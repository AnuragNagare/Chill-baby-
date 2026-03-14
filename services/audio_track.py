"""
services/audio_track.py – Phase 3

AudioTransformTrack
--------------------
Receives raw audio frames from the WebRTC stream, buffers 1 second of audio,
then runs YAMNet (Google's sound classifier) in a thread executor.

Detected events:
  cough  (YAMNet class 370) – with noise filtering and acoustic severity
  sneeze (YAMNet class 411)

On detection:
  - Broadcasts {"_state": 7, "event": "cough"|"sneeze", "confidence": 0.87,
                "severity": "mild"|"moderate"|"severe"}  (severity for cough only)
  - Inserts an alert_events document into MongoDB.

Noise handling: spectral reduction, SNR gate, quiet gate.
Cough severity: mild / moderate / severe from loudness, duration, burst count.

Install: pip install tensorflow-hub noisereduce
"""

import logging
import asyncio
import concurrent.futures
import time
from collections import deque
from datetime import datetime

import numpy as np
from av import AudioFrame
from aiortc import MediaStreamTrack
import tensorflow_hub as hub
import tensorflow as tf

import db
from config import (
    YAMNET_MODEL_URL,
    YAMNET_SAMPLE_RATE,
    YAMNET_BUFFER_SECONDS,
    YAMNET_COUGH_CLASS_ID,
    YAMNET_SNEEZE_CLASS_ID,
    YAMNET_CONFIDENCE_THRESH,
    COUGH_BURST_WINDOW_SEC,
)
from services.cough_analyzer import (
    reduce_noise,
    estimate_snr,
    compute_acoustic_features,
    estimate_cough_severity,
    should_skip_due_to_noise,
)

logger   = logging.getLogger(__name__)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# ── YAMNet singleton ─────────────────────────────────────────────────────────
_yamnet_model = None

def get_yamnet():
    global _yamnet_model
    if _yamnet_model is None:
        logger.info("Loading YAMNet from TF Hub: %s", YAMNET_MODEL_URL)
        _yamnet_model = hub.load(YAMNET_MODEL_URL)
        logger.info("YAMNet model ready.")
    return _yamnet_model


# ── Classification (runs in thread executor) ─────────────────────────────────
def _yamnet_classify(waveform: np.ndarray) -> tuple[str | None, float]:
    """Run YAMNet on waveform. Returns (label, confidence)."""
    model = get_yamnet()
    waveform_tf = tf.constant(waveform, dtype=tf.float32)
    scores, _, _ = model(waveform_tf)   # scores shape: [frames, 521]
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    cough_score  = float(mean_scores[YAMNET_COUGH_CLASS_ID])
    sneeze_score = float(mean_scores[YAMNET_SNEEZE_CLASS_ID])
    logger.debug("YAMNet — cough=%.3f sneeze=%.3f", cough_score, sneeze_score)
    if cough_score >= YAMNET_CONFIDENCE_THRESH and cough_score >= sneeze_score:
        return "cough", cough_score
    if sneeze_score >= YAMNET_CONFIDENCE_THRESH:
        return "sneeze", sneeze_score
    return None, 0.0


def analyze_and_classify_audio(
    waveform: np.ndarray,
    sample_rate: int,
    recent_cough_count: int = 0,
) -> tuple[str | None, float, str | None]:
    """
    Full pipeline: noise reduction → SNR gate → YAMNet → cough severity.

    Returns
    -------
    (label, confidence, severity)
        label: 'cough', 'sneeze', or None
        severity: 'mild'|'moderate'|'severe' for cough, None for sneeze
    """
    # 1. Noise reduction
    cleaned = reduce_noise(waveform, sample_rate)

    # 2. SNR check – skip if too noisy
    snr_db = estimate_snr(cleaned)
    skip, reason = should_skip_due_to_noise(cleaned, snr_db)
    if skip:
        logger.debug("Skipping audio: %s (SNR=%.1f dB)", reason, snr_db)
        return None, 0.0, None

    # 3. YAMNet classification
    label, conf = _yamnet_classify(cleaned)
    if not label:
        return None, 0.0, None

    # 4. Cough severity (cough only)
    severity = None
    if label == "cough":
        features = compute_acoustic_features(cleaned)
        severity = estimate_cough_severity(
            features, sample_rate=sample_rate, recent_cough_count=recent_cough_count
        )
        logger.debug("Cough severity: %s (rms=%.4f, bursts=%d)",
                     severity, features["rms"], features["burst_count"])

    return label, conf, severity


# ── Audio resampling helper ───────────────────────────────────────────────────
def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Simple linear resample using scipy if rates differ."""
    if src_rate == dst_rate:
        return audio
    from scipy.signal import resample as scipy_resample
    num_samples = int(len(audio) * dst_rate / src_rate)
    return scipy_resample(audio, num_samples).astype(np.float32)


# ── AudioTransformTrack ───────────────────────────────────────────────────────
class AudioTransformTrack(MediaStreamTrack):
    """
    WebRTC audio track that buffers frames and classifies cough/sneeze sounds.
    """
    kind = "audio"

    def __init__(self, track, user_id: str, connections: dict,
                 globalvars: dict, session_id=None):
        super().__init__()
        self.track       = track
        self.user_id     = user_id
        self.connections = connections
        self.globalvars  = globalvars
        self.session_id  = session_id

        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._target_samples = int(YAMNET_SAMPLE_RATE * YAMNET_BUFFER_SECONDS)
        self._cough_times: deque[float] = deque(maxlen=20)  # for burst severity

        logger.info("AudioTransformTrack created for user=%s", user_id)

    async def recv(self) -> AudioFrame:
        frame = await self.track.recv()

        # Only classify when session is active
        if not self.globalvars.get("processing"):
            return frame

        try:
            # Convert av.AudioFrame → numpy float32 mono
            audio_np = frame.to_ndarray()          # shape: [channels, samples]
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=0)   # mix to mono
            audio_np = audio_np.astype(np.float32)

            # Normalise to [-1.0, 1.0] if integer PCM
            if audio_np.max() > 1.0:
                audio_np = audio_np / 32768.0

            # Resample to 16 kHz if needed
            src_rate = frame.sample_rate
            audio_np = _resample(audio_np, src_rate, YAMNET_SAMPLE_RATE)

            # Accumulate into rolling buffer
            self._buffer.append(audio_np)
            self._buffer_samples += len(audio_np)

            # Once we have 1 second of audio, classify
            if self._buffer_samples >= self._target_samples:
                waveform = np.concatenate(self._buffer)[:self._target_samples]
                self._buffer       = []
                self._buffer_samples = 0

                # Count coughs in burst window for severity
                now = time.time()
                cutoff = now - COUGH_BURST_WINDOW_SEC
                recent_count = sum(1 for t in self._cough_times if t > cutoff)

                loop = asyncio.get_event_loop()
                label, conf, severity = await loop.run_in_executor(
                    executor,
                    lambda: analyze_and_classify_audio(
                        waveform, YAMNET_SAMPLE_RATE, recent_cough_count=recent_count
                    ),
                )

                if label:
                    if label == "cough":
                        self._cough_times.append(now)
                    logger.info(
                        "Audio event detected: %s (conf=%.2f%s) user=%s",
                        label, conf,
                        f" severity={severity}" if severity else "",
                        self.user_id,
                    )

                    # Broadcast to all WebSocket clients (_state 7)
                    payload = {
                        "_state":     7,
                        "event":      label,
                        "confidence": round(conf, 2),
                    }
                    if severity:
                        payload["severity"] = severity
                    for ws in self.connections.values():
                        await ws.send_json(payload)

                    # Persist to MongoDB alert_events
                    try:
                        doc = {
                            "session_id": self.session_id,
                            "timestamp":  datetime.utcnow(),
                            "alert_type": label,
                            "confidence": round(conf, 2),
                            "metadata":   {"severity": severity} if severity else {},
                        }
                        await db.alert_events().insert_one(doc)
                    except Exception:
                        logger.exception("Failed to log audio alert to MongoDB")

        except Exception:
            logger.exception("Error in AudioTransformTrack.recv()")

        return frame
