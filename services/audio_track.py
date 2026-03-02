"""
services/audio_track.py – Phase 3

AudioTransformTrack
--------------------
Receives raw audio frames from the WebRTC stream, buffers 1 second of audio,
then runs YAMNet (Google's sound classifier) in a thread executor.

Detected events:
  cough  (YAMNet class 370)
  sneeze (YAMNet class 411)

On detection:
  - Broadcasts {"_state": 7, "event": "cough"|"sneeze", "confidence": 0.87}
    to all connected WebSocket clients.
  - Inserts an alert_events document into MongoDB.

YAMNet requires:
  - Mono audio at 16 000 Hz
  - Input shape: [num_samples]  (float32, range -1.0 to 1.0)

Install: pip install tensorflow-hub
"""

import logging
import asyncio
import concurrent.futures
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
def classify_audio(waveform: np.ndarray) -> tuple[str | None, float]:
    """
    Run YAMNet on a mono 16 kHz float32 waveform.

    Returns
    -------
    (label, confidence)  where label is 'cough', 'sneeze', or None
    """
    model   = get_yamnet()
    waveform_tf = tf.constant(waveform, dtype=tf.float32)
    scores, _, _ = model(waveform_tf)   # scores shape: [frames, 521]

    # Average across time frames → shape [521]
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()

    cough_score  = float(mean_scores[YAMNET_COUGH_CLASS_ID])
    sneeze_score = float(mean_scores[YAMNET_SNEEZE_CLASS_ID])

    logger.debug("YAMNet — cough=%.3f sneeze=%.3f", cough_score, sneeze_score)

    if cough_score >= YAMNET_CONFIDENCE_THRESH and cough_score >= sneeze_score:
        return "cough", cough_score
    if sneeze_score >= YAMNET_CONFIDENCE_THRESH:
        return "sneeze", sneeze_score
    return None, 0.0


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

                loop  = asyncio.get_event_loop()
                label, conf = await loop.run_in_executor(
                    executor, classify_audio, waveform
                )

                if label:
                    logger.info("Audio event detected: %s (conf=%.2f) user=%s",
                                label, conf, self.user_id)

                    # Broadcast to all WebSocket clients (_state 7)
                    payload = {
                        "_state":     7,
                        "event":      label,
                        "confidence": round(conf, 2),
                    }
                    for ws in self.connections.values():
                        await ws.send_json(payload)

                    # Persist to MongoDB alert_events
                    try:
                        await db.alert_events().insert_one({
                            "session_id": self.session_id,
                            "timestamp":  datetime.utcnow(),
                            "alert_type": label,          # "cough" or "sneeze"
                            "confidence": round(conf, 2),
                            "metadata":   {},
                        })
                    except Exception:
                        logger.exception("Failed to log audio alert to MongoDB")

        except Exception:
            logger.exception("Error in AudioTransformTrack.recv()")

        return frame
