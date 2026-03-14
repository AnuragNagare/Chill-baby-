"""
services/cough_analyzer.py

Cough analysis: noise reduction, SNR estimation, and acoustic severity.
Used to filter out background noise and classify cough severity (mild/moderate/severe).
"""

import logging
from typing import Literal

import numpy as np

from config import (
    COUGH_NOISE_GATE_RMS,
    COUGH_SNR_MIN_DB,
    COUGH_SEVERITY_MILD_RMS,
    COUGH_SEVERITY_MODERATE_RMS,
    COUGH_SEVERITY_SEVERE_RMS,
    COUGH_BURST_WINDOW_SEC,
    COUGH_BURST_COUNT_SEVERE,
)

logger = logging.getLogger(__name__)

SeverityLevel = Literal["mild", "moderate", "severe"]


def reduce_noise(waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Apply spectral gating to reduce stationary background noise.
    Falls back to original waveform if noisereduce is unavailable.
    """
    try:
        import noisereduce as nr
        reduced = nr.reduce_noise(y=waveform, sr=sample_rate, stationary=True)
        return reduced.astype(np.float32)
    except ImportError:
        logger.debug("noisereduce not installed, skipping noise reduction")
        return waveform
    except Exception as e:
        logger.warning("Noise reduction failed: %s", e)
        return waveform


def estimate_snr(waveform: np.ndarray, noise_frac: float = 0.15) -> float:
    """
    Estimate SNR in dB. Uses first `noise_frac` of the chunk as noise estimate.
    Returns dB (higher = cleaner signal).
    """
    n = len(waveform)
    if n < 100:
        return 0.0
    noise_len = max(100, int(n * noise_frac))
    noise_part = waveform[:noise_len]
    signal_part = waveform[noise_len:]

    noise_power = np.mean(noise_part ** 2)
    signal_power = np.mean(signal_part ** 2)
    if noise_power < 1e-10:
        return 30.0  # effectively no noise
    if signal_power < 1e-10:
        return -10.0  # no signal
    snr_db = 10.0 * np.log10(signal_power / noise_power + 1e-10)
    return float(snr_db)


def compute_acoustic_features(waveform: np.ndarray) -> dict:
    """
    Extract acoustic features for severity estimation.
    """
    rms = float(np.sqrt(np.mean(waveform ** 2)))
    peak = float(np.max(np.abs(waveform)))
    # Effective duration: proportion of time above 20% of peak (cough burst length)
    threshold = peak * 0.2
    above = np.abs(waveform) > threshold
    # Count runs of "above threshold"
    run_starts = np.where(np.diff(np.concatenate([[False], above, [False]])) == 1)[0]
    run_ends = np.where(np.diff(np.concatenate([[False], above, [False]])) == -1)[0]
    if len(run_starts) > 0 and len(run_ends) > 0:
        run_lengths = run_ends - run_starts
        max_run = int(np.max(run_lengths)) if len(run_lengths) > 0 else 0
        burst_count = len(run_starts)
    else:
        max_run = 0
        burst_count = 0

    return {
        "rms": rms,
        "peak": peak,
        "max_burst_samples": max_run,
        "burst_count": burst_count,
    }


def estimate_cough_severity(
    features: dict,
    sample_rate: int = 16000,
    recent_cough_count: int = 0,
) -> SeverityLevel:
    """
    Classify cough severity from acoustic features.
    mild: single, soft cough
    moderate: medium loudness or 2-3 coughs
    severe: loud, long, or paroxysmal (many coughs)
    """
    rms = features["rms"]
    burst_count = features["burst_count"]
    max_run = features["max_burst_samples"]
    duration_sec = max_run / sample_rate if sample_rate > 0 else 0

    # Paroxysmal: many coughs in the chunk
    total_coughs = burst_count + recent_cough_count
    if total_coughs >= COUGH_BURST_COUNT_SEVERE:
        return "severe"

    # Long sustained cough
    if duration_sec > 0.5:
        return "severe"

    # RMS-based severity
    if rms >= COUGH_SEVERITY_SEVERE_RMS:
        return "severe"
    if rms >= COUGH_SEVERITY_MODERATE_RMS:
        return "moderate"
    if rms >= COUGH_SEVERITY_MILD_RMS:
        return "mild"

    return "mild"


def should_skip_due_to_noise(waveform: np.ndarray, snr_db: float) -> tuple[bool, str]:
    """
    Return (skip, reason). Skip if audio is too noisy or too quiet.
    """
    rms = float(np.sqrt(np.mean(waveform ** 2)))
    if rms < COUGH_NOISE_GATE_RMS:
        return True, "too_quiet"
    if snr_db < COUGH_SNR_MIN_DB:
        return True, "low_snr"
    return False, ""
