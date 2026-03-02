"""
Emotion detection service.
Wraps the FER library and exposes a single module-level detector instance
so the model is loaded only once at startup.
"""

import logging
from fer import FER

logger = logging.getLogger(__name__)

# Single shared FER instance (loaded once)
emotion_detector: FER | None = None


def get_detector() -> FER:
    global emotion_detector
    if emotion_detector is None:
        logger.info("Loading FER emotion detector (mtcnn=True)…")
        emotion_detector = FER(mtcnn=True)
        logger.info("FER detector ready.")
    return emotion_detector


def get_max_emotion(emotions: dict) -> str:
    """Return the emotion label with the highest score, or ' ' if empty."""
    if emotions:
        return max(emotions, key=emotions.get)
    return " "
