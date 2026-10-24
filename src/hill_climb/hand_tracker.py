"""Hand tracking interface utilizing MediaPipe Tasks HandLandmarker."""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

from hill_climb.config import TrackingConfig

logger = logging.getLogger(__name__)

MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Standard 21 MediaPipe hand landmark connections
HAND_CONNECTIONS: list[tuple[int, int]] = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    # Palm base connections
    (5, 9),
    (9, 13),
    (13, 17),
]


@dataclass
class HandLandmark:
    x: float
    y: float
    z: float


@dataclass
class HandDetectionResult:
    has_hand: bool
    landmarks: list[HandLandmark] = field(default_factory=list)
    handedness: str = "Unknown"
    score: float = 0.0


def ensure_model_file(model_path_str: str) -> Path:
    """Ensure MediaPipe hand landmarker model exists, downloading if necessary."""
    path = Path(model_path_str)
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe hand landmarker model to %s...", path)
    urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, path)
    logger.info("Model download complete.")
    return path


class HandTracker:
    """Detects and tracks hand landmarks from video frames."""

    def __init__(
        self,
        config: TrackingConfig,
        landmarker_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._landmarker: Any = None
        self._last_timestamp_ms: int = 0

        model_path = ensure_model_file(config.model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

        factory = landmarker_factory or vision.HandLandmarker.create_from_options
        self._landmarker = factory(options)

    def process_frame(
        self,
        frame_bgr: Any,
        timestamp_ms: int | None = None,
    ) -> HandDetectionResult:
        """Process a BGR video frame and extract hand landmarks."""
        if self._landmarker is None:
            return HandDetectionResult(has_hand=False, landmarks=[])

        if timestamp_ms is None:
            now_ms = int(time.time() * 1000)
            # Ensure strictly increasing timestamp for MediaPipe VIDEO mode
            timestamp_ms = max(now_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result or not result.hand_landmarks:
            return HandDetectionResult(has_hand=False, landmarks=[])

        # We take the first detected hand
        raw_landmarks = result.hand_landmarks[0]
        landmarks = [
            HandLandmark(x=float(lm.x), y=float(lm.y), z=float(lm.z))
            for lm in raw_landmarks
        ]

        handedness = "Unknown"
        score = 0.0
        if result.handedness and len(result.handedness) > 0:
            top_category = result.handedness[0][0]
            handedness = top_category.category_name or "Unknown"
            score = float(top_category.score)

        return HandDetectionResult(
            has_hand=True,
            landmarks=landmarks,
            handedness=handedness,
            score=score,
        )

    def close(self) -> None:
        """Release tracker resources."""
        if self._landmarker is not None:
            if hasattr(self._landmarker, "close"):
                self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
