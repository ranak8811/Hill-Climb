"""Hand tracking interface utilizing MediaPipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hill_climb.config import TrackingConfig


@dataclass
class HandLandmark:
    x: float
    y: float
    z: float


@dataclass
class HandDetectionResult:
    has_hand: bool
    landmarks: list[HandLandmark]
    handedness: str = "Unknown"


class HandTracker:
    """Detects hand landmarks from video frames."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config

    def process_frame(self, frame_bgr: Any) -> HandDetectionResult:
        """Process a frame and extract hand landmarks.

        Will be fully implemented in Phase 3.
        """
        return HandDetectionResult(has_hand=False, landmarks=[])

    def close(self) -> None:
        """Release tracker resources."""
        pass
