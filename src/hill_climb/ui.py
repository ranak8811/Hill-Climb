"""Preview window and HUD overlay drawing."""

from __future__ import annotations

import cv2
import numpy as np

from hill_climb.config import UIConfig
from hill_climb.hand_tracker import HandDetectionResult


class UIOverlay:
    """Renders overlays, status texts, FPS, and landmarks onto preview frames."""

    def __init__(self, config: UIConfig) -> None:
        self.config = config

    def draw(
        self,
        frame: np.ndarray,
        detection: HandDetectionResult,
        gesture_name: str,
        active_key: str | None,
        fps: float,
    ) -> np.ndarray:
        """Render HUD elements onto frame."""
        # Minimal initial HUD for Phase 1
        canvas = frame.copy()
        text = f"Gesture: {gesture_name} | Key: {active_key or 'None'} | FPS: {fps:.1f}"
        cv2.putText(
            canvas,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return canvas
