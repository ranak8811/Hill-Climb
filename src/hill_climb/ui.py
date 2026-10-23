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

    def draw_preview(
        self,
        frame: np.ndarray,
        fps: float,
        status: str = "Active",
    ) -> np.ndarray:
        """Render Phase 2 preview HUD with FPS, status, and quit hint."""
        canvas = frame.copy()
        height, width = canvas.shape[:2]

        # Draw semi-transparent header bar
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, 42), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        # FPS & Status text
        fps_text = f"FPS: {fps:.1f} | Status: {status} | Res: {width}x{height}"
        cv2.putText(
            canvas,
            fps_text,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 128),
            2,
            cv2.LINE_AA,
        )

        # Bottom quit instructions
        hint_text = "Press 'q' or 'Esc' to exit preview"
        cv2.putText(
            canvas,
            hint_text,
            (12, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def draw(
        self,
        frame: np.ndarray,
        detection: HandDetectionResult,
        gesture_name: str,
        active_key: str | None,
        fps: float,
    ) -> np.ndarray:
        """Render full HUD elements onto frame for gameplay pipeline."""
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
