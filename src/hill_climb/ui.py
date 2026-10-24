"""Preview window and HUD overlay drawing."""

from __future__ import annotations

import cv2
import numpy as np

from hill_climb.config import UIConfig
from hill_climb.hand_tracker import HAND_CONNECTIONS, HandDetectionResult, HandLandmark

# Fingertip landmark indices in MediaPipe 21-landmark topology
FINGERTIP_INDICES = {4, 8, 12, 16, 20}


class UIOverlay:
    """Renders overlays, status texts, FPS, and landmarks onto preview frames."""

    def __init__(self, config: UIConfig) -> None:
        self.config = config

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: list[HandLandmark],
    ) -> None:
        """Render landmark points and skeletal connection lines on the frame."""
        if not landmarks:
            return

        height, width = frame.shape[:2]
        pts: list[tuple[int, int]] = [
            (
                int(np.clip(lm.x * width, 0, width - 1)),
                int(np.clip(lm.y * height, 0, height - 1)),
            )
            for lm in landmarks
        ]

        # Draw skeletal connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(
                    frame,
                    pts[start_idx],
                    pts[end_idx],
                    (255, 180, 50),  # Cyan/Sky blue connection lines
                    2,
                    cv2.LINE_AA,
                )

        # Draw joints and fingertips
        for idx, (px, py) in enumerate(pts):
            if idx in FINGERTIP_INDICES:
                # Highlight fingertips with bright accent color
                cv2.circle(frame, (px, py), 6, (0, 140, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 4, (0, 255, 120), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 5, (0, 80, 0), 1, cv2.LINE_AA)

    def draw_preview(
        self,
        frame: np.ndarray,
        fps: float,
        detection: HandDetectionResult | None = None,
        status: str = "Active",
    ) -> np.ndarray:
        """Render preview HUD with FPS, tracking status, and landmarks."""
        canvas = frame.copy()
        height, width = canvas.shape[:2]

        # Draw hand landmarks if present and enabled
        if detection and detection.has_hand and self.config.show_landmarks:
            self.draw_landmarks(canvas, detection.landmarks)

        # Draw semi-transparent header bar
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, 42), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        # Status text determination
        if detection is not None:
            if detection.has_hand:
                status_color = (0, 255, 120)  # Bright green
                hand_status = f"HAND: {detection.handedness.upper()}"
            else:
                status_color = (50, 180, 255)  # Amber / orange
                hand_status = "NO HAND"
        else:
            status_color = (0, 255, 120)
            hand_status = status

        header_text = f"FPS: {fps:.1f} | {hand_status} | Res: {width}x{height}"
        cv2.putText(
            canvas,
            header_text,
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
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
        if detection.has_hand and self.config.show_landmarks:
            self.draw_landmarks(canvas, detection.landmarks)

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
