"""Temporal gesture smoothing, debouncing, and tracking loss timeout."""

from __future__ import annotations

import time
from collections import deque

from hill_climb.config import GesturesConfig
from hill_climb.gestures import Gesture


class GestureFilter:
    """Stabilizes raw gesture decisions over time and prevents key flickering."""

    def __init__(self, config: GesturesConfig) -> None:
        self.config = config
        self.history: deque[Gesture] = deque(maxlen=config.debounce_frames)
        self.current_gesture: Gesture = Gesture.NO_HAND
        self.last_detection_time: float = time.time()

    def update(
        self,
        raw_gesture: Gesture,
        current_time: float | None = None,
    ) -> Gesture:
        """Update filter with a new observation and return the stabilized gesture."""
        now = time.time() if current_time is None else current_time

        if raw_gesture != Gesture.NO_HAND:
            self.last_detection_time = now

        # Tracking loss timeout check
        if (now - self.last_detection_time) > self.config.tracking_loss_timeout:
            self.current_gesture = Gesture.NO_HAND
            self.history.clear()
            return self.current_gesture

        self.history.append(raw_gesture)

        # Update stabilized gesture only when all debounced frames agree
        if (
            len(self.history) == self.config.debounce_frames
            and len(set(self.history)) == 1
        ):
            self.current_gesture = raw_gesture

        return self.current_gesture

    def reset(self) -> None:
        """Reset internal history and state to neutral."""
        self.history.clear()
        self.current_gesture = Gesture.NO_HAND
        self.last_detection_time = time.time()
