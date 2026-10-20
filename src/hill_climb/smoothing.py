"""Temporal gesture smoothing and debouncing."""

from __future__ import annotations

import time
from collections import deque

from hill_climb.config import GesturesConfig
from hill_climb.gestures import Gesture


class GestureFilter:
    """Stabilizes raw gesture decisions over multiple frames."""

    def __init__(self, config: GesturesConfig) -> None:
        self.config = config
        self.history: deque[Gesture] = deque(maxlen=config.debounce_frames)
        self.current_gesture: Gesture = Gesture.NO_HAND
        self.last_detection_time: float = time.time()

    def update(self, raw_gesture: Gesture) -> Gesture:
        """Update filter with a new raw observation and return the stabilized gesture.

        Will be fully implemented in Phase 5.
        """
        now = time.time()
        if raw_gesture != Gesture.NO_HAND:
            self.last_detection_time = now

        # Timeout check
        if (now - self.last_detection_time) > self.config.tracking_loss_timeout:
            self.current_gesture = Gesture.NO_HAND
            self.history.clear()
            return self.current_gesture

        self.history.append(raw_gesture)
        if (
            len(self.history) == self.config.debounce_frames
            and len(set(self.history)) == 1
        ):
            self.current_gesture = raw_gesture

        return self.current_gesture

    def reset(self) -> None:
        self.history.clear()
        self.current_gesture = Gesture.NO_HAND
