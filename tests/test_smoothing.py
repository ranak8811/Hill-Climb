"""Tests for gesture smoothing and debouncing."""

from hill_climb.config import GesturesConfig
from hill_climb.gestures import Gesture
from hill_climb.smoothing import GestureFilter


def test_smoothing_debounce() -> None:
    config = GesturesConfig(debounce_frames=3, tracking_loss_timeout=1.0)
    g_filter = GestureFilter(config)

    # Initially NO_HAND
    assert g_filter.current_gesture == Gesture.NO_HAND

    # Sending 2 frames of INDEX_UP should not yet activate with debounce=3
    assert g_filter.update(Gesture.INDEX_UP) == Gesture.NO_HAND
    assert g_filter.update(Gesture.INDEX_UP) == Gesture.NO_HAND

    # Third frame activates INDEX_UP
    assert g_filter.update(Gesture.INDEX_UP) == Gesture.INDEX_UP
