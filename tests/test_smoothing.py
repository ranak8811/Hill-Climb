"""Tests for gesture temporal smoothing, debouncing, and tracking loss timeout."""

from hill_climb.config import GesturesConfig
from hill_climb.gestures import Gesture
from hill_climb.smoothing import GestureFilter


def test_smoothing_debounce() -> None:
    config = GesturesConfig(debounce_frames=3, tracking_loss_timeout=1.0)
    g_filter = GestureFilter(config)

    # Initially NO_HAND
    assert g_filter.current_gesture == Gesture.NO_HAND

    # Sending 2 frames of INDEX_UP should not yet activate with debounce=3
    assert g_filter.update(Gesture.INDEX_UP, current_time=0.0) == Gesture.NO_HAND
    assert g_filter.update(Gesture.INDEX_UP, current_time=0.033) == Gesture.NO_HAND

    # Third consecutive frame activates INDEX_UP
    assert g_filter.update(Gesture.INDEX_UP, current_time=0.066) == Gesture.INDEX_UP
    assert g_filter.current_gesture == Gesture.INDEX_UP


def test_smoothing_suppresses_outlier_glitch() -> None:
    config = GesturesConfig(debounce_frames=3, tracking_loss_timeout=1.0)
    g_filter = GestureFilter(config)

    # Establish INDEX_UP
    g_filter.update(Gesture.INDEX_UP, current_time=0.0)
    g_filter.update(Gesture.INDEX_UP, current_time=0.033)
    g_filter.update(Gesture.INDEX_UP, current_time=0.066)
    assert g_filter.current_gesture == Gesture.INDEX_UP

    # A single frame outlier (e.g. glitch to FIST or UNKNOWN) does not switch state
    assert g_filter.update(Gesture.FIST, current_time=0.100) == Gesture.INDEX_UP
    # Next frame back to INDEX_UP preserves INDEX_UP
    assert g_filter.update(Gesture.INDEX_UP, current_time=0.133) == Gesture.INDEX_UP


def test_smoothing_tracking_loss_timeout() -> None:
    config = GesturesConfig(debounce_frames=3, tracking_loss_timeout=0.5)
    g_filter = GestureFilter(config)

    # Establish active gesture at t = 1.0s
    g_filter.update(Gesture.INDEX_UP, current_time=1.0)
    g_filter.update(Gesture.INDEX_UP, current_time=1.033)
    g_filter.update(Gesture.INDEX_UP, current_time=1.066)
    assert g_filter.current_gesture == Gesture.INDEX_UP

    # At t = 1.1s (0.034s since last detection), hand disappears (NO_HAND)
    # Still within the 0.5s timeout window
    assert g_filter.update(Gesture.NO_HAND, current_time=1.1) == Gesture.INDEX_UP

    # At t = 1.6s (> 0.5s timeout elapsed), state safely forces NO_HAND
    assert g_filter.update(Gesture.NO_HAND, current_time=1.6) == Gesture.NO_HAND
    assert g_filter.current_gesture == Gesture.NO_HAND


def test_smoothing_reset() -> None:
    config = GesturesConfig(debounce_frames=3, tracking_loss_timeout=0.5)
    g_filter = GestureFilter(config)

    g_filter.update(Gesture.INDEX_UP, current_time=0.0)
    g_filter.update(Gesture.INDEX_UP, current_time=0.033)
    g_filter.update(Gesture.INDEX_UP, current_time=0.066)
    assert g_filter.current_gesture == Gesture.INDEX_UP

    g_filter.reset()
    assert g_filter.current_gesture == Gesture.NO_HAND
    assert len(g_filter.history) == 0
