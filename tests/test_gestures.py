"""Tests for gesture classification."""

from hill_climb.gestures import Gesture, classify_gesture


def test_no_hand_classification() -> None:
    assert classify_gesture([]) == Gesture.NO_HAND
