"""Gesture recognition module."""

from __future__ import annotations

from enum import Enum

from hill_climb.hand_tracker import HandLandmark


class Gesture(str, Enum):
    NO_HAND = "no_hand"
    UNKNOWN = "unknown"
    OPEN_PALM = "open_palm"
    INDEX_UP = "index_up"
    FIST = "fist"


def classify_gesture(landmarks: list[HandLandmark]) -> Gesture:
    """Classify hand landmarks into a defined Gesture.

    Will be fully implemented in Phase 4.
    """
    if not landmarks:
        return Gesture.NO_HAND
    return Gesture.UNKNOWN
