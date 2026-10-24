"""Gesture recognition module using relative landmark geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from hill_climb.hand_tracker import HandLandmark

# MediaPipe landmark indices
WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


class Gesture(str, Enum):
    NO_HAND = "no_hand"
    UNKNOWN = "unknown"
    OPEN_PALM = "open_palm"
    INDEX_UP = "index_up"
    FIST = "fist"


@dataclass
class FingerState:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    @property
    def extended_count(self) -> int:
        return sum([self.thumb, self.index, self.middle, self.ring, self.pinky])


def _dist(p1: HandLandmark, p2: HandLandmark) -> float:
    """Euclidean distance between two 3D landmarks."""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def is_finger_extended(
    landmarks: list[HandLandmark],
    tip_idx: int,
    pip_idx: int,
    mcp_idx: int,
) -> bool:
    """Determine whether a non-thumb finger is extended.

    Uses relative distance ratio to wrist (landmark 0) and MCP knuckle,
    making classification invariant to camera distance and hand size.
    """
    wrist = landmarks[WRIST]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    mcp = landmarks[mcp_idx]

    dist_tip_wrist = _dist(tip, wrist)
    dist_pip_wrist = _dist(pip, wrist)
    dist_mcp_wrist = _dist(mcp, wrist)

    # To be extended:
    # 1. Tip must be further from wrist than PIP joint
    # 2. Tip must be substantially extended past the MCP knuckle
    return (dist_tip_wrist > dist_pip_wrist) and (
        dist_tip_wrist > 1.15 * dist_mcp_wrist
    )


def is_thumb_extended(
    landmarks: list[HandLandmark],
    handedness: str = "Unknown",
) -> bool:
    """Determine whether the thumb is extended outward from the hand.

    Measures relative distance between thumb tip and pinky MCP/base.
    """
    thumb_tip = landmarks[THUMB_TIP]
    thumb_ip = landmarks[THUMB_IP]
    pinky_mcp = landmarks[PINKY_MCP]
    wrist = landmarks[WRIST]

    dist_tip_pinky = _dist(thumb_tip, pinky_mcp)
    dist_ip_pinky = _dist(thumb_ip, pinky_mcp)
    dist_tip_wrist = _dist(thumb_tip, wrist)
    dist_ip_wrist = _dist(thumb_ip, wrist)

    # Extended when tip is spread further from the pinky base than IP joint
    return (dist_tip_pinky > dist_ip_pinky) and (dist_tip_wrist > dist_ip_wrist)


def get_finger_states(
    landmarks: list[HandLandmark],
    handedness: str = "Unknown",
) -> FingerState:
    """Evaluate extension states for all five fingers."""
    return FingerState(
        thumb=is_thumb_extended(landmarks, handedness),
        index=is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP),
        middle=is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        ring=is_finger_extended(landmarks, RING_TIP, RING_PIP, RING_MCP),
        pinky=is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP, PINKY_MCP),
    )


def classify_gesture(
    landmarks: list[HandLandmark],
    handedness: str = "Unknown",
) -> Gesture:
    """Classify 21 hand landmarks into a defined Gesture.

    Pure function without hardware dependencies, fully unit-testable.
    """
    if not landmarks or len(landmarks) < 21:
        return Gesture.NO_HAND

    state = get_finger_states(landmarks, handedness)

    # OPEN_PALM: All four primary fingers extended (index, middle, ring, pinky)
    if state.index and state.middle and state.ring and state.pinky:
        return Gesture.OPEN_PALM

    # FIST: All fingers curled (none extended)
    if (
        not state.index
        and not state.middle
        and not state.ring
        and not state.pinky
        and not state.thumb
    ):
        return Gesture.FIST

    # INDEX_UP: Only index finger extended, middle, ring, and pinky curled
    if state.index and not state.middle and not state.ring and not state.pinky:
        return Gesture.INDEX_UP

    return Gesture.UNKNOWN
