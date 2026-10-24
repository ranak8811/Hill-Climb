"""Comprehensive unit tests for pure geometric gesture classification."""

from hill_climb.gestures import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    Gesture,
    classify_gesture,
    get_finger_states,
    is_finger_extended,
)
from hill_climb.hand_tracker import HandLandmark


def build_synthetic_hand(
    thumb_extended: bool = False,
    index_extended: bool = False,
    middle_extended: bool = False,
    ring_extended: bool = False,
    pinky_extended: bool = False,
    scale: float = 1.0,
    offset_x: float = 0.5,
    offset_y: float = 0.5,
) -> list[HandLandmark]:
    """Construct a full 21-landmark hand with configurable finger extension states.

    Wrist is at bottom (y=0.9). MCP knuckles are at y=0.55.
    Extended fingers reach up to y=0.15. Curled fingers fold down to y=0.60.
    """
    pts: list[tuple[float, float, float]] = [
        (0.0, 0.4, 0.0)
    ] * 21  # default wrist relative
    pts[0] = (0.0, 0.4, 0.0)  # WRIST

    # Thumb: 1: CMC, 2: MCP, 3: IP, 4: TIP
    if thumb_extended:
        pts[1] = (-0.1, 0.3, 0.0)
        pts[2] = (-0.2, 0.2, 0.0)
        pts[3] = (-0.28, 0.1, 0.0)
        pts[4] = (-0.35, 0.0, 0.0)  # Spread far to the side
    else:
        pts[1] = (-0.08, 0.3, 0.0)
        pts[2] = (-0.12, 0.22, 0.0)
        pts[3] = (-0.08, 0.18, 0.0)
        pts[4] = (-0.02, 0.2, 0.0)  # Tucked across palm

    finger_configs = [
        # (tip_idx, pip_idx, mcp_idx, base_x, is_ext)
        (8, 6, 5, -0.15, index_extended),
        (12, 10, 9, -0.05, middle_extended),
        (16, 14, 13, 0.05, ring_extended),
        (20, 18, 17, 0.15, pinky_extended),
    ]

    for tip_idx, pip_idx, mcp_idx, base_x, is_ext in finger_configs:
        pts[mcp_idx] = (base_x, 0.15, 0.0)
        dip_idx = tip_idx - 1
        if is_ext:
            # Extended straight up towards y = -0.35 relative to wrist
            pts[pip_idx] = (base_x, 0.0, 0.0)
            pts[dip_idx] = (base_x, -0.18, 0.0)
            pts[tip_idx] = (base_x, -0.35, 0.0)
        else:
            # Curled back downwards towards palm
            pts[pip_idx] = (base_x, 0.05, 0.0)
            pts[dip_idx] = (base_x, 0.12, 0.02)
            pts[tip_idx] = (base_x, 0.20, 0.04)

    landmarks: list[HandLandmark] = []
    for x, y, z in pts:
        landmarks.append(
            HandLandmark(
                x=offset_x + x * scale,
                y=offset_y + y * scale,
                z=z * scale,
            )
        )
    return landmarks


def test_classify_no_hand_empty() -> None:
    assert classify_gesture([]) == Gesture.NO_HAND


def test_classify_no_hand_insufficient_landmarks() -> None:
    incomplete = [HandLandmark(0.5, 0.5, 0.0) for _ in range(15)]
    assert classify_gesture(incomplete) == Gesture.NO_HAND


def test_classify_open_palm() -> None:
    hand = build_synthetic_hand(
        thumb_extended=True,
        index_extended=True,
        middle_extended=True,
        ring_extended=True,
        pinky_extended=True,
    )
    assert classify_gesture(hand) == Gesture.OPEN_PALM


def test_classify_fist() -> None:
    hand = build_synthetic_hand(
        thumb_extended=False,
        index_extended=False,
        middle_extended=False,
        ring_extended=False,
        pinky_extended=False,
    )
    assert classify_gesture(hand) == Gesture.FIST


def test_classify_index_up() -> None:
    hand = build_synthetic_hand(
        thumb_extended=False,
        index_extended=True,
        middle_extended=False,
        ring_extended=False,
        pinky_extended=False,
    )
    assert classify_gesture(hand) == Gesture.INDEX_UP


def test_classify_ambiguous_peace_sign_is_unknown() -> None:
    # Peace sign: index and middle extended, ring and pinky curled
    hand = build_synthetic_hand(
        thumb_extended=False,
        index_extended=True,
        middle_extended=True,
        ring_extended=False,
        pinky_extended=False,
    )
    assert classify_gesture(hand) == Gesture.UNKNOWN


def test_classify_ambiguous_pinky_only_is_unknown() -> None:
    hand = build_synthetic_hand(
        thumb_extended=False,
        index_extended=False,
        middle_extended=False,
        ring_extended=False,
        pinky_extended=True,
    )
    assert classify_gesture(hand) == Gesture.UNKNOWN


def test_distance_scale_invariance() -> None:
    # Scaled up 2x (close to camera)
    hand_large = build_synthetic_hand(
        index_extended=True,
        middle_extended=False,
        ring_extended=False,
        pinky_extended=False,
        scale=2.0,
    )
    # Scaled down 0.5x (far from camera)
    hand_small = build_synthetic_hand(
        index_extended=True,
        middle_extended=False,
        ring_extended=False,
        pinky_extended=False,
        scale=0.5,
    )
    assert classify_gesture(hand_large) == Gesture.INDEX_UP
    assert classify_gesture(hand_small) == Gesture.INDEX_UP


def test_finger_extension_helper() -> None:
    hand = build_synthetic_hand(index_extended=True, middle_extended=False)
    assert is_finger_extended(hand, INDEX_TIP, INDEX_PIP, INDEX_MCP) is True

    states = get_finger_states(hand)
    assert states.index is True
    assert states.middle is False
    assert states.ring is False
    assert states.pinky is False
