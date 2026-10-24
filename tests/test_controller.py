"""Tests for keyboard controller transitions, gesture mapping, and release safety."""

from hill_climb.controller import KeyboardController, PynputBackend
from hill_climb.gestures import Gesture


class MockKeyboardBackend:
    def __init__(self) -> None:
        self.pressed_keys: list[str] = []
        self.released_keys: list[str] = []

    def press(self, key: str) -> None:
        self.pressed_keys.append(key)

    def release(self, key: str) -> None:
        self.released_keys.append(key)


DEFAULT_KEY_MAPPING = {
    "open_palm": "neutral",
    "index_up": "accelerate",
    "fist": "brake",
}

DEFAULT_ACTIONS = {
    "neutral": "",
    "accelerate": "w",
    "brake": "s",
}


def test_controller_press_and_release() -> None:
    backend = MockKeyboardBackend()
    ctrl = KeyboardController(backend=backend)

    assert ctrl.active_key is None

    # Press key "w"
    ctrl.set_key("w")
    assert ctrl.active_key == "w"
    assert backend.pressed_keys == ["w"]
    assert backend.released_keys == []

    # Switching to "s" should release "w" and press "s"
    ctrl.set_key("s")
    assert ctrl.active_key == "s"
    assert backend.released_keys == ["w"]
    assert backend.pressed_keys == ["w", "s"]

    # Setting to None should release "s"
    ctrl.set_key(None)
    assert ctrl.active_key is None
    assert backend.released_keys == ["w", "s"]


def test_controller_idempotent_key() -> None:
    backend = MockKeyboardBackend()
    ctrl = KeyboardController(backend=backend)

    ctrl.set_key("w")
    ctrl.set_key("w")  # No duplicate press
    assert backend.pressed_keys == ["w"]
    assert backend.released_keys == []


def test_controller_release_all() -> None:
    backend = MockKeyboardBackend()
    ctrl = KeyboardController(backend=backend)

    ctrl.set_key("w")
    ctrl.release_all()
    assert ctrl.active_key is None
    assert backend.released_keys == ["w"]

    # Calling release_all again should do nothing
    ctrl.release_all()
    assert backend.released_keys == ["w"]


def test_controller_apply_gesture_transitions() -> None:
    backend = MockKeyboardBackend()
    ctrl = KeyboardController(backend=backend)

    # 1. Start with OPEN_PALM (neutral) -> no key pressed
    ctrl.apply_gesture(Gesture.OPEN_PALM, DEFAULT_KEY_MAPPING, DEFAULT_ACTIONS)
    assert ctrl.active_key is None
    assert backend.pressed_keys == []

    # 2. Transition to INDEX_UP -> presses "w"
    ctrl.apply_gesture(Gesture.INDEX_UP, DEFAULT_KEY_MAPPING, DEFAULT_ACTIONS)
    assert ctrl.active_key == "w"
    assert backend.pressed_keys == ["w"]

    # 3. Transition to FIST -> releases "w", presses "s"
    ctrl.apply_gesture(Gesture.FIST, DEFAULT_KEY_MAPPING, DEFAULT_ACTIONS)
    assert ctrl.active_key == "s"
    assert backend.released_keys == ["w"]
    assert backend.pressed_keys == ["w", "s"]

    # 4. Hand disappears (NO_HAND) -> releases "s", active key becomes None
    ctrl.apply_gesture(Gesture.NO_HAND, DEFAULT_KEY_MAPPING, DEFAULT_ACTIONS)
    assert ctrl.active_key is None
    assert backend.released_keys == ["w", "s"]

    # 5. UNKNOWN gesture -> also neutral, nothing pressed
    ctrl.apply_gesture(Gesture.UNKNOWN, DEFAULT_KEY_MAPPING, DEFAULT_ACTIONS)
    assert ctrl.active_key is None


def test_controller_dry_run_mode() -> None:
    backend = MockKeyboardBackend()
    ctrl = KeyboardController(backend=backend, dry_run=True)

    assert ctrl.dry_run is True
    ctrl.set_key("w")
    assert ctrl.active_key == "w"
    # Backend press should NOT have been called in dry run mode
    assert backend.pressed_keys == []
    assert backend.released_keys == []

    ctrl.release_all()
    assert ctrl.active_key is None
    assert backend.released_keys == []


def test_controller_context_manager() -> None:
    backend = MockKeyboardBackend()
    with KeyboardController(backend=backend) as ctrl:
        ctrl.set_key("w")
        assert ctrl.active_key == "w"

    # Context manager exit should guarantee release_all()
    assert ctrl.active_key is None
    assert backend.released_keys == ["w"]


def test_pynput_backend_key_resolution() -> None:
    backend = PynputBackend()
    # Special keys resolve to pynput Key objects
    from pynput.keyboard import Key

    assert backend._resolve_key("up") == Key.up
    assert backend._resolve_key("down") == Key.down
    assert backend._resolve_key("space") == Key.space
    # Standard character keys resolve to lower-case string
    assert backend._resolve_key("W") == "w"
