"""Tests for keyboard controller state transitions and release safety."""

from hill_climb.controller import KeyboardController


class MockKeyboardBackend:
    def __init__(self) -> None:
        self.pressed_keys: list[str] = []
        self.released_keys: list[str] = []

    def press(self, key: str) -> None:
        self.pressed_keys.append(key)

    def release(self, key: str) -> None:
        self.released_keys.append(key)


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
