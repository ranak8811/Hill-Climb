"""Safe keyboard controller for translating game actions to keyboard input."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from hill_climb.gestures import Gesture

logger = logging.getLogger(__name__)


class KeyboardBackend(Protocol):
    """Protocol for press/release keyboard simulation."""

    def press(self, key: str) -> None: ...

    def release(self, key: str) -> None: ...


class PynputBackend:
    """Production backend using pynput for macOS keyboard simulation."""

    def __init__(self) -> None:
        from pynput.keyboard import Controller, Key

        self._controller = Controller()
        self._special_keys: dict[str, Any] = {
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "space": Key.space,
            "esc": Key.esc,
            "enter": Key.enter,
        }

    def _resolve_key(self, key_str: str) -> Any:
        normalized = key_str.lower().strip()
        if normalized in self._special_keys:
            return self._special_keys[normalized]
        return normalized

    def press(self, key: str) -> None:
        try:
            resolved = self._resolve_key(key)
            self._controller.press(resolved)
        except Exception as e:
            logger.warning(
                "Failed to simulate key press '%s': %s. "
                "Ensure Accessibility permission is granted in macOS System Settings.",
                key,
                e,
            )

    def release(self, key: str) -> None:
        try:
            resolved = self._resolve_key(key)
            self._controller.release(resolved)
        except Exception as e:
            logger.warning("Failed to simulate key release '%s': %s", key, e)


class KeyboardController:
    """Tracks active held keys and guarantees safe release."""

    def __init__(
        self,
        backend: KeyboardBackend | None = None,
        dry_run: bool = False,
    ) -> None:
        self._backend = backend
        self._dry_run = dry_run
        self._active_key: str | None = None

    @property
    def active_key(self) -> str | None:
        """Currently held keyboard key or None."""
        return self._active_key

    @property
    def dry_run(self) -> bool:
        """Whether controller is running in dry-run mode (display only)."""
        return self._dry_run

    def apply_gesture(
        self,
        gesture: Gesture,
        key_mapping: dict[str, str],
        actions: dict[str, str],
    ) -> str | None:
        """Map a stabilized gesture to an action and update the held key."""
        action_name = key_mapping.get(gesture.value, "neutral")
        intended_key = actions.get(action_name, "")

        if not intended_key:
            self.set_key(None)
        else:
            self.set_key(intended_key)

        return self._active_key

    def set_key(self, key: str | None) -> None:
        """Set the active game key, releasing the previous key if changed."""
        if key == self._active_key:
            return

        # Release previous key before pressing new key
        if self._active_key:
            self._do_release(self._active_key)
            self._active_key = None

        if key:
            self._do_press(key)
            self._active_key = key

    def release_all(self) -> None:
        """Emergency and normal exit safe release of all tracked keys."""
        if self._active_key:
            key_to_release = self._active_key
            self._active_key = None
            self._do_release(key_to_release)
            logger.info("Safe release: released held key '%s'", key_to_release)

    def _do_press(self, key: str) -> None:
        if self._dry_run:
            logger.debug("[DRY RUN] Key pressed: '%s'", key)
            return
        if self._backend:
            self._backend.press(key)

    def _do_release(self, key: str) -> None:
        if self._dry_run:
            logger.debug("[DRY RUN] Key released: '%s'", key)
            return
        if self._backend:
            self._backend.release(key)

    def __enter__(self) -> KeyboardController:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release_all()
