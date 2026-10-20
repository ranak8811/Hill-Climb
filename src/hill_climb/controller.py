"""Safe keyboard controller for translating game actions to keyboard input."""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class KeyboardBackend(Protocol):
    """Protocol for press/release keyboard simulation."""

    def press(self, key: str) -> None: ...

    def release(self, key: str) -> None: ...


class PynputBackend:
    """Production backend using pynput for macOS keyboard simulation."""

    def __init__(self) -> None:
        from pynput.keyboard import Controller

        self._controller = Controller()

    def press(self, key: str) -> None:
        self._controller.press(key)

    def release(self, key: str) -> None:
        self._controller.release(key)


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
        return self._active_key

    def set_key(self, key: str | None) -> None:
        """Set the single active game key, releasing the previous key if changed."""
        if key == self._active_key:
            return

        if self._active_key:
            self._do_release(self._active_key)
            self._active_key = None

        if key:
            self._do_press(key)
            self._active_key = key

    def release_all(self) -> None:
        """Emergency and normal exit safe release."""
        if self._active_key:
            self._do_release(self._active_key)
            self._active_key = None

    def _do_press(self, key: str) -> None:
        if self._dry_run:
            logger.debug("[DRY RUN] Key pressed: %s", key)
            return
        if self._backend:
            self._backend.press(key)

    def _do_release(self, key: str) -> None:
        if self._dry_run:
            logger.debug("[DRY RUN] Key released: %s", key)
            return
        if self._backend:
            self._backend.release(key)
