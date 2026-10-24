"""Main entry point for Hill Climb Gesture Controller."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import sys
import threading
import time
from typing import Any

import cv2

from hill_climb import __version__
from hill_climb.camera import Camera
from hill_climb.config import AppConfig, load_config
from hill_climb.controller import KeyboardController, PynputBackend
from hill_climb.gestures import Gesture, classify_gesture
from hill_climb.hand_tracker import HandTracker
from hill_climb.smoothing import GestureFilter
from hill_climb.ui import UIOverlay


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-powered hand gesture controller for Hill Climb Racing."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration TOML file.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment dependencies and exit.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Override camera device index (default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display intended keys without sending keyboard events.",
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Disable keyboard automation completely (camera & tracking preview only).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without GUI preview window (keeps Chrome focused during gameplay).",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=None,
        help="Countdown delay in seconds before enabling control (default: 3.0).",
    )
    parser.add_argument(
        "--accelerate-key",
        type=str,
        default=None,
        help="Override key for accelerate (default: 'right').",
    )
    parser.add_argument(
        "--brake-key",
        type=str,
        default=None,
        help="Override key for brake/reverse (default: 'left').",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(args)


def check_environment() -> bool:
    """Verify core dependencies and platform requirements."""
    logger = logging.getLogger("hill_climb")
    logger.info("Checking environment compatibility...")

    try:
        import cv2

        logger.info("OpenCV version: %s", cv2.__version__)
    except ImportError as e:
        logger.error("OpenCV is not available: %s", e)
        return False

    try:
        import mediapipe as mp

        logger.info("MediaPipe version: %s", mp.__version__)
    except ImportError as e:
        logger.error("MediaPipe is not available: %s", e)
        return False

    try:
        pynput_ver = importlib.metadata.version("pynput")
        logger.info("pynput version: %s", pynput_ver)
    except Exception as e:
        logger.error("pynput is not available: %s", e)
        return False

    logger.info("Environment check passed successfully!")
    return True


def run_camera_preview(
    config: AppConfig,
    dry_run: bool = False,
    no_control: bool = False,
    no_preview: bool = False,
    start_delay: float = 0.0,
    max_frames: int | None = None,
) -> int:
    """Run camera capture loop with gesture smoothing and safe control (Phase 6)."""
    logger = logging.getLogger("hill_climb.controller")
    logger.info(
        "Starting camera on index %d (%dx%d, target %d fps)...",
        config.camera.index,
        config.camera.width,
        config.camera.height,
        config.camera.fps,
    )

    camera = Camera(config.camera)
    ui = UIOverlay(config.ui)
    hand_tracker: HandTracker | None = None
    gesture_filter = GestureFilter(config.gestures)
    keyboard_controller: KeyboardController | None = None

    if not no_control:
        backend = PynputBackend()
        keyboard_controller = KeyboardController(backend=backend, dry_run=dry_run)
        if dry_run:
            logger.info("Controller initialized in DRY-RUN mode (display only).")
        else:
            logger.info("Keyboard controller initialized for active game control.")

    try:
        camera.start()
    except RuntimeError as e:
        logger.error("Camera startup failed: %s", e)
        logger.error(
            "Please verify that Camera access is granted to your terminal/IDE in: "
            "System Settings -> Privacy & Security -> Camera"
        )
        return 1

    try:
        logger.info("Initializing MediaPipe HandTracker...")
        hand_tracker = HandTracker(config.tracking)
    except Exception as e:
        logger.error("Failed to initialize HandTracker: %s", e)
        camera.release()
        return 1

    window_name = config.ui.preview_window_name
    if not no_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Startup countdown to give user time to focus Chrome
    if start_delay > 0 and not no_control:
        logger.info(
            "Game focus delay: switch to Chrome "
            "(hill-climb-racing.com/hill-climb-racing-games/hill-climb-race) now!"
        )
        remaining = int(start_delay)
        while remaining > 0:
            logger.info("Game control activates in %d second(s)...", remaining)
            time.sleep(1.0)
            remaining -= 1
        logger.info("Game control is now ACTIVE! Keep Chrome focused.")

    # Global hotkey listener so Esc/q exits even when Chrome is focused
    exit_event = threading.Event()

    def on_global_press(key: Any) -> None:
        try:
            from pynput.keyboard import Key

            if key == Key.esc:
                logger.info("Global [Esc] pressed -> exiting controller safely.")
                exit_event.set()
            elif hasattr(key, "char") and key.char in ("q", "Q"):
                logger.info("Global ['q'] pressed -> exiting controller safely.")
                exit_event.set()
        except Exception:
            pass

    global_listener: Any = None
    try:
        from pynput.keyboard import Listener

        global_listener = Listener(on_press=on_global_press)
        global_listener.daemon = True
        global_listener.start()
    except Exception as e:
        logger.debug("Global listener could not start: %s", e)

    frame_count = 0
    start_time = time.time()
    prev_frame_time = start_time
    fps = float(config.camera.fps)
    prev_active_key: str | None = None
    prev_stabilized_gesture: Gesture | None = None
    prev_has_hand: bool = False

    if no_preview:
        logger.info("Running in HEADLESS mode (no GUI). Press [Esc] or Ctrl+C to stop.")
    else:
        logger.info("Controller running. Press 'q' or 'Esc' (any window) to exit.")

    try:
        while True:
            # Check global exit flag
            if exit_event.is_set():
                break

            ret, frame = camera.read_frame()
            if not ret or frame is None:
                logger.warning("Empty frame received from camera. Retrying...")
                time.sleep(0.01)
                continue

            current_time = time.time()
            delta = current_time - prev_frame_time
            prev_frame_time = current_time

            if delta > 0:
                instant_fps = 1.0 / delta
                fps = 0.9 * fps + 0.1 * instant_fps

            frame_count += 1

            # Process hand detection
            detection = hand_tracker.process_frame(frame)

            # Classify raw gesture
            if detection.has_hand:
                raw_gesture = classify_gesture(
                    detection.landmarks, detection.handedness
                )
            else:
                raw_gesture = Gesture.NO_HAND

            # Stabilize gesture decision with debounce and timeout filter
            stabilized_gesture = gesture_filter.update(raw_gesture)

            # Apply action to safe keyboard controller
            active_key: str | None = None
            if keyboard_controller is not None:
                active_key = keyboard_controller.apply_gesture(
                    stabilized_gesture,
                    config.key_mapping,
                    config.actions,
                )

            # Real-time terminal logging (active in both preview and headless modes)
            if detection.has_hand != prev_has_hand:
                if detection.has_hand:
                    logger.info(">>> Hand acquired: %s", detection.handedness.upper())
                else:
                    logger.info("<<< Hand lost -> idle (all keys released)")
                prev_has_hand = detection.has_hand

            if (
                active_key != prev_active_key
                or stabilized_gesture != prev_stabilized_gesture
            ):
                logger.info(
                    "State: %-10s | Key: %-7s (FPS: %.1f)",
                    stabilized_gesture.value.upper(),
                    active_key.upper() if active_key else "NONE",
                    fps,
                )
                prev_active_key = active_key
                prev_stabilized_gesture = stabilized_gesture

            if no_preview:
                time.sleep(0.005)
            else:
                # Render HUD, gesture badge, and hand landmarks onto preview frame
                display_frame = ui.draw_preview(
                    frame,
                    fps=fps,
                    detection=detection,
                    gesture=raw_gesture,
                    stabilized_gesture=stabilized_gesture,
                    active_key=active_key,
                    dry_run=dry_run,
                )
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    logger.info("Exit requested via preview window key press.")
                    break

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Preview window closed by user.")
                    break

            if max_frames is not None and frame_count >= max_frames:
                logger.debug("Reached requested max_frames limit (%d).", max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C).")
    finally:
        logger.info("Cleaning up controller, camera, tracker, and preview windows...")
        if global_listener is not None and global_listener.is_alive():
            global_listener.stop()
        if keyboard_controller is not None:
            keyboard_controller.release_all()
        if hand_tracker is not None:
            hand_tracker.close()
        camera.release()
        if not no_preview:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        logger.info("Cleanup completed safely.")

    return 0


def main(args: list[str] | None = None) -> int:
    parsed_args = parse_args(args)
    setup_logging(parsed_args.verbose)
    logger = logging.getLogger("hill_climb")

    logger.info("Hill Climb Gesture Controller v%s", __version__)

    if parsed_args.check_env:
        ok = check_environment()
        return 0 if ok else 1

    try:
        cfg = load_config(parsed_args.config)
        if parsed_args.camera_index is not None:
            cfg.camera.index = parsed_args.camera_index
        if parsed_args.accelerate_key is not None:
            cfg.actions["accelerate"] = parsed_args.accelerate_key
        if parsed_args.brake_key is not None:
            cfg.actions["brake"] = parsed_args.brake_key
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        return 1

    # Default start delay: 3s if active control, 0s if dry-run or no-control
    delay = parsed_args.start_delay
    if delay is None:
        delay = 0.0 if (parsed_args.dry_run or parsed_args.no_control) else 3.0

    return run_camera_preview(
        cfg,
        dry_run=parsed_args.dry_run,
        no_control=parsed_args.no_control,
        no_preview=parsed_args.no_preview,
        start_delay=delay,
    )


if __name__ == "__main__":
    sys.exit(main())
