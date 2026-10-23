"""Main entry point for Hill Climb Gesture Controller."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import sys
import time

import cv2

from hill_climb import __version__
from hill_climb.camera import Camera
from hill_climb.config import AppConfig, load_config
from hill_climb.ui import UIOverlay


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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
    max_frames: int | None = None,
) -> int:
    """Run live camera preview loop with FPS and quit controls (Phase 2)."""
    logger = logging.getLogger("hill_climb.preview")
    logger.info(
        "Starting camera preview on index %d (resolution %dx%d, %d fps target)...",
        config.camera.index,
        config.camera.width,
        config.camera.height,
        config.camera.fps,
    )

    camera = Camera(config.camera)
    ui = UIOverlay(config.ui)

    try:
        camera.start()
    except RuntimeError as e:
        logger.error("Camera startup failed: %s", e)
        logger.error(
            "Please verify that Camera access is granted to your terminal/IDE in: "
            "System Settings -> Privacy & Security -> Camera"
        )
        return 1

    window_name = config.ui.preview_window_name
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    start_time = time.time()
    prev_frame_time = start_time
    fps = float(config.camera.fps)

    logger.info("Preview started. Press 'q' or 'Esc' in the preview window to exit.")

    try:
        while True:
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
                # Smooth FPS indicator
                fps = 0.9 * fps + 0.1 * instant_fps

            frame_count += 1

            # Render HUD overlay onto frame
            display_frame = ui.draw_preview(frame, fps=fps, status="Preview Active")
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            # Check for 'q' or Esc (27)
            if key in (ord("q"), ord("Q"), 27):
                logger.info("Exit requested by user key press.")
                break

            # Handle user closing the window via the window manager close button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Preview window closed by user.")
                break

            if max_frames is not None and frame_count >= max_frames:
                logger.debug("Reached requested max_frames limit (%d).", max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C).")
    finally:
        logger.info("Cleaning up camera and preview windows...")
        camera.release()
        cv2.destroyAllWindows()
        # OpenCV requires a short waitKey pump to finalize destroying windows
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
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        return 1

    return run_camera_preview(cfg)


if __name__ == "__main__":
    sys.exit(main())
