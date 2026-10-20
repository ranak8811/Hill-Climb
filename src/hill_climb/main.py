"""Main entry point for Hill Climb Gesture Controller."""

import argparse
import logging
import sys

from hill_climb import __version__
from hill_climb.config import load_config


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
        import importlib.metadata

        pynput_ver = importlib.metadata.version("pynput")
        logger.info(
            "pynput version: %s", pynver if (pynver := pynput_ver) else "available"
        )
    except Exception as e:
        logger.error("pynput is not available: %s", e)
        return False

    logger.info("Environment check passed successfully!")
    return True


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
        logger.info(
            "Configuration loaded: camera index %d, fps %d",
            cfg.camera.index,
            cfg.camera.fps,
        )
    except Exception as e:
        logger.error("Failed to load configuration: %s", e)
        return 1

    logger.info(
        "Phase 1 scaffold ready. "
        "Real-time vision pipeline will be implemented in subsequent phases."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
