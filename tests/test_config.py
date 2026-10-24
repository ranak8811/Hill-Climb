"""Tests for configuration loading and validation."""

import pytest

from hill_climb.config import (
    AppConfig,
    CameraConfig,
    GesturesConfig,
    TrackingConfig,
    load_config,
    validate_config,
)


def test_default_config() -> None:
    cfg = AppConfig()
    validate_config(cfg)
    assert cfg.camera.index == 0
    assert cfg.camera.fps == 30
    assert cfg.tracking.max_num_hands == 1
    assert cfg.gestures.debounce_frames == 3


def test_invalid_camera_index() -> None:
    cfg = AppConfig(camera=CameraConfig(index=-1))
    with pytest.raises(ValueError, match="Camera index must be >= 0"):
        validate_config(cfg)


def test_invalid_confidence() -> None:
    cfg = AppConfig(tracking=TrackingConfig(min_detection_confidence=1.5))
    with pytest.raises(ValueError, match="min_detection_confidence"):
        validate_config(cfg)


def test_invalid_debounce() -> None:
    cfg = AppConfig(gestures=GesturesConfig(debounce_frames=0))
    with pytest.raises(ValueError, match="debounce_frames"):
        validate_config(cfg)


def test_load_default_toml() -> None:
    cfg = load_config()
    assert cfg is not None
    assert cfg.camera.width == 640
    assert cfg.camera.height == 480
    assert cfg.tracking.model_path == "models/hand_landmarker.task"
    assert cfg.key_mapping["open_palm"] == "accelerate"
    assert cfg.key_mapping["fist"] == "brake"
    assert cfg.actions["accelerate"] == "right"
    assert cfg.actions["brake"] == "left"


def test_cli_parse_args_overrides() -> None:
    from hill_climb.main import parse_args

    args = parse_args(
        [
            "--dry-run",
            "--no-preview",
            "--accelerate-key",
            "up",
            "--brake-key",
            "down",
            "--start-delay",
            "5.0",
        ]
    )
    assert args.dry_run is True
    assert args.no_preview is True
    assert args.accelerate_key == "up"
    assert args.brake_key == "down"
    assert args.start_delay == 5.0
