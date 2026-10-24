"""Configuration loading and validation for the Hill Climb gesture controller."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True


@dataclass
class TrackingConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    model_path: str = "models/hand_landmarker.task"


@dataclass
class GesturesConfig:
    debounce_frames: int = 3
    tracking_loss_timeout: float = 0.5


@dataclass
class UIConfig:
    show_preview: bool = True
    show_landmarks: bool = True
    preview_window_name: str = "Hill Climb - Gesture Preview"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gestures: GesturesConfig = field(default_factory=GesturesConfig)
    key_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "open_palm": "neutral",
            "index_up": "accelerate",
            "fist": "brake",
        }
    )
    actions: dict[str, str] = field(
        default_factory=lambda: {
            "neutral": "",
            "accelerate": "w",
            "brake": "s",
        }
    )
    ui: UIConfig = field(default_factory=UIConfig)


def validate_config(config: AppConfig) -> None:
    """Validate configuration parameters.

    Raises ValueError if any setting is invalid.
    """
    if config.camera.index < 0:
        raise ValueError(f"Camera index must be >= 0, got {config.camera.index}")
    if config.camera.width <= 0 or config.camera.height <= 0:
        raise ValueError("Camera dimensions must be positive integers.")
    if config.camera.fps <= 0:
        raise ValueError("Camera FPS must be a positive integer.")

    if config.tracking.max_num_hands < 1:
        raise ValueError("max_num_hands must be at least 1.")
    if not (0.0 <= config.tracking.min_detection_confidence <= 1.0):
        raise ValueError("min_detection_confidence must be between 0.0 and 1.0.")
    if not (0.0 <= config.tracking.min_tracking_confidence <= 1.0):
        raise ValueError("min_tracking_confidence must be between 0.0 and 1.0.")

    if config.gestures.debounce_frames < 1:
        raise ValueError("debounce_frames must be >= 1.")
    if config.gestures.tracking_loss_timeout < 0.0:
        raise ValueError("tracking_loss_timeout must be non-negative.")


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from a TOML file.

    If path is None, tries to load from `config/default.toml` relative to project root,
    or falls back to default AppConfig values if not found.
    """
    if path is None:
        # Check current working directory config/default.toml
        default_path = Path("config/default.toml")
        if default_path.exists():
            path = default_path

    if path is None or not Path(path).exists():
        cfg = AppConfig()
        validate_config(cfg)
        return cfg

    with open(path, "rb") as f:
        data: dict[str, Any] = tomllib.load(f)

    cam_data = data.get("camera", {})
    track_data = data.get("tracking", {})
    gesture_data = data.get("gestures", {})
    ui_data = data.get("ui", {})

    cfg = AppConfig(
        camera=CameraConfig(
            index=cam_data.get("index", 0),
            width=cam_data.get("width", 640),
            height=cam_data.get("height", 480),
            fps=cam_data.get("fps", 30),
            mirror=cam_data.get("mirror", True),
        ),
        tracking=TrackingConfig(
            max_num_hands=track_data.get("max_num_hands", 1),
            min_detection_confidence=track_data.get("min_detection_confidence", 0.7),
            min_tracking_confidence=track_data.get("min_tracking_confidence", 0.5),
            model_path=track_data.get("model_path", "models/hand_landmarker.task"),
        ),
        gestures=GesturesConfig(
            debounce_frames=gesture_data.get("debounce_frames", 3),
            tracking_loss_timeout=gesture_data.get("tracking_loss_timeout", 0.5),
        ),
        key_mapping=data.get(
            "key_mapping",
            {"open_palm": "neutral", "index_up": "accelerate", "fist": "brake"},
        ),
        actions=data.get(
            "actions",
            {"neutral": "", "accelerate": "w", "brake": "s"},
        ),
        ui=UIConfig(
            show_preview=ui_data.get("show_preview", True),
            show_landmarks=ui_data.get("show_landmarks", True),
            preview_window_name=ui_data.get(
                "preview_window_name", "Hill Climb - Gesture Preview"
            ),
        ),
    )
    validate_config(cfg)
    return cfg
