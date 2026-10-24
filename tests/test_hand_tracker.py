"""Unit tests for HandTracker and landmark rendering."""

from unittest.mock import MagicMock

import numpy as np

from hill_climb.config import TrackingConfig, UIConfig
from hill_climb.hand_tracker import (
    HAND_CONNECTIONS,
    HandDetectionResult,
    HandLandmark,
    HandTracker,
)
from hill_climb.ui import UIOverlay


class MockMediaPipeLandmark:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class MockCategory:
    def __init__(self, category_name: str, score: float) -> None:
        self.category_name = category_name
        self.score = score


class MockLandmarkerResult:
    def __init__(
        self,
        landmarks: list[list[MockMediaPipeLandmark]] | None = None,
        handedness: list[list[MockCategory]] | None = None,
    ) -> None:
        self.hand_landmarks = landmarks or []
        self.handedness = handedness or []


def create_synthetic_landmarks() -> list[HandLandmark]:
    return [HandLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]


def test_hand_connections_integrity() -> None:
    assert len(HAND_CONNECTIONS) == 21
    for start, end in HAND_CONNECTIONS:
        assert 0 <= start < 21
        assert 0 <= end < 21
        assert start != end


def test_hand_tracker_no_hand_detected() -> None:
    mock_landmarker = MagicMock()
    mock_landmarker.detect_for_video.return_value = MockLandmarkerResult(landmarks=[])

    cfg = TrackingConfig(model_path="models/hand_landmarker.task")
    tracker = HandTracker(cfg, landmarker_factory=lambda _: mock_landmarker)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = tracker.process_frame(frame)

    assert isinstance(result, HandDetectionResult)
    assert not result.has_hand
    assert len(result.landmarks) == 0

    tracker.close()
    mock_landmarker.close.assert_called_once()


def test_hand_tracker_hand_detected() -> None:
    mock_raw_landmarks = [
        MockMediaPipeLandmark(x=i * 0.04, y=i * 0.03, z=0.01) for i in range(21)
    ]
    mock_handedness = [[MockCategory(category_name="Right", score=0.98)]]
    mock_result = MockLandmarkerResult(
        landmarks=[mock_raw_landmarks],
        handedness=mock_handedness,
    )

    mock_landmarker = MagicMock()
    mock_landmarker.detect_for_video.return_value = mock_result

    cfg = TrackingConfig(model_path="models/hand_landmarker.task")
    tracker = HandTracker(cfg, landmarker_factory=lambda _: mock_landmarker)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = tracker.process_frame(frame)

    assert result.has_hand is True
    assert len(result.landmarks) == 21
    assert result.handedness == "Right"
    assert result.score == 0.98
    assert result.landmarks[0].x == 0.0
    assert abs(result.landmarks[1].x - 0.04) < 1e-5


def test_ui_draw_landmarks_on_frame() -> None:
    ui = UIOverlay(UIConfig(show_landmarks=True))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks = create_synthetic_landmarks()

    ui.draw_landmarks(frame, landmarks)
    # The canvas should have landmark drawings applied
    assert np.any(frame > 0)


def test_ui_draw_preview_with_detection() -> None:
    ui = UIOverlay(UIConfig(show_landmarks=True))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detection = HandDetectionResult(
        has_hand=True,
        landmarks=create_synthetic_landmarks(),
        handedness="Left",
        score=0.95,
    )

    rendered = ui.draw_preview(frame, fps=30.0, detection=detection)
    assert rendered.shape == frame.shape
    assert np.any(rendered > 0)
