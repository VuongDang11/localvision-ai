import cv2
import numpy as np
import pytest


@pytest.fixture
def make_test_video(tmp_path):
    """Return a function that writes a tiny test video and returns its path."""

    def _make(frame_count: int = 5, width: int = 64, height: int = 48, fps: float = 10.0):
        video_path = tmp_path / f"test_{frame_count}_frames.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for _ in range(frame_count):
            writer.write(np.full((height, width, 3), 128, dtype=np.uint8))

        writer.release()
        return video_path

    return _make