import cv2
import numpy as np
import pytest

from app.video_source import VideoSource


def create_test_video(path, frame_count=5, width=64, height=48):
    # Write a tiny video file made of solid gray frames.
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    for _ in range(frame_count):
        writer.write(np.full((height, width, 3), 128, dtype=np.uint8))
    writer.release()


def test_reads_all_frames_from_video_file(tmp_path):
    # Arrange
    video_path = tmp_path / "test.avi"
    create_test_video(video_path, frame_count=5)
    source = VideoSource(str(video_path))

    # Act
    source.open()
    frames = []
    while (frame := source.read()) is not None:
        frames.append(frame)
    source.release()

    # Assert
    assert len(frames) == 5
    assert frames[0].shape == (48, 64, 3)


def test_invalid_source_raises_error():
    source = VideoSource("does_not_exist.mp4")

    with pytest.raises(RuntimeError):
        source.open()
    
def test_read_before_open_raises_error():
    # Arrange: Create a source but do NOT call open()
    source = VideoSource(0)
    
    with pytest.raises(RuntimeError):
        source.read()
        
import time


def test_loop_restarts_video_when_it_ends(tmp_path):
    # Arrange: a video with only 5 frames
    video_path = tmp_path / "short.avi"
    create_test_video(video_path, frame_count=5)
    source = VideoSource(str(video_path), loop=True)
    source.open()

    # Act: read more frames than the video contains
    frames = [source.read() for _ in range(12)]
    source.release()

    # Assert: every read returned a frame, because the video looped
    assert all(frame is not None for frame in frames)


def test_realtime_plays_at_video_fps(tmp_path):
    # Arrange: 5 frames written at 10 FPS -> 0.1 s between frames
    video_path = tmp_path / "paced.avi"
    create_test_video(video_path, frame_count=5)
    source = VideoSource(str(video_path), realtime=True)
    source.open()

    # Act: time how long it takes to read all 5 frames
    start_time = time.perf_counter()
    for _ in range(5):
        source.read()
    elapsed_seconds = time.perf_counter() - start_time
    source.release()

    # Assert: 4 gaps x 0.1 s = 0.4 s (small tolerance for timer precision)
    assert elapsed_seconds >= 0.35
        