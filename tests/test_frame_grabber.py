import time

from app.frame_grabber import FrameGrabber
from app.video_source import VideoSource


def test_grabber_provides_latest_frame(make_test_video):
    # Arrange
    video_path = make_test_video(frame_count=30)
    grabber = FrameGrabber(VideoSource(str(video_path), loop=True, realtime=True))

    # Act: let the background thread read for half a second
    grabber.start()
    time.sleep(0.5)
    frame = grabber.get_latest_frame()
    grabber.stop()

    # Assert
    assert frame is not None
    assert frame.shape == (48, 64, 3)
    assert grabber.frames_read > 0


def test_latest_frame_is_none_before_start():
    grabber = FrameGrabber(VideoSource(0))

    assert grabber.get_latest_frame() is None


def test_stop_ends_the_background_thread(make_test_video):
    video_path = make_test_video(frame_count=30)
    grabber = FrameGrabber(VideoSource(str(video_path), loop=True, realtime=True))

    grabber.start()
    assert grabber.is_running()

    grabber.stop()
    assert not grabber.is_running()