import time

import cv2
import numpy as np


class VideoSource:
    """Open a webcam, video file, or RTSP stream and read frames from it."""

    def __init__(
        self,
        source: int | str = 0,
        loop: bool = False,
        realtime: bool = False,
    ) -> None:
        # int -> webcam index (0 = default webcam)
        # str -> video file path or RTSP URL
        self.source = source

        # loop: rewind to the start when a video file ends
        # realtime: play a video file at its real speed, not as fast as possible
        self.loop = loop
        self.realtime = realtime

        self.capture = None
        self.frame_interval_seconds = 0.0
        self.last_frame_time = None

    def open(self) -> None:
        if isinstance(self.source, int):
            # On Windows, DirectShow opens webcams faster.
            self.capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)

            # Fall back to the default backend if DirectShow fails.
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = cv2.VideoCapture(self.source)
        else:
            # Files and RTSP URLs use the default backend.
            self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        if self.realtime:
            fps = self.capture.get(cv2.CAP_PROP_FPS)

            # Some sources report 0 or nonsense FPS, so fall back to 30.
            if fps <= 0 or fps > 120:
                fps = 30.0

            self.frame_interval_seconds = 1.0 / fps

    def read(self) -> np.ndarray | None:
        """Return the next frame, or None if no frame is available."""
        if self.capture is None:
            raise RuntimeError("Video source is not open. Call open() first.")

        success, frame = self.capture.read()

        if not success and self.loop:
            # Rewind to the first frame and try once more.
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.capture.read()

        if not success:
            return None

        if self.realtime:
            self._wait_for_next_frame()

        return frame

    def _wait_for_next_frame(self) -> None:
        """Sleep only for the time left until the next frame is due."""
        now = time.perf_counter()

        if self.last_frame_time is not None:
            elapsed_seconds = now - self.last_frame_time
            remaining_seconds = self.frame_interval_seconds - elapsed_seconds

            if remaining_seconds > 0:
                time.sleep(remaining_seconds)

        self.last_frame_time = time.perf_counter()

    def release(self) -> None:
        # Safe to call more than once.
        if self.capture is not None:
            self.capture.release()
            self.capture = None