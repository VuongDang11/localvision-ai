import threading 

import numpy as np

from app.video_source import VideoSource

class FrameGrabber:
    """ Read frames from a VideoSource in background thread and keep the lastest one"""
    
    def __init__(self, source: VideoSource) -> None:
        self.source = source 
        self.frames_read = 0
        
        self._latest_frame = None
        self._lock = threading.Lock()   # protest _lastest_frame
        self._stop_event = threading.Event()  # tells the thread to stop 
        self._thread =  None 
        
    def start(self) -> None:
        self.source.open()
        self._stop_event.clear()
        
        # daemon=True : the thread will not keep the program alive on exit. 
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
    def _run(self) -> None: 
        # This loop runs in the background thread.
        while not self._stop_event.is_set():
            frame = self.source.read() 

            if frame is None:
                break # the source ended or failed
            
            # Only one thread may touch _latest_frame at a time. 
            with self._lock:
                self._latest_frame = frame
                self.frames_read += 1
                
    def get_latest_frame(self) -> np.ndarray | None:
        """Return a copy of the newest frame, or None if there is none yet."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop_event.set()

        # Wait for the thread to finish BEFORE releasing the source,
        # so we never release the camera while the thread is reading it.
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self.source.release()            
        