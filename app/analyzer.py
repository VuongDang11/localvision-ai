import threading
from datetime import datetime
from pathlib import Path

import cv2

from app.event_store import EventStore
from app.ollama_client import OllamaError


class Analyzer:
    """Every few seconds, send the latest frame from each camera to the VLM and store events."""

    def __init__(
        self,
        grabbers: dict,
        camera_names: dict[str, str],
        vlm_client,
        event_store: EventStore,
        snapshot_dir: Path,
        interval_seconds: float = 10.0,
    ) -> None:
        self.grabbers = grabbers
        self.camera_names = camera_names
        self.vlm_client = vlm_client          # injected, so tests can pass a fake
        self.event_store = event_store
        self.snapshot_dir = Path(snapshot_dir)
        self.interval_seconds = interval_seconds

        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            # A VLM call may still be running; daemon=True lets the program exit anyway.
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # Cameras take turns, because there is only one GPU and one model.
            for camera_id in list(self.grabbers):
                if self._stop_event.is_set():
                    break
                try:
                    self.analyze_camera(camera_id)
                except Exception as error:
                    # One bad frame must never kill the background thread.
                    print(f"[ERROR] Analyzer failed on camera '{camera_id}': {error!r}")

            # wait() instead of time.sleep(), so stop() wakes the thread immediately.
            self._stop_event.wait(self.interval_seconds)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def analyze_camera(self, camera_id: str, trigger: str = 'auto') -> dict | None:
        """Analyze one frame from one camera. Return the stored event, or None."""
        frame = self.grabbers[camera_id].get_latest_frame()
        if frame is None:
            return None

        # Time comes from the system clock, captured when the frame was taken.
        event_time = datetime.now().astimezone()
        camera_name = self.camera_names.get(camera_id, camera_id)

        try:
            vlm_result = self.vlm_client.describe_frame(
                frame, camera_name, event_time.isoformat(timespec="seconds")
            )
        except OllamaError as error:
            print(f"[WARN] VLM failed for camera '{camera_id}': {error}")
            return None

        # Nothing is happening, so there is nothing to log.
        if vlm_result.get("event_type") == "none":
            return None

        snapshot_path = self._save_snapshot(camera_id, frame, event_time)

        event = {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "timestamp": event_time.isoformat(timespec="seconds"),
            "trigger": trigger,      # "auto" (Analyzer loop) or "manual" (button)
            "snapshot_path": str(snapshot_path),
            **vlm_result,
        }
        
        return self.event_store.add(event)

    def _save_snapshot(self, camera_id: str, frame, event_time: datetime) -> Path:
        camera_dir = self.snapshot_dir / camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = camera_dir / f"{event_time:%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        return snapshot_path