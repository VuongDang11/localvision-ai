import time
import numpy as np

from pathlib import Path
from app.analyzer import Analyzer
from app.event_store import EventStore
from app.ollama_client import OllamaError


class FakeGrabber:
    """Always returns the same black frame."""

    def get_latest_frame(self):
        return np.zeros((48, 64, 3), dtype=np.uint8)


class FakeVlmClient:
    """Returns a fixed answer, or raises an error, without calling Ollama."""

    def __init__(self, answer: dict | None = None, should_fail: bool = False) -> None:
        self.answer = answer
        self.should_fail = should_fail

    def describe_frame(self, frame, camera_name, timestamp):
        if self.should_fail:
            raise OllamaError("Ollama is not running")
        return self.answer


def make_analyzer(tmp_path, vlm_client) -> tuple[Analyzer, EventStore]:
    event_store = EventStore()
    analyzer = Analyzer(
        grabbers={"front_door": FakeGrabber()},
        camera_names={"front_door": "Front Door"},
        vlm_client=vlm_client,
        event_store=event_store,
        snapshot_dir=tmp_path,
    )
    return analyzer, event_store


def test_analyze_camera_stores_event_with_time_and_snapshot(tmp_path):
    vlm_answer = {"event_type": "package", "alert": "Person carrying a box at the front door."}
    analyzer, event_store = make_analyzer(tmp_path, FakeVlmClient(answer=vlm_answer))

    event = analyzer.analyze_camera("front_door")

    assert event["camera_id"] == "front_door"
    assert event["event_type"] == "package"
    assert "timestamp" in event
    assert Path(event["snapshot_path"]).is_file()
    assert event_store.list_recent() == [event]
    assert event["trigger"] == "auto"


def test_none_event_is_not_stored(tmp_path):
    analyzer, event_store = make_analyzer(tmp_path, FakeVlmClient(answer={"event_type": "none"}))

    assert analyzer.analyze_camera("front_door") is None
    assert event_store.list_recent() == []


def test_vlm_failure_does_not_crash(tmp_path):
    analyzer, event_store = make_analyzer(tmp_path, FakeVlmClient(should_fail=True))

    assert analyzer.analyze_camera("front_door") is None
    assert event_store.list_recent() == []


def test_event_store_returns_newest_first_and_drops_oldest():
    event_store = EventStore(max_events=2)

    event_store.add({"alert": "first"})
    event_store.add({"alert": "second"})
    event_store.add({"alert": "third"})

    alerts = [event["alert"] for event in event_store.list_recent()]
    assert alerts == ["third", "second"]
    
    
class CrashingGrabber:
    def get_latest_frame(self):
        raise KeyError("unexpected")


def test_unexpected_error_does_not_kill_the_thread(tmp_path):
    analyzer = Analyzer(
        grabbers={"broken": CrashingGrabber()},
        camera_names={},
        vlm_client=FakeVlmClient(answer={"event_type": "person"}),
        event_store=EventStore(),
        snapshot_dir=tmp_path,
        interval_seconds=0.05,
    )

    analyzer.start()
    time.sleep(0.3)          # several loops, each raising KeyError
    still_running = analyzer.is_running()
    analyzer.stop()

    assert still_running    