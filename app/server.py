# Import the libraries
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.analyzer import Analyzer
from app.config import CameraConfig, load_camera_configs
from app.event_store import EventStore
from app.frame_grabber import FrameGrabber
from app.frame_utils import resize_to_fit
from app.ollama_client import OllamaClient
from app.video_source import VideoSource 

# Settings File Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_HTML_PATH = (
    Path(__file__).resolve().parent / "templates" / "index.html"
)

SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Shared, in-memory event log (Week 2: SQLite).
event_store = EventStore()

# Init the FrameGrabber per camera  Example: grabbers = { "font_door" : front_grabber, "backyard": backyard_grabber}
grabbers: dict[str, FrameGrabber] = {}

# Camera settings from config.yaml, looked up by camera_id
camera_configs : dict[str, CameraConfig] = {}


# Manage server startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create and start one grabber per camera in config.yaml.
    for camera in load_camera_configs():
        grabber = FrameGrabber(
            VideoSource(camera.source, loop=camera.loop, realtime=camera.realtime)
        )

        try:
            grabber.start()
        except RuntimeError as error:
            # One broken camera should not take down the whole system.
            print(f"[WARN] Skipping camera '{camera.camera_id}': {error}")
            continue

        grabbers[camera.camera_id] = grabber
        camera_configs[camera.camera_id] = camera

    # ── NEW ─────────────────────────────────────────────────────────
    # Start the Analyzer after the cameras, so it has frames to read.
    app.state.analyzer = Analyzer(
        grabbers=grabbers,
        camera_names={camera_id: camera.name for camera_id, camera in camera_configs.items()},
        vlm_client=OllamaClient(),
        event_store=event_store,
        snapshot_dir=SNAPSHOT_DIR,
        interval_seconds=10.0,
    )
    app.state.analyzer.start()
    # ────────────────────────────────────────────────────────────────

    yield

    # ── CHANGED: stop the Analyzer FIRST, then the cameras it reads from ──
    app.state.analyzer.stop()
    app.state.analyzer = None

    for grabber in grabbers.values():
        grabber.stop()
    grabbers.clear()
    camera_configs.clear()


# Create the FastAPI Application
app = FastAPI(title="LocalVision_AI", lifespan=lifespan)

# ── NEW (step 1d) ───────────────────────────────────────────────────
# Serve snapshot images at /snapshots/<camera_id>/<file>.jpg
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")


def to_snapshot_url(snapshot_path: str | Path) -> str:
    """Turn a snapshot file path into a URL the browser can load."""
    relative_path = Path(snapshot_path).relative_to(SNAPSHOT_DIR)
    return f"/snapshots/{relative_path.as_posix()}"


# Defination of MJPEG
def generate_mjpeg(grabber: FrameGrabber, stream_fps: float = 15.0):
    """Yield the latest frame as JPEG parts of a multipart MJPEG stream."""
    frame_interval_seconds = 1.0 / stream_fps

    while True:
        frame = grabber.get_latest_frame()

        if frame is None:
            # The camera has not produced a frame yet. Wait briefly.
            time.sleep(0.05)
            continue

        # Shrink for the browser to save bandwidth.
        display_frame = resize_to_fit(frame, max_width=960, max_height=720)

        success, jpeg_buffer = cv2.imencode(
            ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_buffer.tobytes()
            + b"\r\n"
        )

        time.sleep(frame_interval_seconds)

# Avoid leaking local files
def to_public_event(event: dict) -> dict:
    """Prepare an event for the browser: add the URL, hide the local file path."""
    public_event = {key: value for key, value in event.items() if key != "snapshot_path"}
    public_event["snapshot_url"] = to_snapshot_url(event["snapshot_path"])
    return public_event



@app.get("/")
def index():
    return FileResponse(INDEX_HTML_PATH)


@app.get("/video_feed/{camera_id}")
def video_feed(camera_id: str):
    grabber = grabbers.get(camera_id)

    if grabber is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")

    return StreamingResponse(
        generate_mjpeg(grabber),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/health")
def health():
    analyzer = getattr(app.state, "analyzer", None)
    return {
        "analyzer_running": analyzer is not None and analyzer.is_running(),
        "cameras": {
            camera_id: {
                "running": grabber.is_running(),
                "frames_read": grabber.frames_read,
            }
            for camera_id, grabber in grabbers.items()
        },
    }

@app.get("/api/cameras")
def list_cameras():
    return [
        {"id": camera.camera_id, "name": camera.name}
        for camera in camera_configs.values()
    ]

@app.post("/api/cameras/{camera_id}/snapshot")
def take_snapshot(camera_id: str):
    grabber = grabbers.get(camera_id)
    if grabber is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")

    frame = grabber.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera has no frame yet")

    camera_dir = SNAPSHOT_DIR / camera_id
    camera_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().astimezone()
    snapshot_path = camera_dir / f"manual_{now:%Y%m%d_%H%M%S_%f}.jpg"

    if not cv2.imwrite(str(snapshot_path), frame):
        raise HTTPException(status_code=500, detail="Failed to save snapshot")

    return {"snapshot_url": to_snapshot_url(snapshot_path)}


@app.post("/api/cameras/{camera_id}/analyze")
def analyze_now(camera_id: str):
    analyzer = getattr(app.state, "analyzer", None)
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer is not running")

    if camera_id not in grabbers:
        raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")

    # Takes several seconds. FastAPI runs plain `def` endpoints in a thread pool,
    # so the video streams keep playing while this waits.
    event = analyzer.analyze_camera(camera_id, trigger="manual")

    if event is None:
        return {"event": None, "message": "Nothing happening, or the VLM failed. Check the server log."}

    return {"event": to_public_event(event)}

@app.get("/api/events")
def list_events(limit: int = 20):
    return [to_public_event(event) for event in event_store.list_recent(limit)]