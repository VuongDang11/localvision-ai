# LocalVision AI

A local, privacy-first home camera system. Multiple cameras stream to a web dashboard, and a vision-language model running **on the device** (no cloud) describes what happens and logs it as events.

## What it does

- **Multi-camera live dashboard**: cameras come from `config.yaml` (video files, webcams or RTSP streams); click a tile to enlarge it.
- **Local VLM analysis**: a background Analyzer sends frames to `qwen3-vl:4b` through Ollama and gets structured JSON back (event type, people present, action, alert, summary).
- **Events panel**: each event has a timestamp, camera, snapshot and a short alert for the homeowner.
- **Manual controls**: take a snapshot or analyze a camera on demand.

## Architecture

```text
config.yaml → VideoSource → FrameGrabber (thread) ─┬─► MJPEG stream → dashboard
                                                   └─► Analyzer (thread) → Ollama VLM
                                                         → consistency checks → EventStore → /api/events
```

## Design decisions

- **Everything runs locally**: frames, model and events never leave the machine.
- **Structured output**: the VLM must return JSON that matches a schema, and code-level consistency checks correct contradictions (for example, a person listed in objects means `people_present` is true).
- **Identity never comes from the VLM**: the prompt forbids guessing who people are or their roles.
- **Graceful degradation**: a broken camera is skipped, and VLM failures are logged without stopping the Analyzer.

## Run it

```bash
pip install -r requirements.txt
ollama pull qwen3-vl:4b
pytest -v
uvicorn app.server:app
```

Open http://127.0.0.1:8000.

## Roadmap

- [x] Multi-camera dashboard
- [x] Local VLM analysis and events
- [ ] Segment recorder (60-second files named by time)
- [ ] SQLite event log and event merging
- [ ] Motion-triggered analysis
- [ ] Familiar-face recognition (enrolled household members only)
- [ ] Natural-language search over events (RAG) with video playback
