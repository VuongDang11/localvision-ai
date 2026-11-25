import pytest

from app.config import PROJECT_ROOT, load_camera_configs


def write_config(tmp_path, text):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def test_loads_file_webcam_and_rtsp_sources(tmp_path):
    config_path = write_config(tmp_path, """
cameras:
  - id: front_door
    name: Front Door
    source: data/sample_videos/front_door.mp4
    loop: true
  - id: desk
    source: 0
  - id: driveway
    source: rtsp://192.168.1.50:554/stream1
""")

    cameras = load_camera_configs(config_path)

    # File path is resolved from the project root.
    assert cameras[0].source == str(PROJECT_ROOT / "data/sample_videos/front_door.mp4")
    assert cameras[0].loop is True

    # Webcam index stays an int; name falls back to the id.
    assert cameras[1].source == 0
    assert cameras[1].name == "desk"

    # RTSP URL is unchanged.
    assert cameras[2].source == "rtsp://192.168.1.50:554/stream1"


def test_duplicate_camera_id_raises_error(tmp_path):
    config_path = write_config(tmp_path, """
cameras:
  - id: front_door
    source: 0
  - id: front_door
    source: 1
""")

    with pytest.raises(ValueError, match="Duplicate"):
        load_camera_configs(config_path)