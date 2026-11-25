# Import Libraries
from dataclasses import dataclass
from pathlib import Path
import yaml

# Define the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# Defines the structure of each camera configuration
@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    name: str
    source: int | str
    loop: bool = False
    realtime: bool = False
    

def resolve_source(raw_source: int | str) -> int | str:
    """Turn a config source into something VideoSource can open."""
    # Webcam index: keep as an integer.
    if isinstance(raw_source, int):
        return raw_source

    # Stream URL (rtsp://, http://): keep as it is.
    if "://" in raw_source:
        return raw_source

    # File path: make relative paths start from the project root,
    # so the program works no matter which folder it is launched from.
    path = Path(raw_source)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)



def load_camera_configs(config_path: Path = DEFAULT_CONFIG_PATH) -> list[CameraConfig]:
    """Read the camera list from a YAML config file."""
    with open(config_path, encoding="utf-8") as config_file:
        # safe_load only builds plain data (dicts, lists, strings, numbers).
        config_data = yaml.safe_load(config_file) or {}

    cameras = []
    seen_camera_ids = set()

    for entry in config_data.get("cameras", []):
        camera_id = entry["id"]

        if camera_id in seen_camera_ids:
            raise ValueError(f"Duplicate camera id in config: {camera_id}")
        seen_camera_ids.add(camera_id)

        cameras.append(
            CameraConfig(
                camera_id=camera_id,
                name=entry.get("name", camera_id),
                source=resolve_source(entry["source"]),
                loop=entry.get("loop", False),
                realtime=entry.get("realtime", False),
            )
        )

    if not cameras:
        raise ValueError(f"No cameras defined in {config_path}")

    return cameras