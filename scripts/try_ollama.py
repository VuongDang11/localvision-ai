import json
import sys
import time
from datetime import datetime

import cv2

from app.ollama_client import OllamaClient


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m scripts.try_ollama <image_path>")

    image_path = sys.argv[1]
    camera_name = sys.argv[2] if len(sys.argv) > 2 else "Front Door"
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"Cannot read image: {image_path}")

    client = OllamaClient()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    start_time = time.perf_counter()
    result = client.describe_frame(frame, camera_name=camera_name, timestamp=timestamp)
    elapsed_seconds = time.perf_counter() - start_time

    print(json.dumps(result, indent=2))
    print(f"Took {elapsed_seconds:.1f} seconds")


if __name__ == "__main__":
    main()