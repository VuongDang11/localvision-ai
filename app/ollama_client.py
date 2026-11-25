import base64
import json

import cv2
import numpy as np
import requests

from app.frame_utils import resize_to_fit

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-vl:4b"

# The model must return JSON that matches this schema.
EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {
            "type": "string",
            "enum": ["none", "person", "package", "vehicle", "animal", "other"],
        },
        "people_present": {"type": "boolean"},
        "action": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "alert": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": [
        "event_type",
        "people_present",
        "action",
        "objects",
        "alert",
        "summary",
    ],
}

class OllamaError(Exception):
    """Raised when Ollama fails or returns output we cannot use."""


def encode_frame_as_base64_jpeg(frame: np.ndarray, max_side: int = 768) -> str:
    """Shrink the frame, encode it as JPEG, and return it as a base64 string."""
    small_frame = resize_to_fit(frame, max_width=max_side, max_height=max_side)

    success, jpeg_buffer = cv2.imencode(
        ".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
    )
    if not success:
        raise OllamaError("Could not encode frame as JPEG")

    return base64.b64encode(jpeg_buffer.tobytes()).decode("ascii")


def build_prompt(camera_name: str, timestamp: str) -> str:
    return (
        f"Home security camera: {camera_name}. Time: {timestamp}.\n"
        "Describe only what is clearly visible. Never guess who someone is.\n"
        "event_type: package (someone handling a box or package), person, "
        "vehicle, animal, none (nobody and nothing moving), or other.\n"
        "people_present: true if any human is visible, even next to an animal.\n"
        "If a person is visible, event_type is person or package, never animal.\n"
        "action: a short phrase, like 'carrying a box' or 'playing with a dog'.\n"
        "Never invent names.\n"
        "alert: a short notification for the homeowner, "
        "like 'Person carrying a box at the front door.'\n"
        "summary: one or two factual sentences describing the scene.\n"
        "Do not assume roles like delivery person. Do not give advice."
    )
 
HUMAN_WORDS = ("person", "people", "child", "man", "woman", "girl", "boy")


def fix_inconsistent_event(event: dict) -> dict:
    """Correct answers where the fields contradict each other."""
    fixed_event = dict(event)

    # A human appears in the object list, so people are present.
    objects_text = " ".join(fixed_event.get("objects", [])).lower()
    if any(word in objects_text for word in HUMAN_WORDS):
        fixed_event["people_present"] = True

    # People are visible, so the event is about the person, not "none" or "animal".
    if fixed_event.get("people_present") and fixed_event.get("event_type") in ("none", "animal"):
        fixed_event["event_type"] = "person"

    if "alert" in fixed_event:
        fixed_event["alert"] = clean_alert(fixed_event["alert"])

    return fixed_event   



MAX_ALERT_LENGTH = 100


def clean_alert(alert: str) -> str:
    """Keep only the first sentence of the alert and cap its length."""
    first_sentence = alert.strip().split(". ")[0].rstrip(".")

    if len(first_sentence) > MAX_ALERT_LENGTH:
        first_sentence = first_sentence[:MAX_ALERT_LENGTH].rsplit(" ", 1)[0]

    return first_sentence + "."

class OllamaClient:
    """Send camera frames to a local Ollama vision model and get structured JSON back."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def describe_frame(self, frame: np.ndarray, camera_name: str, timestamp: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": build_prompt(camera_name, timestamp),
                    "images": [encode_frame_as_base64_jpeg(frame)],
                }
            ],
            "format": EVENT_SCHEMA,          # force JSON that matches the schema
            "stream": False,                 # wait for the full answer
            "think": False,                  # skip the slow "thinking" step
            "options": {"temperature": 0},   # consistent, less creative answers
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise OllamaError(f"Ollama request failed: {error}") from error

        response_data = response.json()
        content = response_data["message"]["content"].strip()

        # Small models sometimes return nothing. Say so clearly, with the reason.
        if not content:
            done_reason = response_data.get("done_reason", "unknown")
            raise OllamaError(f"Model returned an empty answer (done_reason: {done_reason})")

        try:
            event = json.loads(content)
        except json.JSONDecodeError as error:
            raise OllamaError(f"Model returned invalid JSON: {content[:200]}") from error

        # Apply the consistency rules before anyone sees the answer.
        return fix_inconsistent_event(event)