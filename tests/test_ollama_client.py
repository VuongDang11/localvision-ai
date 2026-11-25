import json

import numpy as np
import pytest
import requests

from app.ollama_client import OllamaClient, OllamaError, fix_inconsistent_event


class FakeResponse:
    """Pretends to be the HTTP response from Ollama."""

    def __init__(self, content: str, status_code: int = 200) -> None:
        self.content_text = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return {"message": {"content": self.content_text}}


@pytest.fixture
def black_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_describe_frame_returns_parsed_json(monkeypatch, black_frame):
    model_answer = {
        "event_type": "delivery",
        "people_present": True,
        "action": "carrying boxes into the house",
        "objects": ["person", "cardboard box"],
        "alert": "Two people are carrying boxes to your front door.",
    }
    sent_payloads = []

    def fake_post(url, json, timeout):
        sent_payloads.append(json)
        return FakeResponse(content=_to_json_text(model_answer))

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)

    result = OllamaClient().describe_frame(black_frame, "Front Door", "2026-11-21T23:30:00")

    assert result == model_answer
    # The request really included an image and the JSON schema.
    assert len(sent_payloads[0]["messages"][0]["images"]) == 1
    assert sent_payloads[0]["format"]["type"] == "object"


def test_http_error_raises_ollama_error(monkeypatch, black_frame):
    def fake_post(url, json, timeout):
        return FakeResponse(content="", status_code=500)

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)

    with pytest.raises(OllamaError, match="request failed"):
        OllamaClient().describe_frame(black_frame, "Front Door", "2026-11-21T23:30:00")


def test_invalid_json_raises_ollama_error(monkeypatch, black_frame):
    def fake_post(url, json, timeout):
        return FakeResponse(content="this is not json")

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)

    with pytest.raises(OllamaError, match="invalid JSON"):
        OllamaClient().describe_frame(black_frame, "Front Door", "2026-11-21T23:30:00")


def _to_json_text(data: dict) -> str:
    return json.dumps(data)

def test_child_in_objects_means_people_present():
    model_answer = {"people_present": False, "event_type": "animal", "objects": ["dog", "child"]}

    fixed_event = fix_inconsistent_event(model_answer)

    assert fixed_event["people_present"] is True
    assert fixed_event["event_type"] == "person"
    
def test_describe_frame_applies_consistency_fixes(monkeypatch, black_frame):
    # The model wrongly says no people, while listing a child.
    model_answer = {
        "event_type": "animal",
        "people_present": False,
        "action": "playing",
        "objects": ["dog", "child"],
        "alert": "Animal in backyard",
        "summary": "A child plays with a dog.",
    }

    def fake_post(url, json, timeout):
        return FakeResponse(content=_to_json_text(model_answer))

    monkeypatch.setattr("app.ollama_client.requests.post", fake_post)

    result = OllamaClient().describe_frame(black_frame, "Back Yard", "2026-11-24T16:00:00")

    assert result["people_present"] is True
    assert result["event_type"] == "person"
    assert result["alert"] == "Animal in backyard."