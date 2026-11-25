from fastapi.testclient import TestClient

from app.server import app


def test_index_returns_dashboard_html():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "LocalVision AI" in response.text


def test_unknown_camera_returns_404():
    client = TestClient(app)

    response = client.get("/video_feed/not_a_camera")

    assert response.status_code == 404
    
def test_events_starts_empty():
    client = TestClient(app)

    response = client.get("/api/events")

    assert response.status_code == 200
    assert response.json() == []


def test_analyze_without_analyzer_returns_503():
    client = TestClient(app)

    response = client.post("/api/cameras/front_door/analyze")

    assert response.status_code == 503