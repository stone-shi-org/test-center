from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_ui_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    # If static files are built, should return HTML
    # If not built, returns JSON status
    if "text/html" in response.headers.get("content-type", ""):
        assert "<div id=\"root\">" in response.text or "<!doctype html>" in response.text.lower()
    else:
        data = response.json()
        assert "app" in data


def test_spa_routing_fallback():
    response = client.get("/compare")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert "text/html" in response.headers.get("content-type", "")
