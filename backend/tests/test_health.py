from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert data.get("app") == "Test Center"
    assert "data_dir" in data
    assert "test_data_dir" in data


def test_api_404():
    response = client.get("/api/nonexistent_route")
    assert response.status_code == 404
