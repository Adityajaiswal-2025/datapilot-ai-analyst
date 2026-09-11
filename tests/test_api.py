from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root '/' endpoint returns 200 OK and expected app metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "DataPilot"
    assert data["status"] == "online"
    assert "version" in data


def test_health_endpoint():
    """Test the '/api/v1/health' endpoint returns 200 OK and health response structure."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["status"] == "healthy"
    assert "version" in data
    assert "services" in data
    assert data["services"]["api"] == "operational"
