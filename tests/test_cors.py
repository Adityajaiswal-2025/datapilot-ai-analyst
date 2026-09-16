import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from app.core.config import Settings
from app.core.security import setup_security


def test_assemble_cors_origins_comma_separated_parsing():
    """Verify comma-separated string of origins (including production Render frontend) is parsed into a list."""
    raw_env = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173,https://datapilot-frontend-txcf.onrender.com"
    s = Settings(BACKEND_CORS_ORIGINS=raw_env)
    
    assert isinstance(s.BACKEND_CORS_ORIGINS, list)
    assert len(s.BACKEND_CORS_ORIGINS) == 5
    assert "https://datapilot-frontend-txcf.onrender.com" in s.BACKEND_CORS_ORIGINS
    assert "http://localhost:3000" in s.BACKEND_CORS_ORIGINS


def test_cors_middleware_production_origin_get_request():
    """Verify GET request with configured production Origin receives Access-Control-Allow-Origin & Credentials headers."""
    prod_origin = "https://datapilot-frontend-txcf.onrender.com"
    test_settings = Settings(
        BACKEND_CORS_ORIGINS=f"http://localhost:3000,{prod_origin}"
    )
    
    app = FastAPI()
    # Temporarily bind test_settings for setup_security
    from unittest.mock import patch
    with patch("app.core.security.settings", test_settings):
        setup_security(app)
        
    @app.get("/api/v1/test-cors")
    def sample_endpoint():
        return {"status": "ok"}
        
    client = TestClient(app)
    
    # 1. Test production origin GET
    response = client.get(
        "/api/v1/test-cors",
        headers={"Origin": prod_origin}
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == prod_origin
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_middleware_production_origin_preflight_options():
    """Verify OPTIONS preflight request from production Origin returns HTTP 200 with CORS headers."""
    prod_origin = "https://datapilot-frontend-txcf.onrender.com"
    test_settings = Settings(
        BACKEND_CORS_ORIGINS=f"http://localhost:3000,{prod_origin}"
    )
    
    app = FastAPI()
    from unittest.mock import patch
    with patch("app.core.security.settings", test_settings):
        setup_security(app)
        
    @app.post("/api/v1/test-cors")
    def sample_post():
        return {"status": "created"}
        
    client = TestClient(app)
    
    # Preflight OPTIONS request
    response = client.options(
        "/api/v1/test-cors",
        headers={
            "Origin": prod_origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == prod_origin
    assert response.headers.get("access-control-allow-credentials") == "true"
    assert "POST" in response.headers.get("access-control-allow-methods", "")


def test_cors_middleware_disallowed_origin_blocked():
    """Verify request from an unallowed origin does NOT receive Access-Control-Allow-Origin header."""
    prod_origin = "https://datapilot-frontend-txcf.onrender.com"
    disallowed_origin = "https://unauthorized-domain.com"
    test_settings = Settings(
        BACKEND_CORS_ORIGINS=f"http://localhost:3000,{prod_origin}"
    )
    
    app = FastAPI()
    from unittest.mock import patch
    with patch("app.core.security.settings", test_settings):
        setup_security(app)
        
    @app.get("/api/v1/test-cors")
    def sample_endpoint():
        return {"status": "ok"}
        
    client = TestClient(app)
    
    response = client.get(
        "/api/v1/test-cors",
        headers={"Origin": disallowed_origin}
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
