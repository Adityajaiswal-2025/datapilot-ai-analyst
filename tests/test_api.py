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


def test_debug_default_and_override():
    """Test that DEBUG defaults to False in Settings, and can be explicitly overridden."""
    from app.core.config import Settings

    # Default Settings (with no parameters) must have DEBUG=False
    default_settings = Settings(_env_file=None)
    assert default_settings.DEBUG is False

    # Explicit override for local development
    dev_settings = Settings(_env_file=None, DEBUG=True)
    assert dev_settings.DEBUG is True


def test_cors_origin_parsing():
    """Test parsing of BACKEND_CORS_ORIGINS as default list, comma-separated string, and list with trailing slashes."""
    from app.core.config import Settings
    from app.core.security import setup_security
    from fastapi import FastAPI

    # 1. Default origins
    default_settings = Settings(_env_file=None)
    assert "http://localhost:5173" in default_settings.BACKEND_CORS_ORIGINS

    # 2. Comma-separated string parsing
    custom_settings = Settings(
        _env_file=None,
        BACKEND_CORS_ORIGINS="https://datapilot.app, https://analytics.company.com/"
    )
    assert custom_settings.BACKEND_CORS_ORIGINS == [
        "https://datapilot.app",
        "https://analytics.company.com/"
    ]


def test_cors_middleware_behavior():
    """Test CORSMiddleware with custom production origins and preflight requests."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.core.config import Settings
    from app.core.security import setup_security
    import app.core.security

    test_app = FastAPI()

    @test_app.get("/test-cors")
    def sample_route():
        return {"msg": "ok"}

    # Mock settings with explicit custom production origins
    orig_origins = app.core.security.settings.BACKEND_CORS_ORIGINS
    try:
        app.core.security.settings.BACKEND_CORS_ORIGINS = [
            "https://app.datapilot.io",
            "https://dashboard.company.com/"
        ]
        setup_security(test_app)

        tc = TestClient(test_app)
        
        # Preflight request from allowed custom origin
        res = tc.options(
            "/test-cors",
            headers={
                "Origin": "https://app.datapilot.io",
                "Access-Control-Request-Method": "GET",
            }
        )
        assert res.status_code == 200
        assert res.headers.get("access-control-allow-origin") == "https://app.datapilot.io"
        assert res.headers.get("access-control-allow-credentials") == "true"

        # Preflight request from unauthorized origin
        res_bad = tc.options(
            "/test-cors",
            headers={
                "Origin": "https://evil-hacker.com",
                "Access-Control-Request-Method": "GET",
            }
        )
        assert res_bad.headers.get("access-control-allow-origin") is None
    finally:
        app.core.security.settings.BACKEND_CORS_ORIGINS = orig_origins


def test_wildcard_cors_disables_credentials():
    """Test that wildcard '*' origin disables allow_credentials=True."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.core.security import setup_security
    import app.core.security

    test_app = FastAPI()

    @test_app.get("/test-wildcard")
    def sample_route():
        return {"msg": "ok"}

    orig_origins = app.core.security.settings.BACKEND_CORS_ORIGINS
    try:
        app.core.security.settings.BACKEND_CORS_ORIGINS = ["*"]
        setup_security(test_app)

        tc = TestClient(test_app)
        res = tc.options(
            "/test-wildcard",
            headers={
                "Origin": "https://any-domain.com",
                "Access-Control-Request-Method": "GET",
            }
        )
        # allow-credentials must NOT be true when wildcard is configured
        assert res.headers.get("access-control-allow-credentials") != "true"
    finally:
        app.core.security.settings.BACKEND_CORS_ORIGINS = orig_origins

