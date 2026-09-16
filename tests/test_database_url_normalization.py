import pytest
from app.core.config import Settings, normalize_database_url


def test_normalize_database_url_postgresql_unadorned():
    """Verify postgresql:// without explicit driver is converted to postgresql+asyncpg://."""
    raw_url = "postgresql://user:password@localhost:5432/datapilot"
    normalized = normalize_database_url(raw_url)
    assert normalized == "postgresql+asyncpg://user:password@localhost:5432/datapilot"


def test_normalize_database_url_postgres_legacy():
    """Verify legacy postgres:// (Render/Heroku standard) is converted to postgresql+asyncpg://."""
    raw_url = "postgres://user:password@localhost:5432/datapilot"
    normalized = normalize_database_url(raw_url)
    assert normalized == "postgresql+asyncpg://user:password@localhost:5432/datapilot"


def test_normalize_database_url_already_correct():
    """Verify postgresql+asyncpg:// remains unchanged."""
    raw_url = "postgresql+asyncpg://user:password@localhost:5432/datapilot"
    normalized = normalize_database_url(raw_url)
    assert normalized == "postgresql+asyncpg://user:password@localhost:5432/datapilot"


def test_normalize_database_url_sqlite_preservation():
    """Verify SQLite URLs (sqlite:// and sqlite+aiosqlite://) remain unchanged."""
    sqlite_sync = "sqlite:///./datapilot.db"
    assert normalize_database_url(sqlite_sync) == "sqlite:///./datapilot.db"

    sqlite_async = "sqlite+aiosqlite:///./datapilot.db"
    assert normalize_database_url(sqlite_async) == "sqlite+aiosqlite:///./datapilot.db"


def test_settings_database_url_validation(monkeypatch):
    """Verify Settings class normalizes DATABASE_URL supplied via environment or instantiation."""
    s_postgres = Settings(DATABASE_URL="postgres://render_user:secret@dpg-host.render.com/datapilot_db")
    assert s_postgres.DATABASE_URL == "postgresql+asyncpg://render_user:secret@dpg-host.render.com/datapilot_db"

    s_postgresql = Settings(DATABASE_URL="postgresql://render_user:secret@dpg-host.render.com/datapilot_db")
    assert s_postgresql.DATABASE_URL == "postgresql+asyncpg://render_user:secret@dpg-host.render.com/datapilot_db"
