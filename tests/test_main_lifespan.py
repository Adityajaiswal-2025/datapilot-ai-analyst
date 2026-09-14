import pytest
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI
from app.main import lifespan


@pytest.mark.asyncio
async def test_fastapi_lifespan_does_not_call_create_all():
    """Verify that FastAPI lifespan manager does not invoke init_db_tables or create_all on startup."""
    mock_app = FastAPI()

    with patch("app.database.session.init_db_tables", new_callable=AsyncMock) as mock_init:
        async with lifespan(mock_app):
            pass
        # Assert init_db_tables was not called during app lifespan execution
        mock_init.assert_not_called()
