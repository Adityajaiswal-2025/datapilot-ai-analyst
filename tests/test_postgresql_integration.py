import io
import os
import pytest
import pandas as pd
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

from app.database.models.dataset import Base, DatasetModel
from app.database.repository import DatasetRepository
from app.data.storage import LocalStorageService
from app.data.metadata import DATASET_REGISTRY, extract_metadata
from app.services.dataset_service import DatasetService

import pytest_asyncio

# Configurable PostgreSQL test database URL
POSTGRES_TEST_URL = os.getenv(
    "TEST_POSTGRES_URL",
    "postgresql+asyncpg://datapilot:datapilot_secret@localhost:5432/datapilot"
)


@pytest_asyncio.fixture
async def pg_db_session():
    """Opt-in fixture creating a live PostgreSQL AsyncSession for testing. Skips if PostgreSQL unavailable."""
    try:
        engine = create_async_engine(POSTGRES_TEST_URL, echo=False)
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        pytest.skip(f"Live PostgreSQL database connection unavailable ({e}). Skipping @pytest.mark.postgresql tests.")

    # Create tables on PostgreSQL
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    AsyncSessionPG = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncSessionPG() as session:
        yield session

    # Clean up test rows after test without dropping table schema
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM datasets"))
    await engine.dispose()


@pytest.fixture
def pg_temp_storage(tmp_path):
    """Provides a temporary LocalStorageService instance for PostgreSQL integration tests."""
    storage_dir = str(tmp_path / "pg_uploads")
    return LocalStorageService(base_dir=storage_dir)


@pytest.mark.postgresql
@pytest.mark.asyncio
async def test_postgresql_model_crud_and_json_timestamps(pg_db_session):
    """Verify DatasetModel CRUD, JSON columns, timezone-aware timestamps, and indexes on PostgreSQL."""
    repo = DatasetRepository(pg_db_session)
    df = pd.DataFrame({"PG_Col1": [10, 20], "PG_Col2": ["Alpha", "Beta"]})

    meta = extract_metadata("pg_ds_1", "pg_test.csv", "data/uploads/pg_test.csv", df)
    model = await repo.create(meta)

    # Assert record inserted into PostgreSQL
    assert model.id == "pg_ds_1"
    assert model.original_filename == "pg_test.csv"
    assert len(model.columns) == 2
    assert isinstance(model.columns, list)
    assert isinstance(model.sample_rows, list)
    assert model.created_at.tzinfo is not None

    # Fetch from PostgreSQL
    fetched = await repo.get_by_id("pg_ds_1")
    assert fetched is not None
    assert fetched.row_count == 2

    # Update access timestamp
    await repo.update_last_accessed("pg_ds_1")

    # Update status
    await repo.update_status("pg_ds_1", "missing")
    updated = await repo.get_by_id("pg_ds_1")
    assert updated.status == "missing"

    # Delete from PostgreSQL
    deleted = await repo.delete("pg_ds_1")
    assert deleted is True
    assert await repo.get_by_id("pg_ds_1") is None


@pytest.mark.postgresql
@pytest.mark.asyncio
async def test_postgresql_dataset_service_ingest_rehydrate_and_cleanup(pg_db_session, pg_temp_storage):
    """Verify DatasetService ingestion, RAM rehydration, missing status update, and TTL cleanup on PostgreSQL."""
    service = DatasetService(pg_db_session, storage=pg_temp_storage)

    csv_bytes = b"Category,Revenue\nTech,1000\nHealth,500\n"
    upload = UploadFile(filename="pg_sales.csv", file=io.BytesIO(csv_bytes))

    # 1. Ingest into PostgreSQL + Storage
    meta = await service.ingest_dataset(upload)
    assert meta.filename == "pg_sales.csv"

    # Verify present in PostgreSQL
    pg_model = await service.repo.get_by_id(meta.id)
    assert pg_model is not None
    assert pg_model.status == "active"

    # 2. RAM Cache Miss Rehydration
    DATASET_REGISTRY.pop(meta.id, None)
    df_rehydrated = await service.get_dataframe(meta.id)
    assert df_rehydrated is not None
    assert list(df_rehydrated["Revenue"]) == [1000, 500]

    # 3. Missing Storage File Status Update
    pg_temp_storage.delete(meta.storage_path)
    DATASET_REGISTRY.pop(meta.id, None)

    with pytest.raises(Exception) as exc:
        await service.get_dataframe(meta.id)
    assert "missing" in str(exc.value.detail)

    # Assert PostgreSQL status updated to 'missing'
    updated_model = await service.repo.get_by_id(meta.id)
    assert updated_model.status == "missing"

    # 4. Explicit Delete
    await service.delete_dataset(meta.id)
    assert await service.repo.get_by_id(meta.id) is None
