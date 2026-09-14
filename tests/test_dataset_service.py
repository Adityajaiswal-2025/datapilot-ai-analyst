import io
import os
import time
import pytest
import pandas as pd
from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.core.config import settings
from app.database.models.dataset import Base, DatasetModel
from app.database.repository import DatasetRepository
from app.data.storage import LocalStorageService
from app.data.metadata import DATASET_REGISTRY, extract_metadata
from app.services.dataset_service import DatasetService, migrate_unindexed_files

import pytest_asyncio

# Use async SQLite for isolated unit testing
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def test_db():

    """Provides an isolated in-memory SQLite AsyncSession for testing."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    AsyncSessionTest = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with AsyncSessionTest() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def temp_storage(tmp_path):
    """Provides a temporary LocalStorageService instance."""
    storage_dir = str(tmp_path / "uploads")
    return LocalStorageService(base_dir=storage_dir)


@pytest.mark.asyncio
async def test_storage_service_operations(temp_storage):
    """Test StorageService save, exists, get_file_path, and idempotent delete."""
    file_bytes = b"ColA,ColB\n1,10\n2,20\n"
    upload = UploadFile(filename="test.csv", file=io.BytesIO(file_bytes))

    storage_key = "test_file.csv"
    bytes_written = await temp_storage.save_stream(upload, storage_key, max_bytes=1024 * 1024)
    assert bytes_written == len(file_bytes)
    assert temp_storage.exists(storage_key) is True

    path = temp_storage.get_file_path(storage_key)
    assert os.path.exists(path)

    # Idempotent delete
    assert temp_storage.delete(storage_key) is True
    assert temp_storage.exists(storage_key) is False
    assert temp_storage.delete(storage_key) is False


@pytest.mark.asyncio
async def test_dataset_repository_crud(test_db, temp_storage):
    """Test DatasetRepository CRUD methods."""
    repo = DatasetRepository(test_db)
    df = pd.DataFrame({"A": [1, 2], "B": ["X", "Y"]})
    file_path = temp_storage.get_file_path("repo_test.csv")
    df.to_csv(file_path, index=False)

    meta = extract_metadata("repo_ds_1", "repo_test.csv", file_path, df)
    model = await repo.create(meta)

    assert model.id == "repo_ds_1"
    assert model.original_filename == "repo_test.csv"

    fetched = await repo.get_by_id("repo_ds_1")
    assert fetched is not None
    assert fetched.row_count == 2

    listed = await repo.list_all(status="active")
    assert len(listed) >= 1

    await repo.update_last_accessed("repo_ds_1")
    deleted = await repo.delete("repo_ds_1")
    assert deleted is True
    assert await repo.get_by_id("repo_ds_1") is None

    temp_storage.delete("repo_test.csv")


@pytest.mark.asyncio
async def test_dataset_service_ingest_cache_hit_and_rehydrate(test_db, temp_storage):
    """Test DatasetService ingestion, RAM cache hits, and rehydration on cache miss."""
    service = DatasetService(test_db, storage=temp_storage)

    csv_bytes = b"Region,Sales\nNorth,100\nSouth,200\n"
    upload = UploadFile(filename="sales.csv", file=io.BytesIO(csv_bytes))

    meta = await service.ingest_dataset(upload)
    assert meta.filename == "sales.csv"
    assert meta.row_count == 2

    # 1. RAM Cache Hit
    df_hit = await service.get_dataframe(meta.id)
    assert df_hit is not None
    assert list(df_hit["Region"]) == ["North", "South"]

    # 2. Simulate Process Restart / RAM Eviction (clear RAM cache)
    DATASET_REGISTRY.pop(meta.id, None)

    # 3. RAM Cache Miss Rehydration
    df_rehydrated = await service.get_dataframe(meta.id)
    assert df_rehydrated is not None
    assert list(df_rehydrated["Sales"]) == [100, 200]

    # Cleanup
    await service.delete_dataset(meta.id)


@pytest.mark.asyncio
async def test_dataset_service_missing_storage_file_handling(test_db, temp_storage):
    """Test DatasetService returns 404 when database metadata exists but storage file is missing."""
    service = DatasetService(test_db, storage=temp_storage)

    csv_bytes = b"X,Y\n1,2\n"
    upload = UploadFile(filename="missing_test.csv", file=io.BytesIO(csv_bytes))
    meta = await service.ingest_dataset(upload)

    # Manually remove storage file from disk
    temp_storage.delete(meta.storage_path)
    DATASET_REGISTRY.pop(meta.id, None)

    # Attempt rehydration should raise 404 and set status='missing' in DB
    with pytest.raises(HTTPException) as exc:
        await service.get_dataframe(meta.id)
    assert exc.value.status_code == 404
    assert "missing" in exc.value.detail

    # Verify status in DB was updated to 'missing'
    model = await service.repo.get_by_id(meta.id)
    assert model.status == "missing"

    await service.delete_dataset(meta.id)


@pytest.mark.asyncio
async def test_migration_of_unindexed_files(test_db, temp_storage):
    """Test idempotent migration routine for indexing unindexed local files into PostgreSQL."""
    df = pd.DataFrame({"Col1": [1, 2, 3], "Col2": ["A", "B", "C"]})
    filePath = temp_storage.get_file_path("unindexed_dataset.csv")
    df.to_csv(filePath, index=False)

    # Run migration
    indexed_count = await migrate_unindexed_files(test_db, storage=temp_storage)
    assert indexed_count >= 1

    # Verify indexed in DB
    service = DatasetService(test_db, storage=temp_storage)
    summaries = await service.list_datasets()
    assert any(s.filename == "unindexed_dataset.csv" for s in summaries)

    # Second migration run should be idempotent and index 0 additional files
    second_run = await migrate_unindexed_files(test_db, storage=temp_storage)
    assert second_run == 0

    temp_storage.delete(filePath)


@pytest.mark.asyncio
async def test_lru_ram_eviction_preserves_source_file_and_db_metadata(test_db, temp_storage):
    """Verify that LRU RAM eviction frees memory without deleting stored source file or altering DB metadata."""
    from app.data.metadata import evict_lru_datasets
    service = DatasetService(test_db, storage=temp_storage)

    csv_bytes = b"Val1,Val2\n10,20\n"
    upload = UploadFile(filename="lru_test.csv", file=io.BytesIO(csv_bytes))
    meta = await service.ingest_dataset(upload)

    # Force LRU eviction of RAM DataFrames
    evicted_count = evict_lru_datasets(max_allowed=0)
    assert evicted_count >= 1

    # Verify RAM DataFrame evicted
    assert DATASET_REGISTRY[meta.id]["dataframe"] is None

    # Assert source file STILL EXISTS on disk
    assert temp_storage.exists(meta.storage_path) is True

    # Assert PostgreSQL metadata record STILL EXISTS and active
    model = await service.repo.get_by_id(meta.id)
    assert model is not None
    assert model.status == "active"

    # Cleanup
    await service.delete_dataset(meta.id)


@pytest.mark.asyncio
async def test_dataset_service_ttl_cleanup_removes_db_ram_and_storage(test_db, temp_storage):
    """Verify that DatasetService.cleanup_expired_datasets coordinates DB + RAM + storage deletion consistently."""
    from datetime import datetime, timezone, timedelta
    service = DatasetService(test_db, storage=temp_storage)

    csv_bytes = b"ColA\n100\n"
    upload = UploadFile(filename="ttl_test.csv", file=io.BytesIO(csv_bytes))
    meta = await service.ingest_dataset(upload)

    # Manually age last_accessed_at in DB to 100 seconds ago
    old_time = datetime.now(timezone.utc) - timedelta(seconds=100)
    meta_model = await service.repo.get_by_id(meta.id)
    meta_model.last_accessed_at = old_time
    await test_db.commit()

    # Also update RAM registry timestamp
    DATASET_REGISTRY[meta.id]["last_accessed"] = old_time

    # Run TTL cleanup with 10s cutoff
    purged_count = await service.cleanup_expired_datasets(ttl_seconds=10)
    assert purged_count == 1

    # Assert DB record deleted
    assert await service.repo.get_by_id(meta.id) is None

    # Assert RAM entry removed
    assert meta.id not in DATASET_REGISTRY

    # Assert storage file deleted from disk
    assert temp_storage.exists(meta.storage_path) is False

    # Assert idempotency: running cleanup again purges 0 additional items
    second_purge = await service.cleanup_expired_datasets(ttl_seconds=10)
    assert second_purge == 0


@pytest.mark.asyncio
async def test_explicit_delete_removes_db_ram_and_storage(test_db, temp_storage):
    """Verify that explicit DELETE removes DB record, RAM entry, and storage file."""
    service = DatasetService(test_db, storage=temp_storage)

    csv_bytes = b"X,Y\n1,2\n"
    upload = UploadFile(filename="explicit_delete.csv", file=io.BytesIO(csv_bytes))
    meta = await service.ingest_dataset(upload)

    assert temp_storage.exists(meta.storage_path) is True
    assert meta.id in DATASET_REGISTRY

    deleted = await service.delete_dataset(meta.id)
    assert deleted is True

    assert await service.repo.get_by_id(meta.id) is None
    assert meta.id not in DATASET_REGISTRY
    assert temp_storage.exists(meta.storage_path) is False


@pytest.mark.asyncio
async def test_multi_worker_empty_ram_cache_rehydration(test_db, temp_storage):
    """Verify DatasetService rehydrates dataset when DATASET_REGISTRY starts empty (multi-worker simulation)."""
    # Worker A ingests dataset
    worker_a_service = DatasetService(test_db, storage=temp_storage)
    csv_bytes = b"Category,Revenue\nTech,500\nRetail,300\n"
    upload = UploadFile(filename="worker_test.csv", file=io.BytesIO(csv_bytes))
    meta = await worker_a_service.ingest_dataset(upload)

    # Worker B starts with empty RAM registry
    DATASET_REGISTRY.clear()
    assert meta.id not in DATASET_REGISTRY

    # Worker B queries dataset via DatasetService
    worker_b_service = DatasetService(test_db, storage=temp_storage)
    df_worker_b = await worker_b_service.get_dataframe(meta.id)

    assert df_worker_b is not None
    assert list(df_worker_b["Category"]) == ["Tech", "Retail"]

    # Verify rehydrated into Worker B's RAM cache
    assert meta.id in DATASET_REGISTRY
    assert DATASET_REGISTRY[meta.id]["dataframe"] is not None

    await worker_b_service.delete_dataset(meta.id)

