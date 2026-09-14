import os
import uuid
import logging
from typing import Optional, List, Dict, Any
import pandas as pd
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.data.validator import (
    validate_filename_safety,
    validate_file_extension,
    DatasetValidationError,
    DatasetSizeLimitError,
)
from app.data.loader import load_dataset, DatasetLoadError
from app.data.metadata import (
    extract_metadata,
    register_dataset,
    get_registered_dataset,
    delete_registered_dataset,
    DATASET_REGISTRY,
)
from app.data.storage import StorageService, default_storage_service
from app.database.models.dataset import DatasetModel
from app.database.repository import DatasetRepository, to_dataset_metadata
from app.schemas.dataset import DatasetMetadata, DatasetSummary

logger = logging.getLogger("datapilot.services.dataset_service")


class DatasetService:
    """Unified orchestration service for persistent dataset lifecycle management.

    Coordinates PostgreSQL metadata persistence, StorageService file I/O,
    and RAM DataFrame caching (DATASET_REGISTRY).
    """

    def __init__(
        self,
        db_session: AsyncSession,
        storage: Optional[StorageService] = None,
    ):
        self.repo = DatasetRepository(db_session)
        self.storage = storage or default_storage_service

    async def ingest_dataset(self, file: UploadFile) -> DatasetMetadata:
        """Handles bounded streaming upload, file storage, metadata extraction, DB persistence, and RAM caching."""
        raw_filename = file.filename or "uploaded_dataset"

        # 1. Validate filename safety and extension
        try:
            safe_filename = validate_filename_safety(raw_filename)
            ext = validate_file_extension(safe_filename)
        except DatasetValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )


        # 2. Prepare unique dataset ID and safe storage path
        dataset_id = str(uuid.uuid4())
        stored_filename = f"{dataset_id}{ext}"
        storage_path = self.storage.get_file_path(stored_filename)

        # 3. Stream file to storage provider with max size limit
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        try:
            bytes_written = await self.storage.save_stream(file, storage_path, max_bytes)
        except DatasetSizeLimitError as e:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=str(e),
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store dataset file: {str(e)}",
            )

        # 4. Load dataset and extract metadata schema
        try:
            df = load_dataset(storage_path)
            metadata = extract_metadata(
                dataset_id=dataset_id,
                original_filename=safe_filename,
                storage_path=storage_path,
                df=df,
            )

            # Validate row and column safety limits
            if len(df) > settings.MAX_DATASET_ROWS:
                raise DatasetValidationError(
                    f"Dataset row count ({len(df)}) exceeds maximum allowed limit of {settings.MAX_DATASET_ROWS} rows."
                )
            if len(df.columns) > settings.MAX_DATASET_COLUMNS:
                raise DatasetValidationError(
                    f"Dataset column count ({len(df.columns)}) exceeds maximum allowed limit of {settings.MAX_DATASET_COLUMNS} columns."
                )

            # 5. Persist metadata to PostgreSQL database
            await self.repo.create(metadata)

            # 6. Insert DataFrame into RAM Cache
            register_dataset(metadata, df)

            return metadata
        except DatasetValidationError as e:
            self.storage.delete(storage_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except DatasetLoadError as e:
            self.storage.delete(storage_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset processing error: {str(e)}",
            )
        except Exception as e:
            self.storage.delete(storage_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process dataset: {str(e)}",
            )

    async def get_dataframe(self, dataset_id: str) -> pd.DataFrame:
        """Retrieves loaded pandas.DataFrame. Returns RAM cache hit immediately or rehydrates from DB/Storage."""
        # 1. RAM Cache Hit Check (Fast path: no DB query on RAM hit)
        entry = get_registered_dataset(dataset_id)
        if entry and entry.get("dataframe") is not None:
            return entry["dataframe"]

        # 2. RAM Cache Miss: Query PostgreSQL metadata
        model = await self.repo.get_by_id(dataset_id)
        if not model or model.status != "active":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset with ID '{dataset_id}' not found in database.",
            )

        # 3. Verify storage file existence
        if not self.storage.exists(model.storage_path):
            await self.repo.update_status(dataset_id, "missing")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset source file is missing from storage for ID '{dataset_id}'.",
            )

        # 4. Rehydrate DataFrame from storage
        try:
            df = load_dataset(model.storage_path)
            metadata = to_dataset_metadata(model)

            # Validate safety limits on rehydration
            if len(df) > settings.MAX_DATASET_ROWS:
                raise DatasetValidationError(
                    f"Dataset row count ({len(df)}) exceeds limit of {settings.MAX_DATASET_ROWS}."
                )
            if len(df.columns) > settings.MAX_DATASET_COLUMNS:
                raise DatasetValidationError(
                    f"Dataset column count ({len(df.columns)}) exceeds limit of {settings.MAX_DATASET_COLUMNS}."
                )

            # Cache in RAM & update DB access timestamp
            register_dataset(metadata, df)
            await self.repo.update_last_accessed(dataset_id)
            return df
        except Exception as e:
            await self.repo.update_status(dataset_id, "corrupted")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to rehydrate dataset '{dataset_id}': {str(e)}",
            )

    async def get_metadata(self, dataset_id: str) -> DatasetMetadata:
        """Retrieves DatasetMetadata schema from PostgreSQL."""
        model = await self.repo.get_by_id(dataset_id)
        if model and model.status == "active":
            await self.repo.update_last_accessed(dataset_id)
            return to_dataset_metadata(model)

        # Fallback to RAM cache if DB model not available
        entry = get_registered_dataset(dataset_id)
        if entry:
            return entry["metadata"]

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found.",
        )

    async def list_datasets(self) -> List[DatasetSummary]:
        """Lists DatasetSummary objects for all active datasets in PostgreSQL."""
        models = await self.repo.list_all(status="active")
        summaries: List[DatasetSummary] = []
        for m in models:
            summaries.append(
                DatasetSummary(
                    id=m.id,
                    filename=m.original_filename,
                    file_type=m.file_type,
                    file_size_bytes=m.file_size_bytes,
                    row_count=m.row_count,
                    column_count=m.column_count,
                    created_at=m.created_at,
                )
            )
        return summaries

    async def delete_dataset(self, dataset_id: str) -> bool:
        """Deletes a dataset: removes DB record, evicts RAM cache, and deletes storage file."""
        model = await self.repo.get_by_id(dataset_id)
        storage_path = model.storage_path if model else None

        # 1. Delete DB metadata
        db_deleted = await self.repo.delete(dataset_id)

        # 2. Evict RAM cache entry
        delete_registered_dataset(dataset_id)

        # 3. Delete file from storage
        if storage_path:
            self.storage.delete(storage_path)

        return db_deleted or (storage_path is not None)

    async def cleanup_expired_datasets(self, ttl_seconds: Optional[int] = None) -> int:
        """Purges active datasets whose last_accessed_at exceeds TTL seconds (DB + Storage + RAM)."""
        ttl = ttl_seconds if ttl_seconds is not None else settings.DATASET_TTL_SECONDS
        if ttl <= 0:
            return 0

        expired = await self.repo.get_expired_datasets(ttl)
        count = 0
        for m in expired:
            await self.delete_dataset(m.id)
            count += 1
        return count


async def migrate_unindexed_files(
    db_session: AsyncSession, storage: Optional[StorageService] = None
) -> int:
    """Idempotent migration routine that scans local storage directory and indexes unindexed files into PostgreSQL."""
    storage_svc = storage or default_storage_service
    base_dir = storage_svc.get_file_path("")
    if not os.path.exists(base_dir):
        return 0

    repo = DatasetRepository(db_session)
    indexed_count = 0

    for fname in os.listdir(base_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            continue

        filePath = os.path.join(base_dir, fname)
        if not os.path.isfile(filePath):
            continue

        # Extract or infer dataset ID
        name_part = os.path.splitext(fname)[0]
        try:
            uuid.UUID(name_part)
            d_id = name_part
        except ValueError:
            d_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, fname))

        # Check if already indexed in DB
        existing = await repo.get_by_id(d_id)
        if existing:
            continue

        # Try loading and indexing valid file
        try:
            df = load_dataset(filePath)
            if len(df) > settings.MAX_DATASET_ROWS or len(df.columns) > settings.MAX_DATASET_COLUMNS:
                continue

            metadata = extract_metadata(
                dataset_id=d_id,
                original_filename=fname,
                storage_path=filePath,
                df=df,
            )
            await repo.create(metadata)
            register_dataset(metadata, df)
            indexed_count += 1
        except Exception as e:
            logger.warning(f"Skipping indexing of file '{fname}': {e}")
            continue

    return indexed_count
