from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.models.dataset import DatasetModel
from app.schemas.dataset import DatasetMetadata, ColumnSummary


class DatasetRepository:
    """Async repository for PostgreSQL dataset metadata CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, metadata: DatasetMetadata) -> DatasetModel:
        """Persists a new DatasetMetadata model into PostgreSQL."""
        record = DatasetModel(
            id=metadata.id,
            original_filename=metadata.filename,
            storage_path=metadata.storage_path,
            file_size_bytes=metadata.file_size_bytes,
            file_type=metadata.file_type,
            row_count=metadata.row_count,
            column_count=metadata.column_count,
            columns=[col.model_dump() for col in metadata.columns],
            sample_rows=metadata.sample_rows,
            status="active",
            created_at=metadata.created_at or datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_accessed_at=datetime.now(timezone.utc),
        )
        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)
        return record

    async def get_by_id(self, dataset_id: str) -> Optional[DatasetModel]:
        """Retrieves a dataset metadata record by ID."""
        stmt = select(DatasetModel).where(DatasetModel.id == dataset_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(self, status: str = "active") -> List[DatasetModel]:
        """Lists dataset metadata records matching status ordered by creation time."""
        stmt = select(DatasetModel).where(DatasetModel.status == status).order_by(DatasetModel.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_last_accessed(self, dataset_id: str) -> None:
        """Updates the last_accessed_at timestamp for a dataset."""
        stmt = (
            update(DatasetModel)
            .where(DatasetModel.id == dataset_id)
            .values(last_accessed_at=datetime.now(timezone.utc))
        )
        await self.session.execute(stmt)
        await self.session.commit()

    async def update_status(self, dataset_id: str, status: str) -> None:
        """Updates the status flag for a dataset (e.g. 'active', 'missing', 'corrupted')."""
        stmt = (
            update(DatasetModel)
            .where(DatasetModel.id == dataset_id)
            .values(status=status, updated_at=datetime.now(timezone.utc))
        )
        await self.session.execute(stmt)
        await self.session.commit()

    async def delete(self, dataset_id: str) -> bool:
        """Deletes a dataset metadata record from PostgreSQL by ID."""
        stmt = delete(DatasetModel).where(DatasetModel.id == dataset_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0

    async def get_expired_datasets(self, ttl_seconds: int) -> List[DatasetModel]:
        """Retrieves active dataset records whose last_accessed_at exceeds TTL."""
        if ttl_seconds <= 0:
            return []
        cutoff = datetime.now(timezone.utc)
        stmt = select(DatasetModel).where(DatasetModel.status == "active")
        result = await self.session.execute(stmt)
        expired = []
        for row in result.scalars().all():
            elapsed = (cutoff - row.last_accessed_at.replace(tzinfo=timezone.utc)).total_seconds()
            if elapsed > ttl_seconds:
                expired.append(row)
        return expired


def to_dataset_metadata(model: DatasetModel) -> DatasetMetadata:
    """Helper converting SQLAlchemy DatasetModel instance to Pydantic DatasetMetadata schema."""
    columns = [
        ColumnSummary(**col) if isinstance(col, dict) else col for col in model.columns
    ]
    return DatasetMetadata(
        id=model.id,
        filename=model.original_filename,
        storage_path=model.storage_path,
        file_size_bytes=model.file_size_bytes,
        file_type=model.file_type,
        row_count=model.row_count,
        column_count=model.column_count,
        columns=columns,
        sample_rows=model.sample_rows,
        created_at=model.created_at,
    )
