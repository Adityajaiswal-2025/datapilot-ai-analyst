from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_db_session
from app.services.dataset_service import DatasetService
from app.schemas.dataset import DatasetUploadResponse

router = APIRouter()


@router.post(
    "",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Dataset File",
    description="Uploads a CSV or XLSX dataset file using bounded chunked streaming, validates file integrity, extracts schema metadata, persists to PostgreSQL, and caches in RAM.",
)
async def upload_dataset(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
) -> DatasetUploadResponse:
    """Handles multipart file upload for CSV and XLSX files via DatasetService."""
    service = DatasetService(db)
    metadata = await service.ingest_dataset(file)
    return DatasetUploadResponse(
        success=True,
        message=f"Dataset '{metadata.filename}' successfully uploaded and processed.",
        dataset=metadata,
    )


