import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.core.config import settings
from app.data.validator import validate_dataset_file, DatasetValidationError
from app.data.loader import load_dataset, DatasetLoadError
from app.data.metadata import extract_metadata, register_dataset
from app.schemas.dataset import DatasetUploadResponse

router = APIRouter()


@router.post(
    "",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Dataset File",
    description="Uploads a CSV or XLSX dataset file, validates file integrity, extracts schema metadata, and registers dataset.",
)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    """Handles multipart file upload for CSV and XLSX files."""
    filename = file.filename or "uploaded_dataset"

    # Step 1: Read file content bytes to determine file size
    file_bytes = await file.read()
    file_size = len(file_bytes)

    # Step 2: Validate extension and size
    try:
        ext = validate_dataset_file(filename, file_size)
    except DatasetValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Step 3: Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Step 4: Generate unique dataset ID and safe storage path
    dataset_id = str(uuid.uuid4())
    stored_filename = f"{dataset_id}{ext}"
    storage_path = os.path.join(settings.UPLOAD_DIR, stored_filename)

    # Step 5: Save file to disk securely
    try:
        with open(storage_path, "wb") as buffer:
            buffer.write(file_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}",
        )

    # Step 6: Load dataset into Pandas DataFrame and extract metadata
    try:
        df = load_dataset(storage_path)
        metadata = extract_metadata(
            dataset_id=dataset_id,
            original_filename=filename,
            storage_path=storage_path,
            df=df,
        )
        register_dataset(metadata, df)
    except (DatasetValidationError, DatasetLoadError) as e:
        # Cleanup file if loading/parsing fails
        if os.path.exists(storage_path):
            os.remove(storage_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset processing error: {str(e)}",
        )

    return DatasetUploadResponse(
        success=True,
        message=f"Dataset '{filename}' successfully uploaded and processed.",
        dataset=metadata,
    )
