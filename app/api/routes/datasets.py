from fastapi import APIRouter, HTTPException, status
from app.schemas.dataset import (
    DatasetListResponse,
    DatasetUploadResponse,
    DatasetProfileResponse,
    DatasetContextResponse,
    DatasetMetadata,
)
from app.schemas.response import BaseResponse
from app.data.metadata import (
    list_registered_datasets,
    get_registered_dataset,
    delete_registered_dataset,
)
from app.data.context import build_dataset_context, ContextGenerationError
from app.tools.profiling import profile_dataset, ProfilingError

router = APIRouter()


@router.get(
    "",
    response_model=DatasetListResponse,
    summary="List Ingested Datasets",
    description="Retrieves a list of all datasets uploaded to the system.",
)
async def list_datasets() -> DatasetListResponse:
    """Lists metadata summaries for all registered datasets."""
    datasets = list_registered_datasets()
    return DatasetListResponse(
        success=True,
        message=f"Retrieved {len(datasets)} dataset(s).",
        datasets=datasets,
        total=len(datasets),
    )


@router.get(
    "/{dataset_id}",
    response_model=DatasetUploadResponse,
    summary="Get Dataset Details",
    description="Retrieves full metadata summary, schema, and sample rows for a specific dataset ID.",
)
async def get_dataset(dataset_id: str) -> DatasetUploadResponse:
    """Retrieves full metadata details for a single dataset by ID."""
    entry = get_registered_dataset(dataset_id)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found.",
        )
    metadata: DatasetMetadata = entry["metadata"]
    return DatasetUploadResponse(
        success=True,
        message=f"Retrieved metadata for dataset ID '{dataset_id}'.",
        dataset=metadata,
    )


@router.get(
    "/{dataset_id}/profile",
    response_model=DatasetProfileResponse,
    summary="Profile Dataset",
    description="Generates an automated, detailed profiling report for a dataset including data quality audit, column classification, statistics, and warning flags.",
)
async def profile_dataset_endpoint(dataset_id: str) -> DatasetProfileResponse:
    """Generates comprehensive dataset profiling report."""
    try:
        profile = profile_dataset(dataset_id=dataset_id)
        return DatasetProfileResponse(
            success=True,
            message=f"Successfully generated profile for dataset ID '{dataset_id}'.",
            profile=profile,
        )
    except ProfilingError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.get(
    "/{dataset_id}/context",
    response_model=DatasetContextResponse,
    summary="Get Dataset LLM Context",
    description="Generates a token-optimized, high-density Markdown context block representation of the dataset for LLM prompt insertion.",
)
async def get_dataset_context_endpoint(dataset_id: str) -> DatasetContextResponse:
    """Generates formatted dataset context string for LLM agents."""
    entry = get_registered_dataset(dataset_id)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found.",
        )
    try:
        context_text = build_dataset_context(dataset_id=dataset_id)
        filename = entry["metadata"].filename
        return DatasetContextResponse(
            success=True,
            message=f"Successfully generated LLM context for dataset ID '{dataset_id}'.",
            dataset_id=dataset_id,
            filename=filename,
            context_text=context_text,
        )
    except ContextGenerationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete(
    "/{dataset_id}",
    response_model=BaseResponse,
    summary="Delete Ingested Dataset",
    description="Removes dataset metadata from registry and deletes stored file from disk.",
)
async def delete_dataset(dataset_id: str) -> BaseResponse:
    """Deletes a dataset by ID."""
    deleted = delete_registered_dataset(dataset_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID '{dataset_id}' not found.",
        )
    return BaseResponse(
        success=True,
        message=f"Dataset '{dataset_id}' successfully deleted.",
    )
