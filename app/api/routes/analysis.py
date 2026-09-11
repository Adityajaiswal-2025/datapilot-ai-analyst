import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

from app.schemas.analysis import (
    JoinRequest,
    JoinResponse,
    CompareRequest,
    CompareResponse,
    JoinExecutionStats,
    JoinKeyCandidate,
)
from app.tools.multi_dataset import join_datasets, compare_datasets, auto_detect_join_keys, MultiDatasetError

logger = logging.getLogger("datapilot.api.routes.analysis")

router = APIRouter()


@router.post(
    "/join",
    response_model=JoinResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-dataset join operation",
    description="Joins 2 or more registered datasets with candidate key detection, cardinality validation, and row explosion auditing.",
)
def execute_join_endpoint(req: JoinRequest) -> JoinResponse:
    """Executes multi-dataset join endpoint."""
    if not req.dataset_ids or len(req.dataset_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 dataset IDs must be provided to execute a join operation.",
        )

    # Evaluate join candidate keys
    detected_candidates: List[JoinKeyCandidate] = []
    try:
        det_res = auto_detect_join_keys(dataset_ids=req.dataset_ids)
        if "candidates" in det_res:
            detected_candidates = [JoinKeyCandidate(**c) for c in det_res["candidates"]]
    except Exception:
        pass

    try:
        join_res = join_datasets(
            dataset_ids=req.dataset_ids,
            join_keys=req.join_keys,
            join_type=req.join_type,
            new_filename=req.new_filename,
            register_result=req.register_result,
        )
        stats = JoinExecutionStats(**join_res["stats"])
        return JoinResponse(
            stats=stats,
            merged_dataset_id=join_res.get("merged_dataset_id"),
            detected_keys=detected_candidates,
            warnings=join_res.get("warnings", []),
        )
    except MultiDatasetError as e:
        logger.warning(f"Multi-dataset join failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Join operation failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during dataset join: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset join internal error: {str(e)}",
        )


@router.post(
    "/compare",
    response_model=CompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute side-by-side multi-dataset comparison",
    description="Compares metadata schemas, row counts, common columns, data types, missingness, duplicates, and statistical profiles across datasets.",
)
def execute_compare_endpoint(req: CompareRequest) -> CompareResponse:
    """Executes multi-dataset comparison endpoint."""
    if not req.dataset_ids or len(req.dataset_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 dataset IDs must be provided to perform dataset comparison.",
        )

    try:
        comp_res = compare_datasets(dataset_ids=req.dataset_ids)
        return CompareResponse(
            dataset_ids=comp_res["dataset_ids"],
            comparison_details=comp_res["comparison_details"],
            warnings=comp_res.get("warnings", []),
        )
    except MultiDatasetError as e:
        logger.warning(f"Multi-dataset comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Comparison failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during dataset comparison: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset comparison internal error: {str(e)}",
        )
