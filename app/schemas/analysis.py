"""Analysis request and response schemas for single and multi-dataset analytical operations."""

from typing import Optional, Any, List, Dict, Literal
from pydantic import BaseModel, Field
from app.schemas.response import BaseResponse


class AnalysisRequest(BaseModel):
    dataset_id: str
    query: str


class AnalysisResponse(BaseModel):
    analysis_id: str
    dataset_id: str
    query: str
    result: Optional[Any] = None
    insights: List[str] = []
    visualization_url: Optional[str] = None


# --- Multi-Dataset Join & Comparison Schemas (Phase 16) ---

class JoinKeyCandidate(BaseModel):
    """Evidence and metrics for an automatically detected join key candidate."""
    left_column: str = Field(description="Column name in left dataset")
    right_column: str = Field(description="Column name in right dataset")
    left_dtype: str = Field(description="Data type of left column")
    right_dtype: str = Field(description="Data type of right column")
    cardinality_left: str = Field(description="Uniqueness description for left column (e.g. 100% unique)")
    cardinality_right: str = Field(description="Uniqueness description for right column")
    value_overlap_percentage: float = Field(description="Percentage of overlapping unique values (0.0 to 100.0)")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
    recommendation_reason: str = Field(description="Explanation for join key candidate selection")
    is_ambiguous: bool = Field(default=False, description="Flag indicating if multiple candidate keys compete")


class JoinExecutionStats(BaseModel):
    """Execution statistics and cardinality audit produced by a join operation."""
    input_dataset_ids: List[str] = Field(description="IDs of input datasets joined")
    input_row_counts: Dict[str, int] = Field(description="Map of dataset ID or alias to original row count")
    join_keys_used: List[Dict[str, str]] = Field(description="List of join key mappings used (e.g. [{'left': 'customer_id', 'right': 'customer_id'}])")
    join_type: Literal["inner", "left", "right", "outer"] = Field(description="Type of join executed")
    output_row_count: int = Field(description="Final row count of merged DataFrame")
    matched_rows_count: int = Field(description="Number of matching key rows merged")
    unmatched_left_count: int = Field(description="Count of unmatched left rows")
    unmatched_right_count: int = Field(description="Count of unmatched right rows")
    columns_added: List[str] = Field(description="Names of columns added from right dataset")
    cardinality: Literal["one-to-one", "one-to-many", "many-to-one", "many-to-many", "unknown"] = Field(
        description="Relationship cardinality between join keys"
    )
    row_explosion_warning: bool = Field(default=False, description="Flag set to True if output row count expanded significantly")
    execution_status: Literal["success", "partial_success", "failed"] = Field(default="success")


class JoinRequest(BaseModel):
    """Request schema for POST /api/v1/analyze/join."""
    dataset_ids: List[str] = Field(description="List of 2 or more dataset IDs to join", min_length=2)
    join_keys: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Explicit join key mappings. Example: [{'left_column': 'customer_id', 'right_column': 'customer_id'}]"
    )
    join_type: Literal["inner", "left", "right", "outer"] = Field(default="inner", description="Type of SQL join to perform")
    auto_detect_keys: bool = Field(default=True, description="Whether to auto-detect join keys if join_keys is omitted")
    register_result: bool = Field(default=True, description="Whether to register the joined DataFrame as a new dataset in memory/disk")
    new_filename: Optional[str] = Field(default=None, description="Optional filename for newly registered merged dataset")


class JoinResponse(BaseResponse):
    """Response schema for POST /api/v1/analyze/join."""
    stats: JoinExecutionStats = Field(description="Execution statistics and cardinality analysis")
    merged_dataset_id: Optional[str] = Field(default=None, description="ID of newly registered merged dataset if registered")
    detected_keys: List[JoinKeyCandidate] = Field(default_factory=list, description="Evaluated candidate join keys")
    warnings: List[str] = Field(default_factory=list, description="Quality, cardinality, or row explosion warnings")


class CompareRequest(BaseModel):
    """Request schema for POST /api/v1/analyze/compare."""
    dataset_ids: List[str] = Field(description="List of 2 or more dataset IDs to compare", min_length=2)


class CompareResponse(BaseResponse):
    """Response schema for POST /api/v1/analyze/compare."""
    dataset_ids: List[str] = Field(description="List of dataset IDs compared")
    comparison_details: Dict[str, Any] = Field(description="Structured side-by-side comparative metadata and statistics")
    warnings: List[str] = Field(default_factory=list, description="Schema mismatch or data quality warnings")
