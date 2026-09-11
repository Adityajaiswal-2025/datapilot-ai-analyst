from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from app.schemas.response import BaseResponse


class ColumnSummary(BaseModel):
    """Metadata summary for an individual dataset column."""
    name: str
    dtype: str
    null_count: int = 0
    unique_count: int = 0


class DatasetMetadata(BaseModel):
    """Complete metadata contract for an ingested dataset."""
    id: str
    filename: str
    storage_path: str
    file_size_bytes: int
    file_type: str
    row_count: int
    column_count: int
    columns: List[ColumnSummary]
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DatasetUploadResponse(BaseResponse):
    """Response model returned after successful dataset upload."""
    dataset: DatasetMetadata


class DatasetSummary(BaseModel):
    """Condensed dataset model for listing endpoint."""
    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    row_count: int
    column_count: int
    created_at: datetime


class DatasetListResponse(BaseResponse):
    """Response model for listing datasets."""
    datasets: List[DatasetSummary]
    total: int


# --- Profiling Schemas (Phase 3) ---

class NumericStats(BaseModel):
    """Statistical summary for numeric columns."""
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None


class CategoricalStats(BaseModel):
    """Statistical summary for categorical/text columns."""
    mode: Optional[str] = None
    top_frequent_values: List[Dict[str, Any]] = Field(default_factory=list)


class DatetimeStats(BaseModel):
    """Statistical summary for datetime columns."""
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_range_days: Optional[float] = None


class ColumnProfile(BaseModel):
    """Full profiling breakdown for a single column."""
    name: str
    raw_dtype: str
    classified_type: str  # numeric, categorical, datetime, boolean, text
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    is_constant: bool = False
    numeric_stats: Optional[NumericStats] = None
    categorical_stats: Optional[CategoricalStats] = None
    datetime_stats: Optional[DatetimeStats] = None


class DatasetQualityReport(BaseModel):
    """Data quality and anomaly audit report for a dataset."""
    total_cells: int
    missing_cells: int
    completeness_percentage: float
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    quality_score: float  # 0.0 to 100.0 score based on completeness & duplicates
    warnings: List[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    """Complete dataset profile report."""
    dataset_id: str
    filename: str
    row_count: int
    column_count: int
    quality_report: DatasetQualityReport
    column_profiles: List[ColumnProfile]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DatasetProfileResponse(BaseResponse):
    """API response model for dataset profile endpoint."""
    profile: DatasetProfile


class DatasetContextResponse(BaseResponse):
    """API response model for dataset LLM context endpoint."""
    dataset_id: str
    filename: str
    context_text: str

