import math
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from app.data.metadata import get_registered_dataset
from app.schemas.dataset import (
    ColumnProfile,
    NumericStats,
    CategoricalStats,
    DatetimeStats,
    DatasetQualityReport,
    DatasetProfile,
)


class ProfilingError(Exception):
    """Custom exception raised when data profiling fails."""
    pass


def classify_column(series: pd.Series) -> str:
    """Classifies a Pandas Series into: 'boolean', 'numeric', 'datetime', 'categorical', or 'text'."""
    clean_series = series.dropna()
    if clean_series.empty:
        return "text"

    # Check for boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if clean_series.dtype == "object":
        unique_vals = set(clean_series.astype(str).str.lower().unique())
        if unique_vals.issubset({"true", "false", "1", "0", "yes", "no", "t", "f"}):
            return "boolean"

    # Check for numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if series.dtype == "object" and len(clean_series) > 0:
        sample = clean_series.head(10).astype(str)
        try:
            parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
            if parsed.notnull().sum() / len(sample) >= 0.8:
                return "datetime"
        except Exception:
            pass

    # Categorical vs Text distinction for string/object dtypes
    unique_count = clean_series.nunique()
    total_count = len(clean_series)
    unique_ratio = unique_count / total_count if total_count > 0 else 1.0

    if unique_count <= 50 or unique_ratio < 0.25:
        return "categorical"
    return "text"


def profile_numeric_column(series: pd.Series) -> NumericStats:
    """Computes statistical metrics for a numeric column."""
    clean = series.dropna().astype(float)
    if clean.empty:
        return NumericStats()

    def safe_float(val: Any) -> Optional[float]:
        if val is None or pd.isna(val) or math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)

    return NumericStats(
        min=safe_float(clean.min()),
        max=safe_float(clean.max()),
        mean=safe_float(clean.mean()),
        std=safe_float(clean.std()) if len(clean) > 1 else 0.0,
        median=safe_float(clean.median()),
        q25=safe_float(clean.quantile(0.25)),
        q75=safe_float(clean.quantile(0.75)),
        skewness=safe_float(clean.skew()) if len(clean) > 2 else 0.0,
    )


def profile_categorical_column(series: pd.Series) -> CategoricalStats:
    """Computes value frequencies and mode for categorical/text columns."""
    clean = series.dropna().astype(str)
    if clean.empty:
        return CategoricalStats()

    value_counts = clean.value_counts().head(5)
    top_frequent = [
        {"value": str(val), "count": int(count)}
        for val, count in value_counts.items()
    ]
    mode_val = str(clean.mode()[0]) if not clean.mode().empty else None

    return CategoricalStats(
        mode=mode_val,
        top_frequent_values=top_frequent,
    )


def profile_datetime_column(series: pd.Series) -> DatetimeStats:
    """Computes min date, max date, and range for datetime columns."""
    parsed = pd.to_datetime(series, format="mixed", errors="coerce").dropna()
    if parsed.empty:
        return DatetimeStats()

    min_d = parsed.min()
    max_d = parsed.max()
    diff_days = (max_d - min_d).total_seconds() / (24 * 3600)

    return DatetimeStats(
        min_date=min_d.isoformat(),
        max_date=max_d.isoformat(),
        date_range_days=round(diff_days, 2),
    )


def profile_column(series: pd.Series) -> ColumnProfile:
    """Generates a complete ColumnProfile for a Pandas Series."""
    col_name = str(series.name)
    total_len = len(series)
    null_count = int(series.isnull().sum())
    null_pct = round((null_count / total_len * 100.0) if total_len > 0 else 0.0, 2)

    unique_count = int(series.nunique(dropna=True))
    unique_pct = round((unique_count / total_len * 100.0) if total_len > 0 else 0.0, 2)
    is_constant = (unique_count <= 1) and (null_count < total_len)

    classified_type = classify_column(series)

    num_stats = None
    cat_stats = None
    dt_stats = None

    if classified_type == "numeric":
        num_stats = profile_numeric_column(series)
    elif classified_type in ["categorical", "text", "boolean"]:
        cat_stats = profile_categorical_column(series)
    elif classified_type == "datetime":
        dt_stats = profile_datetime_column(series)

    return ColumnProfile(
        name=col_name,
        raw_dtype=str(series.dtype),
        classified_type=classified_type,
        null_count=null_count,
        null_percentage=null_pct,
        unique_count=unique_count,
        unique_percentage=unique_pct,
        is_constant=is_constant,
        numeric_stats=num_stats,
        categorical_stats=cat_stats,
        datetime_stats=dt_stats,
    )


def audit_data_quality(df: pd.DataFrame) -> DatasetQualityReport:
    """Audits dataset quality, missing values, duplicates, and anomaly flags."""
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    missing_cells = int(df.isnull().sum().sum())
    completeness_pct = (
        round(((total_cells - missing_cells) / total_cells * 100.0), 2)
        if total_cells > 0
        else 100.0
    )

    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = (
        round((duplicate_rows / total_rows * 100.0), 2) if total_rows > 0 else 0.0
    )

    # Compute composite quality score (0.0 to 100.0)
    score = (completeness_pct * 0.70) + ((100.0 - duplicate_pct) * 0.30)
    quality_score = round(max(0.0, min(100.0, score)), 2)

    warnings: List[str] = []

    if duplicate_rows > 0:
        warnings.append(
            f"Dataset contains {duplicate_rows} duplicate row(s) ({duplicate_pct}%)."
        )

    if completeness_pct < 80.0:
        warnings.append(
            f"Dataset has significant missing values. Overall completeness is only {completeness_pct}%."
        )

    for col in df.columns:
        series = df[col]
        null_count = series.isnull().sum()
        null_pct = (null_count / total_rows * 100.0) if total_rows > 0 else 0.0
        if null_pct > 50.0:
            warnings.append(
                f"Column '{col}' has high missingness ({null_pct:.1f}% missing values)."
            )

        unique_cnt = series.nunique(dropna=True)
        if unique_cnt == 1:
            warnings.append(
                f"Column '{col}' is constant (contains only 1 unique value)."
            )

    return DatasetQualityReport(
        total_cells=total_cells,
        missing_cells=missing_cells,
        completeness_percentage=completeness_pct,
        duplicate_rows_count=duplicate_rows,
        duplicate_rows_percentage=duplicate_pct,
        quality_score=quality_score,
        warnings=warnings,
    )


def profile_dataset(
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> DatasetProfile:
    """Core profiling tool function. Profiling target can be passed via dataset_id or raw DataFrame."""
    filename = "dataframe"
    target_id = dataset_id or "direct_dataframe"

    if dataset_id:
        entry = get_registered_dataset(dataset_id)
        if not entry:
            raise ProfilingError(f"Dataset with ID '{dataset_id}' not found in registry.")
        df = entry["dataframe"]
        filename = entry["metadata"].filename

    if df is None:
        raise ProfilingError("Either dataset_id or df must be provided to profile_dataset().")

    quality_report = audit_data_quality(df)
    column_profiles = [profile_column(df[col]) for col in df.columns]

    return DatasetProfile(
        dataset_id=target_id,
        filename=filename,
        row_count=len(df),
        column_count=len(df.columns),
        quality_report=quality_report,
        column_profiles=column_profiles,
    )


def get_column_info(
    dataset_id: Optional[str] = None,
    column_name: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
) -> ColumnProfile:
    """Targeted column profiling tool function."""
    if not column_name:
        raise ProfilingError("column_name is required for get_column_info().")

    if dataset_id:
        entry = get_registered_dataset(dataset_id)
        if not entry:
            raise ProfilingError(f"Dataset with ID '{dataset_id}' not found in registry.")
        df = entry["dataframe"]

    if df is None:
        raise ProfilingError("Either dataset_id or df must be provided to get_column_info().")

    if column_name not in df.columns:
        raise ProfilingError(
            f"Column '{column_name}' not found in dataset. Available columns: {list(df.columns)}"
        )

    return profile_column(df[column_name])
