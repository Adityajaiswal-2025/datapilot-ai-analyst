import pandas as pd
from typing import Optional, List, Dict, Any
from app.tools.analysis import _resolve_dataframe, AnalysisToolError
from app.data.metadata import sanitize_for_json


def calculate_statistics(
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculates detailed descriptive statistics for specified numeric columns."""
    target_df = _resolve_dataframe(dataset_id, df)

    if not columns:
        # Default to all numeric columns
        columns = list(target_df.select_dtypes(include=["number"]).columns)

    if not columns:
        raise AnalysisToolError("No numeric columns found in dataset for statistical calculation.")

    stats_output: Dict[str, Dict[str, Any]] = {}
    for col in columns:
        if col not in target_df.columns:
            raise AnalysisToolError(f"Column '{col}' not found in dataset.")

        series = target_df[col].dropna().astype(float)
        if series.empty:
            stats_output[col] = {"error": "Column contains no valid numeric values."}
            continue

        stats_output[col] = {
            "count": int(series.count()),
            "mean": sanitize_for_json(series.mean()),
            "std": sanitize_for_json(series.std()) if len(series) > 1 else 0.0,
            "variance": sanitize_for_json(series.var()) if len(series) > 1 else 0.0,
            "min": sanitize_for_json(series.min()),
            "q25": sanitize_for_json(series.quantile(0.25)),
            "median": sanitize_for_json(series.median()),
            "q75": sanitize_for_json(series.quantile(0.75)),
            "max": sanitize_for_json(series.max()),
            "skewness": sanitize_for_json(series.skew()) if len(series) > 2 else 0.0,
            "kurtosis": sanitize_for_json(series.kurtosis()) if len(series) > 3 else 0.0,
        }

    return {
        "columns_analyzed": columns,
        "statistics": stats_output,
    }


def calculate_correlation(
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
) -> Dict[str, Any]:
    """Calculates pairwise correlation matrix across numeric columns."""
    target_df = _resolve_dataframe(dataset_id, df)

    if not columns:
        columns = list(target_df.select_dtypes(include=["number"]).columns)

    if len(columns) < 2:
        raise AnalysisToolError(
            f"Correlation requires at least 2 numeric columns. Found: {len(columns)}."
        )

    for col in columns:
        if col not in target_df.columns:
            raise AnalysisToolError(f"Column '{col}' not found in dataset.")

    sub_df = target_df[columns].apply(pd.to_numeric, errors="coerce")
    corr_df = sub_df.corr(method=method)

    corr_matrix: Dict[str, Dict[str, Optional[float]]] = {}
    for col1 in columns:
        corr_matrix[col1] = {}
        for col2 in columns:
            val = corr_df.loc[col1, col2]
            corr_matrix[col1][col2] = sanitize_for_json(val)

    return {
        "method": method,
        "columns": columns,
        "correlation_matrix": corr_matrix,
    }
