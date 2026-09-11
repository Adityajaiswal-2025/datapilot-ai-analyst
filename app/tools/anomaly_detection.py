import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from app.tools.analysis import _resolve_dataframe, AnalysisToolError
from app.data.metadata import sanitize_for_json


def analyze_trend(
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    date_column: str = "",
    value_column: str = "",
    period: str = "ME",
) -> Dict[str, Any]:
    """Analyzes temporal trends, computes growth rates, and determines overall trend direction."""
    target_df = _resolve_dataframe(dataset_id, df)

    if not date_column or date_column not in target_df.columns:
        raise AnalysisToolError(f"Date column '{date_column}' not found in dataset.")
    if not value_column or value_column not in target_df.columns:
        raise AnalysisToolError(f"Value column '{value_column}' not found in dataset.")

    temp_df = target_df[[date_column, value_column]].copy()
    temp_df[date_column] = pd.to_datetime(temp_df[date_column], format="mixed", errors="coerce")
    temp_df[value_column] = pd.to_numeric(temp_df[value_column], errors="coerce")
    temp_df = temp_df.dropna()

    if temp_df.empty:
        raise AnalysisToolError("No valid date/numeric data available for trend analysis.")

    temp_df = temp_df.set_index(date_column)
    resampled = temp_df[value_column].resample(period).sum()

    periods_data = []
    prev_val = None
    growth_rates = []

    for date_idx, val in resampled.items():
        pct_change = None
        if prev_val is not None and prev_val != 0:
            pct_change = round(((val - prev_val) / abs(prev_val)) * 100.0, 2)
            growth_rates.append(pct_change)

        periods_data.append({
            "period": str(date_idx),
            "value": sanitize_for_json(val),
            "growth_percentage": pct_change,
        })
        prev_val = val

    # Determine overall trend direction
    if len(resampled) >= 2:
        first_val = resampled.iloc[0]
        last_val = resampled.iloc[-1]
        overall_change_pct = (
            round(((last_val - first_val) / abs(first_val)) * 100.0, 2)
            if first_val != 0
            else 0.0
        )
        if overall_change_pct > 5.0:
            direction = "increasing"
        elif overall_change_pct < -5.0:
            direction = "decreasing"
        else:
            direction = "stable"
    else:
        overall_change_pct = 0.0
        direction = "insufficient_data"

    avg_growth = (
        round(float(np.mean(growth_rates)), 2) if growth_rates else 0.0
    )

    return {
        "date_column": date_column,
        "value_column": value_column,
        "period": period,
        "overall_trend_direction": direction,
        "overall_change_percentage": overall_change_pct,
        "average_period_growth_percentage": avg_growth,
        "period_breakdown": periods_data,
    }


def detect_anomalies(
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    column: str = "",
    method: str = "zscore",
    threshold: float = 3.0,
) -> Dict[str, Any]:
    """Detects statistical outliers/anomalies in a numerical column."""
    target_df = _resolve_dataframe(dataset_id, df)

    if not column or column not in target_df.columns:
        raise AnalysisToolError(f"Target column '{column}' not found in dataset.")

    series = pd.to_numeric(target_df[column], errors="coerce").dropna()
    if series.empty:
        raise AnalysisToolError(f"Column '{column}' contains no numeric data for anomaly detection.")

    anomalies_idx = []

    if method.lower() == "zscore":
        mean = series.mean()
        std = series.std()
        if std > 0:
            z_scores = np.abs((series - mean) / std)
            anomalies_series = series[z_scores > threshold]
            anomalies_idx = list(anomalies_series.index)
    elif method.lower() == "iqr":
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - (threshold * iqr)
        upper_bound = q75 + (threshold * iqr)
        anomalies_series = series[(series < lower_bound) | (series > upper_bound)]
        anomalies_idx = list(anomalies_series.index)
    else:
        raise AnalysisToolError(
            f"Unsupported anomaly detection method '{method}'. Supported: 'zscore', 'iqr'."
        )

    anomaly_rows = []
    for idx in anomalies_idx:
        row_dict = target_df.loc[idx].to_dict()
        clean_row = {str(k): sanitize_for_json(v) for k, v in row_dict.items()}
        anomaly_rows.append(clean_row)

    total_count = len(series)
    anomaly_count = len(anomalies_idx)
    anomaly_pct = round((anomaly_count / total_count * 100.0) if total_count > 0 else 0.0, 2)

    return {
        "column": column,
        "method": method,
        "threshold": threshold,
        "total_rows_analyzed": total_count,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": anomaly_pct,
        "anomalies": anomaly_rows[:50],
    }
