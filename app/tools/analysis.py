import pandas as pd
from typing import Optional, List, Dict, Any
from app.data.metadata import get_registered_dataset, sanitize_for_json, DATASET_REGISTRY


class AnalysisToolError(Exception):
    """Custom exception raised when data filtering or aggregation fails."""
    pass


def _resolve_dataframe(
    dataset_id: Optional[str] = None, df: Optional[pd.DataFrame] = None, query: Optional[str] = None
) -> pd.DataFrame:
    """Helper to resolve DataFrame from dataset_id or raw df parameter.

    If query is provided and the primary dataset does not contain requested columns,
    checks if a merged or comprehensive dataset in DATASET_REGISTRY contains the columns.
    """
    target_df = None
    if dataset_id:
        entry = get_registered_dataset(dataset_id)
        if not entry:
            raise AnalysisToolError(f"Dataset with ID '{dataset_id}' not found in registry.")
        target_df = entry["dataframe"]
    elif df is not None:
        target_df = df

    if target_df is None and DATASET_REGISTRY:
        # Pick latest registered dataset
        target_df = list(DATASET_REGISTRY.values())[-1]["dataframe"]

    # Check if query needs columns missing from target_df but present in another registered dataset
    if query and target_df is not None and DATASET_REGISTRY:
        q_lower = query.lower()
        needs_state = any(kw in q_lower for kw in ["state", "states", "province"])
        has_state = any("state" in c.lower() for c in target_df.columns)

        if needs_state and not has_state:
            for entry in DATASET_REGISTRY.values():
                candidate_df = entry.get("dataframe")
                if candidate_df is not None and any("state" in c.lower() for c in candidate_df.columns):
                    return candidate_df

    if target_df is not None:
        return target_df

    raise AnalysisToolError("Either dataset_id or df must be provided.")


def filter_data(
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Filters a DataFrame based on structured condition dicts.

    Each filter dict format: {"column": "col_name", "operator": "==", "value": val}
    Supported operators: ==, !=, >, >=, <, <=, contains, in
    """
    target_df = _resolve_dataframe(dataset_id, df)
    if not filters:
        filtered_df = target_df.copy()
    else:
        filtered_df = target_df.copy()
        for cond in filters:
            col = cond.get("column")
            op = cond.get("operator", "==")
            val = cond.get("value")

            if not col or col not in filtered_df.columns:
                raise AnalysisToolError(f"Filter column '{col}' does not exist in dataset.")

            series = filtered_df[col]

            if op == "==":
                filtered_df = filtered_df[series == val]
            elif op == "!=":
                filtered_df = filtered_df[series != val]
            elif op == ">":
                filtered_df = filtered_df[series > float(val)]
            elif op == ">=":
                filtered_df = filtered_df[series >= float(val)]
            elif op == "<":
                filtered_df = filtered_df[series < float(val)]
            elif op == "<=":
                filtered_df = filtered_df[series <= float(val)]
            elif op == "contains":
                filtered_df = filtered_df[
                    series.astype(str).str.contains(str(val), case=False, na=False)
                ]
            elif op == "in":
                if not isinstance(val, list):
                    val = [val]
                filtered_df = filtered_df[series.isin(val)]
            else:
                raise AnalysisToolError(f"Unsupported filter operator '{op}'.")

    # Serialize top 100 rows for output
    records = []
    for _, row in filtered_df.head(100).iterrows():
        records.append({str(k): sanitize_for_json(v) for k, v in row.items()})

    return {
        "original_row_count": len(target_df),
        "filtered_row_count": len(filtered_df),
        "columns": list(filtered_df.columns),
        "rows": records,
    }


NUMERIC_ONLY_AGGS = {"sum", "mean", "std", "var", "median"}


def group_data(
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    group_by: Optional[List[str]] = None,
    aggregations: Optional[Dict[str, List[str]]] = None,
    sort_by: Optional[str] = None,
    sort_direction: Optional[str] = None,
    limit: Optional[int] = None,
    time_grain: Optional[str] = None,
) -> Dict[str, Any]:
    """Groups DataFrame by specified columns and applies aggregation functions.

    group_by: ["Region", "Category"] or ["Order Date"]
    aggregations: {"Revenue": ["sum", "mean"], "Units": ["sum"]}
    sort_by: Column name to sort by (e.g., "Revenue_sum" or "Sales")
    sort_direction: "ascending" or "descending"
    limit: Top-N integer row limit
    time_grain: "day" | "week" | "month" | "quarter" | "year"
    """
    target_df = _resolve_dataframe(dataset_id, df)

    if not group_by:
        raise AnalysisToolError("group_by list cannot be empty.")

    for col in group_by:
        if col not in target_df.columns:
            raise AnalysisToolError(f"Group column '{col}' not found in dataset.")

    working_df = target_df.copy()
    input_rows = len(target_df)
    valid_datetime_rows = input_rows
    excluded_datetime_rows = 0

    # Temporal Bucketing & Null Date Handling if time_grain is specified
    if time_grain:
        date_col = group_by[0]
        parsed_dates = pd.to_datetime(working_df[date_col], format="mixed", dayfirst=True, errors="coerce")
        valid_mask = parsed_dates.notnull()
        valid_datetime_rows = int(valid_mask.sum())
        excluded_datetime_rows = input_rows - valid_datetime_rows

        if valid_datetime_rows == 0:
            raise AnalysisToolError(f"Date column '{date_col}' contains no valid parseable dates for time grain '{time_grain}'.")

        # Exclude null/unparseable date rows from temporal grouping
        working_df = working_df[valid_mask].copy()
        valid_dates = parsed_dates[valid_mask]

        grain_lower = str(time_grain).lower()
        if grain_lower == "day":
            working_df[date_col] = valid_dates.dt.strftime("%Y-%m-%d")
        elif grain_lower == "week":
            working_df[date_col] = valid_dates.apply(lambda d: f"{d.isocalendar().year}-W{d.isocalendar().week:02d}")
        elif grain_lower in ("month", "year_month"):
            working_df[date_col] = valid_dates.dt.strftime("%Y-%m")
        elif grain_lower == "quarter":
            working_df[date_col] = valid_dates.apply(lambda d: f"{d.year}-Q{(d.month - 1) // 3 + 1}")
        elif grain_lower == "year":
            working_df[date_col] = valid_dates.dt.strftime("%Y")

    if not aggregations:
        # Default count aggregation
        grouped_df = working_df.groupby(group_by, as_index=False).size().rename(columns={"size": "count"})
    else:
        # Normalize count_distinct to nunique for pandas compatibility
        normalized_aggs = {}
        for col, funcs in aggregations.items():
            if col not in working_df.columns:
                raise AnalysisToolError(f"Aggregation column '{col}' not found in dataset.")
            col_dtype = working_df[col].dtype
            is_numeric = pd.api.types.is_numeric_dtype(col_dtype)
            norm_funcs = []
            for func in funcs:
                func_lower = str(func).lower()
                if func_lower in NUMERIC_ONLY_AGGS and not is_numeric:
                    raise AnalysisToolError(
                        f"Invalid aggregation '{func}' for non-numeric column '{col}' (dtype: {col_dtype})."
                    )
                if func_lower == "count_distinct":
                    norm_funcs.append("nunique")
                else:
                    norm_funcs.append(func)
            normalized_aggs[col] = norm_funcs

        try:
            grouped_df = working_df.groupby(group_by, as_index=False).agg(normalized_aggs)
        except Exception as e:
            raise AnalysisToolError(f"Aggregation failed on dataset: {str(e)}")

        # Flatten multi-level column names if produced
        if isinstance(grouped_df.columns, pd.MultiIndex):
            new_cols = []
            for col_pair in grouped_df.columns:
                if col_pair[1]:
                    new_cols.append(f"{col_pair[0]}_{col_pair[1]}")
                else:
                    new_cols.append(str(col_pair[0]))
            grouped_df.columns = new_cols

    # Explicit sorting if specified, else default to chronological sort for time_grain
    if sort_by:
        target_sort_col = sort_by
        if target_sort_col not in grouped_df.columns:
            matched = [c for c in grouped_df.columns if c.startswith(target_sort_col)]
            if matched:
                target_sort_col = matched[0]

        if target_sort_col in grouped_df.columns:
            is_asc = (sort_direction or "descending").lower() in ["ascending", "asc"]
            grouped_df = grouped_df.sort_values(by=target_sort_col, ascending=is_asc)
        else:
            raise AnalysisToolError(
                f"Sort column '{sort_by}' not found in grouped results. Available columns: {list(grouped_df.columns)}"
            )
    elif time_grain and group_by:
        # Default to chronological sorting on period column
        date_sort_col = group_by[0]
        if date_sort_col in grouped_df.columns:
            is_asc = (sort_direction or "ascending").lower() in ["ascending", "asc"]
            grouped_df = grouped_df.sort_values(by=date_sort_col, ascending=is_asc)

    # Optional top-N limit
    if limit is not None and limit > 0:
        grouped_df = grouped_df.head(limit)

    records = []
    head_rows = limit if (limit and limit > 0) else 200
    for _, row in grouped_df.head(head_rows).iterrows():
        records.append({str(k): sanitize_for_json(v) for k, v in row.items()})

    exec_metadata = {
        "input_rows": input_rows,
        "valid_datetime_rows": valid_datetime_rows,
        "excluded_datetime_rows": excluded_datetime_rows,
        "groups_generated": len(grouped_df),
    }

    return {
        "group_by": group_by,
        "time_grain": time_grain,
        "result_row_count": len(grouped_df),
        "groups_generated": len(grouped_df),
        "input_rows": input_rows,
        "valid_datetime_rows": valid_datetime_rows,
        "excluded_datetime_rows": excluded_datetime_rows,
        "execution_metadata": exec_metadata,
        "columns": list(grouped_df.columns),
        "rows": records,
        "sort_by": sort_by,
        "sort_direction": sort_direction,
        "limit": limit,
    }


def aggregate_data(
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    aggregations: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Applies dataset-wide aggregations without grouping."""
    target_df = _resolve_dataframe(dataset_id, df)
    if not aggregations:
        raise AnalysisToolError("aggregations dictionary must be provided.")

    results: Dict[str, Any] = {}
    for col, funcs in aggregations.items():
        if col not in target_df.columns:
            raise AnalysisToolError(f"Column '{col}' not found in dataset.")
        series = target_df[col].dropna()
        col_res: Dict[str, Any] = {}
        for func in funcs:
            if func == "sum":
                col_res["sum"] = sanitize_for_json(series.sum())
            elif func == "mean":
                col_res["mean"] = sanitize_for_json(series.mean())
            elif func in ("count", "count_non_null"):
                col_res["count"] = int(series.count())
            elif func in ("nunique", "count_distinct"):
                col_res[func] = int(series.nunique())
            elif func == "min":
                col_res["min"] = sanitize_for_json(series.min())
            elif func == "max":
                col_res["max"] = sanitize_for_json(series.max())
            elif func == "std":
                col_res["std"] = sanitize_for_json(series.std())
            else:
                raise AnalysisToolError(f"Unsupported aggregation function '{func}'.")
        results[col] = col_res

    return {
        "row_count": len(target_df),
        "aggregations": results,
    }
