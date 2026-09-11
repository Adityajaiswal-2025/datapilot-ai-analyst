import logging
import os
import uuid
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from app.tools.analysis import _resolve_dataframe, AnalysisToolError
from app.data.metadata import register_merged_dataset
from app.schemas.analysis import JoinKeyCandidate, JoinExecutionStats

logger = logging.getLogger("datapilot.tools.multi_dataset")


class MultiDatasetError(Exception):
    """Custom exception raised when multi-dataset processing encounters unrecoverable errors."""
    pass


def auto_detect_join_keys(
    dataset_ids: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Auto-detects candidate join keys across 2 datasets with evidence scoring and ambiguity detection.

    Evaluates:
    - Column name match / similarity
    - Compatible data types
    - Unique key cardinality
    - Value overlap percentage
    - Confidence score & reason
    """
    resolved_dfs: List[pd.DataFrame] = []
    if dfs is not None and len(dfs) >= 2:
        resolved_dfs = [df.copy() for df in dfs]
    elif dataset_ids is not None and len(dataset_ids) >= 2:
        try:
            resolved_dfs = [_resolve_dataframe(ds_id, None).copy() for ds_id in dataset_ids]
        except Exception as e:
            raise MultiDatasetError(f"Dataset resolution failed for join key detection: {str(e)}") from e
    else:
        raise MultiDatasetError("At least 2 datasets or DataFrames must be provided for join key detection.")

    df_left = resolved_dfs[0]
    df_right = resolved_dfs[1]

    left_cols = list(df_left.columns)
    right_cols = list(df_right.columns)

    candidates: List[JoinKeyCandidate] = []
    warnings: List[str] = []

    for c_left in left_cols:
        s_left = df_left[c_left].dropna()
        dtype_left = str(df_left[c_left].dtype)
        is_left_unique = (df_left[c_left].nunique() == len(df_left)) and (len(df_left) > 0)
        card_left_str = "100% unique (Primary Key Candidate)" if is_left_unique else f"{df_left[c_left].nunique()} unique values"

        for c_right in right_cols:
            s_right = df_right[c_right].dropna()
            dtype_right = str(df_right[c_right].dtype)
            is_right_unique = (df_right[c_right].nunique() == len(df_right)) and (len(df_right) > 0)
            card_right_str = "100% unique (Primary Key Candidate)" if is_right_unique else f"{df_right[c_right].nunique()} unique values"

            # 1. Name match score
            name_score = 0.0
            left_norm = c_left.lower().strip()
            right_norm = c_right.lower().strip()

            if left_norm == right_norm:
                name_score = 1.0
            elif left_norm.endswith("_id") and right_norm.endswith("_id") and left_norm == right_norm:
                name_score = 1.0
            elif left_norm in right_norm or right_norm in left_norm:
                name_score = 0.6

            if name_score == 0.0:
                continue

            # 2. Type compatibility score
            type_compat = False
            left_is_num = pd.api.types.is_numeric_dtype(df_left[c_left])
            right_is_num = pd.api.types.is_numeric_dtype(df_right[c_right])
            left_is_str = pd.api.types.is_string_dtype(df_left[c_left]) or df_left[c_left].dtype == "object"
            right_is_str = pd.api.types.is_string_dtype(df_right[c_right]) or df_right[c_right].dtype == "object"

            if (left_is_num and right_is_num) or (left_is_str and right_is_str):
                type_compat = True

            if not type_compat:
                continue

            # 3. Value overlap calculation
            overlap_pct = 0.0
            if len(s_left) > 0 and len(s_right) > 0:
                set_left = set(s_left.astype(str))
                set_right = set(s_right.astype(str))
                intersection = set_left.intersection(set_right)
                union = set_left.union(set_right)
                if union:
                    overlap_pct = round((len(intersection) / len(union)) * 100.0, 2)

            # 4. Confidence Score Calculation
            score = (name_score * 0.4) + ((overlap_pct / 100.0) * 0.4)
            if is_left_unique or is_right_unique:
                score += 0.2

            score = round(min(1.0, score), 4)

            reason_parts = []
            if left_norm == right_norm:
                reason_parts.append("Exact column name match")
            else:
                reason_parts.append("Partial column name match")

            reason_parts.append(f"Compatible data types ({dtype_left} / {dtype_right})")
            if overlap_pct > 0:
                reason_parts.append(f"{overlap_pct}% value set overlap")
            if is_left_unique:
                reason_parts.append(f"'{c_left}' is unique in left dataset")
            if is_right_unique:
                reason_parts.append(f"'{c_right}' is unique in right dataset")

            recommendation_reason = ". ".join(reason_parts) + "."

            candidates.append(
                JoinKeyCandidate(
                    left_column=c_left,
                    right_column=c_right,
                    left_dtype=dtype_left,
                    right_dtype=dtype_right,
                    cardinality_left=card_left_str,
                    cardinality_right=card_right_str,
                    value_overlap_percentage=overlap_pct,
                    confidence_score=score,
                    recommendation_reason=recommendation_reason,
                    is_ambiguous=False,
                )
            )

    candidates.sort(key=lambda x: x.confidence_score, reverse=True)

    # Detect ambiguity (multiple candidates with score > 0.6 within 0.15 of top score)
    is_ambiguous = False
    best_keys = None

    if len(candidates) >= 2:
        top_score = candidates[0].confidence_score
        competing = [c for c in candidates if c.confidence_score >= 0.6 and (top_score - c.confidence_score) <= 0.15]
        if len(competing) >= 2:
            is_ambiguous = True
            for c in competing:
                c.is_ambiguous = True
            competing_names = [f"('{c.left_column}' -> '{c.right_column}')" for c in competing]
            warnings.append(
                f"Ambiguous join keys detected: multiple plausible join keys found {', '.join(competing_names)}. Please specify explicit join_keys."
            )

    if candidates and not is_ambiguous:
        best_keys = [{"left_column": candidates[0].left_column, "right_column": candidates[0].right_column}]

    return {
        "best_join_keys": best_keys,
        "is_ambiguous": is_ambiguous,
        "candidates": [c.model_dump() for c in candidates],
        "warnings": warnings,
    }


def validate_join_cardinality(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_key: str,
    right_key: str,
) -> Dict[str, Any]:
    """Validates join keys existence, data types, uniqueness, and relationship cardinality."""
    if left_key not in df_left.columns:
        raise MultiDatasetError(f"Left join column '{left_key}' not found in left DataFrame. Columns: {list(df_left.columns)}")
    if right_key not in df_right.columns:
        raise MultiDatasetError(f"Right join column '{right_key}' not found in right DataFrame. Columns: {list(df_right.columns)}")

    s_left = df_left[left_key].dropna()
    s_right = df_right[right_key].dropna()

    left_unique_cnt = df_left[left_key].nunique()
    right_unique_cnt = df_right[right_key].nunique()

    left_is_unique = (left_unique_cnt == len(df_left)) and (len(df_left) > 0)
    right_is_unique = (right_unique_cnt == len(df_right)) and (len(df_right) > 0)

    # Relationship cardinality
    if left_is_unique and right_is_unique:
        cardinality = "one-to-one"
    elif left_is_unique and not right_is_unique:
        cardinality = "one-to-many"
    elif not left_is_unique and right_is_unique:
        cardinality = "many-to-one"
    else:
        cardinality = "many-to-many"

    left_dups = len(df_left) - left_unique_cnt
    right_dups = len(df_right) - right_unique_cnt

    set_left = set(s_left.astype(str))
    set_right = set(s_right.astype(str))
    matched_keys_count = len(set_left.intersection(set_right))

    unmatched_left = len(set_left - set_right)
    unmatched_right = len(set_right - set_left)

    warnings = []
    if cardinality == "many-to-many":
        warnings.append(
            f"Many-to-many relationship detected on join keys ('{left_key}' <-> '{right_key}'). Output row count may expand significantly."
        )

    if matched_keys_count == 0:
        warnings.append(
            f"Zero matching keys found between left column '{left_key}' and right column '{right_key}'."
        )

    return {
        "cardinality": cardinality,
        "left_key": left_key,
        "right_key": right_key,
        "left_unique_count": left_unique_cnt,
        "right_unique_count": right_unique_cnt,
        "left_duplicate_keys": left_dups,
        "right_duplicate_keys": right_dups,
        "matched_unique_keys": matched_keys_count,
        "unmatched_left_unique_keys": unmatched_left,
        "unmatched_right_unique_keys": unmatched_right,
        "warnings": warnings,
    }


def join_datasets(
    dataset_ids: Optional[List[str]] = None,
    join_keys: Optional[List[Dict[str, str]]] = None,
    join_type: str = "inner",
    new_filename: Optional[str] = None,
    register_result: bool = True,
    dfs: Optional[List[pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Executes a deterministic multi-table join pipeline while ensuring original datasets remain immutable.

    Supports 2+ datasets with explicit join order, cardinality validation, and row explosion detection.
    """
    valid_types = {"inner", "left", "right", "outer"}
    j_type = join_type.lower().strip()
    if j_type not in valid_types:
        raise MultiDatasetError(f"Invalid join_type '{join_type}'. Must be one of {valid_types}.")

    resolved_dfs: List[pd.DataFrame] = []
    input_ids: List[str] = []

    if dfs is not None and len(dfs) >= 2:
        for idx, df_item in enumerate(dfs):
            if df_item.empty:
                raise MultiDatasetError(f"Dataset at position {idx+1} is empty (0 rows). Cannot perform join.")
        resolved_dfs = [df.copy() for df in dfs]
        input_ids = [f"df_{i+1}" for i in range(len(dfs))]
    elif dataset_ids is not None and len(dataset_ids) >= 2:
        input_ids = list(dataset_ids)
        for ds_id in dataset_ids:
            try:
                target_df = _resolve_dataframe(ds_id, None).copy()
                if target_df.empty:
                    raise MultiDatasetError(f"Dataset '{ds_id}' is empty (0 rows). Cannot perform join.")
                resolved_dfs.append(target_df)
            except Exception as e:
                raise MultiDatasetError(f"Dataset resolution failed for '{ds_id}': {str(e)}") from e
    else:
        raise MultiDatasetError("At least 2 datasets or DataFrames must be provided to join_datasets().")

    all_warnings: List[str] = []
    keys_used: List[Dict[str, str]] = []

    # Pipeline initialization (Original DataFrames are IMMUTABLE)
    df_current = resolved_dfs[0].copy()
    initial_left_rows = len(df_current)

    row_counts_map = {input_ids[0]: initial_left_rows}

    for idx in range(1, len(resolved_dfs)):
        df_next = resolved_dfs[idx].copy()
        right_id = input_ids[idx]
        row_counts_map[right_id] = len(df_next)

        # Resolve join key mapping for current pair step
        l_col = None
        r_col = None

        if join_keys and len(join_keys) >= idx:
            pair_key = join_keys[idx - 1]
            l_col = pair_key.get("left_column") or pair_key.get("left")
            r_col = pair_key.get("right_column") or pair_key.get("right")

        if not l_col or not r_col:
            # Auto-detect keys for pair
            det_res = auto_detect_join_keys(dfs=[df_current, df_next])
            if det_res.get("is_ambiguous"):
                all_warnings.extend(det_res.get("warnings", []))
                raise MultiDatasetError(
                    f"Join step {idx} between dataset {idx} and dataset {idx+1} failed due to ambiguous candidate keys: {det_res.get('warnings')}"
                )
            best = det_res.get("best_join_keys")
            if not best:
                raise MultiDatasetError(f"Join step {idx} failed: No candidate join keys found between datasets.")
            l_col = best[0]["left_column"]
            r_col = best[0]["right_column"]

        # Validate columns exist
        if l_col not in df_current.columns:
            raise MultiDatasetError(f"Join step {idx} failed: Left column '{l_col}' not found in current merged DataFrame.")
        if r_col not in df_next.columns:
            raise MultiDatasetError(f"Join step {idx} failed: Right column '{r_col}' not found in target DataFrame.")

        # Validate cardinality
        card_meta = validate_join_cardinality(df_current, df_next, l_col, r_col)
        all_warnings.extend(card_meta.get("warnings", []))

        # Safe data type alignment if needed
        l_type = df_current[l_col].dtype
        r_type = df_next[r_col].dtype
        if l_type != r_type:
            try:
                if pd.api.types.is_numeric_dtype(df_current[l_col]) and pd.api.types.is_numeric_dtype(df_next[r_col]):
                    df_current[l_col] = pd.to_numeric(df_current[l_col], errors="coerce")
                    df_next[r_col] = pd.to_numeric(df_next[r_col], errors="coerce")
                else:
                    df_current[l_col] = df_current[l_col].astype(str)
                    df_next[r_col] = df_next[r_col].astype(str)
            except Exception as e:
                raise MultiDatasetError(f"Incompatible data types for join keys '{l_col}' ({l_type}) and '{r_col}' ({r_type}): {str(e)}") from e

        # Pre-join row count check
        pre_join_rows = len(df_current)

        # Columns added from right
        cols_before = set(df_current.columns)

        # Execute pandas merge
        try:
            df_current = pd.merge(
                df_current,
                df_next,
                left_on=l_col,
                right_on=r_col,
                how=j_type, # type: ignore
                suffixes=("", f"_{right_id}")
            )
        except Exception as e:
            raise MultiDatasetError(f"Pandas merge execution failed at step {idx}: {str(e)}") from e

        cols_added = list(set(df_current.columns) - cols_before)

        # Check for row explosion
        row_explosion = False
        max_exp_limit = max(pre_join_rows, len(df_next)) * 2
        if len(df_current) > max_exp_limit:
            row_explosion = True
            all_warnings.append(
                f"Row explosion warning at step {idx}: Merged row count expanded from {pre_join_rows} to {len(df_current)} rows."
            )

        keys_used.append({"left": l_col, "right": r_col})

    output_rows = len(df_current)

    # Matched vs unmatched calculation
    matched_rows = output_rows if j_type == "inner" else min(initial_left_rows, output_rows)
    unmatched_left = max(0, initial_left_rows - matched_rows)
    unmatched_right = max(0, output_rows - initial_left_rows)

    stats = JoinExecutionStats(
        input_dataset_ids=input_ids,
        input_row_counts=row_counts_map,
        join_keys_used=keys_used,
        join_type=j_type, # type: ignore
        output_row_count=output_rows,
        matched_rows_count=matched_rows,
        unmatched_left_count=unmatched_left,
        unmatched_right_count=unmatched_right,
        columns_added=list(df_current.columns),
        cardinality=card_meta["cardinality"] if 'card_meta' in locals() else "unknown", # type: ignore
        row_explosion_warning=any("row explosion" in w.lower() for w in all_warnings),
        execution_status="success" if output_rows > 0 else "partial_success",
    )

    merged_dataset_id = None
    if register_result:
        try:
            fname = new_filename or f"merged_{'_'.join(input_ids[:2])}.csv"
            new_entry = register_merged_dataset(df_current, filename=fname)
            merged_dataset_id = new_entry["id"]
        except Exception as e:
            logger.warning(f"Failed to register merged dataset entry: {e}")

    # Build JSON-safe head sample
    head_sample = []
    if not df_current.empty:
        head_sample = df_current.head(3).replace({np.nan: None}).to_dict(orient="records")

    return {
        "stats": stats.model_dump(),
        "merged_dataset_id": merged_dataset_id,
        "output_row_count": output_rows,
        "sample_rows": head_sample,
        "warnings": all_warnings,
        "merged_df": df_current,
    }


def compare_datasets(
    dataset_ids: Optional[List[str]] = None,
    dfs: Optional[List[pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Computes structured side-by-side comparative metadata and statistical profile analysis across datasets."""
    resolved_dfs: List[pd.DataFrame] = []
    input_ids: List[str] = []

    if dfs is not None and len(dfs) >= 2:
        resolved_dfs = [df.copy() for df in dfs]
        input_ids = [f"dataset_{i+1}" for i in range(len(dfs))]
    elif dataset_ids is not None and len(dataset_ids) >= 2:
        input_ids = list(dataset_ids)
        for ds_id in dataset_ids:
            try:
                target_df = _resolve_dataframe(ds_id, None).copy()
                resolved_dfs.append(target_df)
            except Exception as e:
                raise MultiDatasetError(f"Dataset resolution failed for '{ds_id}': {str(e)}") from e
    else:
        raise MultiDatasetError("At least 2 datasets or DataFrames must be provided for comparison.")

    row_counts = {input_ids[i]: len(resolved_dfs[i]) for i in range(len(resolved_dfs))}
    col_counts = {input_ids[i]: len(resolved_dfs[i].columns) for i in range(len(resolved_dfs))}

    all_col_sets = [set(df.columns) for df in resolved_dfs]
    common_cols = list(set.intersection(*all_col_sets))

    ds_specific_cols = {}
    for i, ds_id in enumerate(input_ids):
        other_cols = set.union(*[all_col_sets[j] for j in range(len(all_col_sets)) if j != i])
        ds_specific_cols[ds_id] = list(all_col_sets[i] - other_cols)

    # Data type differences in common columns
    dtype_diffs = {}
    for col in common_cols:
        types_map = {input_ids[i]: str(resolved_dfs[i][col].dtype) for i in range(len(resolved_dfs))}
        if len(set(types_map.values())) > 1:
            dtype_diffs[col] = types_map

    # Missingness comparison
    missingness = {}
    for i, ds_id in enumerate(input_ids):
        df = resolved_dfs[i]
        total_cells = df.size
        null_cells = int(df.isnull().sum().sum())
        missingness[ds_id] = {
            "total_null_cells": null_cells,
            "null_percentage": round((null_cells / total_cells * 100.0), 2) if total_cells > 0 else 0.0,
        }

    # Duplicate comparison
    duplicates = {}
    for i, ds_id in enumerate(input_ids):
        df = resolved_dfs[i]
        dup_rows = int(df.duplicated().sum())
        duplicates[ds_id] = {
            "duplicate_rows": dup_rows,
            "duplicate_percentage": round((dup_rows / len(df) * 100.0), 2) if len(df) > 0 else 0.0,
        }

    # Statistical differences for common numeric columns
    stat_diffs = {}
    for col in common_cols:
        is_num_all = all(pd.api.types.is_numeric_dtype(df[col]) for df in resolved_dfs)
        if is_num_all:
            col_stats = {}
            for i, ds_id in enumerate(input_ids):
                s = resolved_dfs[i][col].dropna()
                col_stats[ds_id] = {
                    "mean": round(float(s.mean()), 2) if not s.empty else None,
                    "std": round(float(s.std()), 2) if not s.empty and len(s) > 1 else None,
                    "min": round(float(s.min()), 2) if not s.empty else None,
                    "max": round(float(s.max()), 2) if not s.empty else None,
                }
            stat_diffs[col] = col_stats

    warnings = []
    if not common_cols:
        warnings.append("No common columns found across specified datasets.")

    if dtype_diffs:
        warnings.append(f"Common columns with conflicting data types detected: {list(dtype_diffs.keys())}.")

    comparison_details = {
        "dataset_ids": input_ids,
        "row_counts": row_counts,
        "column_counts": col_counts,
        "common_columns": common_cols,
        "dataset_specific_columns": ds_specific_cols,
        "dtype_differences": dtype_diffs,
        "missingness_comparison": missingness,
        "duplicate_comparison": duplicates,
        "statistical_differences": stat_diffs,
    }

    return {
        "dataset_ids": input_ids,
        "comparison_details": comparison_details,
        "warnings": warnings,
    }
