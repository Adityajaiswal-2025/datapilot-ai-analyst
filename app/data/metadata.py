import os
import uuid
import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone
from app.schemas.dataset import DatasetMetadata, ColumnSummary, DatasetSummary



# In-memory registry storing dataset metadata and loaded DataFrames
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {}


import re

def is_identifier_column(col_name: str, series: Optional[pd.Series] = None) -> bool:
    """Evaluates whether a column represents an identifier key rather than a business dimension.

    Uses combined evidence:
    - Explicit identifier-like name patterns (Order ID, Customer_ID, UUID, SKU, etc.)
    - Excludes standard code dimensions (e.g., Product Code, Category Code) unless values are ID keys
    - Value pattern heuristics (e.g. B-25601, ORD_123, UUIDs)
    - High uniqueness combined with non-human text
    """
    cl = str(col_name).lower().strip()

    # Explicit identifier names & patterns
    exact_id_names = {
        "id", "index", "uuid", "row_id", "row id", "user_id", "user id",
        "customer_id", "customer id", "order_id", "order id", "product_id", "product id",
        "transaction_id", "transaction id", "invoice_id", "invoice id", "sku", "sku_id", "sku id"
    }
    if cl in exact_id_names:
        return True

    # ID Suffix/Prefix patterns (excluding generic "code")
    id_patterns = [
        r"(?:^|[\s_\-])id$", r"^id[\s_\-]", r"(?:^|[\s_\-])uuid$", r"^uuid[\s_\-]",
        r"order[\s_\-]?id", r"customer[\s_\-]?id", r"product[\s_\-]?id",
        r"transaction[\s_\-]?id", r"invoice[\s_\-]?id"
    ]
    if any(re.search(pat, cl) for pat in id_patterns):
        return True

    # If col_name ends with "code", check combined evidence (do NOT classify by "code" alone)
    if series is not None and not series.empty:
        n_unique = series.nunique(dropna=True)
        total_count = len(series.dropna())

        if total_count >= 10:
            ratio = n_unique / total_count
            sample_vals = [str(x).strip() for x in series.dropna().head(20)]

            # Check if values match common ID format patterns (e.g., B-25601, CUST_001, UUID)
            id_val_regex = re.compile(r"^(?:[A-Za-z]{1,4}[-_/]?\d{3,}|[0-9a-fA-F\-]{32,36})$")
            match_count = sum(1 for v in sample_vals if id_val_regex.match(v))

            if match_count / max(1, len(sample_vals)) > 0.7 and ratio > 0.5:
                # High proportion of ID-structured values + high ratio -> identifier
                return True

            # High uniqueness (> 0.95) with non-human string values
            if ratio > 0.95 and cl in ("code", "item_code", "item code"):
                if match_count > 0:
                    return True

    return False


def sanitize_for_json(val: Any) -> Any:
    """Converts Pandas/NumPy NaN/Inf or non-serializable values into JSON safe formats."""
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    return str(val) if not isinstance(val, (int, float, bool, str)) else val



def extract_metadata(
    dataset_id: str,
    original_filename: str,
    storage_path: str,
    df: pd.DataFrame,
) -> DatasetMetadata:
    """Extracts schema, column summaries, and sample rows from a Pandas DataFrame."""
    file_size = os.path.getsize(storage_path) if os.path.exists(storage_path) else 0
    file_type = os.path.splitext(original_filename)[1].lower()

    columns: List[ColumnSummary] = []
    for col_name in df.columns:
        series = df[col_name]
        col_summary = ColumnSummary(
            name=str(col_name),
            dtype=str(series.dtype),
            null_count=int(series.isnull().sum()),
            unique_count=int(series.nunique(dropna=True)),
        )
        columns.append(col_summary)

    # Extract sample head (up to 5 rows) with JSON serialization safety
    sample_df = df.head(5)
    sample_rows: List[Dict[str, Any]] = []
    for _, row in sample_df.iterrows():
        clean_row = {str(k): sanitize_for_json(v) for k, v in row.items()}
        sample_rows.append(clean_row)

    metadata = DatasetMetadata(
        id=dataset_id,
        filename=original_filename,
        storage_path=storage_path,
        file_size_bytes=file_size,
        file_type=file_type,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        sample_rows=sample_rows,
        created_at=datetime.now(timezone.utc),
    )
    return metadata


def register_dataset(metadata: DatasetMetadata, df: pd.DataFrame) -> None:
    """Registers metadata and loaded DataFrame into dataset storage registry."""
    DATASET_REGISTRY[metadata.id] = {
        "metadata": metadata,
        "dataframe": df,
    }


def get_registered_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves dataset entry by ID."""
    return DATASET_REGISTRY.get(dataset_id)


def list_registered_datasets() -> List[DatasetSummary]:
    """Returns condensed dataset summaries for all registered datasets."""
    summaries: List[DatasetSummary] = []
    for entry in DATASET_REGISTRY.values():
        meta: DatasetMetadata = entry["metadata"]
        summaries.append(
            DatasetSummary(
                id=meta.id,
                filename=meta.filename,
                file_type=meta.file_type,
                file_size_bytes=meta.file_size_bytes,
                row_count=meta.row_count,
                column_count=meta.column_count,
                created_at=meta.created_at,
            )
        )
    return summaries


def delete_registered_dataset(dataset_id: str) -> bool:
    """Removes a dataset from registry and deletes stored file if present."""
    if dataset_id not in DATASET_REGISTRY:
        return False

    entry = DATASET_REGISTRY.pop(dataset_id)
    meta: DatasetMetadata = entry["metadata"]
    if os.path.exists(meta.storage_path):
        try:
            os.remove(meta.storage_path)
        except OSError:
            pass
    return True


def register_merged_dataset(df: pd.DataFrame, filename: str = "merged_dataset.csv") -> Dict[str, Any]:
    """Saves merged DataFrame to storage directory and registers its metadata into DATASET_REGISTRY."""
    from app.core.config import settings

    dataset_id = str(uuid.uuid4())
    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)

    storage_path = os.path.join(upload_dir, f"{dataset_id}_{filename}")
    df.to_csv(storage_path, index=False)

    metadata = extract_metadata(
        dataset_id=dataset_id,
        original_filename=filename,
        storage_path=storage_path,
        df=df,
    )
    register_dataset(metadata, df)
    return {"id": dataset_id, "metadata": metadata}
