import os
import threading
import uuid
import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any
from datetime import datetime, timezone
from app.schemas.dataset import DatasetMetadata, ColumnSummary, DatasetSummary
from app.core.config import settings
from app.data.validator import DatasetValidationError


# In-memory registry storing dataset metadata and loaded DataFrames
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {}
_registry_lock = threading.Lock()



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
    if isinstance(val, dict):
        return {str(k): sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [sanitize_for_json(v) for v in val]
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


def cleanup_failed_upload(storage_path: str) -> None:
    """Safely removes a temporary file on disk if it exists (idempotent)."""
    if storage_path and os.path.exists(storage_path):
        try:
            os.remove(storage_path)
        except OSError:
            pass


def _is_expired_locked(entry: Dict[str, Any], ttl_seconds: Optional[int] = None) -> bool:
    """Checks if an entry has exceeded its TTL (caller must hold _registry_lock)."""
    ttl = ttl_seconds if ttl_seconds is not None else settings.DATASET_TTL_SECONDS
    if ttl <= 0:
        return False
    last_accessed = entry.get("last_accessed") or entry.get("created_at")
    if not last_accessed:
        return False
    if isinstance(last_accessed, datetime):
        now = datetime.now(timezone.utc)
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)
        elapsed = (now - last_accessed).total_seconds()
    else:
        elapsed = 0
    return elapsed > ttl


def _delete_entry_locked(dataset_id: str, delete_file: bool = False) -> bool:
    """Removes dataset entry from RAM registry and optionally cleans up storage file (caller must hold _registry_lock)."""
    if dataset_id not in DATASET_REGISTRY:
        return False
    entry = DATASET_REGISTRY.pop(dataset_id)
    if delete_file:
        meta = entry.get("metadata")
        if meta and hasattr(meta, "storage_path") and meta.storage_path:
            cleanup_failed_upload(meta.storage_path)
    return True


def cleanup_expired_datasets(ttl_seconds: Optional[int] = None) -> int:
    """Evicts all datasets older than TTL from RAM registry (idempotent, thread-safe). Does NOT delete storage files."""
    count = 0
    with _registry_lock:
        expired_ids = [
            did for did, entry in list(DATASET_REGISTRY.items())
            if _is_expired_locked(entry, ttl_seconds)
        ]
        for did in expired_ids:
            if _delete_entry_locked(did, delete_file=False):
                count += 1
    return count


def _evict_ram_lru_locked(max_allowed: Optional[int] = None) -> int:
    """Frees RAM by setting 'dataframe' to None for LRU entries until RAM count <= max_allowed.

    Does NOT delete source files on disk (caller must hold _registry_lock).
    """
    limit = max_allowed if max_allowed is not None else settings.MAX_DATASETS_IN_MEMORY
    count = 0
    ram_entries = [
        (did, entry) for did, entry in DATASET_REGISTRY.items()
        if entry.get("dataframe") is not None
    ]
    while len(ram_entries) > limit:
        lru_id, _ = min(
            ram_entries,
            key=lambda item: item[1].get("last_accessed") or datetime.min.replace(tzinfo=timezone.utc)
        )
        DATASET_REGISTRY[lru_id]["dataframe"] = None
        count += 1
        ram_entries = [
            (did, entry) for did, entry in DATASET_REGISTRY.items()
            if entry.get("dataframe") is not None
        ]
    return count


def evict_lru_datasets(max_allowed: Optional[int] = None) -> int:
    """Evicts least recently used DataFrames from RAM until count <= max_allowed (thread-safe).

    Source files on disk are preserved for rehydration.
    """
    with _registry_lock:
        return _evict_ram_lru_locked(max_allowed)


def register_dataset(metadata: DatasetMetadata, df: pd.DataFrame) -> None:
    """Registers metadata and loaded DataFrame into dataset storage registry.

    Enforces max rows, max columns, TTL cleanup, and RAM LRU eviction.
    """
    if len(df) > settings.MAX_DATASET_ROWS:
        raise DatasetValidationError(
            f"Dataset row count ({len(df)}) exceeds maximum allowed limit of {settings.MAX_DATASET_ROWS} rows."
        )
    if len(df.columns) > settings.MAX_DATASET_COLUMNS:
        raise DatasetValidationError(
            f"Dataset column count ({len(df.columns)}) exceeds maximum allowed limit of {settings.MAX_DATASET_COLUMNS} columns."
        )

    now = datetime.now(timezone.utc)
    with _registry_lock:
        # Purge TTL expired items from RAM cache
        expired_ids = [
            did for did, entry in list(DATASET_REGISTRY.items())
            if _is_expired_locked(entry)
        ]
        for did in expired_ids:
            _delete_entry_locked(did, delete_file=False)

        DATASET_REGISTRY[metadata.id] = {
            "metadata": metadata,
            "dataframe": df,
            "created_at": now,
            "last_accessed": now,
        }

        # Evict RAM LRU DataFrames if RAM count exceeds limit
        _evict_ram_lru_locked(settings.MAX_DATASETS_IN_MEMORY)


def get_registered_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves dataset entry by ID. If evicted from RAM, rehydrates DataFrame from disk."""
    from app.data.loader import load_dataset

    with _registry_lock:
        entry = DATASET_REGISTRY.get(dataset_id)
        if not entry:
            return None

        if _is_expired_locked(entry):
            _delete_entry_locked(dataset_id, delete_file=False)
            return None

        entry["last_accessed"] = datetime.now(timezone.utc)

        # Check if RAM rehydration from disk storage_path is needed
        if entry.get("dataframe") is None:
            meta: DatasetMetadata = entry["metadata"]
            if meta.storage_path and os.path.exists(meta.storage_path):
                try:
                    df = load_dataset(meta.storage_path)

                    # Re-enforce safety limits
                    if len(df) > settings.MAX_DATASET_ROWS:
                        raise DatasetValidationError(
                            f"Dataset row count ({len(df)}) exceeds maximum allowed limit of {settings.MAX_DATASET_ROWS} rows."
                        )
                    if len(df.columns) > settings.MAX_DATASET_COLUMNS:
                        raise DatasetValidationError(
                            f"Dataset column count ({len(df.columns)}) exceeds maximum allowed limit of {settings.MAX_DATASET_COLUMNS} columns."
                        )

                    entry["dataframe"] = df
                    _evict_ram_lru_locked(settings.MAX_DATASETS_IN_MEMORY)
                except Exception:
                    _delete_entry_locked(dataset_id, delete_file=False)
                    return None
            else:
                _delete_entry_locked(dataset_id, delete_file=False)
                return None

        return entry



def get_all_registered_entries() -> List[Dict[str, Any]]:
    """Thread-safe getter returning snapshots of active dataset entries (rehydrating from disk if needed)."""
    with _registry_lock:
        dataset_ids = list(DATASET_REGISTRY.keys())

    entries = []
    for d_id in dataset_ids:
        entry = get_registered_dataset(d_id)
        if entry:
            entries.append(entry)
    return entries


def list_registered_datasets() -> List[DatasetSummary]:
    """Returns condensed dataset summaries for all valid registered datasets."""
    with _registry_lock:
        expired_ids = [
            did for did, entry in list(DATASET_REGISTRY.items())
            if _is_expired_locked(entry)
        ]
        for did in expired_ids:
            _delete_entry_locked(did, delete_file=False)

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
    """Removes a dataset from registry and deletes stored file if present (idempotent)."""
    with _registry_lock:
        return _delete_entry_locked(dataset_id, delete_file=True)




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
