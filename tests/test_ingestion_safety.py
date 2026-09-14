import io
import os
import time
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.data.metadata import (
    DATASET_REGISTRY,
    register_dataset,
    get_registered_dataset,
    get_all_registered_entries,
    delete_registered_dataset,
    cleanup_expired_datasets,
    evict_lru_datasets,
    cleanup_failed_upload,
    extract_metadata,
)
from app.data.validator import DatasetValidationError, DatasetSizeLimitError, validate_filename_safety

client = TestClient(app)


def test_filename_safety_path_traversal():
    """Test path traversal prevention and safe filename extraction."""
    assert validate_filename_safety("../../../etc/passwd.csv") == "passwd.csv"
    assert validate_filename_safety("C:\\Windows\\System32\\test.xlsx") == "test.xlsx"
    with pytest.raises(DatasetValidationError):
        validate_filename_safety("")
    with pytest.raises(DatasetValidationError):
        validate_filename_safety("file\x00with_null.csv")


def test_upload_size_limit_returns_413():
    """Test that uploading a file larger than MAX_UPLOAD_SIZE_MB returns 413 Payload Too Large."""
    orig_limit = settings.MAX_UPLOAD_SIZE_MB
    try:
        settings.MAX_UPLOAD_SIZE_MB = 1
        oversized_data = b"x" * (1 * 1024 * 1024 + 500 * 1024)
        file = ("large_file.csv", io.BytesIO(oversized_data), "text/csv")
        
        response = client.post("/api/v1/upload", files={"file": file})
        assert response.status_code == 413
        assert "exceeds maximum allowed limit" in response.json()["detail"]
    finally:
        settings.MAX_UPLOAD_SIZE_MB = orig_limit


def test_valid_csv_and_xlsx_uploads():
    """Test standard CSV and XLSX uploads succeed and register correctly."""
    csv_bytes = b"ColA,ColB\n1,A\n2,B\n"
    res_csv = client.post("/api/v1/upload", files={"file": ("sample.csv", io.BytesIO(csv_bytes), "text/csv")})
    assert res_csv.status_code == 201
    csv_id = res_csv.json()["dataset"]["id"]

    df = pd.DataFrame({"X": [10, 20], "Y": [30, 40]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    res_xlsx = client.post("/api/v1/upload", files={"file": ("sample.xlsx", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert res_xlsx.status_code == 201
    xlsx_id = res_xlsx.json()["dataset"]["id"]

    delete_registered_dataset(csv_id)
    delete_registered_dataset(xlsx_id)


def test_malformed_file_rejection():
    """Test malformed binary payload submitted as CSV returns 400 Bad Request."""
    malformed_data = b"\x00\x01\x02\x03\xff\xfe\xfd"
    file = ("bad.csv", io.BytesIO(malformed_data), "text/csv")
    res = client.post("/api/v1/upload", files={"file": file})
    assert res.status_code == 400


def test_row_limit_rejection():
    """Test that datasets exceeding MAX_DATASET_ROWS are rejected with structured error."""
    orig_max_rows = settings.MAX_DATASET_ROWS
    try:
        settings.MAX_DATASET_ROWS = 5
        df = pd.DataFrame({"A": range(10)})
        meta = extract_metadata("test_row_lim", "test.csv", "data/uploads/dummy.csv", df)
        with pytest.raises(DatasetValidationError) as exc:
            register_dataset(meta, df)
        assert "row count (10) exceeds maximum allowed limit" in str(exc.value)
    finally:
        settings.MAX_DATASET_ROWS = orig_max_rows


def test_column_limit_rejection():
    """Test that datasets exceeding MAX_DATASET_COLUMNS are rejected with structured error."""
    orig_max_cols = settings.MAX_DATASET_COLUMNS
    try:
        settings.MAX_DATASET_COLUMNS = 3
        df = pd.DataFrame({f"col_{i}": [1, 2] for i in range(5)})
        meta = extract_metadata("test_col_lim", "test.csv", "data/uploads/dummy.csv", df)
        with pytest.raises(DatasetValidationError) as exc:
            register_dataset(meta, df)
        assert "column count (5) exceeds maximum allowed limit" in str(exc.value)
    finally:
        settings.MAX_DATASET_COLUMNS = orig_max_cols


def test_lru_ram_eviction_preserves_disk_file_and_rehydrates():
    """Test that LRU RAM eviction sets 'dataframe' to None, keeps disk file, and rehydrates on demand."""
    orig_max = settings.MAX_DATASETS_IN_MEMORY
    try:
        settings.MAX_DATASETS_IN_MEMORY = 1

        csv_bytes1 = b"ColA\n10\n20\n"
        res1 = client.post("/api/v1/upload", files={"file": ("ds1.csv", io.BytesIO(csv_bytes1), "text/csv")})
        assert res1.status_code == 201
        id1 = res1.json()["dataset"]["id"]
        storage_path1 = res1.json()["dataset"]["storage_path"]

        # Sleep briefly to establish timestamp difference
        time.sleep(0.01)

        csv_bytes2 = b"ColB\n30\n40\n"
        res2 = client.post("/api/v1/upload", files={"file": ("ds2.csv", io.BytesIO(csv_bytes2), "text/csv")})
        assert res2.status_code == 201
        id2 = res2.json()["dataset"]["id"]

        # ds1 should now be evicted from RAM (dataframe is None) BUT its disk file MUST exist
        assert DATASET_REGISTRY[id1]["dataframe"] is None
        assert os.path.exists(storage_path1)

        # Access ds1 via get_registered_dataset to trigger rehydration
        entry1 = get_registered_dataset(id1)
        assert entry1 is not None
        assert entry1["dataframe"] is not None
        assert list(entry1["dataframe"]["ColA"]) == [10, 20]

        # Cleanup
        delete_registered_dataset(id1)
        delete_registered_dataset(id2)
    finally:
        settings.MAX_DATASETS_IN_MEMORY = orig_max


def test_explicit_delete_removes_disk_file():
    """Test that explicit DELETE removes both registry entry and storage file on disk."""
    csv_bytes = b"X,Y\n1,2\n"
    res = client.post("/api/v1/upload", files={"file": ("to_delete.csv", io.BytesIO(csv_bytes), "text/csv")})
    dataset_id = res.json()["dataset"]["id"]
    storage_path = res.json()["dataset"]["storage_path"]

    assert os.path.exists(storage_path)

    del_res = client.delete(f"/api/v1/datasets/{dataset_id}")
    assert del_res.status_code == 200
    assert not os.path.exists(storage_path)
    assert get_registered_dataset(dataset_id) is None


def test_ram_ttl_expiration_preserves_disk_file():
    """Test RAM dataset TTL expiration purges RAM registry entry while preserving disk source file."""
    csv_bytes = b"A,B\n1,2\n"
    res = client.post("/api/v1/upload", files={"file": ("ttl_test.csv", io.BytesIO(csv_bytes), "text/csv")})
    dataset_id = res.json()["dataset"]["id"]
    storage_path = res.json()["dataset"]["storage_path"]

    time.sleep(0.02)
    cleaned = cleanup_expired_datasets(ttl_seconds=0.01)
    assert cleaned >= 1
    assert get_registered_dataset(dataset_id) is None
    assert os.path.exists(storage_path)
    delete_registered_dataset(dataset_id)


def test_cleanup_idempotency():
    """Test calling cleanup functions multiple times is idempotent and error-free."""
    cleanup_failed_upload("non_existent_file.tmp")
    cleanup_failed_upload("non_existent_file.tmp")

    assert delete_registered_dataset("non_existent_id") is False
    assert delete_registered_dataset("non_existent_id") is False


def test_concurrent_registry_access():
    """Test thread-safe concurrent dataset registration and retrieval."""
    import threading

    errors = []

    def worker(worker_id):
        try:
            for i in range(10):
                d_id = f"w_{worker_id}_{i}"
                df = pd.DataFrame({"val": [i]})
                meta = extract_metadata(d_id, f"w{worker_id}_{i}.csv", f"data/uploads/w{worker_id}_{i}.csv", df)
                register_dataset(meta, df)
                entry = get_registered_dataset(d_id)
                assert entry is not None
                delete_registered_dataset(d_id)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
