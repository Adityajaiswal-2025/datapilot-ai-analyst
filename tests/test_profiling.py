import io
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.tools.profiling import (
    classify_column,
    profile_numeric_column,
    audit_data_quality,
    profile_dataset,
    get_column_info,
)

client = TestClient(app)


def test_column_classification():
    """Test column type classification logic."""
    num_series = pd.Series([10.5, 20.0, 30.2, None], name="price")
    assert classify_column(num_series) == "numeric"

    cat_series = pd.Series(["Electronics", "Furniture", "Electronics", "Clothing"], name="category")
    assert classify_column(cat_series) == "categorical"

    bool_series = pd.Series([True, False, True, True], name="is_active")
    assert classify_column(bool_series) == "boolean"

    date_series = pd.Series(["2026-01-01", "2026-02-15", "2026-03-30"], name="created_at")
    assert classify_column(date_series) == "datetime"


def test_numeric_profiling():
    """Test numeric column statistics calculation."""
    series = pd.Series([10, 20, 30, 40, 50], name="values")
    stats = profile_numeric_column(series)
    assert stats.min == 10.0
    assert stats.max == 50.0
    assert stats.mean == 30.0
    assert stats.median == 30.0
    assert stats.q25 == 20.0
    assert stats.q75 == 40.0


def test_data_quality_audit_and_warnings():
    """Test missing value audit, duplicate detection, and warning flags."""
    df = pd.DataFrame({
        "ID": [1, 2, 2, 4],  # duplicate row
        "Name": ["Alice", "Bob", "Bob", None],  # missing cell
        "Constant": ["Same", "Same", "Same", "Same"],  # constant column
        "HighNull": [None, None, None, 100],  # 75% missing
    })

    audit = audit_data_quality(df)
    assert audit.duplicate_rows_count == 1
    assert audit.missing_cells == 4
    assert len(audit.warnings) >= 3
    # Verify constant column warning
    assert any("Constant" in w for w in audit.warnings)
    # Verify high null warning
    assert any("HighNull" in w for w in audit.warnings)


def test_profile_dataset_tool():
    """Test profile_dataset tool function directly with a DataFrame."""
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40],
        "City": ["NYC", "LA", "NYC", "Chicago"],
    })
    profile = profile_dataset(df=df)
    assert profile.row_count == 4
    assert profile.column_count == 2
    assert len(profile.column_profiles) == 2

    # Targeted get_column_info
    age_col = get_column_info(df=df, column_name="Age")
    assert age_col.classified_type == "numeric"
    assert age_col.numeric_stats is not None
    assert age_col.numeric_stats.mean == 32.5


def test_profile_endpoint():
    """Test GET /api/v1/datasets/{dataset_id}/profile API endpoint."""
    csv_content = "Product,Price,Quantity\nLaptop,1200,2\nMouse,25,10\nKeyboard,75,4"
    file = ("inventory.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")

    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    # Test Profile Endpoint
    profile_res = client.get(f"/api/v1/datasets/{dataset_id}/profile")
    assert profile_res.status_code == 200
    data = profile_res.json()
    assert data["success"] is True
    prof = data["profile"]
    assert prof["dataset_id"] == dataset_id
    assert prof["row_count"] == 3
    assert prof["column_count"] == 3
    assert len(prof["column_profiles"]) == 3

    # Cleanup
    client.delete(f"/api/v1/datasets/{dataset_id}")
