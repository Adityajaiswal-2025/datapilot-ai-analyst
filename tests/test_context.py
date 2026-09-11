import io
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.data.context import build_dataset_context, build_column_context, format_sample_preview
from app.tools.profiling import profile_dataset

client = TestClient(app)


def test_build_column_context():
    """Test column context formatting logic."""
    df = pd.DataFrame({
        "Revenue": [100.0, 200.0, 300.0],
        "Category": ["A", "B", "A"],
    })
    prof = profile_dataset(df=df)

    rev_col = prof.column_profiles[0]
    rev_str = build_column_context(rev_col)
    assert "Revenue" in rev_str
    assert "numeric" in rev_str
    assert "Stats: min=100.0" in rev_str

    cat_col = prof.column_profiles[1]
    cat_str = build_column_context(cat_col)
    assert "Category" in cat_str
    assert "categorical" in cat_str
    assert "Mode: 'A'" in cat_str


def test_format_sample_preview():
    """Test sample row formatting."""
    rows = [
        {"Product": "Laptop", "Price": 1200},
        {"Product": "Mouse", "Price": 25},
    ]
    preview = format_sample_preview(rows)
    assert "Row 1: { Product: Laptop, Price: 1200 }" in preview
    assert "Row 2: { Product: Mouse, Price: 25 }" in preview


def test_build_dataset_context_full():
    """Test full dataset context string generation."""
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "Score": [90.5, 88.0, 95.0],
    })
    prof = profile_dataset(df=df)
    context_text = build_dataset_context(profile=prof)

    assert "### DATASET OVERVIEW" in context_text
    assert "3 rows × 2 columns" in context_text
    assert "### COLUMN SCHEMA & STATISTICS" in context_text
    assert "Score" in context_text


def test_dataset_context_endpoint():
    """Test GET /api/v1/datasets/{dataset_id}/context API endpoint."""
    csv_content = "Product,Sales\nWidget A,500\nWidget B,750"
    file = ("widgets.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")

    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    # Call Context Endpoint
    ctx_res = client.get(f"/api/v1/datasets/{dataset_id}/context")
    assert ctx_res.status_code == 200
    data = ctx_res.json()
    assert data["success"] is True
    assert data["dataset_id"] == dataset_id
    assert "DATASET OVERVIEW" in data["context_text"]
    assert "widgets.csv" in data["context_text"]
    assert "Widget A" in data["context_text"]

    # Cleanup
    client.delete(f"/api/v1/datasets/{dataset_id}")
