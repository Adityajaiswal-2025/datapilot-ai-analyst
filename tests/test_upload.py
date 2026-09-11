import io
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_upload_valid_csv():
    """Test uploading a valid CSV dataset."""
    csv_content = "Product,Category,Revenue,Units\nLaptop,Electronics,1200.50,5\nPhone,Electronics,800.00,10\nDesk,Furniture,350.00,2"
    file = ("sales.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")

    response = client.post("/api/v1/upload", files={"file": file})
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    dataset = data["dataset"]
    assert dataset["filename"] == "sales.csv"
    assert dataset["file_type"] == ".csv"
    assert dataset["row_count"] == 3
    assert dataset["column_count"] == 4
    assert len(dataset["columns"]) == 4
    assert len(dataset["sample_rows"]) == 3

    # Cleanup dataset
    dataset_id = dataset["id"]
    del_res = client.delete(f"/api/v1/datasets/{dataset_id}")
    assert del_res.status_code == 200


def test_upload_valid_xlsx():
    """Test uploading a valid Excel (.xlsx) dataset."""
    df = pd.DataFrame({
        "Region": ["North", "South", "East", "West"],
        "Sales": [1500, 2300, 1800, 3100],
        "Growth": [0.05, 0.12, 0.08, 0.15],
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)

    file = ("regional_sales.xlsx", buffer, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response = client.post("/api/v1/upload", files={"file": file})
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    dataset = data["dataset"]
    assert dataset["filename"] == "regional_sales.xlsx"
    assert dataset["file_type"] == ".xlsx"
    assert dataset["row_count"] == 4
    assert dataset["column_count"] == 3

    # Cleanup
    dataset_id = dataset["id"]
    client.delete(f"/api/v1/datasets/{dataset_id}")


def test_upload_invalid_file_extension():
    """Test uploading an unsupported file format (.txt)."""
    file = ("document.txt", io.BytesIO(b"Hello World"), "text/plain")
    response = client.post("/api/v1/upload", files={"file": file})
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file format" in data["detail"]


def test_upload_empty_file():
    """Test uploading an empty (0 byte) CSV file."""
    file = ("empty.csv", io.BytesIO(b""), "text/csv")
    response = client.post("/api/v1/upload", files={"file": file})
    assert response.status_code == 400
    data = response.json()
    assert "Uploaded file is empty" in data["detail"]


def test_get_dataset_and_list():
    """Test dataset listing and retrieval endpoints."""
    csv_content = "ID,Value\n1,100\n2,200"
    file = ("test_data.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")

    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    # Test GET /datasets
    list_res = client.get("/api/v1/datasets")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1

    # Test GET /datasets/{dataset_id}
    get_res = client.get(f"/api/v1/datasets/{dataset_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["dataset"]["id"] == dataset_id
    assert get_data["dataset"]["row_count"] == 2

    # Test DELETE /datasets/{dataset_id}
    del_res = client.delete(f"/api/v1/datasets/{dataset_id}")
    assert del_res.status_code == 200

    # Test GET non-existent dataset
    get_404 = client.get(f"/api/v1/datasets/{dataset_id}")
    assert get_404.status_code == 404
