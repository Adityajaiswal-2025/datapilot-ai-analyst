import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.tools.multi_dataset import (
    join_datasets,
    compare_datasets,
    auto_detect_join_keys,
    validate_join_cardinality,
    MultiDatasetError,
)
from app.agents.analyst import run_analyst_agent
from app.schemas.agents import AnalystOutput

client = TestClient(app)


@pytest.fixture
def sample_customers_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "customer_name": ["Alice", "Bob", "Charlie", "David"],
        "city": ["New York", "London", "Paris", "Tokyo"],
    })


@pytest.fixture
def sample_orders_df():
    return pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105],
        "customer_id": [1, 2, 1, 3, 999],  # 999 is unmatched
        "amount": [250.0, 150.0, 300.0, 450.0, 50.0],
    })


@pytest.fixture
def sample_shipping_df():
    return pd.DataFrame({
        "order_id": [101, 102, 103, 104],
        "shipping_status": ["Delivered", "Shipped", "Delivered", "Pending"],
    })


# --- 1. Key Auto-Detection & Ambiguity Tests ---

def test_auto_detect_join_keys_success(sample_customers_df, sample_orders_df):
    """Test auto-detecting candidate join key customer_id between customers and orders."""
    res = auto_detect_join_keys(dfs=[sample_customers_df, sample_orders_df])
    assert res["is_ambiguous"] is False
    assert res["best_join_keys"] is not None
    assert res["best_join_keys"][0]["left_column"] == "customer_id"
    assert res["best_join_keys"][0]["right_column"] == "customer_id"
    assert len(res["candidates"]) >= 1


def test_auto_detect_join_keys_ambiguity():
    """Test detecting ambiguous join key candidates when multiple columns match."""
    df1 = pd.DataFrame({"id": [1, 2], "code": ["A", "B"], "val": [10, 20]})
    df2 = pd.DataFrame({"id": [1, 2], "code": ["A", "B"], "amount": [100, 200]})
    res = auto_detect_join_keys(dfs=[df1, df2])
    assert res["is_ambiguous"] is True
    assert len(res["warnings"]) >= 1


# --- 2. Cardinality Validation Tests ---

def test_validate_join_cardinality_one_to_many(sample_customers_df, sample_orders_df):
    """Test cardinality validation identifies one-to-many relationship."""
    meta = validate_join_cardinality(sample_customers_df, sample_orders_df, "customer_id", "customer_id")
    assert meta["cardinality"] == "one-to-many"
    assert meta["left_duplicate_keys"] == 0
    assert meta["right_duplicate_keys"] > 0


def test_validate_join_cardinality_many_to_many():
    """Test cardinality validation identifies many-to-many relationship and emits warning."""
    df1 = pd.DataFrame({"tag": ["A", "A", "B"], "val1": [1, 2, 3]})
    df2 = pd.DataFrame({"tag": ["A", "A", "C"], "val2": [10, 20, 30]})
    meta = validate_join_cardinality(df1, df2, "tag", "tag")
    assert meta["cardinality"] == "many-to-many"
    assert len(meta["warnings"]) >= 1


# --- 3. Join Types & Execution Tests ---

def test_join_datasets_inner(sample_customers_df, sample_orders_df):
    """Test inner join returns matching rows only."""
    res = join_datasets(dfs=[sample_customers_df, sample_orders_df], join_type="inner", register_result=False)
    assert res["output_row_count"] == 4  # orders 101, 102, 103, 104
    assert res["stats"]["join_type"] == "inner"
    assert "city" in res["merged_df"].columns
    assert "amount" in res["merged_df"].columns


def test_join_datasets_left(sample_customers_df, sample_orders_df):
    """Test left join retains all left customer rows."""
    res = join_datasets(dfs=[sample_customers_df, sample_orders_df], join_type="left", register_result=False)
    assert res["output_row_count"] >= 4
    assert res["stats"]["join_type"] == "left"


def test_join_datasets_right(sample_customers_df, sample_orders_df):
    """Test right join retains all order rows including unmatched order 105."""
    res = join_datasets(dfs=[sample_customers_df, sample_orders_df], join_type="right", register_result=False)
    assert res["output_row_count"] == 5
    assert res["stats"]["join_type"] == "right"


def test_join_datasets_outer(sample_customers_df, sample_orders_df):
    """Test outer join retains all rows from both datasets."""
    res = join_datasets(dfs=[sample_customers_df, sample_orders_df], join_type="outer", register_result=False)
    assert res["output_row_count"] >= 5
    assert res["stats"]["join_type"] == "outer"


def test_join_datasets_three_tables(sample_customers_df, sample_orders_df, sample_shipping_df):
    """Test 3-table sequential pipeline join."""
    keys = [
        {"left": "customer_id", "right": "customer_id"},
        {"left": "order_id", "right": "order_id"},
    ]
    res = join_datasets(dfs=[sample_customers_df, sample_orders_df, sample_shipping_df], join_keys=keys, join_type="inner", register_result=False)
    assert res["output_row_count"] == 4
    assert "shipping_status" in res["merged_df"].columns


def test_original_datasets_immutability(sample_customers_df, sample_orders_df):
    """Test original DataFrames remain unchanged after join operation."""
    orig_cust_len = len(sample_customers_df)
    orig_orders_len = len(sample_orders_df)
    orig_cust_cols = list(sample_customers_df.columns)

    join_datasets(dfs=[sample_customers_df, sample_orders_df], join_type="inner", register_result=False)

    assert len(sample_customers_df) == orig_cust_len
    assert len(sample_orders_df) == orig_orders_len
    assert list(sample_customers_df.columns) == orig_cust_cols


# --- 4. Join Error & Edge Case Tests ---

def test_join_missing_key_column(sample_customers_df, sample_orders_df):
    """Test joining on non-existent column raises MultiDatasetError."""
    keys = [{"left": "non_existent_col", "right": "customer_id"}]
    with pytest.raises(MultiDatasetError, match="not found"):
        join_datasets(dfs=[sample_customers_df, sample_orders_df], join_keys=keys, register_result=False)


def test_join_invalid_join_type(sample_customers_df, sample_orders_df):
    """Test passing invalid join_type raises MultiDatasetError."""
    with pytest.raises(MultiDatasetError, match="Invalid join_type"):
        join_datasets(dfs=[sample_customers_df, sample_orders_df], join_type="invalid_type", register_result=False)


def test_join_empty_dataset(sample_customers_df):
    """Test joining with empty DataFrame raises MultiDatasetError."""
    empty_df = pd.DataFrame(columns=["customer_id", "amount"])
    with pytest.raises(MultiDatasetError, match="empty"):
        join_datasets(dfs=[sample_customers_df, empty_df], register_result=False)


def test_join_no_matching_rows(sample_customers_df):
    """Test join when datasets share zero matching key values."""
    other_orders = pd.DataFrame({"customer_id": [901, 902], "amount": [10.0, 20.0]})
    res = join_datasets(dfs=[sample_customers_df, other_orders], join_type="inner", register_result=False)
    assert res["output_row_count"] == 0
    assert len(res["warnings"]) >= 1


# --- 5. Dataset Comparison Tests ---

def test_compare_datasets_structured(sample_customers_df, sample_orders_df):
    """Test compare_datasets returns side-by-side metadata and statistics."""
    res = compare_datasets(dfs=[sample_customers_df, sample_orders_df])
    comp = res["comparison_details"]
    assert "row_counts" in comp
    assert "common_columns" in comp
    assert "customer_id" in comp["common_columns"]
    assert "missingness_comparison" in comp
    assert "duplicate_comparison" in comp


# --- 6. Analyst Agent Integration Tests ---

def test_analyst_agent_multi_dataset_query(sample_customers_df, sample_orders_df):
    """Test Analyst Agent generating and executing multi-step join plan for multi-table query."""
    # Combine dataframes for analyst runner
    res = run_analyst_agent(
        query="Which city generated the highest revenue using customers and orders?",
        df=sample_customers_df,
    )
    assert isinstance(res, AnalystOutput)
    assert res.execution_status == "success"
    assert res.requires_visualization is True


# --- 7. REST API Endpoints Tests ---

def test_api_join_and_compare_endpoints():
    """Test POST /api/v1/analyze/join and POST /api/v1/analyze/compare REST API endpoints."""
    # 1. Upload 2 datasets
    cust_csv = "customer_id,name,city\n1,Alice,New York\n2,Bob,London\n"
    ord_csv = "order_id,customer_id,amount\n101,1,250.0\n102,2,150.0\n"

    f1 = ("customers.csv", io.BytesIO(cust_csv.encode("utf-8")), "text/csv")
    f2 = ("orders.csv", io.BytesIO(ord_csv.encode("utf-8")), "text/csv")

    up1 = client.post("/api/v1/upload", files={"file": f1})
    up2 = client.post("/api/v1/upload", files={"file": f2})

    assert up1.status_code == 201
    assert up2.status_code == 201

    id1 = up1.json()["dataset"]["id"]
    id2 = up2.json()["dataset"]["id"]

    try:
        # 2. Call /api/v1/analyze/compare
        comp_res = client.post("/api/v1/analyze/compare", json={"dataset_ids": [id1, id2]})
        assert comp_res.status_code == 200
        comp_data = comp_res.json()
        assert comp_data["comparison_details"]["common_columns"] == ["customer_id"]

        # 3. Call /api/v1/analyze/join
        join_payload = {
            "dataset_ids": [id1, id2],
            "join_type": "inner",
            "register_result": True,
            "new_filename": "joined_cust_orders.csv",
        }
        join_res = client.post("/api/v1/analyze/join", json=join_payload)
        assert join_res.status_code == 200
        join_data = join_res.json()
        assert join_data["stats"]["output_row_count"] == 2
        assert join_data["merged_dataset_id"] is not None

        # Clean up merged dataset
        client.delete(f"/api/v1/datasets/{join_data['merged_dataset_id']}")
    finally:
        client.delete(f"/api/v1/datasets/{id1}")
        client.delete(f"/api/v1/datasets/{id2}")


def test_api_join_invalid_request():
    """Test POST /api/v1/analyze/join returns HTTP 400 for less than 2 dataset IDs."""
    res = client.post("/api/v1/analyze/join", json={"dataset_ids": ["ds_1"]})
    assert res.status_code == 422 or res.status_code == 400


def test_phase19_compare_dataset_metrics():
    """Phase 19: Verifies compare_dataset_metrics computes exact deltas and percentage shifts."""
    from app.tools.multi_dataset import compare_dataset_metrics

    # Create 2 uploaded datasets
    csv1 = "Order ID,Revenue\nO1,100.0\nO2,200.0\n"
    csv2 = "Order ID,Revenue\nO3,150.0\nO4,250.0\n"

    f1 = ("q1.csv", io.BytesIO(csv1.encode("utf-8")), "text/csv")
    f2 = ("q2.csv", io.BytesIO(csv2.encode("utf-8")), "text/csv")

    up1 = client.post("/api/v1/upload", files={"file": f1})
    up2 = client.post("/api/v1/upload", files={"file": f2})

    id1 = up1.json()["dataset"]["id"]
    id2 = up2.json()["dataset"]["id"]

    try:
        res = compare_dataset_metrics(dataset_ids=[id1, id2], metric_column="Revenue", aggregation="sum")
        assert res["execution_status"] == "success"
        deltas = res["deltas"]
        assert len(deltas) == 1
        d = deltas[0]
        assert d["dataset_1_value"] == 300.0
        assert d["dataset_2_value"] == 400.0
        assert d["absolute_delta"] == 100.0
        assert d["percentage_delta"] == 33.33
    finally:
        client.delete(f"/api/v1/datasets/{id1}")
        client.delete(f"/api/v1/datasets/{id2}")


def test_phase19_multi_dataset_edge_cases():
    """Phase 19: Tests multi-dataset edge cases: invalid dataset ID, missing metric column, single dataset isolation."""
    from app.tools.multi_dataset import compare_dataset_metrics, MultiDatasetError

    # 1. Invalid dataset ID raises MultiDatasetError
    with pytest.raises(MultiDatasetError, match="Failed to resolve dataset"):
        compare_dataset_metrics(dataset_ids=["invalid_id_1", "invalid_id_2"], metric_column="Sales")

    # 2. Upload 2 datasets: 1 with Revenue, 1 without Revenue (missing metric column)
    csv1 = "Order ID,Revenue\nO1,100.0\nO2,200.0\n"
    csv2 = "Order ID,Sales\nO3,150.0\nO4,250.0\n"

    up1 = client.post("/api/v1/upload", files={"file": ("d1.csv", io.BytesIO(csv1.encode("utf-8")), "text/csv")})
    up2 = client.post("/api/v1/upload", files={"file": ("d2.csv", io.BytesIO(csv2.encode("utf-8")), "text/csv")})

    id1 = up1.json()["dataset"]["id"]
    id2 = up2.json()["dataset"]["id"]

    try:
        res = compare_dataset_metrics(dataset_ids=[id1, id2], metric_column="Revenue", aggregation="sum")
        assert res["execution_status"] == "success"
        assert len(res["warnings"]) >= 1
        assert "not found in dataset" in res["warnings"][0]
        assert len(res["deltas"]) == 1
    finally:
        client.delete(f"/api/v1/datasets/{id1}")
        client.delete(f"/api/v1/datasets/{id2}")


def test_phase19_single_dataset_query_isolation():
    """Phase 19: Verifies single-dataset queries do NOT enter comparison mode."""
    csv1 = "Order ID,Sales\nO1,100.0\nO2,200.0\n"
    up1 = client.post("/api/v1/upload", files={"file": ("single.csv", io.BytesIO(csv1.encode("utf-8")), "text/csv")})
    id1 = up1.json()["dataset"]["id"]

    try:
        res = client.post("/api/v1/query", json={"query": "What are total sales?", "dataset_id": id1})
        assert res.status_code == 200
        data = res.json()
        assert data["analyst"]["execution_status"] == "success"
        # Must not have multi_dataset comparison in tool calls executed
        tools = [call["tool"] for call in data["analyst"]["tool_calls_executed"]]
        assert "compare_datasets" not in tools
    finally:
        client.delete(f"/api/v1/datasets/{id1}")


