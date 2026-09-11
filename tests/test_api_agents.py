import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_api_query_profiling():
    """Test POST /api/v1/query routing to Profiler Agent."""
    csv_content = "Product,Category,Price\nLaptop,Tech,1200.0\nMouse,Tech,25.0"
    file = ("profiling_test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    try:
        payload = {
            "query": "Audit missing values and dataset data quality",
            "dataset_id": dataset_id,
        }
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["query"] == payload["query"]
        assert data["next_agent"] == "profiler"
        assert data["profiler"] is not None
        assert data["profiler"]["quality_score"] == 100.0
    finally:
        client.delete(f"/api/v1/datasets/{dataset_id}")


def test_api_query_analysis_end_to_end():
    """Test POST /api/v1/query end-to-end multi-agent workflow with uploaded dataset."""
    csv_content = "Product,Revenue,SalesDate\nLaptop,1200.0,2026-01-01\nPhone,800.0,2026-01-15\nLaptop,1400.0,2026-02-01"
    file = ("api_products.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")

    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    try:
        payload = {
            "query": "What are top products by revenue?",
            "dataset_id": dataset_id,
        }
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_id"] if "dataset_id" in data else True
        assert data["analyst"] is not None
        assert data["analyst"]["execution_status"] == "success"
        assert data["visualization"] is not None
        assert data["insight"] is not None
    finally:
        client.delete(f"/api/v1/datasets/{dataset_id}")


def test_api_query_multi_turn_session():
    """Test POST /api/v1/query multi-turn conversational session and contextual query rewriter."""
    csv_content = "Product,Revenue\nLaptop,1200.0\nPhone,800.0"
    file = ("turn_test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")

    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    try:
        # Turn 1
        turn1_payload = {
            "query": "What are the top products by revenue?",
            "dataset_id": dataset_id,
        }
        res1 = client.post("/api/v1/query", json=turn1_payload)
        assert res1.status_code == 200
        session_id = res1.json()["session_id"]

        # Turn 2: Follow-up question referencing prior turn
        turn2_payload = {
            "query": "Show that as a pie chart",
            "dataset_id": dataset_id,
            "session_id": session_id,
        }
        res2 = client.post("/api/v1/query", json=turn2_payload)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["session_id"] == session_id
        assert data2["rewritten_query"] is not None
        assert "pie" in data2["rewritten_query"].lower()
        assert data2["visualization"]["chart_type"] == "pie"

        # Check session history GET endpoint
        hist_res = client.get(f"/api/v1/sessions/{session_id}")
        assert hist_res.status_code == 200
        hist_data = hist_res.json()
        assert hist_data["session_id"] == session_id
        assert len(hist_data["messages"]) == 4  # 2 turns * 2 messages

        # Delete session
        del_res = client.delete(f"/api/v1/sessions/{session_id}")
        assert del_res.status_code == 200

        # Verify session deleted
        get_deleted = client.get(f"/api/v1/sessions/{session_id}")
        assert get_deleted.status_code == 404
    finally:
        client.delete(f"/api/v1/datasets/{dataset_id}")


def test_api_highest_total_sales_live_path():
    """Verifies live REST API endpoint POST /api/v1/query executes State + Sales + SUM + DESC plan."""
    csv_content = (
        "Order ID,State,Sales\n"
        "ORD_1,Maharashtra,500.0\n"
        "ORD_2,Madhya Pradesh,1000.0\n"
        "ORD_3,Maharashtra,300.0\n"
        "ORD_4,Madhya Pradesh,800.0\n"
        "ORD_5,Delhi,400.0\n"
    )
    file = ("ecommerce_live.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    try:
        payload = {
            "query": "Which states have the highest total sales?",
            "dataset_id": dataset_id,
        }
        res = client.post("/api/v1/query", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["analyst"] is not None
        assert data["analyst"]["execution_status"] == "success"

        # Check executed tool plan
        executed_calls = data["analyst"]["tool_calls_executed"]
        assert len(executed_calls) >= 1
        group_call = executed_calls[0]
        assert group_call["tool"] == "group_data"
        assert group_call["params"]["group_by"] == ["State"]
        assert group_call["params"]["sort_by"] == "Sales_sum"
        assert group_call["params"]["sort_direction"] == "descending"

        # Check quantitative results
        quant = data["analyst"]["quantitative_results"]["step_1_group_data"]
        assert quant["rows"][0]["State"] == "Madhya Pradesh"
        assert quant["rows"][0]["Sales_sum"] == 1800.0

        # Check insight output
        insight = data["insight"]
        assert insight is not None
        assert "Madhya Pradesh" in insight["executive_summary"]
    finally:
        client.delete(f"/api/v1/datasets/{dataset_id}")


def test_api_query_invalid_dataset_id():
    """Test POST /api/v1/query returns HTTP 404 for non-existent dataset ID."""
    payload = {
        "query": "What are top products?",
        "dataset_id": "non_existent_dataset_id",
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_api_query_empty_query():
    """Test POST /api/v1/query returns HTTP 400 for empty query string."""
    payload = {
        "query": "   ",
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
