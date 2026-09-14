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


def test_api_query_list_of_orders_production_endpoint():
    """11. Mandatory production endpoint test for List of Orders CSV via POST /api/v1/query."""
    csv_content = (
        "Order ID,Order Date,CustomerName,State,City\n"
        "B-25601,25-01-2019,Bharat,Maharashtra,Mumbai\n"
        "B-25602,26-01-2019,Pearl,Madhya Pradesh,Bhopal\n"
        "B-25603,27-01-2019,Jash,Madhya Pradesh,Indore\n"
        "B-25604,28-01-2019,Divya,Maharashtra,Pune\n"
        "B-25605,29-01-2019,Khitish,Madhya Pradesh,Gwalior\n"
    )
    file = ("List of Orders.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    try:
        payload = {
            "query": "Which states have the highest number of orders?",
            "dataset_id": dataset_id,
        }
        res = client.post("/api/v1/query", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["analyst"] is not None
        assert data["analyst"]["execution_status"] == "success"

        # 1. State is grouping dimension & Order ID is count entity & DESC sorting
        executed_calls = data["analyst"]["tool_calls_executed"]
        assert len(executed_calls) >= 1
        group_call = executed_calls[0]
        assert group_call["tool"] == "group_data"
        assert group_call["params"]["group_by"] == ["State"]
        assert group_call["params"]["aggregations"] == {"Order ID": ["count"]}
        assert group_call["params"]["sort_direction"] == "descending"

        # 2. Result contains state-level groups (not a single dataset-wide count)
        quant = data["analyst"]["quantitative_results"]["step_1_group_data"]
        assert quant["result_row_count"] > 1
        assert quant["rows"][0]["State"] == "Madhya Pradesh"
        assert quant["rows"][0]["Order ID_count"] == 3
        assert quant["rows"][1]["State"] == "Maharashtra"
        assert quant["rows"][1]["Order ID_count"] == 2

        # 3. No Month / Target substitution
        for col in quant["columns"]:
            assert "Month" not in col
            assert "Target" not in col

        # 4. Insight executive summary contains state-level ranking and no Chi-Square primary answer
        insight = data["insight"]
        assert insight is not None
        assert "Madhya Pradesh" in insight["executive_summary"]
        for k in insight.get("key_insights", []):
            assert "Chi-Square test of independence reveals a statistically significant association between 'State' and 'City'" not in k
    finally:
        client.delete(f"/api/v1/datasets/{dataset_id}")


def test_api_temporal_grouping_orders_each_month():
    """Mandatory REST API endpoint test for temporal query 'How many orders were placed each month?'."""
    csv_content = (
        "Order ID,Order Date,CustomerName,State,City\n"
        "B-25601,25-04-2018,Bharat,Maharashtra,Mumbai\n"
        "B-25602,26-04-2018,Pearl,Madhya Pradesh,Bhopal\n"
        "B-25603,27-05-2018,Jash,Madhya Pradesh,Indore\n"
        "B-25604,28-05-2018,Divya,Maharashtra,Pune\n"
        "B-25605,29-06-2018,Khitish,Madhya Pradesh,Gwalior\n"
    )
    file = ("List of Orders.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    try:
        payload = {
            "query": "How many orders were placed each month?",
            "dataset_id": dataset_id,
        }
        res = client.post("/api/v1/query", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["analyst"] is not None
        assert data["analyst"]["execution_status"] == "success"

        # Assertions:
        # - temporal dimension = Order Date
        # - time_grain = month
        # - metric = Order ID
        # - aggregation = count
        executed_calls = data["analyst"]["tool_calls_executed"]
        assert len(executed_calls) >= 1
        group_call = executed_calls[0]
        assert group_call["tool"] == "group_data"
        assert group_call["params"]["group_by"] == ["Order Date"]
        assert group_call["params"]["time_grain"] == "month"
        assert group_call["params"]["aggregations"] == {"Order ID": ["count"]}

        # - multiple month groups
        # - chronological ordering
        # - no dataset-wide-only count
        quant = data["analyst"]["quantitative_results"]["step_1_group_data"]
        assert quant["result_row_count"] == 3
        dates = [r["Order Date"] for r in quant["rows"]]
        assert dates == ["2018-04", "2018-05", "2018-06"]
        assert quant["rows"][0]["Order ID_count"] == 2
        assert quant["rows"][1]["Order ID_count"] == 2
        assert quant["rows"][2]["Order ID_count"] == 1

        assert "Ranked by 'Order Date' (lowest)" not in data["analyst"]["findings_summary"]

        # - no State/City hypothesis replacing the answer
        insight = data["insight"]
        assert insight is not None
        assert "2018-04: 2" in insight["executive_summary"] or "Monthly Breakdown" in str(insight["key_insights"])
        for k in insight.get("key_insights", []):
            assert "Chi-Square test of independence reveals a statistically significant association between 'State' and 'City'" not in k
    finally:
        client.delete(f"/api/v1/datasets/{dataset_id}")


def test_api_exploratory_query_endpoint():
    """Verifies REST API endpoint POST /api/v1/query returns deterministic exploratory finding for open-ended queries."""
    csv_content = (
        "Order ID,Order Date,State,Category,Sales\n"
        "O1,2019-01-05,Madhya Pradesh,Electronics,500.0\n"
        "O2,2019-01-10,Madhya Pradesh,Electronics,800.0\n"
        "O3,2019-01-15,Madhya Pradesh,Electronics,700.0\n"
        "O4,2019-02-05,Maharashtra,Clothing,200.0\n"
        "O5,2019-02-10,Maharashtra,Clothing,100.0\n"
    )
    file = ("exploratory_test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    try:
        payload = {
            "query": "Tell me something interesting about the orders.",
            "dataset_id": dataset_id,
        }
        res = client.post("/api/v1/query", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["analyst"] is not None
        assert data["analyst"]["execution_status"] == "success"

        # Check executed tool calls
        executed_calls = data["analyst"]["tool_calls_executed"]
        assert len(executed_calls) >= 1
        assert executed_calls[0]["tool"] == "group_data"

        # Check findings summary is NOT merely "Aggregations: 'Order ID': count=500"
        summary = data["analyst"]["findings_summary"]
        assert "Aggregations: 'Order ID': count=500" not in summary

        # Check insight output
        insight = data["insight"]
        assert insight is not None
        assert insight["confidence_score"] > 0.50
    finally:
        client.delete(f"/api/v1/datasets/{dataset_id}")


def test_api_agent_query_rehydrates_after_lru_eviction_and_empty_ram():
    """Verifies POST /api/v1/query succeeds when dataset was LRU evicted or RAM registry cleared."""
    from app.data.metadata import DATASET_REGISTRY, evict_lru_datasets

    csv_content = "Product,Price\nLaptop,1000.0\nMouse,50.0\n"
    file = ("rehydrate_agent.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    try:
        # 1. Force LRU RAM Eviction
        evict_lru_datasets(max_allowed=0)
        assert DATASET_REGISTRY[dataset_id]["dataframe"] is None

        # Agent query should trigger rehydration and succeed cleanly
        payload1 = {"query": "Audit missing values", "dataset_id": dataset_id}
        res1 = client.post("/api/v1/query", json=payload1)
        assert res1.status_code == 200
        assert res1.json()["profiler"]["quality_score"] == 100.0

        # 2. Force Empty RAM Cache (simulating backend restart or multi-worker routing)
        DATASET_REGISTRY.clear()
        assert dataset_id not in DATASET_REGISTRY

        # Agent query should rehydrate dataset from DB + Storage into RAM and succeed cleanly
        payload2 = {"query": "Summarize dataset quality", "dataset_id": dataset_id}
        res2 = client.post("/api/v1/query", json=payload2)
        assert res2.status_code == 200
        assert res2.json()["profiler"] is not None
    finally:
        client.delete(f"/api/v1/datasets/{dataset_id}")




