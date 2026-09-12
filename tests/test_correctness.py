import pytest
import pandas as pd
import numpy as np

from app.tools.analysis import group_data, AnalysisToolError
from app.agents.analyst import (
    build_heuristic_plan,
    validate_analysis_plan,
    execute_analysis_plan,
    run_analyst_agent,
)
from app.agents.insight import run_insight_agent, is_hypothesis_relevant_to_query
from app.schemas.agents import AnalysisPlan, AnalysisPlanStep, AnalystOutput


@pytest.fixture
def ecommerce_df():
    """Sample E-Commerce dataset with State, City, Category, and Sales."""
    return pd.DataFrame({
        "State": ["Maharashtra", "Maharashtra", "Karnataka", "Karnataka", "Delhi", "Delhi"],
        "City": ["Mumbai", "Pune", "Bengaluru", "Mysuru", "New Delhi", "Dwarka"],
        "Category": ["Electronics", "Clothing", "Electronics", "Furniture", "Clothing", "Electronics"],
        "Sales": [15000.0, 8000.0, 22000.0, 5000.0, 12000.0, 18000.0],
        "Units": [10, 25, 15, 5, 30, 20],
    })


def test_highest_total_sales_mapping_and_execution(ecommerce_df):
    """Verifies 'Which states have the highest total sales?' infers State + Sales + SUM + descending sort."""
    query = "Which states have the highest total sales?"
    plan = build_heuristic_plan(query, ecommerce_df)

    assert len(plan.steps) >= 1
    step = plan.steps[0]
    assert step.tool_name == "group_data"
    assert step.params["group_by"] == ["State"]
    assert step.params["aggregations"] == {"Sales": ["sum"]}
    assert step.params["sort_by"] == "Sales_sum"
    assert step.params["sort_direction"] == "descending"

    # Execute plan
    calls, quant, errors, status = execute_analysis_plan(plan, ecommerce_df)
    assert status == "success"
    assert "step_1_group_data" in quant
    res = quant["step_1_group_data"]
    assert res["rows"][0]["State"] == "Delhi"  # 12000 + 18000 = 30000
    assert res["rows"][0]["Sales_sum"] == 30000.0


def test_lowest_total_sales_mapping(ecommerce_df):
    """Verifies 'Which states have the lowest total sales?' infers SUM + ascending sort."""
    query = "Which states have the lowest total sales?"
    plan = build_heuristic_plan(query, ecommerce_df)

    step = plan.steps[0]
    assert step.tool_name == "group_data"
    assert step.params["group_by"] == ["State"]
    assert step.params["aggregations"] == {"Sales": ["sum"]}
    assert step.params["sort_by"] == "Sales_sum"
    assert step.params["sort_direction"] == "ascending"

    calls, quant, errors, status = execute_analysis_plan(plan, ecommerce_df)
    assert status == "success"
    res = quant["step_1_group_data"]
    assert res["rows"][0]["State"] == "Maharashtra"  # 15000 + 8000 = 23000
    assert res["rows"][0]["Sales_sum"] == 23000.0


def test_average_sales_mapping(ecommerce_df):
    """Verifies 'What is the average sales by state?' infers MEAN aggregation."""
    query = "What is the average sales by state?"
    plan = build_heuristic_plan(query, ecommerce_df)

    step = plan.steps[0]
    assert step.tool_name == "group_data"
    assert step.params["group_by"] == ["State"]
    assert step.params["aggregations"] == {"Sales": ["mean"]}


def test_explicitly_invalid_aggregation_rejected(ecommerce_df):
    """Verifies requesting numeric aggregation 'mean' on string column 'City' raises validation/tool error."""
    # 1. Direct tool invocation check
    with pytest.raises(AnalysisToolError) as exc_info:
        group_data(df=ecommerce_df, group_by=["State"], aggregations={"City": ["mean"]})
    assert "Invalid aggregation 'mean' for non-numeric column 'City'" in str(exc_info.value)

    # 2. Plan validation check
    invalid_plan = AnalysisPlan(
        analysis_goal="Invalid mean on string column",
        steps=[
            AnalysisPlanStep(
                tool_name="group_data",
                params={"group_by": ["State"], "aggregations": {"City": ["mean"]}},
                purpose="Invalid test",
            )
        ],
    )
    val_plan, warnings = validate_analysis_plan(invalid_plan, ecommerce_df)
    assert any("Cannot perform numeric aggregation 'mean' on non-numeric column 'City'" in w for w in warnings)


def test_ambiguous_aggregation_no_silent_correction(ecommerce_df):
    """Verifies invalid explicitly planned operations are not silently mutated to dummy defaults."""
    bad_plan = AnalysisPlan(
        analysis_goal="Explicit invalid group plan",
        steps=[
            AnalysisPlanStep(
                tool_name="group_data",
                params={"group_by": ["State"], "aggregations": {"Category": ["sum"]}},
                purpose="Explicit sum on Category",
            )
        ],
    )
    calls, quant, errors, status = execute_analysis_plan(bad_plan, ecommerce_df)
    assert status == "failed"
    assert len(errors) > 0
    assert "Invalid aggregation 'sum' for non-numeric column 'Category'" in errors[0]


def test_top_n_ranking_limit(ecommerce_df):
    """Verifies 'top 2 states by total sales' applies limit parameter."""
    query = "What are the top 2 states by total sales?"
    plan = build_heuristic_plan(query, ecommerce_df)

    step = plan.steps[0]
    assert step.params.get("limit") == 2

    calls, quant, errors, status = execute_analysis_plan(plan, ecommerce_df)
    assert status == "success"
    res = quant["step_1_group_data"]
    assert len(res["rows"]) == 2


def test_failed_analysis_no_unrelated_phase17_insights(ecommerce_df):
    """Verifies that if analysis fails, Insight Agent returns failure explanation and zero unrelated Chi-Square hypotheses."""
    query = "Which states have the highest total sales?"
    failed_analyst_out = AnalystOutput(
        analysis_goal="Failed execution test",
        execution_status="failed",
        errors=["Step 1 (group_data) failed: Invalid aggregation 'mean' for non-numeric column 'City'."],
        findings_summary="Analysis completed with errors: Step 1 (group_data) failed: Invalid aggregation 'mean' for non-numeric column 'City'.",
        quantitative_results={},
    )

    insight_out = run_insight_agent(
        query=query,
        df=ecommerce_df,
        analyst_output=failed_analyst_out,
    )

    assert insight_out.confidence_score == 0.0
    assert "Analysis execution failed" in insight_out.executive_summary
    assert len(insight_out.hypotheses) == 0  # NO unrelated Phase 17 hypotheses!
    # Confirm State vs City Chi-Square test is NOT present
    for h in insight_out.hypotheses:
        assert "State vs City" not in str(h)
    for k in insight_out.key_insights:
        assert "Chi-Square" not in k


def test_hypothesis_relevance_filtering():
    """Verifies is_hypothesis_relevant_to_query correctly filters out unrelated hypotheses."""
    query = "Which states have the highest total sales?"
    active_cols = ["State", "Sales"]

    relevant_hypo = {
        "statement": "State vs Sales ANOVA test...",
        "target_column": "Sales",
        "group_column": "State",
    }

    assert is_hypothesis_relevant_to_query(relevant_hypo, query, active_cols) is True


def test_successful_analysis_with_relevant_insights(ecommerce_df):
    """Verifies end-to-end Analyst and Insight Agent execution on 'Which states have the highest total sales?'."""
    query = "Which states have the highest total sales?"
    analyst_out = run_analyst_agent(query=query, df=ecommerce_df)

    assert analyst_out.execution_status == "success"
    assert "Delhi" in analyst_out.findings_summary

    insight_out = run_insight_agent(query=query, df=ecommerce_df, analyst_output=analyst_out)
    assert insight_out.confidence_score > 0.0
    assert "Delhi" in insight_out.executive_summary or "Top performing" in str(insight_out.key_insights)
    # Ensure no unrelated State-vs-City Chi-Square finding appears
    for k in insight_out.key_insights:
        assert "Chi-Square test of independence reveals a statistically significant association between 'State' and 'City'" not in k


def test_partial_multi_step_analysis_status_tracking(ecommerce_df):
    """Verifies partial success status when step 1 succeeds and step 2 fails."""
    partial_plan = AnalysisPlan(
        analysis_goal="Partial multi-step test",
        steps=[
            AnalysisPlanStep(
                tool_name="filter_data",
                params={"filters": [{"column": "Sales", "operator": ">", "value": 10000}]},
                purpose="Filter high sales",
            ),
            AnalysisPlanStep(
                tool_name="group_data",
                params={"group_by": ["State"], "aggregations": {"City": ["mean"]}},
                purpose="Invalid group step",
            ),
        ],
    )
    calls, quant, errors, status = execute_analysis_plan(partial_plan, ecommerce_df)
    assert status == "partial_success"
    assert len(errors) == 1
    assert "step_1_filter_data" in quant


def test_identifier_detection_combined_evidence():
    """Verifies combined-evidence identifier protection logic."""
    from app.data.metadata import is_identifier_column

    # Explicit identifier names
    assert is_identifier_column("Order ID") is True
    assert is_identifier_column("customer_id") is True
    assert is_identifier_column("UUID") is True
    assert is_identifier_column("Transaction ID") is True

    # Legitimate business dimensions (must NOT be classified as identifiers)
    assert is_identifier_column("Product") is False
    assert is_identifier_column("CustomerName") is False
    assert is_identifier_column("State") is False
    assert is_identifier_column("City") is False
    assert is_identifier_column("Product Code") is False
    assert is_identifier_column("Category Code") is False

    # ID format value pattern check
    s_order_ids = pd.Series([f"B-256{i:02d}" for i in range(50)])
    assert is_identifier_column("Order_Number", series=s_order_ids) is True


def test_explicit_order_id_request():
    """Verifies Order ID is allowed ONLY when explicitly requested by user."""
    df = pd.DataFrame({
        "Order ID": ["ORD_1", "ORD_2", "ORD_3"],
        "Sales": [100.0, 200.0, 150.0],
    })
    query = "Show sales by order ID"
    plan = build_heuristic_plan(query, df)

    assert len(plan.steps) == 1
    assert plan.steps[0].params["group_by"] == ["Order ID"]


def test_generic_question_no_order_id_default():
    """Verifies generic query does NOT default to Order ID."""
    df = pd.DataFrame({
        "Order ID": [f"ORD_{i}" for i in range(10)],
        "Category": ["A", "B"] * 5,
        "Sales": [10.0] * 10,
    })
    query = "What are the total sales?"
    plan = build_heuristic_plan(query, df)

    if plan.steps and plan.steps[0].tool_name == "group_data":
        assert "Order ID" not in plan.steps[0].params["group_by"]


def test_missing_state_column_structured_error():
    """Verifies asking for 'states' on a dataset without State returns structured resolution without substituting Order ID."""
    df = pd.DataFrame({
        "Order ID": ["ORD_1", "ORD_2"],
        "Category": ["Tech", "Office"],
        "Sales": [500.0, 300.0],
    })
    query = "Which states have the highest total sales?"
    plan = build_heuristic_plan(query, df)

    # Must NOT substitute Order ID
    for step in plan.steps:
        if step.tool_name == "group_data":
            assert "Order ID" not in step.params["group_by"]


def test_multiple_candidate_metrics_resolution():
    """Verifies exact metric matching when both Sales and Revenue columns exist."""
    df = pd.DataFrame({
        "State": ["CA", "NY"],
        "Sales": [100.0, 200.0],
        "Revenue": [120.0, 240.0],
    })
    query = "Which states have the highest total sales?"
    plan = build_heuristic_plan(query, df)

    assert plan.steps[0].params["aggregations"] == {"Sales": ["sum"]}


def test_concentration_math_safety():
    """Verifies concentration analysis excludes identifiers, handles negative profits and zero totals safely."""
    from app.tools.insights_engine import discover_automated_insights

    # 1. Dataset with negative profit
    df_negative = pd.DataFrame({
        "Order ID": [f"ORD_{i}" for i in range(20)],
        "Category": ["Electronics"] * 10 + ["Clothing"] * 10,
        "Sales": [1000.0] * 20,
        "Profit": [500.0] * 10 + [-600.0] * 10,  # Mixed positive/negative
    })
    insights_neg = discover_automated_insights(df_negative)

    # Identifiers must NOT appear in concentration headlines
    for ins in insights_neg["insights"]:
        assert "Order ID" not in ins["headline"]
        # Share percentage must be within [0, 100]
        if "top_segment_share_percentage" in ins.get("evidence", {}):
            pct = ins["evidence"]["top_segment_share_percentage"]
            assert 0.0 <= pct <= 100.0

    # 2. Dataset with zero total
    df_zero = pd.DataFrame({
        "Category": ["A", "B", "C"],
        "Sales": [0.0, 0.0, 0.0],
    })
    insights_zero = discover_automated_insights(df_zero)
    assert len(insights_zero["insights"]) == 0


def test_highest_number_of_orders_by_state():
    """1. Regression test: highest number of orders by state."""
    df = pd.DataFrame({
        "Order ID": ["O1", "O2", "O3", "O4", "O5"],
        "State": ["MH", "MH", "KA", "MH", "KA"],
        "City": ["Mumbai", "Pune", "Bengaluru", "Mumbai", "Bengaluru"],
    })
    query = "highest number of orders by state"
    plan = build_heuristic_plan(query, df)
    v_plan, warnings = validate_analysis_plan(plan, df)

    assert len(v_plan.steps) == 1
    step = v_plan.steps[0]
    assert step.tool_name == "group_data"
    assert step.params["group_by"] == ["State"]
    assert step.params["aggregations"] == {"Order ID": ["count"]}
    assert step.params["sort_by"] == "Order ID_count"
    assert step.params["sort_direction"] == "descending"

    calls, quant, errors, status = execute_analysis_plan(v_plan, df)
    assert status == "success"
    res = quant["step_1_group_data"]
    assert res["rows"][0]["State"] == "MH"
    assert res["rows"][0]["Order ID_count"] == 3


def test_lowest_number_of_orders_by_state():
    """2. Regression test: lowest number of orders by state."""
    df = pd.DataFrame({
        "Order ID": ["O1", "O2", "O3", "O4", "O5"],
        "State": ["MH", "MH", "KA", "MH", "KA"],
    })
    query = "lowest number of orders by state"
    plan = build_heuristic_plan(query, df)
    v_plan, warnings = validate_analysis_plan(plan, df)

    step = v_plan.steps[0]
    assert step.tool_name == "group_data"
    assert step.params["sort_direction"] == "ascending"

    calls, quant, errors, status = execute_analysis_plan(v_plan, df)
    assert status == "success"
    res = quant["step_1_group_data"]
    assert res["rows"][0]["State"] == "KA"
    assert res["rows"][0]["Order ID_count"] == 2


def test_top_5_states_by_order_count():
    """3. Regression test: top 5 states by order count."""
    df = pd.DataFrame({
        "Order ID": [f"O{i}" for i in range(20)],
        "State": [f"S{i%10}" for i in range(20)],
    })
    query = "top 5 states by order count"
    plan = build_heuristic_plan(query, df)
    v_plan, warnings = validate_analysis_plan(plan, df)

    step = v_plan.steps[0]
    assert step.params["limit"] == 5
    calls, quant, errors, status = execute_analysis_plan(v_plan, df)
    assert status == "success"
    assert len(quant["step_1_group_data"]["rows"]) == 5


def test_highest_number_of_orders_by_city():
    """4. Regression test: highest number of orders by city."""
    df = pd.DataFrame({
        "Order ID": ["O1", "O2", "O3", "O4"],
        "City": ["Mumbai", "Delhi", "Mumbai", "Mumbai"],
    })
    query = "highest number of orders by city"
    plan = build_heuristic_plan(query, df)
    v_plan, warnings = validate_analysis_plan(plan, df)

    step = v_plan.steps[0]
    assert step.params["group_by"] == ["City"]
    assert step.params["aggregations"] == {"Order ID": ["count"]}


def test_highest_total_profit_by_state(ecommerce_df):
    """6. Regression test: highest total profit by state."""
    df = ecommerce_df.copy()
    df["Profit"] = [1000.0, 500.0, 3000.0, 200.0, 1500.0, 2000.0]
    query = "highest total profit by state"
    plan = build_heuristic_plan(query, df)
    v_plan, warnings = validate_analysis_plan(plan, df)

    step = v_plan.steps[0]
    assert step.params["group_by"] == ["State"]
    assert step.params["aggregations"] == {"Profit": ["sum"]}
    assert step.params["sort_direction"] == "descending"


def test_plain_total_order_count_without_grouping():
    """8. Regression test: plain total order count without grouping."""
    df = pd.DataFrame({
        "Order ID": ["O1", "O2", "O3"],
        "State": ["MH", "KA", "DL"],
    })
    query = "What is the total order count?"
    plan = build_heuristic_plan(query, df)
    v_plan, warnings = validate_analysis_plan(plan, df)

    step = v_plan.steps[0]
    assert step.tool_name == "aggregate_data"
    assert step.params["aggregations"] == {"Order ID": ["count"]}


def test_malformed_wrong_llm_plan_replanning():
    """9. Regression test: validator rejects un-grouped plan for ranking question and re-plans with dimension grouping."""
    df = pd.DataFrame({
        "Order ID": ["O1", "O2", "O3"],
        "State": ["MH", "KA", "MH"],
    })
    # Simulate wrong LLM output: aggregate_data instead of group_data for a ranking query
    bad_plan = AnalysisPlan(
        analysis_goal="Which states have the highest number of orders?",
        steps=[
            AnalysisPlanStep(
                tool_name="aggregate_data",
                params={"aggregations": {"Order ID": ["count"]}},
                purpose="Wrong un-grouped count",
            )
        ]
    )
    v_plan, warnings = validate_analysis_plan(bad_plan, df)

    assert len(v_plan.steps) == 1
    step = v_plan.steps[0]
    assert step.tool_name == "group_data"
    assert step.params["group_by"] == ["State"]
    assert step.params["aggregations"] == {"Order ID": ["count"]}
    assert step.params["sort_direction"] == "descending"
    assert any("Re-planning with group_data" in w for w in warnings)


def test_irrelevant_insight_hypothesis_suppression():
    """10. Regression test: suppresses State vs City Chi-Square as primary answer for direct state ranking query."""
    df = pd.DataFrame({
        "Order ID": [f"O{i}" for i in range(10)],
        "State": ["MH"] * 5 + ["KA"] * 5,
        "City": ["Mumbai"] * 5 + ["Bengaluru"] * 5,
    })
    query = "Which states have the highest number of orders?"
    analyst_out = run_analyst_agent(query=query, df=df)
    insight_out = run_insight_agent(query=query, df=df, analyst_output=analyst_out)

    assert "Ranked by" in insight_out.executive_summary or "State" in insight_out.executive_summary
    for k in insight_out.key_insights:
        assert "Chi-Square test of independence reveals a statistically significant association between 'State' and 'City'" not in k


def test_order_id_resolution_priority_and_exclusion():
    """Verifies resolution priority order and exclusion of Customer ID / Product ID."""
    from app.agents.analyst import resolve_query_intent

    # Priority 1: Exact 'Order ID'
    df1 = pd.DataFrame({"Order ID": ["O1"], "Customer ID": ["C1"], "State": ["MH"]})
    res1 = resolve_query_intent("highest number of orders by state", df1)
    assert res1["metric"] == "Order ID"

    # Priority 2: Normalized 'order_id'
    df2 = pd.DataFrame({"order_id": ["O1"], "customer_id": ["C1"], "State": ["MH"]})
    res2 = resolve_query_intent("number of orders by state", df2)
    assert res2["metric"] == "order_id"

    # Priority 3: Order_No
    df3 = pd.DataFrame({"Order_No": ["O1"], "Customer ID": ["C1"], "State": ["MH"]})
    res3 = resolve_query_intent("most orders by state", df3)
    assert res3["metric"] == "Order_No"

    # Priority 4 & Exclusion: Dataset with ONLY Customer ID and Product ID (NO Order ID)
    df4 = pd.DataFrame({"Customer ID": ["C1"], "Product ID": ["P1"], "State": ["MH"]})
    res4 = resolve_query_intent("highest number of orders by state", df4)
    # Must NOT select Customer ID or Product ID for order count
    assert res4["metric"] != "Customer ID"
    assert res4["metric"] != "Product ID"


def test_13_object_string_date_column_safely_parsed():
    """13. Object/string date column that can be safely parsed."""
    df = pd.DataFrame({
        "Order Date": ["2018-04-15", "2018-04-20", "2018-05-10", "2018-05-15"],
        "Order ID": ["O1", "O2", "O3", "O4"],
    })
    query = "How many orders were placed each month?"
    plan = build_heuristic_plan(query, df)
    v_plan, warnings = validate_analysis_plan(plan, df)

    assert v_plan.steps[0].params["group_by"] == ["Order Date"]
    assert v_plan.steps[0].params["time_grain"] == "month"

    calls, quant, errors, status = execute_analysis_plan(v_plan, df)
    assert status == "success"
    res = quant["step_1_group_data"]
    assert len(res["rows"]) == 2
    assert res["rows"][0]["Order Date"] == "2018-04"
    assert res["rows"][1]["Order Date"] == "2018-05"


def test_14_invalid_null_datetime_values_excluded_and_reported():
    """14. Invalid/null datetime values are excluded and reported in execution metadata."""
    df = pd.DataFrame({
        "Order Date": ["2018-04-15", None, "2018-04-20", "invalid_date", "2018-05-10"],
        "Order ID": ["O1", "O2", "O3", "O4", "O5"],
    })
    res = group_data(df=df, group_by=["Order Date"], aggregations={"Order ID": ["count"]}, time_grain="month")
    
    meta = res["execution_metadata"]
    assert meta["input_rows"] == 5
    assert meta["valid_datetime_rows"] == 3
    assert meta["excluded_datetime_rows"] == 2
    assert meta["groups_generated"] == 2


def test_15_iso_week_grouping_preserves_year():
    """15. ISO week grouping preserves year (YYYY-Www)."""
    df = pd.DataFrame({
        "Order Date": ["2018-01-05", "2019-01-05"],
        "Order ID": ["O1", "O2"],
    })
    res = group_data(df=df, group_by=["Order Date"], aggregations={"Order ID": ["count"]}, time_grain="week")
    labels = [r["Order Date"] for r in res["rows"]]
    assert any("2018-W" in l for l in labels)
    assert any("2019-W" in l for l in labels)
    assert labels[0] != labels[1]  # Does NOT merge week 1 of 2018 with week 1 of 2019!


def test_16_which_month_highest_performs_ranking():
    """16. 'which month highest' performs ranking (descending sort + limit=1)."""
    df = pd.DataFrame({
        "Order Date": ["2018-04-15", "2018-04-20", "2018-05-10", "2018-05-11", "2018-05-12"],
        "Order ID": ["O1", "O2", "O3", "O4", "O5"],
    })
    query = "Which month had the highest number of orders?"
    plan = build_heuristic_plan(query, df)
    v_plan, warnings = validate_analysis_plan(plan, df)

    step = v_plan.steps[0]
    assert step.params["sort_direction"] == "descending"
    assert step.params.get("limit") == 1

    calls, quant, errors, status = execute_analysis_plan(v_plan, df)
    res = quant["step_1_group_data"]
    assert res["rows"][0]["Order Date"] == "2018-05"
    assert res["rows"][0]["Order ID_count"] == 3


def test_17_which_month_lowest_performs_ascending_ranking():
    """17. 'which month lowest' performs ascending ranking (ascending sort + limit=1)."""
    df = pd.DataFrame({
        "Order Date": ["2018-04-15", "2018-04-20", "2018-05-10"],
        "Order ID": ["O1", "O2", "O3"],
    })
    query = "Which month had the lowest number of orders?"
    plan = build_heuristic_plan(query, df)
    v_plan, warnings = validate_analysis_plan(plan, df)

    step = v_plan.steps[0]
    assert step.params["sort_direction"] == "ascending"
    assert step.params.get("limit") == 1

    calls, quant, errors, status = execute_analysis_plan(v_plan, df)
    res = quant["step_1_group_data"]
    assert res["rows"][0]["Order Date"] == "2018-05"
    assert res["rows"][0]["Order ID_count"] == 1


def test_18_monthly_trend_requests_visualization():
    """18. 'monthly trend' requests visualization."""
    df = pd.DataFrame({
        "Order Date": ["2018-04-15", "2018-05-10"],
        "Sales": [100.0, 200.0],
    })
    query = "Show monthly order trend"
    analyst_out = run_analyst_agent(query=query, df=df)
    assert analyst_out.requires_visualization is True


def test_19_date_like_column_poor_parse_rate_not_treated_as_datetime():
    """19. Date-like column with poor parse rate is not blindly treated as datetime."""
    from app.agents.analyst import resolve_datetime_column

    df = pd.DataFrame({
        "date_code": ["invalid_1", "invalid_2", "invalid_3", "2018-04-15"],  # 25% parse rate < 50% threshold
        "Sales": [10, 20, 30, 40],
    })
    dt_col, rate = resolve_datetime_column(df)
    assert dt_col is None
    assert rate < 0.50


def test_n_temporal_breakdown_no_ranking_label():
    """N. Ensures temporal breakdown findings string contains NO 'Ranked by Order Date (lowest)' label."""
    df = pd.DataFrame({
        "Order Date": ["2018-04-15", "2018-05-10"],
        "Order ID": ["O1", "O2"],
    })
    query = "How many orders were placed each month?"
    analyst_out = run_analyst_agent(query=query, df=df)
    
    assert "Ranked by 'Order Date' (lowest)" not in analyst_out.findings_summary
    assert "Temporal breakdown grouped by 'Order Date'" in analyst_out.findings_summary
    assert "2018-04: 1" in analyst_out.findings_summary
    assert "2018-05: 1" in analyst_out.findings_summary


def test_o_count_order_id_distinct_from_group_column():
    """O. Ensures COUNT(Order ID) is present in executed tool calls and group results contain distinct numeric counts."""
    df = pd.DataFrame({
        "Order Date": ["2018-04-15", "2018-04-20", "2018-05-10"],
        "Order ID": ["O1", "O2", "O3"],
    })
    query = "How many orders were placed each month?"
    analyst_out = run_analyst_agent(query=query, df=df)
    
    executed = analyst_out.tool_calls_executed[0]
    assert executed["tool"] == "group_data"
    assert executed["params"]["group_by"] == ["Order Date"]
    assert executed["params"]["aggregations"] == {"Order ID": ["count"]}
    
    rows = analyst_out.quantitative_results["step_1_group_data"]["rows"]
    assert len(rows) == 2
    assert rows[0]["Order Date"] == "2018-04"
    assert rows[0]["Order ID_count"] == 2
    assert rows[1]["Order Date"] == "2018-05"
    assert rows[1]["Order ID_count"] == 1


def test_exploratory_dataset_no_state_column():
    """Verifies exploratory query on dataset with no State column works dynamically."""
    df = pd.DataFrame({
        "Product": ["Laptop", "Laptop", "Laptop", "Keyboard", "Mouse"],
        "Revenue": [25000.0, 30000.0, 23000.0, 1500.0, 500.0],
    })
    query = "Tell me something interesting about the orders."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    assert "State" not in analyst_out.findings_summary
    assert len(analyst_out.tool_calls_executed) >= 1
    call = analyst_out.tool_calls_executed[0]
    assert call["tool"] == "group_data"
    assert call["params"]["group_by"] == ["Product"]


def test_exploratory_dataset_no_order_id_column():
    """Verifies exploratory query on dataset with no Order ID column works dynamically."""
    df = pd.DataFrame({
        "Category": ["Electronics", "Electronics", "Electronics", "Clothing", "Furniture"],
        "Sales": [5000.0, 6000.0, 5000.0, 2000.0, 1500.0],
    })
    query = "Find something interesting in this dataset."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    assert "Order ID" not in analyst_out.findings_summary
    assert len(analyst_out.tool_calls_executed) >= 1
    call = analyst_out.tool_calls_executed[0]
    assert call["tool"] == "group_data"
    assert call["params"]["group_by"] == ["Category"]


def test_exploratory_numeric_pattern_stronger_than_categorical_frequency():
    """Verifies exploratory engine selects numeric sum concentration over categorical frequency when score is higher."""
    df = pd.DataFrame({
        "Category": ["Laptops", "Laptops", "Laptops", "Pens", "Pens", "Pens", "Pencils", "Pencils", "Pencils", "Erasers"],
        "Revenue": [100000.0, 150000.0, 200000.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 2.0],
    })
    query = "What stands out in the data?"
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    call = analyst_out.tool_calls_executed[0]
    assert call["tool"] == "group_data"
    assert call["params"]["group_by"] == ["Category"]
    assert call["params"]["aggregations"] == {"Revenue": ["sum"]}


def test_exploratory_temporal_pattern_strongest():
    """Verifies exploratory engine selects temporal peak candidate when temporal anomaly is highest score."""
    dates = ["2019-01-01"] * 50 + ["2019-02-01"] * 5 + ["2019-03-01"] * 5
    df = pd.DataFrame({
        "Order Date": dates,
        "Category": ["CatA", "CatB"] * 30,
    })
    query = "Tell me something interesting about the orders."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    call = analyst_out.tool_calls_executed[0]
    assert call["tool"] == "group_data"
    assert call["params"]["group_by"] == ["Order Date"]
    assert call["params"]["time_grain"] == "month"


def test_exploratory_candidate_cannot_use_identifier_fields():
    """Verifies identifier columns are never selected as grouping dimensions or metric targets for exploratory candidates."""
    df = pd.DataFrame({
        "Order ID": [f"ORD_{i}" for i in range(50)],
        "Customer_ID": [f"CUST_{i}" for i in range(50)],
        "Category": ["Electronics"] * 40 + ["Furniture"] * 10,
    })
    query = "Tell me something interesting about the orders."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    call = analyst_out.tool_calls_executed[0]
    assert "Order ID" not in call["params"]["group_by"]
    assert "Customer_ID" not in call["params"]["group_by"]
    assert call["params"]["group_by"] == ["Category"]


def test_exploratory_selected_candidate_execution_matches_quantitative_results():
    """Verifies selected exploratory candidate execution is present in tool_calls_executed and quantitatively matches quantitative_results."""
    df = pd.DataFrame({
        "State": ["MH"] * 30 + ["KA"] * 10 + ["DL"] * 10,
        "Order ID": [f"O_{i}" for i in range(50)],
    })
    query = "Tell me something interesting about the orders."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    assert len(analyst_out.tool_calls_executed) >= 1
    call = analyst_out.tool_calls_executed[0]
    assert call["tool"] == "group_data"

    quant_res = analyst_out.quantitative_results["step_1_group_data"]["rows"]
    assert quant_res[0]["State"] == "MH"
    assert quant_res[0]["Order ID_count"] == 30


def test_exploratory_changing_dataset_changes_selected_candidate():
    """Verifies candidate selection changes dynamically when dataset content/schema changes."""
    df1 = pd.DataFrame({
        "City": ["Mumbai"] * 40 + ["Delhi"] * 10,
        "Sales": [10.0] * 50,
    })
    out1 = run_analyst_agent(query="Tell me something interesting", df=df1)
    call1 = out1.tool_calls_executed[0]

    df2 = pd.DataFrame({
        "Category": ["Laptops"] * 45 + ["Phones"] * 5,
        "Sales": [10.0] * 50,
    })
    out2 = run_analyst_agent(query="Tell me something interesting", df=df2)
    call2 = out2.tool_calls_executed[0]

    assert call1["params"]["group_by"] != call2["params"]["group_by"]
    assert call1["params"]["group_by"] == ["City"]
    assert call2["params"]["group_by"] == ["Category"]


def test_exploratory_no_meaningful_candidate_returns_standard_message():
    """Verifies that if all candidate scores are below 0.50, system returns standard no-pattern message and confidence <= 0.50."""
    df = pd.DataFrame({
        "Category": ["A", "B", "C", "D", "E"] * 2,
        "Val": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    })
    query = "Tell me something interesting about the orders."
    analyst_out = run_analyst_agent(query=query, df=df)
    assert analyst_out.findings_summary == "No strong empirical pattern was found in the available data."

    insight_out = run_insight_agent(query=query, df=df, analyst_output=analyst_out)
    assert insight_out.confidence_score <= 0.50
    assert "No strong empirical pattern" in insight_out.executive_summary


def test_exploratory_temporal_pattern_no_state_column():
    """Adversarial Test: Temporal pattern present, but NO State column in dataset."""
    df = pd.DataFrame({
        "Transaction Date": ["2026-01-01"] * 40 + ["2026-02-01"] * 5 + ["2026-03-01"] * 5,
        "Product": ["Laptop"] * 50,
        "Amount": [100.0] * 50,
    })
    query = "Tell me something interesting about this data."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    assert "State" not in analyst_out.findings_summary
    call = analyst_out.tool_calls_executed[0]
    assert call["tool"] == "group_data"
    assert call["params"]["group_by"] == ["Transaction Date"]


def test_exploratory_categorical_concentration_no_order_id():
    """Adversarial Test: Categorical concentration present, but NO Order ID column in dataset."""
    df = pd.DataFrame({
        "Department": ["Engineering"] * 45 + ["Sales"] * 3 + ["HR"] * 2,
        "Budget": [50000.0] * 50,
    })
    query = "Find something interesting in this dataset."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    assert "Order ID" not in analyst_out.findings_summary
    call = analyst_out.tool_calls_executed[0]
    assert call["tool"] == "group_data"
    assert call["params"]["group_by"] == ["Department"]


def test_exploratory_numeric_concentration_no_state():
    """Adversarial Test: Numeric concentration present, but NO State column."""
    df = pd.DataFrame({
        "Segment": ["Enterprise", "Enterprise", "Enterprise", "Retail", "Retail"],
        "Revenue": [500000.0, 400000.0, 300000.0, 1000.0, 500.0],
    })
    query = "What stands out in the data?"
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    assert "State" not in analyst_out.findings_summary
    call = analyst_out.tool_calls_executed[0]
    assert call["tool"] == "group_data"
    assert call["params"]["group_by"] == ["Segment"]
    assert call["params"]["aggregations"] == {"Revenue": ["sum"]}


def test_exploratory_numeric_anomaly_extreme_values():
    """Adversarial Test: Numeric anomaly present (extreme outlier Z-score >= 3.0), no dates/categories."""
    vals = [10.0] * 20 + [10000.0]
    df = pd.DataFrame({"SensorValue": vals})
    query = "Tell me something interesting about the metrics."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    call = analyst_out.tool_calls_executed[0]
    assert call["tool"] == "detect_anomalies"
    assert call["params"]["column"] == "SensorValue"


def test_exploratory_no_datetime_column():
    """Adversarial Test: Dataset with NO datetime column cleanly falls back to categorical or numeric pattern."""
    df = pd.DataFrame({
        "Region": ["North"] * 40 + ["South"] * 10,
        "Score": [90.0] * 50,
    })
    query = "What should I know about this dataset?"
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    call = analyst_out.tool_calls_executed[0]
    assert call["tool"] == "group_data"
    assert call["params"]["group_by"] == ["Region"]


def test_exploratory_no_identifier_column():
    """Adversarial Test: Dataset with NO identifier column (e.g. Order ID, Customer ID)."""
    df = pd.DataFrame({
        "Category": ["A"] * 35 + ["B"] * 15,
        "Units": [5] * 50,
    })
    query = "Find key findings in the dataset."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    call = analyst_out.tool_calls_executed[0]
    assert call["params"]["group_by"] == ["Category"]


def test_exploratory_constants_and_insufficient_columns():
    """Adversarial Test: Dataset with only constant values or insufficient variation returns baseline no-pattern message."""
    df = pd.DataFrame({
        "ConstCat": ["Same"] * 10,
        "ConstVal": [100.0] * 10,
    })
    query = "Tell me something interesting about this data."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.findings_summary == "No strong empirical pattern was found in the available data."
    assert "Order ID" not in analyst_out.findings_summary

    insight_out = run_insight_agent(query=query, df=df, analyst_output=analyst_out)
    assert insight_out.confidence_score <= 0.50


def test_exploratory_competing_candidate_scores_highest_selected():
    """Adversarial Test: Dataset with multiple competing patterns selects candidate with highest valid score."""
    df = pd.DataFrame({
        "Cat": ["TypeA"] * 30 + ["TypeB"] * 20,
        "Revenue": [10000.0] * 30 + [10.0] * 20,
    })
    query = "Tell me something interesting."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    call = analyst_out.tool_calls_executed[0]
    assert call["params"]["aggregations"] == {"Revenue": ["sum"]}


def test_phase19_secondary_drilldown_execution():
    """Phase 19: Verifies secondary segment drill-down is populated in step_2_drilldown when secondary dimension exists."""
    df = pd.DataFrame({
        "State": ["MH"] * 20 + ["KA"] * 10,
        "City": ["Mumbai"] * 15 + ["Pune"] * 5 + ["Bengaluru"] * 10,
        "Sales": [100.0] * 30,
    })
    query = "Which states have the highest sales?"
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    step2 = analyst_out.quantitative_results.get("step_2_drilldown", {})
    assert step2.get("status") == "success"
    assert step2.get("primary_dimension") == "State"
    assert step2.get("primary_segment") == "MH"
    assert step2.get("grouping_dimension") == "City"
    assert len(step2.get("rows", [])) == 2


def test_phase19_secondary_drilldown_failsafe_isolation():
    """Phase 19: Verifies secondary drill-down failure/skip does NOT cause primary analysis failure."""
    df = pd.DataFrame({
        "State": ["MH", "KA"],
        "Sales": [100.0, 200.0],
    })
    query = "Which states have the highest sales?"
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    step2 = analyst_out.quantitative_results.get("step_2_drilldown", {})
    assert step2.get("status") == "skipped"
    assert analyst_out.quantitative_results["step_1_group_data"]["result_row_count"] == 2


def test_phase19_suggested_followups_generation():
    """Phase 19: Verifies Insight Agent generates up to 3 context-aware suggested follow-ups."""
    df = pd.DataFrame({
        "State": ["MH"] * 20 + ["KA"] * 10,
        "City": ["Mumbai"] * 15 + ["Pune"] * 5 + ["Bengaluru"] * 10,
        "Sales": [100.0] * 30,
    })
    query = "Which states have the highest sales?"
    analyst_out = run_analyst_agent(query=query, df=df)
    insight_out = run_insight_agent(query=query, df=df, analyst_output=analyst_out)

    assert len(insight_out.suggested_followups) <= 3
    assert len(insight_out.suggested_followups) > 0
    assert any("City breakdown" in s or "MH" in s for s in insight_out.suggested_followups)


def test_phase19_sandbox_ast_hardening():
    """Phase 19: Verifies AST sandbox security checks reject unsafe code structures."""
    from app.tools.sandbox import validate_code_security, execute_sandboxed_code

    # 1. Reject unsafe imports
    assert len(validate_code_security("import os\nos.system('dir')")) > 0
    assert len(validate_code_security("from subprocess import Popen")) > 0

    # 2. Reject eval/exec/open
    assert len(validate_code_security("eval('1 + 1')")) > 0
    assert len(validate_code_security("open('secret.txt', 'r')")) > 0

    # 3. Reject dunders
    assert len(validate_code_security("x.__class__.__subclasses__()")) > 0

    # 4. Valid analytical code passes
    valid_res = execute_sandboxed_code("res = df['Sales'].sum()", df=pd.DataFrame({"Sales": [10, 20]}))
    assert valid_res["success"] is True
    assert valid_res["result"] == 30


def test_phase19_list_of_orders_exploratory_drilldown():
    """Phase 19: Targeted drill-down validation on List of Orders.csv."""
    csv_path = r"e:\Data Pilot\data\uploads\4bda4e63-290c-479f-b164-599fd863aff8.csv"
    df = pd.read_csv(csv_path)

    query = "Tell me something interesting about the orders."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"

    # Primary finding: 2019-01 = 61 orders
    step1 = analyst_out.quantitative_results.get("step_1_group_data", {})
    rows1 = step1.get("rows", [])
    assert len(rows1) == 1
    assert rows1[0]["Order Date"] == "2019-01"
    assert rows1[0]["Order ID_count"] == 61

    # Secondary drilldown: operates INSIDE 2019-01
    step2 = analyst_out.quantitative_results.get("step_2_drilldown", {})
    assert step2.get("status") == "success"
    assert step2.get("primary_dimension") == "Order Date"
    assert step2.get("primary_segment") == "2019-01"

    sec_dim = step2.get("grouping_dimension")
    assert sec_dim != "Order Date"
    assert sec_dim not in ("Order ID", "CustomerName")  # excluded date and identifier
    assert sec_dim in ("State", "City")

    # Primary result remains unchanged
    assert rows1[0]["Order Date"] == "2019-01"
    assert rows1[0]["Order ID_count"] == 61


def test_phase19_synthetic_no_secondary_dimension_skip():
    """Phase 19: Synthetic dataset with no secondary dimension skips drilldown cleanly without failing primary analysis."""
    df = pd.DataFrame({
        "Order Date": ["2019-01-01"] * 50 + ["2019-02-01"] * 10,
        "Sales": [10.0] * 60,
    })
    query = "Tell me something interesting about the orders."
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    step2 = analyst_out.quantitative_results.get("step_2_drilldown", {})
    assert step2.get("status") == "skipped"
    assert "reason" in step2 and len(step2["reason"]) > 0
    assert len(analyst_out.quantitative_results) >= 1


def test_phase19_exploratory_suggested_followups():
    """Phase 19: Suggested follow-ups for 'Tell me something interesting about the orders.'."""
    csv_path = r"e:\Data Pilot\data\uploads\4bda4e63-290c-479f-b164-599fd863aff8.csv"
    df = pd.read_csv(csv_path)

    query = "Tell me something interesting about the orders."
    analyst_out = run_analyst_agent(query=query, df=df)
    insight_out = run_insight_agent(query=query, df=df, analyst_output=analyst_out)

    suggs = insight_out.suggested_followups
    assert len(suggs) <= 3
    assert len(suggs) > 0
    for s in suggs:
        assert isinstance(s, str) and len(s) > 5









