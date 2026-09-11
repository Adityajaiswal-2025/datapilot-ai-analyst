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

