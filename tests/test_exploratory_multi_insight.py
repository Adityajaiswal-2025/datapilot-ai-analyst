import pytest
import pandas as pd
import numpy as np
from app.agents.analyst import run_analyst_agent, resolve_query_intent, select_best_exploratory_candidates
from app.agents.insight import run_insight_agent, calculate_confidence_score


@pytest.fixture
def ecommerce_df():
    """Sample e-commerce dataset containing Order Date, State, Sales, Order ID."""
    dates = pd.date_range(start="2019-01-01", periods=100, freq="D")
    states = ["Madhya Pradesh"] * 40 + ["Maharashtra"] * 30 + ["Rajasthan"] * 20 + ["Punjab"] * 10
    sales = [150.0] * 99 + [5000.0]  # includes outlier anomaly
    order_ids = [f"ORD-{i+1:04d}" for i in range(100)]
    return pd.DataFrame({
        "Order Date": dates,
        "State": states,
        "Sales": sales,
        "Order ID": order_ids,
    })


@pytest.fixture
def synthetic_df():
    """Synthetic dataset with completely different column names (no State, City, Order ID)."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=120, freq="D")
    regions = ["North"] * 60 + ["South"] * 35 + ["West"] * 25
    revenue = np.random.normal(500, 50, 120)
    revenue[10] = 9500.0  # outlier anomaly
    return pd.DataFrame({
        "TxDate": dates,
        "Region": regions,
        "Revenue": revenue,
    })


def test_exploratory_multi_insight_production_query(ecommerce_df):
    """Test production query: 'Analyze this dataset and tell me the 3 most important insights you can find, with supporting numbers'."""
    query = "Analyze this dataset and tell me the 3 most important insights you can find, with supporting numbers"

    # 1. Intent check
    intent = resolve_query_intent(query, ecommerce_df)
    assert intent["is_exploratory"] is True, "Query should resolve to exploratory intent."

    # 2. Candidate discovery
    steps, score, findings, scores = select_best_exploratory_candidates(ecommerce_df, max_candidates=3)
    assert len(steps) >= 2, f"Should discover at least 2 strong candidates, got {len(steps)}"
    assert score > 0.50, f"Expected candidate score > 0.50, got {score}"

    # 3. Full Analyst Execution
    analyst_out = run_analyst_agent(query=query, df=ecommerce_df)
    assert analyst_out.execution_status == "success"
    assert "Aggregations: 'Order ID': count=100" not in analyst_out.findings_summary
    assert "Aggregations: Order ID count=500" not in analyst_out.findings_summary
    assert "1 total groups" not in analyst_out.findings_summary
    assert "Ranked by 'Order ID_count'" not in analyst_out.findings_summary
    assert any(char.isdigit() for char in analyst_out.findings_summary), "Findings must contain empirical numbers."

    # User-facing findings vs Technical details check
    parts = analyst_out.findings_summary.split("Execution warnings:")
    user_facing_findings = parts[0].strip()
    tech_details = parts[1].strip() if len(parts) > 1 else ""

    assert "(Note:" not in user_facing_findings, "Internal note must not appear in user-facing findings."
    assert "meeting significance threshold" not in user_facing_findings, "Threshold wording must not appear in user-facing findings."
    assert "recorded the highest" in user_facing_findings, "Temporal peak finding must use natural formatting."
    if tech_details:
        assert "meeting significance threshold" in tech_details or "Candidate note:" in tech_details, "Technical threshold note must be available under execution details."

    # 4. Insight Agent Execution & Confidence Score
    insight_out = run_insight_agent(query=query, df=ecommerce_df, analyst_output=analyst_out)
    assert insight_out.confidence_score == score, f"Confidence score should match candidate score {score}, got {insight_out.confidence_score}"
    assert insight_out.confidence_score < 1.0, "Confidence score should not be defaulted to 100% (1.0)."
    assert len(insight_out.key_insights) >= 2, "Should return at least 2 empirical key insights."
    assert "1 total groups" not in insight_out.executive_summary
    assert "Ranked by 'Order ID_count'" not in insight_out.executive_summary
    assert "(Note:" not in insight_out.executive_summary
    assert "meeting significance threshold" not in insight_out.executive_summary
    assert not any("(Note:" in k for k in insight_out.key_insights)
    assert not any("meeting significance threshold" in k for k in insight_out.key_insights)
    assert "strong empirical pattern" in insight_out.executive_summary.lower()


@pytest.mark.parametrize("query", [
    "What are the 3 most important insights in this dataset?",
    "Find 3 interesting patterns in this data.",
    "What stands out in this dataset?",
    "Tell me something interesting about the data.",
])
def test_exploratory_multi_insight_phrasings(ecommerce_df, query):
    """Test equivalent open-ended exploratory query phrasings."""
    intent = resolve_query_intent(query, ecommerce_df)
    assert intent["is_exploratory"] is True

    analyst_out = run_analyst_agent(query=query, df=ecommerce_df)
    assert analyst_out.execution_status == "success"
    assert "Aggregations: 'Order ID': count=100" not in analyst_out.findings_summary

    insight_out = run_insight_agent(query=query, df=ecommerce_df, analyst_output=analyst_out)
    assert insight_out.confidence_score > 0.45
    assert len(insight_out.key_insights) >= 1


def test_exploratory_multi_insight_synthetic_dataset(synthetic_df):
    """Test multi-candidate discovery on synthetic dataset with non-ecommerce column names."""
    query = "Analyze this dataset and tell me the 3 most important insights you can find, with supporting numbers"

    steps, score, findings, scores = select_best_exploratory_candidates(synthetic_df, max_candidates=3)
    assert len(steps) >= 2, "Should discover candidates on synthetic dataset."

    analyst_out = run_analyst_agent(query=query, df=synthetic_df)
    assert analyst_out.execution_status == "success"
    assert "Region" in analyst_out.findings_summary or "Revenue" in analyst_out.findings_summary or "TxDate" in analyst_out.findings_summary

    insight_out = run_insight_agent(query=query, df=synthetic_df, analyst_output=analyst_out)
    assert insight_out.confidence_score > 0.45


def test_explicit_analytical_queries_preservation(ecommerce_df):
    """Verify that explicit analytical queries continue to route to specialized paths and NOT exploratory mode."""

    # 1. Categorical Ranking
    q_rank = "Which states have the highest orders?"
    intent_rank = resolve_query_intent(q_rank, ecommerce_df)
    assert intent_rank["is_exploratory"] is False
    assert intent_rank["dimension"] == "State"
    assert intent_rank["is_ranking"] is True

    analyst_rank = run_analyst_agent(query=q_rank, df=ecommerce_df)
    assert "Madhya Pradesh" in analyst_rank.findings_summary

    # 2. Temporal Breakdown
    q_temp = "How many orders were placed each month?"
    intent_temp = resolve_query_intent(q_temp, ecommerce_df)
    assert intent_temp["is_exploratory"] is False
    assert intent_temp["time_grain"] == "month"

    analyst_temp = run_analyst_agent(query=q_temp, df=ecommerce_df)
    assert "Temporal breakdown" in analyst_temp.findings_summary or "2019-01" in analyst_temp.findings_summary

    # 3. Entity Count
    q_count = "What is the total number of orders?"
    intent_count = resolve_query_intent(q_count, ecommerce_df)
    assert intent_count["is_exploratory"] is False


def test_threshold_note_in_technical_details_only():
    """Verify that when <3 candidates are found, candidate note is placed under Execution warnings / technical details only."""
    # Simple dataset with single temporal date column and uniform values (1 candidate)
    dates = pd.date_range("2019-01-01", periods=60, freq="D")
    df = pd.DataFrame({"Order Date": dates, "Order ID": [f"ORD-{i}" for i in range(60)]})

    query = "Analyze this dataset and tell me the 3 most important insights you can find, with supporting numbers"
    analyst_out = run_analyst_agent(query=query, df=df)

    assert analyst_out.execution_status == "success"
    parts = analyst_out.findings_summary.split("Execution warnings:")
    user_facing = parts[0].strip()
    assert len(parts) > 1, "Execution warnings section must exist when <3 candidates found."
    tech_details = parts[1].strip()

    assert "(Note:" not in user_facing
    assert "meeting significance threshold" not in user_facing
    assert "Candidate note:" in tech_details
    assert "meeting significance threshold" in tech_details

    insight_out = run_insight_agent(query=query, df=df, analyst_output=analyst_out)
    assert "(Note:" not in insight_out.executive_summary
    assert "meeting significance threshold" not in insight_out.executive_summary
    assert not any("(Note:" in k for k in insight_out.key_insights)
    assert not any("meeting significance threshold" in k for k in insight_out.key_insights)


def test_exploratory_1_strong_insight_returns_1():
    """Verify that when only 1 strong insight exists, exactly 1 insight is returned without fabrication."""
    dates = pd.date_range("2019-01-01", periods=60, freq="D")
    df = pd.DataFrame({"Order Date": dates, "Order ID": [f"ORD-{i}" for i in range(60)]})

    query = "Analyze this dataset and tell me the 3 most important insights you can find, with supporting numbers"
    analyst_out = run_analyst_agent(query=query, df=df)
    insight_out = run_insight_agent(query=query, df=df, analyst_output=analyst_out)

    assert len(insight_out.key_insights) == 1, f"Expected 1 key insight, got {len(insight_out.key_insights)}"
    assert "1 strong empirical pattern was identified; no additional findings were included without sufficient supporting evidence." in insight_out.executive_summary
    assert not any("Dataset schema contains" in k for k in insight_out.key_insights), "No generic schema fallback should be added to reach requested count."


def test_exploratory_2_strong_insights_returns_2():
    """Verify that when 2 strong insights exist, exactly 2 insights are returned without fabrication."""
    dates = pd.date_range("2019-01-01", periods=100, freq="D")
    states = ["Madhya Pradesh"] * 70 + ["Maharashtra"] * 30
    categories = ["Electronics"] * 90 + ["Furniture"] * 10
    order_ids = [f"ORD-{i+1:04d}" for i in range(100)]
    df = pd.DataFrame({"Order Date": dates, "State": states, "Category": categories, "Order ID": order_ids})

    query = "Analyze this dataset and tell me the 3 most important insights you can find, with supporting numbers"
    analyst_out = run_analyst_agent(query=query, df=df)
    insight_out = run_insight_agent(query=query, df=df, analyst_output=analyst_out)

    assert len(insight_out.key_insights) == 2, f"Expected 2 key insights, got {len(insight_out.key_insights)}"
    assert "2 strong empirical patterns were identified; no additional findings were included without sufficient supporting evidence." in insight_out.executive_summary
    assert not any("Dataset schema contains" in k for k in insight_out.key_insights)


def test_exploratory_3_plus_insights_returns_max_3(ecommerce_df):
    """Verify that when 3+ strong insights exist, maximum 3 insights are returned."""
    query = "Analyze this dataset and tell me the 3 most important insights you can find, with supporting numbers"
    analyst_out = run_analyst_agent(query=query, df=ecommerce_df)
    insight_out = run_insight_agent(query=query, df=ecommerce_df, analyst_output=analyst_out)

    assert len(insight_out.key_insights) == 3, f"Expected 3 key insights, got {len(insight_out.key_insights)}"
    assert "3 strong empirical patterns were identified in the dataset." in insight_out.executive_summary


def test_exploratory_0_strong_insights_returns_honest_no_pattern():
    """Verify that when 0 strong insights exist, an honest no-pattern summary is returned without fabrication."""
    # Dataset with uniform non-significant numerical data and 2 columns
    df = pd.DataFrame({
        "MetricA": [10.0] * 50,
        "MetricB": [100.0] * 50,
    })

    query = "Analyze this dataset and tell me the 3 most important insights you can find, with supporting numbers"
    analyst_out = run_analyst_agent(query=query, df=df)
    insight_out = run_insight_agent(query=query, df=df, analyst_output=analyst_out)

    assert analyst_out.execution_status == "success"
    assert "No strong empirical pattern" in analyst_out.findings_summary
    assert "No strong empirical patterns were identified in the dataset." in insight_out.executive_summary
    assert insight_out.key_insights == ["No strong empirical patterns met the significance threshold in the dataset."]


