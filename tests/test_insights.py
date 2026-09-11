import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.data.metadata import register_dataset, extract_metadata, DATASET_REGISTRY
from app.tools.hypothesis import (
    benjamini_hochberg_fdr,
    test_normality,
    test_numeric_difference,
    test_categorical_association,
    test_correlation_significance,
    generate_automated_hypotheses,
)
from app.tools.insights_engine import discover_automated_insights
from app.schemas.insights import HypothesisResult, AutomatedInsight

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_clean_registry():
    """Ensure clean registry before each test."""
    DATASET_REGISTRY.clear()
    yield
    DATASET_REGISTRY.clear()


def test_benjamini_hochberg_fdr():
    """Verifies BH FDR adjustment controls false discovery rates properly."""
    raw_p = [0.001, 0.008, 0.039, 0.041, 0.045, 0.15, 0.20, 0.50]
    adj_p = benjamini_hochberg_fdr(raw_p)

    assert len(adj_p) == len(raw_p)
    # Adjusted p-values must be monotonically non-decreasing when ordered
    assert all(adj_p[i] <= adj_p[i + 1] for i in range(len(adj_p) - 1))
    # Adjusted p-value should be >= raw p-value
    for orig, adj in zip(raw_p, adj_p):
        assert adj >= orig
        assert 0.0 <= adj <= 1.0


def test_t_test_two_groups():
    """Verifies Welch's T-test selection and Cohen's d effect size for 2 normal groups."""
    np.random.seed(42)
    group_a = np.random.normal(loc=100.0, scale=10.0, size=50)
    group_b = np.random.normal(loc=120.0, scale=10.0, size=50)

    df = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "value": np.concatenate([group_a, group_b]),
    })

    res = test_numeric_difference(df, group_col="group", value_col="value")
    assert res["test_name"] == "Independent Two-Sample T-Test (Welch's)"
    assert res["p_value"] < 0.01
    assert res["effect_size_type"] == "Cohen's d"
    assert res["effect_size"] > 0.8  # Large effect
    assert res["statistical_significance"] is True


def test_anova_three_groups():
    """Verifies One-Way ANOVA selection and Eta-squared for 3+ groups."""
    np.random.seed(42)
    g1 = np.random.normal(loc=50, scale=5, size=40)
    g2 = np.random.normal(loc=65, scale=5, size=40)
    g3 = np.random.normal(loc=80, scale=5, size=40)

    df = pd.DataFrame({
        "category": ["Low"] * 40 + ["Medium"] * 40 + ["High"] * 40,
        "score": np.concatenate([g1, g2, g3]),
    })

    res = test_numeric_difference(df, group_col="category", value_col="score")
    assert res["test_name"] == "One-Way ANOVA"
    assert res["p_value"] < 0.001
    assert res["effect_size_type"] == "Eta-Squared (η²)"
    assert res["effect_size"] > 0.14  # Large effect
    assert res["statistical_significance"] is True


def test_chi_square_categorical():
    """Verifies Chi-Square test and Cramér's V effect size."""
    # Strong association: Dept A mostly Senior, Dept B mostly Junior
    df = pd.DataFrame({
        "department": ["Eng"] * 50 + ["Sales"] * 50,
        "seniority": ["Senior"] * 45 + ["Junior"] * 5 + ["Senior"] * 5 + ["Junior"] * 45,
    })

    res = test_categorical_association(df, col1="department", col2="seniority")
    assert res["test_name"] == "Chi-Square Test of Independence"
    assert res["p_value"] < 0.001
    assert res["effect_size_type"] == "Cramér's V"
    assert res["effect_size"] > 0.5  # Large effect
    assert res["statistical_significance"] is True


def test_pearson_spearman_correlation():
    """Verifies Pearson linear and Spearman rank correlation with exact p-value."""
    np.random.seed(42)
    x = np.linspace(1, 100, 100)
    y = 2.5 * x + np.random.normal(0, 10, 100)

    df = pd.DataFrame({"x": x, "y": y})

    res_pearson = test_correlation_significance(df, col1="x", col2="y", method="pearson")
    assert res_pearson["test_name"] == "Pearson Linear Correlation"
    assert res_pearson["statistic"] > 0.9
    assert res_pearson["p_value"] < 0.001
    assert "does not establish causation" in res_pearson["statement"]

    res_spearman = test_correlation_significance(df, col1="x", col2="y", method="spearman")
    assert res_spearman["test_name"] == "Spearman Rank Correlation"
    assert res_spearman["statistic"] > 0.9


def test_normality_check():
    """Verifies Shapiro-Wilk (<5000) and D'Agostino-Pearson (>5000) normality tests."""
    np.random.seed(42)
    small_norm = pd.Series(np.random.normal(0, 1, 100))
    res_small = test_normality(small_norm)
    assert res_small["test_name"] == "Shapiro-Wilk Normality Test"
    assert res_small["is_normal"] is True

    large_norm = pd.Series(np.random.normal(0, 1, 6000))
    res_large = test_normality(large_norm)
    assert res_large["test_name"] == "D'Agostino-Pearson Normality Test"
    assert res_large["is_normal"] is True


def test_computational_safety_and_exclusions():
    """Verifies ID columns, constant columns, and high-cardinality exclusions."""
    df = pd.DataFrame({
        "user_id": [f"usr_{i}" for i in range(100)],
        "constant_col": [42] * 100,
        "high_card_code": [f"code_{i}" for i in range(100)],
        "valid_group": ["Alpha"] * 50 + ["Beta"] * 50,
        "valid_metric": np.concatenate([np.random.normal(10, 2, 50), np.random.normal(20, 2, 50)]),
    })

    res = generate_automated_hypotheses(df=df, max_pairs=50)
    tested_cols = [h["target_column"] for h in res["hypotheses"]] + [h["group_column"] for h in res["hypotheses"] if h["group_column"]]

    assert "user_id" not in tested_cols
    assert "constant_col" not in tested_cols
    assert "high_card_code" not in tested_cols
    assert "valid_metric" in tested_cols
    assert "valid_group" in tested_cols


def test_associative_non_causal_phrasing():
    """Verifies that generated insights enforce non-causal language."""
    df = pd.DataFrame({
        "feature_a": np.arange(50),
        "feature_b": np.arange(50) * 3 + np.random.normal(0, 1, 50),
    })

    insights_res = discover_automated_insights(df=df)
    for insight in insights_res["insights"]:
        explanation = insight["explanation"]
        assert "cause" not in explanation.lower() or "does not establish causation" in explanation.lower() or "non-causal" in explanation.lower() or "not imply causality" in explanation.lower() or "associat" in explanation.lower()


def test_api_auto_insights(setup_clean_registry):
    """Tests POST /api/v1/insights/auto endpoint."""
    df = pd.DataFrame({
        "region": ["North"] * 30 + ["South"] * 30,
        "revenue": np.concatenate([np.random.normal(500, 50, 30), np.random.normal(800, 50, 30)]),
        "satisfaction": np.concatenate([np.random.normal(3.5, 0.2, 30), np.random.normal(4.5, 0.2, 30)]),
    })
    meta = extract_metadata("insights_ds", "insights.csv", "insights.csv", df)
    register_dataset(meta, df)

    resp = client.post("/api/v1/insights/auto", json={"dataset_id": "insights_ds", "max_hypotheses": 10})
    assert resp.status_code == 200
    data = resp.json()

    assert data["success"] is True
    assert data["dataset_id"] == "insights_ds"
    assert len(data["hypotheses"]) > 0
    assert len(data["insights"]) > 0
    assert "adjusted_p_value" in data["hypotheses"][0]


def test_api_targeted_hypothesis(setup_clean_registry):
    """Tests POST /api/v1/insights/hypothesis endpoint."""
    df = pd.DataFrame({
        "tier": ["Gold"] * 40 + ["Silver"] * 40,
        "spend": np.concatenate([np.random.normal(1000, 100, 40), np.random.normal(500, 50, 40)]),
    })
    meta = extract_metadata("hyp_ds", "hyp.csv", "hyp.csv", df)
    register_dataset(meta, df)

    resp = client.post(
        "/api/v1/insights/hypothesis",
        json={
            "dataset_id": "hyp_ds",
            "test_type": "numeric_difference",
            "primary_column": "spend",
            "secondary_column": "tier",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["success"] is True
    assert data["hypothesis_result"] is not None
    assert data["hypothesis_result"]["test_name"] == "Independent Two-Sample T-Test (Welch's)"
    assert data["hypothesis_result"]["effect_size_type"] == "Cohen's d"

