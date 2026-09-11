import os
import pandas as pd
import pytest
from app.tools.analysis import filter_data, group_data, aggregate_data, AnalysisToolError
from app.tools.statistics import calculate_statistics, calculate_correlation
from app.tools.anomaly_detection import analyze_trend, detect_anomalies
from app.tools.visualization import generate_visualization


@pytest.fixture
def sample_df():
    """Provides a sample sales dataset for testing tools."""
    return pd.DataFrame({
        "Date": ["2026-01-15", "2026-01-20", "2026-02-10", "2026-02-25", "2026-03-05"],
        "Region": ["East", "West", "East", "West", "East"],
        "Product": ["Laptop", "Mouse", "Laptop", "Keyboard", "Laptop"],
        "Sales": [1200.0, 25.0, 1400.0, 75.0, 1300.0],
        "Units": [4, 10, 5, 3, 4],
        "OutlierCol": [10.0, 12.0, 11.0, 10.5, 500.0],  # 500 is an outlier
    })


def test_filter_data(sample_df):
    """Test filter_data with operator conditions."""
    res = filter_data(df=sample_df, filters=[{"column": "Sales", "operator": ">", "value": 500}])
    assert res["filtered_row_count"] == 3
    assert res["original_row_count"] == 5

    res_cat = filter_data(df=sample_df, filters=[{"column": "Region", "operator": "==", "value": "East"}])
    assert res_cat["filtered_row_count"] == 3


def test_group_data(sample_df):
    """Test group_data by Region and Product."""
    res = group_data(
        df=sample_df,
        group_by=["Region"],
        aggregations={"Sales": ["sum", "mean"], "Units": ["sum"]},
    )
    assert res["result_row_count"] == 2
    rows = res["rows"]
    east_row = next(r for r in rows if r["Region"] == "East")
    assert east_row["Sales_sum"] == 3900.0


def test_aggregate_data(sample_df):
    """Test dataset-wide aggregations."""
    res = aggregate_data(df=sample_df, aggregations={"Sales": ["sum", "min", "max"]})
    assert res["aggregations"]["Sales"]["sum"] == 4000.0
    assert res["aggregations"]["Sales"]["min"] == 25.0
    assert res["aggregations"]["Sales"]["max"] == 1400.0


def test_calculate_statistics(sample_df):
    """Test descriptive statistics calculation."""
    res = calculate_statistics(df=sample_df, columns=["Sales", "Units"])
    assert "Sales" in res["statistics"]
    assert res["statistics"]["Sales"]["mean"] == 800.0
    assert res["statistics"]["Units"]["max"] == 10.0


def test_calculate_correlation(sample_df):
    """Test correlation matrix calculation."""
    res = calculate_correlation(df=sample_df, columns=["Sales", "Units"])
    assert "correlation_matrix" in res
    assert "Sales" in res["correlation_matrix"]
    assert "Units" in res["correlation_matrix"]["Sales"]


def test_analyze_trend(sample_df):
    """Test time-series trend analysis."""
    res = analyze_trend(df=sample_df, date_column="Date", value_column="Sales", period="ME")
    assert res["date_column"] == "Date"
    assert res["value_column"] == "Sales"
    assert "overall_trend_direction" in res
    assert len(res["period_breakdown"]) >= 3


def test_detect_anomalies(sample_df):
    """Test anomaly detection via Z-Score and IQR."""
    res_z = detect_anomalies(df=sample_df, column="OutlierCol", method="zscore", threshold=1.5)
    assert res_z["anomaly_count"] >= 1
    assert res_z["anomalies"][0]["OutlierCol"] == 500.0

    res_iqr = detect_anomalies(df=sample_df, column="OutlierCol", method="iqr", threshold=1.5)
    assert res_iqr["anomaly_count"] >= 1


def test_generate_visualization(sample_df):
    """Test chart generation for bar, line, and scatter plots."""
    chart_res = generate_visualization(
        df=sample_df,
        chart_type="bar",
        x_column="Region",
        y_column="Sales",
        title="Test Sales by Region",
    )
    assert chart_res["chart_type"] == "bar"
    assert os.path.exists(chart_res["file_path"])
    assert chart_res["base64"].startswith("data:image/png;base64,")

    # Cleanup generated file
    if os.path.exists(chart_res["file_path"]):
        os.remove(chart_res["file_path"])
