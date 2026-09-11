import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessageChunk

from app.main import app
from app.agents.profiler import run_profiler_agent, DataProfilerAgentError
from app.agents.analyst import (
    run_analyst_agent,
    DataAnalystAgentError,
    validate_analysis_plan,
    determine_requires_visualization,
)
from app.agents.visualization import (
    run_visualization_agent,
    DataVisualizationAgentError,
    validate_and_fix_chart_params,
)
from app.agents.insight import (
    run_insight_agent,
    DataInsightAgentError,
    calculate_confidence_score,
)
from app.agents.supervisor import run_supervisor_agent
from app.agents.graph import build_datapilot_graph, DataPilotState
from app.schemas.agents import (
    SupervisorOutput,
    ProfilerOutput,
    AnalystOutput,
    VisualizationOutput,
    InsightOutput,
    AnalysisPlan,
    AnalysisPlanStep,
)
from app.llm.model import MockChatModel

client = TestClient(app)


# Mock LLM helper classes for edge case testing
class FailingLLM(BaseChatModel):
    """Mock LLM that simulates network or API failures."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("LLM API Service Unavailable")

    @property
    def _llm_type(self) -> str:
        return "failing_mock"

    def with_structured_output(self, schema, **kwargs):
        raise RuntimeError("Structured Output API Error")


class InvalidToolPlannerLLM(BaseChatModel):
    """Mock LLM that proposes invalid/non-existent tools in its analysis plan."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = AIMessageChunk(content="Plan with invalid tools")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "invalid_tool_planner_mock"

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, input, config=None, **kwargs):
        return AnalysisPlan(
            analysis_goal="Malicious execution test",
            steps=[
                AnalysisPlanStep(
                    tool_name="delete_all_tables_tool",
                    params={"table": "*"},
                    purpose="Destructive action"
                ),
                AnalysisPlanStep(
                    tool_name="calculate_statistics",
                    params={"columns": ["Revenue"]},
                    purpose="Valid statistics"
                )
            ],
            recommended_visualization=False
        )


# --- Phase 8: Data Profiler Agent Tests ---

def test_profiler_agent_with_raw_df():
    """Test run_profiler_agent with a raw Pandas DataFrame."""
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, None],
        "City": ["NYC", "LA", "NYC", "LA", "NYC"],
        "Constant": ["A", "A", "A", "A", "A"],
    })

    result = run_profiler_agent(df=df, llm=MockChatModel())
    assert isinstance(result, ProfilerOutput)
    assert result.quality_score < 100.0
    assert len(result.critical_warnings) >= 1
    assert any("Constant" in w for w in result.critical_warnings)
    assert len(result.key_column_observations) >= 2


def test_profiler_agent_with_registered_dataset():
    """Test run_profiler_agent with a registered dataset ID from upload."""
    csv_content = "Product,Price,Quantity\nLaptop,1200,2\nMouse,25,10\nKeyboard,75,4"
    file = ("tech_inventory.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")

    upload_res = client.post("/api/v1/upload", files={"file": file})
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset"]["id"]

    try:
        result = run_profiler_agent(dataset_id=dataset_id, llm=MockChatModel())
        assert isinstance(result, ProfilerOutput)
        assert "tech_inventory.csv" in result.dataset_summary
        assert result.quality_score == 100.0
        assert len(result.critical_warnings) == 0
    finally:
        client.delete(f"/api/v1/datasets/{dataset_id}")


def test_profiler_agent_invalid_dataset_id():
    """Test that run_profiler_agent handles invalid dataset_id gracefully."""
    result = run_profiler_agent(dataset_id="non_existent_id")
    assert isinstance(result, ProfilerOutput)
    assert result.quality_score == 0.0
    assert "Profiling failed" in result.dataset_summary


# --- Phase 9: Data Analyst Agent Tests ---

def test_analyst_agent_top_revenue_query():
    """Test run_analyst_agent executing grouping for top products query."""
    df = pd.DataFrame({
        "Product": ["Laptop", "Mouse", "Laptop", "Keyboard", "Laptop"],
        "Revenue": [1200.0, 25.0, 1400.0, 75.0, 1300.0],
    })

    result = run_analyst_agent(
        query="What are the top products by revenue?",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "success"
    assert "Laptop" in result.findings_summary
    assert result.requires_visualization is True
    assert len(result.tool_calls_planned) >= 1
    assert result.tool_calls_planned[0]["tool"] == "group_data"
    assert len(result.tool_calls_executed) >= 1


def test_analyst_agent_trend_query():
    """Test run_analyst_agent executing trend analysis."""
    df = pd.DataFrame({
        "Date": ["2026-01-01", "2026-02-01", "2026-03-01"],
        "Sales": [1000, 1500, 2200],
    })

    result = run_analyst_agent(
        query="Show monthly revenue trends",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "success"
    assert "trend" in result.findings_summary.lower()
    assert result.requires_visualization is True
    assert result.tool_calls_planned[0]["tool"] == "analyze_trend"


def test_analyst_agent_anomaly_query():
    """Test run_analyst_agent executing anomaly detection."""
    df = pd.DataFrame({
        "ID": [1, 2, 3, 4, 5],
        "Metrics": [10.0, 10.5, 11.0, 10.2, 500.0],
    })

    result = run_analyst_agent(
        query="Find unusual values or anomalies",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "success"
    assert "anomaly" in result.findings_summary.lower()
    assert result.requires_visualization is True
    assert result.tool_calls_planned[0]["tool"] == "detect_anomalies"


def test_analyst_agent_filtering_query():
    """Test run_analyst_agent executing a filtering query."""
    df = pd.DataFrame({
        "Age": [20, 26, 30, 45, 18],
        "Sales": [100, 250, 300, 500, 50],
    })

    result = run_analyst_agent(
        query="Filter customers above 25",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "success"
    assert "Filtered dataset" in result.findings_summary
    assert len(result.tool_calls_executed) >= 1
    assert result.tool_calls_executed[0]["tool"] == "filter_data"


def test_analyst_agent_correlation_query():
    """Test run_analyst_agent executing correlation calculation."""
    df = pd.DataFrame({
        "Age": [25, 30, 35, 40, 45],
        "Income": [50000, 60000, 75000, 90000, 110000],
        "Sales": [200, 300, 450, 600, 800],
    })

    result = run_analyst_agent(
        query="Calculate correlation between numerical columns",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "success"
    assert "correlation" in result.findings_summary.lower()
    assert result.requires_visualization is True


def test_analyst_agent_descriptive_statistics_query():
    """Test run_analyst_agent calculating descriptive statistics."""
    df = pd.DataFrame({
        "Sales": [100.0, 200.0, 300.0, 400.0],
        "Profit": [10.0, 25.0, 40.0, 60.0],
    })

    result = run_analyst_agent(
        query="Calculate descriptive summary statistics",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "success"
    assert "Summary statistics" in result.findings_summary
    assert "step_1_calculate_statistics" in result.quantitative_results


def test_analyst_agent_multi_step_query():
    """Test multi-step analytical query chaining filter and group data."""
    df = pd.DataFrame({
        "Age": [30, 20, 40, 50, 22],
        "Region": ["North", "South", "North", "North", "South"],
        "Sales": [100.0, 200.0, 300.0, 500.0, 150.0],
    })

    # "Which region had the highest average sales for customers above 25?"
    result = run_analyst_agent(
        query="Which region had the highest average sales for customers above 25?",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "success"
    assert len(result.tool_calls_planned) == 2
    assert result.tool_calls_planned[0]["tool"] == "filter_data"
    assert result.tool_calls_planned[1]["tool"] == "group_data"
    assert len(result.tool_calls_executed) == 2
    assert "North" in result.findings_summary


def test_analyst_agent_invalid_column():
    """Test analyst agent handling queries referencing non-existent columns safely."""
    df = pd.DataFrame({
        "Sales": [100, 200, 300],
        "Category": ["A", "B", "C"],
    })

    result = run_analyst_agent(
        query="What is the average of NonExistentColumn?",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status in ("success", "partial_success")
    assert result.errors != [] or result.findings_summary != ""


def test_analyst_agent_empty_result():
    """Test filtering query that returns zero matching rows."""
    df = pd.DataFrame({
        "Age": [18, 20, 22],
        "Sales": [100, 150, 200],
    })

    result = run_analyst_agent(
        query="Filter customers above 100",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "success"
    assert "0 matching rows" in result.findings_summary or "0 rows" in result.findings_summary


def test_analyst_agent_unsupported_query():
    """Test analyst agent handling unsupported query gracefully."""
    df = pd.DataFrame({"Data": [1, 2, 3]})

    result = run_analyst_agent(
        query="Perform quantum telepathic prediction",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "success"
    assert result.findings_summary != ""


def test_analyst_agent_invalid_dataset_id():
    """Test analyst agent returning failure output for non-existent dataset ID."""
    result = run_analyst_agent(
        query="What is total sales?",
        dataset_id="non_existent_dataset_id",
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "failed"
    assert len(result.errors) >= 1
    assert "not found" in result.errors[0].lower()


def test_analyst_agent_empty_dataframe():
    """Test analyst agent returning failure output for empty DataFrame."""
    df = pd.DataFrame()

    result = run_analyst_agent(
        query="What are top products?",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "failed"
    assert len(result.errors) >= 1
    assert "empty" in result.errors[0].lower()


def test_analyst_agent_llm_failure():
    """Test analyst agent falling back gracefully when LLM raises exception."""
    df = pd.DataFrame({
        "Product": ["A", "B", "A"],
        "Revenue": [100, 200, 300],
    })

    result = run_analyst_agent(
        query="What are top products by revenue?",
        df=df,
        llm=FailingLLM(),
    )
    assert isinstance(result, AnalystOutput)
    assert result.execution_status == "success"
    assert "Product" in result.findings_summary or "Revenue" in result.findings_summary


def test_analyst_agent_invalid_tool_requested_by_llm():
    """Test that plan validator strips non-existent tools requested by LLM."""
    df = pd.DataFrame({"Revenue": [100, 200]})

    result = run_analyst_agent(
        query="Run test query",
        df=df,
        llm=InvalidToolPlannerLLM(),
    )
    assert isinstance(result, AnalystOutput)
    assert any("delete_all_tables_tool" in err for err in result.errors)
    assert not any(call.get("tool") == "delete_all_tables_tool" for call in result.tool_calls_executed)
    assert any(call.get("tool") == "calculate_statistics" for call in result.tool_calls_executed)


def test_analyst_agent_planned_vs_executed_tracking():
    """Verify tool_calls_planned and tool_calls_executed are separate lists and tracked."""
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone"],
        "Revenue": [1000, 500],
    })

    result = run_analyst_agent(
        query="What are the top products by revenue?",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result.tool_calls_planned, list)
    assert isinstance(result.tool_calls_executed, list)
    assert len(result.tool_calls_planned) == len(result.tool_calls_executed)
    assert "purpose" in result.tool_calls_planned[0]
    assert "status" in result.tool_calls_executed[0]


def test_analyst_agent_findings_based_on_actual_results():
    """Verify that findings are derived strictly from quantitative tool outputs."""
    df = pd.DataFrame({
        "Product": ["Alpha", "Beta"],
        "Sales": [5000.0, 1000.0],
    })

    result = run_analyst_agent(
        query="Top products by sales",
        df=df,
        llm=MockChatModel(),
    )
    assert "Alpha" in result.findings_summary
    assert "5000" in result.findings_summary or "5000.0" in result.findings_summary
    assert result.quantitative_results != {}


def test_analyst_agent_requires_visualization_behavior():
    """Verify application-level heuristic logic for requires_visualization."""
    df = pd.DataFrame({
        "Date": ["2026-01-01", "2026-02-01"],
        "Sales": [100, 200],
    })

    # Trend query -> True
    res_trend = run_analyst_agent(query="Show sales trend over time", df=df, llm=MockChatModel())
    assert res_trend.requires_visualization is True

    # Total row count query -> False
    res_count = determine_requires_visualization(query="how many total rows", execution_status="success")
    assert res_count is False

    # Failed status -> False
    res_failed = determine_requires_visualization(query="trend sales", execution_status="failed")
    assert res_failed is False


# --- Phase 10: Data Visualization Agent Tests ---

def test_visualization_agent_bar_chart():
    """Test run_visualization_agent generating a bar chart for categorical comparison."""
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Revenue": [1200.0, 800.0, 450.0],
    })

    result = run_visualization_agent(
        query="What are the top products by revenue?",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, VisualizationOutput)
    assert result.is_useful is True
    assert result.chart_type == "bar"
    assert result.x_column == "Product"
    assert result.y_column == "Revenue"
    assert result.chart_file_path is not None
    assert result.base64_image is not None
    assert result.base64_image.startswith("data:image/png;base64,")


def test_visualization_agent_line_chart():
    """Test run_visualization_agent generating a line chart for time-series trends."""
    df = pd.DataFrame({
        "Date": ["2026-01-01", "2026-02-01", "2026-03-01"],
        "Sales": [1000, 1500, 2200],
    })

    result = run_visualization_agent(
        query="Show monthly revenue trends over time",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, VisualizationOutput)
    assert result.is_useful is True
    assert result.chart_type == "line"
    assert result.x_column == "Date"
    assert result.y_column == "Sales"
    assert result.chart_file_path is not None


def test_visualization_agent_scatter_chart():
    """Test run_visualization_agent generating a scatter plot for numerical correlation."""
    df = pd.DataFrame({
        "Height": [160, 170, 180, 190],
        "Weight": [60, 70, 80, 90],
    })

    result = run_visualization_agent(
        query="Calculate scatter correlation between height and weight",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, VisualizationOutput)
    assert result.is_useful is True
    assert result.chart_type == "scatter"
    assert result.x_column == "Height"
    assert result.y_column == "Weight"


def test_visualization_agent_histogram_chart():
    """Test run_visualization_agent generating a histogram for distribution analysis."""
    df = pd.DataFrame({
        "Age": [20, 22, 25, 27, 30, 32, 35, 40, 45, 50],
    })

    result = run_visualization_agent(
        query="Show age distribution frequency histogram",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, VisualizationOutput)
    assert result.is_useful is True
    assert result.chart_type == "histogram"
    assert result.x_column == "Age"


def test_visualization_agent_boxplot_chart():
    """Test run_visualization_agent generating a boxplot for outlier inspection."""
    df = pd.DataFrame({
        "Metric": [10, 12, 11, 13, 500],
        "Group": ["A", "A", "B", "B", "B"],
    })

    result = run_visualization_agent(
        query="Show boxplot to inspect anomalies and outliers",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, VisualizationOutput)
    assert result.is_useful is True
    assert result.chart_type == "boxplot"
    assert result.x_column == "Metric"


def test_visualization_agent_pie_chart():
    """Test run_visualization_agent generating a pie chart for proportional shares."""
    df = pd.DataFrame({
        "Region": ["North", "South", "East"],
        "Sales": [500, 300, 200],
    })

    result = run_visualization_agent(
        query="Show pie chart of proportional sales share",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, VisualizationOutput)
    assert result.is_useful is True
    assert result.chart_type == "pie"
    assert result.x_column == "Region"


def test_visualization_agent_not_useful():
    """Test run_visualization_agent identifying queries where charts are not useful."""
    df = pd.DataFrame({"Age": [25, 30]})

    result = run_visualization_agent(
        query="how many total rows",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, VisualizationOutput)
    assert result.is_useful is False
    assert result.chart_file_path is None


def test_visualization_agent_invalid_dataset_id():
    """Test run_visualization_agent handling non-existent dataset ID safely."""
    result = run_visualization_agent(
        query="Show sales trend",
        dataset_id="invalid_dataset_id",
        llm=MockChatModel(),
    )
    assert isinstance(result, VisualizationOutput)
    assert result.is_useful is False
    assert "Dataset not found" in result.reasoning


def test_visualization_agent_empty_dataframe():
    """Test run_visualization_agent handling empty DataFrame safely."""
    df = pd.DataFrame()

    result = run_visualization_agent(
        query="Show sales trend",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, VisualizationOutput)
    assert result.is_useful is False
    assert "empty" in result.reasoning.lower()


def test_visualization_agent_invalid_column_repair():
    """Test auto-repair helper fixing non-existent column names against DataFrame schema."""
    df = pd.DataFrame({
        "ValidX": ["A", "B"],
        "ValidY": [10, 20],
    })

    c_type, x_col, y_col = validate_and_fix_chart_params(
        chart_type="line",
        x_column="FakeColumnX",
        y_column="FakeColumnY",
        target_df=df,
    )
    assert c_type == "line"
    assert x_col == "ValidX"
    assert y_col == "ValidY"


def test_visualization_agent_with_analyst_output_context():
    """Test run_visualization_agent utilizing AnalystOutput prior context."""
    df = pd.DataFrame({
        "Product": ["A", "B"],
        "Revenue": [100, 200],
    })
    analyst_out = AnalystOutput(
        analysis_goal="Grouping analysis",
        findings_summary="Top product is B with 200",
        requires_visualization=True,
    )

    result = run_visualization_agent(
        query="Top products by revenue",
        df=df,
        analyst_output=analyst_out,
        llm=MockChatModel(),
    )
    assert isinstance(result, VisualizationOutput)
    assert result.is_useful is True
    assert result.chart_file_path is not None


def test_visualization_agent_llm_failure():
    """Test run_visualization_agent falling back to heuristic engine when LLM fails."""
    df = pd.DataFrame({
        "Date": ["2026-01-01", "2026-02-01"],
        "Sales": [100, 200],
    })

    result = run_visualization_agent(
        query="Show sales trend over time",
        df=df,
        llm=FailingLLM(),
    )
    assert isinstance(result, VisualizationOutput)
    assert result.is_useful is True
    assert result.chart_type == "line"
    assert result.chart_file_path is not None


# --- Phase 11: Executive Insight Agent Tests ---

def test_insight_agent_with_full_context():
    """Test run_insight_agent with prior analyst and visualization outputs."""
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Tablet"],
        "Revenue": [1200.0, 800.0, 450.0],
    })

    analyst_out = AnalystOutput(
        analysis_goal="Grouping top products",
        tool_calls_executed=[{"tool": "group_data", "status": "success"}],
        quantitative_results={"step_1_group_data": {"group_by": ["Product"], "rows": [{"Product": "Laptop", "Revenue_mean": 1200.0}]}},
        findings_summary="Top product by revenue is Laptop with 1200.0.",
        requires_visualization=True,
    )
    vis_out = VisualizationOutput(
        is_useful=True,
        chart_type="bar",
        title="Revenue by Product",
        reasoning="Bar chart rendered.",
    )

    result = run_insight_agent(
        query="Provide executive insights on top products",
        df=df,
        analyst_output=analyst_out,
        vis_output=vis_out,
        llm=MockChatModel(),
    )
    assert isinstance(result, InsightOutput)
    assert "Laptop" in result.executive_summary or "top products" in result.executive_summary.lower()
    assert len(result.key_insights) >= 1
    assert len(result.actionable_recommendations) >= 1
    assert result.confidence_score == 1.0


def test_insight_agent_raw_df_only():
    """Test run_insight_agent with raw DataFrame only."""
    df = pd.DataFrame({
        "Sales": [100, 200, 300],
        "Region": ["North", "South", "North"],
    })

    result = run_insight_agent(
        query="Executive summary of regional sales",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, InsightOutput)
    assert result.executive_summary != ""
    assert len(result.key_insights) >= 1
    assert len(result.actionable_recommendations) >= 1
    assert result.confidence_score > 0.0


def test_insight_agent_invalid_dataset_id():
    """Test run_insight_agent handling invalid dataset ID safely."""
    result = run_insight_agent(
        query="Executive insights query",
        dataset_id="non_existent_dataset_id",
        llm=MockChatModel(),
    )
    assert isinstance(result, InsightOutput)
    assert result.confidence_score == 0.0
    assert "cannot be performed" in result.executive_summary.lower()


def test_insight_agent_empty_dataframe():
    """Test run_insight_agent handling empty DataFrame safely."""
    df = pd.DataFrame()

    result = run_insight_agent(
        query="Executive insights query",
        df=df,
        llm=MockChatModel(),
    )
    assert isinstance(result, InsightOutput)
    assert result.confidence_score == 0.0
    assert "empty" in result.executive_summary.lower()


def test_insight_agent_llm_failure_fallback():
    """Test run_insight_agent falling back to empirical engine when LLM fails."""
    df = pd.DataFrame({
        "Sales": [100, 200],
        "Category": ["A", "B"],
    })

    result = run_insight_agent(
        query="Executive summary for categories",
        df=df,
        llm=FailingLLM(),
    )
    assert isinstance(result, InsightOutput)
    assert result.executive_summary != ""
    assert len(result.key_insights) >= 1
    assert result.confidence_score > 0.0


def test_insight_agent_confidence_score_calculation():
    """Verify confidence score calculations under perfect and partial execution states."""
    df = pd.DataFrame({"Metric": [10, 20, 30]})

    # Perfect state
    score_perfect = calculate_confidence_score(df)
    assert score_perfect == 1.0

    # Partial execution penalty
    analyst_partial = AnalystOutput(
        analysis_goal="Partial goal",
        findings_summary="Partial findings",
        execution_status="partial_success",
        errors=["Step 2 failed"],
    )
    score_partial = calculate_confidence_score(df, analyst_partial)
    assert score_partial < 1.0


# --- Phase 12: Supervisor & LangGraph Workflow Tests ---

def test_supervisor_agent_routing_profiler():
    """Test supervisor agent routing to profiler node."""
    res = run_supervisor_agent(query="Audit missing values and data quality", llm=MockChatModel())
    assert isinstance(res, SupervisorOutput)
    assert res.next_agent == "profiler"


def test_supervisor_agent_routing_analyst():
    """Test supervisor agent routing to analyst node."""
    res = run_supervisor_agent(query="What are top products by revenue?", llm=MockChatModel())
    assert isinstance(res, SupervisorOutput)
    assert res.next_agent == "analyst"


def test_supervisor_agent_routing_visualization():
    """Test supervisor agent routing directly to visualization node."""
    res = run_supervisor_agent(query="Plot a bar chart of product sales", llm=MockChatModel())
    assert isinstance(res, SupervisorOutput)
    assert res.next_agent == "visualization"


def test_supervisor_agent_routing_insight():
    """Test supervisor agent routing to insight node."""
    res = run_supervisor_agent(query="Provide executive strategic recommendations", llm=MockChatModel())
    assert isinstance(res, SupervisorOutput)
    assert res.next_agent == "insight"


def test_langgraph_workflow_profiler_execution():
    """Test LangGraph StateGraph compiled workflow executing profiler route."""
    graph = build_datapilot_graph()
    df = pd.DataFrame({
        "Age": [20, 30, None],
        "City": ["NYC", "LA", "NYC"],
    })

    initial_state = {
        "query": "Audit missing values and data quality",
        "df": df,
        "llm": MockChatModel(),
    }

    final_state = graph.invoke(initial_state)
    assert "supervisor_output" in final_state
    assert final_state["supervisor_output"].next_agent == "profiler"
    assert "profiler_output" in final_state
    assert final_state["profiler_output"].quality_score < 100.0
    assert len(final_state.get("history", [])) >= 2


def test_langgraph_workflow_end_to_end_analysis():
    """Test LangGraph StateGraph compiled workflow executing full multi-agent analysis loop."""
    graph = build_datapilot_graph()
    df = pd.DataFrame({
        "Product": ["Laptop", "Phone", "Laptop", "Keyboard"],
        "Revenue": [1200.0, 800.0, 1400.0, 75.0],
        "Date": ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15"],
    })

    initial_state = {
        "query": "What are the top products by revenue?",
        "df": df,
        "llm": MockChatModel(),
    }

    final_state = graph.invoke(initial_state)
    assert "supervisor_output" in final_state
    assert final_state["supervisor_output"].next_agent == "analyst"
    assert "analyst_output" in final_state
    assert final_state["analyst_output"].execution_status == "success"
    assert "vis_output" in final_state
    assert final_state["vis_output"].is_useful is True
    assert "insight_output" in final_state
    assert final_state["insight_output"].confidence_score > 0.0
    assert len(final_state.get("history", [])) >= 4



