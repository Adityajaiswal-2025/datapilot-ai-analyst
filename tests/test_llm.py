import pytest
from pydantic import ValidationError
from langchain_core.messages import HumanMessage
from app.llm.model import get_llm_model, MockChatModel
from app.core.config import settings
from app.schemas.agents import (
    SupervisorOutput,
    ProfilerOutput,
    AnalystOutput,
    VisualizationOutput,
    InsightOutput,
)
from app.llm.prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    DATA_PROFILER_SYSTEM_PROMPT,
    DATA_ANALYST_SYSTEM_PROMPT,
    VISUALIZATION_SYSTEM_PROMPT,
    INSIGHT_GENERATOR_SYSTEM_PROMPT,
)


def test_mock_chat_model():
    """Test MockChatModel returns AIMessage without live API keys."""
    llm = MockChatModel(mock_response="Test Mock Output")
    res = llm.invoke([HumanMessage(content="Hello DataPilot")])
    assert res.content == "Test Mock Output"


def test_provider_selection_mock():
    """Test provider selection for 'mock'."""
    llm = get_llm_model(provider="mock")
    assert isinstance(llm, MockChatModel)


def test_missing_api_keys_raises_value_error(monkeypatch):
    """Test that requesting real providers (openai, google, anthropic) without API keys raises ValueError."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        get_llm_model(provider="openai")

    with pytest.raises(ValueError, match="GOOGLE_API_KEY is not set"):
        get_llm_model(provider="google")

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        get_llm_model(provider="anthropic")


def test_invalid_provider_handling():
    """Test that requesting an unsupported provider raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported LLM provider 'invalid_provider'"):
        get_llm_model(provider="invalid_provider")


def test_agent_structured_output_schemas():
    """Test instantiating Pydantic structured output models for all 5 agents."""
    sup = SupervisorOutput(
        intent_summary="User wants revenue by region",
        next_agent="analyst",
        plan_steps=["Group dataset by Region", "Sum Revenue"],
        reasoning="Simple grouping and aggregation required",
    )
    assert sup.next_agent == "analyst"

    prof = ProfilerOutput(
        dataset_summary="100 rows x 5 columns dataset",
        quality_score=95.0,
        critical_warnings=["Column 'Age' has 5 missing values"],
        key_column_observations=["Revenue mean is 500.0"],
    )
    assert prof.quality_score == 95.0

    ana = AnalystOutput(
        analysis_goal="Calculate revenue by region",
        tool_calls_planned=[{"tool": "group_data", "params": {"group_by": ["Region"]}}],
        findings_summary="East region generated highest revenue ($50,000)",
        requires_visualization=True,
    )
    assert ana.requires_visualization is True

    vis = VisualizationOutput(
        is_useful=True,
        chart_type="bar",
        x_column="Region",
        y_column="Revenue",
        title="Revenue by Region",
        reasoning="Bar chart is optimal for categorical revenue comparison",
    )
    assert vis.chart_type == "bar"

    ins = InsightOutput(
        executive_summary="East region is the primary revenue driver.",
        key_insights=["East region contributed 45% of total sales."],
        actionable_recommendations=["Increase inventory in East region."],
        confidence_score=0.95,
    )
    assert ins.confidence_score == 0.95


def test_malformed_llm_output_validation():
    """Test Pydantic validation error when LLM output is malformed or invalid."""
    malformed_data = {
        "intent_summary": "Missing required fields",
        "next_agent": "invalid_node_name",  # invalid literal
    }
    with pytest.raises(ValidationError):
        SupervisorOutput.model_validate(malformed_data)


def test_prompt_rendering():
    """Test system prompt formatting and prompt content integrity."""
    prompts = [
        SUPERVISOR_SYSTEM_PROMPT,
        DATA_PROFILER_SYSTEM_PROMPT,
        DATA_ANALYST_SYSTEM_PROMPT,
        VISUALIZATION_SYSTEM_PROMPT,
        INSIGHT_GENERATOR_SYSTEM_PROMPT,
    ]
    for prompt in prompts:
        assert isinstance(prompt, str)
        assert len(prompt) > 50
