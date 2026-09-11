import logging
import pandas as pd
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from app.tools.profiling import profile_dataset, ProfilingError
from app.data.context import build_dataset_context
from app.schemas.agents import ProfilerOutput
from app.llm.model import get_llm_model, MockChatModel
from app.llm.prompts import DATA_PROFILER_SYSTEM_PROMPT

logger = logging.getLogger("datapilot.agents.profiler")


class DataProfilerAgentError(Exception):
    """Custom exception raised when Data Profiler Agent execution fails."""
    pass


def run_profiler_agent(
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    llm: Optional[BaseChatModel] = None,
) -> ProfilerOutput:
    """Executes the Data Profiler Agent node.

    1. Gathers empirical profile & quality metrics from deterministic profiling tools.
    2. Builds LLM dataset context.
    3. Invokes LLM with_structured_output(ProfilerOutput) or constructs fallback ProfilerOutput.
    """
    # Step 1: Run deterministic profiling
    try:
        profile = profile_dataset(dataset_id=dataset_id, df=df)
    except ProfilingError as e:
        logger.warning(f"Profiler Agent profiling failed: {e}")
        return ProfilerOutput(
            dataset_summary=f"Profiling failed: {str(e)}",
            quality_score=0.0,
            critical_warnings=[str(e)],
            key_column_observations=[],
        )

    qr = profile.quality_report
    context_text = build_dataset_context(profile=profile)

    # Step 2: Resolve LLM instance
    if llm is not None:
        agent_llm = llm
    else:
        try:
            agent_llm = get_llm_model()
        except Exception as e:
            logger.warning(f"Data Profiler Agent failed to initialize LLM model: {e}. Using MockChatModel.")
            agent_llm = MockChatModel()

    # Step 3: If LLM is MockChatModel or deterministic fallback
    if isinstance(agent_llm, MockChatModel):
        obs = []
        for col in profile.column_profiles[:5]:
            if col.classified_type == "numeric" and col.numeric_stats:
                ns = col.numeric_stats
                obs.append(
                    f"Column '{col.name}' ({col.classified_type}): mean={ns.mean}, min={ns.min}, max={ns.max}"
                )
            elif col.classified_type in ["categorical", "boolean"] and col.categorical_stats:
                cs = col.categorical_stats
                obs.append(
                    f"Column '{col.name}' ({col.classified_type}): mode='{cs.mode}', unique_count={col.unique_count}"
                )

        summary = f"Dataset '{profile.filename}' contains {profile.row_count} rows and {profile.column_count} columns across {len(profile.column_profiles)} profiled fields."

        return ProfilerOutput(
            dataset_summary=summary,
            quality_score=qr.quality_score,
            critical_warnings=qr.warnings,
            key_column_observations=obs,
        )

    # Live LLM execution with structured output
    try:
        structured_llm = agent_llm.with_structured_output(ProfilerOutput)
        messages = [
            SystemMessage(content=DATA_PROFILER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Please analyze and profile this dataset based on its context:\n\n{context_text}"
            ),
        ]
        result: ProfilerOutput = structured_llm.invoke(messages)
        return result
    except Exception as e:
        logger.warning(
            f"LLM structured output failed for Data Profiler Agent: {e}. Falling back to empirical output."
        )
        obs = [
            f"Column '{col.name}' classified as {col.classified_type}"
            for col in profile.column_profiles[:5]
        ]
        return ProfilerOutput(
            dataset_summary=f"Dataset '{profile.filename}' contains {profile.row_count} rows and {profile.column_count} columns.",
            quality_score=qr.quality_score,
            critical_warnings=qr.warnings,
            key_column_observations=obs,
        )
