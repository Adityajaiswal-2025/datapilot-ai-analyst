import logging
import pandas as pd
from typing import Optional, Dict, Any, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.tools.analysis import _resolve_dataframe, AnalysisToolError
from app.data.context import build_dataset_context
from app.schemas.agents import SupervisorOutput
from app.llm.model import get_llm_model, MockChatModel
from app.llm.prompts import SUPERVISOR_SYSTEM_PROMPT

logger = logging.getLogger("datapilot.agents.supervisor")


class DataSupervisorAgentError(Exception):
    """Custom exception raised when Supervisor Agent execution fails."""
    pass


def select_heuristic_routing(query: str) -> SupervisorOutput:
    """Determines supervisor routing decision based on query intent keyword matching."""
    q_lower = query.lower()

    # Intent 1: Data Profiling / Quality Audit
    if any(k in q_lower for k in ["profile", "quality", "missing", "schema", "audit", "null", "duplicates"]):
        return SupervisorOutput(
            intent_summary=f"Data profiling & quality audit for: '{query}'",
            next_agent="profiler",
            plan_steps=["Audit data quality", "Classify column data types", "Highlight missingness & warnings"],
            reasoning="Query requests dataset profiling and data quality assessment."
        )

    # Intent 2: Visualization Direct Request
    if any(k in q_lower for k in ["chart", "plot", "graph", "histogram", "boxplot", "pie chart"]) and not any(k in q_lower for k in ["calculate", "filter", "group"]):
        return SupervisorOutput(
            intent_summary=f"Data visualization for: '{query}'",
            next_agent="visualization",
            plan_steps=["Determine chart parameters", "Render chart artifact"],
            reasoning="Query directly requests a data visualization chart."
        )

    # Intent 3: Executive Strategy / Insight Request
    if any(k in q_lower for k in ["executive summary", "recommendation", "strategic advice", "high-level insight"]):
        return SupervisorOutput(
            intent_summary=f"Executive insight synthesis for: '{query}'",
            next_agent="insight",
            plan_steps=["Synthesize executive summary", "Draft strategic recommendations"],
            reasoning="Query requests high-level business insights and recommendations."
        )

    # Intent 4: Default Data Analysis (Statistics, Trends, Anomalies, Grouping, Filtering)
    return SupervisorOutput(
        intent_summary=f"Quantitative data analysis for: '{query}'",
        next_agent="analyst",
        plan_steps=["Plan analysis steps", "Execute deterministic tools/sandbox", "Synthesize findings"],
        reasoning="Query requires quantitative calculation, filtering, statistics, or trend analysis."
    )


def run_supervisor_agent(
    query: str,
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    llm: Optional[BaseChatModel] = None,
) -> SupervisorOutput:
    """Executes the Supervisor Agent node to route user query to appropriate specialized agent."""
    if llm is not None:
        agent_llm = llm
    else:
        try:
            agent_llm = get_llm_model()
        except Exception as e:
            logger.warning(f"Supervisor failed to initialize LLM model: {e}. Using MockChatModel.")
            agent_llm = MockChatModel()

    if isinstance(agent_llm, MockChatModel):
        return select_heuristic_routing(query)

    try:
        context_text = ""
        try:
            target_df = _resolve_dataframe(dataset_id, df)
            context_text = build_dataset_context(dataset_id=dataset_id, df=target_df)
        except Exception:
            pass

        structured_llm = agent_llm.with_structured_output(SupervisorOutput)
        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(
                content=f"User Query: '{query}'" + (f"\n\nDataset Schema:\n{context_text}" if context_text else "")
            ),
        ]
        result: SupervisorOutput = structured_llm.invoke(messages)
        if result and result.next_agent in ("profiler", "analyst", "visualization", "insight", "end"):
            return result
        return select_heuristic_routing(query)
    except Exception as e:
        logger.warning(f"LLM Supervisor routing failed: {e}. Falling back to heuristic router.")
        return select_heuristic_routing(query)
