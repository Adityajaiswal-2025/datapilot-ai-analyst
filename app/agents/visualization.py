import logging
import pandas as pd
from typing import Optional, Dict, Any, Tuple, Literal
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.tools.analysis import _resolve_dataframe, AnalysisToolError
from app.tools.visualization import generate_visualization
from app.data.context import build_dataset_context
from app.schemas.agents import VisualizationOutput, AnalystOutput
from app.llm.model import get_llm_model, MockChatModel
from app.llm.prompts import VISUALIZATION_SYSTEM_PROMPT

logger = logging.getLogger("datapilot.agents.visualization")


class DataVisualizationAgentError(Exception):
    """Custom exception raised when Visualization Agent execution fails catastrophically."""
    pass


def select_heuristic_chart_params(
    query: str,
    target_df: pd.DataFrame,
    analyst_output: Optional[AnalystOutput] = None,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str], Optional[str], str]:
    """Determines chart usefulness, type, x/y columns, title, and reasoning using heuristics."""
    q_lower = query.lower()
    cols = list(target_df.columns)
    num_cols = list(target_df.select_dtypes(include=["number"]).columns)
    cat_cols = list(target_df.select_dtypes(include=["object", "category"]).columns)
    date_cols = [
        c for c in cols
        if any(k in c.lower() for k in ["date", "time", "quarter", "month"])
    ]

    # Explicit check if scalar count / single statistic with no grouping is requested
    if any(k in q_lower for k in ["count rows", "total rows", "how many rows"]) and not any(k in q_lower for k in ["by", "group", "trend"]):
        return False, None, None, None, None, "Single row count or scalar query does not benefit from visualization."

    # Check analyst output recommendation if available
    if analyst_output and not analyst_output.requires_visualization:
        if not any(w in q_lower for w in ["chart", "plot", "graph", "visualize"]):
            return False, None, None, None, None, "Prior analysis determined visualization is not required."

    # Pattern 1: Trend / Line Chart
    if any(k in q_lower for k in ["trend", "over time", "monthly", "quarterly", "growth"]) or (date_cols and num_cols):
        x_col = date_cols[0] if date_cols else cols[0]
        y_col = num_cols[0] if num_cols else cols[-1]
        title = f"Trend Chart: {y_col} over {x_col}"
        reasoning = f"Generated line chart to visualize temporal trend of '{y_col}' across '{x_col}'."
        return True, "line", x_col, y_col, title, reasoning

    # Pattern 2: Outlier / Anomaly / Boxplot
    if any(k in q_lower for k in ["anomaly", "anomalies", "outlier", "outliers", "boxplot"]) and num_cols:
        x_col = num_cols[0]
        y_col = cat_cols[0] if cat_cols else None
        title = f"Outlier Boxplot: {x_col}" + (f" by {y_col}" if y_col else "")
        reasoning = f"Generated boxplot to visualize statistical spread and outliers in '{x_col}'."
        return True, "boxplot", x_col, y_col, title, reasoning

    # Pattern 3: Correlation / Scatter Chart
    if any(k in q_lower for k in ["correlation", "scatter", "relationship"]) and len(num_cols) >= 2:
        x_col = num_cols[0]
        y_col = num_cols[1]
        title = f"Scatter Correlation: {x_col} vs {y_col}"
        reasoning = f"Generated scatter plot to examine numerical relationship between '{x_col}' and '{y_col}'."
        return True, "scatter", x_col, y_col, title, reasoning

    # Pattern 4: Distribution / Histogram
    if any(k in q_lower for k in ["distribution", "histogram", "frequency"]) and num_cols:
        x_col = num_cols[0]
        title = f"Distribution Histogram: {x_col}"
        reasoning = f"Generated histogram to display frequency distribution of '{x_col}'."
        return True, "histogram", x_col, None, title, reasoning

    # Pattern 5: Proportional / Pie Chart
    if any(k in q_lower for k in ["pie", "proportion", "share"]) and cat_cols:
        x_col = cat_cols[0]
        title = f"Proportional Share by {x_col}"
        reasoning = f"Generated pie chart to show proportional breakdown across '{x_col}'."
        return True, "pie", x_col, None, title, reasoning

    # Pattern 6: Default Bar Chart for categorical comparisons
    if cat_cols and num_cols:
        x_col = cat_cols[0]
        y_col = num_cols[0]
        title = f"Comparison: {y_col} by {x_col}"
        reasoning = f"Generated bar chart to compare average '{y_col}' across '{x_col}' categories."
        return True, "bar", x_col, y_col, title, reasoning

    if num_cols:
        x_col = num_cols[0]
        title = f"Distribution of {x_col}"
        reasoning = f"Generated histogram for numeric column '{x_col}'."
        return True, "histogram", x_col, None, title, reasoning

    return False, None, None, None, None, "No suitable columns available for visualization."


def validate_and_fix_chart_params(
    chart_type: Optional[str],
    x_column: Optional[str],
    y_column: Optional[str],
    target_df: pd.DataFrame,
) -> Tuple[str, str, Optional[str]]:
    """Validates and auto-repairs proposed chart type and x/y columns against dataset schema."""
    df_cols = list(target_df.columns)
    num_cols = list(target_df.select_dtypes(include=["number"]).columns)

    valid_types = {"bar", "line", "scatter", "histogram", "boxplot", "pie"}
    c_type = chart_type.lower() if chart_type and chart_type.lower() in valid_types else "bar"

    # Fix X column
    x_col = x_column if x_column and x_column in df_cols else df_cols[0]

    # Fix Y column
    y_col = y_column
    if c_type in ("line", "scatter"):
        if not y_col or y_col not in df_cols:
            y_col = num_cols[0] if num_cols and num_cols[0] != x_col else (df_cols[1] if len(df_cols) > 1 else df_cols[0])

    if y_col and y_col not in df_cols:
        y_col = None

    return c_type, x_col, y_col


def run_visualization_agent(
    query: str,
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    analyst_output: Optional[AnalystOutput] = None,
    llm: Optional[BaseChatModel] = None,
) -> VisualizationOutput:
    """Executes the Data Visualization Agent node.

    1. Resolves target DataFrame safely.
    2. Determines chart usefulness, chart type, and X/Y column parameters via LLM or heuristics.
    3. Validates and auto-repairs column names against dataset schema.
    4. Invokes generate_visualization tool to save PNG file and return image metadata.
    """
    try:
        target_df = _resolve_dataframe(dataset_id, df)
    except Exception as e:
        logger.warning(f"Visualization Agent dataset resolution failed: {e}")
        return VisualizationOutput(
            is_useful=False,
            reasoning=f"Visualization failed: Dataset not found ({str(e)})."
        )

    if target_df.empty:
        return VisualizationOutput(
            is_useful=False,
            reasoning="Visualization skipped: Dataset is empty (0 rows)."
        )

    # 1. Determine Chart Proposal (LLM or Heuristics)
    is_useful = False
    c_type = None
    x_col = None
    y_col = None
    title = None
    reasoning = ""

    if llm is not None:
        agent_llm = llm
    else:
        try:
            agent_llm = get_llm_model()
        except Exception as e:
            logger.warning(f"Visualization Agent failed to initialize LLM model: {e}. Using MockChatModel.")
            agent_llm = MockChatModel()

    if isinstance(agent_llm, MockChatModel):
        is_useful, c_type, x_col, y_col, title, reasoning = select_heuristic_chart_params(
            query, target_df, analyst_output
        )
    else:
        try:
            context_text = build_dataset_context(dataset_id=dataset_id, df=target_df)
            structured_llm = agent_llm.with_structured_output(VisualizationOutput)
            messages = [
                SystemMessage(content=VISUALIZATION_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"User Query: '{query}'\n\nDataset Schema:\n{context_text}"
                    + (f"\n\nPrior Analysis Findings:\n{analyst_output.findings_summary}" if analyst_output else "")
                ),
            ]
            proposal: VisualizationOutput = structured_llm.invoke(messages)
            if proposal and proposal.is_useful:
                is_useful = True
                c_type = proposal.chart_type
                x_col = proposal.x_column
                y_col = proposal.y_column
                title = proposal.title
                reasoning = proposal.reasoning
            else:
                is_useful = False
                reasoning = proposal.reasoning if proposal else "LLM recommended no visualization."
        except Exception as e:
            logger.warning(f"LLM visualization proposal failed: {e}. Using heuristic fallback.")
            is_useful, c_type, x_col, y_col, title, reasoning = select_heuristic_chart_params(
                query, target_df, analyst_output
            )

    if not is_useful or not c_type:
        return VisualizationOutput(
            is_useful=False,
            reasoning=reasoning or "Visualization is not required or beneficial for this query."
        )

    # 2. Validate and Auto-repair Parameters
    c_type, x_col, y_col = validate_and_fix_chart_params(c_type, x_col, y_col, target_df)
    chart_title = title or f"{c_type.capitalize()} Chart: {x_col}" + (f" vs {y_col}" if y_col else "")

    # 3. Render Chart Artifact
    try:
        vis_result = generate_visualization(
            df=target_df,
            chart_type=c_type,
            x_column=x_col,
            y_column=y_col,
            title=chart_title,
        )
        return VisualizationOutput(
            is_useful=True,
            chart_type=c_type, # type: ignore
            x_column=x_col,
            y_column=y_col,
            title=chart_title,
            chart_file_path=vis_result["file_path"],
            base64_image=vis_result["base64"],
            reasoning=reasoning,
        )
    except Exception as e:
        logger.error(f"Visualization rendering failed: {e}")
        return VisualizationOutput(
            is_useful=False,
            reasoning=f"Chart rendering failed: {str(e)}"
        )
