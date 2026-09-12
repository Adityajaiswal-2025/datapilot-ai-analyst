import logging
import pandas as pd
from typing import Optional, Dict, Any, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.tools.analysis import _resolve_dataframe, AnalysisToolError
from app.tools.profiling import audit_data_quality
from app.tools.hypothesis import generate_automated_hypotheses
from app.tools.insights_engine import discover_automated_insights
from app.data.context import build_dataset_context
from app.schemas.agents import InsightOutput, AnalystOutput, VisualizationOutput
from app.llm.model import get_llm_model, MockChatModel
from app.llm.prompts import INSIGHT_GENERATOR_SYSTEM_PROMPT

logger = logging.getLogger("datapilot.agents.insight")


class DataInsightAgentError(Exception):
    """Custom exception raised when Executive Insight Agent execution fails."""
    pass


def calculate_confidence_score(
    target_df: Optional[pd.DataFrame] = None,
    analyst_output: Optional[AnalystOutput] = None,
) -> float:
    """Calculates empirical confidence score (0.0 to 1.0) based on data quality and analysis precision."""
    if target_df is None or target_df.empty:
        return 0.0

    score = 1.0

    # Quality score penalty
    try:
        quality_audit = audit_data_quality(target_df)
        q_score = quality_audit.get("quality_score", 100.0)
        if q_score < 100.0:
            score -= (100.0 - q_score) / 200.0  # max 0.5 penalty for 0 quality score
    except Exception:
        pass

    # Analyst output execution penalties
    if analyst_output:
        if analyst_output.execution_status == "failed":
            return 0.0
        elif analyst_output.execution_status == "partial_success":
            score -= 0.25

        if analyst_output.findings_summary and "No strong empirical pattern" in analyst_output.findings_summary:
            return 0.40

        if analyst_output.errors:
            score -= min(0.2, len(analyst_output.errors) * 0.05)

    return round(max(0.0, min(1.0, score)), 2)


def is_hypothesis_relevant_to_query(h: Dict[str, Any], query: str, active_cols: List[str]) -> bool:
    """Validates whether an automated hypothesis result is relevant to the user query context."""
    from app.data.metadata import is_identifier_column

    q_lower = query.lower()
    target_col = (h.get("target_column") or h.get("var1") or h.get("column1") or "").lower().strip()
    group_col = (h.get("group_column") or h.get("var2") or h.get("column2") or "").lower().strip()

    # Exclude identifier columns strictly
    if (target_col and is_identifier_column(target_col)) or (group_col and is_identifier_column(group_col)):
        return False

    # For cross-tabulation tests (like Chi-Square), require both columns to be explicitly present in query
    statement = str(h.get("statement") or "").lower()
    test_type = str(h.get("test_type") or "").lower()
    if "chi" in test_type or "chi-square" in statement or "crosstab" in test_type:
        if not (target_col and target_col in q_lower and group_col and group_col in q_lower):
            return False

    # Target or group column must be explicitly mentioned in query
    has_target = bool(target_col) and (target_col in q_lower)
    has_group = bool(group_col) and (group_col in q_lower)

    return has_target or has_group


def generate_empirical_insight(
    query: str,
    target_df: pd.DataFrame,
    analyst_output: Optional[AnalystOutput] = None,
    vis_output: Optional[VisualizationOutput] = None,
) -> InsightOutput:
    """Generates empirical executive insights without LLM hallucination for fallbacks or offline mode."""
    conf = calculate_confidence_score(target_df, analyst_output)

    # 1. Strict Failure Handling
    if analyst_output and analyst_output.execution_status == "failed":
        err_msg = analyst_output.findings_summary or (
            "; ".join(analyst_output.errors) if analyst_output.errors else "Tool execution failed."
        )
        return InsightOutput(
            executive_summary=f"Analysis execution failed for query '{query}': {err_msg}",
            key_insights=[f"Execution warning: {err}" for err in (analyst_output.errors or [err_msg])],
            actionable_recommendations=[
                "Verify dataset schema and ensure numeric metric columns exist for requested aggregations."
            ],
            confidence_score=0.0,
            hypotheses=[],
            structured_insights=[],
        )

    findings = analyst_output.findings_summary if analyst_output else f"Dataset analyzed containing {len(target_df)} rows."
    exec_summary = f"Executive Summary for '{query}': {findings}"

    insights: List[str] = []
    recommendations: List[str] = []
    relevant_hypotheses: List[Dict[str, Any]] = []
    relevant_struct_insights: List[Dict[str, Any]] = []

    # Extract active columns from analyst results
    active_cols: List[str] = []
    if analyst_output and analyst_output.quantitative_results:
        for res in analyst_output.quantitative_results.values():
            if isinstance(res, dict):
                active_cols.extend(res.get("group_by", []))
                active_cols.extend(res.get("columns", []))

    # Add query-specific analytical findings FIRST
    if analyst_output and analyst_output.quantitative_results:
        q_res = analyst_output.quantitative_results
        for step_key, res in q_res.items():
            if isinstance(res, dict):
                if "group_data" in step_key and "rows" in res and res["rows"]:
                    g_cols_list = res.get("group_by", ["category"])
                    g_col = g_cols_list[0]
                    rows = res.get("rows", [])
                    sort_b = res.get("sort_by")
                    sort_dir = res.get("sort_direction") or "descending"
                    lim = res.get("limit")
                    is_rev = sort_dir.lower() in ["descending", "desc"]

                    first_row = rows[0]
                    metric_keys = [k for k in first_row.keys() if k not in g_cols_list]
                    target_metric = sort_b if (sort_b and sort_b in metric_keys) else (metric_keys[0] if metric_keys else "value")
                    is_temporal = any(kw in g_col.lower() for kw in ["date", "month", "year", "quarter", "week", "day"]) or (res.get("time_grain") is not None)

                    if is_temporal and not is_rev:
                        top_limit = lim if (lim and lim > 0) else len(rows)
                        top_rows = rows[:top_limit]
                        period_strs = [
                            f"{r.get(g_col, 'N/A')}: {r.get(target_metric)}"
                            for r in top_rows
                        ]
                        period_summary = ", ".join(period_strs)
                        insights.append(f"Chronological Monthly Breakdown by '{g_col}' ({len(rows)} periods): {period_summary}.")
                        if top_rows:
                            recommendations.append(f"Monitor ongoing trend and period-over-period variance starting from '{top_rows[0].get(g_col, 'period')}'.")
                    else:
                        sorted_rows = sorted(
                            rows,
                            key=lambda x: (x.get(target_metric) if x.get(target_metric) is not None else -float('inf')),
                            reverse=is_rev,
                        )
                        top_limit = lim if (lim and lim > 0) else 5
                        top_rows = sorted_rows[:top_limit]
                        ranked_strs = [
                            f"{i+1}. {r.get(g_col, 'N/A')}: {r.get(target_metric)}"
                            for i, r in enumerate(top_rows)
                        ]
                        top_summary = ", ".join(ranked_strs)
                        order_label = "highest" if is_rev else "lowest"

                        insights.append(f"Primary Ranking by '{target_metric}' ({order_label}) grouped by '{g_col}': {top_summary}.")
                        recommendations.append(f"Focus strategic initiatives on top segment '{top_rows[0].get(g_col, 'segment')}'.")

                elif "analyze_trend" in step_key:
                    direction = res.get("overall_trend_direction", "stable")
                    insights.append(f"Performance trend direction is currently '{direction}' with {res.get('overall_change_percentage', 0.0)}% total change.")
                    if direction == "increasing":
                        recommendations.append("Maintain positive growth momentum through sustained operational efficiency.")
                    else:
                        recommendations.append("Investigate root causes for declining or stagnant trend performance.")

                elif "detect_anomalies" in step_key:
                    anom_cnt = res.get("anomaly_count", 0)
                    insights.append(f"Identified {anom_cnt} statistical anomaly row(s) requiring operational review.")
                    if anom_cnt > 0:
                        recommendations.append("Audit identified anomaly transactions for potential data entry errors or operational risk.")

    # Contextual Automated Hypotheses & Insights (Only if relevant and explicitly separated)
    auto_hypo = generate_automated_hypotheses(target_df, max_pairs=5)
    auto_ins = discover_automated_insights(target_df, max_insights=3)

    hypo_list = auto_hypo.get("hypotheses", [])
    struct_insights = auto_ins.get("insights", [])

    for h in hypo_list:
        if h.get("statistical_significance") and is_hypothesis_relevant_to_query(h, query, active_cols):
            relevant_hypotheses.append(h)
            insights.append(f"[Additional Relevant Insight] Contextual Hypothesis: {h['statement']}")

    for item in struct_insights:
        if is_hypothesis_relevant_to_query({"target_column": item.get("headline")}, query, active_cols):
            relevant_struct_insights.append(item)
            insights.append(f"[Additional Relevant Insight] {item['headline']}: {item['explanation']}")

    if not insights:
        insights.append(f"Dataset schema contains {len(target_df.columns)} columns across {len(target_df)} records.")
        insights.append(f"Empirical analysis completed with confidence score of {conf * 100}%.")

    if not recommendations:
        recommendations.append("Perform granular sub-segment comparisons for deeper quantitative insights.")
        recommendations.append("Establish ongoing monitoring dashboard for real-time tracking.")

    if vis_output and vis_output.is_useful and vis_output.chart_type:
        insights.append(f"Visualization artifact generated: {vis_output.title or vis_output.chart_type} chart.")

    followups = generate_suggested_followups(target_df, analyst_output)

    return InsightOutput(
        executive_summary=exec_summary,
        key_insights=insights,
        actionable_recommendations=recommendations,
        confidence_score=conf,
        hypotheses=relevant_hypotheses,
        structured_insights=relevant_struct_insights,
        suggested_followups=followups,
    )


def generate_suggested_followups(
    target_df: Optional[pd.DataFrame] = None,
    analyst_output: Optional[AnalystOutput] = None,
) -> List[str]:
    """Deterministically generates up to 3 context-aware follow-up question suggestions derived strictly from empirical findings and valid columns."""
    from app.data.metadata import is_identifier_column

    if target_df is None or target_df.empty or analyst_output is None:
        return []

    if analyst_output.execution_status == "failed" or not analyst_output.quantitative_results:
        return []

    cols = list(target_df.columns)
    non_id_cols = [c for c in cols if not is_identifier_column(c, target_df[c])]
    cat_cols = [c for c in non_id_cols if not pd.api.types.is_numeric_dtype(target_df[c].dtype)]
    num_cols = [c for c in non_id_cols if pd.api.types.is_numeric_dtype(target_df[c].dtype)]

    suggestions: List[str] = []

    # 1. Extract primary finding segment if available
    step1_res = analyst_output.quantitative_results.get("step_1_group_data", {})
    rows = step1_res.get("rows", [])
    g_cols = step1_res.get("columns", [])

    if rows:
        top_row = rows[0]
        primary_dim = g_cols[0] if g_cols else None
        if primary_dim and primary_dim in top_row:
            top_val = top_row[primary_dim]
            avail_sec = [c for c in cat_cols if c != primary_dim]
            if avail_sec:
                suggestions.append(f"Show {avail_sec[0]} breakdown for {top_val}")

            if num_cols:
                suggestions.append(f"What is total {num_cols[0]} by {primary_dim}?")

    # 2. Suggest temporal breakdown if date column exists
    date_cols = [c for c in cols if "date" in c.lower() or "time" in c.lower() or "month" in c.lower()]
    if date_cols and not any("month" in s.lower() for s in suggestions):
        suggestions.append("How many orders were placed each month?")

    # 3. Suggest top numeric metric ranking
    if num_cols and cat_cols:
        s_num = f"What are the highest {num_cols[0]} by {cat_cols[0]}?"
        if s_num not in suggestions:
            suggestions.append(s_num)

    seen = set()
    final_suggestions = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            final_suggestions.append(s)
        if len(final_suggestions) == 3:
            break

    return final_suggestions


def run_insight_agent(
    query: str,
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    analyst_output: Optional[AnalystOutput] = None,
    vis_output: Optional[VisualizationOutput] = None,
    llm: Optional[BaseChatModel] = None,
) -> InsightOutput:
    """Executes the Executive Insight Agent node."""
    try:
        target_df = _resolve_dataframe(dataset_id, df)
    except Exception as e:
        logger.warning(f"Insight Agent dataset resolution failed: {e}")
        return InsightOutput(
            executive_summary=f"Analysis cannot be performed for '{query}': Dataset ID not found or unresolvable.",
            key_insights=["No valid dataset available."],
            actionable_recommendations=["Upload a valid CSV or XLSX dataset before requesting executive insights."],
            confidence_score=0.0,
            suggested_followups=[],
        )

    if target_df.empty:
        return InsightOutput(
            executive_summary=f"Analysis cannot be performed for '{query}': Dataset is empty (0 rows).",
            key_insights=["Dataset contains zero rows."],
            actionable_recommendations=["Upload a dataset with valid records."],
            confidence_score=0.0,
            suggested_followups=[],
        )

    # 1. Strict Failure Propagation: If analyst failed, return failure output immediately
    if analyst_output and analyst_output.execution_status == "failed":
        return generate_empirical_insight(query, target_df, analyst_output, vis_output)

    if llm is not None:
        agent_llm = llm
    else:
        try:
            agent_llm = get_llm_model()
        except Exception as e:
            logger.warning(f"Executive Insight Agent failed to initialize LLM model: {e}. Using MockChatModel.")
            agent_llm = MockChatModel()

    if isinstance(agent_llm, MockChatModel):
        return generate_empirical_insight(query, target_df, analyst_output, vis_output)

    # Live LLM execution with structured output
    try:
        context_text = build_dataset_context(dataset_id=dataset_id, df=target_df)
        conf_score = calculate_confidence_score(target_df, analyst_output)

        structured_llm = agent_llm.with_structured_output(InsightOutput)
        messages = [
            SystemMessage(content=INSIGHT_GENERATOR_SYSTEM_PROMPT),
            HumanMessage(
                content=f"User Query: '{query}'\n\nDataset Overview:\n{context_text}"
                + (f"\n\nAnalyst Findings:\n{analyst_output.findings_summary}" if analyst_output else "")
                + (f"\nQuantitative Results:\n{analyst_output.quantitative_results}" if analyst_output else "")
                + (f"\nVisualization Reasoning:\n{vis_output.reasoning}" if vis_output else "")
                + f"\n\nCalculated Empirical Confidence Score: {conf_score}"
            ),
        ]
        result: InsightOutput = structured_llm.invoke(messages)
        if result:
            result.confidence_score = conf_score
            if not result.suggested_followups:
                result.suggested_followups = generate_suggested_followups(target_df, analyst_output)
            return result
        return generate_empirical_insight(query, target_df, analyst_output, vis_output)
    except Exception as e:
        logger.warning(f"LLM Insight generation failed: {e}. Falling back to empirical insights.")
        return generate_empirical_insight(query, target_df, analyst_output, vis_output)

