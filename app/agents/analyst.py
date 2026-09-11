import logging
import re
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from app.tools.analysis import (
    _resolve_dataframe,
    group_data,
    aggregate_data,
    filter_data,
    AnalysisToolError,
)
from app.tools.statistics import calculate_statistics, calculate_correlation
from app.tools.anomaly_detection import analyze_trend, detect_anomalies
from app.tools.sandbox import execute_sandboxed_code
from app.tools.multi_dataset import join_datasets, compare_datasets, auto_detect_join_keys
from app.data.context import build_dataset_context
from app.data.metadata import sanitize_for_json
from app.schemas.agents import AnalystOutput, AnalysisPlan, AnalysisPlanStep
from app.llm.model import get_llm_model, MockChatModel
from app.llm.prompts import (
    DATA_ANALYST_PLANNER_PROMPT,
    DATA_ANALYST_INTERPRETER_PROMPT,
)

logger = logging.getLogger("datapilot.agents.analyst")

ALLOWED_TOOLS = {
    "filter_data",
    "group_data",
    "aggregate_data",
    "calculate_statistics",
    "calculate_correlation",
    "analyze_trend",
    "detect_anomalies",
    "custom_sandbox",
    "join_datasets",
    "compare_datasets",
}


class DataAnalystAgentError(Exception):
    """Custom exception raised when Data Analyst Agent execution fails catastrophically."""
    pass


from app.data.metadata import is_identifier_column


def resolve_query_intent(query: str, target_df: pd.DataFrame) -> Dict[str, Any]:
    """Deterministically resolves query intent to dimension, metric, aggregation, sort_direction, limit."""
    q_lower = query.lower().strip()
    cols = list(target_df.columns)

    # 1. Check for explicit identifier request in query
    explicit_id_keywords = ["order id", "customer id", "product id", "transaction id", "invoice id", "by order", "by customer", "sku"]
    is_explicit_id_request = any(kw in q_lower for kw in explicit_id_keywords)

    non_id_cols = [c for c in cols if is_explicit_id_request or not is_identifier_column(c, target_df[c])]
    cat_cols = [c for c in non_id_cols if not pd.api.types.is_numeric_dtype(target_df[c].dtype)]
    num_cols = [c for c in non_id_cols if pd.api.types.is_numeric_dtype(target_df[c].dtype)]

    resolved_dim = None
    requested_dim_term = None

    dim_concepts = {
        "state": ["state", "states", "province", "provinces", "region", "regions"],
        "city": ["city", "cities", "town", "towns", "location", "locations"],
        "category": ["category", "categories", "department", "departments"],
        "subcategory": ["sub-category", "subcategory", "sub-categories", "subcategories", "sub category", "sub categories"],
        "customer": ["customer", "customers", "client", "clients", "buyer", "buyers"],
        "product": ["product", "products", "item", "items", "goods"],
        "segment": ["segment", "segments", "tier", "tiers"],
        "country": ["country", "countries", "nation", "nations"],
    }

    for concept, keywords in dim_concepts.items():
        matched_kw = next((kw for kw in keywords if kw in q_lower), None)
        if matched_kw:
            requested_dim_term = matched_kw
            for c in cat_cols:
                cl = c.lower()
                if concept in cl or matched_kw in cl or cl in keywords:
                    resolved_dim = c
                    break
                if concept == "subcategory" and ("sub" in cl or "cat" in cl):
                    resolved_dim = c
                    break
            if resolved_dim:
                break

    if not resolved_dim:
        for c in cat_cols:
            cl = c.lower()
            if cl in q_lower or (len(cl) > 3 and cl[:-1] in q_lower):
                resolved_dim = c
                break

    # 2. Metric Resolution
    resolved_metric = None
    metric_keywords = ["sales", "amount", "revenue", "profit", "quantity", "units", "spend", "turnover"]
    mentioned_metric_kw = next((kw for kw in metric_keywords if kw in q_lower), None)

    if mentioned_metric_kw:
        exact_col = next((c for c in num_cols if c.lower() == mentioned_metric_kw), None)
        if exact_col:
            resolved_metric = exact_col
        else:
            if mentioned_metric_kw in ("sales", "total sales"):
                resolved_metric = next((c for c in num_cols if c.lower() in ("sales", "amount", "revenue")), None)
            elif mentioned_metric_kw == "revenue":
                resolved_metric = next((c for c in num_cols if c.lower() in ("revenue", "sales", "amount")), None)
            elif mentioned_metric_kw == "amount":
                resolved_metric = next((c for c in num_cols if c.lower() in ("amount", "sales", "revenue")), None)
            elif mentioned_metric_kw == "profit":
                resolved_metric = next((c for c in num_cols if c.lower() in ("profit", "net_profit", "earnings")), None)
            elif mentioned_metric_kw in ("quantity", "units"):
                resolved_metric = next((c for c in num_cols if c.lower() in ("quantity", "units", "volume")), None)

    if not resolved_metric and num_cols:
        resolved_metric = num_cols[0]

    # 3. Aggregation & Sort Direction & Limit
    if any(k in q_lower for k in ["average", "mean"]):
        agg = "mean"
    else:
        agg = "sum"

    if any(k in q_lower for k in ["lowest", "worst", "least", "bottom"]):
        sort_dir = "ascending"
    else:
        sort_dir = "descending"

    limit_val = None
    lim_match = re.search(r"(?:top|first|highest|lowest)\s*(\d+)", q_lower)
    if lim_match:
        limit_val = int(lim_match.group(1))

    return {
        "dimension": resolved_dim,
        "requested_dim_term": requested_dim_term,
        "metric": resolved_metric,
        "aggregation": agg,
        "sort_direction": sort_dir,
        "limit": limit_val,
        "is_explicit_id_request": is_explicit_id_request,
    }


def build_heuristic_plan(query: str, target_df: pd.DataFrame) -> AnalysisPlan:
    """Generates a deterministic AnalysisPlan based on query pattern matching."""
    q_lower = query.lower()
    cols = list(target_df.columns)
    num_cols = list(target_df.select_dtypes(include=["number"]).columns)
    date_cols = [
        c for c in cols
        if "date" in c.lower() or "time" in c.lower() or "quarter" in c.lower() or "month" in c.lower()
    ]

    steps: List[AnalysisPlanStep] = []
    rec_vis = False

    # Filter pattern
    filter_step = None
    age_col = next((c for c in num_cols if "age" in c.lower()), None)
    filter_match = re.search(r"(?:above|>|greater than|over)\s*(\d+)", q_lower)
    if filter_match:
        val = int(filter_match.group(1))
        col_to_filter = age_col or (num_cols[0] if num_cols else None)
        if col_to_filter:
            filter_step = AnalysisPlanStep(
                tool_name="filter_data",
                params={"filters": [{"column": col_to_filter, "operator": ">", "value": val}]},
                purpose=f"Filter dataset where {col_to_filter} > {val}"
            )
            steps.append(filter_step)

    # Trend Query Pattern
    if any(k in q_lower for k in ["trend", "over time", "monthly", "quarterly", "growth"]) and (date_cols or len(cols) >= 2):
        d_col = date_cols[0] if date_cols else cols[0]
        v_col = num_cols[0] if num_cols else cols[-1]
        steps.append(AnalysisPlanStep(
            tool_name="analyze_trend",
            params={"date_column": d_col, "value_column": v_col, "period": "ME"},
            purpose=f"Analyze temporal trend of {v_col} over {d_col}"
        ))
        rec_vis = True

    # Anomaly Query Pattern
    elif any(k in q_lower for k in ["anomaly", "anomalies", "outlier", "outliers", "unusual"]) and num_cols:
        target_num = num_cols[0]
        steps.append(AnalysisPlanStep(
            tool_name="detect_anomalies",
            params={"column": target_num, "method": "zscore", "threshold": 3.0},
            purpose=f"Detect statistical outliers in column {target_num}"
        ))
        rec_vis = True

    # Correlation Query Pattern
    elif any(k in q_lower for k in ["correlation", "correlate", "relationship"]) and len(num_cols) >= 2:
        steps.append(AnalysisPlanStep(
            tool_name="calculate_correlation",
            params={"columns": num_cols[:5], "method": "pearson"},
            purpose=f"Calculate correlation matrix across numerical columns"
        ))
        rec_vis = True

    # Descriptive Statistics Query Pattern
    elif any(k in q_lower for k in ["statistics", "summary", "describe"]) and num_cols:
        steps.append(AnalysisPlanStep(
            tool_name="calculate_statistics",
            params={"columns": num_cols[:4]},
            purpose="Calculate summary statistics across numeric columns"
        ))

    # Grouping / Category / Top Query Pattern
    elif any(k in q_lower for k in ["top", "best", "worst", "highest", "lowest", "total", "average", "mean", "sum", "by ", "group"]) or len(cols) >= 2:
        intent = resolve_query_intent(query, target_df)
        g_col = intent["dimension"]
        v_col = intent["metric"]

        if not g_col and intent["requested_dim_term"]:
            # User requested a dimension (e.g. "states") that does not exist in dataset.
            # Do NOT fall back to arbitrary columns.
            pass
        elif g_col and v_col:
            aggs = {v_col: [intent["aggregation"]]}
            s_by = f"{v_col}_{intent['aggregation']}"
            s_dir = intent["sort_direction"]

            group_params: Dict[str, Any] = {
                "group_by": [g_col],
                "aggregations": aggs,
                "sort_by": s_by,
                "sort_direction": s_dir,
            }
            if intent["limit"]:
                group_params["limit"] = intent["limit"]

            steps.append(AnalysisPlanStep(
                tool_name="group_data",
                params=group_params,
                purpose=f"Group by {g_col} and calculate {intent['aggregation']} for {v_col}"
            ))
            rec_vis = True

    # Fallback to calculate statistics if no steps were added
    if not steps:
        non_id_num = [c for c in num_cols if not is_identifier_column(c, target_df[c])]
        if non_id_num:
            steps.append(AnalysisPlanStep(
                tool_name="calculate_statistics",
                params={"columns": non_id_num[:4]},
                purpose="Calculate descriptive statistics for numeric columns"
            ))
        else:
            steps.append(AnalysisPlanStep(
                tool_name="aggregate_data",
                params={"aggregations": {cols[0]: ["count"]}},
                purpose="Calculate dataset-wide row count"
            ))

    return AnalysisPlan(
        analysis_goal=f"Execute analysis for query: '{query}'",
        steps=steps,
        recommended_visualization=rec_vis,
    )


def generate_analysis_plan(
    query: str,
    target_df: pd.DataFrame,
    dataset_id: Optional[str] = None,
    llm: Optional[BaseChatModel] = None,
) -> AnalysisPlan:
    """Generates an analysis plan using LLM structured output, falling back to heuristic plan."""
    context_text = build_dataset_context(dataset_id=dataset_id, df=target_df)

    if llm is not None:
        agent_llm = llm
    else:
        try:
            agent_llm = get_llm_model()
        except Exception as e:
            logger.warning(f"Data Analyst Agent failed to initialize LLM model: {e}. Using MockChatModel.")
            agent_llm = MockChatModel()

    if isinstance(agent_llm, MockChatModel):
        return build_heuristic_plan(query, target_df)

    try:
        structured_llm = agent_llm.with_structured_output(AnalysisPlan)
        messages = [
            SystemMessage(content=DATA_ANALYST_PLANNER_PROMPT),
            HumanMessage(content=f"User Query: '{query}'\n\nDataset Schema & Context:\n{context_text}"),
        ]
        plan: AnalysisPlan = structured_llm.invoke(messages)
        if plan and plan.steps:
            return plan
        logger.warning("LLM planner returned plan with no steps. Falling back to heuristic planner.")
        return build_heuristic_plan(query, target_df)
    except Exception as e:
        logger.warning(f"LLM planner execution failed: {e}. Using heuristic planner fallback.")
        return build_heuristic_plan(query, target_df)


def validate_analysis_plan(
    plan: AnalysisPlan, target_df: pd.DataFrame
) -> Tuple[AnalysisPlan, List[str]]:
    """Validates analysis plan steps against allowed tools and dataset schema."""
    warnings: List[str] = []
    valid_steps: List[AnalysisPlanStep] = []
    df_cols = set(target_df.columns)

    for idx, step in enumerate(plan.steps):
        t_name = step.tool_name
        params = step.params or {}

        # 1. Validate tool name existence
        if t_name not in ALLOWED_TOOLS:
            warnings.append(
                f"Step {idx+1} ignored: Invalid or non-existent tool '{t_name}' requested."
            )
            continue

        # 2. Validate column references & dtypes & identifier protection
        invalid_cols = []
        if t_name == "filter_data":
            filters = params.get("filters", [])
            for cond in filters:
                col = cond.get("column")
                if col and col not in df_cols:
                    invalid_cols.append(col)

        elif t_name == "group_data":
            g_cols = params.get("group_by", [])
            query_goal = plan.analysis_goal or ""
            intent = resolve_query_intent(query_goal, target_df)

            # 1. Identifier protection check
            id_cols_used = [c for c in g_cols if c in df_cols and is_identifier_column(c, target_df[c])]
            if id_cols_used and not intent["is_explicit_id_request"]:
                warnings.append(
                    f"Step {idx+1} ({t_name}) validation error: Cannot group by identifier column(s) {id_cols_used} unless explicitly requested by user."
                )
                if intent["dimension"] and intent["metric"]:
                    step = AnalysisPlanStep(
                        tool_name="group_data",
                        params={
                            "group_by": [intent["dimension"]],
                            "aggregations": {intent["metric"]: [intent["aggregation"]]},
                            "sort_by": f"{intent['metric']}_{intent['aggregation']}",
                            "sort_direction": intent["sort_direction"],
                            "limit": intent["limit"],
                        },
                        purpose=f"Group by {intent['dimension']} and calculate {intent['aggregation']} for {intent['metric']}",
                    )
                    warnings.append(
                        f"Re-planned Step {idx+1} to use validated dimension '{intent['dimension']}'."
                    )
                else:
                    invalid_cols.extend(id_cols_used)

            # 2. Semantic Dimension Mismatch Check (e.g. Month of Order Date instead of requested State)
            elif intent["requested_dim_term"] and intent["dimension"]:
                primary_planned_dim = g_cols[0] if g_cols else None
                if primary_planned_dim and primary_planned_dim != intent["dimension"]:
                    warnings.append(
                        f"Step {idx+1} ({t_name}) semantic mismatch: Planned group column '{primary_planned_dim}' does not match requested dimension '{intent['requested_dim_term']}' (resolved to '{intent['dimension']}')."
                    )
                    step = AnalysisPlanStep(
                        tool_name="group_data",
                        params={
                            "group_by": [intent["dimension"]],
                            "aggregations": {intent["metric"]: [intent["aggregation"]]} if intent["metric"] else params.get("aggregations", {}),
                            "sort_by": f"{intent['metric']}_{intent['aggregation']}" if intent["metric"] else params.get("sort_by"),
                            "sort_direction": intent["sort_direction"],
                            "limit": intent["limit"],
                        },
                        purpose=f"Group by {intent['dimension']} and calculate {intent['aggregation']} for {intent['metric'] or 'values'}",
                    )
                    warnings.append(f"Re-planned Step {idx+1} to group by validated dimension '{intent['dimension']}'.")

            # 3. Missing Requested Dimension Check
            elif intent["requested_dim_term"] and not intent["dimension"]:
                warnings.append(
                    f"Step {idx+1} ({t_name}) validation error: Requested dimension matching '{intent['requested_dim_term']}' is not present in dataset columns."
                )
                invalid_cols.extend(g_cols or ["missing_dimension"])

            for col in params.get("group_by", []):
                if col not in df_cols:
                    invalid_cols.append(col)
            for col, funcs in params.get("aggregations", {}).items():
                if col not in df_cols:
                    invalid_cols.append(col)
                else:
                    col_dtype = target_df[col].dtype
                    is_num = pd.api.types.is_numeric_dtype(col_dtype)
                    for func in funcs:
                        if str(func).lower() in {"sum", "mean", "std", "var", "median"} and not is_num:
                            warnings.append(
                                f"Step {idx+1} ({t_name}) validation error: Cannot perform numeric aggregation '{func}' on non-numeric column '{col}' (dtype: {col_dtype})."
                            )

        elif t_name == "aggregate_data":
            for col, funcs in params.get("aggregations", {}).items():
                if col not in df_cols:
                    invalid_cols.append(col)
                else:
                    col_dtype = target_df[col].dtype
                    is_num = pd.api.types.is_numeric_dtype(col_dtype)
                    for func in funcs:
                        if str(func).lower() in {"sum", "mean", "std", "var", "median"} and not is_num:
                            warnings.append(
                                f"Step {idx+1} ({t_name}) validation error: Cannot perform numeric aggregation '{func}' on non-numeric column '{col}' (dtype: {col_dtype})."
                            )
        elif t_name in ("calculate_statistics", "calculate_correlation"):
            for col in params.get("columns", []):
                if col not in df_cols:
                    invalid_cols.append(col)
        elif t_name == "analyze_trend":
            d_col = params.get("date_column")
            v_col = params.get("value_column")
            if d_col and d_col not in df_cols:
                invalid_cols.append(d_col)
            if v_col and v_col not in df_cols:
                invalid_cols.append(v_col)
        elif t_name == "detect_anomalies":
            col = params.get("column")
            if col and col not in df_cols:
                invalid_cols.append(col)

        if invalid_cols:
            warnings.append(
                f"Step {idx+1} ({t_name}) warnings: Invalid column(s) {invalid_cols} not present in dataset."
            )
            continue

        valid_steps.append(step)

    validated_plan = AnalysisPlan(
        analysis_goal=plan.analysis_goal,
        steps=valid_steps,
        recommended_visualization=plan.recommended_visualization,
    )
    return validated_plan, warnings


def _apply_filter_step(
    df: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Applies filter step to DataFrame and returns (filtered_df, tool_result_dict)."""
    filters = params.get("filters", [])
    if not filters and "column" in params:
        filters = [{
            "column": params.get("column"),
            "operator": params.get("operator", "=="),
            "value": params.get("value"),
        }]

    res_dict = filter_data(df=df, filters=filters)

    filtered_df = df.copy()
    for cond in filters:
        col = cond.get("column")
        op = cond.get("operator", "==")
        val = cond.get("value")

        if not col or col not in filtered_df.columns:
            continue

        series = filtered_df[col]
        try:
            if op == "==":
                filtered_df = filtered_df[series == val]
            elif op == "!=":
                filtered_df = filtered_df[series != val]
            elif op == ">":
                filtered_df = filtered_df[series > float(val)]
            elif op == ">=":
                filtered_df = filtered_df[series >= float(val)]
            elif op == "<":
                filtered_df = filtered_df[series < float(val)]
            elif op == "<=":
                filtered_df = filtered_df[series <= float(val)]
            elif op == "contains":
                filtered_df = filtered_df[
                    series.astype(str).str.contains(str(val), case=False, na=False)
                ]
            elif op == "in":
                if not isinstance(val, list):
                    val = [val]
                filtered_df = filtered_df[series.isin(val)]
        except Exception:
            pass

    return filtered_df, res_dict


def execute_analysis_plan(
    plan: AnalysisPlan,
    target_df: pd.DataFrame,
    dataset_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str], str]:
    """Executes validated analysis plan step-by-step with DataFrame chaining."""
    current_df = target_df.copy()
    tool_calls_executed: List[Dict[str, Any]] = []
    quantitative_results: Dict[str, Any] = {}
    execution_errors: List[str] = []

    if not plan.steps:
        return ([], {}, ["No valid execution steps present in analysis plan."], "failed")

    successful_steps = 0

    for idx, step in enumerate(plan.steps):
        t_name = step.tool_name
        params = step.params or {}
        step_key = f"step_{idx+1}_{t_name}"

        executed_call_record = {
            "tool": t_name,
            "params": params,
            "purpose": step.purpose,
            "status": "pending",
        }

        try:
            if t_name == "filter_data":
                current_df, res = _apply_filter_step(current_df, params)
                quantitative_results[step_key] = res
                executed_call_record["status"] = "success"
                successful_steps += 1

            elif t_name == "group_data":
                g_cols = params.get("group_by", [])
                aggs = params.get("aggregations")
                s_by = params.get("sort_by")
                s_dir = params.get("sort_direction")
                lim = params.get("limit")
                res = group_data(
                    df=current_df,
                    group_by=g_cols,
                    aggregations=aggs,
                    sort_by=s_by,
                    sort_direction=s_dir,
                    limit=lim,
                )
                quantitative_results[step_key] = res
                executed_call_record["status"] = "success"
                successful_steps += 1

            elif t_name == "aggregate_data":
                aggs = params.get("aggregations", {})
                res = aggregate_data(df=current_df, aggregations=aggs)
                quantitative_results[step_key] = res
                executed_call_record["status"] = "success"
                successful_steps += 1

            elif t_name == "calculate_statistics":
                cols = params.get("columns", [])
                res = calculate_statistics(df=current_df, columns=cols)
                quantitative_results[step_key] = res
                executed_call_record["status"] = "success"
                successful_steps += 1

            elif t_name == "calculate_correlation":
                cols = params.get("columns", [])
                method = params.get("method", "pearson")
                res = calculate_correlation(df=current_df, columns=cols, method=method)
                quantitative_results[step_key] = res
                executed_call_record["status"] = "success"
                successful_steps += 1

            elif t_name == "analyze_trend":
                d_col = params.get("date_column", "")
                v_col = params.get("value_column", "")
                period = params.get("period", "ME")
                res = analyze_trend(
                    df=current_df, date_column=d_col, value_column=v_col, period=period
                )
                quantitative_results[step_key] = res
                executed_call_record["status"] = "success"
                successful_steps += 1

            elif t_name == "detect_anomalies":
                col = params.get("column", "")
                method = params.get("method", "zscore")
                thresh = float(params.get("threshold", 3.0))
                res = detect_anomalies(
                    df=current_df, column=col, method=method, threshold=thresh
                )
                quantitative_results[step_key] = res
                executed_call_record["status"] = "success"
                successful_steps += 1

            elif t_name == "custom_sandbox":
                code = params.get("code", "")
                res = execute_sandboxed_code(code=code, df=current_df)
                quantitative_results[step_key] = res
                if res.get("success"):
                    executed_call_record["status"] = "success"
                    successful_steps += 1
                else:
                    executed_call_record["status"] = "failed"
                    err_msg = f"Sandbox execution failed: {', '.join(res.get('security_violations', [])) or res.get('output')}"
                    execution_errors.append(err_msg)

            elif t_name == "join_datasets":
                ds_ids = params.get("dataset_ids") or ([dataset_id] if dataset_id else None)
                j_keys = params.get("join_keys")
                j_type = params.get("join_type", "inner")
                res = join_datasets(dataset_ids=ds_ids, join_keys=j_keys, join_type=j_type, register_result=False, dfs=params.get("dfs"))
                quantitative_results[step_key] = res
                if "merged_df" in res and isinstance(res["merged_df"], pd.DataFrame):
                    current_df = res["merged_df"]
                executed_call_record["status"] = "success"
                successful_steps += 1

            elif t_name == "compare_datasets":
                ds_ids = params.get("dataset_ids") or ([dataset_id] if dataset_id else None)
                res = compare_datasets(dataset_ids=ds_ids, dfs=params.get("dfs"))
                quantitative_results[step_key] = res
                executed_call_record["status"] = "success"
                successful_steps += 1

            else:
                executed_call_record["status"] = "failed"
                execution_errors.append(f"Execution engine encountered unsupported tool '{t_name}'.")

        except Exception as e:
            executed_call_record["status"] = "failed"
            execution_errors.append(f"Step {idx+1} ({t_name}) failed: {str(e)}")

        tool_calls_executed.append(executed_call_record)

    if successful_steps == len(plan.steps):
        status = "success"
    elif successful_steps > 0:
        status = "partial_success"
    else:
        status = "failed"

    return tool_calls_executed, quantitative_results, execution_errors, status


def determine_requires_visualization(
    query: str,
    plan: Optional[AnalysisPlan] = None,
    quantitative_results: Optional[Dict[str, Any]] = None,
    execution_status: str = "success",
) -> bool:
    """Application-level validation heuristic for requires_visualization."""
    if execution_status == "failed" or not quantitative_results:
        return False

    q_lower = query.lower()

    # Explicit chart keywords requested by user
    if any(w in q_lower for w in ["chart", "plot", "visualize", "graph"]):
        return True

    # Rule 1: Single scalar statistic or simple row count -> False
    if any(w in q_lower for w in ["count rows", "total rows", "how many rows", "single mean", "overall average"]) and not any(w in q_lower for w in ["by", "group", "trend"]):
        return False

    executed_tools = list(quantitative_results.keys())

    # Rule 2: Trend analysis, anomaly detection, correlation matrix -> True
    if any("analyze_trend" in t or "detect_anomalies" in t or "calculate_correlation" in t for t in executed_tools):
        return True

    # Rule 3: Group data with 2+ rows -> True
    for t_key, res in quantitative_results.items():
        if "group_data" in t_key:
            if isinstance(res, dict) and res.get("result_row_count", 0) >= 2:
                return True

    # Rule 4: Key query indicators -> True
    if any(k in q_lower for k in ["trend", "top", "distribution", "monthly", "quarterly", "over time", "correlation", "highest", "lowest", "best", "worst", "most"]):
        return True

    if plan:
        return plan.recommended_visualization

    return False


def format_empirical_findings(
    quantitative_results: Dict[str, Any], errors: List[str]
) -> str:
    """Formats quantitative tool results into empirical text findings without LLM hallucination."""
    if not quantitative_results:
        if errors:
            return f"Analysis completed with errors: {'; '.join(errors)}"
        return "No quantitative findings produced."

    findings = []

    for step_key, res in quantitative_results.items():
        if not isinstance(res, dict):
            continue

        if "analyze_trend" in step_key:
            d_col = res.get("date_column", "date")
            v_col = res.get("value_column", "value")
            direction = res.get("overall_trend_direction", "unknown")
            pct = res.get("overall_change_percentage", 0.0)
            avg_g = res.get("average_period_growth_percentage", 0.0)
            findings.append(
                f"Trend analysis for '{v_col}' over '{d_col}': overall trend is {direction} ({pct}% overall change, {avg_g}% average period growth)."
            )

        elif "detect_anomalies" in step_key:
            col = res.get("column", "column")
            cnt = res.get("anomaly_count", 0)
            pct = res.get("anomaly_percentage", 0.0)
            findings.append(
                f"Anomaly detection on '{col}': found {cnt} anomaly row(s) ({pct}% of total rows)."
            )

        elif "group_data" in step_key:
            g_cols_list = res.get("group_by", [])
            g_cols = ", ".join(g_cols_list)
            rows = res.get("rows", [])
            row_cnt = res.get("result_row_count", 0)
            sort_b = res.get("sort_by")
            sort_dir = res.get("sort_direction") or "descending"
            lim = res.get("limit")

            if rows:
                first_row = rows[0]
                metric_keys = [k for k in first_row.keys() if k not in g_cols_list]
                target_k = sort_b if (sort_b and sort_b in first_row) else (metric_keys[0] if metric_keys else None)

                if target_k and g_cols_list:
                    primary_g = g_cols_list[0]
                    is_rev = sort_dir.lower() in ["descending", "desc"]
                    sorted_rows = sorted(
                        rows,
                        key=lambda x: (x.get(target_k) if x.get(target_k) is not None else -float('inf')),
                        reverse=is_rev,
                    )
                    top_limit = lim if (lim and lim > 0) else 5
                    top_rows = sorted_rows[:top_limit]
                    ranked_items = [
                        f"{idx+1}. {r.get(primary_g, 'N/A')}: {r.get(target_k)}"
                        for idx, r in enumerate(top_rows)
                    ]
                    order_label = "highest" if is_rev else "lowest"
                    findings.append(
                        f"Ranked by '{target_k}' ({order_label}) grouped by '{g_cols}' ({row_cnt} total groups): {', '.join(ranked_items)}."
                    )
                else:
                    findings.append(f"Grouped by '{g_cols}' returning {row_cnt} groups.")
            else:
                findings.append(f"Grouped by '{g_cols}' returning 0 groups.")

        elif "filter_data" in step_key:
            orig = res.get("original_row_count", 0)
            filt = res.get("filtered_row_count", 0)
            findings.append(f"Filtered dataset from {orig} rows down to {filt} matching rows.")

        elif "aggregate_data" in step_key:
            aggs = res.get("aggregations", {})
            agg_str_parts = []
            for col, fdict in aggs.items():
                col_parts = [f"{k}={v}" for k, v in fdict.items()]
                agg_str_parts.append(f"'{col}': {', '.join(col_parts)}")
            findings.append(f"Aggregations: {'; '.join(agg_str_parts)}.")

        elif "calculate_statistics" in step_key:
            stats = res.get("statistics", {})
            stat_parts = []
            for col, sdict in stats.items():
                if isinstance(sdict, dict) and "mean" in sdict:
                    stat_parts.append(
                        f"'{col}': mean={sdict['mean']}, std={sdict.get('std')}, min={sdict['min']}, max={sdict['max']}"
                    )
            findings.append(f"Summary statistics: {'; '.join(stat_parts)}.")

        elif "calculate_correlation" in step_key:
            cols = res.get("columns", [])
            findings.append(f"Calculated pairwise correlation matrix across {len(cols)} columns: {', '.join(cols)}.")

        elif "custom_sandbox" in step_key:
            output = res.get("output", "")
            result = res.get("result", "")
            findings.append(f"Custom sandbox code execution result: {result or output}.")

        elif "join_datasets" in step_key:
            stats = res.get("stats", {})
            out_cnt = res.get("output_row_count", 0)
            j_type = stats.get("join_type", "inner")
            matched = stats.get("matched_rows_count", 0)
            card = stats.get("cardinality", "unknown")
            findings.append(f"Joined datasets using '{j_type}' join (cardinality: {card}): merged output contains {out_cnt} rows with {matched} matching key rows.")

        elif "compare_datasets" in step_key:
            comp = res.get("comparison_details", {})
            rows = comp.get("row_counts", {})
            common = comp.get("common_columns", [])
            row_str = ", ".join([f"'{k}': {v} rows" for k, v in rows.items()])
            findings.append(f"Multi-dataset comparison ({row_str}): found {len(common)} common column(s) ({', '.join(common[:5])}).")

    if errors:
        findings.append(f"Execution warnings: {'; '.join(errors)}")

    return " ".join(findings) if findings else "Analysis executed cleanly."


def generate_analyst_findings(
    query: str,
    context_text: str,
    quantitative_results: Dict[str, Any],
    errors: List[str],
    llm: Optional[BaseChatModel] = None,
) -> str:
    """Synthesizes empirical tool results into final findings using LLM interpreter or fallback."""
    if llm is not None:
        agent_llm = llm
    else:
        try:
            agent_llm = get_llm_model()
        except Exception as e:
            logger.warning(f"Data Analyst Agent failed to initialize LLM model: {e}. Using MockChatModel.")
            agent_llm = MockChatModel()

    if isinstance(agent_llm, MockChatModel):
        return format_empirical_findings(quantitative_results, errors)

    try:
        results_str = str(quantitative_results)
        messages = [
            SystemMessage(content=DATA_ANALYST_INTERPRETER_PROMPT),
            HumanMessage(
                content=f"User Query: '{query}'\n\nDataset Context:\n{context_text}\n\nQuantitative Results:\n{results_str}\n\nWarnings/Errors:\n{errors}"
            ),
        ]
        response = agent_llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, str) and content.strip():
            return content.strip()
        return format_empirical_findings(quantitative_results, errors)
    except Exception as e:
        logger.warning(f"LLM interpreter invocation failed: {e}. Using empirical fallback.")
        return format_empirical_findings(quantitative_results, errors)


def run_analyst_agent(
    query: str,
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    llm: Optional[BaseChatModel] = None,
) -> AnalystOutput:
    """Executes the Data Analyst Agent node using modular planner-validator-executor design.

    1. Resolves target DataFrame safely (handles invalid dataset ID, empty DataFrame).
    2. Generates analysis plan (LLM planner or heuristic fallback).
    3. Validates analysis plan against tool registry and dataset schema.
    4. Executes analysis plan step-by-step with DataFrame chaining.
    5. Determines visualization recommendation using application rules.
    6. Generates findings strictly based on actual quantitative tool results.
    """
    try:
        target_df = _resolve_dataframe(dataset_id, df)
    except Exception as e:
        logger.warning(f"Analyst Agent dataset resolution failed: {e}")
        return AnalystOutput(
            analysis_goal=f"Execute analysis for query: '{query}'",
            tool_calls_planned=[],
            tool_calls_executed=[],
            execution_status="failed",
            quantitative_results={},
            findings_summary=f"Analysis failed: {str(e)}",
            requires_visualization=False,
            errors=[str(e)],
        )

    if target_df.empty:
        return AnalystOutput(
            analysis_goal=f"Execute analysis for query: '{query}'",
            tool_calls_planned=[],
            tool_calls_executed=[],
            execution_status="failed",
            quantitative_results={},
            findings_summary="Analysis failed: Dataset is empty (0 rows).",
            requires_visualization=False,
            errors=["Dataset is empty (0 rows). Cannot perform analysis."],
        )

    context_text = build_dataset_context(dataset_id=dataset_id, df=target_df)

    if llm is not None:
        agent_llm = llm
    else:
        try:
            agent_llm = get_llm_model()
        except Exception as e:
            logger.warning(f"Data Analyst Agent failed to initialize LLM model: {e}. Using MockChatModel.")
            agent_llm = MockChatModel()

    # 1. Generate Plan
    plan = generate_analysis_plan(query, target_df, dataset_id, agent_llm)

    # 2. Validate Plan
    validated_plan, validation_errors = validate_analysis_plan(plan, target_df)

    if not validated_plan.steps:
        # Fallback to heuristic plan if validation stripped all invalid steps
        heur_plan = build_heuristic_plan(query, target_df)
        validated_plan, heur_errors = validate_analysis_plan(heur_plan, target_df)
        validation_errors.extend(heur_errors)

    planned_calls = [
        {"tool": s.tool_name, "params": s.params, "purpose": s.purpose}
        for s in validated_plan.steps
    ]

    # 3. Execute Plan
    executed_calls, quantitative_results, execution_errors, status = execute_analysis_plan(
        validated_plan, target_df, dataset_id
    )

    all_errors = validation_errors + execution_errors

    # 4. Determine Visualization
    requires_vis = determine_requires_visualization(
        query, validated_plan, quantitative_results, status
    )

    # 5. Generate Findings
    findings_text = generate_analyst_findings(
        query, context_text, quantitative_results, all_errors, agent_llm
    )

    return AnalystOutput(
        analysis_goal=validated_plan.analysis_goal,
        tool_calls_planned=planned_calls,
        tool_calls_executed=executed_calls,
        execution_status=status,
        quantitative_results=quantitative_results,
        findings_summary=findings_text,
        requires_visualization=requires_vis,
        errors=all_errors,
    )
