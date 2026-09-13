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


def resolve_datetime_column(target_df: pd.DataFrame) -> Tuple[Optional[str], float]:
    """Resolves temporal date column with robust validation priority.

    Priority:
    1. Existing pandas datetime dtype (100% valid)
    2. Safely parseable string/object date column with >50% valid parse rate
    Returns (column_name, parse_rate) or (None, 0.0).
    """
    cols = list(target_df.columns)

    # Priority A: Native datetime dtype
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(target_df[c]):
            return c, 1.0

    # Priority B & C: Parseable date column with >50% parse success rate
    best_cand = None
    best_rate = 0.0

    for c in cols:
        series = target_df[c].dropna()
        if series.empty or pd.api.types.is_numeric_dtype(series.dtype):
            continue

        cl = c.lower()
        has_date_name = any(kw in cl for kw in ["date", "time", "month", "year", "quarter", "day"])

        # Test parse rate safely on series without mutating original dataframe
        parsed = pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")
        valid_cnt = parsed.notnull().sum()
        total_cnt = len(series)
        rate = float(valid_cnt / total_cnt) if total_cnt > 0 else 0.0

        if rate > 0.50:
            if has_date_name:
                return c, rate  # Priority C supporting evidence + parse rate > 50%
            if rate > best_rate:
                best_cand = c
                best_rate = rate

    if best_cand and best_rate > 0.50:
        return best_cand, best_rate

    return None, 0.0


def select_best_exploratory_candidates(
    df: pd.DataFrame, max_candidates: int = 3
) -> Tuple[List[AnalysisPlanStep], float, List[str], List[float]]:
    """Dynamically scans dataset for up to max_candidates independent, high-scoring analytical candidates.

    Returns (plan_steps, overall_score, summary_findings, candidate_scores).
    Candidate types evaluated:
    1. Temporal peak periods (group_data by datetime column)
    2. Categorical frequency / concentration (group_data by non-identifier categorical column)
    3. Numeric concentration / sum dominance (group_data by categorical for numeric sum)
    4. Numeric statistical anomalies (detect_anomalies for Z-score >= 3.0)

    Enforces minimum candidate strength threshold of 0.45.
    Filters out identifier columns, constant columns, high-cardinality noise, and generic row counts.
    Candidates are guaranteed independent.
    """
    from app.data.metadata import is_identifier_column

    if df.empty or len(df) < 2:
        return [], 0.30, ["No strong empirical pattern was found in the available data."], []

    candidates: List[Tuple[float, AnalysisPlanStep, str, str]] = []
    cols = list(df.columns)
    non_id_cols = [c for c in cols if not is_identifier_column(c, df[c]) and df[c].nunique() > 1]

    num_cols = [c for c in non_id_cols if pd.api.types.is_numeric_dtype(df[c].dtype)]
    cat_cols = [c for c in non_id_cols if not pd.api.types.is_numeric_dtype(df[c].dtype)]

    # 1. Temporal Peak Candidate Scan (if datetime column exists)
    dt_col, dt_rate = resolve_datetime_column(df)
    if dt_col and dt_rate > 0.50:
        order_id_col = next((c for c in cols if "order" in c.lower() and "id" in c.lower()), None)
        agg_col = order_id_col or (num_cols[0] if num_cols else cols[0])
        agg_func = "count" if (agg_col in cols and not pd.api.types.is_numeric_dtype(df[agg_col].dtype)) else ("sum" if num_cols else "count")

        try:
            parsed = pd.to_datetime(df[dt_col].dropna(), format="mixed", dayfirst=True, errors="coerce")
            valid = parsed.dropna()
            if len(valid) >= 5:
                months = valid.dt.strftime("%Y-%m")
                counts = months.value_counts()
                if len(counts) >= 2:
                    top_period = counts.index[0]
                    top_val = int(counts.iloc[0])
                    total_orders = int(counts.sum())
                    avg_vol = total_orders / len(counts)
                    surge_ratio = top_val / avg_vol if avg_vol > 0 else 1.0

                    if surge_ratio >= 1.25:
                        score = min(0.95, 0.50 + (surge_ratio - 1.0) * 0.30)
                    else:
                        score = 0.20 + (surge_ratio - 1.0) * 0.20

                    pct_above_avg = round(((top_val - avg_vol) / avg_vol) * 100.0, 1) if avg_vol > 0 else 0.0
                    cand_step = AnalysisPlanStep(
                        tool_name="group_data",
                        params={
                            "group_by": [dt_col],
                            "aggregations": {agg_col: [agg_func]},
                            "sort_by": f"{agg_col}_{agg_func}",
                            "sort_direction": "descending",
                            "limit": 1,
                            "time_grain": "month",
                        },
                        purpose=f"Find peak month for {agg_col} over {dt_col}"
                    )
                    finding = f"Peak monthly volume occurred in '{top_period}' over {dt_col} with {top_val} recorded entries ({pct_above_avg}% above monthly average)."
                    candidates.append((score, cand_step, finding, f"temporal_{dt_col}"))
        except Exception:
            pass

    # 2. Categorical Frequency / Concentration Candidate Scan
    for c_cat in cat_cols:
        card = df[c_cat].nunique()
        if 2 <= card <= 50 and len(df[c_cat].dropna()) >= 5:
            counts = df[c_cat].value_counts()
            top_cat = counts.index[0]
            top_val = int(counts.iloc[0])
            total_val = len(df[c_cat].dropna())
            share_pct = round((top_val / total_val) * 100.0, 1)
            expected_uniform_share = 100.0 / card
            concentration_ratio = share_pct / expected_uniform_share if expected_uniform_share > 0 else 1.0

            if concentration_ratio >= 1.25 and share_pct >= 25.0:
                score = min(0.90, 0.45 + (concentration_ratio - 1.0) * 0.30)
            else:
                score = 0.20 + (share_pct / 100.0) * 0.20

            count_col = next((c for c in cols if c != c_cat and not is_identifier_column(c, df[c])), None)
            if not count_col:
                count_col = next((c for c in cols if c != c_cat), cols[0])

            agg_func = "count"

            cand_step = AnalysisPlanStep(
                tool_name="group_data",
                params={
                    "group_by": [c_cat],
                    "aggregations": {count_col: [agg_func]},
                    "sort_by": f"{count_col}_{agg_func}",
                    "sort_direction": "descending",
                    "limit": 5,
                },
                purpose=f"Find top category distribution for {c_cat}"
            )
            second_str = f", followed by '{counts.index[1]}' ({counts.iloc[1]})" if len(counts) > 1 else ""
            finding = f"Top segment in '{c_cat}' is '{top_cat}' with {top_val} entries ({share_pct}% share){second_str}."
            candidates.append((score, cand_step, finding, f"cat_count_{c_cat}"))

    # 3. Numeric Sum / Concentration Candidate Scan
    for c_num in num_cols:
        for c_cat in cat_cols:
            if df[c_cat].nunique() >= 2 and len(df) >= 5:
                grouped = df.groupby(c_cat)[c_num].sum().sort_values(ascending=False)
                tot = grouped.sum()
                if tot > 0 and (df[c_num] >= 0).all():
                    top_cat = grouped.index[0]
                    top_val = float(grouped.iloc[0])
                    share_pct = round((top_val / tot) * 100.0, 1)
                    card = len(grouped)
                    expected_uniform_share = 100.0 / card
                    concentration_ratio = share_pct / expected_uniform_share if expected_uniform_share > 0 else 1.0

                    if concentration_ratio >= 1.25 and share_pct >= 25.0:
                        score = min(0.92, 0.50 + (concentration_ratio - 1.0) * 0.35)
                    else:
                        score = 0.20 + (share_pct / 100.0) * 0.20

                    cand_step = AnalysisPlanStep(
                        tool_name="group_data",
                        params={
                            "group_by": [c_cat],
                            "aggregations": {c_num: ["sum"]},
                            "sort_by": f"{c_num}_sum",
                            "sort_direction": "descending",
                            "limit": 5,
                        },
                        purpose=f"Group by {c_cat} and calculate sum for {c_num}"
                    )
                    finding = f"Segment '{top_cat}' in '{c_cat}' leads total '{c_num}' with {top_val:,.1f} ({share_pct}% share)."
                    candidates.append((score, cand_step, finding, f"num_sum_{c_cat}_{c_num}"))

    # 4. Numeric Anomaly / Outlier Candidate Scan
    for c_num in num_cols:
        s_clean = pd.to_numeric(df[c_num], errors="coerce").dropna()
        if len(s_clean) >= 5:
            mean_val = s_clean.mean()
            std_val = s_clean.std()
            if std_val > 0:
                z_scores = ((s_clean - mean_val) / std_val).abs()
                max_z = float(z_scores.max())
                if max_z >= 3.0:
                    score = min(0.95, 0.55 + (max_z - 3.0) * 0.10)
                    outlier_val = float(s_clean[z_scores == max_z].iloc[0])
                    cand_step = AnalysisPlanStep(
                        tool_name="detect_anomalies",
                        params={
                            "column": c_num,
                            "method": "zscore",
                            "threshold": 3.0,
                        },
                        purpose=f"Detect statistical anomalies in column {c_num}"
                    )
                    finding = f"Statistical anomaly detected in '{c_num}' with outlier value {outlier_val:,.1f} (Z-score: {max_z:.2f})."
                    candidates.append((score, cand_step, finding, f"anomaly_{c_num}"))

    # Filter by minimum candidate strength threshold 0.45
    candidates = [c for c in candidates if c[0] >= 0.45]

    if not candidates:
        return [], 0.30, ["No strong empirical pattern was found in the available data."], []

    # Sort candidates by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate candidates to ensure candidate independence without overlapping redundant target features
    selected_steps: List[AnalysisPlanStep] = []
    selected_findings: List[str] = []
    selected_scores: List[float] = []
    used_dims: set = set()

    for score, step, finding, key in candidates:
        dim = step.params.get("group_by", [step.params.get("column")])[0] if (step.params and ("group_by" in step.params or "column" in step.params)) else key
        if dim in used_dims:
            continue

        used_dims.add(dim)
        selected_steps.append(step)
        selected_findings.append(finding)
        selected_scores.append(score)

        if len(selected_steps) >= max_candidates:
            break

    overall_score = round(max(selected_scores), 2) if selected_scores else 0.30
    return selected_steps, overall_score, selected_findings, selected_scores


def select_best_exploratory_candidate(df: pd.DataFrame) -> Tuple[Optional[AnalysisPlanStep], float, str]:
    """Backward-compatible single-candidate scanner wrapper."""
    steps, score, findings, scores = select_best_exploratory_candidates(df, max_candidates=1)
    if steps:
        return steps[0], score, findings[0]
    return None, 0.30, "No strong empirical pattern was found in the available data."



def resolve_query_intent(query: str, target_df: pd.DataFrame) -> Dict[str, Any]:
    """Deterministically resolves query intent to dimension, metric, aggregation, sort_direction, limit, time_grain, is_exploratory."""
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
    time_grain = None

    # Temporal Grain & Phrase Resolution
    temporal_phrases = {
        "month": ["each month", "by month", "monthly", "per month", "every month", "which month", "month by month"],
        "year": ["each year", "by year", "yearly", "annually", "per year", "every year", "which year"],
        "quarter": ["each quarter", "by quarter", "quarterly", "per quarter", "which quarter"],
        "week": ["each week", "by week", "weekly", "per week", "which week"],
        "day": ["each day", "by day", "daily", "per day", "which day"],
    }

    for grain, phrases in temporal_phrases.items():
        matched_phrase = next((p for p in phrases if p in q_lower), None)
        if matched_phrase:
            time_grain = grain
            requested_dim_term = matched_phrase
            break

    dt_col, dt_rate = resolve_datetime_column(target_df)
    if time_grain and dt_col:
        resolved_dim = dt_col

    if not resolved_dim:
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

    # Check for explicit exploratory request phrases
    exploratory_phrases = [
        "something interesting", "stands out", "interesting insight",
        "interesting insights", "what should i know", "discover insight", "discover insights",
        "tell me about", "key findings", "explore", "find something", "analyze this dataset",
        "analyze dataset", "most important insights", "top insights", "key insights",
        "discover patterns", "supporting numbers", "interesting patterns", "3 insights",
        "three insights", "important insights", "patterns in this data", "in this dataset"
    ]
    has_exploratory_phrase = any(phrase in q_lower for phrase in exploratory_phrases)

    # Check if user query explicitly asks for a specific known dimension or specific temporal breakdown or specific metric ask
    has_explicit_analytical_ask = any(kw in q_lower for kw in [
        "which state", "which city", "which category", "which month", "which year", "which quarter",
        "highest orders", "most orders", "lowest orders", "fewest orders", "top states", "top cities",
        "how many orders", "total number of orders", "count of orders", "total sales", "average sales"
    ])

    is_exploratory = has_exploratory_phrase and not has_explicit_analytical_ask

    # 2. Metric & Count Intent Resolution
    resolved_metric = None
    agg = "sum"

    # Check if query is an order count / entity count query
    order_count_terms = ["order", "orders", "number of orders", "order count", "most orders", "highest number of orders", "fewest orders", "lowest number of orders", "top states by orders"]
    has_explicit_count_request = any(kw in q_lower for kw in ["how many", "number of", "count of", "total count", "which state", "which month", "highest orders", "lowest orders"])
    
    is_order_count_query = (not is_exploratory or has_explicit_count_request) and (
        any(term in q_lower for term in order_count_terms) or (
            ("order" in q_lower or "orders" in q_lower) and any(w in q_lower for w in ["number", "count", "most", "highest", "fewest", "lowest", "top", "bottom", "each", "by", "per"])
        )
    )

    if is_order_count_query:
        # Priority for Order ID Resolution:
        # 1. Exact case-insensitive 'Order ID'
        exact_order_id = next((c for c in cols if c.lower() == "order id"), None)
        if exact_order_id:
            resolved_metric = exact_order_id
        else:
            # 2. Normalized order id variants
            norm_variants = ["order_id", "orderid", "orderno", "order_no", "ordernumber", "order_number", "ordernum", "order_num"]
            norm_order_id = next((c for c in cols if c.lower() in norm_variants), None)
            if norm_order_id:
                resolved_metric = norm_order_id
            else:
                # 3. Contains 'order' AND ('id', 'no', 'num', 'number', '#'), EXCLUDING customer, product, employee, vendor, transaction, invoice
                exclude_keywords = ["customer", "product", "employee", "user", "client", "vendor", "transaction", "invoice"]
                order_id_cand = next(
                    (c for c in cols if "order" in c.lower() and any(kw in c.lower() for kw in ["id", "no", "num", "number", "#"]) and not any(ex in c.lower() for ex in exclude_keywords)),
                    None
                )
                if order_id_cand:
                    resolved_metric = order_id_cand
                else:
                    # 4. Generic 'ID' / 'id' ONLY as last resort if no other order col exists AND no specific other entity ID exists
                    has_other_entity_ids = any(any(ex in c.lower() for ex in exclude_keywords) for c in cols)
                    if not has_other_entity_ids:
                        generic_id = next((c for c in cols if c.lower() in ("id", "row_id", "row id")), None)
                        if generic_id:
                            resolved_metric = generic_id

        if not resolved_metric:
            # Fallback to any column containing 'order'
            resolved_metric = next((c for c in cols if "order" in c.lower()), None)
            if not resolved_metric:
                # If no order column exists, select first column that is NOT an excluded identifier
                exclude_keywords = ["customer", "product", "employee", "user", "client", "vendor", "transaction", "invoice"]
                resolved_metric = next(
                    (c for c in cols if not any(ex in c.lower() for ex in exclude_keywords)),
                    None
                )

        if any(w in q_lower for w in ["distinct", "unique"]):
            agg = "nunique"
        else:
            agg = "count"

    else:
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

        if any(k in q_lower for k in ["average", "mean"]):
            agg = "mean"
        else:
            agg = "sum"

    # 3. Sort Direction & Limit & Ranking Intent Distinctions
    is_ranking = any(k in q_lower for k in ["highest", "worst", "top", "bottom", "most", "least", "best", "rank", "peak", "lowest", "fewest"])
    if any(k in q_lower for k in ["lowest", "worst", "least", "bottom", "fewest"]):
        sort_dir = "ascending"
    else:
        sort_dir = "descending"

    # For pure temporal breakdown without explicit ranking (e.g. "orders each month"), default to ascending (chronological)
    if time_grain and not is_ranking and not any(k in q_lower for k in ["lowest", "least", "fewest"]):
        sort_dir = "ascending"

    limit_val = None
    lim_match = re.search(r"(?:top|first|highest|lowest|bottom|best|worst)\s*(\d+)", q_lower)
    if lim_match:
        limit_val = int(lim_match.group(1))
    elif is_ranking and any(re.search(rf"\b{kw}\b", q_lower) for kw in ["which month", "which year", "which quarter", "which state", "which city", "which category"]):
        limit_val = 1

    return {
        "dimension": resolved_dim,
        "requested_dim_term": requested_dim_term,
        "metric": resolved_metric,
        "aggregation": agg,
        "sort_direction": sort_dir,
        "limit": limit_val,
        "time_grain": time_grain,
        "is_ranking": is_ranking,
        "is_exploratory": is_exploratory,
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

    intent = resolve_query_intent(query, target_df)
    if intent.get("is_exploratory"):
        cand_steps, overall_score, summary_findings, cand_scores = select_best_exploratory_candidates(target_df, max_candidates=3)
        if cand_steps and overall_score >= 0.45:
            return AnalysisPlan(
                analysis_goal=f"Execute exploratory analysis (score: {overall_score}) for query: '{query}'",
                steps=cand_steps,
                recommended_visualization=True,
            )
        else:
            steps.append(AnalysisPlanStep(
                tool_name="aggregate_data",
                params={"aggregations": {cols[0]: ["count"]}},
                purpose="No strong empirical pattern found."
            ))
            return AnalysisPlan(
                analysis_goal="No strong empirical pattern was found in the available data.",
                steps=steps,
                recommended_visualization=False,
            )

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

    # Trend Query Pattern (e.g. "trend", "growth", "over time")
    if any(k in q_lower for k in ["trend", "trends", "over time", "growth"]) and (date_cols or len(cols) >= 2):
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
    elif any(k in q_lower for k in ["top", "best", "worst", "highest", "lowest", "total", "average", "mean", "sum", "by ", "group", "each", "per", "monthly", "weekly", "yearly", "quarterly", "daily"]) or len(cols) >= 2:
        intent = resolve_query_intent(query, target_df)
        g_col = intent["dimension"]
        v_col = intent["metric"]

        if not g_col and intent["requested_dim_term"]:
            # User requested a dimension (e.g. "states") that does not exist in dataset.
            # Do NOT fall back to arbitrary columns.
            pass
        elif g_col and v_col:
            aggs = {v_col: [intent["aggregation"]]}
            if intent["time_grain"] and not intent["is_ranking"]:
                s_by = g_col
                s_dir = "ascending"
            else:
                s_by = f"{v_col}_{intent['aggregation']}"
                s_dir = intent["sort_direction"]

            group_params: Dict[str, Any] = {
                "group_by": [g_col],
                "aggregations": aggs,
                "sort_by": s_by,
                "sort_direction": s_dir,
            }
            if intent["time_grain"]:
                group_params["time_grain"] = intent["time_grain"]
            if intent["limit"]:
                group_params["limit"] = intent["limit"]

            steps.append(AnalysisPlanStep(
                tool_name="group_data",
                params=group_params,
                purpose=f"Group by {g_col} ({intent['time_grain'] or 'raw'}) and calculate {intent['aggregation']} for {v_col}"
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

            if intent.get("time_grain") and not params.get("time_grain"):
                params["time_grain"] = intent["time_grain"]

            # 1. Identifier protection check
            id_cols_used = [c for c in g_cols if c in df_cols and is_identifier_column(c, target_df[c])]
            if id_cols_used and not intent["is_explicit_id_request"]:
                warnings.append(
                    f"Step {idx+1} ({t_name}) validation error: Cannot group by identifier column(s) {id_cols_used} unless explicitly requested by user."
                )
                if intent["dimension"] and intent["metric"]:
                    g_params = {
                        "group_by": [intent["dimension"]],
                        "aggregations": {intent["metric"]: [intent["aggregation"]]},
                        "sort_by": f"{intent['metric']}_{intent['aggregation']}",
                        "sort_direction": intent["sort_direction"],
                        "limit": intent["limit"],
                    }
                    if intent["time_grain"]:
                        g_params["time_grain"] = intent["time_grain"]
                    step = AnalysisPlanStep(
                        tool_name="group_data",
                        params=g_params,
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
                    g_params = {
                        "group_by": [intent["dimension"]],
                        "aggregations": {intent["metric"]: [intent["aggregation"]]} if intent["metric"] else params.get("aggregations", {}),
                        "sort_by": f"{intent['metric']}_{intent['aggregation']}" if intent["metric"] else params.get("sort_by"),
                        "sort_direction": intent["sort_direction"],
                        "limit": intent["limit"],
                    }
                    if intent["time_grain"]:
                        g_params["time_grain"] = intent["time_grain"]
                    step = AnalysisPlanStep(
                        tool_name="group_data",
                        params=g_params,
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

    # Validation Rule: Ranking or Temporal questions MUST contain group_data step
    query_goal = plan.analysis_goal or ""
    intent = resolve_query_intent(query_goal, target_df)

    has_valid_temporal_analysis = any(
        (s.tool_name == "group_data" and intent["dimension"] in s.params.get("group_by", []))
        or (s.tool_name == "analyze_trend")
        for s in valid_steps
    )
    is_ranking_dimension_query = (
        bool(intent["dimension"]) and 
        bool(intent["requested_dim_term"]) and
        any(kw in query_goal.lower() for kw in ["which", "highest", "lowest", "top", "bottom", "most", "least", "best", "worst", "rank"])
    )
    is_temporal_grouping_query = bool(intent["time_grain"]) and bool(intent["dimension"])

    if (is_ranking_dimension_query or is_temporal_grouping_query) and not has_valid_temporal_analysis:
        warnings.append(
            f"Plan validation error: Query '{query_goal}' requires grouping on '{intent['dimension']}' but plan lacks a valid group_data step. Re-planning with group_data."
        )
        g_col = intent["dimension"]
        v_col = intent["metric"] or list(target_df.columns)[0]
        agg_func = intent["aggregation"]
        s_by = f"{v_col}_{agg_func}"
        s_dir = intent["sort_direction"]
        
        group_params = {
            "group_by": [g_col],
            "aggregations": {v_col: [agg_func]},
            "sort_by": s_by,
            "sort_direction": s_dir,
        }
        if intent["time_grain"]:
            group_params["time_grain"] = intent["time_grain"]
        if intent["limit"]:
            group_params["limit"] = intent["limit"]

        replanned_step = AnalysisPlanStep(
            tool_name="group_data",
            params=group_params,
            purpose=f"Group by {g_col} ({intent['time_grain'] or 'raw'}) and calculate {agg_func} for {v_col}",
        )
        valid_steps = [replanned_step]

    validated_plan = AnalysisPlan(
        analysis_goal=plan.analysis_goal,
        steps=valid_steps,
        recommended_visualization=plan.recommended_visualization or is_ranking_dimension_query or is_temporal_grouping_query,
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
                t_grain = params.get("time_grain")
                res = group_data(
                    df=current_df,
                    group_by=g_cols,
                    aggregations=aggs,
                    sort_by=s_by,
                    sort_direction=s_dir,
                    limit=lim,
                    time_grain=t_grain,
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
                target_k = sort_b if (sort_b and sort_b in metric_keys) else (metric_keys[0] if metric_keys else None)

                is_temporal = any(kw in g_cols.lower() for kw in ["date", "month", "year", "quarter", "week", "day"]) or (res.get("time_grain") is not None)
                is_ranking = res.get("is_ranking", False) or (lim is not None and lim > 0) or any(kw in sort_dir.lower() for kw in ["desc", "descending"])

                if is_temporal and not is_ranking and target_k and g_cols_list:
                    primary_g = g_cols_list[0]
                    period_strs = [
                        f"{r.get(primary_g, 'N/A')}: {r.get(target_k)}"
                        for r in rows
                    ]
                    findings.append(
                        f"Temporal breakdown grouped by '{g_cols}' ({row_cnt} periods): {', '.join(period_strs)}."
                    )
                elif target_k and g_cols_list:
                    primary_g = g_cols_list[0]
                    is_rev = sort_dir.lower() in ["descending", "desc"]
                    sorted_rows = sorted(
                        rows,
                        key=lambda x: (x.get(target_k) if x.get(target_k) is not None else -float('inf')),
                        reverse=is_rev,
                    )
                    top_limit = lim if (lim and lim > 0) else 5
                    top_rows = sorted_rows[:top_limit]
                    is_exploratory_mode = quantitative_results.get("exploratory_metadata", {}).get("is_exploratory", False)

                    if is_exploratory_mode:
                        top_item = top_rows[0]
                        top_seg = top_item.get(primary_g, "N/A")
                        top_val = top_item.get(target_k)
                        if is_temporal:
                            try:
                                formatted_period = pd.to_datetime(str(top_seg), format="%Y-%m").strftime("%B %Y")
                            except Exception:
                                formatted_period = str(top_seg)
                            metric_label = "monthly order volume" if "order" in str(target_k).lower() else f"monthly '{target_k}'"
                            unit_label = "orders" if "order" in str(target_k).lower() else "entries"
                            findings.append(f"{formatted_period} recorded the highest {metric_label}, with {top_val} {unit_label}.")
                        else:
                            findings.append(f"Top segment in '{g_cols}' is '{top_seg}' with {top_val} recorded entries.")
                    else:
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

        elif "detect_anomalies" in step_key:
            col = res.get("column", "")
            cnt = res.get("anomaly_count", 0)
            findings.append(f"Anomaly detection on '{col}': detected {cnt} statistical outliers.")


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

    is_exploratory_mode = quantitative_results.get("exploratory_metadata", {}).get("is_exploratory", False)
    if is_exploratory_mode:
        real_findings = [f for f in findings if not f.startswith("Execution warnings:")]
        warn_findings = [f for f in findings if f.startswith("Execution warnings:")]

        if not real_findings:
            full_str = "No strong empirical pattern was found in the available data."
            warn_findings.append("Candidate note: 0 strong empirical pattern(s) were discovered meeting significance threshold.")
        else:
            numbered = [f"{idx+1}. {item}" for idx, item in enumerate(real_findings)]
            if len(real_findings) < 3:
                warn_findings.append(f"Candidate note: {len(real_findings)} strong empirical pattern(s) were discovered meeting significance threshold.")
            full_str = " ".join(numbered)

        if errors or warn_findings:
            all_warns = errors + [w for w in warn_findings if w not in errors]
            full_str += f" Execution warnings: {'; '.join(all_warns)}"
        return full_str

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


def _execute_secondary_drilldown(
    primary_step: Optional[AnalysisPlanStep],
    quantitative_results: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Deterministically plans and executes secondary segment drill-down on the top primary segment.

    Runs ONLY if a valid secondary dimension exists (2 <= nunique <= 50, obs >= 5),
    excluding identifier columns and the primary grouping column.
    Guaranteed fail-safe: returns status='skipped' or status='success', never raises error.
    """
    from app.data.metadata import is_identifier_column

    try:
        if not primary_step or primary_step.tool_name != "group_data":
            return {"status": "skipped", "reason": "Primary step is not group_data."}

        step1_res = quantitative_results.get("step_1_group_data", {})
        rows = step1_res.get("rows", [])
        if not rows:
            return {"status": "skipped", "reason": "No primary group rows produced."}

        g_cols = primary_step.params.get("group_by", [])
        if not g_cols:
            return {"status": "skipped", "reason": "Primary step has no group_by columns."}

        primary_dim = g_cols[0]
        top_row = rows[0]
        top_seg_val = top_row.get(primary_dim)
        if top_seg_val is None:
            for k, v in top_row.items():
                if k.lower() == primary_dim.lower():
                    top_seg_val = v
                    break
        if top_seg_val is None:
            first_k = list(top_row.keys())[0] if top_row else None
            if first_k:
                top_seg_val = top_row[first_k]

        if top_seg_val is None or str(top_seg_val).strip() == "":
            return {"status": "skipped", "reason": "Top primary segment value is empty."}

        if primary_step.params.get("time_grain") == "month":
            parsed_dates = pd.to_datetime(df[primary_dim], format="mixed", dayfirst=True, errors="coerce")
            filtered_df = df[parsed_dates.dt.strftime("%Y-%m") == str(top_seg_val)].copy()
        else:
            filtered_df = df[df[primary_dim] == top_seg_val].copy()

        if len(filtered_df) < 3:
            return {"status": "skipped", "reason": "Insufficient observations in top primary segment."}

        cols = list(df.columns)
        dt_col, _ = resolve_datetime_column(df)
        cand_sec_dims = [
            c for c in cols
            if c != primary_dim 
            and c != dt_col 
            and not is_identifier_column(c, df[c]) 
            and (2 <= filtered_df[c].nunique() <= 50)
            and not any(kw in c.lower() for kw in ["date", "time", "year", "quarter", "month", "day"])
        ]

        if not cand_sec_dims:
            return {"status": "skipped", "reason": "No valid secondary dimension found."}

        sec_dim = cand_sec_dims[0]

        aggs = primary_step.params.get("aggregations", {})
        metric_col = list(aggs.keys())[0] if aggs else next((c for c in cols if c != sec_dim and not is_identifier_column(c, df[c])), cols[0])
        agg_func = list(aggs[metric_col])[0] if (aggs and metric_col in aggs) else "count"

        sec_res = group_data(
            df=filtered_df,
            group_by=[sec_dim],
            aggregations={metric_col: [agg_func]},
            sort_by=f"{metric_col}_{agg_func}",
            sort_direction="descending",
            limit=5,
        )

        return {
            "status": "success",
            "primary_dimension": primary_dim,
            "primary_segment": top_seg_val,
            "grouping_dimension": sec_dim,
            "metric": metric_col,
            "aggregation": agg_func,
            "rows": sec_res.get("rows", []),
            "result_row_count": sec_res.get("result_row_count", 0),
            "warnings": [],
        }

    except Exception as e:
        logger.warning(f"Secondary drill-down execution skipped due to error: {e}")
        return {"status": "skipped", "reason": f"Execution error: {str(e)}", "warnings": [str(e)]}


def run_analyst_agent(
    query: str,
    dataset_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    llm: Optional[BaseChatModel] = None,
) -> AnalystOutput:
    """Executes the Data Analyst Agent node using modular planner-validator-executor design."""
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

    # 4. Secondary Segment Drill-Down Stage (Fail-safe, separate post-execution stage)
    if status == "success" and validated_plan.steps:
        primary_step = validated_plan.steps[0]
        drilldown_res = _execute_secondary_drilldown(primary_step, quantitative_results, target_df)
        quantitative_results["step_2_drilldown"] = drilldown_res

    # Store exploratory metadata for confidence calculation and formatting if goal is exploratory
    if validated_plan.analysis_goal and "exploratory analysis" in validated_plan.analysis_goal.lower():
        score_match = re.search(r"score:\s*([\d\.]+)", validated_plan.analysis_goal)
        exp_score = float(score_match.group(1)) if score_match else 0.85
        quantitative_results["exploratory_metadata"] = {
            "is_exploratory": True,
            "overall_score": exp_score,
            "step_count": len(executed_calls),
            "status": status,
        }

    all_errors = validation_errors + execution_errors

    # 5. Determine Visualization
    requires_vis = determine_requires_visualization(
        query, validated_plan, quantitative_results, status
    )

    # 6. Generate Findings
    if "No strong empirical pattern" in (validated_plan.analysis_goal or ""):
        findings_text = "No strong empirical pattern was found in the available data."
    else:
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

