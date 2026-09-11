import logging
import uuid
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

from app.tools.hypothesis import generate_automated_hypotheses
from app.schemas.insights import AutomatedInsight

logger = logging.getLogger("datapilot.tools.insights_engine")


def discover_automated_insights(
    df: pd.DataFrame,
    dataset_id: Optional[str] = None,
    max_insights: int = 5,
    max_pairs: Optional[int] = None,
) -> Dict[str, Any]:
    """Discovers evidence-based business insights from Pareto distributions, drivers, and statistical hypotheses."""
    if df.empty:
        return {"insights": [], "warnings": ["DataFrame is empty (0 rows)."]}

    insights: List[Dict[str, Any]] = []
    warnings: List[str] = []

    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    from app.data.metadata import is_identifier_column

    # 1. Pareto 80/20 Concentration Risk & Driver Discovery
    for c_cat in cat_cols:
        if is_identifier_column(c_cat, df[c_cat]):
            continue
        for c_num in num_cols:
            if is_identifier_column(c_num, df[c_num]):
                continue
            if df[c_cat].nunique() >= 3 and len(df) >= 5:
                # Pareto concentration requires non-negative values and total > 0
                s_num = df[c_num].dropna()
                if (s_num < 0).any():
                    # Mixed-sign metric (e.g. Profit with losses) is invalid for Pareto concentration
                    continue

                grouped = df.groupby(c_cat)[c_num].sum().sort_values(ascending=False)
                total_val = grouped.sum()
                if total_val <= 0:
                    continue

                top_n = max(1, int(len(grouped) * 0.2))
                top_sum = grouped.iloc[:top_n].sum()
                share_pct = round((top_sum / total_val) * 100.0, 2)

                if 50.0 <= share_pct <= 100.0:
                    top_items = ", ".join([f"'{k}' ({v:.1f})" for k, v in grouped.iloc[:top_n].items()])
                    headline = f"Concentration Risk: Top {top_n} {c_cat} accounts for {share_pct}% of total {c_num}"
                    explanation = (
                        f"Statistical analysis reveals high concentration in '{c_num}' driven by top category '{c_cat}'. "
                        f"Specifically, top sub-segments [{top_items}] contribute {share_pct}% of total volume across {len(grouped)} total categories; "
                        f"this analysis highlights revenue/volume dependency without establishing causality."
                    )
                    insight_obj = AutomatedInsight(
                        insight_id=f"ins_{uuid.uuid4().hex[:8]}",
                        category="concentration",
                        headline=headline,
                        explanation=explanation,
                        evidence={
                            "metric": "Pareto Concentration",
                            "group_column": c_cat,
                            "value_column": c_num,
                            "top_segments_count": top_n,
                            "total_segments_count": len(grouped),
                            "top_segment_share_percentage": share_pct,
                            "total_value": float(total_val),
                        },
                        confidence_score=0.9,
                        limitations=["Assumes static categorization across historical rows."],
                    )
                    insights.append(insight_obj.model_dump())

    # 2. Statistical Hypotheses Findings Integration
    hypo_res = generate_automated_hypotheses(df=df, max_pairs=max_pairs or 10)
    hypotheses = hypo_res.get("hypotheses", [])

    for h in hypotheses[:3]:
        if h.get("statistical_significance"):
            headline = f"Significant Statistical Finding: {h['target_column']} and {h['group_column']}"
            explanation = (
                f"{h['statement']} The statistical proof ({h['test_name']}, adj. p={h['adjusted_p_value']:.4f}, "
                f"effect size={h['effect_size']} [{h['effect_size_interpretation']}]) confirms a robust empirical relationship."
            )
            insight_obj = AutomatedInsight(
                insight_id=f"ins_{uuid.uuid4().hex[:8]}",
                category="driver" if "T-Test" in h["test_name"] or "ANOVA" in h["test_name"] else "correlation",
                headline=headline,
                explanation=explanation,
                evidence={
                    "test_name": h["test_name"],
                    "target_column": h["target_column"],
                    "group_column": h["group_column"],
                    "statistic": h["statistic"],
                    "p_value": h["p_value"],
                    "adjusted_p_value": h["adjusted_p_value"],
                    "effect_size": h["effect_size"],
                    "effect_size_type": h["effect_size_type"],
                    "sample_size": h["sample_size"],
                },
                confidence_score=0.95 if h["adjusted_p_value"] < 0.01 else 0.85,
                limitations=["Statistical association does not imply operational causation."],
            )
            insights.append(insight_obj.model_dump())

    # Limit to max_insights
    return {
        "dataset_id": dataset_id,
        "insights": insights[:max_insights],
        "hypotheses_evaluated": len(hypotheses),
        "warnings": warnings,
    }
