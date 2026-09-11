import logging
import math
import uuid
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from scipy import stats

from app.tools.analysis import _resolve_dataframe, AnalysisToolError
from app.schemas.insights import HypothesisResult, NormalityTestResult

logger = logging.getLogger("datapilot.tools.hypothesis")


class HypothesisError(Exception):
    """Custom exception raised when statistical hypothesis testing fails."""
    pass


def benjamini_hochberg_fdr(p_values: List[float]) -> List[float]:
    """Calculates Benjamini-Hochberg False Discovery Rate (FDR) adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    if m == 1:
        return [min(1.0, max(0.0, p_values[0]))]

    # Sort p-values while preserving original indices
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * m

    # Step-up adjustment
    cum_min = 1.0
    for i in range(m - 1, -1, -1):
        orig_idx, p_val = indexed_p[i]
        rank = i + 1
        adj_p = (p_val * m) / rank
        cum_min = min(cum_min, adj_p)
        adjusted[orig_idx] = round(min(1.0, max(0.0, cum_min)), 6)

    return adjusted


def test_normality(series: pd.Series) -> Dict[str, Any]:
    """Evaluates column distribution normality using Shapiro-Wilk or D'Agostino-Pearson tests."""
    s_clean = series.dropna()
    n = len(s_clean)

    if n < 8:
        return {
            "test_name": "Normality Check (Sample Size Limit)",
            "statistic": 0.0,
            "p_value": 1.0,
            "sample_size": n,
            "is_normal": True,
            "interpretation": f"Sample size (N={n}) is too small for reliable normality testing. Assuming parametric distribution.",
        }

    # Variance check
    if s_clean.std() == 0 or s_clean.nunique() == 1:
        return {
            "test_name": "Normality Check (Zero Variance)",
            "statistic": 0.0,
            "p_value": 0.0,
            "sample_size": n,
            "is_normal": False,
            "interpretation": "Column has zero variance (constant value). Distribution is non-normal.",
        }

    try:
        if n <= 5000:
            stat, p_val = stats.shapiro(s_clean)
            test_name = "Shapiro-Wilk Normality Test"
        else:
            stat, p_val = stats.normaltest(s_clean)
            test_name = "D'Agostino-Pearson Normality Test"

        p_val = float(p_val)
        is_normal = p_val > 0.05
        interp = (
            f"Distribution appears normal (p={p_val:.4f} > 0.05)."
            if is_normal
            else f"Distribution deviates significantly from normal (p={p_val:.4f} <= 0.05)."
        )

        return {
            "test_name": test_name,
            "statistic": round(float(stat), 4),
            "p_value": round(p_val, 6),
            "sample_size": n,
            "is_normal": is_normal,
            "interpretation": interp,
        }
    except Exception as e:
        logger.warning(f"Normality test failed: {e}")
        return {
            "test_name": "Normality Check (Execution Warning)",
            "statistic": 0.0,
            "p_value": 1.0,
            "sample_size": n,
            "is_normal": True,
            "interpretation": f"Normality check encountered warning: {str(e)}.",
        }


def test_numeric_difference(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> Dict[str, Any]:
    """Tests for statistical differences in a numeric column across categorical groups (T-test / ANOVA / Mann-Whitney)."""
    if group_col not in df.columns or value_col not in df.columns:
        raise HypothesisError(f"Columns '{group_col}' or '{value_col}' not present in DataFrame.")

    clean_df = df[[group_col, value_col]].dropna()
    if clean_df.empty:
        raise HypothesisError("Zero valid non-null rows present for hypothesis test.")

    groups = clean_df.groupby(group_col)[value_col]
    group_names = list(groups.groups.keys())
    num_groups = len(group_names)

    if num_groups < 2:
        raise HypothesisError(f"Group column '{group_col}' contains fewer than 2 distinct categories ({num_groups}). Cannot perform group comparison.")

    group_data = [group.values for _, group in groups if len(group) >= 2]
    if len(group_data) < 2:
        raise HypothesisError(f"Fewer than 2 groups in '{group_col}' have sufficient sample size (>= 2).")

    total_n = sum(len(g) for g in group_data)
    warnings = []

    # Normality check across groups
    is_normal_all = all(test_normality(pd.Series(g))["is_normal"] for g in group_data[:4])

    if num_groups == 2:
        g1 = group_data[0]
        g2 = group_data[1]

        if is_normal_all:
            # Welch's T-test
            test_name = "Independent Two-Sample T-Test (Welch's)"
            stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)

            # Cohen's d effect size
            n1, n2 = len(g1), len(g2)
            s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
            s_pooled = math.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / max(1, n1 + n2 - 2))
            effect_size = abs(float(np.mean(g1) - np.mean(g2)) / s_pooled) if s_pooled > 0 else 0.0
            effect_type = "Cohen's d"
        else:
            # Mann-Whitney U test fallback
            test_name = "Mann-Whitney U Test (Non-Parametric)"
            stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            warnings.append("Non-normal distribution detected: used non-parametric Mann-Whitney U test.")

            # Rank biserial correlation / standardized effect
            n1, n2 = len(g1), len(g2)
            effect_size = abs(1.0 - (2.0 * float(stat)) / (n1 * n2)) if (n1 * n2) > 0 else 0.0
            effect_type = "Rank-Biserial r"

        # Interpretation
        if effect_size < 0.2:
            effect_interp = "Negligible"
        elif effect_size < 0.5:
            effect_interp = "Small"
        elif effect_size < 0.8:
            effect_interp = "Medium"
        else:
            effect_interp = "Large"

    else:
        # 3+ Groups
        if is_normal_all:
            test_name = "One-Way ANOVA"
            stat, p_val = stats.f_oneway(*group_data)

            # Eta-squared effect size: SS_between / SS_total
            all_vals = np.concatenate(group_data)
            grand_mean = np.mean(all_vals)
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            ss_between = sum(len(g) * ((np.mean(g) - grand_mean) ** 2) for g in group_data)
            effect_size = float(ss_between / ss_total) if ss_total > 0 else 0.0
            effect_type = "Eta-Squared (η²)"
        else:
            test_name = "Kruskal-Wallis H Test (Non-Parametric)"
            stat, p_val = stats.kruskal(*group_data)
            warnings.append("Non-normal distribution detected: used non-parametric Kruskal-Wallis test.")

            # H / (N - 1) epsilon squared
            effect_size = float(stat / (total_n - 1)) if (total_n - 1) > 0 else 0.0
            effect_type = "Epsilon-Squared (ε²)"

        # Interpretation for ANOVA / Kruskal effect sizes
        if effect_size < 0.01:
            effect_interp = "Negligible"
        elif effect_size < 0.06:
            effect_interp = "Small"
        elif effect_size < 0.14:
            effect_interp = "Medium"
        else:
            effect_interp = "Large"

    p_val = float(p_val) if not math.isnan(p_val) else 1.0
    stat = float(stat) if not math.isnan(stat) else 0.0

    statement = (
        f"Statistical analysis ({test_name}) indicates that values of '{value_col}' differ significantly across '{group_col}' categories "
        f"(statistic={stat:.2f}, p={p_val:.4f}, effect size {effect_type}={effect_size:.2f} [{effect_interp}])."
        if p_val < 0.05
        else f"No statistically significant difference in '{value_col}' was detected across '{group_col}' categories (p={p_val:.4f} >= 0.05)."
    )

    return {
        "hypothesis_id": f"hyp_{uuid.uuid4().hex[:8]}",
        "test_name": test_name,
        "target_column": value_col,
        "group_column": group_col,
        "statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "adjusted_p_value": round(p_val, 6),
        "effect_size": round(effect_size, 4),
        "effect_size_type": effect_type,
        "effect_size_interpretation": effect_interp,
        "sample_size": total_n,
        "statistical_significance": p_val < 0.05,
        "practical_relevance": f"{effect_interp} magnitude effect across {num_groups} '{group_col}' sub-groups.",
        "statement": statement,
        "warnings": warnings,
    }


def test_categorical_association(
    df: pd.DataFrame,
    col1: str,
    col2: str,
) -> Dict[str, Any]:
    """Conducts Chi-Square (χ²) test of independence between two categorical columns."""
    if col1 not in df.columns or col2 not in df.columns:
        raise HypothesisError(f"Columns '{col1}' or '{col2}' not present in DataFrame.")

    clean_df = df[[col1, col2]].dropna()
    n = len(clean_df)
    if n < 10:
        raise HypothesisError(f"Insufficient sample size (N={n} < 10) for Chi-Square test.")

    contingency = pd.crosstab(clean_df[col1], clean_df[col2])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        raise HypothesisError(f"Contingency table must be at least 2x2. Shape: {contingency.shape}")

    try:
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        chi2 = float(chi2)
        p_val = float(p_val)

        # Cramér's V effect size: sqrt(chi2 / (N * (min(r,c) - 1)))
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = math.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0

        if cramers_v < 0.1:
            effect_interp = "Negligible"
        elif cramers_v < 0.3:
            effect_interp = "Small"
        elif cramers_v < 0.5:
            effect_interp = "Medium"
        else:
            effect_interp = "Large"

        statement = (
            f"Chi-Square test of independence reveals a statistically significant association between '{col1}' and '{col2}' "
            f"(χ²={chi2:.2f}, df={dof}, p={p_val:.4f}, Cramér's V={cramers_v:.2f} [{effect_interp}])."
            if p_val < 0.05
            else f"No statistically significant association detected between '{col1}' and '{col2}' (χ²={chi2:.2f}, p={p_val:.4f} >= 0.05)."
        )

        return {
            "hypothesis_id": f"hyp_{uuid.uuid4().hex[:8]}",
            "test_name": "Chi-Square Test of Independence",
            "target_column": col1,
            "group_column": col2,
            "statistic": round(chi2, 4),
            "p_value": round(p_val, 6),
            "adjusted_p_value": round(p_val, 6),
            "effect_size": round(cramers_v, 4),
            "effect_size_type": "Cramér's V",
            "effect_size_interpretation": effect_interp,
            "sample_size": n,
            "statistical_significance": p_val < 0.05,
            "practical_relevance": f"{effect_interp} categorical association between '{col1}' and '{col2}'.",
            "statement": statement,
            "warnings": [],
        }
    except Exception as e:
        raise HypothesisError(f"Chi-Square test failed: {str(e)}") from e


def test_correlation_significance(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    method: str = "pearson",
) -> Dict[str, Any]:
    """Calculates Pearson or Spearman correlation coefficient and exact p-value."""
    if col1 not in df.columns or col2 not in df.columns:
        raise HypothesisError(f"Columns '{col1}' or '{col2}' not present in DataFrame.")

    s1 = pd.to_numeric(df[col1], errors="coerce")
    s2 = pd.to_numeric(df[col2], errors="coerce")
    clean_df = pd.DataFrame({col1: s1, col2: s2}).dropna()

    n = len(clean_df)
    if n < 5:
        raise HypothesisError(f"Insufficient sample size (N={n} < 5) for correlation test.")

    if clean_df[col1].std() == 0 or clean_df[col2].std() == 0:
        raise HypothesisError(f"Cannot calculate correlation: one or both columns have zero variance.")

    m_clean = method.lower().strip()
    try:
        if m_clean == "spearman":
            r_val, p_val = stats.spearmanr(clean_df[col1], clean_df[col2])
            test_name = "Spearman Rank Correlation"
        else:
            r_val, p_val = stats.pearsonr(clean_df[col1], clean_df[col2])
            test_name = "Pearson Linear Correlation"

        r_val = float(r_val)
        p_val = float(p_val)
        abs_r = abs(r_val)

        if abs_r < 0.2:
            effect_interp = "Negligible"
        elif abs_r < 0.5:
            effect_interp = "Moderate"
        elif abs_r < 0.8:
            effect_interp = "Strong"
        else:
            effect_interp = "Very Strong"

        # Non-causal phrasing mandate
        direction = "positive" if r_val > 0 else "negative"
        statement = (
            f"Correlation test ({test_name}) demonstrates a statistically significant {direction} association between '{col1}' and '{col2}' "
            f"(r={r_val:.2f}, p={p_val:.4f}, N={n}); this analysis does not establish causation."
            if p_val < 0.05
            else f"No statistically significant correlation detected between '{col1}' and '{col2}' (r={r_val:.2f}, p={p_val:.4f} >= 0.05)."
        )

        return {
            "hypothesis_id": f"hyp_{uuid.uuid4().hex[:8]}",
            "test_name": test_name,
            "target_column": col1,
            "group_column": col2,
            "statistic": round(r_val, 4),
            "p_value": round(p_val, 6),
            "adjusted_p_value": round(p_val, 6),
            "effect_size": round(r_val, 4),
            "effect_size_type": f"{method.capitalize()} r",
            "effect_size_interpretation": effect_interp,
            "sample_size": n,
            "statistical_significance": p_val < 0.05,
            "practical_relevance": f"{effect_interp} {direction} correlation; causation not implied.",
            "statement": statement,
            "warnings": [],
        }
    except Exception as e:
        raise HypothesisError(f"Correlation test failed: {str(e)}") from e


def generate_automated_hypotheses(
    df: pd.DataFrame,
    max_pairs: int = 50,
    fdr_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Scans column pairs, formulates hypotheses, applies Benjamini-Hochberg FDR correction, and ranks results."""
    if df.empty:
        return {"total_tested": 0, "significant_count": 0, "hypotheses": [], "warnings": ["DataFrame is empty (0 rows)."]}

    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)

    from app.data.metadata import is_identifier_column

    def _is_id_column(col_name: str) -> bool:
        return is_identifier_column(col_name, series=df[col_name] if col_name in df.columns else None)

    # Filter out ID-like, constant, and high-cardinality columns
    safe_num = []
    for c in num_cols:
        if _is_id_column(c):
            continue
        if df[c].std() == 0 or df[c].nunique() <= 1:
            continue
        safe_num.append(c)

    safe_cat = []
    for c in cat_cols:
        if _is_id_column(c):
            continue
        n_uniq = df[c].nunique()
        if n_uniq <= 1 or n_uniq > 30 or (n_uniq / len(df) > 0.5 and len(df) > 20):
            continue
        safe_cat.append(c)

    raw_results: List[Dict[str, Any]] = []
    pair_count = 0
    max_limit = min(50, max(5, max_pairs))

    # 1. Numeric vs Categorical (T-test / ANOVA)
    for num_c in safe_num[:8]:
        for cat_c in safe_cat[:5]:
            if pair_count >= max_limit:
                break
            try:
                res = test_numeric_difference(df, group_col=cat_c, value_col=num_c)
                raw_results.append(res)
                pair_count += 1
            except Exception:
                pass
        if pair_count >= max_limit:
            break

    # 2. Categorical vs Categorical (Chi-Square)
    if pair_count < max_limit:
        for i in range(len(safe_cat)):
            for j in range(i + 1, len(safe_cat)):
                if pair_count >= max_limit:
                    break
                try:
                    res = test_categorical_association(df, safe_cat[i], safe_cat[j])
                    raw_results.append(res)
                    pair_count += 1
                except Exception:
                    pass

    # 3. Numeric vs Numeric (Correlation)
    if pair_count < max_limit:
        for i in range(len(safe_num)):
            for j in range(i + 1, len(safe_num)):
                if pair_count >= max_limit:
                    break
                try:
                    res = test_correlation_significance(df, safe_num[i], safe_num[j])
                    raw_results.append(res)
                    pair_count += 1
                except Exception:
                    pass

    if not raw_results:
        return {
            "total_tested": 0,
            "significant_count": 0,
            "hypotheses": [],
            "warnings": ["No candidate column pairs met sample size and cardinality criteria for hypothesis testing."],
        }

    # Extract raw p-values and apply Benjamini-Hochberg FDR correction
    raw_p_vals = [r["p_value"] for r in raw_results]
    adj_p_vals = benjamini_hochberg_fdr(raw_p_vals)

    sig_count = 0
    final_hypotheses: List[Dict[str, Any]] = []

    for idx, r in enumerate(raw_results):
        adj_p = adj_p_vals[idx]
        is_sig = adj_p < fdr_alpha
        if is_sig:
            sig_count += 1

        r["adjusted_p_value"] = adj_p
        r["statistical_significance"] = is_sig

        final_hypotheses.append(r)

    # Rank findings primarily by adjusted p-value ascending
    final_hypotheses.sort(key=lambda x: (x["adjusted_p_value"], -abs(x["effect_size"])))

    return {
        "total_tested": len(final_hypotheses),
        "significant_count": sig_count,
        "hypotheses": final_hypotheses,
        "warnings": [],
    }


# Prevent pytest from collecting hypothesis tool functions as tests
test_normality.__test__ = False
test_numeric_difference.__test__ = False
test_categorical_association.__test__ = False
test_correlation_significance.__test__ = False

