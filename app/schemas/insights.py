"""Pydantic schemas for automated insights and statistical hypothesis testing."""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from app.schemas.response import BaseResponse


class NormalityTestResult(BaseModel):
    """Result model for column distribution normality testing."""
    test_name: str = Field(description="Name of statistical normality test (e.g. Shapiro-Wilk, D'Agostino-Pearson)")
    statistic: float = Field(description="Test statistic value")
    p_value: float = Field(description="Raw p-value derived from normality test")
    sample_size: int = Field(description="Number of non-null observations tested")
    is_normal: bool = Field(description="Whether column distribution satisfies normality (p > 0.05)")
    interpretation: str = Field(description="Human-readable interpretation of distribution normality")


class HypothesisResult(BaseModel):
    """Structured result contract for a statistical hypothesis test."""
    hypothesis_id: str = Field(description="Unique identifier for the tested hypothesis")
    test_name: str = Field(description="Statistical test name (e.g. Independent T-Test, One-Way ANOVA, Chi-Square, Pearson Correlation)")
    target_column: str = Field(description="Primary numeric or categorical column analyzed")
    group_column: Optional[str] = Field(default=None, description="Grouping column if applicable")
    statistic: float = Field(description="Computed test statistic value")
    p_value: float = Field(description="Raw p-value derived from statistical test")
    adjusted_p_value: float = Field(description="Multiple-testing adjusted p-value (Benjamini-Hochberg FDR)")
    effect_size: float = Field(description="Computed magnitude of effect (Cohen's d, Eta-squared, Cramér's V, Correlation coefficient)")
    effect_size_type: str = Field(description="Type of effect size metric (e.g. Cohen's d, Eta-squared, Cramér's V, Pearson r)")
    effect_size_interpretation: str = Field(description="Qualitative effect size rating (Negligible, Small, Medium, Large)")
    sample_size: int = Field(description="Number of valid records evaluated in test")
    statistical_significance: bool = Field(description="True if adjusted p-value < 0.05")
    practical_relevance: str = Field(description="Business context description of finding magnitude")
    statement: str = Field(description="Formal statistical hypothesis finding statement using associative phrasing")
    warnings: List[str] = Field(default_factory=list, description="Warnings or assumption violation notices")


class AutomatedInsight(BaseModel):
    """Structured business insight with underlying statistical evidence."""
    insight_id: str = Field(description="Unique identifier for insight")
    category: Literal["driver", "anomaly", "concentration", "correlation", "trend", "distribution"] = Field(
        description="Category of business insight"
    )
    headline: str = Field(description="Concise executive headline")
    explanation: str = Field(description="Detailed explanation adhering strictly to associative, non-causal language")
    evidence: Dict[str, Any] = Field(description="Underlying deterministic numerical metrics and statistical test proof")
    confidence_score: float = Field(description="Composite confidence score between 0.0 and 1.0")
    limitations: List[str] = Field(default_factory=list, description="Key assumptions, limitations, or non-causal disclaimers")


class AutoInsightsRequest(BaseModel):
    """Request schema for POST /api/v1/insights/auto."""
    dataset_id: str = Field(description="ID of registered dataset to analyze")
    max_hypotheses: int = Field(default=20, ge=1, le=50, description="Max number of candidate hypotheses to scan and rank")
    fdr_alpha: float = Field(default=0.05, ge=0.01, le=0.20, description="False Discovery Rate alpha threshold for Benjamini-Hochberg correction")


class AutoInsightsResponse(BaseResponse):
    """Response schema for POST /api/v1/insights/auto."""
    dataset_id: str = Field(description="ID of analyzed dataset")
    total_hypotheses_tested: int = Field(description="Total candidate hypotheses evaluated")
    significant_hypotheses_count: int = Field(description="Count of hypotheses achieving adjusted p < fdr_alpha")
    hypotheses: List[HypothesisResult] = Field(default_factory=list, description="Ranked list of evaluated statistical hypotheses")
    insights: List[AutomatedInsight] = Field(default_factory=list, description="Discovered evidence-based business insights")
    warnings: List[str] = Field(default_factory=list, description="Quality, variance, or sample size warnings")


class HypothesisTestRequest(BaseModel):
    """Request schema for POST /api/v1/insights/hypothesis."""
    dataset_id: str = Field(description="ID of registered dataset")
    test_type: Literal["numeric_difference", "categorical_association", "correlation", "normality"] = Field(
        description="Type of targeted hypothesis test to run"
    )
    primary_column: str = Field(description="Primary target column name")
    secondary_column: Optional[str] = Field(default=None, description="Secondary column name for difference or correlation tests")
    method: Optional[str] = Field(default="pearson", description="Correlation method: pearson or spearman")


class HypothesisTestResponse(BaseResponse):
    """Response schema for POST /api/v1/insights/hypothesis."""
    dataset_id: str = Field(description="ID of analyzed dataset")
    hypothesis_result: Optional[HypothesisResult] = Field(default=None, description="Result of requested hypothesis test")
    normality_result: Optional[NormalityTestResult] = Field(default=None, description="Result if normality test was requested")
    warnings: List[str] = Field(default_factory=list, description="Execution warnings")
