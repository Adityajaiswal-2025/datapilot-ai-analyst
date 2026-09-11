import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status

from app.tools.analysis import _resolve_dataframe, AnalysisToolError
from app.tools.hypothesis import (
    test_normality,
    test_numeric_difference,
    test_categorical_association,
    test_correlation_significance,
    generate_automated_hypotheses,
    HypothesisError,
)
from app.tools.insights_engine import discover_automated_insights
from app.schemas.insights import (
    AutoInsightsRequest,
    AutoInsightsResponse,
    HypothesisTestRequest,
    HypothesisTestResponse,
    HypothesisResult,
    AutomatedInsight,
    NormalityTestResult,
)

logger = logging.getLogger("datapilot.api.routes.insights")

router = APIRouter()


@router.post(
    "/auto",
    response_model=AutoInsightsResponse,
    status_code=status.HTTP_200_OK,
    summary="Discover automated statistical insights & test hypotheses",
    description="Scans column distributions, formulates hypotheses, applies Benjamini-Hochberg FDR correction, and generates evidence-based non-causal insights.",
)
def discover_auto_insights_endpoint(req: AutoInsightsRequest) -> AutoInsightsResponse:
    """Discovers automated insights and statistical hypotheses for a dataset."""
    try:
        df = _resolve_dataframe(req.dataset_id, None)
    except AnalysisToolError as e:
        logger.warning(f"Dataset resolution failed for insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    try:
        # Run hypothesis scanning and insight discovery engine
        hyp_res = generate_automated_hypotheses(
            df=df,
            max_pairs=req.max_hypotheses,
            fdr_alpha=req.fdr_alpha,
        )
        insight_res = discover_automated_insights(
            df=df,
            max_pairs=req.max_hypotheses,
        )

        hypotheses_models = [HypothesisResult(**h) for h in hyp_res.get("hypotheses", [])]
        insight_models = [AutomatedInsight(**i) for i in insight_res.get("insights", [])]

        all_warnings = hyp_res.get("warnings", []) + insight_res.get("warnings", [])

        return AutoInsightsResponse(
            success=True,
            message=f"Successfully analyzed dataset '{req.dataset_id}' and discovered {len(insight_models)} insights across {len(hypotheses_models)} tested hypotheses.",
            dataset_id=req.dataset_id,
            total_hypotheses_tested=hyp_res.get("total_tested", 0),
            significant_hypotheses_count=hyp_res.get("significant_count", 0),
            hypotheses=hypotheses_models,
            insights=insight_models,
            warnings=all_warnings,
        )
    except Exception as e:
        logger.error(f"Unexpected error during automated insight discovery: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Automated insight discovery error: {str(e)}",
        )


@router.post(
    "/hypothesis",
    response_model=HypothesisTestResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute targeted statistical hypothesis test",
    description="Executes a specific statistical test (T-Test/ANOVA, Chi-Square, Pearson/Spearman correlation, or Normality check) with exact p-value and effect size.",
)
def execute_hypothesis_test_endpoint(req: HypothesisTestRequest) -> HypothesisTestResponse:
    """Executes a targeted hypothesis test or normality check on a dataset."""
    try:
        df = _resolve_dataframe(req.dataset_id, None)
    except AnalysisToolError as e:
        logger.warning(f"Dataset resolution failed for hypothesis test: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    try:
        hyp_model = None
        norm_model = None
        warnings = []

        if req.test_type == "normality":
            if req.primary_column not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{req.primary_column}' not found in dataset.",
                )
            norm_res = test_normality(df[req.primary_column])
            norm_model = NormalityTestResult(**norm_res)

        elif req.test_type == "numeric_difference":
            if not req.secondary_column:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Secondary column (group column) is required for numeric difference tests.",
                )
            res = test_numeric_difference(df, group_col=req.secondary_column, value_col=req.primary_column)
            hyp_model = HypothesisResult(**res)

        elif req.test_type == "categorical_association":
            if not req.secondary_column:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Secondary column is required for categorical association tests.",
                )
            res = test_categorical_association(df, col1=req.primary_column, col2=req.secondary_column)
            hyp_model = HypothesisResult(**res)

        elif req.test_type == "correlation":
            if not req.secondary_column:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Secondary column is required for correlation tests.",
                )
            res = test_correlation_significance(
                df,
                col1=req.primary_column,
                col2=req.secondary_column,
                method=req.method or "pearson",
            )
            hyp_model = HypothesisResult(**res)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported test type '{req.test_type}'.",
            )

        return HypothesisTestResponse(
            success=True,
            message=f"Successfully executed hypothesis test '{req.test_type}' on dataset '{req.dataset_id}'.",
            dataset_id=req.dataset_id,
            hypothesis_result=hyp_model,
            normality_result=norm_model,
            warnings=warnings,
        )

    except HypothesisError as e:
        logger.warning(f"Hypothesis test validation failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Hypothesis test failure: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during hypothesis execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hypothesis execution internal error: {str(e)}",
        )
