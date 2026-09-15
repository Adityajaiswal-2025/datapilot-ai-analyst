import asyncio
import logging
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.response import HealthCheckResponse
from app.core.config import settings
from app.database.session import get_db_session

logger = logging.getLogger("datapilot.api.routes.health")

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Service Liveness Check",
    description="Liveness probe returning HTTP 200 if the FastAPI application process is alive and responding.",
)
async def check_health() -> HealthCheckResponse:
    """Liveness probe checking if FastAPI process is operational."""
    return HealthCheckResponse(
        success=True,
        message="DataPilot service is running normally.",
        version=settings.VERSION,
        environment="development" if settings.DEBUG else "production",
        status="healthy",
        services={
            "api": "operational",
            "agent_orchestrator": "ready",
        },
    )


@router.get(
    "/health/readiness",
    response_model=HealthCheckResponse,
    summary="Service Readiness Check",
    description="Readiness probe executing SELECT 1 against the configured database to verify connectivity.",
)
async def check_readiness(
    db: AsyncSession = Depends(get_db_session),
) -> JSONResponse:
    """Readiness probe checking database connectivity and operational status."""
    try:
        await asyncio.wait_for(db.execute(text("SELECT 1")), timeout=3.0)
        payload = HealthCheckResponse(
            success=True,
            message="DataPilot service and database connection are ready.",
            version=settings.VERSION,
            environment="development" if settings.DEBUG else "production",
            status="ready",
            services={
                "api": "operational",
                "database": "connected",
                "agent_orchestrator": "ready",
            },
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=payload.model_dump(mode="json"),
        )
    except Exception as e:
        logger.error(f"Readiness check failed: Database connection error: {e}")
        payload = HealthCheckResponse(
            success=False,
            message="Database connection unavailable.",
            version=settings.VERSION,
            environment="development" if settings.DEBUG else "production",
            status="not_ready",
            services={
                "api": "operational",
                "database": "disconnected",
                "agent_orchestrator": "ready",
            },
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(mode="json"),
        )
