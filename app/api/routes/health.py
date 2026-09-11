from fastapi import APIRouter
from app.schemas.response import HealthCheckResponse
from app.core.config import settings

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Service Health Check",
    description="Returns current health status of the DataPilot application and underlying services.",
)
async def check_health() -> HealthCheckResponse:
    """Returns the operational status of the service."""
    return HealthCheckResponse(
        success=True,
        message="DataPilot service is running normally.",
        version=settings.VERSION,
        environment="development" if settings.DEBUG else "production",
        status="healthy",
        services={
            "api": "operational",
            "database": "not_configured",
            "agent_orchestrator": "ready",
        },
    )
