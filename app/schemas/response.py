from typing import Any, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class BaseResponse(BaseModel):
    """Standardized API response wrapper."""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))



class HealthCheckResponse(BaseResponse):
    """Response model for system health check endpoint."""
    version: str
    environment: str
    status: str = "healthy"
    services: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseResponse):
    """Response model for error details."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Any] = None
