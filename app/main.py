from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.security import setup_security
from app.api.routes import api_router

# Initialize structured logging
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan context manager for application startup and shutdown lifecycle.

    Note: Production database migrations must be executed out-of-band via Alembic:
        alembic upgrade head
    """
    yield


# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    description="DataPilot — Modular AI Data Analyst Agent API Platform",
    lifespan=lifespan,
)


# Apply CORS & Security configurations
setup_security(app)

# Include API V1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", summary="Root Health Endpoint")
async def root():
    """Root endpoint for sanity check."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
    }
