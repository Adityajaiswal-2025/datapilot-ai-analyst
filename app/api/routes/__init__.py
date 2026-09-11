from fastapi import APIRouter
from . import health, upload, datasets, analysis, agents, rag, insights

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(upload.router, prefix="/upload", tags=["Upload"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(analysis.router, prefix="/analyze", tags=["Analysis"])
api_router.include_router(agents.router, tags=["Agent Workflow Execution"])
api_router.include_router(rag.router, prefix="/rag", tags=["RAG Knowledge Engine"])
api_router.include_router(insights.router, prefix="/insights", tags=["Automated Insights & Hypotheses"])


