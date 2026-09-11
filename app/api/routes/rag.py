import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.rag import default_rag_store, default_rag_retriever

logger = logging.getLogger("datapilot.api.routes.rag")

router = APIRouter()


class KnowledgeIngestRequest(BaseModel):
    """Request schema for ingesting domain knowledge into RAG store."""
    title: str = Field(description="Document or glossary item title")
    content: str = Field(description="Body content or definition text")
    category: str = Field(
        default="general",
        description="Category: glossary, domain, column_def, general"
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional custom metadata")


class KnowledgeSearchRequest(BaseModel):
    """Request schema for querying the RAG knowledge index."""
    query: str = Field(description="Search query or analytical concept")
    category: Optional[str] = Field(default=None, description="Optional category filter")
    top_k: int = Field(default=3, ge=1, le=10, description="Max number of top relevance chunks to return")


@router.post(
    "/knowledge",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest domain knowledge document",
    description="Ingests, chunks, and indexes a domain document or business glossary item into the RAG engine.",
)
def ingest_knowledge_document(req: KnowledgeIngestRequest) -> Dict[str, Any]:
    """Ingests a new knowledge document into RAG index."""
    if not req.title or not req.title.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Title cannot be empty.")
    if not req.content or not req.content.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Content cannot be empty.")

    try:
        doc = default_rag_store.add_document(
            title=req.title,
            content=req.content,
            category=req.category,
            metadata=req.metadata,
        )
        return {
            "message": f"Successfully ingested knowledge document '{req.title}'.",
            "document": doc,
        }
    except Exception as e:
        logger.error(f"Error ingesting knowledge document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest knowledge document: {str(e)}",
        )


@router.get(
    "/knowledge",
    status_code=status.HTTP_200_OK,
    summary="List domain knowledge documents",
    description="Returns list of all ingested domain knowledge documents, optionally filtered by category.",
)
def list_knowledge_documents(category: Optional[str] = None) -> Dict[str, Any]:
    """Lists ingested knowledge documents."""
    docs = default_rag_store.list_documents(category=category)
    return {
        "count": len(docs),
        "documents": docs,
    }


@router.post(
    "/search",
    status_code=status.HTTP_200_OK,
    summary="Search RAG knowledge index",
    description="Performs hybrid BM25 keyword and vector similarity search across indexed domain knowledge chunks.",
)
def search_knowledge_index(req: KnowledgeSearchRequest) -> Dict[str, Any]:
    """Searches RAG knowledge index."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query string cannot be empty.")

    results = default_rag_retriever.search(
        query=req.query,
        category=req.category,
        top_k=req.top_k,
    )
    return {
        "query": req.query,
        "result_count": len(results),
        "results": results,
    }


@router.delete(
    "/knowledge/{doc_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete domain knowledge document",
    description="Deletes a knowledge document from the RAG store and updates the index.",
)
def delete_knowledge_document(doc_id: str) -> Dict[str, str]:
    """Deletes a knowledge document."""
    doc = default_rag_store.get_document(doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge document with ID '{doc_id}' not found.",
        )

    default_rag_store.delete_document(doc_id)
    return {"message": f"Knowledge document '{doc_id}' deleted successfully."}
