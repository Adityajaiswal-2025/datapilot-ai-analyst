import pytest
import os
import tempfile
from fastapi.testclient import TestClient

from app.main import app
from app.rag.chunker import DocumentChunker
from app.rag.store import RAGStore
from app.rag.retriever import HybridRetriever
from app.data.context import build_dataset_context
from app.schemas.dataset import DatasetProfile, DatasetQualityReport, ColumnProfile

client = TestClient(app)


def test_document_chunker_basic():
    """Test DocumentChunker splits long text into chunks."""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    text = (
        "Paragraph one is a long description about software architecture and domain knowledge. "
        "It contains multiple detailed sentences.\n\n"
        "Paragraph two describes financial metrics like MRR, ARR, and Net Revenue Retention. "
        "These terms are essential for enterprise analytics."
    )
    chunks = chunker.split_text(text, metadata={"test_id": "123"})
    assert len(chunks) >= 2
    assert all("chunk_index" in c for c in chunks)
    assert all("content" in c for c in chunks)
    assert all(c["test_id"] == "123" for c in chunks)


def test_rag_store_crud():
    """Test RAGStore add, list, get, delete operations and index persistence."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = RAGStore(storage_dir=tmp_dir, chunk_size=200)

        # Add document
        doc1 = store.add_document(
            title="SaaS Financial Glossary",
            content="MRR stands for Monthly Recurring Revenue. ARR stands for Annual Recurring Revenue.",
            category="glossary",
        )
        doc_id = doc1["doc_id"]
        assert doc_id in store.documents
        assert doc1["chunk_count"] >= 1

        # Retrieve document
        retrieved = store.get_document(doc_id)
        assert retrieved is not None
        assert retrieved["title"] == "SaaS Financial Glossary"

        # List documents
        docs = store.list_documents(category="glossary")
        assert len(docs) == 1
        assert docs[0]["doc_id"] == doc_id

        # Delete document
        deleted = store.delete_document(doc_id)
        assert deleted is True
        assert store.get_document(doc_id) is None


def test_hybrid_retriever_search():
    """Test HybridRetriever relevance scoring for keyword and semantic matches."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = RAGStore(storage_dir=tmp_dir)
        store.add_document(
            title="Churn Calculation Rule",
            content="Customer churn rate is calculated as the percentage of subscribers lost over a given period.",
            category="domain",
        )
        store.add_document(
            title="Revenue Metric Rule",
            content="Gross revenue is calculated before deducting operating expenses and cost of goods sold.",
            category="domain",
        )

        retriever = HybridRetriever(store=store)

        # Search for churn
        churn_results = retriever.search(query="how to calculate customer churn", top_k=2)
        assert len(churn_results) >= 1
        assert churn_results[0]["doc_title"] == "Churn Calculation Rule"
        assert churn_results[0]["relevance_score"] > 0.1

        # Search for revenue
        rev_results = retriever.search(query="gross revenue calculation expenses", top_k=2)
        assert len(rev_results) >= 1
        assert rev_results[0]["doc_title"] == "Revenue Metric Rule"


def test_rag_context_integration():
    """Test build_dataset_context appends RAG domain knowledge section."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        from app.rag import default_rag_store
        
        # Ingest test glossary doc
        doc = default_rag_store.add_document(
            title="Revenue Recognition Rule",
            content="Recognize recurring software revenue on a monthly linear basis.",
            category="glossary",
        )

        try:
            profile = DatasetProfile(
                dataset_id="test_rag_ds",
                filename="sales.csv",
                row_count=10,
                column_count=2,
                column_profiles=[
                    ColumnProfile(
                        name="Revenue",
                        raw_dtype="float64",
                        classified_type="numeric",
                        null_count=0,
                        null_percentage=0.0,
                        unique_count=10,
                        unique_percentage=100.0,
                        is_constant=False,
                    )
                ],
                quality_report=DatasetQualityReport(
                    total_cells=20,
                    missing_cells=0,
                    total_rows=10,
                    completeness_percentage=100.0,
                    duplicate_rows_count=0,
                    duplicate_rows_percentage=0.0,
                    quality_score=100.0,
                    warnings=[],
                ),
            )

            context_str = build_dataset_context(
                profile=profile,
                query="What is the total revenue?",
                include_rag_context=True,
            )

            assert "DOMAIN KNOWLEDGE & GLOSSARY (RAG)" in context_str
            assert "Revenue Recognition Rule" in context_str
        finally:
            default_rag_store.delete_document(doc["doc_id"])


def test_rag_api_endpoints():
    """Test REST API endpoints /api/v1/rag/* for knowledge ingestion, listing, search, and deletion."""
    # 1. Ingest document via API
    ingest_payload = {
        "title": "API Test Glossary Term",
        "content": "Net Retention Rate (NRR) measures percentage of recurring revenue retained from existing customers.",
        "category": "glossary",
    }
    res_ingest = client.post("/api/v1/rag/knowledge", json=ingest_payload)
    assert res_ingest.status_code == 201
    data_ingest = res_ingest.json()
    assert "document" in data_ingest
    doc_id = data_ingest["document"]["doc_id"]

    try:
        # 2. List documents via API
        res_list = client.get("/api/v1/rag/knowledge?category=glossary")
        assert res_list.status_code == 200
        data_list = res_list.json()
        assert data_list["count"] >= 1
        assert any(d["doc_id"] == doc_id for d in data_list["documents"])

        # 3. Search RAG index via API
        search_payload = {
            "query": "Net Retention Rate NRR calculation",
            "top_k": 2,
        }
        res_search = client.post("/api/v1/rag/search", json=search_payload)
        assert res_search.status_code == 200
        data_search = res_search.json()
        assert data_search["result_count"] >= 1
        assert data_search["results"][0]["doc_title"] == "API Test Glossary Term"

    finally:
        # 4. Delete document via API
        res_del = client.delete(f"/api/v1/rag/knowledge/{doc_id}")
        assert res_del.status_code == 200
        assert "deleted successfully" in res_del.json()["message"]
