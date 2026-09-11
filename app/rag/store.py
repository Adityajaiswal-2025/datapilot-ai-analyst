import json
import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from app.core.config import settings
from app.rag.chunker import DocumentChunker

logger = logging.getLogger("datapilot.rag.store")


class RAGStore:
    """Persistent storage engine for RAG knowledge documents and chunk indices."""

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        if not storage_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            storage_dir = os.path.join(base_dir, "data", "rag")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.index_file = os.path.join(self.storage_dir, "index.json")
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        self.documents: Dict[str, Dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Loads persistent document index from JSON storage."""
        if not os.path.exists(self.index_file):
            self.documents = {}
            return

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            logger.info(f"Loaded {len(self.documents)} domain documents into RAGStore index.")
        except Exception as e:
            logger.error(f"Failed to load RAG store index: {e}", exc_info=True)
            self.documents = {}

    def _save_index(self) -> None:
        """Saves current document index to JSON storage."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save RAG store index: {e}", exc_info=True)

    def add_document(
        self,
        title: str,
        content: str,
        category: str = "general",
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingests a new knowledge document, chunks it, and updates the persistent index."""
        if not title or not title.strip():
            raise ValueError("Document title cannot be empty.")
        if not content or not content.strip():
            raise ValueError("Document content cannot be empty.")

        document_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        extra_meta = metadata or {}

        # Generate chunks
        chunk_metadata = {
            "doc_id": document_id,
            "doc_title": title.strip(),
            "category": category.strip().lower(),
        }
        chunks = self.chunker.split_text(content, metadata=chunk_metadata)

        doc_record = {
            "doc_id": document_id,
            "title": title.strip(),
            "category": category.strip().lower(),
            "content": content.strip(),
            "chunk_count": len(chunks),
            "created_at": now_iso,
            "metadata": extra_meta,
            "chunks": chunks,
        }

        self.documents[document_id] = doc_record
        self._save_index()
        logger.info(f"Added document '{title}' ({document_id}) with {len(chunks)} chunks to RAGStore.")
        return doc_record

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a document by its ID."""
        return self.documents.get(doc_id)

    def list_documents(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists stored documents, optionally filtered by category."""
        docs = list(self.documents.values())
        if category:
            cat_lower = category.strip().lower()
            docs = [d for d in docs if d.get("category") == cat_lower]
        return docs

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document from the store and persists the updated index."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._save_index()
            logger.info(f"Deleted document {doc_id} from RAGStore.")
            return True
        return False

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Extracts all chunks across all ingested documents."""
        all_chunks = []
        for doc in self.documents.values():
            all_chunks.extend(doc.get("chunks", []))
        return all_chunks

    def clear(self) -> None:
        """Clears all documents from store and deletes persistent file."""
        self.documents = {}
        if os.path.exists(self.index_file):
            try:
                os.remove(self.index_file)
            except Exception:
                pass
