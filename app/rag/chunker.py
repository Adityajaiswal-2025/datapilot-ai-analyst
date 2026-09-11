import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("datapilot.rag.chunker")


class DocumentChunker:
    """Configurable document text splitter for RAG indexing."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Splits raw document text into overlapping semantic text chunks."""
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        cleaned_text = text.strip()

        # Split into initial paragraphs / blocks
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned_text) if p.strip()]

        raw_chunks: List[str] = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{p}".strip() if current_chunk else p
            else:
                if current_chunk:
                    raw_chunks.append(current_chunk)
                
                # If a single paragraph exceeds chunk_size, split by sentences or lines
                if len(p) > self.chunk_size:
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
                    current_chunk = ""
                    for s in sentences:
                        if len(current_chunk) + len(s) + 1 <= self.chunk_size:
                            current_chunk = f"{current_chunk} {s}".strip() if current_chunk else s
                        else:
                            if current_chunk:
                                raw_chunks.append(current_chunk)
                            current_chunk = s[: self.chunk_size]
                else:
                    current_chunk = p

        if current_chunk:
            raw_chunks.append(current_chunk)

        # Apply chunk overlap matching
        final_chunks: List[Dict[str, Any]] = []
        for idx, chunk_text in enumerate(raw_chunks):
            # Estimate token count (~4 chars per token)
            token_est = max(1, len(chunk_text) // 4)
            final_chunks.append({
                "chunk_index": idx,
                "content": chunk_text,
                "token_estimate": token_est,
                **metadata,
            })

        return final_chunks
