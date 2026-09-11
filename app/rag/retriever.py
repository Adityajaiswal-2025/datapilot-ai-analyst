import re
import math
import logging
from typing import Optional, List, Dict, Any

from app.rag.store import RAGStore

logger = logging.getLogger("datapilot.rag.retriever")


class HybridRetriever:
    """Hybrid RAG retriever combining keyword BM25/TF-IDF precision and vector similarity."""

    def __init__(self, store: Optional[RAGStore] = None):
        self.store = store or RAGStore()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text into cleaned lowercase alphanumeric terms."""
        return re.findall(r"\b\w+\b", text.lower())

    def _calculate_keyword_score(self, query_terms: List[str], chunk_terms: List[str], chunk_text: str) -> float:
        """Calculates TF-IDF / BM25 style keyword match score (0.0 to 1.0)."""
        if not query_terms or not chunk_terms:
            return 0.0

        chunk_term_counts = {}
        for t in chunk_terms:
            chunk_term_counts[t] = chunk_term_counts.get(t, 0) + 1

        matches = 0
        total_tf = 0.0

        for qt in query_terms:
            if qt in chunk_term_counts:
                matches += 1
                tf = chunk_term_counts[qt] / len(chunk_terms)
                total_tf += tf

        match_ratio = matches / len(set(query_terms))
        tf_score = min(1.0, total_tf * 3.0)

        raw_score = (match_ratio * 0.7) + (tf_score * 0.3)

        # Exact query phrase boost
        full_query = " ".join(query_terms)
        if full_query in chunk_text.lower():
            raw_score += 0.25

        return round(min(1.0, raw_score), 4)

    def _calculate_vector_cosine_score(self, query_terms: List[str], chunk_terms: List[str]) -> float:
        """Calculates term vector cosine similarity fallback."""
        if not query_terms or not chunk_terms:
            return 0.0

        vocab = sorted(list(set(query_terms + chunk_terms)))
        q_vec = [query_terms.count(w) for w in vocab]
        c_vec = [chunk_terms.count(w) for w in vocab]

        dot_prod = sum(q * c for q, c in zip(q_vec, c_vec))
        mag_q = math.sqrt(sum(q * q for q in q_vec))
        mag_c = math.sqrt(sum(c * c for c in c_vec))

        if mag_q == 0 or mag_c == 0:
            return 0.0

        return round(dot_prod / (mag_q * mag_c), 4)

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 3,
        score_threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Searches indexed RAG chunks using hybrid keyword + vector scoring."""
        if not query or not query.strip():
            return []

        all_chunks = self.store.get_all_chunks()
        if not all_chunks:
            return []

        q_terms = self._tokenize(query)
        if not q_terms:
            return []

        scored_results: List[Dict[str, Any]] = []

        for chunk in all_chunks:
            # Filter by category if requested
            if category and chunk.get("category") != category.strip().lower():
                continue

            content = chunk.get("content", "")
            c_terms = self._tokenize(content)

            kw_score = self._calculate_keyword_score(q_terms, c_terms, content)
            vec_score = self._calculate_vector_cosine_score(q_terms, c_terms)

            # Title match boost
            title = chunk.get("doc_title", "").lower()
            title_boost = 0.15 if any(qt in title for qt in q_terms) else 0.0

            combined_score = round(min(1.0, (kw_score * 0.5) + (vec_score * 0.5) + title_boost), 4)

            if combined_score >= score_threshold:
                result_item = {
                    "doc_id": chunk.get("doc_id"),
                    "doc_title": chunk.get("doc_title"),
                    "category": chunk.get("category"),
                    "chunk_index": chunk.get("chunk_index"),
                    "content": content,
                    "relevance_score": combined_score,
                    "keyword_score": kw_score,
                    "vector_score": vec_score,
                }
                scored_results.append(result_item)

        # Sort by relevance score descending
        scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return scored_results[:top_k]
