import React, { useState, useEffect } from 'react';
import { getKnowledgeDocsApi, searchKnowledgeApi } from '../services/api';
import type { RAGDocument, RAGSearchResult } from '../types/api';
import {
  BookOpen,
  Search,
  FileText,
  AlertCircle,
} from 'lucide-react';

export const RAGKnowledgeHub: React.FC = () => {
  const [documents, setDocuments] = useState<RAGDocument[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<RAGSearchResult[] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDocs = async () => {
    setLoading(true);
    setError(null);
    try {
      const docs = await getKnowledgeDocsApi();
      setDocuments(docs);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await searchKnowledgeApi(searchQuery);
      setSearchResults(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocs();
  }, []);

  const safeDocs = Array.isArray(documents) ? documents : [];
  const safeResults = Array.isArray(searchResults) ? searchResults : null;

  return (
    <div className="animate-fade-in" style={{ padding: '1.5rem', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <BookOpen color="var(--accent-cyan)" size={24} /> RAG Domain Knowledge & Glossary Hub
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
          Hybrid BM25 keyword + vector semantic search over business domain glossaries and dataset documentation.
        </p>
      </div>

      {/* Search Input Bar */}
      <form onSubmit={handleSearch} style={{ display: 'flex', gap: '0.75rem', marginBottom: '2rem' }}>
        <div style={{ position: 'relative', flexGrow: 1 }}>
          <Search
            size={18}
            color="var(--text-muted)"
            style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)' }}
          />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search business domain glossary or metric definitions..."
            style={{
              width: '100%',
              padding: '0.85rem 1rem 0.85rem 2.75rem',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--bg-card)',
              border: '1px solid var(--border-glass)',
              color: '#fff',
              fontSize: '0.925rem',
            }}
          />
        </div>
        <button
          type="submit"
          disabled={loading || !searchQuery.trim()}
          style={{
            padding: '0.85rem 1.5rem',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'var(--accent-indigo)',
            color: '#fff',
            border: 'none',
            fontWeight: 600,
            cursor: loading || !searchQuery.trim() ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <Search size={16} /> Search Glossary
        </button>
      </form>

      {/* Error alert */}
      {error && (
        <div
          style={{
            padding: '1rem 1.25rem',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            color: 'var(--accent-rose)',
            marginBottom: '2rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            fontSize: '0.9rem',
          }}
        >
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* Search Results Display */}
      {safeResults !== null && (
        <div style={{ marginBottom: '2.5rem' }}>
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '1rem', color: 'var(--accent-cyan)' }}>
            Search Results ({safeResults.length} chunks retrieved)
          </h3>

          {safeResults.length === 0 ? (
            <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              No matching knowledge snippets found for query "{searchQuery}".
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {safeResults.map((res) => (
                <div key={res.chunk_id} className="glass-card" style={{ padding: '1.25rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff' }}>{res.title}</span>
                    <span className="badge badge-cyan">Score: {res.score.toFixed(2)}</span>
                  </div>
                  <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    {res.content}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Knowledge Documents Grid */}
      <div>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '1rem', color: '#fff' }}>
          Ingested RAG Knowledge Documents ({safeDocs.length})
        </h3>

        {safeDocs.length === 0 ? (
          <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            <FileText size={48} style={{ opacity: 0.3, marginBottom: '1rem', color: 'var(--accent-cyan)' }} />
            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#fff' }}>No Domain Knowledge Documents Registered</h4>
            <p style={{ fontSize: '0.85rem', maxWidth: '420px', margin: '0.5rem auto 0' }}>
              The backend RAG engine is active. You can ingest text documents via <code>POST /api/v1/rag/knowledge</code> to enhance LLM context retrieval.
            </p>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.25rem' }}>
            {safeDocs.map((doc) => (
              <div key={doc.doc_id} className="glass-card" style={{ padding: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                  <span style={{ fontSize: '1rem', fontWeight: 600, color: '#fff' }}>{doc.title}</span>
                  <span className="badge badge-indigo">{doc.category}</span>
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Chunks: <strong>{doc.chunk_count}</strong>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  Ingested: {new Date(doc.created_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
