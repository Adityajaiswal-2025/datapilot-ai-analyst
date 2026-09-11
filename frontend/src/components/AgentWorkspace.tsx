import React, { useState } from 'react';
import { useDataPilot } from '../context/DataPilotContext';
import {
  Send,
  Sparkles,
  Bot,
  User,
  Code,
  ZoomIn,
  Download,
  AlertCircle,
  ChevronRight,
  Brain,
  X,
} from 'lucide-react';

export const AgentWorkspace: React.FC = () => {
  const {
    selectedDatasetId,
    selectedMetadata,
    messages,
    sendQuery,
    loading,
  } = useDataPilot();

  const [inputQuery, setInputQuery] = useState('');
  const [activeZoomImage, setActiveZoomImage] = useState<string | null>(null);
  const [expandedCodeId, setExpandedCodeId] = useState<string | null>(null);

  const samplePrompts = [
    'Analyze column totals and distributions across groups',
    'Detect anomalies and outliers in numeric values',
    'Generate a trend analysis and bar visualization chart',
    'Provide executive recommendations and key insights',
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || loading) return;
    sendQuery(inputQuery);
    setInputQuery('');
  };

  const handlePromptClick = (prompt: string) => {
    if (loading) return;
    sendQuery(prompt);
  };

  const downloadImage = (base64Str: string, filename: string) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${base64Str}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="animate-fade-in" style={{ padding: '1.5rem', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Workspace Header */}
      <div style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Brain color="var(--accent-cyan)" size={24} /> AI Agent Analytics Workspace
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
            Multi-agent reasoning flow (Supervisor → Profiler → Analyst → Visualization → Executive Insight)
          </p>
        </div>

        {selectedMetadata && (
          <div className="badge badge-indigo">
            Dataset: {selectedMetadata.filename} ({selectedMetadata.row_count} rows)
          </div>
        )}
      </div>

      {/* Suggestion Chips */}
      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1.5rem' }}>
        {samplePrompts.map((p, idx) => (
          <button
            key={idx}
            onClick={() => handlePromptClick(p)}
            disabled={loading || !selectedDatasetId}
            style={{
              padding: '0.4rem 0.85rem',
              borderRadius: 'var(--radius-full)',
              backgroundColor: 'var(--bg-card)',
              border: '1px solid var(--border-glass)',
              color: 'var(--text-secondary)',
              fontSize: '0.8rem',
              cursor: loading || !selectedDatasetId ? 'not-allowed' : 'pointer',
              transition: 'all var(--transition-fast)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
            }}
          >
            <Sparkles size={12} color="var(--accent-cyan)" /> {p}
          </button>
        ))}
      </div>

      {/* Messages Feed */}
      <div
        className="glass-panel"
        style={{
          minHeight: '400px',
          maxHeight: '650px',
          overflowY: 'auto',
          padding: '1.5rem',
          marginBottom: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem',
        }}
      >
        {messages.length === 0 ? (
          <div style={{ margin: 'auto', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Bot size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
            <h4 style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Ask DataPilot Anything
            </h4>
            <p style={{ fontSize: '0.85rem', maxWidth: '400px', margin: '0.5rem auto 0' }}>
              Select a dataset and enter an analytical prompt below to trigger the multi-agent execution pipeline.
            </p>
          </div>
        ) : (
          messages.map((msg) => {
            const resp = msg.response;
            return (
              <div key={msg.id} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {/* User Message */}
                <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
                  <div
                    style={{
                      maxWidth: '75%',
                      backgroundColor: 'rgba(99, 102, 241, 0.2)',
                      border: '1px solid rgba(99, 102, 241, 0.4)',
                      borderRadius: 'var(--radius-md)',
                      padding: '0.85rem 1.15rem',
                      fontSize: '0.925rem',
                    }}
                  >
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.25rem', textAlign: 'right' }}>
                      {msg.timestamp}
                    </div>
                    {msg.query}
                  </div>
                  <div
                    style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: 'var(--radius-full)',
                      backgroundColor: 'var(--accent-indigo)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <User size={16} color="#fff" />
                  </div>
                </div>

                {/* Agent Response Block */}
                <div style={{ display: 'flex', gap: '0.75rem', width: '100%' }}>
                  <div
                    style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: 'var(--radius-full)',
                      background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-purple))',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                  >
                    <Bot size={18} color="#fff" />
                  </div>

                  <div className="glass-panel" style={{ flexGrow: 1, padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {/* Execution Flow Badges */}
                    {Array.isArray(resp?.execution_flow) && resp.execution_flow.length > 0 && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>FLOW:</span>
                        {resp.execution_flow.map((node, idx) => (
                          <React.Fragment key={idx}>
                            <span className="badge badge-cyan">{node}</span>
                            {idx < resp.execution_flow!.length - 1 && <ChevronRight size={12} color="var(--text-muted)" />}
                          </React.Fragment>
                        ))}
                      </div>
                    )}

                    {/* Status Spinner if running */}
                    {msg.status === 'running' && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--accent-cyan)', fontSize: '0.9rem' }}>
                        <Sparkles className="animate-spin" size={18} />
                        Multi-agent orchestrator executing analysis steps...
                      </div>
                    )}

                    {/* Error display if failed */}
                    {msg.status === 'failed' && (
                      <div style={{ color: 'var(--accent-rose)', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <AlertCircle size={18} /> {msg.error || 'Execution failed'}
                      </div>
                    )}

                    {/* Analyst Findings */}
                    {resp?.analyst && (
                      <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.02)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-glass)' }}>
                        <div style={{ fontWeight: 600, fontSize: '0.95rem', marginBottom: '0.5rem', color: 'var(--accent-cyan)' }}>
                          Analytical Findings
                        </div>
                        <p style={{ fontSize: '0.9rem', color: 'var(--text-primary)', lineHeight: 1.5 }}>
                          {resp.analyst.findings_summary}
                        </p>

                        {/* Sandboxed Code Details Toggle */}
                        {Array.isArray(resp.analyst.tool_calls_executed) && resp.analyst.tool_calls_executed.length > 0 && (
                          <div style={{ marginTop: '0.75rem' }}>
                            <button
                              onClick={() => setExpandedCodeId(expandedCodeId === msg.id ? null : msg.id)}
                              style={{
                                background: 'none',
                                border: 'none',
                                color: 'var(--text-secondary)',
                                fontSize: '0.8rem',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.35rem',
                              }}
                            >
                              <Code size={14} /> {expandedCodeId === msg.id ? 'Hide' : 'Show'} Sandboxed Tools Executed ({resp.analyst.tool_calls_executed.length})
                            </button>

                            {expandedCodeId === msg.id && (
                              <pre
                                style={{
                                  marginTop: '0.5rem',
                                  padding: '0.75rem',
                                  backgroundColor: '#070a12',
                                  borderRadius: 'var(--radius-sm)',
                                  fontSize: '0.775rem',
                                  color: 'var(--accent-cyan)',
                                  overflowX: 'auto',
                                  fontFamily: 'var(--font-mono)',
                                }}
                              >
                                {JSON.stringify(resp.analyst.tool_calls_executed, null, 2)}
                              </pre>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Visualization Image Viewer */}
                    {resp?.visualization && resp.visualization.base64_image && (
                      <div style={{ marginTop: '0.5rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-indigo)' }}>
                            {resp.visualization.title} ({resp.visualization.chart_type.toUpperCase()})
                          </span>
                          <div style={{ display: 'flex', gap: '0.5rem' }}>
                            <button
                              onClick={() => setActiveZoomImage(resp.visualization!.base64_image!)}
                              style={{
                                background: 'rgba(255, 255, 255, 0.08)',
                                border: 'none',
                                color: '#fff',
                                padding: '0.25rem 0.5rem',
                                borderRadius: 'var(--radius-sm)',
                                fontSize: '0.75rem',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.25rem',
                              }}
                            >
                              <ZoomIn size={12} /> Zoom
                            </button>
                            <button
                              onClick={() => downloadImage(resp.visualization!.base64_image!, `${resp.visualization!.chart_type}_chart.png`)}
                              style={{
                                background: 'rgba(255, 255, 255, 0.08)',
                                border: 'none',
                                color: '#fff',
                                padding: '0.25rem 0.5rem',
                                borderRadius: 'var(--radius-sm)',
                                fontSize: '0.75rem',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.25rem',
                              }}
                            >
                              <Download size={12} /> PNG
                            </button>
                          </div>
                        </div>

                        <img
                          src={`data:image/png;base64,${resp.visualization.base64_image}`}
                          alt={resp.visualization.title}
                          style={{
                            width: '100%',
                            maxHeight: '350px',
                            objectFit: 'contain',
                            borderRadius: 'var(--radius-md)',
                            border: '1px solid var(--border-glass)',
                            backgroundColor: '#070a12',
                          }}
                        />
                      </div>
                    )}

                    {/* Executive Insights & Recommendations */}
                    {resp?.insight && (
                      <div style={{ backgroundColor: 'rgba(16, 185, 129, 0.05)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                        <div style={{ fontWeight: 600, fontSize: '0.95rem', marginBottom: '0.5rem', color: 'var(--accent-emerald)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <span>Executive Summary & Strategic Insights</span>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            Confidence: <strong>{((resp.insight.confidence_score || 0) * 100).toFixed(0)}%</strong>
                          </span>
                        </div>
                        <p style={{ fontSize: '0.875rem', marginBottom: '0.75rem', lineHeight: 1.5 }}>
                          {resp.insight.executive_summary}
                        </p>

                        {Array.isArray(resp.insight.key_insights) && resp.insight.key_insights.length > 0 && (
                          <div style={{ marginBottom: '0.75rem' }}>
                            <strong style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Key Observations:</strong>
                            <ul style={{ paddingLeft: '1.25rem', marginTop: '0.25rem', fontSize: '0.85rem' }}>
                              {resp.insight.key_insights.map((ins, idx) => (
                                <li key={idx}>{ins}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {Array.isArray(resp.insight.strategic_recommendations) && resp.insight.strategic_recommendations.length > 0 && (
                          <div>
                            <strong style={{ fontSize: '0.8rem', color: 'var(--accent-emerald)' }}>Strategic Recommendations:</strong>
                            <ul style={{ paddingLeft: '1.25rem', marginTop: '0.25rem', fontSize: '0.85rem' }}>
                              {resp.insight.strategic_recommendations.map((rec, idx) => (
                                <li key={idx}>{rec}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Query Submission Form */}
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '0.75rem' }}>
        <input
          type="text"
          value={inputQuery}
          onChange={(e) => setInputQuery(e.target.value)}
          placeholder={selectedDatasetId ? 'Enter analytical question or visualization prompt...' : 'Please select a dataset to enable query workspace'}
          disabled={loading || !selectedDatasetId}
          style={{
            flexGrow: 1,
            padding: '0.85rem 1.25rem',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-glass)',
            color: 'var(--text-primary)',
            fontSize: '0.925rem',
          }}
        />
        <button
          type="submit"
          disabled={loading || !inputQuery.trim() || !selectedDatasetId}
          style={{
            padding: '0.85rem 1.5rem',
            borderRadius: 'var(--radius-md)',
            backgroundColor: loading || !inputQuery.trim() || !selectedDatasetId ? 'rgba(255, 255, 255, 0.1)' : 'var(--accent-indigo)',
            color: '#fff',
            border: 'none',
            fontWeight: 600,
            cursor: loading || !inputQuery.trim() || !selectedDatasetId ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
          }}
        >
          <Send size={16} /> Send
        </button>
      </form>

      {/* High-Resolution Zoom Modal */}
      {activeZoomImage && (
        <div
          onClick={() => setActiveZoomImage(null)}
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.85)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
          }}
        >
          <div style={{ position: 'relative', maxWidth: '90vw', maxHeight: '90vh' }}>
            <button
              onClick={() => setActiveZoomImage(null)}
              style={{
                position: 'absolute',
                top: '-40px',
                right: '0',
                background: 'none',
                border: 'none',
                color: '#fff',
                cursor: 'pointer',
              }}
            >
              <X size={28} />
            </button>
            <img
              src={`data:image/png;base64,${activeZoomImage}`}
              alt="Zoomed Chart Preview"
              style={{ maxWidth: '100%', maxHeight: '85vh', borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-card)' }}
            />
          </div>
        </div>
      )}
    </div>
  );
};
