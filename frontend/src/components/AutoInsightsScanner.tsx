import React from 'react';
import { useDataPilot } from '../context/DataPilotContext';
import {
  Sparkles,
  Award,
  AlertCircle,
  Layers,
  RefreshCw,
  Info,
} from 'lucide-react';

export const AutoInsightsScanner: React.FC = () => {
  const {
    selectedDatasetId,
    selectedMetadata,
    autoInsights,
    fetchAutoInsights,
    loading,
    error,
  } = useDataPilot();

  if (!selectedDatasetId || !selectedMetadata) {
    return (
      <div className="glass-panel" style={{ margin: '2rem auto', padding: '3rem', maxWidth: '600px', textAlign: 'center' }}>
        <Sparkles size={48} style={{ opacity: 0.3, marginBottom: '1rem', color: 'var(--accent-cyan)' }} />
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.5rem' }}>No Dataset Selected</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          Please select a dataset to launch automated statistical hypothesis testing and discovery scans.
        </p>
      </div>
    );
  }

  return (
    <div className="animate-fade-in" style={{ padding: '1.5rem', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header & Scan Button */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Sparkles color="var(--accent-cyan)" size={24} /> Automated Statistical Insight Scanner
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
            Scans column combinations, formulates statistical hypotheses, applies Benjamini-Hochberg FDR correction, and extracts Pareto distributions.
          </p>
        </div>

        <button
          onClick={() => fetchAutoInsights()}
          disabled={loading}
          style={{
            padding: '0.75rem 1.25rem',
            borderRadius: 'var(--radius-md)',
            background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-indigo))',
            color: '#fff',
            border: 'none',
            fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            boxShadow: 'var(--shadow-glow)',
          }}
        >
          <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          {loading ? 'Scanning Dataset...' : 'Run Automated Scan'}
        </button>
      </div>

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

      {!autoInsights && !loading && (
        <div className="glass-panel" style={{ padding: '4rem 2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <Award size={48} style={{ opacity: 0.3, marginBottom: '1rem', color: 'var(--accent-cyan)' }} />
          <h3 style={{ fontSize: '1.15rem', fontWeight: 600, color: '#fff', marginBottom: '0.5rem' }}>
            Ready to Scan {selectedMetadata.filename}
          </h3>
          <p style={{ fontSize: '0.85rem', maxWidth: '480px', margin: '0 auto 1.5rem' }}>
            Click "Run Automated Scan" to perform multi-testing corrected statistical hypothesis scans (T-test, ANOVA, Chi-Square, Pearson/Spearman) across candidate feature pairs.
          </p>
          <button
            onClick={() => fetchAutoInsights()}
            style={{
              padding: '0.6rem 1.2rem',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--accent-indigo)',
              color: '#fff',
              border: 'none',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Launch Auto Discovery
          </button>
        </div>
      )}

      {/* Scan Summary Banner */}
      {autoInsights && (
        <div>
          <div
            className="glass-panel"
            style={{
              padding: '1.25rem 1.5rem',
              marginBottom: '2rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '1rem',
            }}
          >
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                Tested Column Pairs
              </span>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#fff' }}>
                {autoInsights.total_hypotheses_tested}
              </div>
            </div>
            <div style={{ width: '1px', height: '30px', backgroundColor: 'var(--border-glass)' }} />
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                Statistically Significant (BH FDR adj. p &lt; 0.05)
              </span>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--accent-emerald)' }}>
                {autoInsights.significant_hypotheses_count}
              </div>
            </div>
            <div style={{ width: '1px', height: '30px', backgroundColor: 'var(--border-glass)' }} />
            <div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                Correction Method
              </span>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>
                Benjamini-Hochberg FDR
              </div>
            </div>
          </div>

          {/* Discovered Business Insights Section */}
          {Array.isArray(autoInsights?.insights) && autoInsights.insights.length > 0 && (
            <div style={{ marginBottom: '2.5rem' }}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Award color="var(--accent-emerald)" size={20} /> Executive Business Drivers & Concentration Risks
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '1.25rem' }}>
                {autoInsights.insights.map((insight) => (
                  <div key={insight.insight_id} className="glass-card" style={{ padding: '1.25rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                      <span className="badge badge-emerald">{(insight.category || '').toUpperCase()}</span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        Confidence: {((insight.confidence_score || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#fff', marginBottom: '0.5rem' }}>
                      {insight.headline}
                    </h4>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.75rem' }}>
                      {insight.explanation}
                    </p>
                    {Array.isArray(insight.limitations) && insight.limitations.length > 0 && (
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                        <Info size={12} /> {insight.limitations[0]}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Ranked Statistical Hypotheses Cards */}
          <div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Layers color="var(--accent-cyan)" size={20} /> Evaluated Statistical Hypotheses (Ranked by BH FDR Adjusted p-value)
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {(autoInsights?.hypotheses || []).map((hyp) => (
                <div
                  key={hyp.hypothesis_id}
                  className="glass-card"
                  style={{
                    padding: '1.25rem',
                    borderLeft: hyp.statistical_significance
                      ? '4px solid var(--accent-emerald)'
                      : '4px solid var(--border-glass)',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.75rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span className="badge badge-cyan">{hyp.test_name}</span>
                      <span style={{ fontSize: '0.95rem', fontWeight: 600, color: '#fff' }}>
                        {hyp.target_column} {hyp.group_column ? `vs ${hyp.group_column}` : ''}
                      </span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.8rem' }}>
                      <div>
                        Raw p: <strong style={{ color: '#fff' }}>{(hyp.p_value || 0).toFixed(4)}</strong>
                      </div>
                      <div>
                        BH Adj. p: <strong style={{ color: hyp.statistical_significance ? 'var(--accent-emerald)' : 'var(--text-secondary)' }}>
                          {(hyp.adjusted_p_value || 0).toFixed(4)}
                        </strong>
                      </div>
                      <span className={`badge ${hyp.statistical_significance ? 'badge-emerald' : 'badge-amber'}`}>
                        {hyp.statistical_significance ? 'SIGNIFICANT' : 'NON-SIGNIFICANT'}
                      </span>
                    </div>
                  </div>

                  <p style={{ fontSize: '0.875rem', color: 'var(--text-primary)', lineHeight: 1.5, marginBottom: '0.75rem' }}>
                    {hyp.statement}
                  </p>

                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)', paddingTop: '0.5rem', borderTop: '1px solid var(--border-glass)' }}>
                    <span>
                      Effect Size: <strong style={{ color: '#fff' }}>{hyp.effect_size}</strong> ({hyp.effect_size_type} - {hyp.effect_size_interpretation})
                    </span>
                    <span>Sample Size (N): {hyp.sample_size}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
