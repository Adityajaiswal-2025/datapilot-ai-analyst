import React, { useState } from 'react';
import { useDataPilot } from '../context/DataPilotContext';
import { executeJoinApi, executeCompareApi } from '../services/api';
import type { JoinResponse, CompareResponse } from '../types/api';
import {
  Layers,
  AlertTriangle,
  CheckCircle,
  FileSpreadsheet,
  GitMerge,
  BarChart2,
} from 'lucide-react';

export const MultiDatasetStudio: React.FC = () => {
  const { datasets = [], refreshDatasets, selectDataset, setActiveTab } = useDataPilot();

  const safeDatasets = Array.isArray(datasets) ? datasets : [];

  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [joinType, setJoinType] = useState<string>('inner');
  const [leftKey, setLeftKey] = useState<string>('');
  const [rightKey, setRightKey] = useState<string>('');
  const [joinResult, setJoinResult] = useState<JoinResponse | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [showWarningModal, setShowWarningModal] = useState<boolean>(false);

  const safeSelectedIds = Array.isArray(selectedIds) ? selectedIds : [];

  const handleToggleSelect = (id: string) => {
    setSelectedIds((prev) =>
      (prev || []).includes(id) ? prev.filter((i) => i !== id) : [...(prev || []), id]
    );
  };

  const handleRunJoin = async () => {
    if (safeSelectedIds.length < 2) {
      setError('Please select at least 2 datasets to execute a join operation.');
      return;
    }
    if (!leftKey.trim() || !rightKey.trim()) {
      setError('Please specify join key column names for both datasets.');
      return;
    }

    setLoading(true);
    setError(null);
    setShowWarningModal(false);

    try {
      const res = await executeJoinApi({
        dataset_ids: safeSelectedIds,
        join_keys: { [leftKey]: rightKey },
        join_type: joinType,
        register_result: true,
      });
      setJoinResult(res);
      await refreshDatasets();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunCompare = async () => {
    if (safeSelectedIds.length < 2) {
      setError('Please select at least 2 datasets to perform side-by-side comparison.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await executeCompareApi({ dataset_ids: safeSelectedIds });
      setCompareResult(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in" style={{ padding: '1.5rem', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers color="var(--accent-cyan)" size={24} /> Multi-Dataset Synthesis & Join Studio
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
          Combine 2+ datasets with join cardinality auditing, candidate key detection, and side-by-side comparison.
        </p>
      </div>

      {/* Dataset Selection Matrix */}
      <div style={{ marginBottom: '2rem' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', color: '#fff' }}>
          Select Datasets to Synthesize (Select 2 or more)
        </h3>

        {safeDatasets.length < 2 ? (
          <div className="glass-panel" style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            <FileSpreadsheet size={40} style={{ opacity: 0.4, marginBottom: '0.75rem' }} />
            <p style={{ fontSize: '0.95rem' }}>Multi-dataset operations require at least 2 registered datasets.</p>
            <p style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>
              Please upload another dataset in the Datasets tab to enable joining and comparisons.
            </p>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1rem' }}>
            {safeDatasets.map((ds) => {
              const isChecked = safeSelectedIds.includes(ds.id);
              return (
                <div
                  key={ds.id}
                  className="glass-card"
                  onClick={() => handleToggleSelect(ds.id)}
                  style={{
                    padding: '1rem',
                    border: isChecked ? '1px solid var(--accent-cyan)' : '1px solid var(--border-glass)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <FileSpreadsheet color={isChecked ? 'var(--accent-cyan)' : 'var(--text-muted)'} size={20} />
                    <div>
                      <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#fff' }}>{ds.filename}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {ds.row_count} rows × {ds.column_count} cols
                      </div>
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => {}}
                    style={{ width: '18px', height: '18px', cursor: 'pointer' }}
                  />
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Error Alert */}
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
          <AlertTriangle size={20} />
          <span>{error}</span>
        </div>
      )}

      {/* Action Controls */}
      {safeSelectedIds.length >= 2 && (
        <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <GitMerge color="var(--accent-cyan)" size={20} /> Configure Join Parameters
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                Dataset 1 Join Key Column:
              </label>
              <input
                type="text"
                value={leftKey}
                onChange={(e) => setLeftKey(e.target.value)}
                placeholder="e.g. customer_id"
                style={{
                  width: '100%',
                  padding: '0.6rem 0.85rem',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'var(--bg-dark)',
                  border: '1px solid var(--border-glass)',
                  color: '#fff',
                  fontSize: '0.875rem',
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                Dataset 2 Join Key Column:
              </label>
              <input
                type="text"
                value={rightKey}
                onChange={(e) => setRightKey(e.target.value)}
                placeholder="e.g. cust_id"
                style={{
                  width: '100%',
                  padding: '0.6rem 0.85rem',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'var(--bg-dark)',
                  border: '1px solid var(--border-glass)',
                  color: '#fff',
                  fontSize: '0.875rem',
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.35rem' }}>
                Join Type:
              </label>
              <select
                value={joinType}
                onChange={(e) => setJoinType(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.6rem 0.85rem',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'var(--bg-dark)',
                  border: '1px solid var(--border-glass)',
                  color: '#fff',
                  fontSize: '0.875rem',
                }}
              >
                <option value="inner">Inner Join (Matching rows in both)</option>
                <option value="left">Left Join (All left dataset rows)</option>
                <option value="right">Right Join (All right dataset rows)</option>
                <option value="outer">Full Outer Join (All rows from both)</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            <button
              onClick={() => setShowWarningModal(true)}
              disabled={loading}
              style={{
                padding: '0.75rem 1.5rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'var(--accent-indigo)',
                color: '#fff',
                border: 'none',
                fontWeight: 600,
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
              }}
            >
              <GitMerge size={16} /> Execute Dataset Join
            </button>

            <button
              onClick={handleRunCompare}
              disabled={loading}
              style={{
                padding: '0.75rem 1.5rem',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'rgba(255, 255, 255, 0.08)',
                color: '#fff',
                border: '1px solid var(--border-glass)',
                fontWeight: 600,
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
              }}
            >
              <BarChart2 size={16} /> Run Side-by-Side Comparison
            </button>
          </div>
        </div>
      )}

      {/* Confirmation & Risk Audit Modal */}
      {showWarningModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1.5rem',
          }}
        >
          <div className="glass-panel" style={{ maxWidth: '500px', width: '100%', padding: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem', color: 'var(--accent-amber)' }}>
              <AlertTriangle size={28} />
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0, color: '#fff' }}>
                Confirm Join Execution & Cardinality Audit
              </h3>
            </div>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '1.5rem' }}>
              DataPilot will execute an <strong>{joinType.toUpperCase()} JOIN</strong> mapping column <code>{leftKey}</code> to <code>{rightKey}</code> across {selectedIds.length} datasets.
            </p>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <button
                onClick={() => setShowWarningModal(false)}
                style={{
                  padding: '0.6rem 1.2rem',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'rgba(255, 255, 255, 0.08)',
                  color: '#fff',
                  border: 'none',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleRunJoin}
                style={{
                  padding: '0.6rem 1.2rem',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'var(--accent-indigo)',
                  color: '#fff',
                  border: 'none',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Confirm & Join Datasets
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Join Results Output */}
      {joinResult && (
        <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-emerald)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <CheckCircle size={20} /> Join Operation Completed
            </h3>
            {joinResult.merged_dataset_id && (
              <button
                onClick={() => {
                  selectDataset(joinResult.merged_dataset_id!);
                  setActiveTab('profile');
                }}
                style={{
                  padding: '0.4rem 0.85rem',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'var(--accent-cyan)',
                  color: '#000',
                  border: 'none',
                  fontWeight: 600,
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                }}
              >
                View Merged Dataset Profile
              </button>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            <div>Left Rows: <strong>{joinResult.stats.left_row_count}</strong></div>
            <div>Right Rows: <strong>{joinResult.stats.right_row_count}</strong></div>
            <div>Joined Result Rows: <strong style={{ color: 'var(--accent-cyan)' }}>{joinResult.stats.joined_row_count}</strong></div>
            <div>Execution Time: <strong>{joinResult.stats.execution_time_seconds.toFixed(2)}s</strong></div>
          </div>
        </div>
      )}

      {/* Comparison Results Output */}
      {compareResult && (
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', color: 'var(--accent-cyan)' }}>
            Side-by-Side Dataset Comparison Breakdown
          </h3>
          <pre
            style={{
              padding: '1rem',
              backgroundColor: '#070a12',
              borderRadius: 'var(--radius-md)',
              fontSize: '0.85rem',
              color: 'var(--text-primary)',
              overflowX: 'auto',
            }}
          >
            {JSON.stringify(compareResult.comparison_details, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};
