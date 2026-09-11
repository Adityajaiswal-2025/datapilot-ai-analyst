import React from 'react';
import { useDataPilot } from '../context/DataPilotContext';
import {
  BarChart2,
  AlertTriangle,
  Grid,
  Hash,
  Type,
  Calendar,
  Layers,
  Table,
} from 'lucide-react';

export const DatasetProfileView: React.FC = () => {
  const { selectedMetadata, datasetProfile, selectedDatasetId, loading } = useDataPilot();

  if (loading) {
    return (
      <div style={{ padding: '4rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
        <BarChart2 size={48} className="animate-spin" style={{ opacity: 0.5, marginBottom: '1rem' }} />
        <p style={{ fontSize: '1.1rem', fontWeight: 600 }}>Analyzing Schema & Statistical Profile...</p>
      </div>
    );
  }

  if (!selectedDatasetId || !selectedMetadata) {
    return (
      <div className="glass-panel" style={{ margin: '2rem auto', padding: '3rem', maxWidth: '600px', textAlign: 'center' }}>
        <Grid size={48} style={{ opacity: 0.3, marginBottom: '1rem', color: 'var(--accent-cyan)' }} />
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.5rem' }}>No Dataset Selected</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          Please upload or select an active dataset from the Datasets tab to inspect schema metadata and statistics.
        </p>
      </div>
    );
  }

  const quality = datasetProfile?.quality_report;
  const columnList = Array.isArray(datasetProfile?.column_profiles)
    ? datasetProfile!.column_profiles
    : Array.isArray(selectedMetadata.columns)
    ? selectedMetadata.columns
    : [];
  const sampleRows = Array.isArray(selectedMetadata.sample_rows) ? selectedMetadata.sample_rows : [];
  const schemaColumns = Array.isArray(selectedMetadata.columns) ? selectedMetadata.columns : [];

  const getTypeBadge = (type: string) => {
    switch ((type || '').toLowerCase()) {
      case 'numeric':
        return <span className="badge badge-cyan"><Hash size={10} /> NUMERIC</span>;
      case 'categorical':
        return <span className="badge badge-indigo"><Layers size={10} /> CATEGORICAL</span>;
      case 'datetime':
        return <span className="badge badge-emerald"><Calendar size={10} /> DATETIME</span>;
      default:
        return <span className="badge badge-amber"><Type size={10} /> TEXT</span>;
    }
  };

  return (
    <div className="animate-fade-in" style={{ padding: '1.5rem', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Header Info */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 700, margin: 0 }}>
            {selectedMetadata.filename} Profile
          </h2>
          <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            {(selectedMetadata.row_count || 0).toLocaleString()} Rows × {selectedMetadata.column_count || 0} Columns • {((selectedMetadata.file_size_bytes || 0) / 1024).toFixed(1)} KB
          </span>
        </div>

        {quality && (
          <div className="glass-panel" style={{ padding: '0.75rem 1.25rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                Quality Score
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: (quality.quality_score || 0) > 80 ? 'var(--accent-emerald)' : 'var(--accent-amber)' }}>
                {quality.quality_score || 0}/100
              </div>
            </div>
            <div style={{ width: '1px', height: '30px', backgroundColor: 'var(--border-glass)' }} />
            <div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                Completeness: <strong>{quality.completeness_percentage || 0}%</strong>
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                Duplicates: <strong>{quality.duplicate_rows_count || 0} ({quality.duplicate_rows_percentage || 0}%)</strong>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Warning Flags */}
      {quality && (quality.warnings || []).length > 0 && (
        <div
          style={{
            padding: '1rem 1.25rem',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            color: 'var(--accent-amber)',
            marginBottom: '2rem',
            fontSize: '0.875rem',
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: '0.35rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertTriangle size={16} /> Schema Quality Warnings:
          </div>
          <ul style={{ paddingLeft: '1.5rem', margin: 0 }}>
            {(quality.warnings || []).map((w, idx) => (
              <li key={idx}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Column Schema Grid */}
      <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Layers color="var(--accent-cyan)" size={20} /> Column Classification & Statistics
      </h3>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
          gap: '1.25rem',
          marginBottom: '2.5rem',
        }}
      >
        {columnList.map((col: any) => (
          <div key={col.name} className="glass-card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
              <span style={{ fontSize: '1rem', fontWeight: 600, color: '#fff' }}>{col.name}</span>
              {getTypeBadge(col.classified_type || col.dtype || col.raw_dtype)}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              <div>
                Nulls: <strong>{col.null_count || 0}</strong> ({col.null_percentage || 0}%)
              </div>
              <div>
                Unique: <strong>{col.unique_count || 0}</strong>
              </div>
            </div>

            {col.numeric_stats && (
              <div style={{ paddingTop: '0.75rem', borderTop: '1px solid var(--border-glass)', fontSize: '0.775rem', color: 'var(--text-muted)' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.25rem' }}>
                  <span>Min: <strong style={{ color: '#fff' }}>{col.numeric_stats.min}</strong></span>
                  <span>Max: <strong style={{ color: '#fff' }}>{col.numeric_stats.max}</strong></span>
                  <span>Mean: <strong style={{ color: '#fff' }}>{col.numeric_stats.mean?.toFixed(2)}</strong></span>
                  <span>Std: <strong style={{ color: '#fff' }}>{col.numeric_stats.std?.toFixed(2)}</strong></span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Sample Data Table Preview */}
      {sampleRows.length > 0 && (
        <div>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Table color="var(--accent-cyan)" size={20} /> Sample Rows Preview (First 5 Rows)
          </h3>
          <div className="glass-panel" style={{ overflowX: 'auto', padding: '0.5rem' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-glass)', color: 'var(--text-secondary)' }}>
                  {schemaColumns.map((c) => (
                    <th key={c.name} style={{ padding: '0.75rem 1rem', fontWeight: 600 }}>
                      {c.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sampleRows.map((row, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--border-glass)' }}>
                    {schemaColumns.map((c) => (
                      <td key={c.name} style={{ padding: '0.75rem 1rem', color: 'var(--text-primary)' }}>
                        {row[c.name] !== null && row[c.name] !== undefined ? String(row[c.name]) : '-'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
