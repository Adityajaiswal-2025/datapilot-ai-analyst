import React, { useState, useRef } from 'react';
import { useDataPilot } from '../context/DataPilotContext';
import { deleteDatasetApi } from '../services/api';
import {
  UploadCloud,
  FileSpreadsheet,
  Trash2,
  CheckCircle,
  AlertCircle,
  HardDrive,
  Calendar,
} from 'lucide-react';

export const DatasetUploader: React.FC = () => {
  const {
    datasets = [],
    selectedDatasetId,
    selectDataset,
    refreshDatasets,
    uploadFile,
    setActiveTab,
    error: globalError,
  } = useDataPilot();

  const [dragActive, setDragActive] = useState<boolean>(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const safeDatasets = Array.isArray(datasets) ? datasets : [];

  const validateFile = (file: File): string | null => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!ext || !['csv', 'xlsx'].includes(ext)) {
      return 'Unsupported file format. Please upload a CSV (.csv) or Excel (.xlsx) dataset.';
    }
    if (file.size === 0) {
      return 'The uploaded file is empty (0 bytes). Please upload a valid dataset.';
    }
    if (file.size > 50 * 1024 * 1024) {
      return 'File size exceeds the 50 MB maximum threshold. Please upload a smaller file.';
    }
    return null;
  };

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    const validationError = validateFile(file);
    if (validationError) {
      setLocalError(validationError);
      return;
    }

    setLocalError(null);
    setUploading(true);
    try {
      await uploadFile(file);
    } catch (err: any) {
      setLocalError(err.message || 'Upload failed due to network or validation error.');
    } finally {
      setUploading(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this dataset?')) return;
    try {
      await deleteDatasetApi(id);
      await refreshDatasets();
    } catch (err: any) {
      setLocalError(err.message);
    }
  };

  return (
    <div className="animate-fade-in" style={{ padding: '1.5rem', maxWidth: '1200px', margin: '0 auto' }}>
      {/* Page Title */}
      <div style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '0.5rem' }}>
          Dataset Ingestion Hub
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.925rem' }}>
          Upload structured CSV or Excel datasets to launch autonomous multi-agent statistical profiling, analysis, and hypothesis discovery.
        </p>
      </div>

      {/* Drag & Drop Upload Zone */}
      <div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: dragActive
            ? '2px dashed var(--accent-cyan)'
            : '2px dashed var(--border-glass)',
          backgroundColor: dragActive
            ? 'rgba(0, 242, 254, 0.05)'
            : 'var(--bg-card)',
          borderRadius: 'var(--radius-lg)',
          padding: '3rem 2rem',
          textAlign: 'center',
          cursor: uploading ? 'not-allowed' : 'pointer',
          transition: 'all var(--transition-normal)',
          marginBottom: '2rem',
        }}
        role="button"
        tabIndex={0}
        aria-label="Upload dataset file dropzone"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx"
          style={{ display: 'none' }}
          onChange={(e) => handleFiles(e.target.files)}
          disabled={uploading}
        />
        <div
          style={{
            width: '64px',
            height: '64px',
            borderRadius: 'var(--radius-full)',
            background: 'rgba(0, 242, 254, 0.1)',
            color: 'var(--accent-cyan)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.25rem',
          }}
        >
          <UploadCloud size={32} />
        </div>
        <h3 style={{ fontSize: '1.15rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          {uploading ? 'Processing & Ingesting Dataset...' : 'Drag & drop dataset file here, or click to browse'}
        </h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          Supported file formats: <strong>CSV (.csv)</strong> or <strong>Excel (.xlsx)</strong> up to 50 MB
        </p>
      </div>

      {/* Error Alert Display */}
      {(localError || globalError) && (
        <div
          style={{
            padding: '1rem 1.25rem',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            color: 'var(--accent-rose)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '2rem',
            fontSize: '0.9rem',
          }}
        >
          <AlertCircle size={20} style={{ flexShrink: 0 }} />
          <span>{localError || globalError}</span>
        </div>
      )}

      {/* Active Ingested Datasets Section */}
      <div>
        <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <HardDrive size={20} color="var(--accent-cyan)" />
          Registered Datasets ({safeDatasets.length})
        </h3>

        {safeDatasets.length === 0 ? (
          <div
            className="glass-panel"
            style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}
          >
            <FileSpreadsheet size={48} style={{ opacity: 0.4, marginBottom: '1rem' }} />
            <p style={{ fontSize: '1rem', fontWeight: 500 }}>No datasets ingested yet.</p>
            <p style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>
              Upload your first CSV or XLSX file above to start analyzing with DataPilot.
            </p>
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
              gap: '1.25rem',
            }}
          >
            {safeDatasets.map((ds) => {
              const isSelected = ds.id === selectedDatasetId;
              return (
                <div
                  key={ds.id}
                  className="glass-card"
                  onClick={() => selectDataset(ds.id)}
                  style={{
                    padding: '1.25rem',
                    border: isSelected
                      ? '1px solid var(--accent-cyan)'
                      : '1px solid var(--border-glass)',
                    boxShadow: isSelected ? 'var(--shadow-glow)' : 'none',
                    cursor: 'pointer',
                    position: 'relative',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <FileSpreadsheet color="var(--accent-cyan)" size={20} />
                      <h4
                        style={{
                          fontSize: '1rem',
                          fontWeight: 600,
                          color: '#fff',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          maxWidth: '180px',
                        }}
                      >
                        {ds.filename}
                      </h4>
                    </div>
                    <button
                      onClick={(e) => handleDelete(ds.id, e)}
                      style={{
                        background: 'none',
                        border: 'none',
                        color: 'var(--text-muted)',
                        cursor: 'pointer',
                        padding: '0.25rem',
                      }}
                      title="Delete dataset"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                    <div>
                      <strong>Rows:</strong> {(ds.row_count || 0).toLocaleString()}
                    </div>
                    <div>
                      <strong>Columns:</strong> {ds.column_count || 0}
                    </div>
                    <div>
                      <strong>Type:</strong> {(ds.file_type || '').toUpperCase()}
                    </div>
                    <div>
                      <strong>Size:</strong> {((ds.file_size_bytes || 0) / 1024).toFixed(1)} KB
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '0.75rem', borderTop: '1px solid var(--border-glass)' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                      <Calendar size={12} /> {ds.created_at ? new Date(ds.created_at).toLocaleDateString() : '-'}
                    </span>
                    {isSelected ? (
                      <span className="badge badge-emerald">
                        <CheckCircle size={12} /> Active
                      </span>
                    ) : (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          selectDataset(ds.id);
                          setActiveTab('profile');
                        }}
                        style={{
                          padding: '0.25rem 0.6rem',
                          fontSize: '0.75rem',
                          backgroundColor: 'rgba(255, 255, 255, 0.08)',
                          color: '#fff',
                          border: 'none',
                          borderRadius: 'var(--radius-sm)',
                          cursor: 'pointer',
                        }}
                      >
                        Select & Profile
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
