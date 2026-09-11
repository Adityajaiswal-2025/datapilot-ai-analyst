import React, { useState, useEffect } from 'react';
import { useDataPilot } from '../context/DataPilotContext';
import type { TabType } from '../context/DataPilotContext';
import { checkHealthApi } from '../services/api';
import {
  BrainCircuit,
  Database,
  BarChart3,
  MessageSquareCode,
  Sparkles,
  Layers,
  BookOpen,
  CheckCircle2,
  XCircle,
} from 'lucide-react';

export const Navbar: React.FC = () => {
  const {
    activeTab,
    setActiveTab,
    datasets,
    selectedDatasetId,
    selectDataset,
  } = useDataPilot();

  const [isOnline, setIsOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    const checkHealth = async () => {
      const status = await checkHealthApi();
      if (mounted) setIsOnline(status);
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'uploader', label: 'Datasets', icon: <Database size={16} /> },
    { id: 'profile', label: 'Schema Profile', icon: <BarChart3 size={16} /> },
    { id: 'workspace', label: 'Agent Workspace', icon: <MessageSquareCode size={16} /> },
    { id: 'insights', label: 'Auto-Insights', icon: <Sparkles size={16} /> },
    { id: 'studio', label: 'Multi-Dataset Studio', icon: <Layers size={16} /> },
    { id: 'rag', label: 'RAG Knowledge', icon: <BookOpen size={16} /> },
  ];

  return (
    <header
      className="glass-panel"
      style={{
        margin: '1rem 1.5rem',
        padding: '0.75rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem',
      }}
    >
      {/* Brand Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div
          style={{
            width: '40px',
            height: '40px',
            borderRadius: 'var(--radius-md)',
            background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-purple))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: 'var(--shadow-glow)',
          }}
        >
          <BrainCircuit color="#fff" size={24} />
        </div>
        <div>
          <h1 style={{ fontSize: '1.25rem', fontWeight: 800, margin: 0 }} className="gradient-text">
            DataPilot
          </h1>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
            AI DATA ANALYST PLATFORM
          </span>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }} aria-label="Main Navigation">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.5rem 0.9rem',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.85rem',
                fontWeight: 600,
                color: isActive ? '#fff' : 'var(--text-secondary)',
                backgroundColor: isActive ? 'rgba(0, 242, 254, 0.12)' : 'transparent',
                border: isActive ? '1px solid rgba(0, 242, 254, 0.3)' : '1px solid transparent',
                cursor: 'pointer',
                transition: 'all var(--transition-fast)',
              }}
              aria-current={isActive ? 'page' : undefined}
            >
              {tab.icon}
              {tab.label}
            </button>
          );
        })}
      </nav>

      {/* Right Controls: Dataset Switcher & Backend Health */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {Array.isArray(datasets) && datasets.length > 0 && (
          <select
            value={selectedDatasetId || ''}
            onChange={(e) => selectDataset(e.target.value)}
            style={{
              padding: '0.45rem 0.75rem',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'var(--bg-dark)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-glass)',
              fontSize: '0.825rem',
              cursor: 'pointer',
            }}
            aria-label="Select active dataset"
          >
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>
                {ds.filename} ({ds.row_count} rows)
              </option>
            ))}
          </select>
        )}

        <div
          className={`badge ${isOnline ? 'badge-emerald' : isOnline === false ? 'badge-rose' : 'badge-amber'}`}
          title={isOnline ? 'FastAPI Backend Online' : 'FastAPI Backend Disconnected'}
        >
          {isOnline ? (
            <>
              <CheckCircle2 size={12} /> ONLINE
            </>
          ) : isOnline === false ? (
            <>
              <XCircle size={12} /> OFFLINE
            </>
          ) : (
            'CHECKING...'
          )}
        </div>
      </div>
    </header>
  );
};
