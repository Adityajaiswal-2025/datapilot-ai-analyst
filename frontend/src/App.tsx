import React from 'react';
import { DataPilotProvider, useDataPilot } from './context/DataPilotContext';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Navbar } from './components/Navbar';
import { DatasetUploader } from './components/DatasetUploader';
import { DatasetProfileView } from './components/DatasetProfileView';
import { AgentWorkspace } from './components/AgentWorkspace';
import { AutoInsightsScanner } from './components/AutoInsightsScanner';
import { MultiDatasetStudio } from './components/MultiDatasetStudio';
import { RAGKnowledgeHub } from './components/RAGKnowledgeHub';

const MainDashboard: React.FC = () => {
  const { activeTab } = useDataPilot();

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'uploader':
        return <DatasetUploader />;
      case 'profile':
        return <DatasetProfileView />;
      case 'workspace':
        return <AgentWorkspace />;
      case 'insights':
        return <AutoInsightsScanner />;
      case 'studio':
        return <MultiDatasetStudio />;
      case 'rag':
        return <RAGKnowledgeHub />;
      default:
        return <DatasetUploader />;
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar />
      <main style={{ flexGrow: 1, paddingBottom: '3rem' }}>
        <ErrorBoundary fallbackTitle="Section Error">
          {renderActiveTab()}
        </ErrorBoundary>
      </main>
      <footer
        style={{
          borderTop: '1px solid var(--border-glass)',
          padding: '1.25rem',
          textAlign: 'center',
          fontSize: '0.8rem',
          color: 'var(--text-muted)',
          backgroundColor: 'var(--bg-dark)',
        }}
      >
        DataPilot AI Data Analyst Platform • Autonomous Multi-Agent Engine (FastAPI + LangGraph + React)
      </footer>
    </div>
  );
};

export default function App() {
  return (
    <ErrorBoundary fallbackTitle="Application Error">
      <DataPilotProvider>
        <MainDashboard />
      </DataPilotProvider>
    </ErrorBoundary>
  );
}
