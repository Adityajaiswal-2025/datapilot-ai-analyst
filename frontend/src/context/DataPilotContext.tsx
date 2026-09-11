import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import type {
  DatasetSummary,
  DatasetMetadata,
  DatasetProfile,
  AgentQueryResponse,
  AutoInsightsResponse,
} from '../types/api';
import {
  getDatasetsApi,
  getDatasetDetailApi,
  getDatasetProfileApi,
  uploadDatasetApi,
  runAgentQueryApi,
  getAutoInsightsApi,
} from '../services/api';

export type TabType = 'uploader' | 'profile' | 'workspace' | 'insights' | 'studio' | 'rag';
export type AgentStage = 'idle' | 'queued' | 'supervisor' | 'profiler' | 'analyst' | 'visualization' | 'insight' | 'completed' | 'failed';

export interface ChatMessage {
  id: string;
  query: string;
  timestamp: string;
  response: AgentQueryResponse | null;
  status: 'queued' | 'running' | 'completed' | 'failed';
  error?: string;
}

interface DataPilotContextType {
  datasets: DatasetSummary[];
  selectedDatasetId: string | null;
  selectedMetadata: DatasetMetadata | null;
  datasetProfile: DatasetProfile | null;
  activeTab: TabType;
  sessionId: string;
  messages: ChatMessage[];
  agentStage: AgentStage;
  autoInsights: AutoInsightsResponse | null;
  loading: boolean;
  error: string | null;
  // Actions
  setActiveTab: (tab: TabType) => void;
  selectDataset: (id: string) => Promise<void>;
  refreshDatasets: () => Promise<void>;
  uploadFile: (file: File) => Promise<DatasetMetadata>;
  sendQuery: (query: string) => Promise<void>;
  fetchAutoInsights: (datasetId?: string) => Promise<void>;
  clearError: () => void;
}

const DataPilotContext = createContext<DataPilotContextType | undefined>(undefined);

export const DataPilotProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [selectedMetadata, setSelectedMetadata] = useState<DatasetMetadata | null>(null);
  const [datasetProfile, setDatasetProfile] = useState<DatasetProfile | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('uploader');
  const [sessionId] = useState<string>(() => `session_${Math.random().toString(36).substring(2, 11)}`);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [agentStage, setAgentStage] = useState<AgentStage>('idle');
  const [autoInsights, setAutoInsights] = useState<AutoInsightsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const clearError = () => setError(null);

  const refreshDatasets = useCallback(async () => {
    try {
      setLoading(true);
      const list = await getDatasetsApi();
      const safeList = Array.isArray(list) ? list : [];
      setDatasets(safeList);
      if (safeList.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(safeList[0].id);
      }
    } catch (err: any) {
      setError(err.message);
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  }, [selectedDatasetId]);

  const selectDataset = useCallback(async (id: string) => {
    setSelectedDatasetId(id);
    setLoading(true);
    setError(null);
    try {
      const [meta, prof] = await Promise.all([
        getDatasetDetailApi(id),
        getDatasetProfileApi(id).catch(() => null),
      ]);
      setSelectedMetadata(meta);
      setDatasetProfile(prof);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const uploadFile = async (file: File): Promise<DatasetMetadata> => {
    setLoading(true);
    setError(null);
    try {
      const meta = await uploadDatasetApi(file);
      await refreshDatasets();
      await selectDataset(meta.id);
      setActiveTab('profile');
      return meta;
    } catch (err: any) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const sendQuery = async (queryText: string) => {
    if (!selectedDatasetId) {
      setError('Please select or upload a dataset before running queries.');
      return;
    }
    if (!queryText.trim()) return;

    const msgId = `msg_${Date.now()}`;
    const newMsg: ChatMessage = {
      id: msgId,
      query: queryText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      response: null,
      status: 'running',
    };

    setMessages((prev) => [...prev, newMsg]);
    setAgentStage('queued');
    setLoading(true);
    setError(null);

    try {
      const res = await runAgentQueryApi({
        query: queryText,
        dataset_id: selectedDatasetId,
        session_id: sessionId,
      });

      // Update message with actual backend response
      setMessages((prev) =>
        prev.map((m) => (m.id === msgId ? { ...m, response: res, status: 'completed' } : m))
      );
      setAgentStage('completed');
    } catch (err: any) {
      const errMsg = err.message;
      setMessages((prev) =>
        prev.map((m) => (m.id === msgId ? { ...m, status: 'failed', error: errMsg } : m))
      );
      setAgentStage('failed');
      setError(errMsg);
    } finally {
      setLoading(false);
    }
  };

  const fetchAutoInsights = async (targetId?: string) => {
    const dsId = targetId || selectedDatasetId;
    if (!dsId) {
      setError('Select a dataset to discover automated statistical insights.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const insightsRes = await getAutoInsightsApi({ dataset_id: dsId, max_hypotheses: 20 });
      setAutoInsights(insightsRes);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshDatasets();
  }, []);

  useEffect(() => {
    if (selectedDatasetId && !selectedMetadata) {
      selectDataset(selectedDatasetId);
    }
  }, [selectedDatasetId]);

  return (
    <DataPilotContext.Provider
      value={{
        datasets,
        selectedDatasetId,
        selectedMetadata,
        datasetProfile,
        activeTab,
        sessionId,
        messages,
        agentStage,
        autoInsights,
        loading,
        error,
        setActiveTab,
        selectDataset,
        refreshDatasets,
        uploadFile,
        sendQuery,
        fetchAutoInsights,
        clearError,
      }}
    >
      {children}
    </DataPilotContext.Provider>
  );
};

export const useDataPilot = (): DataPilotContextType => {
  const context = useContext(DataPilotContext);
  if (!context) {
    throw new Error('useDataPilot must be used within a DataPilotProvider');
  }
  return context;
};
