import axios, { AxiosError } from 'axios';
import type {
  DatasetSummary,
  DatasetMetadata,
  DatasetProfile,
  AgentQueryRequest,
  AgentQueryResponse,
  AutoInsightsRequest,
  AutoInsightsResponse,
  HypothesisTestRequest,
  HypothesisTestResponse,
  JoinRequest,
  JoinResponse,
  CompareRequest,
  CompareResponse,
  RAGDocument,
  RAGSearchResult,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 45000, // 45 seconds for analytical operations
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Standardized error message extraction helper.
 */
export const handleApiError = (error: unknown, fallbackMessage: string): string => {
  if (axios.isAxiosError(error)) {
    const axiosErr = error as AxiosError<any>;
    if (axiosErr.code === 'ECONNABORTED' || axiosErr.message.includes('timeout')) {
      return 'Request timed out. The operation is taking longer than expected. Please try again.';
    }
    if (axiosErr.response) {
      const data = axiosErr.response.data;
      if (data && data.detail) {
        if (typeof data.detail === 'string') return data.detail;
        if (Array.isArray(data.detail)) {
          return data.detail.map((err: any) => `${err.loc?.join('.')}: ${err.msg}`).join(', ');
        }
      }
      return `Server Error (${axiosErr.response.status}): ${axiosErr.response.statusText}`;
    } else if (axiosErr.request) {
      return 'Unable to reach the DataPilot API server. Please check your connection and try again.';
    }
  }
  return error instanceof Error ? error.message : fallbackMessage;
};

// --- Dataset Ingestion Services ---
export const uploadDatasetApi = async (file: File): Promise<DatasetMetadata> => {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const res = await apiClient.post<any>('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data.dataset || res.data;
  } catch (err) {
    throw new Error(handleApiError(err, 'Failed to upload dataset file.'));
  }
};

export const getDatasetsApi = async (): Promise<DatasetSummary[]> => {
  try {
    const res = await apiClient.get<any>('/datasets');
    return res.data.datasets || (Array.isArray(res.data) ? res.data : []);
  } catch (err) {
    throw new Error(handleApiError(err, 'Failed to fetch registered datasets.'));
  }
};

export const getDatasetDetailApi = async (datasetId: string): Promise<DatasetMetadata> => {
  try {
    const res = await apiClient.get<any>(`/datasets/${datasetId}`);
    return res.data.dataset || res.data;
  } catch (err) {
    throw new Error(handleApiError(err, `Failed to fetch metadata for dataset ${datasetId}.`));
  }
};

export const getDatasetProfileApi = async (datasetId: string): Promise<DatasetProfile> => {
  try {
    const res = await apiClient.get<any>(`/datasets/${datasetId}/profile`);
    return res.data.profile || res.data;
  } catch (err) {
    throw new Error(handleApiError(err, `Failed to profile dataset ${datasetId}.`));
  }
};

export const deleteDatasetApi = async (datasetId: string): Promise<void> => {
  try {
    await apiClient.delete(`/datasets/${datasetId}`);
  } catch (err) {
    throw new Error(handleApiError(err, `Failed to delete dataset ${datasetId}.`));
  }
};

// --- Agent Execution & Session Services ---
export const runAgentQueryApi = async (
  req: AgentQueryRequest,
  signal?: AbortSignal
): Promise<AgentQueryResponse> => {
  try {
    const res = await apiClient.post<AgentQueryResponse>('/query', req, { signal });
    return res.data;
  } catch (err) {
    throw new Error(handleApiError(err, 'Agent workflow query execution failed.'));
  }
};

export const getSessionApi = async (sessionId: string): Promise<any> => {
  try {
    const res = await apiClient.get(`/sessions/${sessionId}`);
    return res.data;
  } catch (err) {
    throw new Error(handleApiError(err, `Failed to retrieve session ${sessionId}.`));
  }
};

export const deleteSessionApi = async (sessionId: string): Promise<void> => {
  try {
    await apiClient.delete(`/sessions/${sessionId}`);
  } catch (err) {
    throw new Error(handleApiError(err, `Failed to delete session ${sessionId}.`));
  }
};

// --- Statistical Insights Services ---
export const getAutoInsightsApi = async (req: AutoInsightsRequest): Promise<AutoInsightsResponse> => {
  try {
    const res = await apiClient.post<AutoInsightsResponse>('/insights/auto', req);
    return res.data;
  } catch (err) {
    throw new Error(handleApiError(err, 'Automated statistical insight scanning failed.'));
  }
};

export const runTargetedHypothesisApi = async (
  req: HypothesisTestRequest
): Promise<HypothesisTestResponse> => {
  try {
    const res = await apiClient.post<HypothesisTestResponse>('/insights/hypothesis', req);
    return res.data;
  } catch (err) {
    throw new Error(handleApiError(err, 'Targeted hypothesis test execution failed.'));
  }
};

// --- Multi-Dataset Services ---
export const executeJoinApi = async (req: JoinRequest): Promise<JoinResponse> => {
  try {
    const res = await apiClient.post<JoinResponse>('/analyze/join', req);
    return res.data;
  } catch (err) {
    throw new Error(handleApiError(err, 'Multi-dataset join operation failed.'));
  }
};

export const executeCompareApi = async (req: CompareRequest): Promise<CompareResponse> => {
  try {
    const res = await apiClient.post<CompareResponse>('/analyze/compare', req);
    return res.data;
  } catch (err) {
    throw new Error(handleApiError(err, 'Multi-dataset side-by-side comparison failed.'));
  }
};

// --- RAG Services ---
export const getKnowledgeDocsApi = async (): Promise<RAGDocument[]> => {
  try {
    const res = await apiClient.get<any>('/rag/knowledge');
    return res.data.documents || (Array.isArray(res.data) ? res.data : []);
  } catch (err) {
    throw new Error(handleApiError(err, 'Failed to list RAG knowledge documents.'));
  }
};

export const searchKnowledgeApi = async (query: string): Promise<RAGSearchResult[]> => {
  try {
    const res = await apiClient.post<any>('/rag/search', { query });
    return res.data.results || (Array.isArray(res.data) ? res.data : []);
  } catch (err) {
    throw new Error(handleApiError(err, 'RAG domain knowledge search failed.'));
  }
};

export const checkHealthApi = async (): Promise<boolean> => {
  try {
    const res = await apiClient.get('/health');
    return res.status === 200;
  } catch (err) {
    return false;
  }
};
