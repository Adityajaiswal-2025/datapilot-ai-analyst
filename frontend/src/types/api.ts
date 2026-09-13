/**
 * Exact TypeScript interfaces matching DataPilot FastAPI backend Pydantic contracts.
 */

export interface ColumnSummary {
  name: string;
  dtype: string;
  null_count: number;
  unique_count: number;
}

export interface DatasetMetadata {
  id: string;
  filename: string;
  storage_path: string;
  file_size_bytes: number;
  file_type: string;
  row_count: number;
  column_count: number;
  columns: ColumnSummary[];
  sample_rows: Record<string, any>[];
  created_at: string;
}

export interface DatasetSummary {
  id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  row_count: number;
  column_count: number;
  created_at: string;
}

export interface DatasetQualityReport {
  total_cells: number;
  missing_cells: number;
  completeness_percentage: number;
  duplicate_rows_count: number;
  duplicate_rows_percentage: number;
  quality_score: number;
  warnings: string[];
}

export interface ColumnProfile {
  name: string;
  classified_type: string;
  null_count: number;
  null_percentage: number;
  unique_count: number;
  numeric_stats?: {
    min: number;
    max: number;
    mean: number;
    std: number;
    median: number;
    q25: number;
    q75: number;
    skewness: number;
  };
  categorical_stats?: {
    mode: string;
    top_values: Record<string, number>;
  };
  datetime_stats?: {
    min_date: string;
    max_date: string;
  };
}

export interface DatasetProfile {
  dataset_id: string;
  filename: string;
  row_count: number;
  column_count: number;
  quality_report: DatasetQualityReport;
  column_profiles: ColumnProfile[];
}

// Agent Output Contracts
export interface SupervisorOutput {
  selected_agent: string;
  reasoning: string;
  requires_visualization: boolean;
}

export interface ProfilerOutput {
  dataset_summary: string;
  quality_score: number;
  critical_warnings: string[];
  key_column_observations: string[];
}

export interface AnalystOutput {
  analysis_goal: string;
  tool_calls_planned: string[];
  tool_calls_executed: string[];
  execution_status: string;
  quantitative_results: Record<string, any>;
  findings_summary: string;
  requires_visualization: boolean;
  errors: string[];
}

export interface VisualizationOutput {
  chart_type: string;
  x_column: string;
  y_column?: string;
  title: string;
  base64_image?: string;
  chart_file_path?: string;
  reasoning: string;
  errors: string[];
}

export interface InsightOutput {
  executive_summary: string;
  key_insights: string[];
  strategic_recommendations: string[];
  confidence_score: number;
  limitations: string[];
  suggested_followups?: string[];
  hypotheses?: HypothesisResult[];
  structured_insights?: AutomatedInsight[];
}

export interface AgentQueryRequest {
  query: string;
  dataset_id: string;
  session_id?: string;
}

export interface AgentQueryResponse {
  success: boolean;
  message: string;
  session_id: string;
  dataset_id: string;
  query: string;
  execution_flow: string[];
  supervisor?: SupervisorOutput;
  profiler?: ProfilerOutput;
  analyst?: AnalystOutput;
  visualization?: VisualizationOutput;
  insight?: InsightOutput;
  history?: any[];
}

// Hypothesis & Insights Contracts
export interface HypothesisResult {
  hypothesis_id: string;
  test_name: string;
  target_column: string;
  group_column?: string;
  statistic: number;
  p_value: number;
  adjusted_p_value: number;
  effect_size: number;
  effect_size_type: string;
  effect_size_interpretation: string;
  sample_size: number;
  statistical_significance: boolean;
  practical_relevance: string;
  statement: string;
  warnings: string[];
}

export interface AutomatedInsight {
  insight_id: string;
  category: 'driver' | 'anomaly' | 'concentration' | 'correlation' | 'trend' | 'distribution';
  headline: string;
  explanation: string;
  evidence: Record<string, any>;
  confidence_score: number;
  limitations: string[];
}

export interface AutoInsightsRequest {
  dataset_id: string;
  max_hypotheses?: number;
  fdr_alpha?: number;
}

export interface AutoInsightsResponse {
  success: boolean;
  message: string;
  dataset_id: string;
  total_hypotheses_tested: number;
  significant_hypotheses_count: number;
  hypotheses: HypothesisResult[];
  insights: AutomatedInsight[];
  warnings: string[];
}

export interface HypothesisTestRequest {
  dataset_id: string;
  test_type: 'numeric_difference' | 'categorical_association' | 'correlation' | 'normality';
  primary_column: string;
  secondary_column?: string;
  method?: string;
}

export interface HypothesisTestResponse {
  success: boolean;
  message: string;
  dataset_id: string;
  hypothesis_result?: HypothesisResult;
  normality_result?: {
    test_name: string;
    statistic: number;
    p_value: number;
    sample_size: number;
    is_normal: boolean;
    interpretation: string;
  };
  warnings: string[];
}

// Multi-Dataset Contracts
export interface JoinKeyCandidate {
  col_left: string;
  col_right: string;
  dtype_compatible: boolean;
  cardinality_left: string;
  cardinality_right: string;
  value_overlap_percentage: number;
  confidence_score: number;
  is_recommended: boolean;
  recommendation_reason: string;
}

export interface JoinExecutionStats {
  left_row_count: number;
  right_row_count: number;
  joined_row_count: number;
  row_count_change: number;
  row_explosion_detected: boolean;
  execution_time_seconds: number;
}

export interface JoinRequest {
  dataset_ids: string[];
  join_keys: Record<string, string>;
  join_type?: string;
  new_filename?: string;
  register_result?: boolean;
}

export interface JoinResponse {
  success: boolean;
  message: string;
  stats: JoinExecutionStats;
  merged_dataset_id?: string;
  detected_keys: JoinKeyCandidate[];
  warnings: string[];
}

export interface CompareRequest {
  dataset_ids: string[];
}

export interface CompareResponse {
  success: boolean;
  message: string;
  dataset_ids: string[];
  comparison_details: Record<string, any>;
  warnings: string[];
}

// RAG Knowledge Base Contracts
export interface RAGDocument {
  doc_id: string;
  title: string;
  category: string;
  chunk_count: number;
  created_at: string;
}

export interface RAGSearchResult {
  chunk_id: string;
  doc_id: string;
  title: string;
  content: string;
  score: number;
}
