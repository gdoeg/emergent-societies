import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface AgentSnapshot {
  id: number;
  wealth: number;
  power: number;
  strategy: string;
  risk_tolerance: string;
  social_preference: string;
  memory_bias: string;
  goal: string;
  latest_decision?: string;
  latest_confidence?: number | null;
  latest_reasoning?: string;
  decision_history?: string[];
  confidence_history?: number[];
}

export interface AvgConfidenceByPersonaType {
  risk_tolerance: Record<string, number>;
  social_preference: Record<string, number>;
  memory_bias: Record<string, number>;
  goal: Record<string, number>;
}

export interface LlmStats {
  calls: number;
  total_agent_decisions?: number;
  success: number;
  fallbacks: number;
  errors: number;
  success_rate?: number;
  fallback_rate?: number;
  latency: number;
}

export interface LlmProviderHealth {
  provider: string;
  model: string;
  healthy: boolean | null;
  status: string;
  provider_status?: "ok" | "error" | null;
  status_code: number | null;
  error_type: string | null;
  error_code: string | null;
  provider_error?: string | null;
  message: string | null;
}

export interface MetricEntry {
  tick: number;
  gini: number;
  total_wealth: number;
  avg_wealth: number;
  avg_power: number;
  max_power: number;
  average_degree: number;
  network_density: number;
  wealth_distribution: number[];
  llm_call_count: number;
  llm_success_count?: number;
  llm_success_rate?: number;
  llm_fallback_count: number;
  llm_fallback_rate: number;
  total_agent_decisions?: number;
  success_agent_decisions?: number;
  fallback_agent_decisions?: number;
  avg_llm_latency: number;
  pct_cooperating?: number;
  strategy_counts?: Record<string, number>;
  agents?: AgentSnapshot[];
  cooperation_rate_by_social_preference?: Record<string, number>;
  defect_rate_by_social_preference?: Record<string, number>;
  avg_confidence_by_persona_type?: AvgConfidenceByPersonaType;
  strategy_switching_rate_by_risk_tolerance?: Record<string, number>;
  llm_stats?: LlmStats;
  provider_status?: "ok" | "error" | null;
  provider_error?: string | null;
  llm_provider_health?: LlmProviderHealth | null;
  /** Present only in aggregate entries – number of runs contributing to this step. */
  run_count?: number;
}

export interface TrackerInfo {
  run_count: number;
  runs: Array<{ run_id: string; timestamp: string; steps: number }>;
}

export async function fetchMetrics(): Promise<MetricEntry[]> {
  const { data } = await axios.get<MetricEntry[]>(`${API_BASE}/metrics`);
  return data;
}

export async function fetchAggregateMetrics(): Promise<MetricEntry[]> {
  const { data } = await axios.get<MetricEntry[]>(`${API_BASE}/aggregate-metrics`);
  return data;
}

export async function fetchTrackerInfo(): Promise<TrackerInfo> {
  const { data } = await axios.get<TrackerInfo>(`${API_BASE}/tracker-info`);
  return data;
}

export async function runSimulation(steps: number): Promise<void> {
  await axios.post(`${API_BASE}/run`, { steps });
}

export async function runMultipleSimulations(steps: number, numRuns: number): Promise<void> {
  await axios.post(`${API_BASE}/run-multiple`, { steps, num_runs: numRuns });
}

export async function resetSimulation(): Promise<void> {
  await axios.post(`${API_BASE}/reset`);
}
