import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
  llm_fallback_count: number;
  llm_fallback_rate: number;
  avg_llm_latency: number;
}

export async function fetchMetrics(): Promise<MetricEntry[]> {
  const { data } = await axios.get<MetricEntry[]>(`${API_BASE}/metrics`);
  return data;
}

export async function runSimulation(steps: number): Promise<void> {
  await axios.post(`${API_BASE}/run`, { steps });
}

export async function resetSimulation(): Promise<void> {
  await axios.post(`${API_BASE}/reset`);
}
