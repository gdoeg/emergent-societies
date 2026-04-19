import axios from "axios";

const API_BASE = "http://localhost:8000";

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
