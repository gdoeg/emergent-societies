"use client";

import { MetricEntry, AgentSnapshot } from "@/lib/api";
import LLMDiagnostics from "@/components/LLMDiagnostics";
import TopAgentsPanel from "@/components/TopAgentsPanel";
import AgentTable from "@/components/AgentTable";
import StrategyOverTimeChart from "@/components/StrategyOverTimeChart";

interface AgentInsightsProps {
  metrics: MetricEntry[];
}

export default function AgentInsights({ metrics }: AgentInsightsProps) {
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const agents: AgentSnapshot[] = latest?.agents ?? [];

  return (
    <div className="grid flex-1 min-h-0 min-w-0 grid-cols-1 auto-rows-fr gap-1.5 lg:grid-cols-2">
      <LLMDiagnostics metrics={metrics} />
      <TopAgentsPanel agents={agents} />
      <AgentTable agents={agents} />
      <StrategyOverTimeChart metricsHistory={metrics} />
    </div>
  );
}

