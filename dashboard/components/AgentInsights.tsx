"use client";

import { MetricEntry } from "@/lib/api";
import ChartCard from "@/components/ChartCard";

interface AgentInsightsProps {
  metrics: MetricEntry[];
}

function AgentTable({ metrics }: { metrics: MetricEntry[] }) {
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  return (
    <ChartCard title="Agent Overview" delay={0.1} bodyClassName="h-full">
      <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 p-4 text-center text-sm text-slate-400">
        {latest
          ? `Tick ${latest.tick} · Agent table coming soon`
          : "No data available"}
      </div>
    </ChartCard>
  );
}

function TopAgentsPanel({ metrics }: { metrics: MetricEntry[] }) {
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  return (
    <ChartCard title="Top Agents" delay={0.15} bodyClassName="h-full">
      <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 p-4 text-center text-sm text-slate-400">
        {latest
          ? `Avg wealth ${latest.avg_wealth.toFixed(1)} · Top agents panel coming soon`
          : "No data available"}
      </div>
    </ChartCard>
  );
}

function StrategyChart({ metrics }: { metrics: MetricEntry[] }) {
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const strategies = latest?.strategy_counts
    ? Object.entries(latest.strategy_counts)
    : [];

  return (
    <ChartCard title="Strategy Distribution" delay={0.2} bodyClassName="h-full">
      {strategies.length > 0 ? (
        <div className="flex h-full flex-col gap-1.5 overflow-auto">
          {strategies.map(([name, count]) => (
            <div key={name} className="flex items-center justify-between rounded-xl border border-slate-200/70 bg-slate-50/70 px-3 py-1.5">
              <span className="text-xs font-medium capitalize text-slate-700">{name}</span>
              <span className="text-xs font-semibold text-slate-900">{count}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 p-4 text-center text-sm text-slate-400">
          Strategy chart coming soon
        </div>
      )}
    </ChartCard>
  );
}

export default function AgentInsights({ metrics }: AgentInsightsProps) {
  return (
    <div className="grid flex-1 min-h-0 min-w-0 grid-cols-1 auto-rows-fr gap-1.5 lg:grid-cols-2">
      <AgentTable metrics={metrics} />
      <TopAgentsPanel metrics={metrics} />
      <StrategyChart metrics={metrics} />
    </div>
  );
}
