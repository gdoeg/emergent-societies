"use client";

import { AgentSnapshot } from "@/lib/api";
import ChartCard from "@/components/ChartCard";

interface TopAgentsPanelProps {
  agents: AgentSnapshot[];
}

const STRATEGY_COLORS: Record<string, string> = {
  cooperate: "bg-emerald-100 text-emerald-700 border-emerald-200",
  defect: "bg-red-100 text-red-700 border-red-200",
};

export default function TopAgentsPanel({ agents }: TopAgentsPanelProps) {
  if (agents.length === 0) {
    return (
      <ChartCard title="Top Agents" delay={0.15} bodyClassName="h-full">
        <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 p-4 text-center text-sm text-slate-400">
          No agent data available
        </div>
      </ChartCard>
    );
  }

  const topAgents = [...agents].sort((a, b) => b.power - a.power).slice(0, 5);

  return (
    <ChartCard title="Top Agents" delay={0.15} bodyClassName="h-full">
      <div className="flex h-full flex-col gap-1.5 overflow-auto">
        {topAgents.map((agent, rank) => {
          const strategyClass =
            STRATEGY_COLORS[agent.strategy] ?? "bg-slate-100 text-slate-700 border-slate-200";
          return (
            <div
              key={agent.id}
              className="flex items-center justify-between rounded-xl border border-slate-200/70 bg-slate-50/70 px-3 py-1.5"
            >
              <div className="flex items-center gap-2">
                <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-slate-900 text-[10px] font-bold text-white">
                  {rank + 1}
                </span>
                <span className="text-xs font-semibold text-slate-800">Agent {agent.id}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">
                  Power <span className="font-semibold text-slate-900">{agent.power.toFixed(1)}</span>
                </span>
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold capitalize ${strategyClass}`}
                >
                  {agent.strategy}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </ChartCard>
  );
}
