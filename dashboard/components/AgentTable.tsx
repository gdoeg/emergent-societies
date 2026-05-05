"use client";

import { useMemo, useState } from "react";
import { AgentSnapshot } from "@/lib/api";
import ChartCard from "@/components/ChartCard";
import AgentDetailsPanel from "@/components/AgentDetailsPanel";

interface AgentTableProps {
  agents: AgentSnapshot[];
}

type SortKey = "id" | "wealth" | "power" | "strategy";

const STRATEGY_COLORS: Record<string, string> = {
  cooperate: "text-emerald-600",
  defect: "text-red-600",
};

function SortIcon({ sortKey, col, sortAsc }: { sortKey: SortKey; col: SortKey; sortAsc: boolean }) {
  if (sortKey !== col) return <span className="ml-0.5 opacity-30">↕</span>;
  return <span className="ml-0.5">{sortAsc ? "↑" : "↓"}</span>;
}

export default function AgentTable({ agents }: AgentTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("power");
  const [sortAsc, setSortAsc] = useState(false);
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null);
  const selectedAgent = useMemo(
    () => agents.find((agent) => agent.id === selectedAgentId) ?? null,
    [agents, selectedAgentId]
  );

  if (agents.length === 0) {
    return (
      <ChartCard title="Agent Table" delay={0.1} bodyClassName="h-full">
        <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 p-4 text-center text-sm text-slate-400">
          No agent data available
        </div>
      </ChartCard>
    );
  }

  const sortedAgents = [...agents].sort((a, b) => {
    const aVal = a[sortKey];
    const bVal = b[sortKey];
    if (typeof aVal === "number" && typeof bVal === "number") {
      return sortAsc ? aVal - bVal : bVal - aVal;
    }
    const aStr = String(aVal);
    const bStr = String(bVal);
    return sortAsc ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
  });

  // IDs of top 5 agents by power for highlight
  const top5Ids = new Set(
    [...agents].sort((a, b) => b.power - a.power).slice(0, 5).map((a) => a.id)
  );

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortAsc((v) => !v);
    } else {
      setSortKey(key);
      setSortAsc(key === "id" || key === "strategy");
    }
  }

  function toggleSelected(agentId: number) {
    setSelectedAgentId((current) => (current === agentId ? null : agentId));
  }

  const headerCls =
    "cursor-pointer select-none px-2 py-1 text-left text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500 hover:text-slate-800";
  const staticHeaderCls =
    "px-2 py-1 text-left text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-500";

  return (
    <ChartCard title="Agent Table" delay={0.1} bodyClassName="h-full">
      <div className="flex h-full min-h-0 gap-3">
        <div className="min-w-0 flex-1 overflow-auto">
          <table className="w-full min-w-220 table-auto border-collapse text-[10px] leading-tight">
            <thead className="sticky top-0 bg-white/90 backdrop-blur-sm">
              <tr>
                <th className={headerCls} style={{ width: "64px" }} onClick={() => handleSort("id")}>
                  ID <SortIcon sortKey={sortKey} col="id" sortAsc={sortAsc} />
                </th>
                <th className={headerCls} style={{ width: "76px" }} onClick={() => handleSort("wealth")}>
                  Wealth <SortIcon sortKey={sortKey} col="wealth" sortAsc={sortAsc} />
                </th>
                <th className={headerCls} style={{ width: "76px" }} onClick={() => handleSort("power")}>
                  Power <SortIcon sortKey={sortKey} col="power" sortAsc={sortAsc} />
                </th>
                <th className={headerCls} style={{ width: "100px" }} onClick={() => handleSort("strategy")}>
                  Strategy <SortIcon sortKey={sortKey} col="strategy" sortAsc={sortAsc} />
                </th>
                <th className={staticHeaderCls} style={{ width: "84px" }}>
                  Risk
                </th>
                <th className={staticHeaderCls} style={{ width: "96px" }}>
                  Social
                </th>
                <th className={staticHeaderCls} style={{ width: "96px" }}>
                  Memory
                </th>
                <th className={staticHeaderCls} style={{ width: "132px" }}>
                  Goal
                </th>
                <th className={staticHeaderCls} style={{ width: "62px" }}>
                  Conf
                </th>
                <th className={staticHeaderCls} style={{ width: "220px" }}>
                  Reasoning
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedAgents.map((agent) => {
                const isTop = top5Ids.has(agent.id);
                const isSelected = selectedAgentId === agent.id;
                const strategyColor = STRATEGY_COLORS[agent.strategy] ?? "text-slate-700";
                const latestReasoning = agent.latest_reasoning?.trim() || "-";

                return (
                  <tr
                    key={agent.id}
                    aria-selected={isSelected}
                    className={`cursor-pointer border-t border-slate-100 transition-colors hover:bg-slate-50 ${
                      isSelected ? "bg-cyan-50/70" : isTop ? "bg-amber-50/60" : ""
                    }`}
                    onClick={() => toggleSelected(agent.id)}
                  >
                    <td className={`px-2 py-1.5 ${isTop || isSelected ? "font-bold text-slate-900" : "text-slate-700"}`}>
                      {agent.id}
                      {isTop && <span className="ml-1 text-amber-500">★</span>}
                    </td>
                    <td className={`px-2 py-1.5 ${isTop || isSelected ? "font-bold text-slate-900" : "text-slate-700"}`}>
                      {agent.wealth.toFixed(1)}
                    </td>
                    <td className={`px-2 py-1.5 ${isTop || isSelected ? "font-bold text-slate-900" : "text-slate-700"}`}>
                      {agent.power.toFixed(1)}
                    </td>
                    <td className={`px-2 py-1.5 capitalize ${isTop || isSelected ? "font-bold" : ""} ${strategyColor}`}>
                      {agent.strategy}
                    </td>
                    <td className="px-2 py-1.5 capitalize text-slate-600 whitespace-nowrap">{agent.risk_tolerance}</td>
                    <td className="px-2 py-1.5 capitalize text-slate-600 whitespace-nowrap">{agent.social_preference}</td>
                    <td className="px-2 py-1.5 capitalize text-slate-600 whitespace-nowrap">{agent.memory_bias}</td>
                    <td className="px-2 py-1.5 capitalize text-slate-600 whitespace-nowrap">{agent.goal.replace(/_/g, " ")}</td>
                    <td className="px-2 py-1.5 text-slate-600 whitespace-nowrap">
                      {agent.latest_confidence != null ? `${(agent.latest_confidence * 100).toFixed(0)}%` : "-"}
                    </td>
                    <td className="max-w-55 truncate px-2 py-1.5 text-slate-600" title={latestReasoning}>
                      {latestReasoning}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div
          className={`min-h-0 overflow-hidden transition-all duration-300 ease-out ${
            selectedAgent ? "w-full max-w-96 opacity-100" : "w-0 max-w-0 opacity-0"
          }`}
        >
          <div className={`h-full transition-opacity duration-300 ${selectedAgent ? "opacity-100" : "opacity-0"}`}>
            {selectedAgent && <AgentDetailsPanel agent={selectedAgent} onClose={() => setSelectedAgentId(null)} />}
          </div>
        </div>
      </div>
    </ChartCard>
  );
}
