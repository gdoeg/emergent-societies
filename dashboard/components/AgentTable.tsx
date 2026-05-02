"use client";

import { useState } from "react";
import { AgentSnapshot } from "@/lib/api";
import ChartCard from "@/components/ChartCard";

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

  const headerCls =
    "cursor-pointer select-none px-2 py-1 text-left text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500 hover:text-slate-800";

  return (
    <ChartCard title="Agent Table" delay={0.1} bodyClassName="h-full">
      <div className="h-full overflow-auto">
        <table className="w-full table-fixed border-collapse text-xs">
          <thead className="sticky top-0 bg-white/90 backdrop-blur-sm">
            <tr>
              <th className={headerCls} style={{ width: "22%" }} onClick={() => handleSort("id")}>
                ID <SortIcon sortKey={sortKey} col="id" sortAsc={sortAsc} />
              </th>
              <th className={headerCls} style={{ width: "26%" }} onClick={() => handleSort("wealth")}>
                Wealth <SortIcon sortKey={sortKey} col="wealth" sortAsc={sortAsc} />
              </th>
              <th className={headerCls} style={{ width: "26%" }} onClick={() => handleSort("power")}>
                Power <SortIcon sortKey={sortKey} col="power" sortAsc={sortAsc} />
              </th>
              <th className={headerCls} style={{ width: "26%" }} onClick={() => handleSort("strategy")}>
                Strategy <SortIcon sortKey={sortKey} col="strategy" sortAsc={sortAsc} />
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedAgents.map((agent) => {
              const isTop = top5Ids.has(agent.id);
              const strategyColor =
                STRATEGY_COLORS[agent.strategy] ?? "text-slate-700";
              return (
                <tr
                  key={agent.id}
                  className={`border-t border-slate-100 transition-colors hover:bg-slate-50 ${isTop ? "bg-amber-50/60" : ""}`}
                >
                  <td className={`px-2 py-1.5 ${isTop ? "font-bold text-slate-900" : "text-slate-700"}`}>
                    {agent.id}
                    {isTop && (
                      <span className="ml-1 text-amber-500">★</span>
                    )}
                  </td>
                  <td className={`px-2 py-1.5 ${isTop ? "font-bold text-slate-900" : "text-slate-700"}`}>
                    {agent.wealth.toFixed(1)}
                  </td>
                  <td className={`px-2 py-1.5 ${isTop ? "font-bold text-slate-900" : "text-slate-700"}`}>
                    {agent.power.toFixed(1)}
                  </td>
                  <td className={`px-2 py-1.5 capitalize ${isTop ? "font-bold" : ""} ${strategyColor}`}>
                    {agent.strategy}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </ChartCard>
  );
}
