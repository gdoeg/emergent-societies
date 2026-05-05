"use client";

import { AgentSnapshot } from "@/lib/api";

interface AgentDetailsPanelProps {
  agent: AgentSnapshot;
  maxHistoryRows?: number;
  onClose?: () => void;
}

function formatPct(value: number | null | undefined, digits = 1): string {
  if (value == null || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(digits)}%`;
}

function titleCase(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function computeBehaviorMetrics(agent: AgentSnapshot) {
  const decisions = agent.decision_history ?? [];
  const confidences = agent.confidence_history ?? [];

  const decisionVolatility =
    decisions.length > 1
      ? decisions.slice(1).reduce((count, decision, index) => count + (decision !== decisions[index] ? 1 : 0), 0) /
        (decisions.length - 1)
      : null;

  const avgConfidence =
    confidences.length > 0
      ? confidences.reduce((sum, value) => sum + value, 0) / confidences.length
      : null;

  const cooperationRate =
    decisions.length > 0
      ? decisions.filter((decision) => decision.toLowerCase() === "cooperate").length / decisions.length
      : null;

  return { decisionVolatility, avgConfidence, cooperationRate };
}

function ConfidenceSparkline({ values }: { values: number[] }) {
  const points = values.slice(-20);
  if (points.length < 2) {
    return <div className="text-[10px] text-slate-400">Not enough confidence data</div>;
  }

  const width = 220;
  const height = 56;
  const xStep = width / Math.max(1, points.length - 1);
  const d = points
    .map((value, index) => {
      const clamped = Math.max(0, Math.min(1, value));
      const x = index * xStep;
      const y = height - clamped * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="w-full">
      <path d={d} fill="none" stroke="rgb(14 116 144)" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export default function AgentDetailsPanel({ agent, maxHistoryRows = 10, onClose }: AgentDetailsPanelProps) {
  const latestDecision = (agent.latest_decision?.trim() || agent.strategy || "").toLowerCase();
  const latestReasoning = agent.latest_reasoning?.trim() || "No reasoning captured yet.";
  const decisions = agent.decision_history ?? [];
  const confidences = agent.confidence_history ?? [];
  const { decisionVolatility, avgConfidence, cooperationRate } = computeBehaviorMetrics(agent);

  const historyRows = decisions
    .map((decision, index) => ({
      step: index + 1,
      decision,
      confidence: confidences[index] ?? null,
    }))
    .slice(-maxHistoryRows)
    .reverse();

  const recentDecisions = decisions.slice(-20);

  return (
    <div className="relative h-full rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      {onClose && (
        <button
          type="button"
          aria-label="Close agent inspection"
          onClick={onClose}
          className="absolute left-3 top-3 inline-flex h-6 w-6 items-center justify-center rounded-md border border-slate-200 bg-white text-sm text-slate-500 transition-colors hover:bg-slate-50 hover:text-slate-700"
        >
          ×
        </button>
      )}
      <div className="mb-3 flex items-center justify-between">
        <h4 className="pl-8 text-xs font-semibold uppercase tracking-[0.14em] text-slate-700">Agent {agent.id} Inspection</h4>
        <span className="text-[10px] font-medium uppercase tracking-[0.12em] text-slate-400">Active</span>
      </div>

      <div className="space-y-3 text-[11px] text-slate-700">
        <section className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
          <p className="mb-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500">Persona</p>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Risk</span><p className="capitalize">{agent.risk_tolerance}</p></div>
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Social</span><p className="capitalize">{agent.social_preference}</p></div>
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Memory</span><p className="capitalize">{agent.memory_bias}</p></div>
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Goal</span><p className="capitalize">{titleCase(agent.goal)}</p></div>
          </div>
        </section>

        <section className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
          <p className="mb-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500">Current State</p>
          <div className="grid grid-cols-3 gap-x-3">
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Wealth</span><p>{agent.wealth.toFixed(1)}</p></div>
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Power</span><p>{agent.power.toFixed(1)}</p></div>
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Strategy</span><p className="capitalize">{agent.strategy}</p></div>
          </div>
        </section>

        <section className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
          <p className="mb-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500">Latest Decision</p>
          <div className="mb-2 grid grid-cols-2 gap-x-3 gap-y-1.5">
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Decision</span><p className="capitalize">{latestDecision || "-"}</p></div>
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Confidence</span><p>{formatPct(agent.latest_confidence, 1)}</p></div>
          </div>
          <span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Reasoning</span>
          <div className="mt-1 max-h-50 overflow-y-auto rounded-lg border border-slate-200 bg-white p-2 text-[10px] leading-relaxed whitespace-pre-wrap wrap-break-word">
            {latestReasoning}
          </div>
        </section>

        <section className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
          <p className="mb-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500">Decision History</p>
          <div className="max-h-36 overflow-y-auto rounded-lg border border-slate-200 bg-white">
            <table className="w-full table-auto text-[10px]">
              <thead className="sticky top-0 bg-slate-50 text-slate-500">
                <tr>
                  <th className="px-2 py-1 text-left uppercase tracking-[0.12em]">Step</th>
                  <th className="px-2 py-1 text-left uppercase tracking-[0.12em]">Decision</th>
                  <th className="px-2 py-1 text-left uppercase tracking-[0.12em]">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {historyRows.length === 0 ? (
                  <tr>
                    <td colSpan={3} className="px-2 py-2 text-slate-400">No decision history yet</td>
                  </tr>
                ) : (
                  historyRows.map((row) => (
                    <tr key={row.step} className="border-t border-slate-100">
                      <td className="px-2 py-1">{row.step}</td>
                      <td className="px-2 py-1 capitalize">{row.decision}</td>
                      <td className="px-2 py-1">{formatPct(row.confidence, 1)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
          <p className="mb-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500">Behavior Metrics</p>
          <div className="grid grid-cols-3 gap-x-3">
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Volatility</span><p>{formatPct(decisionVolatility, 1)}</p></div>
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Avg Confidence</span><p>{formatPct(avgConfidence, 1)}</p></div>
            <div><span className="text-[9px] uppercase tracking-[0.14em] text-slate-400">Cooperation Rate</span><p>{formatPct(cooperationRate, 1)}</p></div>
          </div>
        </section>

        <section className="rounded-xl border border-slate-200 bg-slate-50/70 p-3">
          <p className="mb-2 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-500">Decision Signals</p>
          <div className="mb-2 flex flex-wrap gap-1">
            {recentDecisions.length === 0 ? (
              <span className="text-[10px] text-slate-400">No decisions yet</span>
            ) : (
              recentDecisions.map((decision, index) => (
                <span
                  key={`${decision}-${index}`}
                  className={`rounded px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-[0.08em] ${decision.toLowerCase() === "cooperate" ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"}`}
                >
                  {decision}
                </span>
              ))
            )}
          </div>
          <ConfidenceSparkline values={confidences} />
        </section>
      </div>
    </div>
  );
}
