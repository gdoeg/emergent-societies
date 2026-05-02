"use client";

import { MetricEntry } from "@/lib/api";
import ChartCard from "@/components/ChartCard";

const FALLBACK_COLOR_THRESHOLD = 20;
const ERROR_COLOR_THRESHOLD = 10;

interface LLMDiagnosticsProps {
  metrics: MetricEntry[];
}

export default function LLMDiagnostics({ metrics }: LLMDiagnosticsProps) {
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const stats = latest?.llm_stats;

  const calls = stats?.calls ?? 0;
  const success = stats?.success ?? 0;
  const fallbacks = stats?.fallbacks ?? 0;
  const errors = stats?.errors ?? 0;
  const latency = stats?.latency ?? 0;

  const successPct = calls > 0 ? (success / calls) * 100 : 0;
  const fallbackPct = calls > 0 ? (fallbacks / calls) * 100 : 0;
  const errorPct = calls > 0 ? (errors / calls) * 100 : 0;
  const highFallbackRate = calls > 0 && (fallbacks + errors) / calls > 0.5;

  const rows = [
    { label: "Total Calls", value: calls.toString(), color: "text-slate-900" },
    { label: "Success %", value: calls > 0 ? `${successPct.toFixed(1)}%` : "—", color: "text-emerald-600" },
    { label: "Fallback %", value: calls > 0 ? `${fallbackPct.toFixed(1)}%` : "—", color: fallbackPct > FALLBACK_COLOR_THRESHOLD ? "text-amber-600" : "text-slate-700" },
    { label: "Error %", value: calls > 0 ? `${errorPct.toFixed(1)}%` : "—", color: errorPct > ERROR_COLOR_THRESHOLD ? "text-red-600" : "text-slate-700" },
    { label: "Avg Latency", value: `${latency.toFixed(0)} ms`, color: "text-slate-700" },
  ];

  return (
    <ChartCard title="LLM Diagnostics" delay={0.05} bodyClassName="h-full">
      <div className="flex h-full flex-col gap-1.5">
        {highFallbackRate && (
          <div className="flex items-center gap-2 rounded-xl border border-amber-300 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-800">
            <span className="text-base leading-none">⚠️</span>
            High fallback rate — results may be unreliable
          </div>
        )}
        <div className="flex flex-col gap-1">
          {rows.map(({ label, value, color }) => (
            <div
              key={label}
              className="flex items-center justify-between rounded-xl border border-slate-200/70 bg-slate-50/70 px-3 py-1.5"
            >
              <span className="text-xs font-medium text-slate-500">{label}</span>
              <span className={`text-xs font-semibold ${color}`}>{value}</span>
            </div>
          ))}
        </div>
        {!stats && (
          <div className="flex flex-1 items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 p-4 text-center text-sm text-slate-400">
            No LLM data yet
          </div>
        )}
      </div>
    </ChartCard>
  );
}
