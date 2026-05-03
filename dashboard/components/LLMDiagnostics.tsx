"use client";

import { MetricEntry, LlmProviderHealth } from "@/lib/api";
import ChartCard from "@/components/ChartCard";

const FALLBACK_COLOR_THRESHOLD = 20;
const ERROR_COLOR_THRESHOLD = 10;

interface LLMDiagnosticsProps {
  metrics: MetricEntry[];
}

function buildHealthSummary(health: LlmProviderHealth | null | undefined) {
  if (!health) {
    return { badge: "No provider data", detail: "No provider health has been recorded yet.", tone: "neutral" as const };
  }

  if (health.healthy) {
    return {
      badge: "Healthy",
      detail: `${health.provider} responded successfully${health.model ? ` on ${health.model}` : ""}.`,
      tone: "healthy" as const,
    };
  }

  const parts = [health.status_code ? `HTTP ${health.status_code}` : null, health.error_code, health.error_type]
    .filter(Boolean)
    .join(" • ");

  return {
    badge: parts || "Provider error",
    detail: health.message || "The last provider request failed.",
    tone: "error" as const,
  };
}

export default function LLMDiagnostics({ metrics }: LLMDiagnosticsProps) {
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;
  const stats = latest?.llm_stats;
  const providerHealth = latest?.llm_provider_health;
  const healthSummary = buildHealthSummary(providerHealth);

  const calls = stats?.calls ?? 0;
  const totalDecisions = stats?.total_agent_decisions ?? latest?.total_agent_decisions ?? calls;
  const success = stats?.success ?? 0;
  const fallbacks = stats?.fallbacks ?? 0;
  const errors = stats?.errors ?? 0;
  const latency = stats?.latency ?? 0;

  const successPct = totalDecisions > 0 ? (success / totalDecisions) * 100 : 0;
  // Prefer the pre-computed fallback_rate (covers both parse failures and provider errors).
  // Fall back to computing from stats.fallbacks only when fallback_rate is unavailable.
  const fallbackPct =
    stats?.fallback_rate != null
      ? stats.fallback_rate * 100
      : totalDecisions > 0
        ? (fallbacks / totalDecisions) * 100
        : 0;
  const errorPct = totalDecisions > 0 ? (errors / totalDecisions) * 100 : 0;
  const highFallbackRate = fallbackPct > 50;
  const healthClasses =
    healthSummary.tone === "healthy"
      ? "border-emerald-300 bg-emerald-50 text-emerald-800"
      : healthSummary.tone === "error"
        ? "border-red-300 bg-red-50 text-red-800"
        : "border-slate-200 bg-slate-50 text-slate-700";

  const rows = [
    { label: "Total Calls", value: calls.toString(), color: "text-slate-900" },
    { label: "Success %", value: totalDecisions > 0 ? `${successPct.toFixed(1)}%` : "—", color: "text-emerald-600" },
    { label: "Fallback %", value: totalDecisions > 0 ? `${fallbackPct.toFixed(1)}%` : "—", color: fallbackPct > FALLBACK_COLOR_THRESHOLD ? "text-amber-600" : "text-slate-700" },
    { label: "Error %", value: totalDecisions > 0 ? `${errorPct.toFixed(1)}%` : "—", color: errorPct > ERROR_COLOR_THRESHOLD ? "text-red-600" : "text-slate-700" },
    { label: "Avg Latency", value: `${latency.toFixed(0)} ms`, color: "text-slate-700" },
  ];

  return (
    <ChartCard title="LLM Diagnostics" delay={0.05} bodyClassName="h-full">
      <div className="flex h-full flex-col gap-1.5">
        <div className={`rounded-xl border px-3 py-2 ${healthClasses}`}>
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.22em]">Provider Health</p>
              <p className="mt-0.5 text-sm font-semibold">
                {providerHealth?.provider ? providerHealth.provider.toUpperCase() : "LLM"}
                {providerHealth?.model ? ` · ${providerHealth.model}` : ""}
              </p>
            </div>
            <span className="rounded-full border border-current/20 bg-white/60 px-2 py-0.5 text-[11px] font-semibold">
              {healthSummary.badge}
            </span>
          </div>
          <p className="mt-1 text-xs leading-4 opacity-90">{healthSummary.detail}</p>
        </div>
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
