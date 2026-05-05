"use client";

import { useEffect, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { fetchMetrics, fetchAggregateMetrics, MetricEntry } from "@/lib/api";
import DashboardLayout from "@/components/DashboardLayout";
import Controls, { ViewMode } from "@/components/Controls";
import SimulationCharts from "@/components/SimulationCharts";
import AgentInsights from "@/components/AgentInsights";

const POLL_INTERVAL_MS = 3000;

export default function Home() {
  const [metrics, setMetrics] = useState<MetricEntry[]>([]);
  const [aggregateMetrics, setAggregateMetrics] = useState<MetricEntry[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>("current");
  const [mainView, setMainView] = useState<"simulation" | "agents">("simulation");
  const [loading, setLoading] = useState(true);
  const [, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [current, aggregate] = await Promise.all([
        fetchMetrics(),
        fetchAggregateMetrics(),
      ]);
      setMetrics(current);
      setAggregateMetrics(aggregate);
      setError(null);
    } catch {
      setError("Could not connect to backend at http://localhost:8000");
    } finally {
      setLoading(false);
    }
  }, []);

  // Polling every second
  useEffect(() => {
    const raf = window.requestAnimationFrame(() => {
      void refresh();
    });
    const id = setInterval(refresh, POLL_INTERVAL_MS);
    return () => {
      window.cancelAnimationFrame(raf);
      clearInterval(id);
    };
  }, [refresh]);

  const activeMetrics = viewMode === "aggregate" ? aggregateMetrics : metrics;
  const latest = activeMetrics.length > 0 ? activeMetrics[activeMetrics.length - 1] : null;

  // run_count is only present on aggregate entries; null when not in aggregate mode.
  const runCount: number | null =
    viewMode === "aggregate" && latest ? (latest.run_count ?? null) : null;

  const metricsConfig = [
    { label: "Current Tick", value: latest ? latest.tick : "—", tint: "rgba(15,118,110,0.12)" },
    {
      label: "Gini (Inequality)",
      value: latest ? latest.gini.toFixed(4) : "—",
      tint: "rgba(20,184,166,0.12)",
    },
    {
      label: "Total Wealth",
      value: latest ? latest.total_wealth.toFixed(0) : "—",
      tint: "rgba(14,165,233,0.1)",
    },
    {
      label: "LLM Fallback Rate",
      value: latest ? `${(latest.llm_fallback_rate * 100).toFixed(1)}%` : "—",
      tint: "rgba(51,65,85,0.08)",
    },
  ];

  const railStats = [
    { label: "Avg Wealth", value: latest ? latest.avg_wealth.toFixed(1) : "—" },
    { label: "Avg Power", value: latest ? latest.avg_power.toFixed(3) : "—" },
    {
      label: "Avg LLM Latency",
      value: latest ? `${(latest.avg_llm_latency * 1000).toFixed(0)} ms` : "—",
    },
    {
      label: "Fallbacks",
      value: (() => {
        if (!latest) return "—";
        const numerator = latest.fallback_agent_decisions ?? latest.llm_fallback_count;
        const denominator = latest.total_agent_decisions ?? latest.llm_call_count;
        return `${numerator}/${denominator}`;
      })(),
    },
  ];

  const chartTitle = (base: string) =>
    viewMode === "aggregate" ? `${base} (avg across ${runCount ?? "?"} runs)` : base;

  return (
    <DashboardLayout>
      <div className="grid w-full grid-cols-1 gap-2 p-2 lg:h-full lg:grid-cols-[232px_1fr] lg:p-2.5">
        <aside className="grid min-w-0 gap-1.5 lg:min-h-0 lg:grid-rows-[auto_auto_minmax(0,1fr)]">
          <motion.section
            initial={{ opacity: 0, x: -24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.45, ease: "easeOut" }}
            className="overflow-hidden rounded-4xl border border-slate-900/10 bg-[linear-gradient(145deg,rgba(16,42,51,0.96),rgba(23,63,71,0.92))] p-2 text-white shadow-[0_28px_90px_rgba(16,42,51,0.24)]"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="inline-flex items-center rounded-full border border-white/15 bg-white/10 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-100/90">
                Live Dashboard
              </span>
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_0_8px_rgba(52,211,153,0.12)]" />
            </div>
            <h1 className="mt-2 max-w-sm text-[1.6rem] font-bold tracking-tight text-white leading-[1.02]">
              Emergent Societies
            </h1>
            <p className="mt-1 max-w-md text-[11px] leading-4 text-slate-200/86">
              An operational view of emergent dynamics in an active simulation.
            </p>
            <div className="mt-2 grid grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)] gap-1.5">
              <div className="rounded-2xl border border-white/10 bg-white/6 px-2 py-1.5">
                <p className="text-[11px] uppercase tracking-[0.22em] text-slate-300">Samples</p>
                <p className="mt-0.5 text-base font-semibold text-white">{activeMetrics.length}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/6 px-2 py-1.5">
                <p className="text-[11px] uppercase tracking-[0.22em] text-slate-300">Mode</p>
                <p className="mt-0.5 text-base font-semibold text-white capitalize">{viewMode}</p>
              </div>
            </div>
          </motion.section>

          <Controls onUpdate={refresh} viewMode={viewMode} onViewModeChange={setViewMode} className="lg:h-full" />

          <motion.section
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.12, ease: "easeOut" }}
            className="rounded-[28px] border border-white/70 bg-white/88 p-1.5 shadow-[0_24px_80px_rgba(16,42,51,0.12)] backdrop-blur-sm lg:min-h-0"
          >
            <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-500">
              Quick View
            </p>
            <div className="mt-1.5 grid grid-cols-2 gap-1.5">
              {railStats.map(({ label, value }) => (
                <div key={label} className="rounded-2xl border border-slate-200/70 bg-slate-50/70 px-2 py-1.5">
                  <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">{label}</p>
                  <p className="mt-0.5 text-sm font-semibold text-slate-900">{value}</p>
                </div>
              ))}
            </div>
          </motion.section>
        </aside>

        <section className="flex min-w-0 flex-col lg:min-h-0">
          <div className="grid h-auto grid-cols-2 gap-1.5 lg:h-15 lg:grid-cols-4 lg:gap-2">
            {metricsConfig.map(({ label, value, tint }, i) => (
              <motion.div
                key={label}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: i * 0.08 }}
                className="rounded-[28px] border border-white/70 p-1.5 shadow-[0_24px_80px_rgba(16,42,51,0.12)] backdrop-blur-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_30px_90px_rgba(16,42,51,0.16)] lg:p-2"
                style={{ backgroundImage: `linear-gradient(180deg, ${tint} 0%, rgba(255,255,255,0.96) 46%)` }}
              >
                <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">{label}</p>
                <p className="mt-0.5 text-base font-semibold leading-none text-slate-950 lg:text-lg">
                  {loading && !latest ? (
                    <span className="inline-block h-6 w-16 animate-pulse rounded-md bg-slate-200" />
                  ) : (
                    value
                  )}
                </p>
              </motion.div>
            ))}
          </div>

          <div className="mt-1.5 flex rounded-[28px] border border-white/60 bg-white/85 p-0.5 shadow-[0_24px_80px_rgba(16,42,51,0.10)] backdrop-blur-sm text-xs font-semibold w-fit">
            <button
              onClick={() => setMainView("simulation")}
              className={`rounded-[22px] px-4 py-1.5 transition ${mainView === "simulation" ? "bg-slate-900 text-white shadow" : "text-slate-600 hover:bg-slate-100"}`}
            >
              Simulation
            </button>
            <button
              onClick={() => setMainView("agents")}
              className={`rounded-[22px] px-4 py-1.5 transition ${mainView === "agents" ? "bg-slate-900 text-white shadow" : "text-slate-600 hover:bg-slate-100"}`}
            >
              Agents
            </button>
          </div>

          {mainView === "simulation" ? (
            <SimulationCharts metrics={activeMetrics} chartTitle={chartTitle} />
          ) : (
            <AgentInsights metrics={activeMetrics} fallbackMetrics={metrics} />
          )}
        </section>
      </div>
    </DashboardLayout>
  );
}
