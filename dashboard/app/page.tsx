"use client";

import { useEffect, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { fetchMetrics, MetricEntry } from "@/lib/api";
import DashboardLayout from "@/components/DashboardLayout";
import ChartCard from "@/components/ChartCard";
import Controls from "@/components/Controls";
import GiniChart from "@/components/charts/GiniChart";
import WealthChart from "@/components/charts/WealthChart";
import PowerChart from "@/components/charts/PowerChart";
import NetworkChart from "@/components/charts/NetworkChart";
import DistributionChart from "@/components/charts/DistributionChart";

const POLL_INTERVAL_MS = 1000;

export default function Home() {
  const [metrics, setMetrics] = useState<MetricEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await fetchMetrics();
      setMetrics(data);
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

  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;
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
      label: "Network Density",
      value: latest ? latest.network_density.toFixed(4) : "—",
      tint: "rgba(51,65,85,0.08)",
    },
  ];

  const railStats = [
    { label: "Avg Wealth", value: latest ? latest.avg_wealth.toFixed(1) : "—" },
    { label: "Avg Power", value: latest ? latest.avg_power.toFixed(3) : "—" },
    { label: "Max Power", value: latest ? latest.max_power.toFixed(3) : "—" },
    { label: "Agents", value: latest ? latest.wealth_distribution.length : "—" },
  ];

  const statusTone = error
    ? {
        label: "Backend disconnected",
        className: "border-rose-200 bg-rose-50/90 text-rose-700",
      }
    : loading && !latest
      ? {
          label: "Fetching live metrics",
          className: "border-amber-200 bg-amber-50/90 text-amber-700",
        }
      : {
          label: "Streaming metrics",
          className: "border-emerald-200 bg-emerald-50/90 text-emerald-700",
        };

  return (
    <DashboardLayout>
      <div className="grid h-full w-full grid-cols-1 gap-4 p-4 xl:grid-cols-[260px_1fr]">
        <aside className="grid min-h-0 gap-4 xl:grid-rows-[auto_auto_auto_minmax(0,1fr)]">
          <motion.section
            initial={{ opacity: 0, x: -24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.45, ease: "easeOut" }}
            className="overflow-hidden rounded-4xl border border-slate-900/10 bg-[linear-gradient(145deg,rgba(16,42,51,0.96),rgba(23,63,71,0.92))] p-4 text-white shadow-[0_28px_90px_rgba(16,42,51,0.24)]"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="inline-flex items-center rounded-full border border-white/15 bg-white/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-100/90">
                Live Dashboard
              </span>
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_0_8px_rgba(52,211,153,0.12)]" />
            </div>
            <h1 className="mt-5 max-w-sm text-3xl font-bold tracking-tight text-white">
              Emergent Societies
            </h1>
            <p className="mt-3 max-w-md text-sm leading-6 text-slate-200/86">
              An operational view of emergent dynamics in an active simulation.
            </p>
            <div className="mt-5 grid grid-cols-2 gap-3">
              <div className="rounded-2xl border border-white/10 bg-white/6 px-4 py-3">
                <p className="text-[11px] uppercase tracking-[0.22em] text-slate-300">Samples</p>
                <p className="mt-1.5 text-xl font-semibold text-white">{metrics.length}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/6 px-4 py-3">
                <p className="text-[11px] uppercase tracking-[0.22em] text-slate-300">Mode</p>
                <p className="mt-1.5 text-xl font-semibold text-white">Live</p>
              </div>
            </div>
          </motion.section>

          <Controls onUpdate={refresh} className="xl:h-full" />

          <motion.section
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.08, ease: "easeOut" }}
            className="rounded-[28px] border border-white/70 bg-white/85 p-4 shadow-[0_24px_80px_rgba(16,42,51,0.12)] backdrop-blur-sm"
          >
            <div className="flex flex-wrap items-center gap-3">
              <span className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] ${statusTone.className}`}>
                {statusTone.label}
              </span>
              {latest && (
                <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-600">
                  Tick {latest.tick}
                </span>
              )}
            </div>
            <p className="mt-4 text-sm leading-6 text-slate-600">
              {error ?? "The dashboard is polling every second and the cards resize to use the full canvas on large displays."}
            </p>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.12, ease: "easeOut" }}
            className="rounded-[28px] border border-white/70 bg-white/88 p-4 shadow-[0_24px_80px_rgba(16,42,51,0.12)] backdrop-blur-sm xl:min-h-0"
          >
            <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-500">
              Quick View
            </p>
            <div className="mt-4 grid grid-cols-2 gap-3">
              {railStats.map(({ label, value }) => (
                <div key={label} className="rounded-2xl border border-slate-200/70 bg-slate-50/70 px-4 py-3">
                  <p className="text-[11px] uppercase tracking-[0.22em] text-slate-500">{label}</p>
                  <p className="mt-1.5 text-lg font-semibold text-slate-900">{value}</p>
                </div>
              ))}
            </div>
          </motion.section>
        </aside>

        <section className="flex min-h-0 min-w-0 flex-col">
          <div className="grid h-22.5 grid-cols-4 gap-4">
            {metricsConfig.map(({ label, value, tint }, i) => (
              <motion.div
                key={label}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: i * 0.08 }}
                className="rounded-[28px] border border-white/70 p-4 shadow-[0_24px_80px_rgba(16,42,51,0.12)] backdrop-blur-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_30px_90px_rgba(16,42,51,0.16)]"
                style={{ backgroundImage: `linear-gradient(180deg, ${tint} 0%, rgba(255,255,255,0.96) 46%)` }}
              >
                <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">{label}</p>
                <p className="mt-2 text-2xl font-semibold leading-none text-slate-950">
                  {loading && !latest ? (
                    <span className="inline-block h-6 w-16 animate-pulse rounded-md bg-slate-200" />
                  ) : (
                    value
                  )}
                </p>
              </motion.div>
            ))}
          </div>

          <div className="mt-4 grid flex-1 min-h-0 min-w-0 grid-cols-3 auto-rows-fr gap-4">
            <ChartCard title="Wealth Distribution (latest tick)" delay={0.3} className="col-span-1 row-span-2" bodyClassName="h-full">
              <DistributionChart latest={latest} />
            </ChartCard>

            <ChartCard title="Wealth Inequality (Gini)" delay={0.1} className="col-span-1" bodyClassName="h-full">
              <GiniChart data={metrics} />
            </ChartCard>

            <ChartCard title="Total Wealth" delay={0.15} className="col-span-1" bodyClassName="h-full">
              <WealthChart data={metrics} />
            </ChartCard>

            <ChartCard title="Power Metrics" delay={0.2} className="col-span-1" bodyClassName="h-full">
              <PowerChart data={metrics} />
            </ChartCard>

            <ChartCard title="Network Metrics" delay={0.25} className="col-span-1" bodyClassName="h-full">
              <NetworkChart data={metrics} />
            </ChartCard>
          </div>
        </section>
      </div>
    </DashboardLayout>
  );
}
