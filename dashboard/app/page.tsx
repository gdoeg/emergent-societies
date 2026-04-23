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
    { label: "Current Tick", value: latest ? latest.tick : "—" },
    { label: "Gini (Inequality)", value: latest ? latest.gini.toFixed(4) : "—" },
    { label: "Total Wealth", value: latest ? latest.total_wealth.toFixed(0) : "—" },
    { label: "Network Density", value: latest ? latest.network_density.toFixed(4) : "—" },
  ];

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header row */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-white">Emergent Societies</h1>
            <p className="text-sm text-white/50 mt-0.5">Real-time simulation</p>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1.5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-400" />
            </span>
            <span className="text-xs font-semibold text-cyan-400">Live Simulation</span>
          </div>
        </div>

        {/* Status / error */}
        {loading && (
          <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/70">
            Loading metrics...
          </div>
        )}
        {error && (
          <div className="rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm px-4 py-3">
            {error}
          </div>
        )}

        {/* Metrics row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {metricsConfig.map(({ label, value }, i) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              whileHover={{ scale: 1.02 }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
              className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-6 shadow-[0_0_40px_rgba(34,211,238,0.05)] transition duration-300"
            >
              <p className="text-xs uppercase tracking-wide text-white/50">{label}</p>
              <p className="mt-3 text-2xl font-bold text-white leading-none">
                {loading && !latest ? (
                  <span className="inline-block h-7 w-20 rounded-md bg-white/10 animate-pulse" />
                ) : (
                  value
                )}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Main content grid: primary chart + controls */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <ChartCard title="Wealth Inequality (Gini)" delay={0.1}>
              <GiniChart data={metrics} />
            </ChartCard>
          </div>
          <div className="lg:col-span-1">
            <div className="space-y-6">
              <p className="text-xs uppercase tracking-wide text-white/50">Simulation Controls</p>
              <Controls onUpdate={refresh} />
            </div>
          </div>
        </div>

        {/* Wide chart */}
        <ChartCard title="Total Wealth" delay={0.15}>
          <WealthChart data={metrics} />
        </ChartCard>

        {/* Bottom row: two charts side-by-side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <ChartCard title="Network Metrics" delay={0.2}>
            <NetworkChart data={metrics} />
          </ChartCard>

          <ChartCard title="Power Metrics" delay={0.25}>
            <PowerChart data={metrics} />
          </ChartCard>
        </div>

        {/* Wealth distribution */}
        <ChartCard title="Wealth Distribution (latest tick)" delay={0.3}>
          <DistributionChart latest={latest} />
        </ChartCard>
      </div>
    </DashboardLayout>
  );
}
