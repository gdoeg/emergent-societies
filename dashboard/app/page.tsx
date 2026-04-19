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
    { label: "Current Tick", value: latest ? latest.tick : "—", tint: "rgba(22,163,74,0.1)" },
    {
      label: "Gini (Inequality)",
      value: latest ? latest.gini.toFixed(4) : "—",
      tint: "rgba(21,128,61,0.12)",
    },
    {
      label: "Total Wealth",
      value: latest ? latest.total_wealth.toFixed(0) : "—",
      tint: "rgba(34,197,94,0.11)",
    },
    {
      label: "Network Density",
      value: latest ? latest.network_density.toFixed(4) : "—",
      tint: "rgba(5,150,105,0.12)",
    },
  ];

  return (
    <DashboardLayout>
      <div className="space-y-10">
        {/* Stat strip */}
        <div>
          <p className="mb-4 text-sm font-semibold uppercase tracking-wide text-green-700">Key Metrics</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {metricsConfig.map(({ label, value, tint }, i) => (
            <motion.div
              key={label}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
              className="bg-white rounded-2xl border border-green-100 shadow-lg hover:shadow-xl hover:-translate-y-0.5 transition-all duration-200 p-6"
              style={{ backgroundImage: `linear-gradient(180deg, ${tint} 0%, rgba(255,255,255,1) 44%)` }}
            >
              <div className="mb-4 h-1.5 w-16 rounded-full bg-green-500/40" />
              <p className="text-sm text-green-600 uppercase tracking-wide">{label}</p>
              <p className="mt-3 text-3xl font-semibold text-green-900 leading-none">
                {loading && !latest ? (
                  <span className="inline-block h-8 w-20 rounded-md bg-green-100 animate-pulse" />
                ) : (
                  value
                )}
              </p>
            </motion.div>
          ))}
        </div>
        </div>

        {/* Controls */}
        <div>
          <p className="mb-4 text-sm font-semibold uppercase tracking-wide text-green-700">Simulation Controls</p>
          <Controls onUpdate={refresh} />
        </div>

        {/* Status / error */}
        {loading && (
          <div className="rounded-xl border border-green-200 bg-white/80 px-4 py-3 text-sm text-green-700 shadow-sm">
            Loading metrics...
          </div>
        )}
        {error && (
          <div className="rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm px-4 py-3">
            {error}
          </div>
        )}

        {/* Charts grid */}
        <div>
          <p className="mb-4 text-sm font-semibold uppercase tracking-wide text-green-700">Analytics</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <ChartCard title="Wealth Inequality (Gini)" delay={0.1}>
            <GiniChart data={metrics} />
          </ChartCard>

          <ChartCard title="Total Wealth" delay={0.15}>
            <WealthChart data={metrics} />
          </ChartCard>

          <ChartCard title="Power Metrics" delay={0.2}>
            <PowerChart data={metrics} />
          </ChartCard>

          <ChartCard title="Network Metrics" delay={0.25}>
            <NetworkChart data={metrics} />
          </ChartCard>

          <ChartCard title="Wealth Distribution (latest tick)" delay={0.3}>
            <DistributionChart latest={latest} />
          </ChartCard>
        </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
