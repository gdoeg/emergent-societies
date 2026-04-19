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

  // Initial load
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Polling every second
  useEffect(() => {
    const id = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  return (
    <DashboardLayout>
      {/* Stat strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        {[
          { label: "Current Tick", value: latest ? latest.tick : "—" },
          {
            label: "Gini (Inequality)",
            value: latest ? latest.gini.toFixed(4) : "—",
          },
          {
            label: "Total Wealth",
            value: latest ? latest.total_wealth.toFixed(0) : "—",
          },
          {
            label: "Network Density",
            value: latest ? latest.network_density.toFixed(4) : "—",
          },
        ].map(({ label, value }, i) => (
          <motion.div
            key={label}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: i * 0.07 }}
            className="bg-white rounded-2xl border border-green-100 shadow-sm px-5 py-4"
          >
            <p className="text-xs font-medium text-green-500 uppercase tracking-wide">
              {label}
            </p>
            <p className="mt-1 text-2xl font-bold text-green-900">{value}</p>
          </motion.div>
        ))}
      </div>

      {/* Controls */}
      <div className="mb-6">
        <Controls onUpdate={refresh} />
      </div>

      {/* Status / error */}
      {loading && (
        <p className="text-sm text-green-500 mb-4">Loading metrics…</p>
      )}
      {error && (
        <div className="rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm px-4 py-3 mb-4">
          {error}
        </div>
      )}

      {/* Charts grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
    </DashboardLayout>
  );
}
