"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { MetricEntry } from "@/lib/api";

interface DistributionChartProps {
  latest: MetricEntry | null;
}

const BUCKET_COUNT = 20;

function bucketWealth(values: number[]): { bucket: string; count: number }[] {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const bucketSize = range / BUCKET_COUNT;

  const counts = Array(BUCKET_COUNT).fill(0);
  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / bucketSize), BUCKET_COUNT - 1);
    counts[idx]++;
  }

  return counts.map((count, i) => ({
    bucket: (min + i * bucketSize).toFixed(0),
    count,
  }));
}

export default function DistributionChart({ latest }: DistributionChartProps) {
  const chartData = latest ? bucketWealth(latest.wealth_distribution) : [];

  if (chartData.length === 0) {
    return (
      <div className="flex h-[220px] items-center justify-center rounded-xl border border-dashed border-green-200 bg-green-50/50 text-sm text-green-600">
        No data yet - run simulation to generate insights.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#bbf7d0" />
        <XAxis dataKey="bucket" tick={{ fontSize: 10, fill: "#166534" }} stroke="#86efac" tickMargin={8} />
        <YAxis tick={{ fontSize: 11, fill: "#166534" }} stroke="#86efac" tickMargin={8} />
        <Tooltip
          contentStyle={{ borderRadius: 8, borderColor: "#bbf7d0", fontSize: 12 }}
        />
        <Bar dataKey="count" fill="#16a34a" name="Agents" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
