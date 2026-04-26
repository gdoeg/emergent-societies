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
      <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-emerald-200 bg-emerald-50/60 text-sm text-emerald-700">
        No data yet - run simulation to generate insights.
      </div>
    );
  }

  return (
    <div className="h-full min-h-[200px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#d7e9e3" vertical={false} />
          <XAxis dataKey="bucket" tick={{ fontSize: 10, fill: "#4b5f67" }} stroke="#c6ddd6" tickMargin={8} />
          <YAxis tick={{ fontSize: 11, fill: "#4b5f67" }} stroke="#c6ddd6" tickMargin={8} />
          <Tooltip contentStyle={{ borderRadius: 16, borderColor: "#d7e9e3", fontSize: 12 }} />
          <Bar dataKey="count" fill="#0f766e" name="Agents" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
