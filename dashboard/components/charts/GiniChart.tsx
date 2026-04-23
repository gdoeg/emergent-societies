"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { MetricEntry } from "@/lib/api";

interface GiniChartProps {
  data: MetricEntry[];
}

export default function GiniChart({ data }: GiniChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex h-[220px] items-center justify-center rounded-xl border border-dashed border-white/10 bg-white/5 text-sm text-white/50">
        No data yet - run simulation to generate insights.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
        <XAxis dataKey="tick" tick={{ fontSize: 11, fill: "rgba(255,255,255,0.4)" }} stroke="rgba(255,255,255,0.1)" tickMargin={8} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "rgba(255,255,255,0.4)" }} stroke="rgba(255,255,255,0.1)" tickMargin={8} />
        <Tooltip
          contentStyle={{ borderRadius: 8, borderColor: "rgba(34,211,238,0.2)", fontSize: 12, backgroundColor: "#0f172a", color: "#fff" }}
        />
        <Line
          type="monotone"
          dataKey="gini"
          stroke="#22d3ee"
          strokeWidth={3}
          dot={false}
          name="Gini"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
