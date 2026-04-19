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
      <div className="flex h-[220px] items-center justify-center rounded-xl border border-dashed border-green-200 bg-green-50/50 text-sm text-green-600">
        No data yet - run simulation to generate insights.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#bbf7d0" />
        <XAxis dataKey="tick" tick={{ fontSize: 11, fill: "#166534" }} stroke="#86efac" tickMargin={8} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#166534" }} stroke="#86efac" tickMargin={8} />
        <Tooltip
          contentStyle={{ borderRadius: 8, borderColor: "#bbf7d0", fontSize: 12 }}
        />
        <Line
          type="monotone"
          dataKey="gini"
          stroke="#16a34a"
          strokeWidth={3}
          dot={false}
          name="Gini"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
