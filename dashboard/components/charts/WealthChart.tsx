"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { MetricEntry } from "@/lib/api";

interface WealthChartProps {
  data: MetricEntry[];
}

export default function WealthChart({ data }: WealthChartProps) {
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
        <YAxis tick={{ fontSize: 11, fill: "#166534" }} stroke="#86efac" tickMargin={8} />
        <Tooltip
          contentStyle={{ borderRadius: 8, borderColor: "#bbf7d0", fontSize: 12 }}
        />
        <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
        <Line
          type="monotone"
          dataKey="total_wealth"
          stroke="#16a34a"
          strokeWidth={3}
          dot={false}
          name="Total Wealth"
        />
        <Line
          type="monotone"
          dataKey="avg_wealth"
          stroke="#22c55e"
          strokeWidth={3}
          dot={false}
          name="Avg Wealth"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
