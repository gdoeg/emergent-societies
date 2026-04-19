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
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#dcfce7" />
        <XAxis dataKey="tick" tick={{ fontSize: 11 }} stroke="#86efac" />
        <YAxis tick={{ fontSize: 11 }} stroke="#86efac" />
        <Tooltip
          contentStyle={{ borderRadius: 8, borderColor: "#bbf7d0", fontSize: 12 }}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line
          type="monotone"
          dataKey="total_wealth"
          stroke="#16a34a"
          strokeWidth={2}
          dot={false}
          name="Total Wealth"
        />
        <Line
          type="monotone"
          dataKey="avg_wealth"
          stroke="#4ade80"
          strokeWidth={2}
          dot={false}
          name="Avg Wealth"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
