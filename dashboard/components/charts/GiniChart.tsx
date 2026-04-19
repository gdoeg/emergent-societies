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
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#dcfce7" />
        <XAxis dataKey="tick" tick={{ fontSize: 11 }} stroke="#86efac" />
        <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} stroke="#86efac" />
        <Tooltip
          contentStyle={{ borderRadius: 8, borderColor: "#bbf7d0", fontSize: 12 }}
        />
        <Line
          type="monotone"
          dataKey="gini"
          stroke="#16a34a"
          strokeWidth={2}
          dot={false}
          name="Gini"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
