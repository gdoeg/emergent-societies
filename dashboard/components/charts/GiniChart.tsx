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
      <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-emerald-200 bg-emerald-50/60 text-sm text-emerald-700">
        No data yet - run simulation to generate insights.
      </div>
    );
  }

  return (
    <div className="h-full min-h-24 lg:min-h-26">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#d7e9e3" vertical={false} />
          <XAxis dataKey="tick" tick={{ fontSize: 11, fill: "#4b5f67" }} stroke="#c6ddd6" tickMargin={8} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#4b5f67" }} stroke="#c6ddd6" tickMargin={8} />
          <Tooltip contentStyle={{ borderRadius: 16, borderColor: "#d7e9e3", fontSize: 12 }} />
          <Line type="monotone" dataKey="gini" stroke="#0f766e" strokeWidth={3} dot={false} name="Gini" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
