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
import ChartCard from "@/components/ChartCard";

interface StrategyOverTimeChartProps {
  metricsHistory: MetricEntry[];
}

export default function StrategyOverTimeChart({ metricsHistory }: StrategyOverTimeChartProps) {
  const hasData = metricsHistory.some((m) => m.strategy_counts != null);

  if (!hasData) {
    return (
      <ChartCard title="Strategy Over Time" delay={0.25} bodyClassName="h-full">
        <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 p-4 text-center text-sm text-slate-400">
          No strategy data yet — run simulation to see trends.
        </div>
      </ChartCard>
    );
  }

  const chartData = metricsHistory
    .filter((m) => m.strategy_counts != null)
    .map((m) => {
      const cooperate = m.strategy_counts!.cooperate ?? 0;
      const defect = m.strategy_counts!.defect ?? 0;
      const total = cooperate + defect;
      return {
        tick: m.tick,
        cooperate: total > 0 ? Math.round((cooperate / total) * 100) : 0,
        defect: total > 0 ? Math.round((defect / total) * 100) : 0,
      };
    });

  return (
    <ChartCard title="Strategy Over Time" delay={0.25} bodyClassName="h-full">
      <div className="h-full min-h-24 lg:min-h-26">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#d7e9e3" vertical={false} />
            <XAxis
              dataKey="tick"
              tick={{ fontSize: 11, fill: "#4b5f67" }}
              stroke="#c6ddd6"
              tickMargin={8}
              label={{ value: "Step", position: "insideBottomRight", offset: -4, fontSize: 10, fill: "#4b5f67" }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "#4b5f67" }}
              stroke="#c6ddd6"
              tickMargin={8}
              tickFormatter={(v) => `${v}%`}
              domain={[0, 100]}
            />
            <Tooltip
              contentStyle={{ borderRadius: 16, borderColor: "#d7e9e3", fontSize: 12 }}
              formatter={(value) => [`${value}%`]}
            />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
            <Line
              type="monotone"
              dataKey="cooperate"
              stroke="#10b981"
              strokeWidth={2.5}
              dot={false}
              name="Cooperate %"
            />
            <Line
              type="monotone"
              dataKey="defect"
              stroke="#ef4444"
              strokeWidth={2.5}
              dot={false}
              name="Defect %"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}
