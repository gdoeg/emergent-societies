"use client";

import ChartCard from "@/components/ChartCard";
import GiniChart from "@/components/charts/GiniChart";
import WealthChart from "@/components/charts/WealthChart";
import PowerChart from "@/components/charts/PowerChart";
import NetworkChart from "@/components/charts/NetworkChart";
import DistributionChart from "@/components/charts/DistributionChart";
import { MetricEntry } from "@/lib/api";

interface SimulationChartsProps {
  metrics: MetricEntry[];
  chartTitle: (base: string) => string;
}

export default function SimulationCharts({ metrics, chartTitle }: SimulationChartsProps) {
  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  return (
    <div className="grid flex-1 min-h-0 min-w-0 grid-cols-1 auto-rows-fr gap-1.5 lg:grid-cols-2">
      <ChartCard title={chartTitle("Wealth Distribution (latest tick)")} delay={0.3} className="col-span-1" bodyClassName="h-full">
        <DistributionChart latest={latest} />
      </ChartCard>

      <ChartCard title={chartTitle("Wealth Inequality (Gini)")} delay={0.1} className="col-span-1" bodyClassName="h-full">
        <GiniChart data={metrics} />
      </ChartCard>

      <ChartCard title={chartTitle("Total Wealth")} delay={0.15} className="col-span-1" bodyClassName="h-full">
        <WealthChart data={metrics} />
      </ChartCard>

      <ChartCard title={chartTitle("Power Metrics")} delay={0.2} className="col-span-1" bodyClassName="h-full">
        <PowerChart data={metrics} />
      </ChartCard>

      <ChartCard title={chartTitle("Network Metrics")} delay={0.25} className="col-span-1" bodyClassName="h-full">
        <NetworkChart data={metrics} />
      </ChartCard>
    </div>
  );
}
