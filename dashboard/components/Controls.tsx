"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { runSimulation, resetSimulation, runMultipleSimulations } from "@/lib/api";

export type ViewMode = "current" | "aggregate";

interface ControlsProps {
  onUpdate: () => void;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  className?: string;
}

function joinClasses(...values: Array<string | undefined>) {
  return values.filter(Boolean).join(" ");
}

export default function Controls({ onUpdate, viewMode, onViewModeChange, className }: ControlsProps) {
  const [steps, setSteps] = useState(10);
  const [numRuns, setNumRuns] = useState(3);
  const [running, setRunning] = useState(false);
  const [runningMultiple, setRunningMultiple] = useState(false);
  const [resetting, setResetting] = useState(false);

  async function handleRun() {
    setRunning(true);
    try {
      await runSimulation(steps);
      onUpdate();
    } finally {
      setRunning(false);
    }
  }

  async function handleRunMultiple() {
    setRunningMultiple(true);
    try {
      await runMultipleSimulations(steps, numRuns);
      onViewModeChange("aggregate");
      onUpdate();
    } finally {
      setRunningMultiple(false);
    }
  }

  async function handleReset() {
    setResetting(true);
    try {
      await resetSimulation();
      onUpdate();
    } finally {
      setResetting(false);
    }
  }

  const busy = running || runningMultiple || resetting;

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={joinClasses(
        "rounded-[28px] border border-white/60 bg-[linear-gradient(180deg,rgba(255,255,255,0.9),rgba(239,246,245,0.94))] p-2 shadow-[0_24px_80px_rgba(16,42,51,0.12)] backdrop-blur-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_30px_90px_rgba(16,42,51,0.16)]",
        className,
      )}
    >
      <div className="mb-2 flex items-start justify-between gap-2">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-500">
            Controls
          </p>
          <h2 className="mt-0.5 text-sm font-semibold text-slate-900 lg:text-[15px]">
            Simulation Runbook
          </h2>
        </div>
        <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-emerald-700">
          Live
        </span>
      </div>

      {/* View mode toggle */}
      <div className="mb-2">
        <p className="mb-1 text-xs font-medium text-slate-700">Data source</p>
        <div className="flex rounded-2xl border border-slate-200 bg-white/90 p-0.5 text-xs font-semibold">
          <button
            onClick={() => onViewModeChange("current")}
            className={joinClasses(
              "flex-1 rounded-[14px] px-2 py-1 transition",
              viewMode === "current"
                ? "bg-slate-900 text-white shadow"
                : "text-slate-600 hover:bg-slate-100",
            )}
          >
            Current
          </button>
          <button
            onClick={() => onViewModeChange("aggregate")}
            className={joinClasses(
              "flex-1 rounded-[14px] px-2 py-1 transition",
              viewMode === "aggregate"
                ? "bg-slate-900 text-white shadow"
                : "text-slate-600 hover:bg-slate-100",
            )}
          >
            Aggregate
          </button>
        </div>
      </div>

      <div className="grid gap-2 xl:grid-cols-1">
        <div className="flex min-w-40 flex-col gap-1.5">
          <label className="text-xs font-medium text-slate-700" htmlFor="steps-input">
            Simulation steps
          </label>
          <input
            id="steps-input"
            type="number"
            min={1}
            max={1000}
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
            className="w-full rounded-2xl border border-slate-200 bg-white/90 px-3 py-1.5 text-sm text-slate-900 shadow-sm outline-none transition focus:border-emerald-400 focus:ring-2 focus:ring-emerald-200"
          />
        </div>

        <div className="flex min-w-40 flex-col gap-1.5">
          <label className="text-xs font-medium text-slate-700" htmlFor="num-runs-input">
            Number of runs
          </label>
          <input
            id="num-runs-input"
            type="number"
            min={1}
            max={20}
            value={numRuns}
            onChange={(e) => setNumRuns(Number(e.target.value))}
            className="w-full rounded-2xl border border-slate-200 bg-white/90 px-3 py-1.5 text-sm text-slate-900 shadow-sm outline-none transition focus:border-emerald-400 focus:ring-2 focus:ring-emerald-200"
          />
        </div>

        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={handleRun}
            disabled={busy}
            className="rounded-2xl bg-slate-900 px-3 py-1.5 text-sm font-semibold text-white shadow-lg shadow-slate-900/20 transition hover:bg-slate-800 disabled:opacity-50"
          >
            {running ? "Running..." : "Run"}
          </button>

          <button
            onClick={handleRunMultiple}
            disabled={busy}
            className="rounded-2xl bg-teal-700 px-3 py-1.5 text-sm font-semibold text-white shadow-lg shadow-teal-900/20 transition hover:bg-teal-600 disabled:opacity-50"
          >
            {runningMultiple ? "Running..." : `Run ×${numRuns}`}
          </button>

          <button
            onClick={handleReset}
            disabled={busy}
            className="rounded-2xl border border-slate-200 bg-white/90 px-3 py-1.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
          >
            {resetting ? "Resetting..." : "Reset"}
          </button>
        </div>
      </div>
    </motion.div>
  );
}
