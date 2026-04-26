"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { runSimulation, resetSimulation } from "@/lib/api";

interface ControlsProps {
  onUpdate: () => void;
  className?: string;
}

function joinClasses(...values: Array<string | undefined>) {
  return values.filter(Boolean).join(" ");
}

export default function Controls({ onUpdate, className }: ControlsProps) {
  const [steps, setSteps] = useState(10);
  const [running, setRunning] = useState(false);
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

  async function handleReset() {
    setResetting(true);
    try {
      await resetSimulation();
      onUpdate();
    } finally {
      setResetting(false);
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={joinClasses(
        "rounded-[28px] border border-white/60 bg-[linear-gradient(180deg,rgba(255,255,255,0.9),rgba(239,246,245,0.94))] p-4 shadow-[0_24px_80px_rgba(16,42,51,0.12)] backdrop-blur-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_30px_90px_rgba(16,42,51,0.16)]",
        className,
      )}
    >
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-500">
            Controls
          </p>
          <h2 className="mt-1.5 text-base font-semibold text-slate-900">
            Simulation Runbook
          </h2>
          <p className="mt-2 max-w-sm text-sm text-slate-600">
            Advance the environment in measured steps or reset the run state instantly.
          </p>
        </div>
        <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-emerald-700">
          Live
        </span>
      </div>

      <div className="grid gap-4 xl:grid-cols-1">
        <div className="flex min-w-40 flex-col gap-2">
          <label className="text-sm font-medium text-slate-700" htmlFor="steps-input">
            Simulation steps
          </label>
          <input
            id="steps-input"
            type="number"
            min={1}
            max={1000}
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
            className="w-full rounded-2xl border border-slate-200 bg-white/90 px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-emerald-400 focus:ring-2 focus:ring-emerald-200"
          />
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            onClick={handleRun}
            disabled={running}
            className="rounded-2xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-slate-900/20 transition hover:bg-slate-800 disabled:opacity-50"
          >
            {running ? "Running..." : "Run Simulation"}
          </button>

          <button
            onClick={handleReset}
            disabled={resetting}
            className="rounded-2xl border border-slate-200 bg-white/90 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:opacity-50"
          >
            {resetting ? "Resetting..." : "Reset"}
          </button>
        </div>
      </div>
    </motion.div>
  );
}
