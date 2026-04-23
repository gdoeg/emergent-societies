"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { runSimulation, resetSimulation } from "@/lib/api";

interface ControlsProps {
  onUpdate: () => void;
}

export default function Controls({ onUpdate }: ControlsProps) {
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
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-6 flex flex-wrap items-end gap-5 shadow-[0_0_40px_rgba(34,211,238,0.05)] transition duration-300"
    >
      <div className="flex flex-col gap-2 min-w-[130px]">
        <label className="text-xs font-medium uppercase tracking-wide text-white/50" htmlFor="steps-input">
          Steps
        </label>
        <input
          id="steps-input"
          type="number"
          min={1}
          max={1000}
          value={steps}
          onChange={(e) => setSteps(Number(e.target.value))}
          className="w-32 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-white shadow-sm focus:outline-none focus:ring-2 focus:ring-cyan-400/50"
        />
      </div>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleRun}
          disabled={running}
          className="rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 px-5 py-2.5 text-sm font-semibold text-cyan-400 shadow-md disabled:opacity-50 transition duration-300"
        >
          {running ? "Running..." : "Run Simulation"}
        </button>

        <button
          onClick={handleReset}
          disabled={resetting}
          className="rounded-xl border border-white/10 bg-white/5 px-5 py-2.5 text-sm font-semibold text-white/70 hover:bg-white/10 disabled:opacity-50 transition duration-300"
        >
          {resetting ? "Resetting..." : "Reset"}
        </button>
      </div>
    </motion.div>
  );
}
