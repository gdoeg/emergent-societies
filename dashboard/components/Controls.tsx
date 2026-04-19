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
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="bg-white rounded-2xl shadow-lg border border-green-100 p-6 flex flex-wrap items-end gap-5 hover:shadow-xl hover:-translate-y-0.5 transition-all duration-200"
    >
      <div className="flex flex-col gap-2 min-w-[130px]">
        <label className="text-sm font-medium text-green-700" htmlFor="steps-input">
          Steps
        </label>
        <input
          id="steps-input"
          type="number"
          min={1}
          max={1000}
          value={steps}
          onChange={(e) => setSteps(Number(e.target.value))}
          className="w-32 rounded-xl border border-green-200 bg-white px-3 py-2 text-sm text-green-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-green-400"
        />
      </div>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={handleRun}
          disabled={running}
          className="rounded-xl bg-green-500 hover:bg-green-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md disabled:opacity-50 transition-all duration-200"
        >
          {running ? "Running..." : "Run Simulation"}
        </button>

        <button
          onClick={handleReset}
          disabled={resetting}
          className="rounded-xl border border-green-300 bg-white px-5 py-2.5 text-sm font-semibold text-green-700 hover:bg-green-50 disabled:opacity-50 transition-all duration-200"
        >
          {resetting ? "Resetting..." : "Reset"}
        </button>
      </div>
    </motion.div>
  );
}
