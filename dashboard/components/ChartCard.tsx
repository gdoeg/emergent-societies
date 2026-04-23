"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

interface ChartCardProps {
  title: string;
  children: ReactNode;
  delay?: number;
}

export default function ChartCard({ title, children, delay = 0 }: ChartCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-6 shadow-[0_0_40px_rgba(34,211,238,0.05)] transition duration-300"
    >
      <div className="mb-5 flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-white">
          {title}
        </h2>
        <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 shadow-[0_0_0_6px_rgba(34,211,238,0.16)]" />
      </div>
      {children}
    </motion.div>
  );
}
