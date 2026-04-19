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
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      className="group bg-white/95 backdrop-blur-sm rounded-2xl shadow-lg border border-green-100 p-6 hover:shadow-xl hover:-translate-y-0.5 transition-all duration-200"
    >
      <div className="mb-5 flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-green-800">
          {title}
        </h2>
        <span className="h-2.5 w-2.5 rounded-full bg-green-500 shadow-[0_0_0_6px_rgba(34,197,94,0.16)]" />
      </div>
      {children}
    </motion.div>
  );
}
