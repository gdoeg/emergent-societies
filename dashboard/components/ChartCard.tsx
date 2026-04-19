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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      className="bg-white rounded-2xl shadow-sm border border-green-100 p-6"
    >
      <h2 className="text-sm font-semibold text-green-700 uppercase tracking-wide mb-4">
        {title}
      </h2>
      {children}
    </motion.div>
  );
}
