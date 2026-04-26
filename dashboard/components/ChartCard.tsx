"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

interface ChartCardProps {
  title: string;
  children: ReactNode;
  delay?: number;
  className?: string;
  bodyClassName?: string;
}

function joinClasses(...values: Array<string | undefined>) {
  return values.filter(Boolean).join(" ");
}

export default function ChartCard({
  title,
  children,
  delay = 0,
  className,
  bodyClassName,
}: ChartCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      className={joinClasses(
        "group h-full rounded-[28px] border border-white/70 bg-[linear-gradient(180deg,rgba(255,255,255,0.94),rgba(241,248,247,0.96))] p-4 shadow-[0_24px_80px_rgba(16,42,51,0.12)] backdrop-blur-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-[0_30px_90px_rgba(16,42,51,0.16)]",
        className,
      )}
    >
      <div className="flex h-full flex-col">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-slate-500">
            Analytics
          </p>
            <h2 className="mt-1.5 text-base font-semibold text-slate-900">
            {title}
            </h2>
          </div>
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 shadow-[0_0_0_6px_rgba(16,185,129,0.14)]" />
        </div>
        <div className={joinClasses("flex-1 min-h-0", bodyClassName)}>{children}</div>
      </div>
    </motion.div>
  );
}
