"use client";

import { ReactNode } from "react";

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="relative min-h-screen px-5 py-12 md:px-10 md:py-14">
      <div className="max-w-6xl mx-auto relative z-10">
        <header className="mb-10 md:mb-14 rounded-2xl border border-white/60 bg-white/70 backdrop-blur-sm shadow-lg px-6 py-6 md:px-8 md:py-7">
          <p className="inline-flex items-center rounded-full border border-green-200 bg-green-50 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-green-700">
            Live Dashboard
          </p>
          <h1 className="text-4xl font-bold tracking-tight text-green-800">
            Emergent Societies
          </h1>
          <p className="text-green-600 mt-3 text-base">
            Real-time economic simulation
          </p>
        </header>
        {children}
      </div>
    </div>
  );
}
