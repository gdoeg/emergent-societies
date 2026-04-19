"use client";

import { ReactNode } from "react";

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="min-h-screen bg-green-50 px-4 py-8 md:px-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-green-900">
            Emergent Societies
          </h1>
          <p className="text-green-600 mt-1 text-sm">
            Real-time simulation dashboard
          </p>
        </header>
        {children}
      </div>
    </div>
  );
}
