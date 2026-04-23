"use client";

import { ReactNode } from "react";

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="relative min-h-screen">
      <div className="max-w-[1400px] mx-auto px-6 py-6 relative z-10">
        {children}
      </div>
    </div>
  );
}
