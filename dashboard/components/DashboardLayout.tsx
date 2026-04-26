"use client";

import { ReactNode } from "react";

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="h-screen w-full overflow-hidden">
      <div className="dashboard-scale relative z-10 h-full w-full">{children}</div>
    </div>
  );
}
