"use client";

import { useSidebar } from "@/context/SidebarContext";
import { cn } from "@/lib/utils";
import AuthGate from "./AuthGate";
import Sidebar from "./Sidebar";

export default function AppShell({ children }) {
  const { isCollapsed } = useSidebar();

  return (
    <div className="min-h-screen bg-[var(--app-bg)] text-[var(--app-text)] flex">
      {/* Fixed Collapsible Sidebar */}
      <Sidebar />

      {/* Main App Container that smoothly adapts padding */}
      <div
        className={cn(
          "flex-1 flex flex-col transition-all duration-300 min-w-0 overflow-x-hidden",
          isCollapsed ? "pl-0 md:pl-[68px]" : "pl-0 md:pl-[220px]",
        )}
      >
        {/* Dynamic Page Content */}
        <main className="p-4 md:p-6 flex-1 min-w-0 overflow-y-auto">
          <div className="max-w-[1700px] mx-auto space-y-5">
            <AuthGate>{children}</AuthGate>
          </div>
        </main>
      </div>
    </div>
  );
}
