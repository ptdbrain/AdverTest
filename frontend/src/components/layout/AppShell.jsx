"use client";

import React from "react";
import Sidebar from "./Sidebar";
import TopNavigation from "./TopNavigation";

export default function AppShell({ children }) {
  return (
    <div className="min-h-screen bg-[#F8FAFC] text-[#0F172A] flex">
      {/* Fixed Sidebar */}
      <Sidebar />

      {/* Main App Container */}
      <div className="flex-1 flex flex-col pl-[220px] transition-all duration-300 min-w-0">
        {/* Top Navbar */}
        <TopNavigation />

        {/* Dynamic Page Content */}
        <main className="mt-[58px] p-5 md:p-6 flex-1 min-w-0 overflow-y-auto">
          <div className="max-w-[1600px] mx-auto space-y-5">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
