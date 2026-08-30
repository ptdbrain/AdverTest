"use client";

import React from "react";
import Link from "next/link";
import { Bell, HelpCircle, Menu } from "lucide-react";
import { cn } from "@/lib/utils";
import UserMenu from "@/components/UserMenu";
import { useSidebar } from "@/context/SidebarContext";
import ThemeToggle from "@/components/ThemeToggle";

export default function TopNavigation() {
  const { isCollapsed, toggleSidebar } = useSidebar();


  return (
    <header
      className={cn(
        "h-[58px] bg-white border-b border-slate-200 fixed top-0 right-0 z-20 flex items-center justify-between px-3 md:px-5 transition-all duration-300",
        isCollapsed ? "left-0 md:left-[68px]" : "left-0 md:left-[220px]"
      )}
    >
        <div className="flex items-center h-full min-w-0">
          {/* Mobile hamburger menu toggle */}
          <button
            type="button"
            onClick={toggleSidebar}
            className="md:hidden min-w-[44px] min-h-[44px] flex items-center justify-center p-2 mr-1 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors shrink-0"
            aria-label="Toggle Sidebar"
          >
            <Menu className="w-5 h-5" />
          </button>

        </div>

        {/* Right Utilities */}
        <div className="flex items-center gap-3 pl-4">
          {/* Simulation Notice */}
          <span className="hidden xl:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
            AdverTest
          </span>

          {/* Notifications */}
          <button
            type="button"
            className="relative p-2 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition-colors"
            title="Thông báo hệ thống"
          >
            <Bell className="w-4 h-4" />
            <span className="absolute top-1.5 right-1.5 w-4 h-4 rounded-full bg-red-500 text-white text-[9px] font-bold flex items-center justify-center border-2 border-white">
              3
            </span>
          </button>

          {/* Settings Link */}
          <Link
            href="/settings"
            className="p-2 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition-colors"
            title="Cài đặt & Tích hợp"
          >
            <HelpCircle className="w-4 h-4" />
          </Link>
          <ThemeToggle compact />

        {/* User Profile / Auth Control */}
        <div className="flex items-center gap-2 pl-2 border-l border-slate-200">
          <UserMenu />
        </div>
      </div>
    </header>
  );
}
