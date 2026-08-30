"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Bell, HelpCircle, User, Sparkles, LogIn, Menu } from "lucide-react";
import { cn } from "@/lib/utils";
import UserMenu from "@/components/UserMenu";
import { useAuth } from "@/context/AuthContext";
import { useSidebar } from "@/context/SidebarContext";

const NAV_TABS = [
  { name: "Tổng quan hệ thống", href: "/dashboard" },
  { name: "Cấu hình bài toán", href: "/experiments/new" },
  { name: "Cấu hình tấn công", href: "/experiments/EXP-2025-0512-001/attack" },
  { name: "Kết quả trực quan", href: "/experiments/EXP-2025-0512-001/results" },
  { name: "Metrics & Benchmark", href: "/benchmark" },
  { name: "Phân tích & Báo cáo", href: "/analysis" },
  { name: "Phòng thủ", href: "/defense" },
  { name: "Quản trị", href: "/admin" },
];

export default function TopNavigation() {
  const rawPathname = usePathname();
  const pathname = rawPathname || "/";
  const { user, isAuthenticated, openAuthModal } = useAuth();
  const { isCollapsed, toggleSidebar } = useSidebar();

  const isActive = (href) => {
    if (!pathname) return false;
    if (href === "/dashboard" && (pathname === "/" || pathname === "/dashboard")) return true;
    if (href.startsWith("/experiments/new") && pathname.startsWith("/experiments/new")) return true;
    if (href.includes("/attack") && pathname.includes("/attack")) return true;
    if (href.includes("/results") && pathname.includes("/results")) return true;
    return pathname.startsWith(href);
  };

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

          {/* Horizontal Tabs */}
          <nav className="flex items-center space-x-1 h-full overflow-x-auto no-scrollbar">
            {NAV_TABS.map((tab) => {
              const active = isActive(tab.href);
              return (
                <Link
                  key={tab.name}
                  href={tab.href}
                  className={cn(
                    "px-3.5 min-h-[44px] flex items-center text-[13px] font-medium transition-colors border-b-2 whitespace-nowrap",
                    active
                      ? "border-blue-600 text-blue-700 bg-blue-50/50 font-semibold"
                      : "border-transparent text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                  )}
                >
                  {tab.name}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Right Utilities */}
        <div className="flex items-center gap-3 pl-4">
          {/* Simulation Notice */}
          <span className="hidden xl:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
            <Sparkles className="w-3 h-3" /> AdversAI Lab v2.5
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

        {/* User Profile / Auth Control */}
        <div className="flex items-center gap-2 pl-2 border-l border-slate-200">
          <UserMenu />
        </div>
      </div>
    </header>
  );
}
