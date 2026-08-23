"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Shield,
  LayoutDashboard,
  Settings2,
  Crosshair,
  Eye,
  BarChart3,
  FileText,
  ShieldCheck,
  Users,
  Sun,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";

const MENU_ITEMS = [
  { name: "Tổng quan hệ thống", href: "/dashboard", icon: LayoutDashboard },
  { name: "Cấu hình bài toán", href: "/experiments/new", icon: Settings2 },
  { name: "Cấu hình tấn công", href: "/experiments/EXP-2025-0512-001/attack", icon: Crosshair },
  { name: "Kết quả trực quan", href: "/experiments/EXP-2025-0512-001/results", icon: Eye },
  { name: "Metrics & Benchmark", href: "/benchmark", icon: BarChart3 },
  { name: "Phân tích & Báo cáo", href: "/analysis", icon: FileText },
  { name: "Phòng thủ", href: "/defense", icon: ShieldCheck },
  { name: "Quản trị", href: "/admin", icon: Users },
];

export default function Sidebar() {
  const rawPathname = usePathname();
  const pathname = rawPathname || "/";
  const [collapsed, setCollapsed] = useState(false);

  const isActive = (href) => {
    if (!pathname) return false;
    if (href === "/dashboard" && (pathname === "/" || pathname === "/dashboard")) return true;
    if (href.startsWith("/experiments/new") && pathname.startsWith("/experiments/new")) return true;
    if (href.includes("/attack") && pathname.includes("/attack")) return true;
    if (href.includes("/results") && pathname.includes("/results")) return true;
    return pathname.startsWith(href);
  };

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 bottom-0 bg-white border-r border-slate-200 z-30 flex flex-col transition-all duration-300 select-none",
        collapsed ? "w-[68px]" : "w-[220px]"
      )}
    >
      {/* Brand Header */}
      <div className="h-[64px] flex items-center gap-2.5 px-4 border-b border-slate-100">
        <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-sm flex-shrink-0">
          <Shield className="w-5 h-5" />
        </div>
        {!collapsed && (
          <div className="overflow-hidden">
            <h1 className="text-[17px] font-bold text-[#153E9D] tracking-tight leading-tight whitespace-nowrap">
              AdversAI Lab
            </h1>
            <p className="text-[10px] text-slate-500 font-medium leading-tight whitespace-nowrap">
              Đánh giá & phòng thủ AI
            </p>
          </div>
        )}
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 px-2.5 py-3 space-y-1 overflow-y-auto">
        {MENU_ITEMS.map((item) => {
          const active = isActive(item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.name}
              href={item.href}
              title={collapsed ? item.name : undefined}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-all duration-150",
                active
                  ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-sm shadow-blue-500/20"
                  : "text-[#253858] hover:bg-blue-50/70 hover:text-blue-700"
              )}
            >
              <Icon className={cn("w-[18px] h-[18px] flex-shrink-0", active ? "text-white" : "text-slate-500")} />
              {!collapsed && <span className="truncate leading-none">{item.name}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Sidebar Footer */}
      <div className="p-3 border-t border-slate-100 space-y-2">
        {/* User Card */}
        {!collapsed ? (
          <div className="flex items-center justify-between p-2 rounded-lg bg-slate-50 border border-slate-150">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-full bg-blue-600 text-white text-xs font-bold flex items-center justify-center">
                AD
              </div>
              <div className="text-left">
                <div className="text-xs font-semibold text-slate-800 leading-tight">admin</div>
                <div className="text-[10px] text-slate-500">Quản trị viên</div>
              </div>
            </div>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          </div>
        ) : (
          <div className="w-8 h-8 mx-auto rounded-full bg-blue-600 text-white text-xs font-bold flex items-center justify-center">
            AD
          </div>
        )}

        {/* Light Theme Tag & Collapse Toggle */}
        <div className={cn("flex items-center justify-between text-[11px] text-slate-500 px-1 pt-1", collapsed && "justify-center")}>
          {!collapsed && (
            <span className="flex items-center gap-1 font-medium">
              <Sun className="w-3.5 h-3.5 text-amber-500" /> Chế độ sáng
            </span>
          )}
          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            className="flex items-center gap-1 text-slate-500 hover:text-slate-800 transition-colors p-1 rounded hover:bg-slate-100"
            title={collapsed ? "Mở rộng sidebar" : "Thu gọn sidebar"}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <span className="flex items-center gap-1"><ChevronLeft className="w-3.5 h-3.5" /> Thu gọn</span>}
          </button>
        </div>
      </div>
    </aside>
  );
}
