"use client";

import React from "react";
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
  PanelLeftClose,
  PanelLeftOpen,
  ClipboardCheck,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/context/AuthContext";
import { useSidebar } from "@/context/SidebarContext";

const MENU_ITEMS = [
  { name: "Tổng quan hệ thống", href: "/dashboard", icon: LayoutDashboard },
  { name: "Cấu hình bài toán", href: "/experiments/new", icon: Settings2 },
  { name: "Cấu hình tấn công", href: "/experiments/EXP-2025-0512-001/attack", icon: Crosshair },
  { name: "Kết quả trực quan", href: "/experiments/EXP-2025-0512-001/results", icon: Eye },
  { name: "Metrics & Benchmark", href: "/benchmark", icon: BarChart3 },
  { name: "Phân tích & Báo cáo", href: "/analysis", icon: FileText },
  { name: "Thẩm định & Review", href: "/reviews", icon: ClipboardCheck },
  { name: "Phòng thủ", href: "/defense", icon: ShieldCheck },
  { name: "Quản trị", href: "/admin", icon: Users },
];

export default function Sidebar() {
  const rawPathname = usePathname();
  const pathname = rawPathname || "/";
  const { isCollapsed, toggleSidebar } = useSidebar();
  const collapsed = isCollapsed;
  const [isTouchViewport, setIsTouchViewport] = React.useState(false);
  const { user, isAuthenticated, openAuthModal } = useAuth();

  const isActive = (href) => {
    if (!pathname) return false;
    if (href === "/dashboard" && (pathname === "/" || pathname === "/dashboard")) return true;
    if (href.startsWith("/experiments/new") && pathname.startsWith("/experiments/new")) return true;
    if (href.includes("/attack") && pathname.includes("/attack")) return true;
    if (href.includes("/results") && pathname.includes("/results")) return true;
    return pathname.startsWith(href);
  };

  const displayName = user?.display_name || "Admin";
  const roleName = user?.role === "ADMIN" ? "Quản trị viên" : "Nghiên cứu viên";
  const initials = displayName
    .split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  // Keyboard accessibility: Close drawer on Escape key
  React.useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape" && !collapsed) {
        toggleSidebar();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [collapsed, toggleSidebar]);

  // A CSS breakpoint alone is not enough here: browser zoom can make a desktop
  // viewport look narrow and leave the mobile backdrop above the whole app.
  // Only touch-first narrow devices need a click-blocking drawer backdrop.
  React.useEffect(() => {
    if (typeof window.matchMedia !== "function") return undefined;
    const media = window.matchMedia("(max-width: 767px) and (pointer: coarse)");
    const update = () => setIsTouchViewport(media.matches);
    update();
    media.addEventListener?.("change", update);
    return () => media.removeEventListener?.("change", update);
  }, []);

  const closeMobileDrawer = () => {
    if (isTouchViewport && !collapsed) toggleSidebar();
  };

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {isTouchViewport && !collapsed && (
        <div
          role="presentation"
          aria-hidden="true"
          onClick={toggleSidebar}
          className="fixed inset-0 bg-slate-900/50 backdrop-blur-xs z-30 md:hidden transition-opacity"
        />
      )}

      <aside
        role="navigation"
        aria-label="Thanh điều hướng chính"
        className={cn(
          "fixed left-0 top-0 bottom-0 bg-white border-r border-slate-200 z-40 flex flex-col transition-all duration-300 select-none",
          collapsed
            ? "hidden md:flex md:w-[68px]"
            : "w-[260px] md:w-[220px] shadow-2xl md:shadow-none"
        )}
      >
      {/* FLOATING EXPAND / COLLAPSE BUTTON ON THE BORDER (ALWAYS VISIBLE & ACCESSIBLE) */}
      <button
        type="button"
        onClick={toggleSidebar}
        className="absolute -right-3.5 top-5 w-7 h-7 bg-white border border-slate-300 hover:border-blue-500 rounded-full shadow-md flex items-center justify-center text-slate-600 hover:text-blue-600 hover:scale-110 z-40 transition-all cursor-pointer"
        title={collapsed ? "Mở rộng sidebar (Click để mở)" : "Thu gọn sidebar (Click để thu gọn)"}
      >
        {collapsed ? (
          <ChevronRight className="w-4 h-4 text-blue-600 font-bold" />
        ) : (
          <ChevronLeft className="w-4 h-4 text-slate-600" />
        )}
      </button>

      {/* Brand Header */}
      <div className="h-[64px] flex items-center justify-between px-3.5 border-b border-slate-100">
        <div className="flex items-center gap-2.5 overflow-hidden">
          <div
            onClick={toggleSidebar}
            className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-sm flex-shrink-0 cursor-pointer hover:bg-blue-700 transition-colors"
            title={collapsed ? "Mở rộng sidebar" : "AdversAI Lab"}
          >
            <Shield className="w-5 h-5" />
          </div>
          {!collapsed && (
            <div className="overflow-hidden">
              <h1 className="text-[16px] font-bold text-[#153E9D] tracking-tight leading-tight whitespace-nowrap">
                AdversAI Lab
              </h1>
              <p className="text-[10px] text-slate-500 font-medium leading-tight whitespace-nowrap">
                Đánh giá & phòng thủ AI
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 px-2 py-3 space-y-1 overflow-y-auto">
        {MENU_ITEMS.map((item) => {
          const active = isActive(item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.name}
              href={item.href}
              onClick={closeMobileDrawer}
              title={collapsed ? item.name : undefined}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-all duration-150",
                active
                  ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-sm shadow-blue-500/20"
                  : "text-[#253858] hover:bg-blue-50/70 hover:text-blue-700",
                collapsed && "justify-center px-0"
              )}
            >
              <Icon className={cn("w-[18px] h-[18px] flex-shrink-0", active ? "text-white" : "text-slate-500")} />
              {!collapsed && <span className="truncate leading-none">{item.name}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Sidebar Footer */}
      <div className="p-2.5 border-t border-slate-100 space-y-2 bg-slate-50/50">
        {/* User Card */}
        {!collapsed ? (
          <button
            type="button"
            onClick={openAuthModal}
            className="w-full flex items-center justify-between p-2 rounded-lg bg-white border border-slate-200 hover:bg-slate-50 transition-colors text-left shadow-2xs"
            title="Nhấp để đổi tài khoản / xem phân quyền"
          >
            <div className="flex items-center gap-2 overflow-hidden">
              <div className="w-7 h-7 rounded-full bg-blue-600 text-white text-xs font-bold flex items-center justify-center flex-shrink-0">
                {initials || "AD"}
              </div>
              <div className="text-left overflow-hidden">
                <div className="text-xs font-semibold text-slate-800 leading-tight truncate">
                  {displayName}
                </div>
                <div className="text-[10px] text-slate-500 truncate">{roleName}</div>
              </div>
            </div>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
          </button>
        ) : (
          <button
            type="button"
            onClick={openAuthModal}
            className="w-8 h-8 mx-auto rounded-full bg-blue-600 text-white text-xs font-bold flex items-center justify-center hover:opacity-90 shadow-xs"
            title={`Tài khoản: ${displayName} (${roleName})`}
          >
            {initials || "AD"}
          </button>
        )}

        {/* Expand / Collapse Button in Footer */}
        <div className={cn("flex items-center justify-between text-[11px] text-slate-500 px-1 pt-0.5", collapsed && "justify-center")}>
          {!collapsed && (
            <span className="flex items-center gap-1 font-medium text-[10px]">
              <Sun className="w-3 h-3 text-amber-500" /> Sáng
            </span>
          )}
          <button
            type="button"
            onClick={toggleSidebar}
            className="flex items-center gap-1 text-slate-600 hover:text-blue-700 font-semibold transition-colors p-1 rounded hover:bg-slate-200/60 text-[11px]"
            title={collapsed ? "Mở rộng sidebar" : "Thu gọn sidebar"}
          >
            {collapsed ? (
              <span className="flex items-center gap-0.5 text-blue-600 font-bold"><ChevronRight className="w-4 h-4" /> Mở</span>
            ) : (
              <span className="flex items-center gap-1"><ChevronLeft className="w-3.5 h-3.5" /> Thu gọn</span>
            )}
          </button>
        </div>
      </div>
    </aside>
  </>
  );
}
