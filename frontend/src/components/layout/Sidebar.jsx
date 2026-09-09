"use client";

import {
  BarChart3,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  Crosshair,
  Eye,
  FileText,
  FolderKanban,
  LayoutDashboard,
  LogOut,
  Moon,
  Settings2,
  Shield,
  ShieldCheck,
  Sun,
  Users,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { useProject } from "@/context/ProjectContext";
import { useSidebar } from "@/context/SidebarContext";
import { useTheme } from "@/context/ThemeContext";
import { cn } from "@/lib/utils";

// The experiment session id is user-defined when creating a project, so the
// experiment-scoped menu links must not hardcode one id. Resolution order:
// the id in the current URL → the last active experiment (localStorage) →
// the default demo experiment.
const DEFAULT_EXPERIMENT_ID = "EXP-2025-0512-001";

function resolveActiveExperimentId(pathname) {
  const match = pathname?.match(/^\/experiments\/([^/]+)/);
  if (match && match[1] !== "new") return decodeURIComponent(match[1]);
  try {
    const parsed = JSON.parse(window.localStorage.getItem("adversai_active_experiment") || "null");
    if (parsed?.expName) return String(parsed.expName);
  } catch {
    // Corrupted cache entry — fall through to the default experiment id.
  }
  return DEFAULT_EXPERIMENT_ID;
}

const MENU_ITEMS = [
  { name: "Tổng quan hệ thống", href: "/dashboard", icon: LayoutDashboard },
  { name: "Cấu hình bài toán", href: "/experiments/new", icon: Settings2 },
  { name: "Cấu hình tấn công", href: (expId) => `/experiments/${expId}/attack`, icon: Crosshair },
  { name: "Kết quả trực quan", href: (expId) => `/experiments/${expId}/results`, icon: Eye },
  { name: "Metrics & Benchmark", href: "/benchmark", icon: BarChart3 },
  { name: "Phân tích & Báo cáo", href: "/analysis", icon: FileText },
  { name: "Thẩm định & Review", href: "/reviews", icon: ClipboardCheck },
  { name: "Phòng thủ", href: "/defense", icon: ShieldCheck },
  { name: "Quản trị", href: "/admin", icon: Users },
];

const ROLE_META = {
  ADMIN: {
    label: "Quản trị viên",
    badge: "bg-[rgba(168,85,247,0.15)] text-[#a855f7]",
  },
  ENGINEER: {
    label: "Nghiên cứu viên",
    badge: "bg-[rgba(56,189,248,0.15)] text-[#0ea5e9]",
  },
};

export default function Sidebar() {
  const rawPathname = usePathname();
  const pathname = rawPathname || "/";
  const { isCollapsed, toggleSidebar } = useSidebar();
  const collapsed = isCollapsed;
  const { user, isAuthenticated, openAuthModal, logout } = useAuth();
  const { projects, activeProjectId, isLoadingProjects, setActiveProjectId } = useProject();
  const { resolvedTheme, setTheme } = useTheme();
  const isLight = resolvedTheme === "light";
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);
  // The default matches the SSR/hydration render; resolve the user's real
  // experiment id (URL → localStorage) after mount.
  const [activeExperimentId, setActiveExperimentId] = useState(DEFAULT_EXPERIMENT_ID);
  useEffect(() => {
    setActiveExperimentId(resolveActiveExperimentId(pathname));
  }, [pathname]);
  const scopedHref = (href) =>
    activeProjectId
      ? `${href}${href.includes("?") ? "&" : "?"}project_id=${encodeURIComponent(activeProjectId)}`
      : href;

  const isActive = (href) => {
    if (!pathname) return false;
    if (href === "/dashboard" && (pathname === "/" || pathname === "/dashboard")) return true;
    if (href.startsWith("/experiments/new") && pathname.startsWith("/experiments/new")) return true;
    if (href.includes("/attack") && pathname.includes("/attack")) return true;
    if (href.includes("/results") && pathname.includes("/results")) return true;
    return pathname.startsWith(href);
  };

  const displayName = user?.display_name || "Admin";
  const roleKey = user?.role === "ADMIN" ? "ADMIN" : "ENGINEER";
  const roleMeta = ROLE_META[roleKey];
  const initials = displayName
    .split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  // Close the user menu on outside click or Escape; Escape also closes the mobile drawer.
  useEffect(() => {
    const handleMouseDown = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false);
      }
    };
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        setMenuOpen(false);
        if (!collapsed) toggleSidebar();
      }
    };
    document.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [collapsed, toggleSidebar]);

  const toggleTheme = () => setTheme(isLight ? "dark" : "light");
  const themeLabel = isLight ? "Chuyển sang chế độ tối" : "Chuyển sang chế độ sáng";
  const activeProject = projects.find((project) => project.id === activeProjectId);

  // Avatars come from Google accounts or DiceBear demo profiles (SVG), which
  // next/image cannot serve without dangerouslyAllowSVG — plain <img> is used.
  const avatar = user?.avatar_url ? (
    // biome-ignore lint/performance/noImgElement: Google/DiceBear avatar URLs (incl. SVG) need a plain img
    <img src={user.avatar_url} alt={displayName} className="w-full h-full object-cover" />
  ) : (
    initials || "AD"
  );

  return (
    <aside
      aria-label="Thanh điều hướng chính"
      className={cn(
        "fixed left-0 top-0 bottom-0 bg-[var(--app-surface)] border-r border-[var(--app-border)] z-40 flex flex-col transition-all duration-300 select-none",
        collapsed ? "hidden md:flex md:w-[68px]" : "w-[260px] md:w-[220px] shadow-2xl md:shadow-none",
      )}
    >
      {/* FLOATING EXPAND / COLLAPSE BUTTON ON THE BORDER (ALWAYS VISIBLE & ACCESSIBLE) */}
      <button
        type="button"
        onClick={toggleSidebar}
        className="absolute -right-3.5 top-5 w-7 h-7 bg-[var(--app-surface)] border border-[var(--app-border)] hover:border-blue-500 rounded-full shadow-md flex items-center justify-center text-[var(--text-secondary)] hover:text-blue-500 hover:scale-110 z-40 transition-all cursor-pointer"
        title={collapsed ? "Mở rộng sidebar (Click để mở)" : "Thu gọn sidebar (Click để thu gọn)"}
      >
        {collapsed ? <ChevronRight className="w-4 h-4 text-blue-500 font-bold" /> : <ChevronLeft className="w-4 h-4" />}
      </button>

      {/* Brand Header */}
      <div className="h-[64px] flex items-center justify-between px-3.5 border-b border-[var(--app-border)]">
        <div className="flex items-center gap-2.5 overflow-hidden">
          <button
            type="button"
            onClick={toggleSidebar}
            className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-sm flex-shrink-0 cursor-pointer hover:bg-blue-700 transition-colors"
            title={collapsed ? "Mở rộng sidebar" : "AdverTest"}
            aria-label={collapsed ? "Mở rộng sidebar" : "Thu gọn sidebar"}
          >
            <Shield className="w-5 h-5" />
          </button>
          {!collapsed && (
            <div className="overflow-hidden">
              <h1 className="text-[16px] font-bold text-[var(--text-primary)] tracking-tight leading-tight whitespace-nowrap">
                AdverTest
              </h1>
              <p className="text-[10px] text-[var(--text-muted)] font-medium leading-tight whitespace-nowrap">
                Đánh giá & phòng thủ AI
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Project Selector */}
      <div className={cn("px-2.5 pt-3", collapsed && "px-0 pt-3")}>
        {!collapsed ? (
          <div>
            <label
              htmlFor="sidebar-project-select"
              className="flex items-center gap-1.5 px-1 pb-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--text-muted)]"
            >
              <FolderKanban className="w-3 h-3" /> Dự án
            </label>
            <select
              id="sidebar-project-select"
              value={activeProjectId}
              onChange={(event) => setActiveProjectId(event.target.value)}
              disabled={isLoadingProjects || !projects.length}
              className="w-full rounded-md border border-[var(--app-border)] bg-[var(--bg-secondary)] px-2 py-1.5 text-xs font-medium text-[var(--text-primary)] outline-none focus:border-blue-500 disabled:text-[var(--text-muted)]"
            >
              <option value="">{isLoadingProjects ? "Đang tải…" : "Chọn project"}</option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <button
            type="button"
            onClick={toggleSidebar}
            className="w-full flex items-center justify-center py-2 rounded-lg text-[var(--text-muted)] hover:text-blue-500 hover:bg-[var(--primary-light)] transition-colors"
            title={`Dự án: ${activeProject?.name || "Chưa chọn"}`}
            aria-label="Dự án đang hoạt động"
          >
            <FolderKanban className="w-[18px] h-[18px]" />
          </button>
        )}
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 px-2 py-3 space-y-1 overflow-y-auto">
        {MENU_ITEMS.map((item) => {
          const itemHref = typeof item.href === "function" ? item.href(activeExperimentId) : item.href;
          const active = isActive(itemHref);
          const Icon = item.icon;
          return (
            <Link
              key={item.name}
              href={scopedHref(itemHref)}
              title={collapsed ? item.name : undefined}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-[13px] font-medium transition-all duration-150",
                active
                  ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-sm shadow-blue-500/20"
                  : "text-[var(--text-primary)] hover:bg-[var(--primary-light)] hover:text-blue-500",
                collapsed && "justify-center px-0",
              )}
            >
              <Icon
                className={cn("w-[18px] h-[18px] flex-shrink-0", active ? "text-white" : "text-[var(--text-muted)]")}
              />
              {!collapsed && <span className="truncate leading-none">{item.name}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Sidebar Footer */}
      <div className="p-2.5 border-t border-[var(--app-border)] space-y-2">
        {/* User Account Area */}
        {!isAuthenticated || !user ? (
          <button
            type="button"
            onClick={openAuthModal}
            className={cn(
              "w-full flex items-center justify-center gap-2 rounded-lg bg-gradient-to-br from-blue-500 to-blue-700 text-white text-xs font-bold py-2 shadow-sm shadow-blue-500/30 hover:from-blue-600 hover:to-blue-800 transition-colors",
              collapsed && "px-0",
            )}
            title="Đăng nhập để quản lý dự án"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
              <path
                fill="#FFF"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="#FFF"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="#FFF"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
              />
              <path
                fill="#FFF"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
              />
            </svg>
            {!collapsed && <span>Đăng nhập bằng Google</span>}
          </button>
        ) : (
          <div className="relative" ref={menuRef}>
            {!collapsed ? (
              <button
                type="button"
                onClick={() => setMenuOpen((open) => !open)}
                aria-expanded={menuOpen}
                className="w-full flex items-center gap-2 p-1.5 rounded-lg bg-[var(--bg-secondary)] border border-[var(--app-border)] hover:bg-[var(--bg-tertiary)] transition-colors text-left"
                title="Tài khoản & phân quyền"
              >
                <div className="w-7 h-7 rounded-full bg-blue-600 text-white text-xs font-bold flex items-center justify-center flex-shrink-0 overflow-hidden">
                  {avatar}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-semibold text-[var(--text-primary)] leading-tight truncate">
                    {displayName}
                  </div>
                  <span className={cn("inline-block mt-0.5 text-[10px] font-bold px-1.5 rounded", roleMeta.badge)}>
                    {roleMeta.label}
                  </span>
                </div>
                <ChevronDown
                  className={cn(
                    "w-3.5 h-3.5 flex-shrink-0 text-[var(--text-muted)] transition-transform",
                    menuOpen && "rotate-180",
                  )}
                />
              </button>
            ) : (
              <button
                type="button"
                onClick={() => setMenuOpen((open) => !open)}
                aria-expanded={menuOpen}
                className="w-8 h-8 mx-auto rounded-full bg-blue-600 text-white text-xs font-bold flex items-center justify-center hover:opacity-90 shadow-xs overflow-hidden"
                title={`Tài khoản: ${displayName} (${roleMeta.label})`}
              >
                {avatar}
              </button>
            )}

            {/* Account Dropdown (opens upward from the sidebar footer) */}
            {menuOpen && (
              <div className="absolute bottom-full left-0 right-0 mb-2 z-50 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)] shadow-xl p-2.5 space-y-2">
                <div className="border-b border-[var(--app-border)] pb-2">
                  <div className="text-[13px] font-bold text-[var(--text-primary)] truncate">{displayName}</div>
                  <div className="text-[11px] text-[var(--text-muted)] truncate">{user.email || "—"}</div>
                  <div className="text-[10px] text-blue-500 font-semibold mt-0.5">Đã đăng nhập bằng Google</div>
                </div>

                <div className="bg-[var(--bg-secondary)] border border-[var(--app-border)] rounded-md p-2">
                  <div className="text-[10px] font-semibold text-[var(--text-muted)]">Vai trò</div>
                  <div className={cn("text-xs font-bold mt-0.5", roleMeta.badge.split(" ")[1])}>{roleMeta.label}</div>
                </div>

                <div className="border-t border-[var(--app-border)] pt-2 space-y-0.5">
                  <Link
                    href="/settings"
                    onClick={() => setMenuOpen(false)}
                    className="flex items-center gap-2 px-2 py-1.5 rounded-md text-xs font-semibold text-[var(--text-primary)] hover:bg-[var(--primary-light)] transition-colors"
                  >
                    <Settings2 className="w-3.5 h-3.5 text-[var(--text-muted)]" /> Cài đặt & Tích hợp
                  </Link>
                  <Link
                    href="/admin"
                    onClick={() => setMenuOpen(false)}
                    className="flex items-center gap-2 px-2 py-1.5 rounded-md text-xs text-[var(--text-secondary)] hover:bg-[var(--primary-light)] transition-colors"
                  >
                    <Users className="w-3.5 h-3.5 text-[var(--text-muted)]" /> Quản trị
                  </Link>
                  <button
                    type="button"
                    onClick={() => {
                      logout();
                      setMenuOpen(false);
                    }}
                    className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs font-bold text-[var(--danger)] hover:bg-[rgba(239,68,68,0.12)] transition-colors text-left cursor-pointer"
                  >
                    <LogOut className="w-3.5 h-3.5" /> Đăng xuất
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Theme Toggle & Expand / Collapse */}
        <div
          className={cn(
            "flex items-center justify-between text-[11px] text-[var(--text-muted)] px-1 pt-0.5",
            collapsed && "justify-center gap-1",
          )}
        >
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={themeLabel}
            title={themeLabel}
            suppressHydrationWarning
            className="flex items-center gap-1 text-[var(--text-secondary)] hover:text-blue-500 font-semibold transition-colors p-1 rounded hover:bg-[var(--bg-tertiary)] cursor-pointer"
          >
            {isLight ? (
              <span className="flex items-center gap-0.5">
                <Moon className="w-3.5 h-3.5" /> {!collapsed && "Tối"}
              </span>
            ) : (
              <span className="flex items-center gap-0.5">
                <Sun className="w-3.5 h-3.5 text-amber-400" /> {!collapsed && "Sáng"}
              </span>
            )}
          </button>
          <button
            type="button"
            onClick={toggleSidebar}
            className="flex items-center gap-1 text-[var(--text-secondary)] hover:text-blue-500 font-semibold transition-colors p-1 rounded hover:bg-[var(--bg-tertiary)] text-[11px] cursor-pointer"
            title={collapsed ? "Mở rộng sidebar" : "Thu gọn sidebar"}
          >
            {collapsed ? (
              <span className="flex items-center gap-0.5 text-blue-500 font-bold">
                <ChevronRight className="w-4 h-4" />
              </span>
            ) : (
              <span className="flex items-center gap-1">
                <ChevronLeft className="w-3.5 h-3.5" /> Thu gọn
              </span>
            )}
          </button>
        </div>
      </div>
    </aside>
  );
}
