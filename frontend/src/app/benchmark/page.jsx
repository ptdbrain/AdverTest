"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  BarChart3,
  Filter,
  Download,
  Calendar,
  Layers,
  Crosshair,
  Shield,
  TrendingDown,
  TrendingUp,
  Award,
  Zap,
  Clock,
  ArrowRight,
  RotateCcw,
  Trash2,
  CheckCircle2,
  AlertCircle,
  Plus,
  Info,
  Sliders,
  FolderOpen,
  Eye,
  Activity,
} from "lucide-react";
import PageHeader from "@/components/layout/PageHeader";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import MetricSparkline from "@/components/metrics/MetricSparkline";
import ComparisonBarChart from "@/components/metrics/ComparisonBarChart";
import TrendLineChart from "@/components/metrics/TrendLineChart";
import RobustnessRadar from "@/components/metrics/RobustnessRadar";
import { cn } from "@/lib/utils";
import { getApiBase, listSessions, deleteRunFromSession } from "@/lib/api";
import EndSessionModal from "@/components/EndSessionModal";
import { useRouter } from "next/navigation";

export default function BenchmarkPage() {
  const router = useRouter();
  const [sessionsList, setSessionsList] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState("EXP-2025-0512-001");
  const [activeSession, setActiveSession] = useState(null);
  const [isLoadingSession, setIsLoadingSession] = useState(true);

  const [isResetModalOpen, setIsResetModalOpen] = useState(false);
  const [isResetSuccess, setIsResetSuccess] = useState(false);
  const [isEndModalOpen, setIsEndModalOpen] = useState(false);

  // 1. Fetch Sessions List on Mount
  const loadSessions = () => {
    setIsLoadingSession(true);
    listSessions()
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setSessionsList(data);
          const current = data.find((s) => s.id === activeSessionId) || data[0];
          setActiveSessionId(current.id);
          setActiveSession(current);
        }
      })
      .catch((err) => console.warn("Error fetching sessions:", err))
      .finally(() => setIsLoadingSession(false));
  };

  useEffect(() => {
    loadSessions();
  }, []);

  const handleSwitchSession = (sessionId) => {
    setActiveSessionId(sessionId);
    const found = sessionsList.find((s) => s.id === sessionId);
    if (found) {
      setActiveSession(found);
    }
  };

  const handleDeleteRun = (runId) => {
    deleteRunFromSession(activeSessionId, runId)
      .then((updatedSession) => {
        setActiveSession(updatedSession);
        setSessionsList((prev) =>
          prev.map((s) => (s.id === updatedSession.id ? updatedSession : s))
        );
      })
      .catch((err) => console.warn("Error deleting run:", err));
  };

  const runs = activeSession?.runs || [];

  // Calculate dynamic session aggregates
  const totalRuns = runs.length;
  const bestRun = runs.length > 0 ? [...runs].sort((a, b) => b.robustness_score - a.robustness_score)[0] : null;
  const worstRun = runs.length > 0 ? [...runs].sort((a, b) => a.robustness_score - b.robustness_score)[0] : null;
  const avgRobustness = runs.length > 0
    ? (runs.reduce((acc, r) => acc + r.robustness_score, 0) / runs.length).toFixed(1)
    : "50.0";
  const avgMapDrop = runs.length > 0
    ? (runs.reduce((acc, r) => acc + r.map_drop_pct, 0) / runs.length).toFixed(1)
    : "0.0";

  // KPIs
  const kpis = [
    {
      name: "Tổng số lần thử nghiệm trong phiên",
      after: `${totalRuns} lần chạy`,
      before: "0 lần",
      drop: `${totalRuns} bài test`,
      pct: "100%",
      sparkline: [1, 2, Math.max(3, totalRuns)],
      isDanger: false,
    },
    {
      name: "Độ sụt giảm mAP@0.5 Trung Bình",
      after: `-${avgMapDrop}%`,
      before: "0.0%",
      drop: `-${avgMapDrop} pp`,
      pct: `-${avgMapDrop}%`,
      sparkline: runs.map((r) => r.map_drop_pct),
      isDanger: parseFloat(avgMapDrop) > 40,
    },
    {
      name: "Điểm Bền Vững Trung Bình (Robustness)",
      after: `${avgRobustness} / 100`,
      before: "100.0",
      drop: `-${(100 - parseFloat(avgRobustness)).toFixed(1)}`,
      pct: `${avgRobustness}%`,
      sparkline: runs.map((r) => r.robustness_score),
      isDanger: parseFloat(avgRobustness) < 50,
    },
    {
      name: "Đòn Tác Động Gây Thiệt Hại Nặng Nhất",
      after: worstRun ? worstRun.attack_name : "Chưa có",
      before: "Clean State",
      drop: worstRun ? `-${worstRun.map_drop_pct}% mAP` : "0%",
      pct: worstRun ? `Score ${worstRun.robustness_score}` : "N/A",
      sparkline: [82, 45, worstRun ? Math.round(worstRun.attacked_map * 100) : 11],
      isDanger: true,
    },
  ];

  // Prepare chart data for Multi-Run Comparison Bar Chart
  const chartCategories = runs.map((r, i) => `Lần ${i + 1}: ${r.attack_name.split("(")[0].trim()}`);
  const cleanMapSeries = runs.map((r) => Math.round(r.clean_map * 100));
  const attackedMapSeries = runs.map((r) => Math.round(r.attacked_map * 100));

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Metrics & Benchmark (Đánh giá định lượng đa lần chạy)"
        subtitle="So sánh đối đầu giữa các lần thực thi tấn công trong cùng phiên làm việc và đánh giá độ bền vững toàn diện của mô hình."
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Metrics & Benchmark" },
        ]}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm" onClick={() => setIsEndModalOpen(true)} className="text-red-600 hover:bg-red-50 hover:border-red-200 bg-white shadow-sm border-slate-200 font-bold">
              Kết thúc phiên (Khóa)
            </Button>
            <Link href="/experiments/new">
              <Button variant="primary" size="sm" icon={Plus}>
                Tạo phiên làm việc mới
              </Button>
            </Link>
          </div>
        }
      />

      {/* 1. SESSION SELECTOR & CONTEXT BANNER */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-blue-950 via-slate-900 to-slate-900 border border-blue-800/60 text-white shadow-md space-y-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-600/30 border border-blue-500 text-blue-400 flex items-center justify-center font-bold">
              <FolderOpen className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-bold text-blue-400 uppercase tracking-wide">
                Phiên làm việc đang xem (Active Session Workspace)
              </div>
              <div className="text-sm font-bold text-white mt-0.5 flex items-center gap-2">
                <span>{activeSession?.name || "Đang tải phiên..."}</span>
                <span className="font-mono text-xs px-2 py-0.5 rounded bg-blue-900/60 text-blue-300 border border-blue-700">
                  {activeSessionId}
                </span>
              </div>
            </div>
          </div>

          {/* Session Switcher Dropdown */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-semibold whitespace-nowrap">Chuyển phiên:</span>
            <select
              value={activeSessionId}
              onChange={(e) => handleSwitchSession(e.target.value)}
              className="text-xs py-1.5 px-3 rounded-lg border border-slate-700 bg-slate-800 text-slate-100 font-semibold shadow-inner focus:ring-2 focus:ring-blue-500"
            >
              {sessionsList.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.runs?.length || 0} lần chạy)
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-300">
          <div className="flex flex-wrap items-center gap-3">
            <span>Mô hình: <strong className="text-white">{activeSession?.model_name || "YOLO11s"}</strong></span>
            <span className="text-slate-600">•</span>
            <span>Bộ dữ liệu: <strong className="text-blue-300">{activeSession?.dataset_name || "KITTI"}</strong></span>
            <span className="text-slate-600">•</span>
            <span>Tổng lần chạy: <strong className="text-emerald-400">{totalRuns} lần</strong></span>
          </div>

          <Link href={`/experiments/${activeSessionId}/attack`}>
            <span className="text-xs text-blue-400 hover:text-blue-300 font-bold flex items-center gap-1 cursor-pointer">
              + Chạy thêm đòn tấn công mới vào phiên này <ArrowRight className="w-3.5 h-3.5" />
            </span>
          </Link>
        </div>
      </div>

      {/* 2. 4 DYNAMIC SESSION KPI CARDS */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {kpis.map((kpi) => (
          <div
            key={kpi.name}
            className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs flex flex-col justify-between space-y-3"
          >
            <div>
              <div className="text-xs text-slate-500 font-medium line-clamp-1">
                {kpi.name}
              </div>
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-xl font-bold text-slate-900">{kpi.after}</span>
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-slate-100">
              <span
                className={cn(
                  "text-xs font-bold px-2 py-0.5 rounded flex items-center gap-1 font-mono",
                  kpi.isDanger
                    ? "bg-red-50 text-red-600 border border-red-200"
                    : "bg-emerald-50 text-emerald-700 border border-emerald-200"
                )}
              >
                {kpi.drop}
              </span>
              <div className="w-16">
                <MetricSparkline data={kpi.sparkline} isNegative={kpi.isDanger} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 3. MULTI-RUN LEADERBOARD & COMPARISON TABLE */}
      <Card
        title={`Bảng so sánh đối đầu các lần chạy trong phiên [${activeSession?.name || activeSessionId}]`}
        subtitle="Liệt kê chi tiết từng kịch bản tấn công đã thử nghiệm và đo lường độ sụt giảm tương ứng"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3">Lần chạy</th>
                <th className="py-2.5 px-3">Kịch bản tấn công</th>
                <th className="py-2.5 px-3">Thời gian ghi nhận</th>
                <th className="py-2.5 px-3 text-right">Clean mAP</th>
                <th className="py-2.5 px-3 text-right">Robust mAP</th>
                <th className="py-2.5 px-3 text-right">Độ sụt giảm</th>
                <th className="py-2.5 px-3 text-right">Điểm Bền vững</th>
                <th className="py-2.5 px-3 text-right">PSNR</th>
                <th className="py-2.5 px-3 text-right">Độ trễ</th>
                <th className="py-2.5 px-3 text-center">Thao tác</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {runs.map((r, idx) => (
                <tr key={r.id} className="hover:bg-blue-50/40 transition-colors font-medium">
                  <td className="py-3 px-3">
                    <span className="w-5 h-5 rounded-full bg-blue-600 text-white font-bold flex items-center justify-center text-[10px]">
                      {idx + 1}
                    </span>
                  </td>
                  <td className="py-3 px-3 font-bold text-slate-900">
                    <div className="flex items-center gap-1.5">
                      <span className="text-red-600 font-bold">{r.attack_name}</span>
                      <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-slate-100 text-slate-600 border border-slate-200">
                        Cấp {r.severity}
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-3 font-mono text-slate-500 text-[11px]">{r.timestamp}</td>
                  <td className="py-3 px-3 text-right font-mono text-slate-700 font-semibold">
                    {(r.clean_map * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-3 text-right font-mono font-bold text-red-600">
                    {(r.attacked_map * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-3 text-right font-mono font-bold text-red-600">
                    -{r.map_drop_pct}%
                  </td>
                  <td className="py-3 px-3 text-right font-mono font-bold text-amber-600">
                    {r.robustness_score} / 100
                  </td>
                  <td className="py-3 px-3 text-right font-mono text-slate-600">{r.psnr}</td>
                  <td className="py-3 px-3 text-right font-mono text-slate-600">{r.inference_ms} ms</td>
                  <td className="py-3 px-3 text-center">
                    <div className="flex items-center justify-center gap-1.5">
                      <Link href={`/experiments/${activeSessionId}/results`}>
                        <button className="p-1 text-blue-600 hover:bg-blue-50 rounded" title="Xem kết quả trực quan">
                          <Eye className="w-3.5 h-3.5" />
                        </button>
                      </Link>
                      <button
                        onClick={() => handleDeleteRun(r.id)}
                        className="p-1 text-red-600 hover:bg-red-50 rounded"
                        title="Xóa lần chạy này"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* 4. VISUAL COMPARISON CHARTS ROW (BAR CHART & RADAR / DEGRADATION) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Multi-Run Performance Bar Chart (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <Card
            title="Biểu đồ so sánh mAP@0.5 qua các lần chạy (Clean vs Attacked)"
            subtitle="Trực quan hóa độ sụt giảm chính xác của từng kịch bản thử nghiệm"
          >
            <div className="space-y-3 text-xs pt-2">
              {runs.map((r, i) => (
                <div key={r.id} className="space-y-1 p-3 rounded-lg border border-slate-200 bg-slate-50/60">
                  <div className="flex items-center justify-between font-bold">
                    <span className="text-slate-800">Lần #{i + 1}: {r.attack_name}</span>
                    <span className="font-mono text-red-600">
                      {(r.attacked_map * 100).toFixed(1)}% (Gốc: {(r.clean_map * 100).toFixed(1)}%) · -{r.map_drop_pct}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-200 h-2.5 rounded-full overflow-hidden flex">
                    <div
                      className="bg-emerald-500 h-full"
                      style={{ width: `${r.attacked_map * 100}%` }}
                    />
                    <div
                      className="bg-red-400 h-full opacity-60"
                      style={{ width: `${(r.clean_map - r.attacked_map) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 flex items-center justify-between text-[11px] text-slate-500 pt-2 border-t border-slate-100">
              <div className="flex items-center gap-3">
                <span className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded bg-emerald-500" /> mAP Còn Lại
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded bg-red-400" /> mAP Bị Triệt Tiêu
                </span>
              </div>
              <span>Dữ liệu thực nghiệm 100%</span>
            </div>
          </Card>
        </div>

        {/* Robustness Radar & Domain Resilience (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <Card
            title="Khả năng chống chịu theo nhóm đòn (Domain Resilience)"
            subtitle="Đánh giá điểm bền vững trung bình qua các nhóm tác động"
          >
            <div className="space-y-3 text-xs">
              {[
                { domain: "🌧️ Mưa (Depth Rain)", score: 26.8, color: "bg-blue-500", desc: "Mất tương phản quang học" },
                { domain: "🌫️ Sương mù (Depth Fog)", score: 54.9, color: "bg-teal-500", desc: "Mờ đối tượng ở xa" },
                { domain: "❄️ Tuyết (Snow & Frost)", score: 34.2, color: "bg-indigo-500", desc: "Che khuất và tán xạ ánh sáng" },
                { domain: "⚡ Nhiễu đối kháng (PGD)", score: 13.4, color: "bg-red-500", desc: "Phá vỡ cấu trúc gradient mạng" },
                { domain: "💨 Mờ chuyển động (Motion Blur)", score: 48.0, color: "bg-amber-500", desc: "Rung lắc khi xe di chuyển" },
              ].map((d) => (
                <div key={d.domain} className="p-2.5 rounded-lg border border-slate-100 bg-slate-50/60 space-y-1">
                  <div className="flex items-center justify-between font-bold">
                    <span className="text-slate-800">{d.domain}</span>
                    <span className="font-mono text-blue-700">{d.score} / 100</span>
                  </div>
                  <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                    <div className={cn("h-full rounded-full", d.color)} style={{ width: `${d.score}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <EndSessionModal
        isOpen={isEndModalOpen}
        onClose={() => setIsEndModalOpen(false)}
        session={activeSession}
        onConfirm={() => loadSessions()}
      />
    </div>
  );
}
