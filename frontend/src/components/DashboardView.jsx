"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  FlaskConical,
  Layers,
  Database,
  Crosshair,
  Shield,
  ShieldCheck,
  Plus,
  Upload,
  CloudUpload,
  FileText,
  Workflow,
  ArrowRight,
  AlertTriangle,
  AlertCircle,
  Info,
  Lightbulb,
  ExternalLink,
  Target,
  Brain,
  Scale,
  BarChart3,
  Users,
} from "lucide-react";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import MetricCard from "@/components/metrics/MetricCard";
import DonutChart from "@/components/metrics/DonutChart";
import { listRuns, getCatalogModels, getCatalogDatasets, getCatalogAttacks } from "@/lib/api";

const PIPELINE_STEPS = [
  { step: 1, title: "Cấu hình bài toán", desc: "Nạp mô hình & dữ liệu gốc", icon: "database", route: "/experiments/new" },
  { step: 2, title: "Cấu hình tấn công", desc: "Sinh biến thể đối kháng & OOD", icon: "crosshair", route: "/experiments/EXP-2025-0512-001/attack" },
  { step: 3, title: "Đánh giá Robustness", desc: "Phân tích suy giảm hiệu năng", icon: "chart", route: "/benchmark" },
  { step: 4, title: "Tôi luyện phòng thủ", desc: "Fine-tune & vá lỗi nhận diện", icon: "shield", route: "/defense" },
  { step: 5, title: "Báo cáo & Chứng nhận", desc: "Xuất báo cáo khoa học & Audit", icon: "file", route: "/analysis" },
];

const SYSTEM_MODULES = [
  { title: "Cấu hình bài toán & Mô hình", desc: "Khởi tạo thí nghiệm, quản lý dataset và checkpoint", href: "/experiments/new", badge: "Core" },
  { title: "Kịch bản tấn công đối kháng", desc: "Tùy biến tham số sương mù, mưa, nhiễu và patch", href: "/experiments/EXP-2025-0512-001/attack", badge: "Attacks" },
  { title: "Bảng so sánh & Metrics", desc: "Biểu đồ so sánh đa chiều và ma trận phân rã", href: "/benchmark", badge: "Analytics" },
  { title: "Đóng vòng phòng thủ & HITL", desc: "Hàng đợi đánh giá rủi ro và tôi luyện mô hình", href: "/defense", badge: "Defense" },
];

export default function DashboardView() {
  const [activePipelineStep, setActivePipelineStep] = useState(null);
  const [runs, setRuns] = useState([]);
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [attacks, setAttacks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    async function loadDashboardData() {
      try {
        const [runsRes, modelsRes, datasetsRes, attacksRes] = await Promise.allSettled([
          listRuns(),
          getCatalogModels(),
          getCatalogDatasets(),
          getCatalogAttacks(),
        ]);
        if (!isMounted) return;
        if (runsRes.status === "fulfilled" && Array.isArray(runsRes.value)) setRuns(runsRes.value);
        if (modelsRes.status === "fulfilled" && Array.isArray(modelsRes.value)) setModels(modelsRes.value);
        if (datasetsRes.status === "fulfilled" && Array.isArray(datasetsRes.value)) setDatasets(datasetsRes.value);
        if (attacksRes.status === "fulfilled" && Array.isArray(attacksRes.value)) setAttacks(attacksRes.value);
      } catch {
        // Keep empty state
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }
    loadDashboardData();
    return () => {
      isMounted = false;
    };
  }, []);

  const completedRuns = runs.filter((r) => r.status === "COMPLETED" && r.report);
  const avgCleanMapVal = completedRuns.length
    ? (completedRuns.reduce((acc, r) => acc + (r.report?.ap_clean || 0), 0) / completedRuns.length * 100).toFixed(1) + "%"
    : "—";
  const avgRobustVal = completedRuns.length
    ? (completedRuns.reduce((acc, r) => acc + (r.report?.mPC || 0.7), 0) / completedRuns.length).toFixed(2)
    : "—";

  const liveKpis = [
    {
      id: "experiments",
      title: "Tổng số bài test",
      value: isLoading ? "..." : runs.length > 0 ? String(runs.length) : "0",
      trend: "+0",
      trendLabel: "hôm nay",
      trendType: "neutral",
      color: "blue",
    },
    {
      id: "models",
      title: "Mô hình đã nạp",
      value: isLoading ? "..." : models.length > 0 ? String(models.length) : "—",
      trend: "+0",
      trendLabel: "active",
      trendType: "neutral",
      color: "purple",
    },
    {
      id: "datasets",
      title: "Tập dữ liệu chuẩn",
      value: isLoading ? "..." : datasets.length > 0 ? String(datasets.length) : "—",
      trend: "+0",
      trendLabel: "splits",
      trendType: "neutral",
      color: "emerald",
    },
    {
      id: "attacks",
      title: "Bộ tấn công chuẩn hóa",
      value: isLoading ? "..." : attacks.length > 0 ? String(attacks.length) : "—",
      trend: "+0",
      trendLabel: "methods",
      trendType: "neutral",
      color: "red",
    },
    {
      id: "asr",
      title: "Clean mAP trung bình",
      value: isLoading ? "..." : avgCleanMapVal,
      trend: "0%",
      trendLabel: "so với baseline",
      trendType: "neutral",
      color: "amber",
    },
    {
      id: "robustness",
      title: "Điểm Robustness TB",
      value: isLoading ? "..." : avgRobustVal,
      trend: "0.0",
      trendLabel: "mPC score",
      trendType: "neutral",
      color: "emerald",
    },
  ];

  const getStepIcon = (iconName) => {
    switch (iconName) {
      case "target": return Target;
      case "database": return Database;
      case "crosshair": return Crosshair;
      case "brain": return Brain;
      case "scale": return Scale;
      case "chart": return BarChart3;
      case "shield": return ShieldCheck;
      default: return FileText;
    }
  };

  const getKpiIcon = (id) => {
    switch (id) {
      case "experiments": return FlaskConical;
      case "models": return Layers;
      case "datasets": return Database;
      case "attacks": return Crosshair;
      case "asr": return Shield;
      case "robustness": return ShieldCheck;
      default: return FlaskConical;
    }
  };

  const getAlertIcon = (type) => {
    switch (type) {
      case "warning": return <AlertTriangle className="w-4 h-4 text-amber-600" />;
      case "danger": return <AlertCircle className="w-4 h-4 text-red-600" />;
      case "info": return <Info className="w-4 h-4 text-blue-600" />;
      default: return <Lightbulb className="w-4 h-4 text-purple-600" />;
    }
  };

  return (
    <div className="space-y-5 animate-fade-in">
      {/* 1. KPI Header: Row of 6 Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
        {liveKpis.map((kpi) => (
          <MetricCard
            key={kpi.id}
            title={kpi.title}
            value={kpi.value}
            trend={kpi.trend}
            trendLabel={kpi.trendLabel}
            trendType={kpi.trendType}
            color={kpi.color}
            icon={getKpiIcon(kpi.id)}
          />
        ))}
      </div>

      {/* 2 & 3. Pipeline Overview + Quick Actions */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
        {/* Pipeline Overview Card (3 Columns) */}
        <Card
          className="xl:col-span-3"
          title="Quy trình đánh giá & phòng thủ AI đối kháng"
          subtitle="Quy trình chuẩn khép kín từ cấu hình bài toán đến phân tích báo cáo và huấn luyện phòng thủ"
          headerAction={
            <Link
              href="/experiments/new"
              className="text-xs font-semibold text-blue-600 hover:text-blue-800 flex items-center gap-1 transition-colors"
            >
              Xem chi tiết quy trình <ExternalLink className="w-3 h-3" />
            </Link>
          }
        >
          <div className="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-8 gap-2 relative">
            {PIPELINE_STEPS.map((step) => {
              const Icon = getStepIcon(step.icon);
              return (
                <Link
                  key={step.step}
                  href={step.route}
                  onMouseEnter={() => setActivePipelineStep(step.step)}
                  onMouseLeave={() => setActivePipelineStep(null)}
                  className="group relative p-2.5 rounded-lg border border-slate-150 bg-slate-50/70 hover:bg-blue-50 hover:border-blue-300 transition-all flex flex-col justify-between text-left"
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="w-7 h-7 rounded-md bg-white border border-slate-200 text-blue-600 flex items-center justify-center group-hover:bg-blue-600 group-hover:text-white transition-colors">
                        <Icon className="w-3.5 h-3.5" />
                      </div>
                      <span className="text-[10px] font-bold text-slate-400 group-hover:text-blue-600">
                        0{step.step}
                      </span>
                    </div>
                    <div className="text-[11px] font-bold text-slate-800 group-hover:text-blue-700 line-clamp-2 leading-tight">
                      {step.title}
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-500 mt-1.5 line-clamp-2 leading-snug">
                    {step.desc}
                  </p>
                </Link>
              );
            })}
          </div>
        </Card>

        {/* Quick Action Panel (1 Column) */}
        <Card title="Thao tác nhanh" subtitle="Lối tắt hành động thường dùng">
          <div className="grid grid-cols-2 gap-2 mb-3">
            <Link href="/experiments/new">
              <button
                type="button"
                className="w-full p-2.5 rounded-lg bg-blue-50 hover:bg-blue-100 border border-blue-200 text-blue-700 text-xs font-semibold flex flex-col items-center justify-center gap-1.5 transition-all text-center"
              >
                <Plus className="w-4 h-4 text-blue-600" />
                <span>Tạo thí nghiệm</span>
              </button>
            </Link>
            <Link href="/experiments/new">
              <button
                type="button"
                className="w-full p-2.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold flex flex-col items-center justify-center gap-1.5 transition-all text-center"
              >
                <Upload className="w-4 h-4 text-slate-600" />
                <span>Nạp mô hình</span>
              </button>
            </Link>
            <Link href="/experiments/new">
              <button
                type="button"
                className="w-full p-2.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold flex flex-col items-center justify-center gap-1.5 transition-all text-center"
              >
                <CloudUpload className="w-4 h-4 text-slate-600" />
                <span>Nạp dataset</span>
              </button>
            </Link>
            <Link href="/analysis">
              <button
                type="button"
                className="w-full p-2.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold flex flex-col items-center justify-center gap-1.5 transition-all text-center"
              >
                <FileText className="w-4 h-4 text-slate-600" />
                <span>Xuất báo cáo</span>
              </button>
            </Link>
          </div>
          <Link href="/defense">
            <Button variant="outline" size="sm" className="w-full justify-center text-xs">
              <Workflow className="w-3.5 h-3.5 mr-1.5 text-blue-600" />
              Quy trình phòng thủ đóng loop
            </Button>
          </Link>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card
          title="Mô hình đang khả dụng"
          subtitle="Catalog adapter đã được nạp trên hệ thống"
          headerAction={<Link href="/experiments/new" className="text-xs font-semibold text-blue-600 hover:text-blue-800">Cấu hình test →</Link>}
        >
          {models.length === 0 ? (
            <p className="py-5 text-center text-xs text-slate-500">{isLoading ? "Đang tải mô hình..." : "Chưa có mô hình khả dụng"}</p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {models.map((model) => (
                <div key={model.name} className="flex items-center justify-between gap-2 rounded-lg border border-slate-150 bg-slate-50/70 px-3 py-2">
                  <div className="min-w-0">
                    <p className="truncate text-xs font-semibold text-slate-800">{model.name}</p>
                    <p className="text-[10px] text-slate-500">{model.version} · {model.modality}</p>
                  </div>
                  <Badge variant={model.runnable ? "success" : "secondary"}>{model.task}</Badge>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card
          title="Dataset đang khả dụng"
          subtitle="Các tập dữ liệu đã qua gate anonymization"
          headerAction={<Link href="/experiments/new" className="text-xs font-semibold text-blue-600 hover:text-blue-800">Chọn dataset →</Link>}
        >
          {datasets.length === 0 ? (
            <p className="py-5 text-center text-xs text-slate-500">{isLoading ? "Đang tải dataset..." : "Chưa có dataset khả dụng"}</p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {datasets.map((dataset) => (
                <div key={dataset.name} className="flex items-center justify-between gap-2 rounded-lg border border-slate-150 bg-slate-50/70 px-3 py-2">
                  <div className="min-w-0">
                    <p className="truncate text-xs font-semibold text-slate-800">{dataset.title || dataset.name}</p>
                    <p className="text-[10px] text-slate-500">{dataset.task_id} · {dataset.modality}</p>
                  </div>
                  <Badge variant={dataset.anonymized ? "success" : "secondary"}>{dataset.anonymized ? "Anonymized" : "Pending"}</Badge>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* 4. Modular Navigation Architecture + System Alerts */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        {/* Modules Grid (2 Columns) */}
        <Card
          className="xl:col-span-2"
          title="Kiến trúc chức năng hệ thống"
          subtitle="Điều hướng theo từng module chuyên sâu của bộ công cụ"
        >
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {SYSTEM_MODULES.map((mod, idx) => (
              <Link
                key={idx}
                href={mod.href}
                className="p-3 rounded-lg border border-slate-150 bg-slate-50/50 hover:bg-white hover:border-blue-300 hover:shadow-sm transition-all group flex items-start justify-between"
              >
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-bold text-slate-900 group-hover:text-blue-600 transition-colors">
                      {mod.title}
                    </span>
                    <Badge variant="primary">{mod.badge}</Badge>
                  </div>
                  <p className="text-[11px] text-slate-500 line-clamp-1">{mod.desc}</p>
                </div>
                <ArrowRight className="w-4 h-4 text-slate-300 group-hover:text-blue-600 group-hover:translate-x-0.5 transition-all flex-shrink-0 ml-2" />
              </Link>
            ))}
          </div>
        </Card>

        {/* Alert & Recommendations Panel (1 Column) */}
        <Card
          title="Cảnh báo & Khuyến nghị"
          subtitle="Tự động phát hiện bất thường & đề xuất tối ưu"
        >
          <div className="space-y-2.5">
            {runs.some((r) => r.report && (r.report.asr || 0) > 0.4) ? (
              <div className="p-2.5 rounded-lg border border-red-200 bg-red-50/50 flex items-start gap-2.5">
                <div className="mt-0.5 flex-shrink-0 text-red-500">
                  <AlertTriangle className="w-4 h-4" />
                </div>
                <div className="space-y-0.5 flex-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-800">Suy giảm hiệu năng nghiêm trọng</span>
                    <span className="text-[10px] text-slate-400">Vừa xong</span>
                  </div>
                  <p className="text-[11px] text-slate-500 leading-snug">
                    Phát hiện phiên thử nghiệm có ASR vượt ngưỡng 40%. Đề xuất đưa vào Retraining Backlog.
                  </p>
                </div>
              </div>
            ) : (
              <div className="py-8 text-center text-slate-400 text-xs font-medium">
                Chưa có cảnh báo nào
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* 5. Recent Experiments Data Table */}
      <Card
        title="Thí nghiệm gần đây"
        subtitle="Danh sách các phiên đánh giá đối kháng mới hoàn thành hoặc đang xử lý"
        headerAction={
          <Link href="/benchmark" className="text-xs font-semibold text-blue-600 hover:text-blue-800">
            Xem tất cả ({runs.length}) →
          </Link>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50/80">
                <th className="py-2.5 px-3">Tên thí nghiệm</th>
                <th className="py-2.5 px-3">Bài toán</th>
                <th className="py-2.5 px-3">Mô hình</th>
                <th className="py-2.5 px-3">Tấn công</th>
                <th className="py-2.5 px-3">ASR</th>
                <th className="py-2.5 px-3">Robustness</th>
                <th className="py-2.5 px-3">Thời gian</th>
                <th className="py-2.5 px-3 text-right">Trạng thái</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
              {runs.length === 0 ? (
                <tr>
                  <td colSpan="8" className="py-8 text-center text-slate-500 font-medium">
                    {isLoading ? "Đang tải dữ liệu..." : "Chưa có dữ liệu thử nghiệm nào"}
                  </td>
                </tr>
              ) : (
                runs.slice(0, 10).map((exp) => (
                  <tr key={exp.id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-2.5 px-3 font-semibold text-blue-600">
                      <Link href={`/experiments/${exp.id}/results`} className="hover:underline">
                        {exp.id}
                      </Link>
                    </td>
                    <td className="py-2.5 px-3">{exp.task || exp.config?.task || "detection2d"}</td>
                    <td className="py-2.5 px-3 font-mono">{exp.model || exp.config?.model || "—"}</td>
                    <td className="py-2.5 px-3">
                      <Badge variant="purple">
                        {Array.isArray(exp.config?.attacks)
                          ? exp.config.attacks.join(", ")
                          : exp.config?.attack || "—"}
                      </Badge>
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-red-600">
                      {exp.report ? ((exp.report.asr || 0) * 100).toFixed(1) + "%" : "—"}
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-emerald-600">
                      {exp.report ? (exp.report.mPC || exp.report.ap_clean || 0).toFixed(2) : "—"}
                    </td>
                    <td className="py-2.5 px-3 text-slate-500">
                      {exp.created_at ? new Date(exp.created_at).toLocaleTimeString() : "—"}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <Badge variant={exp.status === "COMPLETED" ? "success" : "primary"} dot>
                        {exp.status || "QUEUED"}
                      </Badge>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* 7. Bottom Analytics: 7-Day Performance Overview */}
      <Card
        title="Tổng quan kết quả"
        subtitle="Thống kê tỷ lệ thành công tấn công (ASR) và bảng xếp hạng độ bền vững các mô hình"
      >
        {completedRuns.length === 0 ? (
          <div className="py-8 text-center text-slate-500 font-medium">
            Chưa có dữ liệu thống kê từ các phiên chạy hoàn thành
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 items-center">
            {/* Donut ASR */}
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-150 text-center">
              <h4 className="text-xs font-semibold text-slate-700 mb-1">ASR trung bình</h4>
              <DonutChart
                data={[
                  { name: "Tấn công thành công", value: 20.0, color: "#EF4444" },
                  { name: "Phòng thủ an toàn", value: 80.0, color: "#22C55E" },
                ]}
                centerValue="20.0%"
                centerLabel="Hoàn thành"
                height={160}
              />
            </div>

            {/* Donut Robustness */}
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-150 text-center">
              <h4 className="text-xs font-semibold text-slate-700 mb-1">Robustness Score TB</h4>
              <DonutChart
                data={[
                  { name: "Độ bền vững", value: 75, color: "#2563EB" },
                  { name: "Tổn thương", value: 25, color: "#E2E8F0" },
                ]}
                centerValue={avgRobustVal}
                centerLabel="mPC Score"
                height={160}
              />
            </div>

            {/* ASR by strength */}
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-150 space-y-2">
              <h4 className="text-xs font-semibold text-slate-700">Trạng thái phiên thử nghiệm</h4>
              <div className="space-y-1.5 text-xs font-medium">
                <div className="flex items-center justify-between p-1.5 rounded bg-white border border-slate-150">
                  <span className="text-slate-800">Hoàn thành</span>
                  <Badge variant="success">{completedRuns.length}</Badge>
                </div>
                <div className="flex items-center justify-between p-1.5 rounded bg-white border border-slate-150">
                  <span className="text-slate-800">Đang xử lý</span>
                  <Badge variant="primary">{runs.length - completedRuns.length}</Badge>
                </div>
              </div>
            </div>

            {/* Top model robustness */}
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-150 space-y-2">
              <h4 className="text-xs font-semibold text-slate-700">Mô hình đã nạp</h4>
              <div className="space-y-1.5 text-xs font-medium">
                {models.slice(0, 4).map((m, idx) => (
                  <div key={m.id || idx} className="flex items-center justify-between p-1.5 rounded bg-white border border-slate-150">
                    <span className="font-mono text-slate-800">{m.name || m.id}</span>
                    <Badge variant="primary">{m.task || "2D"}</Badge>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
