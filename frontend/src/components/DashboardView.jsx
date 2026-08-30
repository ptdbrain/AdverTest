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
import {
  createProject,
  getActiveProjectId,
  getCatalogAttacks,
  getCatalogDatasets,
  getCatalogModels,
  listProjects,
  listRuns,
  setActiveProjectId,
} from "@/lib/api";
import { useProjectContext } from "@/context/ProjectContext";

const PIPELINE_STEPS = [
  { step: 1, title: "Cấu hình bài toán", desc: "Nạp mô hình & dữ liệu gốc", icon: "database", route: "/experiments/new" },
  { step: 2, title: "Cấu hình tấn công", desc: "Tạo hoặc mở một phiên thuộc project để chọn attack", icon: "crosshair", route: "/experiments/new" },
  { step: 3, title: "Đánh giá Robustness", desc: "Phân tích suy giảm hiệu năng", icon: "chart", route: "/benchmark" },
  { step: 4, title: "Tôi luyện phòng thủ", desc: "Fine-tune & vá lỗi nhận diện", icon: "shield", route: "/defense" },
  { step: 5, title: "Báo cáo & Chứng nhận", desc: "Xuất báo cáo khoa học & Audit", icon: "file", route: "/analysis" },
];

const SYSTEM_MODULES = [
  { title: "Cấu hình bài toán & Mô hình", desc: "Khởi tạo thí nghiệm, quản lý dataset và checkpoint", href: "/experiments/new", badge: "Core" },
  { title: "Kịch bản tấn công đối kháng", desc: "Tạo hoặc mở một phiên thuộc project để tùy biến attack", href: "/experiments/new", badge: "Attacks" },
  { title: "Bảng so sánh & Metrics", desc: "Biểu đồ so sánh đa chiều và ma trận phân rã", href: "/benchmark", badge: "Analytics" },
  { title: "Đóng vòng phòng thủ & HITL", desc: "Hàng đợi đánh giá rủi ro và tôi luyện mô hình", href: "/defense", badge: "Defense" },
];

export default function DashboardView() {
  const { scopedHref } = useProjectContext();
  const [activePipelineStep, setActivePipelineStep] = useState(null);
  const [runs, setRuns] = useState([]);
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [attacks, setAttacks] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [projects, setProjects] = useState([]);
  const [activeProjectId, setActiveProject] = useState(null);
  const [newProjectName, setNewProjectName] = useState("");
  const [projectError, setProjectError] = useState("");

  useEffect(() => {
    setActiveProject(getActiveProjectId());
  }, []);

  useEffect(() => {
    let isMounted = true;
    async function loadDashboardData() {
      try {
        const [projectsRes, runsRes, modelsRes, datasetsRes, attacksRes] = await Promise.allSettled([
          listProjects(),
          listRuns(),
          getCatalogModels(),
          getCatalogDatasets(),
          getCatalogAttacks(),
        ]);
        if (!isMounted) return;
        if (projectsRes.status === "fulfilled" && Array.isArray(projectsRes.value)) setProjects(projectsRes.value);
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
  }, [activeProjectId]);

  const selectProject = (projectId) => {
    setActiveProjectId(projectId || null);
    setActiveProject(projectId || null);
    setProjectError("");
  };

  const handleCreateProject = async () => {
    const name = newProjectName.trim();
    if (!name) {
      setProjectError("Nhập tên project trước khi tạo.");
      return;
    }
    try {
      const project = await createProject({ name });
      setProjects((current) => [...current, project]);
      setNewProjectName("");
      selectProject(project.id);
    } catch (error) {
      setProjectError(error?.message || "Không thể tạo project.");
    }
  };

  const completedRuns = runs.filter((r) => r.status === "COMPLETED" && r.report?.evidence?.status === "VERIFIED");
  const runDegradation = (report) => {
    if (report?.evidence?.status !== "VERIFIED") return null;
    const values = (report?.cells || [])
      .map((cell) => Number(cell.degradation_ratio ?? cell.degradation))
      .filter((value) => Number.isFinite(value) && value >= 0 && value <= 1);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  const degradations = completedRuns
    .map((run) => runDegradation(run.report))
    .filter((value) => value !== null);
  const avgAttackSuccessRatio = degradations.length
    ? degradations.reduce((sum, value) => sum + value, 0) / degradations.length
    : null;
  const cleanMapValues = completedRuns
    .map((run) => Number(run.report?.ap_clean))
    .filter((value) => Number.isFinite(value));
  const avgCleanMapVal = cleanMapValues.length
    ? (cleanMapValues.reduce((sum, value) => sum + value, 0) / cleanMapValues.length * 100).toFixed(1) + "%"
    : "—";
  const avgRobustVal = avgAttackSuccessRatio !== null
    ? (1 - avgAttackSuccessRatio).toFixed(2)
    : "—";

  const liveKpis = [
    {
      id: "experiments",
      title: "Tổng số bài test",
      value: isLoading ? "..." : runs.length > 0 ? String(runs.length) : "0",
      trend: undefined,
      trendLabel: undefined,
      trendType: "neutral",
      color: "blue",
    },
    {
      id: "models",
      title: "Mô hình đã nạp",
      value: isLoading ? "..." : models.length > 0 ? String(models.length) : "—",
      trend: undefined,
      trendLabel: undefined,
      trendType: "neutral",
      color: "purple",
    },
    {
      id: "datasets",
      title: "Tập dữ liệu chuẩn",
      value: isLoading ? "..." : datasets.length > 0 ? String(datasets.length) : "—",
      trend: undefined,
      trendLabel: undefined,
      trendType: "neutral",
      color: "emerald",
    },
    {
      id: "attacks",
      title: "Bộ tấn công chuẩn hóa",
      value: isLoading ? "..." : attacks.length > 0 ? String(attacks.length) : "—",
      trend: undefined,
      trendLabel: undefined,
      trendType: "neutral",
      color: "red",
    },
    {
      id: "asr",
      title: "Clean mAP trung bình",
      value: isLoading ? "..." : avgCleanMapVal,
      trend: undefined,
      trendLabel: undefined,
      trendType: "neutral",
      color: "amber",
    },
    {
      id: "robustness",
      title: "Robustness TB (1 - suy giảm)",
      value: isLoading ? "..." : avgRobustVal,
      trend: undefined,
      trendLabel: undefined,
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

      <Card
        title="Ngữ cảnh project"
        subtitle="Run, session, artifact và báo cáo chỉ hiển thị trong project mà bạn là thành viên."
      >
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <label className="flex min-w-0 flex-1 flex-col gap-1 text-xs font-medium text-slate-700">
            Project đang làm việc
            <select
              value={activeProjectId || ""}
              onChange={(event) => selectProject(event.target.value)}
              className="min-h-11 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-900"
              aria-label="Chọn project đang làm việc"
            >
              <option value="">Chọn project</option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>{project.name}</option>
              ))}
            </select>
          </label>
          <label className="flex min-w-0 flex-1 flex-col gap-1 text-xs font-medium text-slate-700">
            Tạo project mới
            <input
              value={newProjectName}
              onChange={(event) => setNewProjectName(event.target.value)}
              placeholder="Ví dụ: KITTI safety audit"
              className="min-h-11 rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-900"
              aria-label="Tên project mới"
            />
          </label>
          <Button onClick={handleCreateProject} className="min-h-11" aria-label="Tạo project mới">
            <Plus className="h-4 w-4" /> Tạo project
          </Button>
        </div>
        {!activeProjectId && !isLoading && (
          <p className="mt-3 text-sm text-amber-700">
            Chọn hoặc tạo project để xem dữ liệu benchmark, session và artifact thật.
          </p>
        )}
        {projectError && <p className="mt-3 text-sm text-red-700" role="alert">{projectError}</p>}
      </Card>

      {/* 2 & 3. Pipeline Overview + Quick Actions */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
        {/* Pipeline Overview Card (3 Columns) */}
        <Card
          className="xl:col-span-3"
          title="Quy trình đánh giá & phòng thủ AI đối kháng"
          subtitle="Quy trình chuẩn khép kín từ cấu hình bài toán đến phân tích báo cáo và huấn luyện phòng thủ"
          headerAction={
            <Link
              href={scopedHref("/experiments/new")}
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
                  href={scopedHref(step.route)}
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
            <Link href={scopedHref("/experiments/new")}>
              <button
                type="button"
                className="w-full p-2.5 rounded-lg bg-blue-50 hover:bg-blue-100 border border-blue-200 text-blue-700 text-xs font-semibold flex flex-col items-center justify-center gap-1.5 transition-all text-center"
              >
                <Plus className="w-4 h-4 text-blue-600" />
                <span>Tạo thí nghiệm</span>
              </button>
            </Link>
            <Link href={scopedHref("/experiments/new")}>
              <button
                type="button"
                className="w-full p-2.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold flex flex-col items-center justify-center gap-1.5 transition-all text-center"
              >
                <Upload className="w-4 h-4 text-slate-600" />
                <span>Nạp mô hình</span>
              </button>
            </Link>
            <Link href={scopedHref("/experiments/new")}>
              <button
                type="button"
                className="w-full p-2.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold flex flex-col items-center justify-center gap-1.5 transition-all text-center"
              >
                <CloudUpload className="w-4 h-4 text-slate-600" />
                <span>Nạp dataset</span>
              </button>
            </Link>
            <Link href={scopedHref("/analysis")}>
              <button
                type="button"
                className="w-full p-2.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold flex flex-col items-center justify-center gap-1.5 transition-all text-center"
              >
                <FileText className="w-4 h-4 text-slate-600" />
                <span>Xuất báo cáo</span>
              </button>
            </Link>
          </div>
          <Link href={scopedHref("/defense")}>
            <Button variant="outline" size="sm" className="w-full justify-center text-xs">
              <Workflow className="w-3.5 h-3.5 mr-1.5 text-blue-600" />
              Quy trình phòng thủ đóng loop
            </Button>
          </Link>
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
                href={scopedHref(mod.href)}
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
            {runs.some((r) => runDegradation(r.report) !== null && runDegradation(r.report) > 0.4) ? (
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
          <Link href={scopedHref("/benchmark")} className="text-xs font-semibold text-blue-600 hover:text-blue-800">
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
                      <Link href={scopedHref(`/experiments/${exp.id}/results`, exp.run_id || exp.id)} className="hover:underline">
                        {exp.run_id || exp.id}
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
                      {runDegradation(exp.report) !== null ? (runDegradation(exp.report) * 100).toFixed(1) + "%" : "—"}
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-emerald-600">
                      {runDegradation(exp.report) !== null ? (1 - runDegradation(exp.report)).toFixed(2) : "—"}
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
        {avgAttackSuccessRatio === null ? (
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
                  { name: "Suy giảm đo được", value: avgAttackSuccessRatio * 100, color: "#EF4444" },
                  { name: "Không suy giảm", value: (1 - avgAttackSuccessRatio) * 100, color: "#22C55E" },
                ]}
                centerValue={(avgAttackSuccessRatio * 100).toFixed(1) + "%"}
                centerLabel="Suy giảm TB"
                height={160}
              />
            </div>

            {/* Donut Robustness */}
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-150 text-center">
              <h4 className="text-xs font-semibold text-slate-700 mb-1">Robustness Score TB</h4>
              <DonutChart
                data={[
                  { name: "Robustness", value: (1 - avgAttackSuccessRatio) * 100, color: "#2563EB" },
                  { name: "Suy giảm", value: avgAttackSuccessRatio * 100, color: "#E2E8F0" },
                ]}
                centerValue={avgRobustVal}
                centerLabel="1 - suy giảm"
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
