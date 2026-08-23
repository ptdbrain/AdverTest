"use client";

import React, { useState } from "react";
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
  DASHBOARD_KPIS,
  PIPELINE_STEPS,
  SYSTEM_MODULES,
  RECENT_EXPERIMENTS,
  SYSTEM_ALERTS,
} from "@/data/mockData";

export default function DashboardPage() {
  const [activePipelineStep, setActivePipelineStep] = useState(null);

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
        {DASHBOARD_KPIS.map((kpi) => (
          <MetricCard
            key={kpi.id}
            title={kpi.title}
            value={kpi.value}
            trend={kpi.trend}
            trendLabel={kpi.trendLabel}
            trendType={kpi.trendType}
            color={kpi.color}
            icon={getKpiIcon(kpi.id)}
            sparkline={kpi.sparkline}
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
            {PIPELINE_STEPS.map((step, idx) => {
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
                <span>Tải dữ liệu</span>
              </button>
            </Link>
            <Link href="/experiments/new">
              <button
                type="button"
                className="w-full p-2.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold flex flex-col items-center justify-center gap-1.5 transition-all text-center"
              >
                <CloudUpload className="w-4 h-4 text-slate-600" />
                <span>Tải mô hình</span>
              </button>
            </Link>
            <Link href="/analysis">
              <button
                type="button"
                className="w-full p-2.5 rounded-lg bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 text-xs font-semibold flex flex-col items-center justify-center gap-1.5 transition-all text-center"
              >
                <FileText className="w-4 h-4 text-slate-600" />
                <span>Xem báo cáo</span>
              </button>
            </Link>
          </div>
          <Link href="/experiments/new" className="block">
            <Button variant="primary" className="w-full text-xs font-semibold" icon={Workflow}>
              Mở trình dựng pipeline
            </Button>
          </Link>
        </Card>
      </div>

      {/* 4 & 6. System Modules & Alert Panel */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        {/* System Modules (2 Columns) */}
        <Card
          className="xl:col-span-2"
          title="Mô-đun hệ thống"
          subtitle="Truy cập trực tiếp vào các trung tâm điều khiển chuyên biệt"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {SYSTEM_MODULES.map((mod) => (
              <Link
                key={mod.title}
                href={mod.route}
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
            {SYSTEM_ALERTS.map((alert, idx) => (
              <div
                key={idx}
                className="p-2.5 rounded-lg border border-slate-150 bg-slate-50/50 flex items-start gap-2.5"
              >
                <div className="mt-0.5 flex-shrink-0">{getAlertIcon(alert.type)}</div>
                <div className="space-y-0.5 flex-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-800">{alert.title}</span>
                    <span className="text-[10px] text-slate-400">{alert.time}</span>
                  </div>
                  <p className="text-[11px] text-slate-500 leading-snug">{alert.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* 5. Recent Experiments Data Table */}
      <Card
        title="Thí nghiệm gần đây"
        subtitle="Danh sách các phiên đánh giá đối kháng mới hoàn thành hoặc đang xử lý"
        headerAction={
          <Link href="/benchmark" className="text-xs font-semibold text-blue-600 hover:text-blue-800">
            Xem tất cả (128) →
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
              {RECENT_EXPERIMENTS.map((exp) => (
                <tr key={exp.id} className="hover:bg-slate-50/80 transition-colors">
                  <td className="py-2.5 px-3 font-semibold text-blue-600">
                    <Link href={`/experiments/${exp.id}/results`} className="hover:underline">
                      {exp.id}
                    </Link>
                  </td>
                  <td className="py-2.5 px-3">{exp.task}</td>
                  <td className="py-2.5 px-3 font-mono">{exp.model}</td>
                  <td className="py-2.5 px-3">
                    <Badge variant="purple">{exp.attack}</Badge>
                  </td>
                  <td className="py-2.5 px-3 font-semibold text-red-600">{exp.asr}</td>
                  <td className="py-2.5 px-3 font-semibold text-emerald-600">{exp.robustness}</td>
                  <td className="py-2.5 px-3 text-slate-500">{exp.time}</td>
                  <td className="py-2.5 px-3 text-right">
                    <Badge variant={exp.status === "Hoàn thành" ? "success" : "primary"} dot>
                      {exp.status}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* 7. Bottom Analytics: 7-Day Performance Overview */}
      <Card
        title="Tổng quan kết quả (7 ngày qua)"
        subtitle="Thống kê tỷ lệ thành công tấn công (ASR) và bảng xếp hạng độ bền vững các mô hình"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 items-center">
          {/* Donut ASR */}
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-150 text-center">
            <h4 className="text-xs font-semibold text-slate-700 mb-1">ASR trung bình</h4>
            <DonutChart
              data={[
                { name: "Tấn công thành công", value: 23.6, color: "#EF4444" },
                { name: "Phòng thủ an toàn", value: 76.4, color: "#22C55E" },
              ]}
              centerValue="23.6%"
              centerLabel="↑ 18% tuần này"
              height={160}
            />
          </div>

          {/* Donut Robustness */}
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-150 text-center">
            <h4 className="text-xs font-semibold text-slate-700 mb-1">Robustness Score TB</h4>
            <DonutChart
              data={[
                { name: "Độ bền vững", value: 71, color: "#2563EB" },
                { name: "Tổn thương", value: 29, color: "#E2E8F0" },
              ]}
              centerValue="0.71"
              centerLabel="↑ 0.06 cải thiện"
              height={160}
            />
          </div>

          {/* ASR by strength */}
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-150 space-y-2">
            <h4 className="text-xs font-semibold text-slate-700">ASR theo cường độ tấn công</h4>
            <div className="space-y-2 text-xs">
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span>Thấp (ε = 2/255)</span>
                  <span className="font-semibold text-emerald-600">12.5%</span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-200 overflow-hidden">
                  <div className="h-full bg-emerald-500" style={{ width: "12.5%" }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span>Trung bình (ε = 8/255)</span>
                  <span className="font-semibold text-amber-600">22.3%</span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-200 overflow-hidden">
                  <div className="h-full bg-amber-500" style={{ width: "22.3%" }} />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span>Cao (ε = 16/255)</span>
                  <span className="font-semibold text-red-600">36.8%</span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-200 overflow-hidden">
                  <div className="h-full bg-red-500" style={{ width: "36.8%" }} />
                </div>
              </div>
            </div>
          </div>

          {/* Top model robustness */}
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-150 space-y-2">
            <h4 className="text-xs font-semibold text-slate-700">Xếp hạng độ bền mô hình</h4>
            <div className="space-y-1.5 text-xs font-medium">
              <div className="flex items-center justify-between p-1.5 rounded bg-white border border-slate-150">
                <span className="font-mono text-slate-800">1. ViT-B/16</span>
                <Badge variant="success">0.78</Badge>
              </div>
              <div className="flex items-center justify-between p-1.5 rounded bg-white border border-slate-150">
                <span className="font-mono text-slate-800">2. ResNet50</span>
                <Badge variant="success">0.73</Badge>
              </div>
              <div className="flex items-center justify-between p-1.5 rounded bg-white border border-slate-150">
                <span className="font-mono text-slate-800">3. YOLOv8s</span>
                <Badge variant="primary">0.66</Badge>
              </div>
              <div className="flex items-center justify-between p-1.5 rounded bg-white border border-slate-150">
                <span className="font-mono text-slate-800">4. YOLOv8n</span>
                <Badge variant="warning">0.61</Badge>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
