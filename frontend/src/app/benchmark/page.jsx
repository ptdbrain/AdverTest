"use client";

import React, { useState } from "react";
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
} from "lucide-react";
import PageHeader from "@/components/layout/PageHeader";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import ComparisonBarChart from "@/components/metrics/ComparisonBarChart";
import TrendLineChart from "@/components/metrics/TrendLineChart";
import RobustnessRadar from "@/components/metrics/RobustnessRadar";
import MetricSparkline from "@/components/metrics/MetricSparkline";
import {
  BENCHMARK_KPIS,
  MODEL_RANKING,
  ATTACK_RANKING_DATA,
} from "@/data/mockData";
import { cn } from "@/lib/utils";

export default function BenchmarkPage() {
  const [selectedTask, setSelectedTask] = useState("Object Detection");
  const [selectedModel, setSelectedModel] = useState("all");
  const [selectedAttackType, setSelectedAttackType] = useState("all");
  const [dateRange, setDateRange] = useState("05/05/2025 - 12/05/2025");

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Metrics & Benchmark"
        subtitle="So sánh định lượng toàn diện hiệu năng trước / sau tấn công và đánh giá độ bền vững các mô hình"
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Metrics & Benchmark" },
        ]}
        actions={
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm" icon={Filter}>
              Bộ lọc nâng cao
            </Button>
            <Link href="/analysis">
              <Button variant="primary" size="sm" icon={Download}>
                Xuất báo cáo
              </Button>
            </Link>
          </div>
        }
      />

      {/* FILTER TOOLBAR */}
      <Card className="p-3">
        <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-5 gap-3 text-xs">
          <div>
            <label className="text-slate-500 font-medium block mb-1">Bài toán</label>
            <select
              value={selectedTask}
              onChange={(e) => setSelectedTask(e.target.value)}
              className="w-full p-1.5 rounded border border-slate-300 bg-white font-medium"
            >
              <option value="Object Detection">Object Detection</option>
              <option value="SAM">SAM / Segmentation</option>
              <option value="3D Detection">3D Detection</option>
              <option value="Classification">Classification</option>
            </select>
          </div>

          <div>
            <label className="text-slate-500 font-medium block mb-1">Mô hình</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full p-1.5 rounded border border-slate-300 bg-white font-medium"
            >
              <option value="all">Tất cả mô hình (6)</option>
              <option value="yolov8n">YOLOv8n</option>
              <option value="yolov8s">YOLOv8s</option>
              <option value="resnet50">ResNet50</option>
              <option value="vit">ViT-B/16</option>
            </select>
          </div>

          <div>
            <label className="text-slate-500 font-medium block mb-1">Loại tấn công</label>
            <select
              value={selectedAttackType}
              onChange={(e) => setSelectedAttackType(e.target.value)}
              className="w-full p-1.5 rounded border border-slate-300 bg-white font-medium"
            >
              <option value="all">Tất cả loại (48)</option>
              <option value="gradient">Gradient-based (White-box)</option>
              <option value="query">Query-based (Black-box)</option>
              <option value="patch">Physical Patch</option>
            </select>
          </div>

          <div>
            <label className="text-slate-500 font-medium block mb-1">Mức độ tấn công</label>
            <select className="w-full p-1.5 rounded border border-slate-300 bg-white font-medium">
              <option>Tất cả mức độ (ε=2/255 → 16/255)</option>
              <option>Nhẹ (ε=2/255)</option>
              <option>Trung bình (ε=8/255)</option>
              <option>Nặng (ε=16/255)</option>
            </select>
          </div>

          <div>
            <label className="text-slate-500 font-medium block mb-1">Khoảng thời gian</label>
            <div className="p-1.5 rounded border border-slate-300 bg-slate-50 text-slate-700 flex items-center gap-1.5 font-medium">
              <Calendar className="w-3.5 h-3.5 text-slate-400" />
              <span>{dateRange}</span>
            </div>
          </div>
        </div>
      </Card>

      {/* 9 KPI METRIC CARDS */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-9 gap-2.5">
        {BENCHMARK_KPIS.map((kpi) => (
          <div
            key={kpi.id}
            className="p-2.5 rounded-lg bg-white border border-slate-200 shadow-xs flex flex-col justify-between"
          >
            <div>
              <div className="text-[11px] text-slate-500 font-medium truncate">{kpi.name}</div>
              <div className="text-xs font-bold text-slate-900 mt-0.5">
                <span className="text-slate-500 text-[10px] mr-1">{kpi.before} →</span>
                <span className={kpi.isBadIncrease ? "text-red-600" : "text-blue-600"}>
                  {kpi.after}
                </span>
              </div>
            </div>
            <div className="flex items-end justify-between mt-2 pt-1 border-t border-slate-100">
              <span
                className={cn(
                  "text-[10px] font-bold",
                  kpi.isBadIncrease ? "text-red-600" : "text-slate-600"
                )}
              >
                {kpi.drop}
              </span>
              <MetricSparkline
                data={kpi.sparkline}
                color={kpi.isBadIncrease ? "red" : "blue"}
                height={16}
                width={36}
              />
            </div>
          </div>
        ))}
      </div>

      {/* CHARTS ROW: Bar Comparison, Trend Line, Radar & Callouts */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
        {/* Before vs After Bar Chart (4 Columns) */}
        <Card
          className="xl:col-span-4"
          title="So sánh Trước vs Sau"
          subtitle="Biến động các chỉ số thị giác chính"
        >
          <ComparisonBarChart height={240} />
        </Card>

        {/* Multi-run Trend (4 Columns) */}
        <Card
          className="xl:col-span-4"
          title="Xu hướng theo chuỗi thí nghiệm"
          subtitle="Hiệu năng mAP qua các lần thử nghiệm"
        >
          <TrendLineChart height={240} />
        </Card>

        {/* Robustness Radar & Callouts (4 Columns) */}
        <Card
          className="xl:col-span-4"
          title="Độ bền vững đa chiều (Radar)"
          subtitle="Đánh giá toàn diện 5 trục thuộc tính"
        >
          <RobustnessRadar height={240} showDefended={true} />
        </Card>
      </div>

      {/* SUMMARY CALLOUTS */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="p-3 rounded-lg bg-red-50 border border-red-200 flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-red-600 text-white flex items-center justify-center font-bold flex-shrink-0">
            <Shield className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] text-red-700 uppercase font-bold">Tấn công mạnh nhất</div>
            <div className="text-sm font-bold text-red-900">PGD (ε=8/255)</div>
            <div className="text-xs text-red-700 mt-0.5">ASR trung bình đạt tới 82.3%</div>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-600 text-white flex items-center justify-center font-bold flex-shrink-0">
            <Award className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] text-emerald-700 uppercase font-bold">Mô hình bền vững nhất</div>
            <div className="text-sm font-bold text-emerald-900">YOLOv8n + AdvTrain</div>
            <div className="text-xs text-emerald-700 mt-0.5">Robustness Score đạt 0.71</div>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-amber-600 text-white flex items-center justify-center font-bold flex-shrink-0">
            <TrendingDown className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] text-amber-700 uppercase font-bold">Suy giảm lớn nhất</div>
            <div className="text-sm font-bold text-amber-900">mIoU Phân đoạn</div>
            <div className="text-xs text-amber-700 mt-0.5">Giảm sụt tới 29.6% so với gốc</div>
          </div>
        </div>
      </div>

      {/* BENCHMARK TABLES: Model Ranking & Attack Benchmark */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        {/* Overall Model Ranking Table */}
        <Card
          title="Bảng xếp hạng mô hình bền vững"
          subtitle="Sắp xếp theo Robustness Score giảm dần"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50/70">
                  <th className="py-2 px-2.5">#</th>
                  <th className="py-2 px-2.5">Mô hình</th>
                  <th className="py-2 px-2.5">Robustness</th>
                  <th className="py-2 px-2.5">mAP Sau</th>
                  <th className="py-2 px-2.5">ASR</th>
                  <th className="py-2 px-2.5">FPS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                {MODEL_RANKING.map((m) => (
                  <tr key={m.rank} className="hover:bg-slate-50">
                    <td className="py-2 px-2.5 font-bold text-slate-400">{m.rank}</td>
                    <td className="py-2 px-2.5 font-bold font-mono text-slate-900">{m.model}</td>
                    <td className="py-2 px-2.5 font-bold text-emerald-600">{m.robustness}</td>
                    <td className="py-2 px-2.5">{m.mapAfter}</td>
                    <td className="py-2 px-2.5 text-red-600 font-semibold">{m.asr}</td>
                    <td className="py-2 px-2.5 font-mono">{m.fps}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Attack Benchmark Table */}
        <Card
          title="Bảng đánh giá mức độ nguy hiểm thuật toán tấn công"
          subtitle="Tỷ lệ thành công và tác động suy giảm mAP"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50/70">
                  <th className="py-2 px-2.5">#</th>
                  <th className="py-2 px-2.5">Thuật toán</th>
                  <th className="py-2 px-2.5">ASR</th>
                  <th className="py-2 px-2.5">mAP giảm</th>
                  <th className="py-2 px-2.5">Mức ảnh hưởng</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                {ATTACK_RANKING_DATA.slice(0, 5).map((a) => (
                  <tr key={a.rank} className="hover:bg-slate-50">
                    <td className="py-2 px-2.5 font-bold text-slate-400">{a.rank}</td>
                    <td className="py-2 px-2.5 font-bold text-slate-900">{a.attack}</td>
                    <td className="py-2 px-2.5 font-bold text-red-600">{a.asr}</td>
                    <td className="py-2 px-2.5 text-slate-600">{a.mapDrop}</td>
                    <td className="py-2 px-2.5">
                      <Badge
                        variant={
                          a.severity === "Rất cao"
                            ? "danger"
                            : a.severity === "Cao"
                            ? "warning"
                            : "primary"
                        }
                      >
                        {a.severity}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
