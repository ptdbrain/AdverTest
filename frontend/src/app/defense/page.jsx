"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  ShieldCheck,
  Play,
  Pause,
  RotateCcw,
  Cpu,
  HardDrive,
  Activity,
  Layers,
  Settings,
  Sparkles,
  ArrowRight,
  TrendingUp,
  FileCode,
  CheckCircle2,
  AlertCircle,
  Database,
  Sliders,
} from "lucide-react";
import PageHeader from "@/components/layout/PageHeader";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import RobustnessRadar from "@/components/metrics/RobustnessRadar";
import { DEFENSE_COMPARISON } from "@/data/mockData";
import { cn } from "@/lib/utils";

const SUBTABS = [
  "Tổng quan",
  "Adversarial Training",
  "Fine-tuning",
  "Chiến lược phòng thủ",
  "So sánh mô hình",
  "Lịch sử huấn luyện",
  "Logs",
];

export default function DefensePage() {
  const [activeSubtab, setActiveSubtab] = useState("Tổng quan");
  const [isTraining, setIsTraining] = useState(true);
  const [progress, setProgress] = useState(68);
  const [epoch, setEpoch] = useState(68);

  const STRATEGIES = [
    { name: "Adversarial Training", status: "active", desc: "Huấn luyện cùng mẫu nhiễu PGD / TRADES", badge: "Đang sử dụng" },
    { name: "Randomized Smoothing", status: "ready", desc: "Cung cấp độ tin cậy chứng minh được bằng Gaussian noise", badge: "Khả dụng" },
    { name: "Input Preprocessing", status: "ready", desc: "Tự động làm sạch nhiễu đầu vào trước khi inference", badge: "Khả dụng" },
    { name: "JPEG Compression", status: "ready", desc: "Khử nhiễu tần số cao của gradient attack", badge: "Khả dụng" },
    { name: "Feature Denoising", status: "ready", desc: "Triệt tiêu activation bất thường tại các tầng ẩn", badge: "Khả dụng" },
  ];

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Phòng thủ & Huấn luyện lại"
        subtitle="Củng cố độ bền vững của mô hình thông qua Adversarial Training, Fine-tuning và cơ chế tiền xử lý phòng vệ."
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Phòng thủ & Huấn luyện lại" },
        ]}
      />

      {/* SUBTABS NAVIGATION */}
      <div className="flex items-center gap-1 overflow-x-auto border-b border-slate-200 pb-1">
        {SUBTABS.map((tab) => (
          <button
            key={tab}
            type="button"
            onClick={() => setActiveSubtab(tab)}
            className={cn(
              "px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-colors whitespace-nowrap",
              activeSubtab === tab
                ? "bg-blue-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
            )}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* 1. DEFENSE PIPELINE ARCHITECTURE FLOW */}
      <Card
        title="Quy trình phòng thủ & tái huấn luyện khép kín"
        subtitle="Chu trình Closed-Loop từ phát hiện điểm yếu đến sinh dữ liệu đối kháng và tôi luyện mô hình mới"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-3">
          <div className="p-3 rounded-lg bg-blue-50/60 border border-blue-200 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-[11px] font-bold text-blue-800 mb-1">
                <span>1. Dữ liệu gốc</span>
                <Database className="w-3.5 h-3.5" />
              </div>
              <p className="text-xs font-bold text-slate-800 font-mono">120,000 ảnh</p>
            </div>
            <span className="text-[10px] text-slate-500 mt-2">VOC + COCO Train Set</span>
          </div>

          <div className="p-3 rounded-lg bg-red-50/60 border border-red-200 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-[11px] font-bold text-red-800 mb-1">
                <span>2. Dữ liệu bị tấn công</span>
                <ShieldCheck className="w-3.5 h-3.5 text-red-600" />
              </div>
              <p className="text-xs font-bold text-slate-800 font-mono">120,000 ảnh đối kháng</p>
            </div>
            <span className="text-[10px] text-slate-500 mt-2">PGD, FGSM, AutoAttack</span>
          </div>

          <div className="p-3 rounded-lg bg-purple-50/60 border border-purple-200 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-[11px] font-bold text-purple-800 mb-1">
                <span>3. Adv. Training</span>
                <Cpu className="w-3.5 h-3.5 text-purple-600" />
              </div>
              <p className="text-xs font-bold text-slate-800 font-mono">TRADES + CE Loss</p>
            </div>
            <span className="text-[10px] text-slate-500 mt-2">Cosine Annealing Schedule</span>
          </div>

          <div className="p-3 rounded-lg bg-emerald-50/60 border border-emerald-200 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-[11px] font-bold text-emerald-800 mb-1">
                <span>4. Mô hình tôi luyện</span>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              </div>
              <p className="text-xs font-bold text-emerald-700 font-mono">Model v2.1 (Defended)</p>
            </div>
            <span className="text-[10px] text-emerald-700 font-bold mt-2">Best mAP: 48.7%</span>
          </div>

          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-[11px] font-bold text-slate-700 mb-1">
                <span>5. Đánh giá lại</span>
                <Activity className="w-3.5 h-3.5 text-slate-600" />
              </div>
              <p className="text-xs font-bold text-slate-800 font-mono">Robustness: 0.53</p>
            </div>
            <span className="text-[10px] text-emerald-600 font-bold mt-2">↑ +152.4% so với attack</span>
          </div>
        </div>
      </Card>

      {/* 2 & 3. TRAINING CONFIG & LIVE PROGRESS */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
        {/* Left: Training Configuration Form (7 Columns) */}
        <Card
          className="xl:col-span-7"
          title="Cấu hình huấn luyện phòng thủ"
          subtitle="Thiết lập siêu tham số cho Adversarial Training / Fine-tuning"
        >
          <div className="space-y-4 text-xs">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div>
                <label className="text-slate-500 font-medium block mb-1">Chế độ</label>
                <select className="w-full p-1.5 rounded border border-slate-300 bg-white font-medium">
                  <option>Adversarial Training</option>
                  <option>Fine-tuning</option>
                  <option>TRADES Defense</option>
                </select>
              </div>
              <div>
                <label className="text-slate-500 font-medium block mb-1">Epochs</label>
                <input
                  type="number"
                  defaultValue={100}
                  className="w-full p-1.5 rounded border border-slate-300 font-mono"
                />
              </div>
              <div>
                <label className="text-slate-500 font-medium block mb-1">Batch size</label>
                <input
                  type="number"
                  defaultValue={32}
                  className="w-full p-1.5 rounded border border-slate-300 font-mono"
                />
              </div>
              <div>
                <label className="text-slate-500 font-medium block mb-1">Learning Rate</label>
                <input
                  type="text"
                  defaultValue="0.0001"
                  className="w-full p-1.5 rounded border border-slate-300 font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div>
                <label className="text-slate-500 font-medium block mb-1">Scheduler</label>
                <select className="w-full p-1.5 rounded border border-slate-300 bg-white">
                  <option>Cosine Annealing</option>
                  <option>StepLR</option>
                  <option>ReduceLROnPlateau</option>
                </select>
              </div>
              <div>
                <label className="text-slate-500 font-medium block mb-1">Loss Function</label>
                <input
                  type="text"
                  defaultValue="Cross Entropy + TRADES"
                  className="w-full p-1.5 rounded border border-slate-300 font-mono font-medium"
                />
              </div>
              <div>
                <label className="text-slate-500 font-medium block mb-1">Augmentation</label>
                <input
                  type="text"
                  defaultValue="Mosaic + MixUp + ColorJitter"
                  className="w-full p-1.5 rounded border border-slate-300"
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-slate-100">
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" defaultChecked className="rounded text-blue-600" />
                  <span className="text-slate-700 font-medium">Mixed Precision (FP16)</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" defaultChecked className="rounded text-blue-600" />
                  <span className="text-slate-700 font-medium">Early Stopping (Patience: 15)</span>
                </label>
              </div>

              <div className="flex items-center gap-2">
                <Button variant="secondary" size="sm">
                  Lưu cấu hình
                </Button>
                <Button variant="primary" size="sm" icon={Play}>
                  ▶ Khởi chạy huấn luyện
                </Button>
              </div>
            </div>
          </div>
        </Card>

        {/* Right: Live Training Progress Monitor (5 Columns) */}
        <div className="xl:col-span-5 space-y-4">
          <Card
            title="Tiến trình huấn luyện hiện tại"
            headerAction={<Badge variant="primary" dot>Đang huấn luyện</Badge>}
          >
            <div className="space-y-3 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-500">Run ID:</span>
                <span className="font-mono font-bold text-blue-600">run-2025-05-12-001</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Phương pháp:</span>
                <span className="font-semibold text-slate-800">PGD Adversarial Training</span>
              </div>

              {/* Progress bar */}
              <div>
                <div className="flex justify-between text-xs font-bold mb-1">
                  <span>Epoch {epoch} / 100</span>
                  <span className="text-blue-600">{progress}%</span>
                </div>
                <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-600 transition-all" style={{ width: `${progress}%` }} />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 pt-1">
                <div className="p-2 rounded bg-slate-50 border border-slate-150 text-center">
                  <div className="text-[10px] text-slate-500">Đã chạy</div>
                  <div className="font-mono font-bold text-slate-800">01:34:22</div>
                </div>
                <div className="p-2 rounded bg-slate-50 border border-slate-150 text-center">
                  <div className="text-[10px] text-slate-500">Còn lại (Dự kiến)</div>
                  <div className="font-mono font-bold text-slate-800">00:48:11</div>
                </div>
              </div>

              <div className="flex items-center gap-2 pt-2 border-t border-slate-100">
                <Button variant="danger" size="sm" className="flex-1 text-xs" icon={Pause}>
                  Dừng huấn luyện
                </Button>
                <Button variant="secondary" size="sm" className="flex-1 text-xs" icon={FileCode}>
                  Xem Logs
                </Button>
              </div>
            </div>
          </Card>

          {/* Hardware Resource Monitor */}
          <Card title="Tài nguyên & Hardware Runtime">
            <div className="grid grid-cols-3 gap-2 text-xs text-center font-medium">
              <div className="p-2 rounded bg-slate-50 border border-slate-150">
                <div className="text-[10px] text-slate-500">2× RTX 4090</div>
                <div className="text-sm font-bold text-blue-600">67%</div>
              </div>
              <div className="p-2 rounded bg-slate-50 border border-slate-150">
                <div className="text-[10px] text-slate-500">VRAM GPU</div>
                <div className="text-sm font-bold text-purple-600">19.8 / 24GB</div>
              </div>
              <div className="p-2 rounded bg-slate-50 border border-slate-150">
                <div className="text-[10px] text-slate-500">RAM Hệ thống</div>
                <div className="text-sm font-bold text-emerald-600">22.4 / 64GB</div>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* 4. DEFENSE STRATEGIES CARDS */}
      <Card
        title="Chiến lược phòng thủ khả dụng"
        subtitle="Các kỹ thuật nâng cao sức kháng cự cho mạng nơ-ron"
        headerAction={
          <Button variant="outline" size="sm">
            + Thêm chiến lược tùy chỉnh
          </Button>
        }
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
          {STRATEGIES.map((st) => (
            <div
              key={st.name}
              className={cn(
                "p-3 rounded-lg border text-left transition-all space-y-1.5",
                st.status === "active"
                  ? "border-blue-600 bg-blue-50/50"
                  : "border-slate-200 bg-white hover:bg-slate-50"
              )}
            >
              <div className="flex items-center justify-between">
                <Badge variant={st.status === "active" ? "primary" : "default"}>
                  {st.badge}
                </Badge>
              </div>
              <div className="text-xs font-bold text-slate-900">{st.name}</div>
              <p className="text-[11px] text-slate-500 leading-snug">{st.desc}</p>
            </div>
          ))}
        </div>
      </Card>

      {/* 5. 3-WAY COMPARISON: BEFORE vs ATTACKED vs DEFENDED */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
        {/* Comparison Table (7 Columns) */}
        <Card
          className="xl:col-span-7"
          title="Bảng so sánh 3 trạng thái: Trước vs Tấn công vs Phòng thủ"
          subtitle="Đo lường mức độ hồi phục độ chính xác và kháng cự đối kháng"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 font-semibold bg-slate-50">
                  <th className="py-2.5 px-3">Chỉ số Metric</th>
                  <th className="py-2.5 px-3 text-blue-700 font-bold">1. Trước tấn công</th>
                  <th className="py-2.5 px-3 text-red-700 font-bold">2. Sau tấn công</th>
                  <th className="py-2.5 px-3 text-emerald-700 font-bold">3. Sau phòng thủ</th>
                  <th className="py-2.5 px-3 text-right">Mức cải thiện</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                {DEFENSE_COMPARISON.map((row) => (
                  <tr key={row.metric} className="hover:bg-slate-50">
                    <td className="py-2.5 px-3 font-semibold text-slate-900">{row.metric}</td>
                    <td className="py-2.5 px-3 font-mono text-blue-700">{row.before}</td>
                    <td className="py-2.5 px-3 font-mono text-red-600">{row.attacked}</td>
                    <td className="py-2.5 px-3 font-mono font-bold text-emerald-600">{row.defended}</td>
                    <td className="py-2.5 px-3 text-right">
                      <Badge variant="success" className="font-mono">
                        {row.improvement}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Defense Radar Comparison (5 Columns) */}
        <Card
          className="xl:col-span-5"
          title="Radar so sánh đa thuộc tính"
          subtitle="Đối sánh trực diện Trước, Sau tấn công và Sau phòng thủ"
        >
          <RobustnessRadar height={250} showDefended={true} />
        </Card>
      </div>

      {/* RECOMMENDATIONS CARDS */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
        <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 space-y-1">
          <span className="font-bold text-emerald-800 block">✓ Hiệu quả phòng thủ tốt</span>
          <p className="text-emerald-700">mAP@0.5 hồi phục từ 31.6% lên 48.7%, ASR giảm từ 87.6% xuống 22.8%.</p>
        </div>
        <div className="p-3 rounded-lg bg-blue-50 border border-blue-200 space-y-1">
          <span className="font-bold text-blue-800 block">ℹ Tăng cường Adv. Training</span>
          <p className="text-blue-700">Nên bổ sung thêm kịch bản AutoAttack vào tập huấn luyện TRADES.</p>
        </div>
        <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 space-y-1">
          <span className="font-bold text-amber-800 block">⚠ Cân nhắc kết hợp</span>
          <p className="text-amber-700">Kết hợp tầng Feature Denoising để bảo vệ trước các tấn công tần số cao.</p>
        </div>
        <div className="p-3 rounded-lg bg-purple-50 border border-purple-200 space-y-1">
          <span className="font-bold text-purple-800 block">💡 Theo dõi độ trễ FPS</span>
          <p className="text-purple-700">FPS suy giảm nhẹ 3.9% (từ 34.2 xuống 31.8 FPS) vẫn đạt thời gian thực.</p>
        </div>
      </div>
    </div>
  );
}
