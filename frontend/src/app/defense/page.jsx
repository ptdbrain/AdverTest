"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  ShieldCheck,
  Play,
  Copy,
  Check,
  Download,
  Upload,
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
  Terminal,
  Lock,
  RefreshCw,
} from "lucide-react";
import PageHeader from "@/components/layout/PageHeader";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import RobustnessRadar from "@/components/metrics/RobustnessRadar";
import { DEFENSE_COMPARISON } from "@/data/mockData";
import { cn } from "@/lib/utils";

export default function DefensePage() {
  // Strategy & Hyperparameter Config State
  const [strategy, setStrategy] = useState("adversarial_training");
  const [baseModel, setBaseModel] = useState("weights/yolo11s-clean-b0_best.pt");
  const [datasetYaml, setDatasetYaml] = useState("data/anonymized/kitti-de/data.yaml");
  const [targetRecipe, setTargetRecipe] = useState("fgsm,depth_fog,pgd");
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);
  const [device, setDevice] = useState("cuda:0");

  // Copy & Script Download States
  const [isCopied, setIsCopied] = useState(false);

  // Defended Upload & Re-test Locked Protocol States
  const [defendedFile, setDefendedFile] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationDone, setEvaluationDone] = useState(false);

  // Computed CLI Command
  const generatedCommand = `python scripts/train_defence.py --model ${baseModel} --dataset ${datasetYaml} --recipe ${targetRecipe} --strategy ${strategy} --epochs ${epochs} --batch-size ${batchSize} --lr ${learningRate} --output-dir weights/defended --device ${device}`;

  const handleCopyCommand = () => {
    navigator.clipboard.writeText(generatedCommand);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2500);
  };

  const handleDownloadScript = (format) => {
    const isWindows = format === "bat";
    const scriptContent = isWindows
      ? `@echo off\necho ====================================================\necho ADVERSAI LAB - LOCAL DEFENCE TRAINING\necho ====================================================\n${generatedCommand}\npause\n`
      : `#!/usr/bin/env bash\n# AdversAI Lab - Local Defence Training Script\necho "===================================================="\necho "ADVERSAI LAB - LOCAL DEFENCE TRAINING"\necho "===================================================="\n${generatedCommand}\n`;

    const blob = new Blob([scriptContent], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = isWindows ? "run_train_defence.bat" : "run_train_defence.sh";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDefendedUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setDefendedFile(file);
      setEvaluationDone(false);
    }
  };

  const handleRunLockedEvaluation = () => {
    setIsEvaluating(true);
    setTimeout(() => {
      setIsEvaluating(false);
      setEvaluationDone(true);
    }, 1500);
  };

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Huấn luyện phòng thủ & Đánh giá đối kháng"
        subtitle="Cấu hình siêu tham số để sinh lệnh huấn luyện phòng thủ chạy máy cá nhân (Local GPU), sau đó upload mô hình mới để đánh giá phục hồi theo Locked Protocol."
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Huấn luyện phòng thủ" },
        ]}
      />

      {/* BANNER: ARCHITECTURE EXPLANATION */}
      <div className="p-4 rounded-xl bg-blue-50/80 border border-blue-200 text-xs text-blue-900 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-xs">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 text-white flex items-center justify-center font-bold flex-shrink-0 mt-0.5">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <div className="font-bold text-blue-900">
              Cơ chế Phân tách Tính toán (Offloaded Training & Central Evaluation):
            </div>
            <div className="text-[11px] text-blue-700 mt-0.5 leading-relaxed">
              Quá trình huấn luyện tăng cường (Adversarial Training / Fine-tuning) yêu cầu nhiều compute GPU. Web hỗ trợ cấu hình và sinh kịch bản CLI một dòng để bạn chạy trực tiếp trên máy trạm của bạn. Sau khi có file checkpoint mới (`.pt`), hãy tải lên để hệ thống tự động khóa cấu hình tấn công (Locked Protocol) và đo lường tỷ lệ phục hồi độ chính xác.
            </div>
          </div>
        </div>
      </div>

      {/* MAIN TWO COLUMNS */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
        {/* LEFT COLUMN: Defence Strategy & Local Command Generator (7 Columns) */}
        <div className="xl:col-span-7 space-y-5">
          {/* 1. DEFENCE STRATEGY CONFIG */}
          <Card
            title="1. Cấu hình chiến lược phòng thủ"
            subtitle="Thiết lập thuật toán và siêu tham số huấn luyện"
          >
            <div className="space-y-4 text-xs">
              {/* Strategy Radio Grid */}
              <div>
                <label className="font-bold text-slate-700 block mb-1.5">
                  Phương pháp phòng thủ:
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                  {[
                    {
                      id: "adversarial_training",
                      name: "Adversarial Training",
                      desc: "Tiêm mẫu nhiễu PGD / TRADES trong từng epoch",
                    },
                    {
                      id: "augmentation_mix",
                      name: "Augmentation Mix",
                      desc: "Trộn 40% ảnh mô phỏng thời tiết sương mù, mưa, chói",
                    },
                    {
                      id: "robust_finetune",
                      name: "Robust Fine-tuning",
                      desc: "Tối ưu hóa Cosine Annealing bảo toàn mAP clean",
                    },
                  ].map((s) => (
                    <div
                      key={s.id}
                      onClick={() => setStrategy(s.id)}
                      className={cn(
                        "p-3 rounded-lg border cursor-pointer transition-all flex flex-col justify-between",
                        strategy === s.id
                          ? "border-blue-600 bg-blue-50/60 shadow-xs ring-1 ring-blue-500"
                          : "border-slate-200 bg-white hover:border-slate-300"
                      )}
                    >
                      <div className="font-bold text-slate-800">{s.name}</div>
                      <div className="text-[10px] text-slate-500 mt-1">{s.desc}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Hyperparameters Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-2 border-t border-slate-100">
                <div>
                  <label className="text-slate-500 font-medium block mb-1">Mô hình gốc (Base Model)</label>
                  <input
                    type="text"
                    value={baseModel}
                    onChange={(e) => setBaseModel(e.target.value)}
                    className="w-full p-2 border border-slate-300 rounded font-mono text-slate-800 text-xs bg-slate-50"
                  />
                </div>

                <div>
                  <label className="text-slate-500 font-medium block mb-1">Đòn đối kháng cần kháng</label>
                  <input
                    type="text"
                    value={targetRecipe}
                    onChange={(e) => setTargetRecipe(e.target.value)}
                    className="w-full p-2 border border-slate-300 rounded font-mono text-slate-800 text-xs"
                  />
                </div>

                <div>
                  <label className="text-slate-500 font-medium block mb-1">Số Epochs</label>
                  <input
                    type="number"
                    value={epochs}
                    onChange={(e) => setEpochs(Number(e.target.value))}
                    className="w-full p-2 border border-slate-300 rounded font-mono text-slate-800 text-xs"
                  />
                </div>

                <div>
                  <label className="text-slate-500 font-medium block mb-1">Batch Size</label>
                  <input
                    type="number"
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    className="w-full p-2 border border-slate-300 rounded font-mono text-slate-800 text-xs"
                  />
                </div>

                <div>
                  <label className="text-slate-500 font-medium block mb-1">Learning Rate</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={learningRate}
                    onChange={(e) => setLearningRate(Number(e.target.value))}
                    className="w-full p-2 border border-slate-300 rounded font-mono text-slate-800 text-xs"
                  />
                </div>

                <div>
                  <label className="text-slate-500 font-medium block mb-1">Thiết bị (Device)</label>
                  <select
                    value={device}
                    onChange={(e) => setDevice(e.target.value)}
                    className="w-full p-2 border border-slate-300 rounded font-mono text-slate-800 text-xs bg-white"
                  >
                    <option value="cuda:0">cuda:0 (NVIDIA GPU)</option>
                    <option value="cpu">cpu (Máy CPU)</option>
                  </select>
                </div>
              </div>
            </div>
          </Card>

          {/* 2. LOCAL COMMAND GENERATOR & DOWNLOADABLE SCRIPT */}
          <Card
            title="2. Trình sinh lệnh & Kịch bản huấn luyện (Local Script Generator)"
            subtitle="Copy lệnh CLI 1 dòng hoặc tải file script để chạy trực tiếp trên máy GPU cá nhân"
            headerAction={<Badge variant="primary">Standalone Script</Badge>}
          >
            <div className="space-y-3 text-xs">
              <div className="relative">
                <div className="p-3.5 rounded-xl bg-slate-950 text-emerald-400 font-mono text-xs overflow-x-auto leading-relaxed border border-slate-800 shadow-inner">
                  <span className="text-slate-500 select-none">$ </span>
                  {generatedCommand}
                </div>

                <button
                  type="button"
                  onClick={handleCopyCommand}
                  className="absolute top-2.5 right-2.5 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white font-sans text-xs flex items-center gap-1 border border-slate-700 transition-colors shadow-xs"
                >
                  {isCopied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400 font-bold">Đã Copy!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy Lệnh</span>
                    </>
                  )}
                </button>
              </div>

              {/* Download script files buttons */}
              <div className="flex flex-wrap gap-2 pt-2">
                <Button
                  variant="secondary"
                  size="sm"
                  icon={Download}
                  onClick={() => handleDownloadScript("sh")}
                  className="bg-slate-100 text-slate-700 hover:bg-slate-200 text-xs"
                >
                  Tải run_train_defence.sh (Linux / WSL)
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  icon={Download}
                  onClick={() => handleDownloadScript("bat")}
                  className="bg-slate-100 text-slate-700 hover:bg-slate-200 text-xs"
                >
                  Tải run_train_defence.bat (Windows)
                </Button>
              </div>
            </div>
          </Card>
        </div>

        {/* RIGHT COLUMN: Upload Defended Model & Locked Protocol Re-Test (5 Columns) */}
        <div className="xl:col-span-5 space-y-5">
          <Card
            title="3. Upload mô hình đã phòng thủ & Test lại"
            subtitle="Tự động khóa cùng cấu hình tấn công (Locked Protocol) để đo lường tỷ lệ phục hồi"
          >
            <div className="space-y-4 text-xs">
              {/* Protocol lock banner */}
              <div className="p-3 rounded-lg bg-amber-50/80 border border-amber-200 text-amber-900 flex items-center gap-2">
                <Lock className="w-4 h-4 text-amber-700 flex-shrink-0" />
                <span>
                  <strong>Locked Protocol:</strong> Cố định cùng dataset (KITTI), seed (42), và chuỗi tấn công (Depth Fog + PGD) để kết quả đối chiếu có tính khoa học tuyệt đối.
                </span>
              </div>

              {/* Upload Defended Weight Box */}
              <label className="border-2 border-dashed border-slate-300 hover:border-blue-500 rounded-xl p-5 flex flex-col items-center justify-center gap-2 cursor-pointer bg-slate-50/50 transition-colors">
                <Upload className="w-6 h-6 text-slate-400" />
                <div className="text-xs font-bold text-slate-700">
                  {defendedFile ? defendedFile.name : "Tải lên Checkpoint đã tôi luyện (.pt / .pth)"}
                </div>
                <div className="text-[10px] text-slate-400">
                  Ví dụ: yolo11s_defended_adversarial_training.pt
                </div>
                <input type="file" onChange={handleDefendedUpload} className="hidden" />
              </label>

              {/* Action Button */}
              <Button
                variant="primary"
                onClick={handleRunLockedEvaluation}
                disabled={isEvaluating}
                icon={isEvaluating ? RefreshCw : ShieldCheck}
                className={cn(
                  "w-full justify-center py-2.5 text-xs font-bold shadow-sm",
                  isEvaluating ? "animate-pulse" : "bg-emerald-600 hover:bg-emerald-700"
                )}
              >
                {isEvaluating ? "Đang chạy đánh giá đối kháng..." : "Chạy Đánh Giá Đối Chiếu (Locked Protocol)"}
              </Button>

              {/* COMPARATIVE EVALUATION RESULTS TABLE */}
              {evaluationDone && (
                <div className="space-y-3 pt-3 border-t border-slate-200 animate-fade-in">
                  <div className="flex items-center justify-between">
                    <div className="font-bold text-slate-800 text-xs flex items-center gap-1.5">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                      <span>Kết quả phục hồi sau phòng thủ:</span>
                    </div>
                    <Badge variant="success" className="font-mono">
                      Phục hồi +68.4%
                    </Badge>
                  </div>

                  <div className="overflow-x-auto rounded-lg border border-slate-200">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold text-[10px] uppercase">
                        <tr>
                          <th className="p-2">Chỉ số</th>
                          <th className="p-2">Trước Đòn (Clean)</th>
                          <th className="p-2">Bị Tấn Công</th>
                          <th className="p-2 text-emerald-700">Sau Phòng Thủ</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {DEFENSE_COMPARISON.slice(0, 4).map((row, i) => (
                          <tr key={i} className="hover:bg-slate-50">
                            <td className="p-2 font-semibold text-slate-700">{row.metric}</td>
                            <td className="p-2 font-mono text-slate-500">{row.before}</td>
                            <td className="p-2 font-mono text-red-600">{row.attacked}</td>
                            <td className="p-2 font-mono font-bold text-emerald-700 bg-emerald-50/50">
                              {row.defended} ({row.improvement})
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
