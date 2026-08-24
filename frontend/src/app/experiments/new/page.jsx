"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Target,
  Layers,
  Box,
  Grid,
  Activity,
  CheckCircle2,
  Upload,
  Cpu,
  Save,
  ArrowRight,
  ArrowLeft,
  Check,
  Sparkles,
  Info,
  Database,
  FolderCheck,
  FolderOpen,
  FileCheck,
} from "lucide-react";
import PageHeader from "@/components/layout/PageHeader";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import {
  PROBLEM_TYPES,
  MODEL_ARCHITECTURES,
  AVAILABLE_DATASETS,
  AVAILABLE_MODELS,
  CLASS_LABELS_DATA,
} from "@/data/mockData";
import { cn } from "@/lib/utils";

export default function ConfigureProblemPage() {
  const router = useRouter();
  const [selectedTask, setSelectedTask] = useState("detection2d");
  const [selectedArch, setSelectedArch] = useState("yolo");
  
  // Active selected dataset and model objects (Defaulting to the local files in workspace)
  const [selectedDatasetId, setSelectedDatasetId] = useState("kitti_anonymized_de");
  const [selectedModelId, setSelectedModelId] = useState("local_yolo11s_clean");

  const [dataSource, setDataSource] = useState("repository");
  const [modelSource, setModelSource] = useState("default_lib");
  const [mixedPrecision, setMixedPrecision] = useState(true);
  const [batchSize, setBatchSize] = useState(16);
  const [gpuCount, setGpuCount] = useState(1);
  const [expName, setExpName] = useState("EXP-2025-05-12-001");
  const [expDesc, setExpDesc] = useState("Đánh giá khả năng phát hiện đối tượng trên tập dữ liệu KITTI có sẵn trong máy.");
  const [expTags, setExpTags] = useState("yolo11, kitti, object-detection, local");

  const currentDataset = AVAILABLE_DATASETS.find((d) => d.id === selectedDatasetId) || AVAILABLE_DATASETS[0];
  const currentModel = AVAILABLE_MODELS.find((m) => m.id === selectedModelId) || AVAILABLE_MODELS[0];

  const getTaskIcon = (id) => {
    switch (id) {
      case "detection2d": return Target;
      case "segmentation": return Layers;
      case "detection3d": return Box;
      case "classification": return Grid;
      default: return Activity;
    }
  };

  const handleNext = () => {
    router.push(`/experiments/${expName}/attack`);
  };

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Cấu hình bài toán"
        subtitle="Thiết lập bài toán, mô hình, dữ liệu có sẵn trong workspace và các tùy chọn để bắt đầu thí nghiệm đánh giá & phòng thủ AI đối kháng."
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Cấu hình bài toán", href: "/experiments/new" },
          { label: "Tạo thí nghiệm mới" },
        ]}
      />

      {/* BANNER: LOCAL WORKSPACE RESOURCES DETECTED */}
      <div className="p-3.5 rounded-lg bg-blue-50/80 border border-blue-200 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 text-white flex items-center justify-center font-bold">
            <FolderCheck className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs font-bold text-blue-900">
              Đã nhận diện tài nguyên có sẵn trong thư mục máy (Local Workspace)
            </div>
            <div className="text-[11px] text-blue-700">
              Bao gồm: Checkpoints <strong>yolo11s-clean-b0_best.pt</strong>, <strong>pointpillars_kitti_3class.pth</strong>, <strong>yolo11n-face.onnx</strong> và dữ liệu <strong>data/anonymized/kitti-de/</strong>.
            </div>
          </div>
        </div>
        <Badge variant="primary">5 Models · 5 Datasets Local</Badge>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5 items-start">
        {/* LEFT & CENTER: Form Controls (2 Columns) */}
        <div className="xl:col-span-2 space-y-5">
          {/* 1. Chọn loại bài toán */}
          <Card
            title="1. Chọn loại bài toán"
            subtitle="Xác định nhiệm vụ thị giác máy tính cần đánh giá kiểm thử"
          >
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
              {PROBLEM_TYPES.map((task) => {
                const Icon = getTaskIcon(task.id);
                const isSelected = selectedTask === task.id;
                return (
                  <div
                    key={task.id}
                    onClick={() => {
                      setSelectedTask(task.id);
                      if (task.id === "detection3d") {
                        setSelectedArch("pointpillars");
                        setSelectedModelId("local_pointpillars_kitti");
                        setSelectedDatasetId("kitti_3d_lidar");
                      }
                    }}
                    className={cn(
                      "p-3 rounded-lg border text-left cursor-pointer transition-all relative flex flex-col justify-between h-[104px]",
                      isSelected
                        ? "border-blue-600 bg-blue-50/70 shadow-sm"
                        : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                    )}
                  >
                    {isSelected && (
                      <div className="absolute top-2 right-2 w-4 h-4 rounded-full bg-blue-600 text-white flex items-center justify-center">
                        <Check className="w-2.5 h-2.5" />
                      </div>
                    )}
                    <div
                      className={cn(
                        "w-8 h-8 rounded-lg flex items-center justify-center border mb-2",
                        isSelected
                          ? "bg-blue-600 text-white border-blue-600"
                          : "bg-slate-100 text-slate-600 border-slate-200"
                      )}
                    >
                      <Icon className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-xs font-bold text-slate-800 leading-tight">
                        {task.name}
                      </div>
                      <div className="text-[10px] text-slate-500 line-clamp-1 mt-0.5">
                        {task.desc}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* 2. Chọn họ kiến trúc mô hình */}
          <Card
            title="2. Chọn họ kiến trúc mô hình"
            subtitle="Chọn họ kiến trúc deep learning và phiên bản checkpoints cần kiểm thử"
          >
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {MODEL_ARCHITECTURES.map((arch) => {
                const isSelected = selectedArch === arch.id;
                return (
                  <div
                    key={arch.id}
                    onClick={() => {
                      setSelectedArch(arch.id);
                      const matchingModel = AVAILABLE_MODELS.find((m) => m.architecture === arch.id);
                      if (matchingModel) setSelectedModelId(matchingModel.id);
                    }}
                    className={cn(
                      "p-3 rounded-lg border text-left cursor-pointer transition-all space-y-2",
                      isSelected
                        ? "border-blue-600 bg-blue-50/50 shadow-sm"
                        : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-800">{arch.name}</span>
                      {isSelected && (
                        <span className="w-3.5 h-3.5 rounded-full bg-blue-600 text-white flex items-center justify-center">
                          <Check className="w-2.5 h-2.5" />
                        </span>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-500">{arch.family}</p>
                    <select
                      value={currentModel.name}
                      onChange={(e) => {
                        e.stopPropagation();
                        const found = AVAILABLE_MODELS.find((m) => m.name.includes(e.target.value) || m.id === e.target.value);
                        if (found) {
                          setSelectedModelId(found.id);
                          setSelectedArch(found.architecture || arch.id);
                        }
                      }}
                      className="w-full text-xs py-1 px-2 border border-slate-300 rounded bg-white font-medium text-slate-700 focus:outline-none focus:border-blue-500"
                    >
                      {arch.variants.map((v) => (
                        <option key={v} value={v}>
                          {v}
                        </option>
                      ))}
                    </select>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* 3 & 4. Nguồn Dữ liệu & Nguồn Mô hình */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Nguồn dữ liệu */}
            <Card title="3. Nguồn dữ liệu">
              <div className="space-y-2.5 text-xs">
                <label
                  className={cn(
                    "flex items-start gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors",
                    dataSource === "repository"
                      ? "border-blue-500 bg-blue-50/40"
                      : "border-slate-200 hover:bg-slate-50"
                  )}
                >
                  <input
                    type="radio"
                    name="data_src"
                    checked={dataSource === "repository"}
                    onChange={() => setDataSource("repository")}
                    className="mt-0.5 text-blue-600"
                  />
                  <div className="flex-1">
                    <div className="font-semibold text-slate-800 flex items-center gap-1.5">
                      <span>Kho dữ liệu có sẵn trong máy & chuẩn</span>
                      <Badge variant="success" className="text-[9px] py-0 px-1">Local / Repo</Badge>
                    </div>
                    <div className="text-[11px] text-slate-500 mb-2">Chọn bộ dữ liệu KITTI, DPatch hoặc chuẩn COCO/VOC</div>
                    
                    {/* Dataset Dropdown Selector */}
                    <select
                      value={selectedDatasetId}
                      onChange={(e) => setSelectedDatasetId(e.target.value)}
                      className="w-full p-1.5 rounded border border-slate-300 bg-white font-semibold text-blue-700 text-xs focus:outline-none focus:border-blue-500"
                    >
                      {AVAILABLE_DATASETS.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.isLocal ? "📍 [Local] " : "📦 [Chuẩn] "} {d.name} ({d.samples.toLocaleString()} samples · {d.size})
                        </option>
                      ))}
                    </select>
                  </div>
                </label>

                <label
                  className={cn(
                    "flex items-start gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors",
                    dataSource === "user_upload"
                      ? "border-blue-500 bg-blue-50/40"
                      : "border-slate-200 hover:bg-slate-50"
                  )}
                >
                  <input
                    type="radio"
                    name="data_src"
                    checked={dataSource === "user_upload"}
                    onChange={() => setDataSource("user_upload")}
                    className="mt-0.5 text-blue-600"
                  />
                  <div>
                    <div className="font-semibold text-slate-800">Tải lên bộ dữ liệu tùy chỉnh</div>
                    <div className="text-[11px] text-slate-500">Hỗ trợ zip ảnh & nhãn YOLO / COCO JSON</div>
                    <Button variant="secondary" size="sm" icon={Upload} className="mt-1.5 text-xs">
                      Tải lên Dataset
                    </Button>
                  </div>
                </label>
              </div>
            </Card>

            {/* Nguồn mô hình */}
            <Card title="4. Nguồn mô hình">
              <div className="space-y-2.5 text-xs">
                <label
                  className={cn(
                    "flex items-start gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors",
                    modelSource === "default_lib"
                      ? "border-blue-500 bg-blue-50/40"
                      : "border-slate-200 hover:bg-slate-50"
                  )}
                >
                  <input
                    type="radio"
                    name="model_src"
                    checked={modelSource === "default_lib"}
                    onChange={() => setModelSource("default_lib")}
                    className="mt-0.5 text-blue-600"
                  />
                  <div className="flex-1">
                    <div className="font-semibold text-slate-800 flex items-center gap-1.5">
                      <span>Mô hình & Checkpoints có sẵn</span>
                      <Badge variant="success" className="text-[9px] py-0 px-1">Local / Pretrained</Badge>
                    </div>
                    <div className="text-[11px] text-slate-500 mb-2">Checkpoints đã có trong `checkpoints/` và `data/`</div>

                    {/* Model Dropdown Selector */}
                    <select
                      value={selectedModelId}
                      onChange={(e) => {
                        const m = AVAILABLE_MODELS.find((mod) => mod.id === e.target.value);
                        if (m) {
                          setSelectedModelId(m.id);
                          if (m.architecture) setSelectedArch(m.architecture);
                        }
                      }}
                      className="w-full p-1.5 rounded border border-slate-300 bg-white font-semibold text-blue-700 text-xs focus:outline-none focus:border-blue-500"
                    >
                      {AVAILABLE_MODELS.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.isLocal ? "📍 [Local] " : "🤖 "} {m.name} ({m.params} · {m.format})
                        </option>
                      ))}
                    </select>
                  </div>
                </label>

                <label
                  className={cn(
                    "flex items-start gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors",
                    modelSource === "user_upload"
                      ? "border-blue-500 bg-blue-50/40"
                      : "border-slate-200 hover:bg-slate-50"
                  )}
                >
                  <input
                    type="radio"
                    name="model_src"
                    checked={modelSource === "user_upload"}
                    onChange={() => setModelSource("user_upload")}
                    className="mt-0.5 text-blue-600"
                  />
                  <div>
                    <div className="font-semibold text-slate-800">Tải lên file trọng số tùy chỉnh</div>
                    <div className="text-[11px] text-slate-500">Hỗ trợ .pt, .pth, .onnx, .engine</div>
                    <Button variant="secondary" size="sm" icon={Upload} className="mt-1.5 text-xs">
                      Tải lên Weights
                    </Button>
                  </div>
                </label>
              </div>
            </Card>
          </div>

          {/* 5 & 6. Data Preview & Class Labels */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 5. Xem trước dữ liệu */}
            <Card
              title="5. Xem trước dữ liệu"
              subtitle={`Ảnh mẫu trong tập "${currentDataset.name}"`}
            >
              <div className="space-y-2">
                <div className="grid grid-cols-3 gap-2">
                  {[1, 2, 3, 4, 5].map((idx) => (
                    <div
                      key={idx}
                      className="h-20 rounded bg-slate-100 border border-slate-300 relative overflow-hidden flex flex-col items-center justify-center text-slate-500 font-mono text-[10px] p-1 text-center"
                    >
                      <Database className="w-4 h-4 text-slate-400 mb-1" />
                      <span>{currentDataset.id}_{idx}.jpg</span>
                    </div>
                  ))}
                  <div className="h-20 rounded bg-blue-900/80 border border-blue-700 text-white font-bold text-xs flex items-center justify-center p-2 text-center">
                    + {(Math.max(1, currentDataset.samples - 5)).toLocaleString()} mẫu
                  </div>
                </div>
                <div className="p-2 rounded bg-slate-50 border border-slate-200 text-[11px] text-slate-600 space-y-1">
                  <p>{currentDataset.description}</p>
                  {currentDataset.localPath && (
                    <p className="font-mono text-blue-700 font-semibold text-[10px]">
                      📂 Thư mục: {currentDataset.localPath}
                    </p>
                  )}
                </div>
              </div>
            </Card>

            {/* 6. Nhãn lớp (Class labels) */}
            <Card
              title="6. Nhãn lớp (Class labels)"
              subtitle={`Tổng số: ${currentDataset.classes || currentDataset.classLabels?.length || 8} lớp đối tượng`}
              headerAction={
                <Badge variant="primary">
                  {currentDataset.format}
                </Badge>
              }
            >
              <div className="max-h-36 overflow-y-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500 font-semibold">
                      <th className="py-1 px-2">ID</th>
                      <th className="py-1 px-2">Tên lớp</th>
                      <th className="py-1 px-2 text-right">Số lượng mẫu</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                    {(currentDataset.classLabels || CLASS_LABELS_DATA).map((c) => (
                      <tr key={c.id} className="hover:bg-slate-50">
                        <td className="py-1 px-2 font-mono text-slate-400">{c.id}</td>
                        <td className="py-1 px-2 font-semibold text-slate-800">{c.name}</td>
                        <td className="py-1 px-2 text-right font-mono">{c.count?.toLocaleString() || 500}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          {/* 7 & 8. Model Info & Runtime */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 7. Thông tin mô hình */}
            <Card title="7. Thông tin mô hình">
              <div className="space-y-1.5 text-xs font-medium divide-y divide-slate-100">
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Tên mô hình</span>
                  <span className="font-semibold text-slate-800 font-mono">{currentModel.name}</span>
                </div>
                {currentModel.localPath && (
                  <div className="flex justify-between py-1 bg-blue-50/50 px-1 rounded">
                    <span className="text-blue-800 font-semibold">File path</span>
                    <span className="font-mono text-blue-700 text-[11px]">{currentModel.localPath}</span>
                  </div>
                )}
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Họ kiến trúc</span>
                  <span className="text-slate-800">{currentModel.family}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Định dạng file</span>
                  <span className="font-mono text-slate-800">{currentModel.format}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Số lượng tham số</span>
                  <span className="text-blue-600 font-bold font-mono">{currentModel.params}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Kích thước file</span>
                  <span className="text-slate-800 font-mono">{currentModel.fileSize}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Độ phân giải đầu vào</span>
                  <span className="text-slate-800 font-mono">{currentModel.inputSize}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Độ chính xác gốc (Clean Baseline)</span>
                  <span className="text-emerald-600 font-bold font-mono">{currentModel.mapBaseline}</span>
                </div>
              </div>
            </Card>

            {/* 8. Tài nguyên & Runtime */}
            <Card title="8. Tài nguyên & Runtime">
              <div className="space-y-2 text-xs">
                <div>
                  <label className="text-slate-500 font-medium block mb-1">Thiết bị tính toán</label>
                  <input
                    type="text"
                    readOnly
                    value="NVIDIA RTX 4090 (24GB VRAM)"
                    className="w-full px-2.5 py-1.5 rounded border border-slate-300 bg-slate-50 text-slate-800 font-mono font-semibold"
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-slate-500 font-medium block mb-1">Số GPU</label>
                    <select
                      value={gpuCount}
                      onChange={(e) => setGpuCount(Number(e.target.value))}
                      className="w-full px-2 py-1.5 rounded border border-slate-300 bg-white"
                    >
                      <option value={1}>1 GPU</option>
                      <option value={2}>2 GPU (DDP)</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-slate-500 font-medium block mb-1">Batch size</label>
                    <select
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                      className="w-full px-2 py-1.5 rounded border border-slate-300 bg-white"
                    >
                      <option value={8}>8</option>
                      <option value={16}>16</option>
                      <option value={32}>32</option>
                    </select>
                  </div>
                </div>
                <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                  <span className="text-slate-700 font-medium">Mixed Precision (FP16)</span>
                  <button
                    type="button"
                    onClick={() => setMixedPrecision(!mixedPrecision)}
                    className={cn(
                      "w-9 h-5 rounded-full transition-colors relative",
                      mixedPrecision ? "bg-blue-600" : "bg-slate-300"
                    )}
                  >
                    <span
                      className={cn(
                        "w-4 h-4 rounded-full bg-white absolute top-0.5 transition-transform",
                        mixedPrecision ? "right-0.5" : "left-0.5"
                      )}
                    />
                  </button>
                </div>
              </div>
            </Card>
          </div>

          {/* 9. Thông tin thí nghiệm */}
          <Card title="9. Thông tin thí nghiệm">
            <div className="space-y-3 text-xs">
              <div>
                <label className="text-slate-600 font-semibold block mb-1">Tên thí nghiệm</label>
                <input
                  type="text"
                  value={expName}
                  onChange={(e) => setExpName(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 font-mono font-bold text-slate-800 focus:outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="text-slate-600 font-semibold block mb-1">Mô tả thí nghiệm</label>
                <textarea
                  rows={2}
                  value={expDesc}
                  onChange={(e) => setExpDesc(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-slate-300 text-slate-700 focus:outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="text-slate-600 font-semibold block mb-1">Tags (phân cách bằng dấu phẩy)</label>
                <input
                  type="text"
                  value={expTags}
                  onChange={(e) => setExpTags(e.target.value)}
                  className="w-full px-3 py-1.5 rounded-lg border border-slate-300 text-slate-700 focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
          </Card>
        </div>

        {/* RIGHT: Configuration Summary Sticky Panel (1 Column) */}
        <div className="sticky top-[74px] space-y-4">
          <Card
            title="Tóm tắt cấu hình"
            subtitle="Tổng hợp các tham số khởi tạo bài toán"
          >
            <div className="space-y-2 text-xs font-medium divide-y divide-slate-100 mb-4">
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Loại bài toán</span>
                <span className="font-semibold text-slate-800">
                  {PROBLEM_TYPES.find((t) => t.id === selectedTask)?.name}
                </span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Mô hình đã chọn</span>
                <span className="font-mono text-blue-600 font-bold">{currentModel.name}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Bộ dữ liệu</span>
                <span className="text-slate-800 font-semibold">{currentDataset.name}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Vị trí lưu trữ</span>
                <span className="text-slate-800 font-mono text-[11px]">{currentDataset.localPath || "Remote Standard"}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Tổng số mẫu</span>
                <span className="text-slate-800 font-mono">{currentDataset.samples?.toLocaleString()}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Số tham số model</span>
                <span className="text-blue-600 font-mono font-bold">{currentModel.params}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Độ phân giải</span>
                <span className="text-slate-800 font-mono">{currentModel.inputSize}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Thiết bị</span>
                <span className="text-slate-800">NVIDIA RTX 4090 ({gpuCount} GPU)</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Batch size</span>
                <span className="text-slate-800">{batchSize}</span>
              </div>
            </div>

            {/* Validation Badge */}
            <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 mb-4 flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
              <div className="text-xs">
                <div className="font-bold text-emerald-800">✓ Hợp lệ & Sẵn sàng</div>
                <p className="text-emerald-700 text-[11px] mt-0.5">
                  Đã tải file trọng số <strong>{currentModel.name}</strong> và dữ liệu <strong>{currentDataset.name}</strong>.
                </p>
              </div>
            </div>

            {/* Actions */}
            <div className="space-y-2">
              <Button
                variant="primary"
                className="w-full justify-between"
                onClick={handleNext}
              >
                <span>Tiếp tục cấu hình tấn công</span>
                <ArrowRight className="w-4 h-4" />
              </Button>
              <div className="grid grid-cols-2 gap-2">
                <Link href="/dashboard" className="block">
                  <Button variant="secondary" className="w-full text-xs" icon={ArrowLeft}>
                    Quay lại
                  </Button>
                </Link>
                <Button variant="outline" className="w-full text-xs" icon={Save}>
                  Lưu nháp
                </Button>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
