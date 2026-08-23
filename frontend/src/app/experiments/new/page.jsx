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
} from "lucide-react";
import PageHeader from "@/components/layout/PageHeader";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import {
  PROBLEM_TYPES,
  MODEL_ARCHITECTURES,
  CLASS_LABELS_DATA,
} from "@/data/mockData";
import { cn } from "@/lib/utils";

export default function ConfigureProblemPage() {
  const router = useRouter();
  const [selectedTask, setSelectedTask] = useState("detection2d");
  const [selectedArch, setSelectedArch] = useState("yolo");
  const [selectedModelVariant, setSelectedModelVariant] = useState("YOLOv8n");
  const [dataSource, setDataSource] = useState("user_upload");
  const [modelSource, setModelSource] = useState("user_upload");
  const [mixedPrecision, setMixedPrecision] = useState(true);
  const [batchSize, setBatchSize] = useState(16);
  const [gpuCount, setGpuCount] = useState(1);
  const [expName, setExpName] = useState("EXP-2025-05-12-001");
  const [expDesc, setExpDesc] = useState("Đánh giá khả năng phát hiện đối tượng của YOLOv8n trên bộ dữ liệu giao thông đô thị.");
  const [expTags, setExpTags] = useState("yolov8, object-detection, traffic");

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
        subtitle="Thiết lập bài toán, mô hình, dữ liệu và các tùy chọn để bắt đầu thí nghiệm đánh giá & phòng thủ AI đối kháng."
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Cấu hình bài toán", href: "/experiments/new" },
          { label: "Tạo thí nghiệm mới" },
        ]}
      />

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
                    onClick={() => setSelectedTask(task.id)}
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

          {/* 2. Chọn mô hình / Kiến trúc */}
          <Card
            title="2. Chọn mô hình / Kiến trúc"
            subtitle="Chọn họ kiến trúc deep learning và phiên bản checkpoints cần kiểm thử"
          >
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {MODEL_ARCHITECTURES.map((arch) => {
                const isSelected = selectedArch === arch.id;
                return (
                  <div
                    key={arch.id}
                    onClick={() => {
                      setSelectedArch(arch.id);
                      setSelectedModelVariant(arch.defaultModel);
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
                      value={isSelected ? selectedModelVariant : arch.defaultModel}
                      onChange={(e) => {
                        e.stopPropagation();
                        setSelectedArch(arch.id);
                        setSelectedModelVariant(e.target.value);
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
                    <div className="font-semibold text-slate-800">Tải lên bộ dữ liệu người dùng</div>
                    <div className="text-[11px] text-slate-500">Tải lên ảnh, nhãn YOLO/COCO và file cấu hình</div>
                    <Button variant="secondary" size="sm" icon={Upload} className="mt-2 text-xs">
                      Tải lên dữ liệu
                    </Button>
                  </div>
                </label>

                <label
                  className={cn(
                    "flex items-center gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors",
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
                    className="text-blue-600"
                  />
                  <div>
                    <div className="font-semibold text-slate-800">Kho dữ liệu (Repository)</div>
                    <div className="text-[11px] text-slate-500">traffic_dataset_v1 (25,146 ảnh)</div>
                  </div>
                </label>

                <label
                  className={cn(
                    "flex items-center gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors",
                    dataSource === "sample"
                      ? "border-blue-500 bg-blue-50/40"
                      : "border-slate-200 hover:bg-slate-50"
                  )}
                >
                  <input
                    type="radio"
                    name="data_src"
                    checked={dataSource === "sample"}
                    onChange={() => setDataSource("sample")}
                    className="text-blue-600"
                  />
                  <div>
                    <div className="font-semibold text-slate-800">Dataset mẫu (Sample VOC/COCO)</div>
                    <div className="text-[11px] text-slate-500">50 ảnh benchmark chuẩn</div>
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
                    <div className="font-semibold text-slate-800">Tải lên mô hình người dùng</div>
                    <div className="text-[11px] text-slate-500">Hỗ trợ .pt, .pth, .onnx, .engine</div>
                    <Button variant="secondary" size="sm" icon={Upload} className="mt-2 text-xs">
                      Tải lên mô hình
                    </Button>
                  </div>
                </label>

                <label
                  className={cn(
                    "flex items-center gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors",
                    modelSource === "trained"
                      ? "border-blue-500 bg-blue-50/40"
                      : "border-slate-200 hover:bg-slate-50"
                  )}
                >
                  <input
                    type="radio"
                    name="model_src"
                    checked={modelSource === "trained"}
                    onChange={() => setModelSource("trained")}
                    className="text-blue-600"
                  />
                  <div>
                    <div className="font-semibold text-slate-800">Chọn mô hình đã huấn luyện</div>
                    <div className="text-[11px] text-slate-500">yolov8n_custom.pt (v8.0.226)</div>
                  </div>
                </label>

                <label
                  className={cn(
                    "flex items-center gap-2.5 p-2.5 rounded-lg border cursor-pointer transition-colors",
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
                    className="text-blue-600"
                  />
                  <div>
                    <div className="font-semibold text-slate-800">Thư viện mô hình mặc định</div>
                    <div className="text-[11px] text-slate-500">YOLOv8 COCO Pretrained</div>
                  </div>
                </label>
              </div>
            </Card>
          </div>

          {/* 5 & 6. Data Preview & Class Labels */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 5. Xem trước dữ liệu */}
            <Card title="5. Xem trước dữ liệu" subtitle="Ảnh mẫu trong tập dữ liệu giao thông">
              <div className="grid grid-cols-3 gap-2">
                {[1, 2, 3, 4, 5].map((idx) => (
                  <div
                    key={idx}
                    className="h-20 rounded bg-slate-200 border border-slate-300 relative overflow-hidden flex items-center justify-center text-slate-400 font-mono text-[10px]"
                  >
                    <span className="p-1 text-center">sample_{idx}.jpg</span>
                  </div>
                ))}
                <div className="h-20 rounded bg-blue-900/80 border border-blue-700 text-white font-bold text-xs flex items-center justify-center p-2 text-center">
                  + 12,345 ảnh
                </div>
              </div>
            </Card>

            {/* 6. Nhãn lớp (Class labels) */}
            <Card
              title="6. Nhãn lớp (Class labels)"
              subtitle="Tổng số lớp: 8 lớp đối tượng"
              headerAction={
                <button type="button" className="text-xs text-blue-600 hover:underline">
                  Xem tất cả &gt;
                </button>
              }
            >
              <div className="max-h-36 overflow-y-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500 font-semibold">
                      <th className="py-1 px-2">ID</th>
                      <th className="py-1 px-2">Tên lớp</th>
                      <th className="py-1 px-2 text-right">Số lượng</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                    {CLASS_LABELS_DATA.map((c) => (
                      <tr key={c.id} className="hover:bg-slate-50">
                        <td className="py-1 px-2 font-mono text-slate-400">{c.id}</td>
                        <td className="py-1 px-2 font-semibold text-slate-800">{c.name}</td>
                        <td className="py-1 px-2 text-right font-mono">{c.count.toLocaleString()}</td>
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
                  <span className="font-semibold text-slate-800 font-mono">{selectedModelVariant}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Phiên bản</span>
                  <span className="text-slate-800">8.0.226</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Định dạng</span>
                  <span className="font-mono text-slate-800">.pt (PyTorch)</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Số tham số</span>
                  <span className="text-slate-800">3.2M</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Kích thước file</span>
                  <span className="text-slate-800">6.4 MB</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Độ phân giải đầu vào</span>
                  <span className="text-slate-800 font-mono">640 × 640</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-slate-500">Số lớp</span>
                  <span className="text-slate-800">8</span>
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
                <span className="text-slate-500">Kiến trúc mô hình</span>
                <span className="font-mono text-blue-600 font-bold">{selectedModelVariant}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Nguồn dữ liệu</span>
                <span className="text-slate-800">traffic_dataset_v1</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Nguồn mô hình</span>
                <span className="text-slate-800 font-mono">yolov8n_custom.pt</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Số lớp</span>
                <span className="text-slate-800">8 lớp</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Tổng số ảnh</span>
                <span className="text-slate-800 font-mono">25,146</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-500">Độ phân giải</span>
                <span className="text-slate-800 font-mono">640 × 640</span>
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
                <div className="font-bold text-emerald-800">✓ Hợp lệ</div>
                <p className="text-emerald-700 text-[11px] mt-0.5">
                  Tất cả cài đặt đã sẵn sàng để tiếp tục sang cấu hình tấn công.
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
