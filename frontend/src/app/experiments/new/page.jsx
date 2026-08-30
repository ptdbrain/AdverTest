"use client";

import React, { useState, useEffect, useMemo } from "react";
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
  Download,
  AlertCircle,
  HardDrive,
  Zap,
  RefreshCw,
  Sliders,
  CheckCheck,
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
} from "@/lib/constants";
import { cn } from "@/lib/utils";
import { getApiBase, getCatalogDatasets, getModelVersions } from "@/lib/api";
import ProjectAssetPicker from "@/components/ProjectAssetPicker";
import ClassMappingCard from "@/components/ClassMappingCard";

export default function ConfigureProblemPage() {
  const router = useRouter();
  const [selectedTask, setSelectedTask] = useState("detection2d");
  const [selectedArch, setSelectedArch] = useState("yolo");

  // Active selected dataset and model objects
  const [selectedDatasetId, setSelectedDatasetId] = useState("kitti_anonymized_de");
  const [selectedModelId, setSelectedModelId] = useState("local_yolo11s_clean");
  const [catalogDatasets, setCatalogDatasets] = useState([]);
  const [catalogModels, setCatalogModels] = useState([]);
  const [isLoadingCatalog, setIsLoadingCatalog] = useState(true);

  const [classMapping, setClassMapping] = useState({});
  const [expName, setExpName] = useState("EXP-2025-0512-001");
  const [sessionName, setSessionName] = useState("Đánh giá Robustness YOLO11s trên KITTI");
  const [expDesc, setExpDesc] = useState("Kiểm thử độ suy giảm của mô hình YOLO11s trước các điều kiện thời tiết khắc nghiệt và nhiễu đối kháng.");
  const [expTags, setExpTags] = useState("yolo11, kitti, object-detection, local");

  // Experiment Session Workspace Management State
  const [sessionMode, setSessionMode] = useState("new"); // "new" | "existing"
  const [existingSessions, setExistingSessions] = useState([]);
  const [selectedSessionId, setSelectedSessionId] = useState("");

  // Load existing sessions on mount
  useEffect(() => {
    fetch(`${getApiBase()}/api/v1/sessions`)
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setExistingSessions(data);
          setSelectedSessionId(data[0].id);
        }
      })
      .catch((err) => console.warn("Could not load sessions:", err));
  }, []);

  // When choosing an existing session, auto-fill its configuration
  const handleSelectExistingSession = (sessId) => {
    setSelectedSessionId(sessId);
    const found = existingSessions.find((s) => s.id === sessId);
    if (found) {
      setExpName(found.id);
      setSessionName(found.name);
      setExpDesc(found.description || "");
      if (found.task_id) setSelectedTask(found.task_id);
      if (found.model_id) setSelectedModelId(found.model_id);
      if (found.dataset_id) setSelectedDatasetId(found.dataset_id);
    }
  };


  // The experiment form must show the deployable catalog, not the old design
  // mock data.  Dataset `demo` means an anonymised bundle is mounted by the
  // worker; model `runnable` means its checkpoint has passed the run gate.
  useEffect(() => {
    let isMounted = true;
    setIsLoadingCatalog(true);
    Promise.allSettled([
      getModelVersions(),
      getCatalogDatasets({ task_id: selectedTask }),
    ]).then(([modelsResult, datasetsResult]) => {
      if (!isMounted) return;
      if (modelsResult.status === "fulfilled" && Array.isArray(modelsResult.value)) {
        setCatalogModels(modelsResult.value);
      }
      if (datasetsResult.status === "fulfilled" && Array.isArray(datasetsResult.value)) {
        setCatalogDatasets(datasetsResult.value);
      }
      if (modelsResult.status === "rejected" || datasetsResult.status === "rejected") {
        console.warn("Could not load the live model/dataset catalog; showing local fallback entries.");
      }
    }).finally(() => {
      if (isMounted) setIsLoadingCatalog(false);
    });
    return () => { isMounted = false; };
  }, [selectedTask]);

  // 2. Cascade Filter: Filter Models & Architectures based on Selected Task
  const fallbackModels = useMemo(() => AVAILABLE_MODELS.filter((m) => {
    if (selectedTask === "detection2d") return m.task === "detection2d";
    if (selectedTask === "segmentation") return m.task === "segmentation";
    if (selectedTask === "detection3d") return m.task === "detection3d";
    if (selectedTask === "classification") return m.task === "classification" || m.task === "detection2d";
    return true;
  }), [selectedTask]);

  const fallbackDatasets = useMemo(() => AVAILABLE_DATASETS.filter((d) => {
    if (selectedTask === "detection2d") return d.task === "detection2d";
    if (selectedTask === "segmentation") return d.task === "segmentation";
    if (selectedTask === "detection3d") return d.task === "detection3d";
    return true;
  }), [selectedTask]);

  const filteredModels = useMemo(() => {
    const liveModels = catalogModels
      .filter((model) => model.task === selectedTask && model.checkpoint_role === "base")
      .map((model) => ({
        id: model.id,
        name: model.model_name || model.id,
        family: model.model_family_id || "Catalog",
        task: model.task,
        params: model.checkpoint_hash ? "đã xác thực" : "catalog",
        format: model.checkpoint_path?.split(".").pop()?.toUpperCase() || "checkpoint",
        isLocal: Boolean(model.runnable),
        runnable: Boolean(model.runnable),
        reason: model.blocked_reason || (model.runnable ? null : "Checkpoint chưa sẵn sàng"),
        architecture: model.model_family_id || "catalog",
      }));
    return liveModels.length ? liveModels : fallbackModels;
  }, [catalogModels, fallbackModels]);

  const filteredDatasets = useMemo(() => {
    const liveDatasets = catalogDatasets.map((dataset) => ({
      id: dataset.name,
      name: dataset.title || dataset.name,
      task: dataset.task_id,
      description: `${dataset.modality} · ${dataset.ground_truth_status || "ground truth có sẵn"}`,
      sampleCount: dataset.sample_count,
      samples: dataset.sample_count != null ? `${dataset.sample_count} ảnh` : "chưa có ảnh",
      isLocal: Boolean(dataset.demo && dataset.anonymized),
      reason: dataset.demo && dataset.anonymized
        ? null
        : !dataset.anonymized
          ? "Chưa anonymize"
          : "Chưa có bundle trên worker",
      classLabels: Object.entries(dataset.class_map || {}).map(([id, name]) => ({ id, name })),
    }));
    return liveDatasets.length ? liveDatasets : fallbackDatasets;
  }, [catalogDatasets, fallbackDatasets]);

  // Keep selected model and dataset synchronized with task
  useEffect(() => {
    if (filteredModels.length > 0 && !filteredModels.some((m) => m.id === selectedModelId)) {
      setSelectedModelId(filteredModels[0].id);
      setSelectedArch(filteredModels[0].architecture || "yolo");
    }
    if (filteredDatasets.length > 0 && !filteredDatasets.some((d) => d.id === selectedDatasetId)) {
      setSelectedDatasetId(filteredDatasets[0].id);
    }
  }, [selectedTask, filteredModels, filteredDatasets, selectedModelId, selectedDatasetId]);

  const currentDataset = filteredDatasets.find((d) => d.id === selectedDatasetId) || filteredDatasets[0] || AVAILABLE_DATASETS[0];
  const currentModel = filteredModels.find((m) => m.id === selectedModelId) || filteredModels[0] || AVAILABLE_MODELS[0];
  const projectId = process.env.NEXT_PUBLIC_DEFAULT_PROJECT_ID || "advertest-default";

  // Derive Model Classes and Dataset Classes for Label Mapping Matrix
  const modelClasses = currentModel.architecture === "pointpillars"
    ? ["Car", "Pedestrian", "Cyclist"]
    : currentModel.architecture === "anonymization"
    ? ["face", "license_plate"]
    : currentModel.task === "segmentation"
    ? ["road", "sidewalk", "building", "person", "car", "truck"]
    : ["person", "car", "bus", "truck", "motorcycle", "bicycle", "traffic light", "stop sign"];

  const datasetClasses = (currentDataset?.classLabels || []).map((c) => c.name);

  // Compute Matched vs Unmapped Classes
  const matchedClasses = datasetClasses.filter((dc) =>
    modelClasses.some((mc) => mc.toLowerCase() === dc.toLowerCase())
  );
  const unmappedDatasetClasses = datasetClasses.filter(
    (dc) => !modelClasses.some((mc) => mc.toLowerCase() === dc.toLowerCase())
  );
  const unmappedModelClasses = modelClasses.filter(
    (mc) => !datasetClasses.some((dc) => dc.toLowerCase() === mc.toLowerCase())
  );
  const overlapPercentage = datasetClasses.length > 0
    ? Math.round((matchedClasses.length / datasetClasses.length) * 100)
    : 100;

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
    const taskName = PROBLEM_TYPES.find((t) => t.id === selectedTask)?.name || "Object Detection";
    const experimentConfig = {
      expName,
      sessionName: sessionName || `Phiên thử nghiệm ${expName}`,
      sessionDesc: expDesc,
      selectedTask,
      taskName,
      selectedArch,
      selectedModelId: currentModel.id,
      selectedModelName: currentModel.name,
      selectedModelFormat: currentModel.format,
      selectedDatasetId: currentDataset.id,
      selectedDatasetName: currentDataset.name,
      class_mapping: classMapping,
      updatedAt: new Date().toISOString(),
    };
    try {
      localStorage.setItem("adversai_active_experiment", JSON.stringify(experimentConfig));
      // Register or update session in backend API
      fetch(`${getApiBase()}/api/v1/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: expName,
          name: sessionName || `Phiên thử nghiệm ${expName}`,
          description: expDesc,
          task_id: selectedTask,
          task_name: taskName,
          model_id: currentModel.id,
          model_name: currentModel.name,
          dataset_id: currentDataset.id,
          dataset_name: currentDataset.name,
          class_mapping: classMapping,
          created_at: new Date().toLocaleDateString("vi-VN"),
          updated_at: new Date().toLocaleDateString("vi-VN"),
          runs: [],
        }),
      }).catch((e) => console.warn("Could not save session to backend:", e));
    } catch (e) {
      console.warn("Could not save to localStorage:", e);
    }
    router.push(`/experiments/${expName}/attack`);
  };

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Cấu hình bài toán & Mô hình"
        subtitle="Thiết lập bài toán, mô hình, đối chiếu nhãn lớp dataset và kiểm tra tài nguyên runtime trước khi tấn công đối kháng."
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Cấu hình bài toán", href: "/experiments/new" },
          { label: "Tạo thí nghiệm mới" },
        ]}
      />

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5 items-start">
        {/* LEFT & CENTER: Form Controls (2 Columns) */}
        <div className="xl:col-span-2 space-y-5">
          {/* 0. Quản lý Phiên làm việc & Cuộc thử nghiệm */}
          <Card
            title="0. Quản lý Phiên làm việc & Cuộc thử nghiệm (Experiment Session Workspace)"
            subtitle="Tạo phiên làm việc mới hoặc tiếp tục phiên đã có để lưu trữ và cộng dồn các lần chạy tấn công"
          >
            <div className="space-y-4">
              <div className="flex items-center gap-4 border-b border-slate-200 pb-3">
                <label className="flex items-center gap-2 text-xs font-bold text-slate-800 cursor-pointer">
                  <input
                    type="radio"
                    name="session_mode"
                    checked={sessionMode === "new"}
                    onChange={() => setSessionMode("new")}
                    className="text-blue-600 focus:ring-blue-500"
                  />
                  <span>Tạo Phiên Làm Việc Mới</span>
                </label>
                <label className="flex items-center gap-2 text-xs font-bold text-slate-800 cursor-pointer">
                  <input
                    type="radio"
                    name="session_mode"
                    checked={sessionMode === "existing"}
                    onChange={() => setSessionMode("existing")}
                    className="text-blue-600 focus:ring-blue-500"
                  />
                  <span>Tiếp Tục Trên Phiên Đã Có ({existingSessions.length} phiên)</span>
                </label>
              </div>

              {sessionMode === "existing" && existingSessions.length > 0 ? (
                <div className="space-y-2">
                  <label className="text-xs font-semibold text-slate-700">Chọn Phiên Đã Lưu Để Tiếp Tục Thử Nghiệm:</label>
                  <select
                    value={selectedSessionId}
                    onChange={(e) => handleSelectExistingSession(e.target.value)}
                    className="w-full text-xs py-2 px-3 rounded-lg border border-slate-300 bg-white font-medium text-slate-800 shadow-xs focus:ring-2 focus:ring-blue-500"
                  >
                    {existingSessions.map((sess) => (
                      <option key={sess.id} value={sess.id}>
                        {sess.name} ({sess.id}) — {sess.runs?.length || 0} bài test đã chạy [{sess.model_name}]
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  <div className="space-y-1">
                    <label className="font-semibold text-slate-700">Tên Cuộc Thử Nghiệm / Phiên Làm Việc:</label>
                    <input
                      type="text"
                      value={sessionName}
                      onChange={(e) => setSessionName(e.target.value)}
                      placeholder="Ví dụ: Đánh giá Robustness YOLO11s trên KITTI Mùa 1"
                      className="w-full py-1.5 px-3 rounded-lg border border-slate-300 bg-white font-medium text-slate-800 shadow-xs focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="font-semibold text-slate-700">Mã Định Danh Phiên (Session ID):</label>
                    <input
                      type="text"
                      value={expName}
                      onChange={(e) => setExpName(e.target.value)}
                      className="w-full py-1.5 px-3 rounded-lg border border-slate-300 bg-slate-50 font-mono font-bold text-blue-700 shadow-xs"
                    />
                  </div>
                  <div className="md:col-span-2 space-y-1">
                    <label className="font-semibold text-slate-700">Mục Tiêu & Ghi Chú Thử Nghiệm:</label>
                    <input
                      type="text"
                      value={expDesc}
                      onChange={(e) => setExpDesc(e.target.value)}
                      placeholder="Mô tả mục đích kiểm thử và các kịch bản dự kiến..."
                      className="w-full py-1.5 px-3 rounded-lg border border-slate-300 bg-white text-slate-700 shadow-xs"
                    />
                  </div>
                </div>
              )}
            </div>
          </Card>
          <Card
            title="1. Chọn loại bài toán"
            subtitle="Xác định nhiệm vụ thị giác máy tính cần đánh giá kiểm thử"
          >
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {PROBLEM_TYPES.filter((t) => ["detection2d", "segmentation", "detection3d", "classification"].includes(t.id)).map((task) => {
                const Icon = getTaskIcon(task.id);
                const isSelected = selectedTask === task.id;
                return (
                  <div
                    key={task.id}
                    onClick={() => {
                      setSelectedTask(task.id);
                      if (task.id === "detection3d") {
                        setSelectedArch("pointpillars");
                      }
                    }}
                    className={cn(
                      "p-3 rounded-lg border text-left cursor-pointer transition-all relative flex flex-col justify-between h-[104px]",
                      isSelected
                        ? "border-blue-600 bg-blue-50/70 shadow-sm ring-2 ring-blue-500/20"
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

          {/* 2. Chọn Mô hình (Cascade Filtered by Task) */}
          <Card
            title={`2. Chọn mô hình tương thích với [${PROBLEM_TYPES.find((t) => t.id === selectedTask)?.name}]`}
            subtitle={isLoadingCatalog ? "Đang kiểm tra checkpoint trên hệ thống..." : "Chỉ hiển thị checkpoint base tương thích; trạng thái phản ánh khả năng chạy thực tế."}
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
              {filteredModels.map((model) => {
                const isSelected = selectedModelId === model.id;
                return (
                  <div
                    key={model.id}
                    onClick={() => {
                      setSelectedModelId(model.id);
                      setSelectedArch(model.architecture || "yolo");
                    }}
                    className={cn(
                      "p-3 rounded-lg border text-left cursor-pointer transition-all space-y-2 relative flex flex-col justify-between",
                      isSelected
                        ? "border-blue-600 bg-blue-50/50 shadow-sm ring-1 ring-blue-500"
                        : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                    )}
                  >
                    <div>
                      <div className="flex items-start justify-between gap-1 mb-1">
                        <span className="text-xs font-bold text-slate-800 line-clamp-1">
                          {model.name}
                        </span>
                        {isSelected && (
                          <span className="w-4 h-4 rounded-full bg-blue-600 text-white flex items-center justify-center flex-shrink-0">
                            <Check className="w-2.5 h-2.5" />
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5 mb-2">
                        <span className="text-[10px] font-semibold text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded">
                          {model.family}
                        </span>
                        <span className="text-[10px] font-mono text-slate-400">{model.params} params</span>
                      </div>
                    </div>

                    <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[11px]">
                      {model.isLocal ? (
                        <span className="inline-flex items-center gap-1 text-emerald-700 font-semibold bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                          ✓ Sẵn sàng
                        </span>
                      ) : (
                        <div className="flex items-center gap-1" title={model.reason || "Checkpoint chưa sẵn sàng"}>
                          <span className="text-amber-700 text-[10px] font-semibold">Cần tải / xác thực</span>
                          <span className="text-[10px] text-slate-500 font-semibold">Artifact chưa sẵn sàng để tải</span>
                        </div>
                      )}
                      <span className="font-mono text-slate-500">{model.format}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* 3. Kho Dữ Liệu (Catalog Status & Selection) */}
          <Card
            title="3. Chọn tập dữ liệu kiểm thử (Dataset)"
            subtitle={isLoadingCatalog ? "Đang kiểm tra bundle anonymized trên worker..." : "✓ Sẵn sàng = bundle anonymized đã có trên GCS/worker; còn lại cần chuẩn bị trước khi chạy."}
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {filteredDatasets.map((ds) => {
                const isSelected = selectedDatasetId === ds.id;
                return (
                  <div
                    key={ds.id}
                    onClick={() => setSelectedDatasetId(ds.id)}
                    className={cn(
                      "p-3 rounded-lg border text-left cursor-pointer transition-all space-y-2 flex flex-col justify-between",
                      isSelected
                        ? "border-blue-600 bg-blue-50/50 shadow-sm ring-1 ring-blue-500"
                        : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                    )}
                  >
                    <div>
                      <div className="flex items-start justify-between gap-1 mb-1">
                        <span className="text-xs font-bold text-slate-800">{ds.name}</span>
                        {isSelected && (
                          <span className="w-4 h-4 rounded-full bg-blue-600 text-white flex items-center justify-center flex-shrink-0">
                            <Check className="w-2.5 h-2.5" />
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-500 line-clamp-2">{ds.description}</p>
                    </div>

                    <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[11px]">
                      <div className="flex items-center gap-1.5">
                        {ds.isLocal ? (
                          <span className="inline-flex items-center gap-1 text-emerald-700 font-semibold bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                            ✓ Sẵn sàng
                          </span>
                        ) : (
                          <span title={ds.reason || "Dataset chưa sẵn sàng"} className="inline-flex items-center gap-1 text-amber-700 font-semibold bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                            ⚠ {ds.reason || "Cần tải về"}
                          </span>
                        )}
                      </div>
                      <span className="font-mono text-slate-500">{ds.sampleCount != null ? `${ds.sampleCount} ảnh` : ds.samples}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* 4. CLASS LABELS & LABEL MAPPING MATRIX */}
          <Card
            title="4. Đối chiếu nhãn lớp (Class Labels & Mapping Matrix)"
            subtitle="Kiểm tra mức độ trùng khớp giữa nhãn mô hình đã học và nhãn của bộ dữ liệu"
          >
            <ClassMappingCard
              modelClasses={modelClasses}
              datasetClasses={datasetClasses}
              source={catalogDatasets.length > 0 ? "manifest" : "demo"}
              onChange={setClassMapping}
            />
            <div className="space-y-4">
              {/* Header Match Rate */}
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-between">
                <div>
                  <div className="text-xs font-bold text-slate-800">
                    Tỷ lệ khớp nhãn (Label Overlap): <span className="text-emerald-600 font-mono">{overlapPercentage}%</span>
                  </div>
                  <div className="text-[11px] text-slate-500">
                    {matchedClasses.length}/{datasetClasses.length || 1} lớp của Dataset được nhận diện bởi mô hình.
                  </div>
                </div>
                <Badge variant={overlapPercentage >= 80 ? "success" : "warning"}>
                  {overlapPercentage >= 80 ? "✓ Tương thích cao" : "⚠️ Cần ánh xạ"}
                </Badge>
              </div>

              {/* Two Column Class Badges */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                {/* Model Classes */}
                <div className="p-3 rounded-lg border border-slate-200 bg-white">
                  <div className="font-bold text-slate-700 mb-2 flex items-center justify-between">
                    <span>Nhãn mô hình đã học ({modelClasses.length}):</span>
                    <span className="text-[10px] text-slate-400 font-mono">Model Weights</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto">
                    {modelClasses.map((cls) => {
                      const isMatched = datasetClasses.some((dc) => dc.toLowerCase() === cls.toLowerCase());
                      return (
                        <span
                          key={cls}
                          className={cn(
                            "px-2 py-0.5 rounded text-[11px] font-medium border",
                            isMatched
                              ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                              : "bg-slate-100 text-slate-600 border-slate-200"
                          )}
                        >
                          {cls}
                        </span>
                      );
                    })}
                  </div>
                </div>

                {/* Dataset Classes */}
                <div className="p-3 rounded-lg border border-slate-200 bg-white">
                  <div className="font-bold text-slate-700 mb-2 flex items-center justify-between">
                    <span>Nhãn trong Dataset ({datasetClasses.length}):</span>
                    <span className="text-[10px] text-slate-400 font-mono">Ground Truth</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto">
                    {datasetClasses.map((cls) => {
                      const isMatched = modelClasses.some((mc) => mc.toLowerCase() === cls.toLowerCase());
                      return (
                        <span
                          key={cls}
                          className={cn(
                            "px-2 py-0.5 rounded text-[11px] font-medium border",
                            isMatched
                              ? "bg-emerald-50 text-emerald-700 border-emerald-200 font-semibold"
                              : "bg-amber-50 text-amber-700 border-amber-200 font-semibold"
                          )}
                        >
                          {cls} {!isMatched && "(Bỏ qua)"}
                        </span>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* RIGHT COLUMN: Experiment Summary & Hyperparameters (1 Column) */}
        <div className="space-y-5">
          <Card title="Asset của project" subtitle={`Project: ${projectId}`}>
            <div className="space-y-4">
              <ProjectAssetPicker projectId={projectId} taskId={selectedTask} kind="model" modelFamilyId={currentModel.architecture} onComplete={() => getModelVersions().then(setCatalogModels).catch(() => {})} />
              <ProjectAssetPicker projectId={projectId} taskId={selectedTask} kind="dataset" onComplete={() => getCatalogDatasets({ task_id: selectedTask }).then(setCatalogDatasets).catch(() => {})} />
              <p className="text-[11px] text-slate-500">Upload dùng API project-scoped; server quyết định validation và trạng thái READY.</p>
            </div>
          </Card>
          {/* Summary Card */}
          <Card title="Tóm tắt cấu hình bài toán">
            <div className="space-y-3 text-xs">
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Bài toán:</span>
                <span className="font-bold text-slate-800">{PROBLEM_TYPES.find((t) => t.id === selectedTask)?.name}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Mô hình:</span>
                <span className="font-bold text-blue-600 line-clamp-1 max-w-[160px]">{currentModel.name}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Dữ liệu:</span>
                <span className="font-bold text-slate-800 line-clamp-1 max-w-[160px]">{currentDataset.name}</span>
              </div>
            </div>

            <div className="mt-5 space-y-2">
              <Button
                variant="primary"
                className="w-full justify-center text-sm py-2.5 shadow-sm"
                onClick={handleNext}
                icon={ArrowRight}
              >
                Tiếp tục: Cấu hình tấn công
              </Button>
            </div>
          </Card>
        </div>
      </div>
      {/* Uploads are handled by the project asset pickers above. */}
    </div>
  );
}
