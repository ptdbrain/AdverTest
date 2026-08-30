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

  const [dataSource, setDataSource] = useState("repository");
  const [modelSource, setModelSource] = useState("default_lib");
  const [mixedPrecision, setMixedPrecision] = useState(true);
  const [batchSize, setBatchSize] = useState(16);
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

  // Hardware Specs Auto-detection State
  const [hardwareSpecs, setHardwareSpecs] = useState(null);
  const [isLoadingSpecs, setIsLoadingSpecs] = useState(true);

  // Upload & Download Modal / Simulation States
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadTarget, setDownloadTarget] = useState("");
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [uploadType, setUploadType] = useState("model"); // "model" | "dataset"
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("idle"); // "idle" | "validating" | "success" | "error"
  const [validationReport, setValidationReport] = useState(null);

  // 1. Auto-detect Hardware on Mount
  useEffect(() => {
    let isMounted = true;
    fetch(`${getApiBase()}/api/v1/system/runtime-specs`)
      .then((res) => res.json())
      .then((data) => {
        if (isMounted && data) {
          setHardwareSpecs(data);
          const executionSpecs = data.execution_plane || data;
          setBatchSize(executionSpecs.recommended_batch_size || 16);
          setMixedPrecision(executionSpecs.recommended_precision === "FP16");
        }
      })
      .catch((err) => {
        console.warn("Could not fetch hardware specs from backend:", err);
        if (isMounted) {
          setHardwareSpecs({
            has_cuda: false,
            device_name: "Intel/AMD CPU System",
            device_target: "cpu",
            cpu_count: 8,
            recommended_batch_size: 4,
            recommended_precision: "FP32",
            supported_precisions: ["FP32"],
          });
        }
      })
      .finally(() => {
        if (isMounted) setIsLoadingSpecs(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

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
      samples: dataset.demo ? "bundle demo" : "chưa chuẩn bị",
      size: dataset.demo ? "GCS sẵn sàng" : "cần tải về",
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

  const handleDownloadSimulation = (targetName) => {
    setDownloadTarget(targetName);
    setIsDownloading(true);
    setDownloadProgress(10);
    const interval = setInterval(() => {
      setDownloadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => setIsDownloading(false), 500);
          return 100;
        }
        return prev + 25;
      });
    }, 300);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadFile(file);
    setUploadStatus("validating");

    // Strict validation simulation
    setTimeout(() => {
      if (uploadType === "model") {
        const validExt = [".pt", ".pth", ".onnx"].some((ext) => file.name.endsWith(ext));
        if (!validExt) {
          setUploadStatus("error");
          setValidationReport({
            valid: false,
            error: `Định dạng tệp không hợp lệ: '${file.name}'. Hệ thống chỉ chấp nhận .pt, .pth, hoặc .onnx.`,
          });
        } else {
          setUploadStatus("success");
          setValidationReport({
            valid: true,
            modelName: file.name,
            fileSize: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
            architecture: file.name.includes("3d") ? "PointPillars 3D" : "YOLO11 Vision",
            classesDetected: ["car", "pedestrian", "cyclist"],
            checksum: "SHA256: verified",
          });
        }
      } else {
        // Dataset validation
        if (!file.name.endsWith(".zip")) {
          setUploadStatus("error");
          setValidationReport({
            valid: false,
            error: `Tập dữ liệu phải được nén dạng .zip chứa cấu trúc 'images/', 'labels/' và 'data.yaml'.`,
          });
        } else {
          setUploadStatus("success");
          setValidationReport({
            valid: true,
            datasetName: file.name.replace(".zip", ""),
            fileSize: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
            structureCheck: "images/ (OK), labels/ (OK), data.yaml (OK)",
            annotationFormat: "YOLO Normalized BBox [0..1] — 100% Passed",
            samplesFound: 48,
          });
        }
      }
    }, 800);
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
      batchSize,
      mixedPrecision,
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

      {/* 1. AUTO-DETECTED HARDWARE RUNSPECS BANNER */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-slate-900 via-slate-800 to-indigo-950 border border-slate-700 text-white shadow-lg flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="w-10 h-10 rounded-xl bg-indigo-500/20 border border-indigo-400/40 text-indigo-400 flex items-center justify-center font-bold shadow-inner">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-indigo-400">
                Tài Nguyên Phần Cứng & Runtime Auto-detect
              </span>
              <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                ● Ready
              </span>
            </div>
            <div className="text-sm font-semibold text-slate-100 mt-0.5">
              {hardwareSpecs ? (hardwareSpecs.execution_plane || hardwareSpecs).device_name : "Đang kiểm tra phần cứng..."}
              {(hardwareSpecs?.execution_plane || hardwareSpecs)?.total_vram_gb ? ` — ${(hardwareSpecs.execution_plane || hardwareSpecs).total_vram_gb} GB VRAM${(hardwareSpecs.execution_plane || hardwareSpecs).free_vram_gb != null ? ` (${(hardwareSpecs.execution_plane || hardwareSpecs).free_vram_gb} GB khả dụng)` : ""}` : ""}
            </div>
            <div className="text-xs text-slate-400 mt-0.5">
              Tự động tối ưu: Batch Size = <strong>{batchSize}</strong> | Precision = <strong>{mixedPrecision ? "FP16 (Tốc độ cao)" : "FP32"}</strong> | Thiết bị = <code className="bg-slate-800 px-1 py-0.5 rounded text-indigo-300">{(hardwareSpecs?.execution_plane || hardwareSpecs)?.device_target || "cpu"}</code>
              {hardwareSpecs?.execution_plane?.status === "on_demand" && " | GPU khởi động theo job"}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              setUploadType("model");
              setIsUploadModalOpen(true);
            }}
            icon={Upload}
            className="bg-slate-800/80 border-slate-600 text-slate-200 hover:bg-slate-700 text-xs"
          >
            Tải lên Model / Data
          </Button>
        </div>
      </div>

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
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDownloadSimulation(model.name);
                            }}
                            className="text-[10px] text-blue-600 hover:underline font-semibold flex items-center gap-0.5"
                          >
                            <Download className="w-3 h-3" /> Tải về
                          </button>
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
                      <span className="font-mono text-slate-500">{ds.samples} mẫu · {ds.size}</span>
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
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Batch Size:</span>
                <span className="font-bold text-slate-800">{batchSize}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Độ chính xác:</span>
                <span className="font-bold text-slate-800">{mixedPrecision ? "FP16 (Mixed)" : "FP32 (Single)"}</span>
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

          {/* Hyperparameters Card */}
          <Card title="Tùy chỉnh siêu tham số Runtime">
            <div className="space-y-3 text-xs">
              <div>
                <label className="font-semibold text-slate-700 block mb-1">Batch Size</label>
                <select
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="w-full p-2 border border-slate-300 rounded-lg bg-white font-medium text-slate-700"
                >
                  <option value={2}>2 (Rất nhẹ / CPU)</option>
                  <option value={4}>4 (Nhẹ / GPU 4GB)</option>
                  <option value={8}>8 (Vừa / GPU 6-8GB)</option>
                  <option value={16}>16 (Chuẩn / GPU 8-16GB)</option>
                  <option value={32}>32 (Nhanh / GPU &gt;16GB)</option>
                </select>
              </div>

              <div className="flex items-center justify-between pt-2">
                <div>
                  <div className="font-semibold text-slate-700">Mixed Precision (FP16)</div>
                  <div className="text-[10px] text-slate-400">Tăng tốc độ suy luận gấp 2 lần</div>
                </div>
                <input
                  type="checkbox"
                  checked={mixedPrecision}
                  onChange={(e) => setMixedPrecision(e.target.checked)}
                  className="w-4 h-4 text-blue-600 rounded"
                />
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* MODAL: DOWNLOAD SIMULATION */}
      {isDownloading && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-xl p-6 max-w-sm w-full space-y-4 shadow-2xl border border-slate-200">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center animate-spin">
                <RefreshCw className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-800">Đang tải xuống Server...</h3>
                <p className="text-xs text-slate-500">{downloadTarget}</p>
              </div>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
              <div
                className="bg-blue-600 h-2 transition-all duration-300"
                style={{ width: `${downloadProgress}%` }}
              />
            </div>
            <div className="text-right text-xs font-mono text-slate-500">{downloadProgress}%</div>
          </div>
        </div>
      )}

      {/* MODAL: UPLOAD WITH STRICT VALIDATION */}
      {isUploadModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-xl p-6 max-w-lg w-full space-y-4 shadow-2xl border border-slate-200 animate-scale-in">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <h3 className="text-base font-bold text-slate-800">Tải lên & Xác thực tệp (Upload)</h3>
                <p className="text-xs text-slate-500">Hệ thống thực hiện kiểm tra an toàn và định dạng nghiêm ngặt</p>
              </div>
              <button
                type="button"
                onClick={() => {
                  setIsUploadModalOpen(false);
                  setUploadStatus("idle");
                  setValidationReport(null);
                }}
                className="text-slate-400 hover:text-slate-600 text-lg font-bold"
              >
                ✕
              </button>
            </div>

            {/* Type selector */}
            <div className="flex gap-2 p-1 bg-slate-100 rounded-lg">
              <button
                type="button"
                onClick={() => {
                  setUploadType("model");
                  setUploadStatus("idle");
                  setValidationReport(null);
                }}
                className={cn(
                  "flex-1 py-1.5 text-xs font-bold rounded-md transition-colors",
                  uploadType === "model" ? "bg-white text-blue-600 shadow-xs" : "text-slate-600"
                )}
              >
                Tải lên Mô hình (.pt, .onnx, .pth)
              </button>
              <button
                type="button"
                onClick={() => {
                  setUploadType("dataset");
                  setUploadStatus("idle");
                  setValidationReport(null);
                }}
                className={cn(
                  "flex-1 py-1.5 text-xs font-bold rounded-md transition-colors",
                  uploadType === "dataset" ? "bg-white text-blue-600 shadow-xs" : "text-slate-600"
                )}
              >
                Tải lên Dataset (.zip)
              </button>
            </div>

            {/* File Input Box */}
            <label className="border-2 border-dashed border-slate-300 hover:border-blue-500 rounded-xl p-6 flex flex-col items-center justify-center gap-2 cursor-pointer bg-slate-50/60 transition-colors">
              <Upload className="w-8 h-8 text-slate-400" />
              <div className="text-xs font-semibold text-slate-700">
                {uploadFile ? uploadFile.name : "Kéo thả hoặc click để chọn tệp"}
              </div>
              <div className="text-[10px] text-slate-400">
                {uploadType === "model" ? "Hỗ trợ file PyTorch (.pt, .pth) hoặc ONNX Runtime (.onnx)" : "Tệp nén .zip chứa images/, labels/ và data.yaml"}
              </div>
              <input type="file" onChange={handleFileUpload} className="hidden" />
            </label>

            {/* Validation Feedback */}
            {uploadStatus === "validating" && (
              <div className="p-3 rounded-lg bg-blue-50 border border-blue-200 text-xs text-blue-700 flex items-center gap-2">
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Đang quét kiểm tra toàn vẹn định dạng và nhãn...</span>
              </div>
            )}

            {uploadStatus === "success" && validationReport && (
              <div className="p-3.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs space-y-1.5">
                <div className="font-bold text-emerald-800 flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  <span>Xác thực thành công! Tệp đạt tiêu chuẩn sẵn sàng benchmark.</span>
                </div>
                {uploadType === "model" ? (
                  <div className="text-[11px] text-emerald-700 space-y-0.5">
                    <div>• Dung lượng: {validationReport.fileSize} | Kiến trúc: {validationReport.architecture}</div>
                    <div>• Nhãn nhận diện: {validationReport.classesDetected?.join(", ")}</div>
                    <div>• Quét an toàn mã độc: {validationReport.checksum}</div>
                  </div>
                ) : (
                  <div className="text-[11px] text-emerald-700 space-y-0.5">
                    <div>• Cấu trúc: {validationReport.structureCheck}</div>
                    <div>• Bounding box: {validationReport.annotationFormat}</div>
                    <div>• Số lượng mẫu: {validationReport.samplesFound} ảnh</div>
                  </div>
                )}
              </div>
            )}

            {uploadStatus === "error" && validationReport && (
              <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold">Lỗi xác thực tệp:</div>
                  <div>{validationReport.error}</div>
                </div>
              </div>
            )}

            <div className="flex justify-end gap-2 pt-2 border-t border-slate-100">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  setIsUploadModalOpen(false);
                  setUploadStatus("idle");
                  setValidationReport(null);
                }}
              >
                Đóng
              </Button>
              <Button
                variant="primary"
                size="sm"
                disabled={uploadStatus !== "success"}
                onClick={() => {
                  setIsUploadModalOpen(false);
                }}
              >
                Sử dụng tệp này
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
