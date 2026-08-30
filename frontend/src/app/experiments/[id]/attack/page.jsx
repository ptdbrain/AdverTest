"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  Crosshair,
  Search,
  Sliders,
  Play,
  CheckCircle2,
  Trash2,
  Plus,
  Sparkles,
  ArrowRight,
  ArrowLeft,
  CloudRain,
  CloudFog,
  Snowflake,
  Sun,
  Camera,
  Layers,
  Zap,
  Check,
  Eye,
  SlidersHorizontal,
  Info,
  ShieldAlert,
  Database,
  Box,
  Target,
  RefreshCw,
  Cpu,
  Link2,
} from "lucide-react";
import PageHeader from "@/components/layout/PageHeader";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import { ATTACK_CATEGORIES, ATTACK_PRESETS } from "@/lib/constants";
import { useSearchParams } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  createRun,
  getCatalogAttacks,
  getCatalogDatasets,
  getModelVersions,
  getRun,
  getRunReport,
  getRunSamples,
  preflightRun,
  addRunToSession,
} from "@/lib/api";

const TERMINAL_RUN_STATES = new Set(["COMPLETED", "FAILED", "CANCELLED"]);
const ATTACK_ID_ALIASES = {
  low_light: "brightness",
  lidar_jitter: "lidar_xyz_noise",
  cw: "cw_l2",
};

export function normalizeAttackQueue(queue) {
  return queue.map((attack) => ({
    ...attack,
    id: ATTACK_ID_ALIASES[attack.id] || attack.id,
  }));
}

function basename(value) {
  return String(value || "").split(/[\\/]/).pop()?.toLowerCase() || "";
}

function resolveBackendModel(context, versions) {
  const requestedFile = basename(String(context.selectedModelName || "").split(" (")[0]);
  const requestedTokens = requestedFile
    .replace(/\.[^.]+$/, "")
    .split(/[-_ ]+/)
    .filter((token) => token.length > 1 && token !== "best");
  
  const baseVersions = versions.filter(
    (version) => version.runnable && version.checkpoint_role === "base" && (!context.selectedTask || version.task === context.selectedTask),
  );
  const taskVersions = versions.filter(
    (version) => version.runnable && (!context.selectedTask || version.task === context.selectedTask),
  );

  return baseVersions.find((version) => version.id === context.selectedModelId)
    || baseVersions.find((version) => basename(version.checkpoint_path) === requestedFile)
    || baseVersions.find((version) => requestedFile && basename(version.checkpoint_path).includes(requestedFile))
    || baseVersions.find((version) => {
      const modelName = String(version.model_name || "").toLowerCase();
      return modelName.length > 1 && requestedFile.includes(modelName);
    })
    || baseVersions.find((version) => {
      const identity = `${version.id} ${version.model_name} ${version.checkpoint_path || ""}`.toLowerCase();
      return requestedTokens.length > 0 && requestedTokens.every((token) => identity.includes(token));
    })
    || baseVersions[0]
    || taskVersions.find((version) => version.checkpoint_role === "base")
    || taskVersions[0]
    || versions.find((version) => version.runnable)
    || versions[0];
}

function resolveBackendDataset(context, datasets) {
  const requestedName = String(context.selectedDatasetName || "").toLowerCase();
  return datasets.find((dataset) => dataset.name === context.selectedDatasetId)
    || datasets.find((dataset) => dataset.title === context.selectedDatasetName)
    || datasets.find((dataset) => requestedName && requestedName.includes(String(dataset.name || "").toLowerCase()))
    || datasets.find((dataset) => dataset.anonymized)
    || datasets[0];
}

function buildRealRunConfig(context, attackQueue, model, dataset, attackMode = "combined") {
  const isCombined = attackMode === "combined";
  // Each dataset uses a strict parameter schema.  Start with the parameters
  // advertised by the backend and add KITTI-only controls only for KITTI.
  const datasetParams = { ...(dataset.dataset_params || {}) };
  if (dataset.name === "kitti") {
    datasetParams.split = datasetParams.split || "val";
    datasetParams.difficulty = datasetParams.difficulty || "all";
    datasetParams.merge_van_truck = true;
  }
  if (isCombined) {
    return {
      checkpoint_id: model.id,
      model_family_id: model.model_family_id,
      task_id: context.selectedTask,
      dataset: dataset.name,
      dataset_params: datasetParams,
      recipe: {
        name: `recipe-${attackQueue.map((a) => a.id).join("-")}`,
        steps: attackQueue.map((attack, index) => ({
          position: index,
          attack_name: attack.id,
          implementation_version: (
            { cw_l2: "2.0.0", dpatch: "2.0.0", thys_patch: "2.0.0" }[attack.id] || "1.0.0"
          ),
          severity: Number(attack.severity) || 3,
          parameters: {},
          seed: 42 + index,
          expected_cost: 1.0,
        })),
      },
      limit: 8,
      seed: 42,
      iou_threshold: 0.5,
      confidence_threshold: 0.25,
    };
  }

  return {
    checkpoint_id: model.id,
    model_family_id: model.model_family_id,
    task_id: context.selectedTask,
    dataset: dataset.name,
    dataset_params: datasetParams,
    attacks: attackQueue.map((attack) => attack.id),
    severities: Array.from(new Set(attackQueue.map((attack) => Number(attack.severity)))),
    limit: 8,
    seed: 42,
    iou_threshold: 0.5,
    confidence_threshold: 0.25,
  };
}

async function waitForRealRun(runId, onProgress) {
  while (true) {
    const job = await getRun(runId);
    onProgress(job);
    if (TERMINAL_RUN_STATES.has(job.status)) {
      if (job.status !== "COMPLETED") throw new Error(job.error || `Run kết thúc với trạng thái ${job.status}.`);
      const report = await getRunReport(runId);
      const samples = await getRunSamples(runId);
      if (!report || (!samples.length && !report.sample_results?.length)) {
        throw new Error("Backend hoàn tất run nhưng không trả về report/sample evidence.");
      }
      return { job, report, samples: samples.length ? samples : report.sample_results };
    }
    await new Promise((resolve) => window.setTimeout(resolve, 750));
  }
}

function executionStepForJob(job) {
  const stage = job?.detail?.stage || job?.status;
  if (stage === "COMPLETED") return 5;
  if (["EVALUATING", "COMPUTING_METRICS"].includes(stage)) return 4;
  if (stage === "INFERENCING") return 3;
  // The remote worker may report PREPARING, GPU_STARTING, GENERATING, or its
  // durable RUNNING state while it materialises data and synthesises variants.
  return 2;
}

function executionStatusMessageForJob(job) {
  const stage = job?.detail?.stage || job?.status;
  if (stage === "GPU_STARTING") {
    return "GPU Cloud Run đang khởi động theo yêu cầu. Đây không phải lỗi; job sẽ tự chạy khi worker sẵn sàng.";
  }
  if (stage === "PREPARING") {
    return "GPU đã sẵn sàng, đang nạp checkpoint và dữ liệu đã ẩn danh từ GCS.";
  }
  if (stage === "GENERATING") return "Đang sinh biến thể nhiễu đối kháng trên GPU.";
  if (stage === "INFERENCING") return "Đang suy luận clean và attacked trên GPU.";
  if (["EVALUATING", "COMPUTING_METRICS"].includes(stage)) return "Đang tính metric và chuẩn bị evidence.";
  return "Đang gửi job tới GPU worker.";
}

function ConfigureAttackPageContent() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const mode = searchParams?.get("mode"); // "reset" | "extend" | null
  const expId = params?.id || "EXP-2025-0512-001";

  // Experiment Context from Step 1
  const [expContext, setExpContext] = useState({
    expName: expId,
    selectedTask: "detection2d",
    taskName: "Object Detection",
    selectedModelId: "local_yolo11s_clean",
    selectedModelName: "yolo11s-clean-b0_best.pt (Local Workspace)",
    selectedDatasetId: "kitti_anonymized_de",
    selectedDatasetName: "KITTI De-anonymized Set (Local Workspace)",
    batchSize: 16,
    mixedPrecision: true,
  });

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedAttackId, setSelectedAttackId] = useState("depth_fog");
  const [severityLevel, setSeverityLevel] = useState(3);
  const [attackMode, setAttackMode] = useState("combined"); // "combined" | "individual"
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionStep, setExecutionStep] = useState(0);
  const [executionStatusMessage, setExecutionStatusMessage] = useState("");
  const [executionError, setExecutionError] = useState("");

  // Attack Recipe Summary (The active queue of confirmed attacks)
  const [attackQueue, setAttackQueue] = useState([
    {
      id: "depth_fog",
      name: "Depth Fog (Sương mù)",
      severity: 3,
      eps: "Severity 3",
      norm: "Natural",
      desc: "Sương mù 3D tán xạ quang học làm mờ đối tượng ở xa.",
    },
    {
      id: "pgd",
      name: "PGD (Projected Gradient)",
      severity: 3,
      eps: "8/255",
      norm: "Linf",
      desc: "Nhiễu gradient đối kháng lặp 20 bước kiểm thử điểm yếu.",
    },
  ]);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("adversai_active_experiment");
      if (saved) {
        const parsed = JSON.parse(saved);
        window.setTimeout(() => {
          setExpContext((prev) => ({ ...prev, ...parsed }));
          if (mode === "reset") {
            if (parsed.selectedTask === "detection3d") {
              setSelectedAttackId("lidar_beam_drop");
              setAttackQueue([{ id: "lidar_beam_drop", name: "Beam Drop (LiDAR 3D)", severity: 3, eps: "Drop 30%", norm: "Spatial 3D", desc: "Mất chùm tia cảm biến LiDAR mô phỏng bụi bám và thời tiết." }]);
            } else {
              setSelectedAttackId("depth_rain");
              setAttackQueue([{ id: "depth_rain", name: "Depth Rain (Mưa giông)", severity: 3, eps: "Severity 3", norm: "Weather", desc: "Mưa giông tán xạ quang học làm suy giảm tương phản." }]);
            }
          } else if (parsed.selectedTask === "detection3d") {
            setSelectedAttackId("lidar_beam_drop");
            setAttackQueue([
              { id: "lidar_beam_drop", name: "Beam Drop (LiDAR 3D)", severity: 3, eps: "Drop 30%", norm: "Spatial 3D", desc: "Mất chùm tia cảm biến LiDAR mô phỏng bụi bám và thời tiết." },
              { id: "lidar_jitter", name: "Point Jitter (LiDAR 3D)", severity: 3, eps: "σ=0.05m", norm: "Gaussian 3D", desc: "Nhiễu rung tọa độ 3D làm biến dạng voxels." },
            ]);
          }
        }, 0);
      }
    } catch {}
  }, [mode]);

  const allAttacksList = ATTACK_CATEGORIES.flatMap((c) => c.attacks);
  const currentAttack =
    allAttacksList.find((a) => a.id === selectedAttackId) || allAttacksList[0];

  // 1. Preset Selector: Instantly populate queue
  const handleApplyPreset = (preset) => {
    setAttackQueue(
      preset.attacks.map((a) => {
        const found = allAttacksList.find((x) => x.id === a.id);
        return {
          id: a.id,
          name: a.name || found?.name || a.id,
          severity: a.severity || 3,
          eps: a.eps || `Sev-${a.severity || 3}`,
          norm: found?.norm || "Linf",
          desc: found?.desc || "Đòn tấn công đối kháng mô phỏng.",
        };
      })
    );
  };

  const handleAddCurrentAttackToQueue = () => {
    const existingIndex = attackQueue.findIndex((a) => a.id === currentAttack.id);
    const newEntry = {
      id: currentAttack.id,
      name: currentAttack.name,
      severity: severityLevel,
      eps: currentAttack.isWeather ? `Severity ${severityLevel}` : `${severityLevel * 2}/255`,
      norm: currentAttack.norm,
      desc: currentAttack.desc,
    };

    if (existingIndex >= 0) {
      const updated = [...attackQueue];
      updated[existingIndex] = newEntry;
      setAttackQueue(updated);
    } else {
      setAttackQueue([...attackQueue, newEntry]);
    }
  };

  const handleRemoveQueueItem = (id) => {
    if (attackQueue.length <= 1) return;
    setAttackQueue(attackQueue.filter((a) => a.id !== id));
  };

  const handleStartExecution = async () => {
    setIsExecuting(true);
    setExecutionStep(1);
    setExecutionStatusMessage("Đang kiểm tra cấu hình và gửi job tới GPU worker...");
    setExecutionError("");
    try {
      const [versions, datasets] = await Promise.all([
        getModelVersions(),
        getCatalogDatasets({ task_id: expContext.selectedTask }),
      ]);
      const model = resolveBackendModel(expContext, versions);
      const dataset = resolveBackendDataset(expContext, datasets);
      if (!model) throw new Error("Mô hình đã chọn chưa có base checkpoint hợp lệ để chạy attack.");
      if (!dataset) throw new Error("Bộ dữ liệu đã chọn chưa được backend đăng ký.");

      const normalizedAttackQueue = normalizeAttackQueue(attackQueue);
      const attackCatalog = await getCatalogAttacks({
        task_id: expContext.selectedTask,
        model_family_id: model.model_family_id,
        checkpoint_id: model.id,
        dataset: dataset.name,
      });
      const catalogByName = new Map(attackCatalog.map((attack) => [attack.name, attack]));
      const unsupportedAttacks = normalizedAttackQueue.filter(
        (attack) => !catalogByName.has(attack.id) || catalogByName.get(attack.id)?.available === false,
      );
      if (unsupportedAttacks.length) {
        throw new Error(`Đòn tấn công không có trong catalog backend hoặc không tương thích: ${unsupportedAttacks.map((attack) => attack.id).join(", ")}.`);
      }

      const config = buildRealRunConfig(expContext, normalizedAttackQueue, model, dataset, attackMode);
      const preflight = await preflightRun(config);
      if (preflight.fatal_errors?.length) throw new Error(preflight.fatal_errors.join(" "));
      setExecutionStep(2);
      const created = await createRun(config);
      setExecutionStatusMessage(executionStatusMessageForJob(created));
      const { job, report, samples } = await waitForRealRun(created.run_id, (currentJob) => {
        // A poll can observe a lower-level worker state after INFERENCING.
        // Keep the checklist monotonic so the UI never appears to run backward.
        setExecutionStep((previous) => Math.max(previous, executionStepForJob(currentJob)));
        setExecutionStatusMessage(executionStatusMessageForJob(currentJob));
      });

      const executedAt = new Date().toISOString();
      const resultData = {
        ...expContext,
        attackMode,
        attackQueue: normalizedAttackQueue,
        selectedModelId: model.id,
        selectedModelName: report.model_version || report.model || expContext.selectedModelName,
        selectedDatasetId: dataset.name,
        selectedDatasetName: report.dataset || dataset.title || dataset.name,
        executedAt,
        status: job.status,
        activeRunId: job.run_id,
      };
      localStorage.setItem("adversai_active_experiment", JSON.stringify(resultData));
      setExecutionStep(5);
      const previous = JSON.parse(localStorage.getItem("advertest_last_session") || "{}");
      localStorage.setItem("advertest_last_session", JSON.stringify({
        ...previous,
        runId: job.run_id,
        report,
        samples,
        recipe: attackMode === "combined" ? config.recipe : null,
        attackMode,
        selectedAttacks: normalizedAttackQueue.map((attack) => attack.id),
        selectedDataset: dataset.name,
        selectedModelFamily: model.model_family_id,
        selectedModelVersion: model.id,
        activeTab: "evidence",
        runStatus: job.status,
      }));

      // ── Auto-push RunRecord to Session API ──
      const lastCell = report.cells?.[report.cells.length - 1];
      const cleanApVal = report.ap_clean ?? report.metrics?.clean?.ap50 ?? 0;
      const attackedApVal = lastCell?.metrics?.ap50 ?? lastCell?.ap ?? 0;
      const mapDropPctVal = cleanApVal > 0
        ? parseFloat(((cleanApVal - attackedApVal) / cleanApVal * 100).toFixed(1))
        : 0;
      const sampleCleanBoxes = samples?.[0]?.clean_prediction?.boxes || [];
      const sampleAttackedBoxes = samples?.[0]?.attacked_prediction?.boxes || [];
      const avgCleanConfVal = sampleCleanBoxes.length > 0
        ? sampleCleanBoxes.reduce((sum, b) => sum + (b.score || 0), 0) / sampleCleanBoxes.length
        : 0;
      const avgAttackedConfVal = sampleAttackedBoxes.length > 0
        ? sampleAttackedBoxes.reduce((sum, b) => sum + (b.score || 0), 0) / sampleAttackedBoxes.length
        : 0;

      const sessionRunIndex = (() => {
        try {
          return Number(localStorage.getItem("advertest_session_run_count") || "0") + 1;
        } catch { return 1; }
      })();
      localStorage.setItem("advertest_session_run_count", String(sessionRunIndex));

      const sessionRunRecord = {
        id: job.run_id,
        name: `Lần ${sessionRunIndex}: ${normalizedAttackQueue.map(a => a.name || a.id).join(" + ")}`,
        timestamp: new Date().toLocaleString("vi-VN"),
        attack_type: normalizedAttackQueue.map(a => a.id).join("+"),
        attack_name: normalizedAttackQueue.map(a => a.name || a.id).join(" + "),
        severity: normalizedAttackQueue[0]?.severity || 3,
        clean_map: parseFloat(cleanApVal.toFixed(4)),
        attacked_map: parseFloat(attackedApVal.toFixed(4)),
        map_drop_pct: mapDropPctVal,
        clean_conf: parseFloat(avgCleanConfVal.toFixed(3)),
        attacked_conf: parseFloat(avgAttackedConfVal.toFixed(3)),
        psnr: String(lastCell?.psnr ?? "N/A"),
        ssim: String(lastCell?.ssim ?? "N/A"),
        inference_ms: parseFloat(
          ((report.duration_seconds ?? 0) * 1000 / Math.max(report.n_samples || 1, 1)).toFixed(1)
        ),
        robustness_score: parseFloat(Math.max(0, 100 - mapDropPctVal).toFixed(1)),
        clean_bbox_count: sampleCleanBoxes.length,
        attacked_bbox_count: sampleAttackedBoxes.length,
        sample_id: samples?.[0]?.sample_id ?? "000000",
        is_combined: attackMode === "combined" && normalizedAttackQueue.length > 1,
        attack_components: normalizedAttackQueue.map(a => a.id),
        seed: config.seed ?? 42,
        backend_run_id: job.run_id,
      };
      if (typeof addRunToSession === "function") {
        addRunToSession(expId, sessionRunRecord).catch(syncErr =>
          console.warn("Could not sync run to session:", syncErr)
        );
      }

      // ── Cache report for RunHistoryBar ──
      try {
        const existingCache = JSON.parse(localStorage.getItem("advertest_run_reports") || "{}");
        const cacheKeys = Object.keys(existingCache);
        if (cacheKeys.length >= 10) {
          delete existingCache[cacheKeys[0]];
        }
        existingCache[job.run_id] = {
          report,
          samples,
          attackQueue: normalizedAttackQueue,
          executedAt: new Date().toISOString(),
          attackMode,
        };
        localStorage.setItem("advertest_run_reports", JSON.stringify(existingCache));
      } catch (cacheErr) {
        console.warn("Could not cache run report:", cacheErr);
      }

      router.push(`/experiments/${expId}/results`);
    } catch (error) {
      setExecutionError(error.message || "Không thể chạy attack bằng backend.");
      setIsExecuting(false);
      setExecutionStep(0);
    }
  };

  const isTask3D = expContext.selectedTask === "detection3d";
  const filteredCategories = ATTACK_CATEGORIES.filter((cat) => {
    if (isTask3D) return cat.task_compatibility?.includes("detection3d");
    return cat.task_compatibility?.includes("detection2d");
  })
    .map((cat) => ({
      ...cat,
      attacks: cat.attacks.filter(
        (a) =>
          a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          a.fullName.toLowerCase().includes(searchQuery.toLowerCase()) ||
          a.desc.toLowerCase().includes(searchQuery.toLowerCase())
      ),
    }))
    .filter((cat) => cat.attacks.length > 0);

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Cấu hình phương pháp tấn công & Tổ hợp đối kháng"
        subtitle="Chọn các kịch bản tổ hợp mẫu sẵn có hoặc tùy chỉnh từng phương pháp tấn công, xem ảnh ví dụ trực quan và nạp vào chuỗi tác động."
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Cấu hình bài toán", href: "/experiments/new" },
          { label: expId, href: `/experiments/${expId}/attack` },
          { label: "Cấu hình tấn công" },
        ]}
      />

      {/* TOP BANNER: ACTIVE EXPERIMENT SELECTION CONTEXT */}
      <div className="p-3.5 rounded-xl bg-gradient-to-r from-blue-950 via-slate-900 to-slate-900 border border-blue-800/60 text-white flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-md">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 text-white flex items-center justify-center font-bold">
            {isTask3D ? <Box className="w-4 h-4" /> : <Target className="w-4 h-4" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-bold text-blue-400 uppercase tracking-wide">
                Đang cấu hình tấn công cho bài toán: {expContext.taskName}
              </span>
            </div>
            <div className="text-xs text-slate-200 mt-0.5 flex flex-wrap items-center gap-2">
              <span>Mô hình: <strong className="text-white">{expContext.selectedModelName}</strong></span>
              <span className="text-slate-500">•</span>
              <span>Dữ liệu: <strong className="text-blue-300">{expContext.selectedDatasetName}</strong></span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Link href="/experiments/new">
            <Button
              variant="secondary"
              size="sm"
              icon={ArrowLeft}
              className="bg-slate-800 text-slate-300 hover:bg-slate-700 border-slate-700 text-xs py-1"
            >
              Đổi Model / Data
            </Button>
          </Link>
        </div>
      </div>

      {/* 1. PREDEFINED ATTACK PRESETS BAR */}
      <Card
        title="1. Chọn tổ hợp kịch bản mẫu sẵn có (Attack Presets)"
        subtitle={`Các kịch bản chuẩn tối ưu cho ${expContext.taskName}`}
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {ATTACK_PRESETS.filter((p) => (isTask3D ? p.task === "detection3d" : p.task === "detection2d")).map((preset) => (
            <div
              key={preset.id}
              onClick={() => handleApplyPreset(preset)}
              className="p-3.5 rounded-xl border border-slate-200 bg-gradient-to-b from-white to-slate-50/70 hover:border-blue-500 hover:shadow-md transition-all cursor-pointer flex flex-col justify-between space-y-2 group"
            >
              <div>
                <div className="text-xs font-bold text-slate-800 group-hover:text-blue-600 transition-colors">
                  {preset.name}
                </div>
                <p className="text-[11px] text-slate-500 line-clamp-2 mt-1">
                  {preset.desc}
                </p>
              </div>

              <div className="pt-2 border-t border-slate-100 flex items-center justify-between">
                <span className="text-[10px] font-semibold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
                  {preset.attacks.length} Đòn phối hợp
                </span>
                <span className="text-[10px] text-slate-400 font-medium group-hover:text-blue-600 flex items-center gap-0.5">
                  Áp dụng ➔
                </span>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* 2. MAIN ATTACK PICKER + INSPECTOR & RECIPE SUMMARY */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
        {/* LEFT COLUMN: Attack Category Catalog (4 Columns) */}
        <div className="xl:col-span-4 space-y-4">
          <Card
            title="2. Danh mục phương pháp tấn công"
            subtitle={`Lọc theo bài toán [${expContext.taskName}]`}
          >
            {/* Search */}
            <div className="relative mb-3">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Tìm phương pháp tấn công..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-3 py-1.5 text-xs rounded-lg border border-slate-200 focus:outline-none focus:border-blue-500 bg-slate-50/50"
              />
            </div>

            {/* Accordion/Category List */}
            <div className="space-y-4 max-h-[580px] overflow-y-auto pr-1">
              {filteredCategories.map((cat) => (
                <div key={cat.category} className="space-y-1.5">
                  <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider px-1">
                    {cat.category}
                  </div>
                  <div className="space-y-1">
                    {cat.attacks.map((atk) => {
                      const isSelected = selectedAttackId === atk.id;
                      const isInQueue = attackQueue.some((q) => q.id === atk.id);
                      return (
                        <div
                          key={atk.id}
                          onClick={() => {
                            setSelectedAttackId(atk.id);
                          }}
                          className={cn(
                            "p-2.5 rounded-lg border text-left cursor-pointer transition-all flex items-center justify-between",
                            isSelected
                              ? "border-blue-600 bg-blue-50/60 shadow-xs ring-1 ring-blue-500"
                              : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                          )}
                        >
                          <div className="flex-1 pr-2">
                            <div className="flex items-center gap-1.5">
                              <span className="text-xs font-bold text-slate-800">
                                {atk.name}
                              </span>
                              {isInQueue && (
                                <span className="w-2 h-2 rounded-full bg-emerald-500" title="Đã có trong chuỗi đòn" />
                              )}
                            </div>
                            <div className="text-[10px] text-slate-500 line-clamp-1">
                              {atk.fullName}
                            </div>
                          </div>
                          <Badge variant="neutral" className="text-[10px] font-mono">
                            {atk.norm}
                          </Badge>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* CENTER COLUMN: Visual Example Preview & Severity Slider & OK Button (5 Columns) */}
        <div className="xl:col-span-5 space-y-4">
          <Card
            title={`Chi tiết & Tác động của: ${currentAttack.name}`}
            subtitle={currentAttack.fullName}
            headerAction={<Badge variant="primary">{currentAttack.norm}</Badge>}
          >
            <div className="space-y-4">
              {/* Description */}
              <div className="p-3 rounded-lg bg-blue-50/60 border border-blue-200 text-xs text-blue-900 leading-relaxed">
                <div className="font-bold mb-1 flex items-center gap-1.5 text-blue-800">
                  <Info className="w-3.5 h-3.5" />
                  <span>Mô tả & Nguyên lý tác động:</span>
                </div>
                {currentAttack.desc}
              </div>

              {/* Visual Example Preview (Before vs After) */}
              <div className="space-y-2">
                <div className="text-xs font-bold text-slate-700 flex items-center justify-between">
                  <span>Ví dụ minh họa trực quan (Visual Example Preview):</span>
                  <span className="text-[10px] text-slate-400 font-normal">Mức độ {severityLevel}/5</span>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  {/* Clean Preview Frame */}
                  <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-900 relative group aspect-4/3 flex flex-col justify-between p-2 shadow-inner">
                    <img
                      src={currentAttack.previewClean || "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=600&auto=format&fit=crop&q=80"}
                      alt="Clean sample"
                      className="absolute inset-0 w-full h-full object-cover opacity-85"
                    />
                    <span className="z-10 bg-black/70 text-white text-[10px] font-semibold px-2 py-0.5 rounded w-max">
                      Ảnh gốc (Clean)
                    </span>
                    {/* Simulated Clean Bbox */}
                    <div className="z-10 absolute left-[20%] top-[35%] w-[45%] h-[40%] border-2 border-emerald-400 bg-emerald-500/10 rounded pointer-events-none">
                      <span className="bg-emerald-600 text-white text-[8px] px-1 rounded absolute -top-4 left-0 font-mono">
                        {isTask3D ? "Car 3D 0.94" : "car 0.95"}
                      </span>
                    </div>
                  </div>

                  {/* Attacked Preview Frame */}
                  <div className="rounded-xl overflow-hidden border border-red-300 bg-slate-900 relative group aspect-4/3 flex flex-col justify-between p-2 shadow-inner">
                    <img
                      src={currentAttack.previewAttacked || "https://images.unsplash.com/photo-1517411032315-54ef2cb783bb?w=600&auto=format&fit=crop&q=80"}
                      alt="Attacked sample"
                      className={cn(
                        "absolute inset-0 w-full h-full object-cover transition-all duration-300",
                        severityLevel >= 4 ? "opacity-90 blur-[1px]" : "opacity-80"
                      )}
                    />
                    <span className="z-10 bg-red-600 text-white text-[10px] font-semibold px-2 py-0.5 rounded w-max flex items-center gap-1">
                      <ShieldAlert className="w-2.5 h-2.5" /> Sau tác động
                    </span>
                    {/* Simulated Missed / Degraded Bbox */}
                    <div className="z-10 absolute left-[20%] top-[35%] w-[45%] h-[40%] border-2 border-dashed border-red-400 bg-red-500/20 rounded pointer-events-none">
                      <span className="bg-red-600 text-white text-[8px] px-1 rounded absolute -top-4 left-0 font-mono">
                        {isTask3D ? "Car 3D 0.42 (Lệch)" : "car 0.38 (Lệch)"}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Severity Slider (1 to 5) */}
              <div className="p-3.5 rounded-xl border border-slate-200 bg-slate-50/80 space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-slate-800">
                    Mức độ nghiêm trọng (Severity Ladder):
                  </span>
                  <span className="font-mono font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded border border-blue-200">
                    Cấp {severityLevel} / 5
                  </span>
                </div>

                <input
                  type="range"
                  min="1"
                  max="5"
                  step="1"
                  value={severityLevel}
                  onChange={(e) => setSeverityLevel(Number(e.target.value))}
                  className="w-full accent-blue-600 h-2 bg-slate-200 rounded-lg cursor-pointer"
                />

                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                  <span>1 (Rất nhẹ)</span>
                  <span>2 (Nhẹ)</span>
                  <span>3 (Vừa)</span>
                  <span>4 (Mạnh)</span>
                  <span>5 (Khắc nghiệt)</span>
                </div>
              </div>

              {/* OK / Add to Recipe Button */}
              <Button
                variant="primary"
                onClick={handleAddCurrentAttackToQueue}
                icon={Check}
                className="w-full justify-center py-2.5 shadow-sm text-xs font-bold"
              >
                Xác nhận: Thêm vào Tóm tắt cấu hình tác động (OK)
              </Button>
            </div>
          </Card>
        </div>

        {/* RIGHT COLUMN: Attack Recipe Summary (3 Columns) */}
        <div className="xl:col-span-3 space-y-4">
          <Card
            title={attackMode === "combined" ? "3. Chuỗi kết hợp tác động (Recipe)" : "3. Tác động riêng lẻ (Individual)"}
            subtitle={`${attackQueue.length} đòn đối kháng đã nạp`}
          >
            {/* Mode Switcher */}
            <div className="p-1 rounded-xl bg-slate-100 border border-slate-200 grid grid-cols-2 gap-1 mb-3">
              <button
                type="button"
                data-testid="mode-combined-btn"
                onClick={() => setAttackMode("combined")}
                className={cn(
                  "flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg text-xs font-bold transition-all",
                  attackMode === "combined"
                    ? "bg-white text-blue-700 shadow-sm border border-slate-200"
                    : "text-slate-600 hover:text-slate-900",
                )}
              >
                <Link2 className="w-3.5 h-3.5" />
                <span>Chuỗi kết hợp</span>
              </button>
              <button
                type="button"
                data-testid="mode-individual-btn"
                onClick={() => setAttackMode("individual")}
                className={cn(
                  "flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg text-xs font-bold transition-all",
                  attackMode === "individual"
                    ? "bg-white text-blue-700 shadow-sm border border-slate-200"
                    : "text-slate-600 hover:text-slate-900",
                )}
              >
                <SlidersHorizontal className="w-3.5 h-3.5" />
                <span>Tách riêng lẻ</span>
              </button>
            </div>

            {/* Mode Explanation Banner */}
            <div className={cn(
              "p-2.5 rounded-lg text-[11px] leading-relaxed border mb-3 flex items-start gap-2",
              attackMode === "combined"
                ? "bg-blue-50/70 border-blue-200 text-blue-800"
                : "bg-amber-50/70 border-amber-200 text-amber-800",
            )}>
              <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <div>
                {attackMode === "combined" ? (
                  <span>
                    <strong>Chế độ kết hợp (Pipeline):</strong> {attackQueue.length} đòn tấn công sẽ được <strong>chồng liên tiếp lên cùng 1 ảnh</strong>.
                  </span>
                ) : (
                  <span>
                    <strong>Chế độ riêng lẻ (Matrix):</strong> Từng đòn tấn công sẽ được chạy độc lập trên <strong>từng ảnh riêng biệt</strong>.
                  </span>
                )}
              </div>
            </div>

            <div className="space-y-2 text-xs">
              {attackQueue.map((item, idx) => (
                <div key={`${item.id}-${idx}`}>
                  <div
                    className={cn(
                      "p-3 rounded-lg border flex items-start justify-between gap-2 transition-all",
                      attackMode === "combined"
                        ? "border-blue-200 bg-blue-50/30"
                        : "border-slate-200 bg-slate-50/60",
                    )}
                  >
                    <div>
                      <div className="font-bold text-slate-800 flex items-center gap-1.5">
                        <span className={cn(
                          "w-4 h-4 rounded-full text-white text-[10px] flex items-center justify-center font-mono",
                          attackMode === "combined" ? "bg-blue-600" : "bg-slate-700",
                        )}>
                          {idx + 1}
                        </span>
                        <span>{item.name}</span>
                        {attackMode === "combined" ? (
                          <span className="text-[9px] font-semibold text-blue-600 bg-blue-100 px-1.5 py-0.2 rounded">Bước {idx + 1}</span>
                        ) : (
                          <span className="text-[9px] font-semibold text-slate-600 bg-slate-200 px-1.5 py-0.2 rounded">Độc lập</span>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-500 mt-1">
                        Mức độ: <strong>Cấp {item.severity}</strong> · Chuẩn: <code>{item.norm}</code>
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => handleRemoveQueueItem(item.id)}
                      disabled={attackQueue.length <= 1}
                      className="text-slate-400 hover:text-red-600 disabled:opacity-30 p-1"
                      title="Xóa đòn này"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  {attackMode === "combined" && idx < attackQueue.length - 1 && (
                    <div className="flex justify-center my-1 text-blue-500 font-bold text-[10px] tracking-wide">
                      ↓ Chồng tiếp đòn tiếp theo
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Actions */}
            <div className="mt-5 space-y-2 pt-3 border-t border-slate-100">
              <Button
                variant="primary"
                onClick={handleStartExecution}
                disabled={isExecuting || attackQueue.length === 0}
                icon={isExecuting ? Sparkles : Play}
                className="w-full justify-center py-2.5 text-xs font-bold bg-emerald-600 hover:bg-emerald-700 shadow-sm"
              >
                {isExecuting ? "Đang chạy suy luận đối kháng..." : "Bắt đầu chạy suy luận"}
              </Button>
              {executionError && (
                <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-[11px] leading-relaxed text-red-700">
                  {executionError}
                </div>
              )}

              <Link href="/experiments/new" className="block">
                <Button variant="secondary" size="sm" icon={ArrowLeft} className="w-full justify-center text-xs">
                  Quay lại Cấu hình bài toán
                </Button>
              </Link>
            </div>
          </Card>
        </div>
      </div>

      {/* LIVE INFERENCE PROGRESS MODAL */}
      {isExecuting && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl p-6 max-w-md w-full space-y-5 shadow-2xl border border-slate-200 animate-scale-in">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-blue-100 text-blue-600 flex items-center justify-center animate-spin">
                <RefreshCw className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-slate-900">Đang thực thi suy luận đối kháng</h3>
                <p className="text-xs text-slate-500 font-mono">Mô hình: {expContext.selectedModelName}</p>
              </div>
            </div>

            {executionStatusMessage && (
              <div className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-800">
                {executionStatusMessage}
              </div>
            )}

            {/* Step Progress Checklist */}
            <div className="space-y-2.5 text-xs border border-slate-100 p-3 rounded-xl bg-slate-50">
              <div className={cn("flex items-center gap-2", executionStep >= 1 ? "text-emerald-700 font-semibold" : "text-slate-400")}>
                {executionStep >= 1 ? <CheckCircle2 className="w-4 h-4 text-emerald-600" /> : <div className="w-4 h-4 rounded-full border border-slate-300" />}
                <span>1. Nạp checkpoint & cấu hình dữ liệu {expContext.selectedDatasetName}...</span>
              </div>
              <div className={cn("flex items-center gap-2", executionStep >= 2 ? "text-emerald-700 font-semibold" : "text-slate-400")}>
                {executionStep >= 2 ? <CheckCircle2 className="w-4 h-4 text-emerald-600" /> : <div className="w-4 h-4 rounded-full border border-slate-300" />}
                <span>2. Sinh nhiễu đối kháng ({attackQueue.map((a) => a.name).join(", ")})...</span>
              </div>
              <div className={cn("flex items-center gap-2", executionStep >= 3 ? "text-emerald-700 font-semibold" : "text-slate-400")}>
                {executionStep >= 3 ? <CheckCircle2 className="w-4 h-4 text-emerald-600" /> : <div className="w-4 h-4 rounded-full border border-slate-300" />}
                <span>3. Chạy suy luận song song Clean vs Attacked Model Output...</span>
              </div>
              <div className={cn("flex items-center gap-2", executionStep >= 4 ? "text-emerald-700 font-semibold" : "text-slate-400")}>
                {executionStep >= 4 ? <CheckCircle2 className="w-4 h-4 text-emerald-600" /> : <div className="w-4 h-4 rounded-full border border-slate-300" />}
                <span>4. Tính toán độ lệch Bounding Box, IoU & Confidence Drop...</span>
              </div>
              <div className={cn("flex items-center gap-2", executionStep >= 5 ? "text-emerald-700 font-bold" : "text-slate-400")}>
                {executionStep >= 5 ? <Sparkles className="w-4 h-4 text-emerald-600" /> : <div className="w-4 h-4 rounded-full border border-slate-300" />}
                <span>5. Hoàn thành! Đang chuẩn bị màn hình kết quả trực quan...</span>
              </div>
            </div>

            <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
              <div
                className="bg-blue-600 h-2 transition-all duration-300"
                style={{ width: `${(executionStep / 5) * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ConfigureAttackPage() {
  return (
    <React.Suspense
      fallback={
        <div className="min-h-screen bg-slate-900 flex items-center justify-center text-slate-400">
          Loading attack configuration...
        </div>
      }
    >
      <ConfigureAttackPageContent />
    </React.Suspense>
  );
}
