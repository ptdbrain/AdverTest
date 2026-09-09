"use client";

import {
  ArrowLeft,
  ArrowRight,
  Box,
  Camera,
  Check,
  CheckCircle2,
  CloudFog,
  CloudRain,
  Cpu,
  Crosshair,
  Database,
  Eye,
  Info,
  Layers,
  Link2,
  Play,
  Plus,
  RefreshCw,
  Search,
  ShieldAlert,
  Sliders,
  SlidersHorizontal,
  Snowflake,
  Sparkles,
  Sun,
  Target,
  Trash2,
  Zap,
} from "lucide-react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import React, { useEffect, useState } from "react";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import Card from "@/components/common/Card";
import PageHeader from "@/components/layout/PageHeader";
import { useProject } from "@/context/ProjectContext";
import {
  addRunToSession,
  cancelRun,
  createAttackConfig,
  createRun,
  deleteAttackConfig,
  getCatalogAttacks,
  getCatalogDatasets,
  getModelVersions,
  getRun,
  getRunReport,
  getRunSamples,
  listAttackConfigs,
  listProjectCheckpoints,
  listProjectDatasetVersions,
  preflightRun,
} from "@/lib/api";
import { visibleAttacks } from "@/lib/attackCatalog";
import { ATTACK_PRESETS } from "@/lib/constants";
import { cn } from "@/lib/utils";

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

/** Scalar fields of a JSON-schema params model — only types a text/number input can express. */
export function schemaParamFields(schema) {
  if (!schema?.properties) return [];
  return Object.entries(schema.properties)
    .filter(([, prop]) => ["number", "integer", "boolean", "string"].includes(prop.type))
    .map(([name, prop]) => ({ name, ...prop }));
}

/** Coerce raw draft strings into typed values per schema; empty entries are dropped (backend default). */
export function coerceParamDraft(schema, draft = {}) {
  const fields = schemaParamFields(schema);
  const out = {};
  for (const field of fields) {
    const raw = draft[field.name];
    if (raw === undefined || raw === null || raw === "") continue;
    if (field.type === "boolean") out[field.name] = raw === true || raw === "true";
    else if (field.type === "number" || field.type === "integer") {
      const num = Number(raw);
      if (Number.isFinite(num)) out[field.name] = num;
    } else out[field.name] = raw;
  }
  return out;
}

function basename(value) {
  return (
    String(value || "")
      .split(/[\\/]/)
      .pop()
      ?.toLowerCase() || ""
  );
}

export function resolveBackendModel(context, versions, projectCheckpoints = []) {
  if (context.selectedModelSource === "project") {
    const checkpoint = projectCheckpoints.find((item) => item.id === context.selectedModelId);
    if (!checkpoint?.runnable || checkpoint.status !== "READY") return null;
    return {
      ...checkpoint,
      model_name: checkpoint.display_name || checkpoint.id,
      task: checkpoint.task_id,
      checkpoint_role: "base",
      source: "project",
    };
  }
  if (context.selectedModelId === "portable-blob-detector-v1") {
    const portableRef = versions.find(
      (version) => version.id === "portable-blob-detector-v1" && version.runnable,
    );
    if (portableRef) return portableRef;
  }

  const requestedFile = basename(String(context.selectedModelName || "").split(" (")[0]);
  const requestedTokens = requestedFile
    .replace(/\.[^.]+$/, "")
    .split(/[-_ ]+/)
    .filter((token) => token.length > 1 && token !== "best");

  const taskVersions = versions.filter(
    (version) => version.runnable && (!context.selectedTask || version.task === context.selectedTask),
  );
  const baseVersions = taskVersions.filter(
    (version) => version.checkpoint_role === "base",
  );

  return (
    taskVersions.find((version) => version.id === context.selectedModelId) ||
    baseVersions.find((version) => basename(version.checkpoint_path) === requestedFile) ||
    taskVersions.find((version) => basename(version.checkpoint_path) === requestedFile) ||
    baseVersions.find((version) => requestedFile && basename(version.checkpoint_path).includes(requestedFile)) ||
    taskVersions.find((version) => requestedFile && basename(version.checkpoint_path).includes(requestedFile)) ||
    baseVersions.find((version) => {
      const modelName = String(version.model_name || "").toLowerCase();
      return modelName.length > 1 && requestedFile.includes(modelName);
    }) ||
    baseVersions.find((version) => {
      const identity = `${version.id} ${version.model_name} ${version.checkpoint_path || ""}`.toLowerCase();
      return requestedTokens.length > 0 && requestedTokens.every((token) => identity.includes(token));
    }) ||
    baseVersions.find((version) => version.id === "yolo11s-base") ||
    taskVersions.find((version) => version.id === "yolo11s-kitti-clean-b0") ||
    baseVersions.find((version) => version.id === "portable-blob-detector-v1") ||
    baseVersions[0] ||
    taskVersions.find((version) => version.checkpoint_role === "base") ||
    taskVersions[0] ||
    versions.find((version) => version.runnable) ||
    versions[0]
  );
}

export function resolveBackendDataset(context, datasets, projectDatasets = []) {
  if (context.selectedDatasetSource === "project") {
    const version = projectDatasets.find((item) => item.id === context.selectedDatasetId);
    if (!version?.runnable || version.status !== "READY") return null;
    return {
      ...version,
      name: version.dataset,
      title: version.display_name,
      source: "project",
    };
  }
  const requestedName = String(context.selectedDatasetName || "").toLowerCase();
  const runnableDatasets = datasets.filter((dataset) => dataset.runnable !== false && dataset.anonymized !== false);
  return (
    runnableDatasets.find((dataset) => dataset.name === context.selectedDatasetId) ||
    runnableDatasets.find((dataset) => dataset.name === "kitti") ||
    runnableDatasets.find((dataset) => dataset.title === context.selectedDatasetName) ||
    runnableDatasets.find(
      (dataset) => requestedName && requestedName.includes(String(dataset.name || "").toLowerCase()),
    ) ||
    runnableDatasets[0] ||
    null
  );
}

/** Per-attack parameter overrides for the run config; empty map when nothing was edited. */
function attackParamsForQueue(queue) {
  const params = Object.fromEntries(
    queue
      .filter((attack) => attack.params && Object.keys(attack.params).length > 0)
      .map((attack) => [attack.id, attack.params]),
  );
  return Object.keys(params).length ? { attack_params: params } : {};
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
  // Workflow §4: the label mapping chosen in the wizard travels with the run
  // and is validated server-side (CLASS_MAPPING_INCOMPLETE / _TARGET_UNKNOWN).
  const classMapping =
    context.class_mapping && Object.keys(context.class_mapping).length ? context.class_mapping : undefined;
  const sampleLimit = Math.min(
    context.limit || 2,
    dataset.sample_count || 2,
    model.id === "portable-blob-detector-v1" ? 1 : 2,
  );
  if (isCombined) {
    return {
      project_id: context.projectId || undefined,
      checkpoint_id: model.id,
      ...(model.source === "project" ? { model_version_id: model.id } : {}),
      model_family_id: model.model_family_id,
      task_id: context.selectedTask,
      dataset: dataset.name,
      ...(dataset.source === "project" ? { dataset_version_id: dataset.id } : {}),
      dataset_params: datasetParams,
      dataset_class_names: context.datasetClassNames || [],
      ...(classMapping ? { class_mapping: classMapping } : {}),
      recipe: {
        name: `recipe-${attackQueue.map((a) => a.id).join("-")}`,
        steps: attackQueue.map((attack, index) => ({
          position: index,
          attack_name: attack.id,
          implementation_version: { cw_l2: "2.0.0", dpatch: "2.0.0", thys_patch: "2.0.0" }[attack.id] || "1.0.0",
          severity: Number(attack.severity) || 3,
          parameters: attack.params && Object.keys(attack.params).length ? attack.params : {},
          seed: 42 + index,
          expected_cost: 1.0,
        })),
      },
      limit: sampleLimit,
      seed: 42,
      iou_threshold: 0.5,
      confidence_threshold: 0.25,
    };
  }

  return {
    project_id: context.projectId || undefined,
    checkpoint_id: model.id,
    ...(model.source === "project" ? { model_version_id: model.id } : {}),
    model_family_id: model.model_family_id,
    task_id: context.selectedTask,
    dataset: dataset.name,
    ...(dataset.source === "project" ? { dataset_version_id: dataset.id } : {}),
    dataset_params: datasetParams,
    dataset_class_names: context.datasetClassNames || [],
    ...(classMapping ? { class_mapping: classMapping } : {}),
    attacks: attackQueue.map((attack) => attack.id),
    severities: Array.from(new Set(attackQueue.map((attack) => Number(attack.severity)))),
    // Per-attack parameter overrides edited in the params panel (empty values
    // fall back to the backend defaults declared in each params_model).
    ...attackParamsForQueue(attackQueue),
    // Explicit pairs keep the exact (attack, severity) combination per queue
    // item (blur@3 + noise@2 runs blur@3 and noise@2, not a cross-product).
    attack_severity_pairs: attackQueue.map((attack) => [attack.id, Number(attack.severity) || 3]),
    limit: sampleLimit,
    seed: 42,
    iou_threshold: 0.5,
    confidence_threshold: 0.25,
  };
}

async function waitForRealRun(runId, onProgress, projectId) {
  while (true) {
    const job = await getRun(runId, projectId);
    onProgress(job);
    if (TERMINAL_RUN_STATES.has(job.status)) {
      if (job.status !== "COMPLETED") throw new Error(job.error || `Run kết thúc với trạng thái ${job.status}.`);
      const report = await getRunReport(runId, projectId);
      const samples = await getRunSamples(runId, {}, projectId);
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
    return "GPU đang khởi động";
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
  const { activeProjectId } = useProject();

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
    limit: 2,
  });

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedAttackId, setSelectedAttackId] = useState("depth_fog");
  const [severityLevel, setSeverityLevel] = useState(3);
  const [attackMode, setAttackMode] = useState("combined"); // "combined" | "individual"
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionStep, setExecutionStep] = useState(0);
  const [executionStatusMessage, setExecutionStatusMessage] = useState("");
  const [activeRunId, setActiveRunId] = useState("");
  const [executionError, setExecutionError] = useState("");
  const [catalogRows, setCatalogRows] = useState([]);
  // Raw per-attack parameter drafts keyed by attack id; coerced when the
  // attack is added/updated in the queue (see coerceParamDraft).
  const [attackParamDrafts, setAttackParamDrafts] = useState({});
  // Named attack configurations saved for the active project (workflow step 5).
  const [savedConfigs, setSavedConfigs] = useState([]);
  const [configName, setConfigName] = useState("");
  const [configBusy, setConfigBusy] = useState(false);
  const [configMessage, setConfigMessage] = useState("");
  const runProjectId = activeProjectId || expContext.projectId || undefined;

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
      id: "motion_blur",
      name: "Motion Blur (Mờ chuyển động)",
      severity: 3,
      eps: "Severity 3",
      norm: "Spatial",
      desc: "Mô phỏng rung lắc camera và chuyển động tốc độ cao.",
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
              setAttackQueue([
                {
                  id: "lidar_beam_drop",
                  name: "Beam Drop (LiDAR 3D)",
                  severity: 3,
                  eps: "Drop 30%",
                  norm: "Spatial 3D",
                  desc: "Mất chùm tia cảm biến LiDAR mô phỏng bụi bám và thời tiết.",
                },
              ]);
            } else {
              setSelectedAttackId("depth_rain");
              setAttackQueue([
                {
                  id: "depth_rain",
                  name: "Depth Rain (Mưa giông)",
                  severity: 3,
                  eps: "Severity 3",
                  norm: "Weather",
                  desc: "Mưa giông tán xạ quang học làm suy giảm tương phản.",
                },
              ]);
            }
          } else if (mode !== "extend" && parsed.selectedTask === "detection3d") {
            setSelectedAttackId("lidar_beam_drop");
            setAttackQueue([
              {
                id: "lidar_beam_drop",
                name: "Beam Drop (LiDAR 3D)",
                severity: 3,
                eps: "Drop 30%",
                norm: "Spatial 3D",
                desc: "Mất chùm tia cảm biến LiDAR mô phỏng bụi bám và thời tiết.",
              },
              {
                id: "lidar_jitter",
                name: "Point Jitter (LiDAR 3D)",
                severity: 3,
                eps: "σ=0.05m",
                norm: "Gaussian 3D",
                desc: "Nhiễu rung tọa độ 3D làm biến dạng voxels.",
              },
            ]);
          }
        }, 0);
      }
    } catch {}
  }, [mode]);

  useEffect(() => {
    let mounted = true;
    getCatalogAttacks({ task_id: expContext.selectedTask })
      .then((rows) => {
        if (mounted && Array.isArray(rows)) setCatalogRows(rows);
      })
      .catch(() => {});
    return () => {
      mounted = false;
    };
  }, [expContext.selectedTask]);

  const catalogAttacks = visibleAttacks(catalogRows, expContext.selectedTask).map((row) => ({
    id: row.name,
    name: row.display_name || row.title || row.name,
    fullName: row.display_name || row.title || row.name,
    desc: row.plain_summary || row.technical_summary || row.reason || "Backend catalog attack",
    norm: row.threat_model || "Catalog",
    isWeather: row.scenario_kind === "environmental_degradation",
    available: row.available !== false,
    unavailable_reason: row.reason,
    // JSON schema of the attack's params_model (drives the per-attack editor).
    paramsSchema: row.params_schema || null,
  }));
  const allAttacksList = catalogAttacks;
  const currentAttack =
    allAttacksList.find((a) => a.id === selectedAttackId) ||
    allAttacksList[0] || {
      id: "",
      name: "Chưa có attack khả dụng",
      fullName: "Danh mục backend chưa sẵn sàng",
      desc: "Chọn đúng task và asset đã được xác thực để tải danh mục attack.",
      norm: "—",
      paramsSchema: null,
      available: false,
    };

  useEffect(() => {
    if (!activeProjectId) {
      setSavedConfigs([]);
      return undefined;
    }
    let cancelled = false;
    listAttackConfigs(activeProjectId)
      .then((rows) => {
        if (!cancelled) setSavedConfigs(Array.isArray(rows) ? rows : []);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [activeProjectId]);

  const handleSaveConfig = async () => {
    const name = configName.trim();
    if (!name || !activeProjectId || configBusy) return;
    setConfigBusy(true);
    setConfigMessage("");
    try {
      const saved = await createAttackConfig(activeProjectId, {
        name,
        task_id: expContext.selectedTask || "detection2d",
        items: normalizeAttackQueue(attackQueue).map((attack) => ({
          attack: attack.id,
          severity: Number(attack.severity) || 3,
          params: attack.params || {},
        })),
      });
      setSavedConfigs((current) => [saved, ...current.filter((row) => row.config_id !== saved.config_id)]);
      setConfigMessage(`Đã lưu cấu hình “${saved.name}”.`);
    } catch (cause) {
      setConfigMessage(cause.message || "Không lưu được cấu hình tấn công.");
    } finally {
      setConfigBusy(false);
    }
  };

  const handleLoadConfig = (row) => {
    const entries = (row.items || []).map((item) => {
      const found = allAttacksList.find((a) => a.id === item.attack);
      return {
        id: item.attack,
        name: found?.name || item.attack,
        severity: item.severity || 3,
        eps: found?.isWeather ? `Severity ${item.severity || 3}` : `${(item.severity || 3) * 2}/255`,
        norm: found?.norm || "Catalog",
        desc: found?.desc || "Attack từ cấu hình đã lưu.",
        ...(item.params && Object.keys(item.params).length ? { params: item.params } : {}),
      };
    });
    if (!entries.length) return;
    setAttackQueue(entries);
    setSelectedAttackId(entries[0].id);
    setSeverityLevel(entries[0].severity);
    setAttackParamDrafts((current) => ({
      ...current,
      [entries[0].id]: { ...(current[entries[0].id] || {}), ...(entries[0].params || {}) },
    }));
    setConfigMessage(`Đã nạp cấu hình “${row.name}” vào hàng đợi.`);
  };

  const handleDeleteConfig = async (configId) => {
    try {
      await deleteAttackConfig(configId);
      setSavedConfigs((current) => current.filter((row) => row.config_id !== configId));
    } catch {
      // leave the list untouched when deletion fails
    }
  };

  // Workflow §4: a run may not start while any dataset class is left unmapped.
  const unmappedClasses = (expContext.datasetClassNames || []).filter(
    (name) => !expContext.class_mapping?.[name],
  );
  const mappingIncomplete = unmappedClasses.length > 0;

  // 1. Preset Selector: Instantly populate queue
  const handleApplyPreset = (preset) => {
    const backendById = new Map(catalogAttacks.map((attack) => [attack.id, attack]));
    const compatibleAttacks = preset.attacks.filter((attack) => backendById.get(attack.id)?.available !== false);
    if (!compatibleAttacks.length) return;
    setAttackQueue(
      compatibleAttacks.map((a) => {
        const found = allAttacksList.find((x) => x.id === a.id);
        return {
          id: a.id,
          name: a.name || found?.name || a.id,
          severity: a.severity || 3,
          eps: a.eps || `Sev-${a.severity || 3}`,
          norm: found?.norm || "Linf",
          desc: found?.desc || "Đòn tấn công đối kháng mô phỏng.",
        };
      }),
    );
  };

  const handleAddCurrentAttackToQueue = () => {
    if (!currentAttack.id || currentAttack.available === false) return;
    const existingIndex = attackQueue.findIndex((a) => a.id === currentAttack.id);
    const coercedParams = coerceParamDraft(currentAttack.paramsSchema, attackParamDrafts[currentAttack.id]);
    const newEntry = {
      id: currentAttack.id,
      name: currentAttack.name,
      severity: severityLevel,
      eps: currentAttack.isWeather ? `Severity ${severityLevel}` : `${severityLevel * 2}/255`,
      norm: currentAttack.norm,
      desc: currentAttack.desc,
      ...(Object.keys(coercedParams).length ? { params: coercedParams } : {}),
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
      if (mappingIncomplete) {
        throw new Error(
          `Còn ${unmappedClasses.length} lớp dataset chưa được ánh xạ (${unmappedClasses.join(", ")}). ` +
            "Hãy quay lại Cấu hình bài toán để hoàn tất đối chiếu nhãn lớp trước khi chạy.",
        );
      }
      const [versions, datasets, projectCheckpoints, projectDatasets] = await Promise.all([
        getModelVersions(),
        getCatalogDatasets({ task_id: expContext.selectedTask }),
        runProjectId ? listProjectCheckpoints(runProjectId) : Promise.resolve([]),
        runProjectId ? listProjectDatasetVersions(runProjectId) : Promise.resolve([]),
      ]);
      const model = resolveBackendModel(expContext, versions, projectCheckpoints);
      const dataset = resolveBackendDataset(expContext, datasets, projectDatasets);
      if (!model) throw new Error("Mô hình đã chọn chưa có base checkpoint hợp lệ để chạy attack.");
      if (!dataset) throw new Error("Bộ dữ liệu đã chọn chưa được backend đăng ký.");

      const normalizedAttackQueue = normalizeAttackQueue(attackQueue);
      const attackCatalog = await getCatalogAttacks({
        task_id: expContext.selectedTask,
        ...(model.source === "project"
          ? {}
          : { model_family_id: model.model_family_id, checkpoint_id: model.id }),
        dataset: dataset.name,
      });
      const catalogByName = new Map(attackCatalog.map((attack) => [attack.name, attack]));
      const unsupportedAttacks = normalizedAttackQueue.filter(
        (attack) => !catalogByName.has(attack.id) || catalogByName.get(attack.id)?.available === false,
      );
      if (unsupportedAttacks.length) {
        throw new Error(
          `Đòn tấn công không có trong catalog backend hoặc không tương thích: ${unsupportedAttacks.map((attack) => attack.id).join(", ")}.`,
        );
      }

      const config = buildRealRunConfig(expContext, normalizedAttackQueue, model, dataset, attackMode);
      const preflight = await preflightRun(config, runProjectId);
      if (preflight.fatal_errors?.length) throw new Error(preflight.fatal_errors.join(" "));
      setExecutionStep(2);
      const created = await createRun(config, runProjectId);
      setActiveRunId(created.run_id);
      setExecutionStatusMessage(executionStatusMessageForJob(created));
      const { job, report, samples } = await waitForRealRun(created.run_id, (currentJob) => {
        // A poll can observe a lower-level worker state after INFERENCING.
        // Keep the checklist monotonic so the UI never appears to run backward.
        setExecutionStep((previous) => Math.max(previous, executionStepForJob(currentJob)));
        setExecutionStatusMessage(executionStatusMessageForJob(currentJob));
      }, runProjectId);

      const executedAt = new Date().toISOString();
      const resultData = {
        ...expContext,
        attackMode,
        attackQueue: normalizedAttackQueue,
        selectedModelId: model.id,
        selectedModelName:
          model.source === "project"
            ? expContext.selectedModelName
            : report.model_version || report.model || expContext.selectedModelName,
        selectedDatasetId: dataset.source === "project" ? dataset.id : dataset.name,
        selectedDatasetName:
          dataset.source === "project" ? expContext.selectedDatasetName : report.dataset || dataset.title || dataset.name,
        executedAt,
        status: job.status,
        activeRunId: job.run_id,
      };
      localStorage.setItem("adversai_active_experiment", JSON.stringify(resultData));
      setExecutionStep(5);
      const previous = JSON.parse(localStorage.getItem("advertest_last_session") || "{}");
      localStorage.setItem(
        "advertest_last_session",
        JSON.stringify({
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
        }),
      );

      // ── Auto-push RunRecord to Session API ──
      const lastCell = report.cells?.[report.cells.length - 1];
      const cleanApVal = report.ap_clean ?? report.metrics?.clean?.ap50 ?? 0;
      const attackedApVal = lastCell?.metrics?.ap50 ?? lastCell?.ap ?? 0;
      const mapDropPctVal =
        cleanApVal > 0 ? parseFloat((((cleanApVal - attackedApVal) / cleanApVal) * 100).toFixed(1)) : 0;
      const sampleCleanBoxes = samples?.[0]?.clean_prediction?.boxes || [];
      const sampleAttackedBoxes = samples?.[0]?.attacked_prediction?.boxes || [];
      const avgCleanConfVal =
        sampleCleanBoxes.length > 0
          ? sampleCleanBoxes.reduce((sum, b) => sum + (b.score || 0), 0) / sampleCleanBoxes.length
          : 0;
      const avgAttackedConfVal =
        sampleAttackedBoxes.length > 0
          ? sampleAttackedBoxes.reduce((sum, b) => sum + (b.score || 0), 0) / sampleAttackedBoxes.length
          : 0;

      const sessionRunIndex = (() => {
        try {
          return Number(localStorage.getItem("advertest_session_run_count") || "0") + 1;
        } catch {
          return 1;
        }
      })();
      localStorage.setItem("advertest_session_run_count", String(sessionRunIndex));

      const sessionRunRecord = {
        id: job.run_id,
        name: `Lần ${sessionRunIndex}: ${normalizedAttackQueue.map((a) => a.name || a.id).join(" + ")}`,
        timestamp: new Date().toLocaleString("vi-VN"),
        attack_type: normalizedAttackQueue.map((a) => a.id).join("+"),
        attack_name: normalizedAttackQueue.map((a) => a.name || a.id).join(" + "),
        severity: normalizedAttackQueue[0]?.severity || 3,
        clean_map: parseFloat(cleanApVal.toFixed(4)),
        attacked_map: parseFloat(attackedApVal.toFixed(4)),
        map_drop_pct: mapDropPctVal,
        clean_conf: parseFloat(avgCleanConfVal.toFixed(3)),
        attacked_conf: parseFloat(avgAttackedConfVal.toFixed(3)),
        psnr: String(lastCell?.psnr ?? "N/A"),
        ssim: String(lastCell?.ssim ?? "N/A"),
        inference_ms: parseFloat(
          (((report.seconds ?? report.duration_seconds ?? 0) * 1000) / Math.max(report.n_samples || 1, 1)).toFixed(1),
        ),
        robustness_score: parseFloat(Math.max(0, 100 - mapDropPctVal).toFixed(1)),
        clean_bbox_count: sampleCleanBoxes.length,
        attacked_bbox_count: sampleAttackedBoxes.length,
        sample_id: samples?.[0]?.sample_id ?? "000000",
        is_combined: attackMode === "combined" && normalizedAttackQueue.length > 1,
        attack_components: normalizedAttackQueue.map((a) => a.id),
        seed: config.seed ?? 42,
        backend_run_id: job.run_id,
      };
      if (typeof addRunToSession === "function") {
        await addRunToSession(expId, sessionRunRecord, runProjectId);
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

  const handleCancelExecution = async () => {
    if (!activeRunId) return;
    try {
      setExecutionStatusMessage("Đang gửi yêu cầu hủy tiến trình...");
      const cancelled = await cancelRun(
        activeRunId,
        runProjectId || process.env.NEXT_PUBLIC_DEFAULT_PROJECT_ID,
      );
      setExecutionStatusMessage(cancelled.status === "CANCELLED" ? "Đã hủy tiến trình." : "Yêu cầu hủy đã được gửi.");
      if (cancelled.status === "CANCELLED") setIsExecuting(false);
    } catch (error) {
      setExecutionError(error.message || "Không thể hủy tiến trình.");
    }
  };

  const isTask3D = expContext.selectedTask === "detection3d";
  const sourceCategories = Object.values(
    catalogAttacks.reduce((groups, attack) => {
      const group = attack.group || "catalog";
      groups[group] = groups[group] || { id: group, title: group, attacks: [] };
      groups[group].attacks.push(attack);
      return groups;
    }, {}),
  );
  const filteredCategories = sourceCategories
    .filter((cat) => {
      return cat.attacks.some((attack) => attack.available !== false);
    })
    .map((cat) => ({
      ...cat,
      attacks: cat.attacks.filter(
        (a) =>
          a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          a.fullName.toLowerCase().includes(searchQuery.toLowerCase()) ||
          a.desc.toLowerCase().includes(searchQuery.toLowerCase()),
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
              <span>
                Mô hình: <strong className="text-white">{expContext.selectedModelName}</strong>
              </span>
              <span className="text-slate-500">•</span>
              <span>
                Dữ liệu: <strong className="text-blue-300">{expContext.selectedDatasetName}</strong>
              </span>
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
          {ATTACK_PRESETS.filter((p) => p.task === expContext.selectedTask).map(
            (preset) => (
              <div
                key={preset.id}
                onClick={() => handleApplyPreset(preset)}
                className="p-3.5 rounded-xl border border-slate-200 bg-gradient-to-b from-white to-slate-50/70 hover:border-blue-500 hover:shadow-md transition-all cursor-pointer flex flex-col justify-between space-y-2 group"
              >
                <div>
                  <div className="text-xs font-bold text-slate-800 group-hover:text-blue-600 transition-colors">
                    {preset.name}
                  </div>
                  <p className="text-[11px] text-slate-500 line-clamp-2 mt-1">{preset.desc}</p>
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
            ),
          )}
        </div>
      </Card>

      {/* 2. MAIN ATTACK PICKER + INSPECTOR & RECIPE SUMMARY */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
        {/* LEFT COLUMN: Attack Category Catalog (4 Columns) */}
        <div className="xl:col-span-4 space-y-4">
          <Card title="2. Danh mục phương pháp tấn công" subtitle={`Lọc theo bài toán [${expContext.taskName}]`}>
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
                <div key={cat.id || cat.category} className="space-y-1.5">
                  <div className="text-[11px] font-bold text-slate-500 uppercase tracking-wider px-1">
                    {cat.title || cat.category}
                  </div>
                  <div className="space-y-1">
                    {cat.attacks.map((atk) => {
                      const isSelected = selectedAttackId === atk.id;
                      const isInQueue = attackQueue.some((q) => q.id === atk.id);
                      return (
                        <div
                          key={atk.id}
                          title={atk.unavailable_reason || ""}
                          onClick={() => {
                            if (atk.available !== false) setSelectedAttackId(atk.id);
                          }}
                          className={cn(
                            "p-2.5 rounded-lg border text-left transition-all flex items-center justify-between",
                            atk.available === false
                              ? "cursor-not-allowed border-slate-200 bg-slate-100 opacity-60"
                              : "cursor-pointer",
                            isSelected
                              ? "border-blue-600 bg-blue-50/60 shadow-xs ring-1 ring-blue-500"
                              : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50",
                          )}
                        >
                          <div className="flex-1 pr-2">
                            <div className="flex items-center gap-1.5">
                              <span className="text-xs font-bold text-slate-800">{atk.name}</span>
                              {atk.available === false && (
                                <span className="text-[10px] font-semibold text-amber-700">Không khả dụng</span>
                              )}
                              {isInQueue && (
                                <span className="w-2 h-2 rounded-full bg-emerald-500" title="Đã có trong chuỗi đòn" />
                              )}
                            </div>
                            <div className="text-[10px] text-slate-500 line-clamp-1">{atk.fullName}</div>
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
                      src={
                        currentAttack.previewClean ||
                        "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd?w=600&auto=format&fit=crop&q=80"
                      }
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
                      src={
                        currentAttack.previewAttacked ||
                        "https://images.unsplash.com/photo-1517411032315-54ef2cb783bb?w=600&auto=format&fit=crop&q=80"
                      }
                      alt="Attacked sample"
                      className={cn(
                        "absolute inset-0 w-full h-full object-cover transition-all duration-300",
                        severityLevel >= 4 ? "opacity-90 blur-[1px]" : "opacity-80",
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
                  <span className="font-bold text-slate-800">Mức độ nghiêm trọng (Severity Ladder):</span>
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

              {/* Per-attack parameter editor (from the backend params_model schema) */}
              {schemaParamFields(currentAttack.paramsSchema).length > 0 && (
                <div
                  className="p-3.5 rounded-xl border border-slate-200 bg-slate-50/80 space-y-2"
                  data-testid="attack-params-editor"
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-slate-800">Tham số riêng của attack:</span>
                    <span className="text-[10px] text-slate-400">Để trống → dùng mặc định backend</span>
                  </div>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {schemaParamFields(currentAttack.paramsSchema).map((field) => {
                      const draft = attackParamDrafts[currentAttack.id]?.[field.name] ?? "";
                      const setDraft = (value) =>
                        setAttackParamDrafts((current) => ({
                          ...current,
                          [currentAttack.id]: { ...current[currentAttack.id], [field.name]: value },
                        }));
                      const placeholder =
                        field.default !== undefined && field.default !== null ? String(field.default) : "mặc định";
                      const label = field.title || field.name;
                      return (
                        <label key={field.name} className="space-y-0.5 text-xs">
                          <span className="font-medium text-slate-700">
                            {label}
                            {field.description ? (
                              <span className="ml-1 font-normal text-slate-400" title={field.description}>
                                ⓘ
                              </span>
                            ) : null}
                          </span>
                          {field.type === "boolean" ? (
                            <select
                              value={draft === "" ? "" : draft === true || draft === "true" ? "true" : "false"}
                              onChange={(event) => setDraft(event.target.value)}
                              className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-xs"
                            >
                              <option value="">mặc định</option>
                              <option value="true">true</option>
                              <option value="false">false</option>
                            </select>
                          ) : field.enum ? (
                            <select
                              value={draft}
                              onChange={(event) => setDraft(event.target.value)}
                              className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-xs"
                            >
                              <option value="">mặc định</option>
                              {field.enum.map((option) => (
                                <option key={String(option)} value={String(option)}>
                                  {String(option)}
                                </option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type={field.type === "number" || field.type === "integer" ? "number" : "text"}
                              value={draft}
                              min={field.minimum}
                              max={field.maximum}
                              step={field.type === "integer" ? 1 : "any"}
                              placeholder={placeholder}
                              onChange={(event) => setDraft(event.target.value)}
                              className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-xs"
                            />
                          )}
                        </label>
                      );
                    })}
                  </div>
                </div>
              )}

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
            title={
              attackMode === "combined" ? "3. Chuỗi kết hợp tác động (Recipe)" : "3. Tác động riêng lẻ (Individual)"
            }
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
            <div
              className={cn(
                "p-2.5 rounded-lg text-[11px] leading-relaxed border mb-3 flex items-start gap-2",
                attackMode === "combined"
                  ? "bg-blue-50/70 border-blue-200 text-blue-800"
                  : "bg-amber-50/70 border-amber-200 text-amber-800",
              )}
            >
              <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <div>
                {attackMode === "combined" ? (
                  <span>
                    <strong>Chế độ kết hợp (Pipeline):</strong> {attackQueue.length} đòn tấn công sẽ được{" "}
                    <strong>chồng liên tiếp lên cùng 1 ảnh</strong>.
                  </span>
                ) : (
                  <span>
                    <strong>Chế độ riêng lẻ (Matrix):</strong> Từng đòn tấn công sẽ được chạy độc lập trên{" "}
                    <strong>từng ảnh riêng biệt</strong>.
                  </span>
                )}
              </div>
            </div>

            {/* Saved attack configurations (workflow step 5): named, per-project, reloadable */}
            <div
              className="p-3 rounded-lg border border-slate-200 bg-white/70 mb-3 space-y-2"
              data-testid="saved-attack-configs"
            >
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="font-bold text-slate-800">Bộ cấu hình đã lưu</span>
                {mode === "reset" && <Badge variant="secondary">Chế độ tạo cấu hình mới</Badge>}
                {mode === "extend" && <Badge variant="secondary">Chế độ chỉnh sửa hiện tại</Badge>}
              </div>
              {savedConfigs.length > 0 && (
                <ul className="space-y-1">
                  {savedConfigs.map((row) => (
                    <li key={row.config_id} className="flex items-center justify-between gap-2 text-[11px]">
                      <button
                        type="button"
                        onClick={() => handleLoadConfig(row)}
                        className="min-w-0 flex-1 truncate text-left text-blue-600 hover:underline"
                        title={`Nạp “${row.name}” (${(row.items || []).length} attack)`}
                      >
                        {row.name} · {(row.items || []).length} attack
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDeleteConfig(row.config_id)}
                        className="shrink-0 text-slate-400 hover:text-red-600"
                        aria-label={`Xóa cấu hình ${row.name}`}
                      >
                        ×
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              {activeProjectId ? (
                <div className="flex items-center gap-1.5">
                  <input
                    value={configName}
                    onChange={(event) => setConfigName(event.target.value)}
                    placeholder="Tên bộ cấu hình, vd: Sương mù cấp 3"
                    className="min-w-0 flex-1 rounded-md border border-slate-300 px-2 py-1.5 text-[11px]"
                  />
                  <Button
                    variant="secondary"
                    className="min-h-7 shrink-0 px-2 py-1 text-[10px]"
                    disabled={!configName.trim() || configBusy}
                    onClick={handleSaveConfig}
                  >
                    Lưu cấu hình
                  </Button>
                </div>
              ) : (
                <p className="text-[10px] text-slate-400">Cần project đang chọn để lưu cấu hình.</p>
              )}
              {configMessage && <p className="text-[10px] text-slate-500">{configMessage}</p>}
            </div>

            <div className="space-y-2 text-xs">
              {attackQueue.map((item, idx) => (
                <div key={`${item.id}-${idx}`}>
                  <div
                    className={cn(
                      "p-3 rounded-lg border flex items-start justify-between gap-2 transition-all",
                      attackMode === "combined" ? "border-blue-200 bg-blue-50/30" : "border-slate-200 bg-slate-50/60",
                    )}
                  >
                    <div>
                      <div className="font-bold text-slate-800 flex items-center gap-1.5">
                        <span
                          className={cn(
                            "w-4 h-4 rounded-full text-white text-[10px] flex items-center justify-center font-mono",
                            attackMode === "combined" ? "bg-blue-600" : "bg-slate-700",
                          )}
                        >
                          {idx + 1}
                        </span>
                        <span>{item.name}</span>
                        {attackMode === "combined" ? (
                          <span className="text-[9px] font-semibold text-blue-600 bg-blue-100 px-1.5 py-0.2 rounded">
                            Bước {idx + 1}
                          </span>
                        ) : (
                          <span className="text-[9px] font-semibold text-slate-600 bg-slate-200 px-1.5 py-0.2 rounded">
                            Độc lập
                          </span>
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
                disabled={isExecuting || attackQueue.length === 0 || mappingIncomplete}
                icon={isExecuting ? Sparkles : Play}
                className="w-full justify-center py-2.5 text-xs font-bold bg-emerald-600 hover:bg-emerald-700 shadow-sm"
              >
                {mappingIncomplete
                  ? "Còn lớp chưa ánh xạ — không thể chạy"
                  : isExecuting
                    ? "Đang chạy suy luận đối kháng..."
                    : "Bắt đầu chạy suy luận"}
              </Button>
              {executionError && (
                <div
                  role="alert"
                  className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-[11px] leading-relaxed text-red-700"
                >
                  {executionError}
                </div>
              )}
              {mappingIncomplete && !executionError && (
                <div
                  role="alert"
                  className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] leading-relaxed text-amber-700"
                >
                  Còn {unmappedClasses.length} lớp dataset chưa được ánh xạ:{" "}
                  <strong>{unmappedClasses.join(", ")}</strong>. Quay lại &quot;Cấu hình bài toán&quot; để ánh xạ hoặc
                  chọn &quot;Bỏ qua có chủ đích&quot;.
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
            {activeRunId && isExecuting && (
              <Button variant="secondary" size="sm" onClick={handleCancelExecution}>
                Hủy tiến trình
              </Button>
            )}

            {/* Step Progress Checklist */}
            <div className="space-y-2.5 text-xs border border-slate-100 p-3 rounded-xl bg-slate-50">
              <div
                className={cn(
                  "flex items-center gap-2",
                  executionStep >= 1 ? "text-emerald-700 font-semibold" : "text-slate-400",
                )}
              >
                {executionStep >= 1 ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                ) : (
                  <div className="w-4 h-4 rounded-full border border-slate-300" />
                )}
                <span>1. Nạp checkpoint & cấu hình dữ liệu {expContext.selectedDatasetName}...</span>
              </div>
              <div
                className={cn(
                  "flex items-center gap-2",
                  executionStep >= 2 ? "text-emerald-700 font-semibold" : "text-slate-400",
                )}
              >
                {executionStep >= 2 ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                ) : (
                  <div className="w-4 h-4 rounded-full border border-slate-300" />
                )}
                <span>2. Sinh nhiễu đối kháng ({attackQueue.map((a) => a.name).join(", ")})...</span>
              </div>
              <div
                className={cn(
                  "flex items-center gap-2",
                  executionStep >= 3 ? "text-emerald-700 font-semibold" : "text-slate-400",
                )}
              >
                {executionStep >= 3 ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                ) : (
                  <div className="w-4 h-4 rounded-full border border-slate-300" />
                )}
                <span>3. Chạy suy luận song song Clean vs Attacked Model Output...</span>
              </div>
              <div
                className={cn(
                  "flex items-center gap-2",
                  executionStep >= 4 ? "text-emerald-700 font-semibold" : "text-slate-400",
                )}
              >
                {executionStep >= 4 ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                ) : (
                  <div className="w-4 h-4 rounded-full border border-slate-300" />
                )}
                <span>4. Tính toán độ lệch Bounding Box, IoU & Confidence Drop...</span>
              </div>
              <div
                className={cn(
                  "flex items-center gap-2",
                  executionStep >= 5 ? "text-emerald-700 font-bold" : "text-slate-400",
                )}
              >
                {executionStep >= 5 ? (
                  <Sparkles className="w-4 h-4 text-emerald-600" />
                ) : (
                  <div className="w-4 h-4 rounded-full border border-slate-300" />
                )}
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
