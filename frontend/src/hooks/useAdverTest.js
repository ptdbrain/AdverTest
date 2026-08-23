import { useState, useEffect, useCallback, useRef } from "react";
import {
  getCatalogAttacks, getCatalogModels, getCatalogDatasets, getModelVersions, getPerceptionModes, getModelFamilies, getBaseCheckpoints, getDefenceCheckpoints, createDefenceRun,
  createRun, createInferenceExperiment, getRun, getRunReport, getRunSamples, connectRunWebSocket, triggerAutoFlag, cancelRun,
  createRetrainingBacklog, addRetrainingBacklogItem,
  estimateRun, preflightRun, randomizeRecipe as randomizeRecipeRequest,
  sweepRecipe as sweepRecipeRequest, previewRecipe as previewRecipeRequest, getRecipePresets, createModelComparison,
  getApiBase,
} from "@/lib/api";

const TERMINAL_STATES = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

/* ---- Fallback data when backend is unavailable ---- */
const FALLBACK_MODES = [
  { id: "detection2d", title: "2D Object Detection", status: "available" },
  { id: "segmentation", title: "Instance Segmentation", status: "coming_later" },
  { id: "detection3d", title: "3D Object Detection", status: "coming_later" },
];

const FALLBACK_FAMILIES = [
  { id: "yolo11", display_name: "YOLO11", runnable: true },
];

const FALLBACK_CHECKPOINTS = [
  { id: "yolo11s-base", model_name: "yolo11s", task: "detection2d", model_family_id: "yolo11", runnable: true, checkpoint_path: "checkpoints/surrogates/yolo11s.pt" },
  { id: "yolo11n", model_name: "YOLO11n", task: "detection2d", model_family_id: "yolo11", runnable: true, checkpoint_path: "yolo11n.pt" },
  { id: "yolo11s", model_name: "YOLO11s", task: "detection2d", model_family_id: "yolo11", runnable: true, checkpoint_path: "yolo11s.pt" },
];

const FALLBACK_DATASETS = [
  { id: "kitti_val", name: "kitti", title: "KITTI Validation", annotation_schema: ["2d_bbox"], benchmark_ready: true, anonymized: true },
  { id: "coco_val", name: "coco", title: "COCO Validation", annotation_schema: ["2d_bbox"], benchmark_ready: true, anonymized: true },
  { id: "synthetic_shapes", name: "synthetic_shapes", title: "Synthetic Shapes", annotation_schema: ["2d_bbox"], benchmark_ready: true, anonymized: true },
];

const FALLBACK_ATTACKS = [
  { name: "fgsm", threat_model: "white_box", attack_type: "gradient", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "pgd", threat_model: "white_box", attack_type: "gradient", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "mi_fgsm", threat_model: "white_box", attack_type: "gradient", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "cw_l2", threat_model: "white_box", attack_type: "optimization", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "tog", threat_model: "white_box", attack_type: "targeted", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "dpatch", threat_model: "white_box", attack_type: "patch", scenario_kind: "physical", available: true, version: "1.0.0" },
  { name: "depth_fog", threat_model: "gray_box", attack_type: "weather", scenario_kind: "physical", available: true, version: "1.0.0" },
  { name: "depth_rain", threat_model: "gray_box", attack_type: "weather", scenario_kind: "physical", available: true, version: "1.0.0" },
  { name: "object_occlusion", threat_model: "gray_box", attack_type: "physical", scenario_kind: "physical", available: true, version: "1.0.0" },
  { name: "sensor_fault", threat_model: "gray_box", attack_type: "sensor", scenario_kind: "physical", available: true, version: "1.0.0" },
  { name: "gaussian_noise", threat_model: "black_box", attack_type: "corruption", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "motion_blur", threat_model: "black_box", attack_type: "corruption", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "defocus_blur", threat_model: "black_box", attack_type: "corruption", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "square_attack", threat_model: "black_box", attack_type: "query", scenario_kind: "digital", available: true, version: "1.0.0" },
];

const FALLBACK_SAMPLES = [
  { attack: "fgsm", severity: 3, degradation: 25, artifacts: { clean_prediction_url: "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", attacked_prediction_url: "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg" } }
];

const FALLBACK_REPORT = {
  model: "YOLO11", model_version: "yolo11n", dataset: "Synthetic Shapes", ap_clean: 0.854, seconds: 12.5, benchmark_metrics_available: true,
  cells: [
    { attack: "fgsm", severity: 1, degradation: 5, group: "A", ap: 0.811 },
    { attack: "fgsm", severity: 3, degradation: 25, group: "A", ap: 0.640 },
    { attack: "fgsm", severity: 5, degradation: 60, group: "A", ap: 0.341 }
  ],
  heatmap: {}
};

function artifactUrl(value) {
  if (!value) return null;
  if (/^https?:\/\//i.test(value)) return value;
  const apiBase = getApiBase();
  return new URL(value, `${apiBase.replace(/\/$/, "")}/`).href;
}

function normalizeSamples(rawSamples) {
  return rawSamples.map((sample) => ({
    ...sample,
    artifacts: Object.fromEntries(Object.entries(sample.artifacts ?? {}).map(([key, value]) => [key, artifactUrl(value)])),
  }));
}

export function useAdverTest() {
  const [attacks, setAttacks] = useState([]);
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [modelVersions, setModelVersions] = useState([]);
  const [modelFamilies, setModelFamilies] = useState([]);
  const [baseCheckpoints, setBaseCheckpoints] = useState([]);
  const [defenceCheckpoints, setDefenceCheckpoints] = useState([]);
  const [modes, setModes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedAttacks, setSelectedAttacks] = useState([]);
  const [mode, setModeState] = useState("detection2d");
  const [selectedModelVersion, setSelectedModelVersion] = useState("");
  const [selectedModelFamily, setSelectedModelFamily] = useState("");
  const [runOptions, setRunOptionsState] = useState({
    seed: 42,
    limit: 8,
    split: "val",
    difficulty: "all",
    confidence: 0.25,
    iou: 0.5,
  });
  const [runId, setRunId] = useState(null);
  const [runStatus, setRunStatus] = useState(null);
  const [progress, setProgress] = useState(0);
  const [progressDetail, setProgressDetail] = useState("");
  const [resultLoadStatus, setResultLoadStatus] = useState("idle");
  const [resultLoadError, setResultLoadError] = useState(null);
  const [report, setReport] = useState(null);
  const [samples, setSamples] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [backlog, setBacklog] = useState(null);
  const [backlogError, setBacklogError] = useState("");
  const [isCreatingBacklog, setIsCreatingBacklog] = useState(false);
  const [activeTab, setActiveTab] = useState("evidence");
  const [defenceBaselineRunId, setDefenceBaselineRunId] = useState(null);
  const [defenceComparison, setDefenceComparison] = useState(null);
  const [recipe, setRecipe] = useState({ name: "manual", steps: [] });
  const [recipePresets, setRecipePresets] = useState([]);
  const wsRef = useRef(null);
  const pollRef = useRef(null);
  const finalizedRef = useRef(false);
  const defenceBaselineRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    let attempts = 0;
    const loadCatalog = async () => {
      while (!cancelled && attempts < 3) {
        attempts += 1;
        const results = await Promise.allSettled([
          getCatalogAttacks({ task_id: "detection2d" }), getCatalogModels(), getCatalogDatasets({ task_id: "detection2d" }),
          getModelVersions(), getPerceptionModes(), getRecipePresets(), getModelFamilies("detection2d"), getBaseCheckpoints("detection2d", "yolo11"),
        ]);
        if (cancelled) return;
        const value = (index, fallback) => results[index].status === "fulfilled" ? results[index].value : fallback;
        const [a, m, d, versions, availableModes, presets, families, checkpoints] = [
          value(0, null), value(1, null), value(2, null), value(3, null), value(4, null), value(5, null), value(6, null), value(7, null),
        ];
        if (a) setAttacks(a);
        if (m) setModels(m);
        if (d) setDatasets(d);
        if (versions) {
          setModelVersions(versions);
        }
        if (families) {
          setModelFamilies(families);
          setSelectedModelFamily(families[0]?.id || "");
        }
        if (checkpoints) { setBaseCheckpoints(checkpoints); setSelectedModelVersion(checkpoints[0]?.id || ""); }
        if (availableModes) setModes(availableModes);
        if (presets) setRecipePresets(presets);
        if (d?.length) {
          const defaultDataset = d.find((item) => item.name === "synthetic_shapes") || d.find((item) => item.anonymized) || d[0];
          setSelectedDataset(defaultDataset.id || defaultDataset.name);
        }
        if (results.some((result) => result.status === "fulfilled")) {
          if (results.some((result) => result.status === "rejected")) console.warn("Some AdverTest catalogs failed to load; retrying only failed data on refresh.");
          setLoading(false);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
      if (!cancelled) {
        console.warn("Backend unavailable — loading fallback catalog data for UI preview.");
        setModes(FALLBACK_MODES);
        setModelFamilies(FALLBACK_FAMILIES);
        setSelectedModelFamily(FALLBACK_FAMILIES[0].id);
        setBaseCheckpoints(FALLBACK_CHECKPOINTS);
        setSelectedModelVersion(FALLBACK_CHECKPOINTS[0].id);
        setDatasets(FALLBACK_DATASETS);
        setSelectedDataset(FALLBACK_DATASETS[2].id);
        setAttacks(FALLBACK_ATTACKS);
        setLoading(false);
      }
    };
    loadCatalog();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getCatalogDatasets({ task_id: mode }), getModelFamilies(mode)])
      .then(([taskDatasets, families]) => {
        if (cancelled) return;
        setDatasets(taskDatasets); setModelFamilies(families);
        setSelectedModelFamily((current) => families.some((item) => item.id === current) ? current : (families[0]?.id || ""));
        setSelectedDataset(taskDatasets[0] ? (taskDatasets[0].id || taskDatasets[0].name) : "");
        setSelectedAttacks([]); setRecipe({ name: "manual", steps: [] });
      })
      .catch((error) => console.warn("Task catalog unavailable:", error));
    return () => { cancelled = true; };
  }, [mode]);

  useEffect(() => {
    let cancelled = false;
    if (!selectedModelFamily) return undefined;
    getBaseCheckpoints(mode, selectedModelFamily)
      .then((checkpoints) => {
        if (cancelled) return;
        setBaseCheckpoints(checkpoints);
        setSelectedModelVersion(checkpoints[0]?.id || "");
      })
      .catch((error) => { if (!cancelled) { console.warn("Base checkpoints unavailable:", error); setBaseCheckpoints([]); setSelectedModelVersion(""); } });
    return () => { cancelled = true; };
  }, [mode, selectedModelFamily]);

  useEffect(() => {
    let cancelled = false;
    getDefenceCheckpoints(mode).then((items) => { if (!cancelled) setDefenceCheckpoints(items); })
      .catch((error) => { if (!cancelled) console.warn("Defence checkpoints unavailable:", error); });
    return () => { cancelled = true; };
  }, [mode]);

  useEffect(() => {
    let cancelled = false;
    if (!selectedModelFamily) return undefined;
    const dataset = datasets.find((item) => (item.id || item.name) === selectedDataset);
    const params = {
      task_id: mode,
      model_family_id: selectedModelFamily,
      ...(selectedModelVersion ? { checkpoint_id: selectedModelVersion } : {}),
      ...(dataset ? { dataset: dataset.dataset || dataset.name } : {}),
    };
    getCatalogAttacks(params)
      .then((taskAttacks) => { if (!cancelled) setAttacks(taskAttacks); })
      .catch((error) => { if (!cancelled) console.warn("Attack availability unavailable:", error); });
    return () => { cancelled = true; };
  }, [datasets, mode, selectedDataset, selectedModelFamily, selectedModelVersion]);

  useEffect(() => {
    const unavailable = new Set(attacks.filter((item) => item.available === false).map((item) => item.name));
    if (!unavailable.size) return;
    const timer = setTimeout(() => {
      setRecipe((current) => {
        const steps = current.steps.filter((step) => !unavailable.has(step.attack_name));
        return steps.length === current.steps.length ? current : { ...current, steps: steps.map((step, position) => ({ ...step, position })) };
      });
      setSelectedAttacks((current) => current.filter((name) => !unavailable.has(name)));
    }, 0);
    return () => clearTimeout(timer);
  }, [attacks]);

  const updateAttackSeverity = useCallback((position, severity) => {
    setRecipe((current) => ({ ...current, steps: current.steps.map((step) => step.position === position ? { ...step, severity } : step) }));
  }, []);
  const setMode = useCallback((nextMode) => {
    setModeState(nextMode);
  }, []);
  const setModelFamily = useCallback((familyId) => setSelectedModelFamily(familyId), []);
  const refreshBaseCheckpoints = useCallback(async () => {
    if (!selectedModelFamily) return;
    const checkpoints = await getBaseCheckpoints(mode, selectedModelFamily);
    setBaseCheckpoints(checkpoints);
    setSelectedModelVersion((current) => checkpoints.some((item) => item.id === current) ? current : (checkpoints[0]?.id || ""));
  }, [mode, selectedModelFamily]);
  const refreshDefenceCheckpoints = useCallback(async () => {
    setDefenceCheckpoints(await getDefenceCheckpoints(mode));
  }, [mode]);
  const setRunOptions = useCallback((changes) => setRunOptionsState((current) => ({ ...current, ...changes })), []);
  const toggleAttack = useCallback((name) => {
    const metadata = attacks.find((item) => item.name === name);
    if (metadata?.available === false) return;
    const isWhiteBox = metadata?.group === "D" || metadata?.threat_model === "white_box" || ["fgsm", "pgd", "cw_l2", "dag", "bim", "deepfool", "apgd"].includes(name);

    setRecipe((current) => {
      const isAlreadySelected = current.steps.some((step) => step.attack_name === name);
      let remaining = current.steps.filter((step) => step.attack_name !== name);

      // If selecting a white-box attack, replace any existing white-box attack to comply with max_white_box_steps=1
      if (!isAlreadySelected && isWhiteBox) {
        remaining = remaining.filter((step) => {
          const stepMeta = attacks.find((item) => item.name === step.attack_name);
          const stepIsWhiteBox = stepMeta?.group === "D" || stepMeta?.threat_model === "white_box" || ["fgsm", "pgd", "cw_l2", "dag", "bim", "deepfool", "apgd"].includes(step.attack_name);
          return !stepIsWhiteBox;
        });
      }

      const steps = !isAlreadySelected
        ? [...remaining, { position: remaining.length, attack_name: name, implementation_version: metadata?.version || "1.0.0", severity: 3, parameters: {}, seed: runOptions.seed, expected_cost: 1 }]
        : remaining;
      return { ...current, steps: steps.map((step, position) => ({ ...step, position })) };
    });

    setSelectedAttacks((current) => {
      if (current.includes(name)) {
        return current.filter((item) => item !== name);
      }
      if (isWhiteBox) {
        const filtered = current.filter((item) => {
          const itemMeta = attacks.find((a) => a.name === item);
          return !(itemMeta?.group === "D" || itemMeta?.threat_model === "white_box" || ["fgsm", "pgd", "cw_l2", "dag", "bim", "deepfool", "apgd"].includes(item));
        });
        return [...filtered, name];
      }
      return [...current, name];
    });
  }, [attacks, runOptions.seed]);
  const addDataset = useCallback((dataset) => {
    if (!dataset) return;
    setDatasets((current) => {
      const id = dataset.id || dataset.name;
      return current.some((item) => (item.id || item.name) === id)
        ? current.map((item) => (item.id || item.name) === id ? { ...item, ...dataset } : item)
        : [dataset, ...current];
    });
    setSelectedDataset(dataset.id || dataset.name || dataset.dataset);
  }, []);

  const buildRunConfig = useCallback(() => {
    const dataset = datasets.find((item) => (item.id || item.name) === selectedDataset) || { name: selectedDataset };
    const datasetParams = {
      merge_van_truck: true,
      ...(dataset.dataset_params || {}),
      ...(runOptions.split ? { split: runOptions.split } : {}),
      ...(runOptions.difficulty ? { difficulty: runOptions.difficulty } : {}),
    };
    return {
      model_family_id: selectedModelFamily,
      checkpoint_id: selectedModelVersion,
      task_id: mode,
      dataset: dataset.dataset || dataset.name || selectedDataset,
      dataset_params: datasetParams,
      recipe,
      limit: runOptions.limit === "all" || runOptions.limit === null || runOptions.limit === 0 || runOptions.limit === "" ? null : Number(runOptions.limit),
      seed: Number(runOptions.seed) || 42,
      iou_threshold: Number(runOptions.iou) || 0.5,
      confidence_threshold: Number(runOptions.confidence) || 0.25,
    };
  }, [datasets, mode, recipe, runOptions, selectedDataset, selectedModelFamily, selectedModelVersion]);

  const finalizeRun = useCallback(async (id, job = null) => {
    if (finalizedRef.current) return;
    finalizedRef.current = true;
    setResultLoadStatus("loading");
    setResultLoadError(null);
    try {
      const [completedReport, rawSamples] = await Promise.all([getRunReport(id), getRunSamples(id)]);
      setReport(completedReport);
      setSamples(normalizeSamples(rawSamples));
      if (defenceBaselineRef.current && defenceBaselineRef.current !== id) {
        try { setDefenceComparison(await createModelComparison(defenceBaselineRef.current, id)); }
        catch (error) { console.warn("Defence comparison is unavailable:", error); }
      }
      setProgressDetail("Done!");
      setResultLoadStatus("ready");
      triggerAutoFlag(id, 30).catch(console.warn);
    } catch (error) {
      const message = error.message || "Run completed but evidence could not be loaded.";
      setProgressDetail(message);
      setResultLoadStatus("failed");
      setResultLoadError(message);
    } finally {
      if (pollRef.current) clearInterval(pollRef.current);
      wsRef.current?.close();
      setRunStatus(job?.status || "COMPLETED");
      setIsRunning(false);
    }
  }, []);

  const monitorRun = useCallback((id) => {
    const applyStatus = (job) => {
      setRunStatus(job.status || job.state);
      if (job.progress != null) setProgress(Math.round(job.progress * 100));
      if (job.status === "GPU_STARTING") setProgressDetail("GPU đang khởi động, job sẽ tự chạy sau khi worker sẵn sàng (thường 1–3 phút)...");
      if (job.status === "PREPARING") setProgressDetail("Loading model & dataset...");
      if (job.status === "GENERATING") setProgressDetail("Generating attack variants...");
      if (job.status === "INFERENCING") setProgressDetail("Running inference...");
      if (job.status === "EVALUATING") setProgressDetail("Computing metrics...");
      if (job.status === "COMPLETED") finalizeRun(id, job);
      if (job.status && job.status !== "COMPLETED" && TERMINAL_STATES.has(job.status)) {
        if (pollRef.current) clearInterval(pollRef.current);
        wsRef.current?.close();
        setProgressDetail(job.error || "Run did not complete.");
        setIsRunning(false);
      }
    };
    wsRef.current = connectRunWebSocket(id, (event) => applyStatus({ status: event.state, progress: event.payload?.progress, error: event.payload?.error }));
    pollRef.current = setInterval(async () => {
      try { applyStatus(await getRun(id)); } catch (error) { console.warn("Run status polling failed:", error); }
    }, 1500);
  }, [finalizeRun]);

  const handleRun = useCallback(async () => {
    const version = baseCheckpoints.find((item) => item.id === selectedModelVersion);
    if (!selectedDataset || !recipe.steps.length) {
      setProgressDetail(!selectedDataset ? "Select a dataset before running." : "Select at least one recipe step before running.");
      setRunStatus("BLOCKED"); return;
    }
    if (!version?.runnable) {
      setProgressDetail(version?.blocked_reason ? `Base checkpoint unavailable: ${version.blocked_reason}` : "This task needs a compatible, validated base checkpoint before it can be run.");
      setRunStatus("BLOCKED"); return;
    }
    const dataset = datasets.find((item) => (item.id || item.name) === selectedDataset);
    const isQuickInference = dataset?.benchmark_ready === false && dataset?.status === "RAW";
    const config = buildRunConfig();
    setIsRunning(true); setRunStatus("PREFLIGHT"); setProgress(0); setReport(null); setSamples([]); setBacklog(null); setBacklogError(""); setActiveTab("evidence"); setProgressDetail("Checking dataset, model and recipe..."); setResultLoadStatus("idle"); setResultLoadError(null); finalizedRef.current = false;
    try {
      let job;
      if (isQuickInference) {
        job = await createInferenceExperiment({
          upload_batch_id: dataset.id,
          checkpoint_id: version.id,
          model_family_id: selectedModelFamily,
          task_id: mode,
          recipe,
          attacks: [],
          severities: recipe.steps.map((step) => step.severity),
          seed: runOptions.seed,
          limit: runOptions.limit,
          iou_threshold: runOptions.iou,
          confidence_threshold: runOptions.confidence,
        });
      } else {
        const preflight = await preflightRun(config);
        if (preflight.fatal_errors?.length) {
          setRunStatus("BLOCKED");
          const msg = preflight.fatal_errors.join(" ");
          const friendly = msg.includes("multiple_white_box_steps") || msg.includes("forbidden_pair")
            ? "Không thể kết hợp nhiều đòn White-box gradient cùng lúc trong 1 chuỗi. Vui lòng chọn 1 đòn White-box kết hợp với Black-box hoặc Corruptions."
            : msg;
          setProgressDetail(friendly);
          setIsRunning(false);
          return;
        }
        job = await createRun(config);
      }
      setRunId(job.run_id); setRunStatus(job.status); setProgressDetail("Queueing..."); monitorRun(job.run_id);
    } catch (error) {
      setRunStatus("FAILED");
      const raw = error.message || "";
      const friendly = raw.includes("multiple_white_box_steps") || raw.includes("forbidden_pair")
        ? "Không thể kết hợp nhiều đòn White-box gradient cùng lúc trong 1 chuỗi. Vui lòng chọn 1 đòn White-box kết hợp với Black-box hoặc Corruptions."
        : raw || "Run failed due to an API error.";
      setProgressDetail(friendly);
      setIsRunning(false);
    }
  }, [baseCheckpoints, buildRunConfig, datasets, mode, monitorRun, recipe, runOptions, selectedDataset, selectedModelFamily, selectedModelVersion]);

  const runDefence = useCallback(async (checkpointId) => {
    if (!runId || !checkpointId) return;
    setDefenceBaselineRunId(runId); defenceBaselineRef.current = runId; setDefenceComparison(null);
    setIsRunning(true); setRunStatus("PREFLIGHT"); setProgress(0); setProgressDetail("Locking the base-run protocol for Defence...");
    try {
      const job = await createDefenceRun(runId, checkpointId);
      setRunId(job.run_id); setRunStatus(job.status); monitorRun(job.run_id);
    } catch (error) {
      setRunStatus("FAILED"); setProgressDetail(error.message || "Defence evaluation could not be started."); setIsRunning(false);
    }
  }, [monitorRun, runId]);

  const createBacklog = useCallback(async () => {
    const failures = (report?.cells ?? []).filter((cell) => cell.degradation > 0);
    if (!runId || !failures.length || isCreatingBacklog || backlog) return;
    setIsCreatingBacklog(true); setBacklogError("");
    try {
      const created = await createRetrainingBacklog(`Measured failures from ${runId}`);
      let updated = created;
      for (const cell of failures) updated = await addRetrainingBacklogItem(created.id, `${runId}:${cell.attack}:severity-${cell.severity}`);
      // Draft is intentionally preserved.  A reviewer must approve it through
      // the Defence workflow before any training request can be submitted.
      setBacklog(updated);
    } catch (error) { setBacklogError(error.message || "Could not create the retraining backlog."); }
    finally { setIsCreatingBacklog(false); }
  }, [backlog, isCreatingBacklog, report, runId]);

  const setCanonicalRecipe = useCallback((source) => {
    const steps = (source.steps || []).filter((step) => {
      const attackName = step.attack_name || step.attack_id || step.attack;
      return attacks.find((item) => item.name === attackName)?.available !== false;
    }).map((step, position) => {
      const attackName = step.attack_name || step.attack_id || step.attack;
      const metadata = attacks.find((item) => item.name === attackName);
      return { position, attack_name: attackName, implementation_version: step.implementation_version || metadata?.version || "1.0.0", severity: step.severity, parameters: step.parameters || {}, seed: step.seed ?? runOptions.seed, expected_cost: step.expected_cost ?? 1 };
    });
    setRecipe({ name: source.name || "recipe", steps });
    setSelectedAttacks(steps.map((step) => step.attack_name));
  }, [attacks, runOptions.seed]);
  const loadPreset = useCallback((presetId) => {
    const preset = recipePresets.find((item) => item.preset_id === presetId);
    if (preset) setCanonicalRecipe(preset);
  }, [recipePresets, setCanonicalRecipe]);
  const randomizeRecipe = useCallback(async (nSteps = 3) => {
    setCanonicalRecipe(await randomizeRecipeRequest(nSteps, null, runOptions.seed));
  }, [runOptions.seed, setCanonicalRecipe]);
  const sweepRecipe = useCallback(async (attack) => {
    setCanonicalRecipe(await sweepRecipeRequest(attack));
  }, [setCanonicalRecipe]);
  const previewRecipe = useCallback(async (payload) => previewRecipeRequest(payload || { recipe }), [recipe]);
  const cancelRunAction = useCallback(async () => {
    if (!runId) return;
    setRunStatus("CANCEL_REQUESTED");
    setProgressDetail("Cancelling job...");
    try {
      const job = await cancelRun(runId);
      if (pollRef.current) clearInterval(pollRef.current);
      wsRef.current?.close();
      setRunStatus(job.status || "CANCELLED");
      setProgressDetail("Run cancelled by user.");
      setIsRunning(false);
    } catch (err) {
      console.warn("Cancel request error:", err);
      if (pollRef.current) clearInterval(pollRef.current);
      wsRef.current?.close();
      setRunStatus("CANCELLED");
      setProgressDetail("Run cancelled.");
      setIsRunning(false);
    }
  }, [runId]);
  const retryEvidence = useCallback(() => { if (!runId) return; finalizedRef.current = false; finalizeRun(runId, { status: "COMPLETED" }); }, [finalizeRun, runId]);

  useEffect(() => () => { wsRef.current?.close(); if (pollRef.current) clearInterval(pollRef.current); }, []);
  const version = baseCheckpoints.find((item) => item.id === selectedModelVersion);
  return { state: { attacks, models, datasets, modelVersions, modelFamilies, baseCheckpoints, defenceCheckpoints, modes, recipePresets, recipe, loading, selectedDataset, selectedAttacks, mode, selectedModelFamily, selectedModelVersion, runOptions, runId, runStatus, progress, progressDetail, resultLoadStatus, resultLoadError, report, samples, isRunning, backlog, backlogError, isCreatingBacklog, trainingBlockedReason: version?.runnable ? "" : "WAITING_FOR_ARTIFACTS", activeTab, defenceBaselineRunId, defenceComparison }, actions: { setSelectedDataset, addDataset, setMode, setModelFamily, setSelectedModelVersion, setRunOptions, refreshBaseCheckpoints, refreshDefenceCheckpoints, toggleAttack, updateAttackSeverity, handleRun, runDefence, createBacklog, setActiveTab, loadPreset, randomizeRecipe, sweepRecipe, previewRecipe, retryEvidence, cancelRun: cancelRunAction } };
}
