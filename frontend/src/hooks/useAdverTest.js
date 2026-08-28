import { useState, useEffect, useCallback, useRef } from "react";
import {
  getCatalogAttacks, getCatalogModels, getCatalogDatasets, getModelVersions, getPerceptionModes, getModelFamilies, getBaseCheckpoints, getDefenceCheckpoints, createDefenceRun,
  createRun, createInferenceExperiment, getRun, getRunReport, getRunSamples, connectRunWebSocket, triggerAutoFlag, cancelRun,
  createRetrainingBacklog, addRetrainingBacklogItem,
  estimateRun, preflightRun, randomizeRecipe as randomizeRecipeRequest,
  sweepRecipe as sweepRecipeRequest, previewRecipe as previewRecipeRequest, getRecipePresets, createModelComparison,
  getApiBase, artifactUrl,
} from "@/lib/api";
import { getCanonicalAttackKey, getDescriptiveAttackName, normalizeDatasetKey } from "@/lib/attackNaming";

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
  { name: "cw_l2", threat_model: "white_box", attack_type: "optimization", scenario_kind: "digital", available: true, version: "2.0.0" },
  { name: "tog", threat_model: "white_box", attack_type: "targeted", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "dpatch", threat_model: "white_box", attack_type: "patch", scenario_kind: "physical", available: true, version: "2.0.0" },
  { name: "depth_fog", threat_model: "gray_box", attack_type: "weather", scenario_kind: "physical", available: true, version: "1.0.0" },
  { name: "depth_rain", threat_model: "gray_box", attack_type: "weather", scenario_kind: "physical", available: true, version: "1.0.0" },
  { name: "object_occlusion", threat_model: "gray_box", attack_type: "physical", scenario_kind: "physical", available: true, version: "1.0.0" },
  { name: "sensor_fault", threat_model: "gray_box", attack_type: "sensor", scenario_kind: "physical", available: true, version: "1.0.0" },
  { name: "gaussian_noise", threat_model: "black_box", attack_type: "corruption", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "motion_blur", threat_model: "black_box", attack_type: "corruption", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "defocus_blur", threat_model: "black_box", attack_type: "corruption", scenario_kind: "digital", available: true, version: "1.0.0" },
  { name: "square_attack", threat_model: "black_box", attack_type: "query", scenario_kind: "digital", available: true, version: "1.0.0" },
];

function pathToArtifactUri(pathValue) {
  if (!pathValue) return null;
  if (/^https?:\/\//i.test(pathValue)) return pathValue;
  if (typeof pathValue === "string") {
    const normalized = pathValue.replace(/\\/g, "/");
    const dataIdx = normalized.indexOf("/data/");
    if (dataIdx !== -1) {
      return artifactUrl(normalized.slice(dataIdx));
    }
  }
  return artifactUrl(pathValue);
}

function normalizeSamples(rawSamples) {
  const items = Array.isArray(rawSamples)
    ? rawSamples
    : Array.isArray(rawSamples?.items)
      ? rawSamples.items
      : [];
  return items.map((sample) => {
    const rawArtifacts = sample.artifacts || {};
    const clean_input = artifactUrl(rawArtifacts.clean_input_url) || pathToArtifactUri(sample.clean_image_path);
    const attacked_input = artifactUrl(rawArtifacts.attacked_input_url) || pathToArtifactUri(sample.attacked_image_path);
    const clean_pred = artifactUrl(rawArtifacts.clean_prediction_url) || pathToArtifactUri(sample.clean_prediction_path);
    const attacked_pred = artifactUrl(rawArtifacts.attacked_prediction_url) || pathToArtifactUri(sample.attacked_prediction_path);

    return {
      ...sample,
      artifacts: {
        clean_input_url: clean_input,
        attacked_input_url: attacked_input,
        clean_prediction_url: clean_pred,
        attacked_prediction_url: attacked_pred,
      },
    };
  });
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
  const [activeTab, setActiveTabState] = useState("evidence");
  const [defenceBaselineRunId, setDefenceBaselineRunId] = useState(null);
  const [defenceComparison, setDefenceComparison] = useState(null);
  const [recipe, setRecipe] = useState({ name: "manual", steps: [] });
  const [recipePresets, setRecipePresets] = useState([]);
  const wsRef = useRef(null);
  const pollRef = useRef(null);
  const finalizedRef = useRef(false);
  const defenceBaselineRef = useRef(null);
  const datasetReportsMapRef = useRef({});

  // Restore previous attack run session from localStorage on initial load
  useEffect(() => {
    try {
      const savedMap = localStorage.getItem("advertest_dataset_reports_map");
      if (savedMap) {
        datasetReportsMapRef.current = JSON.parse(savedMap) || {};
      }
      const saved = localStorage.getItem("advertest_last_session");
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.runId) setRunId(parsed.runId);
        if (parsed.recipe) setRecipe(parsed.recipe);
        if (parsed.selectedAttacks) setSelectedAttacks(parsed.selectedAttacks);
        if (parsed.selectedDataset) setSelectedDataset(parsed.selectedDataset);
        if (parsed.selectedModelFamily) setSelectedModelFamily(parsed.selectedModelFamily);
        if (parsed.selectedModelVersion) setSelectedModelVersion(parsed.selectedModelVersion);
        if (parsed.activeTab) setActiveTabState(parsed.activeTab);
        if (parsed.runStatus) setRunStatus(parsed.runStatus);

        const currentKey = normalizeDatasetKey(parsed.selectedDataset || parsed.report?.dataset);
        const currentReport = datasetReportsMapRef.current[currentKey] || parsed.report;
        if (currentReport) {
          setReport(currentReport);
          datasetReportsMapRef.current[currentKey] = currentReport;
        }
        if (parsed.samples && Array.isArray(parsed.samples)) {
          setSamples(currentReport?.samples || parsed.samples);
        }
      }
    } catch (e) {
      console.warn("Could not restore last session:", e);
    }
  }, []);

  const setActiveTab = useCallback((tab) => {
    setActiveTabState(tab);
    try {
      const saved = JSON.parse(localStorage.getItem("advertest_last_session") || "{}");
      localStorage.setItem("advertest_last_session", JSON.stringify({ ...saved, activeTab: tab }));
    } catch {}
  }, []);

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
          setSelectedModelFamily((prev) => prev || families[0]?.id || "");
        }
        if (checkpoints) {
          setBaseCheckpoints(checkpoints);
          setSelectedModelVersion((prev) => prev || checkpoints[0]?.id || "");
        }
        if (availableModes) setModes(availableModes);
        if (presets) setRecipePresets(presets);
        if (d?.length) {
          const defaultDataset = d.find((item) => item.name === "synthetic_shapes") || d.find((item) => item.anonymized) || d[0];
          setSelectedDataset((prev) => prev || defaultDataset.id || defaultDataset.name);
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
        setSelectedModelFamily((prev) => prev || FALLBACK_FAMILIES[0].id);
        setBaseCheckpoints(FALLBACK_CHECKPOINTS);
        setSelectedModelVersion((prev) => prev || FALLBACK_CHECKPOINTS[0].id);
        setDatasets(FALLBACK_DATASETS);
        setSelectedDataset((prev) => prev || FALLBACK_DATASETS[2].id);
        setAttacks(FALLBACK_ATTACKS);
        setLoading(false);
      }
    };
    loadCatalog();
    return () => { cancelled = true; };
  }, []);

  const setRunOptions = useCallback((options) => {
    setRunOptionsState((current) => ({ ...current, ...options }));
  }, []);

  const toggleAttack = useCallback((name) => {
    const isSelected = selectedAttacks.includes(name);
    const updated = isSelected ? selectedAttacks.filter((a) => a !== name) : [...selectedAttacks, name];
    setSelectedAttacks(updated);
    setRecipe((current) => {
      const metadata = attacks.find((item) => item.name === name);
      const steps = isSelected
        ? current.steps.filter((step) => step.attack_name !== name)
        : [...current.steps, { position: current.steps.length, attack_name: name, implementation_version: metadata?.version || "1.0.0", severity: 3, parameters: {}, seed: runOptions.seed, expected_cost: 1 }];
      return { ...current, steps: steps.map((step, position) => ({ ...step, position })) };
    });
  }, [attacks, runOptions.seed, selectedAttacks]);

  useEffect(() => {
    let cancelled = false;
    if (!selectedModelFamily) return;
    getBaseCheckpoints(mode, selectedModelFamily)
      .then((checkpoints) => {
        if (cancelled) return;
        setBaseCheckpoints(checkpoints);
        setSelectedModelVersion((prev) => checkpoints.some((c) => c.id === prev) ? prev : (checkpoints[0]?.id || ""));
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

  const handleSetSelectedDataset = useCallback((newDataset) => {
    setSelectedDataset(newDataset);
    const key = normalizeDatasetKey(newDataset);
    const cached = datasetReportsMapRef.current?.[key];
    if (cached) {
      setReport(cached);
      setSamples(cached.samples || []);
    } else {
      setReport(null);
      setSamples([]);
    }
  }, []);

  const finalizeRun = useCallback(async (id, job = null) => {
    if (finalizedRef.current) return;
    finalizedRef.current = true;
    setResultLoadStatus("loading");
    setResultLoadError(null);
    try {
      const [completedReport, rawSamples] = await Promise.all([
        getRunReport(id),
        getRunSamples(id).catch(() => []),
      ]);
      const initialSamples = (Array.isArray(rawSamples) && rawSamples.length > 0)
        ? rawSamples
        : (completedReport?.worst_cases?.length ? completedReport.worst_cases : completedReport?.sample_results || []);
      const normSamples = normalizeSamples(initialSamples);

      const datasetKey = normalizeDatasetKey(completedReport?.dataset || selectedDataset);
      const existing = datasetReportsMapRef.current?.[datasetKey] || null;

      let finalReport;
      if (!existing) {
        finalReport = {
          ...completedReport,
          samples: normSamples,
          accumulated_runs: [id],
          accumulated_attacks: Array.from(
            new Set((completedReport.cells || []).map((c) => getDescriptiveAttackName(c, c.severity, completedReport)))
          ),
        };
      } else {
        // Accumulate cells from existing and completedReport on same dataset with canonical deduplication
        const cellMap = new Map();
        (existing.cells || []).forEach((c) => {
          const key = getCanonicalAttackKey(c, c.severity, existing);
          cellMap.set(key, c);
        });
        (completedReport.cells || []).forEach((c) => {
          const key = getCanonicalAttackKey(c, c.severity, completedReport);
          cellMap.set(key, c);
        });
        const mergedCells = Array.from(cellMap.values());

        // Accumulate worst_cases
        const sampleMap = new Map();
        (existing.worst_cases || []).forEach((s) => {
          const key = `${s.sample_id}::${getCanonicalAttackKey(s, s.severity, existing)}`;
          sampleMap.set(key, s);
        });
        (completedReport.worst_cases || []).forEach((s) => {
          const key = `${s.sample_id}::${getCanonicalAttackKey(s, s.severity, completedReport)}`;
          sampleMap.set(key, s);
        });
        const mergedWorstCases = Array.from(sampleMap.values());

        // Accumulate samples
        const existingSamples = existing.samples || [];
        const fullSamplesMap = new Map();
        existingSamples.forEach((s) => {
          const key = `${s.sample_id}::${getCanonicalAttackKey(s, s.severity, existing)}`;
          fullSamplesMap.set(key, s);
        });
        normSamples.forEach((s) => {
          const key = `${s.sample_id}::${getCanonicalAttackKey(s, s.severity, completedReport)}`;
          fullSamplesMap.set(key, s);
        });
        const mergedSamples = Array.from(fullSamplesMap.values());

        const prevRuns = existing.accumulated_runs || (existing.run_id ? [existing.run_id] : []);
        const allRuns = Array.from(new Set([...prevRuns, id]));
        const allAttacks = Array.from(
          new Set(mergedCells.map((c) => getDescriptiveAttackName(c, c.severity, completedReport)))
        );

        finalReport = {
          ...completedReport,
          cells: mergedCells,
          worst_cases: mergedWorstCases,
          samples: mergedSamples,
          accumulated_runs: allRuns,
          accumulated_attacks: allAttacks,
          ap_clean: completedReport.ap_clean ?? existing.ap_clean,
        };
      }

      if (!datasetReportsMapRef.current) datasetReportsMapRef.current = {};
      datasetReportsMapRef.current[datasetKey] = finalReport;
      setReport(finalReport);
      setSamples(finalReport.samples || normSamples);

      if (defenceBaselineRef.current && defenceBaselineRef.current !== id) {
        try { setDefenceComparison(await createModelComparison(defenceBaselineRef.current, id)); }
        catch (error) { console.warn("Defence comparison is unavailable:", error); }
      }
      setProgressDetail("Done!");
      setResultLoadStatus("ready");
      triggerAutoFlag(id, 30).catch(console.warn);

      // Save session to localStorage
      try {
        localStorage.setItem("advertest_dataset_reports_map", JSON.stringify(datasetReportsMapRef.current));
        localStorage.setItem("advertest_last_session", JSON.stringify({
          runId: id,
          report: finalReport,
          samples: finalReport.samples || normSamples,
          recipe,
          selectedAttacks,
          selectedDataset,
          selectedModelFamily,
          selectedModelVersion,
          activeTab,
          runStatus: "COMPLETED",
        }));
      } catch (e) {
        console.warn("Could not cache session in localStorage:", e);
      }
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
  }, [activeTab, recipe, selectedAttacks, selectedDataset, selectedModelFamily, selectedModelVersion]);

  const monitorRun = useCallback((id) => {
    const applyStatus = (job) => {
      if (!job) return;
      const status = job.status || job.state;
      setRunStatus(status);
      if (job.progress != null) setProgress(Math.round(job.progress * 100));
      if (status === "PREPARING") setProgressDetail("Loading model & dataset...");
      if (status === "GENERATING") setProgressDetail("Generating attack variants...");
      if (status === "INFERENCING") setProgressDetail("Running inference...");
      if (status === "EVALUATING") setProgressDetail("Computing metrics...");
      if (status === "COMPLETED") finalizeRun(id, job);
      if (status && status !== "COMPLETED" && TERMINAL_STATES.has(status)) {
        if (pollRef.current) clearInterval(pollRef.current);
        wsRef.current?.close();
        setProgressDetail(job.error || "Run did not complete.");
        setIsRunning(false);
      }
    };
    wsRef.current = connectRunWebSocket(id, (event) => applyStatus({ status: event.state, progress: event.payload?.progress, error: event.payload?.error }));
    // Immediate first check in case job completed rapidly
    getRun(id).then(applyStatus).catch(() => {});
    pollRef.current = setInterval(async () => {
      try { applyStatus(await getRun(id)); } catch (error) { console.warn("Run status polling failed:", error); }
    }, 1200);
  }, [finalizeRun]);

  const [activeRunMeta, setActiveRunMeta] = useState(null);

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

    // If a run is currently running, cancel it cleanly before launching the new one
    if (runId && (isRunning || !TERMINAL_STATES.has(runStatus))) {
      try { cancelRun(runId); } catch {}
      if (pollRef.current) clearInterval(pollRef.current);
      wsRef.current?.close();
    }

    setIsRunning(true);
    setRunStatus("PREFLIGHT");
    setProgress(0);
    setBacklog(null);
    setBacklogError("");
    setProgressDetail("Checking dataset, model and recipe...");
    setResultLoadStatus("idle");
    setResultLoadError(null);
    finalizedRef.current = false;
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
      setRunId(job.run_id);
      setRunStatus(job.status);
      setActiveRunMeta({
        runId: job.run_id,
        task: mode,
        modelFamily: selectedModelFamily,
        modelVersion: version?.model_name || selectedModelVersion,
        dataset: dataset?.title || dataset?.name || selectedDataset,
        attacks: selectedAttacks,
        severity: recipe.steps?.[0]?.severity || 3,
        startedAt: new Date().toLocaleTimeString(),
      });
      setProgressDetail(`Đang chạy phiên #${job.run_id.slice(0, 8)} (${selectedAttacks.join(", ") || "Attack"})...`);
      monitorRun(job.run_id);
    } catch (error) {
      setRunStatus("FAILED");
      const raw = error.message || "";
      const friendly = raw.includes("multiple_white_box_steps") || raw.includes("forbidden_pair")
        ? "Không thể kết hợp nhiều đòn White-box gradient cùng lúc trong 1 chuỗi. Vui lòng chọn 1 đòn White-box kết hợp với Black-box hoặc Corruptions."
        : raw || "Run failed due to an API error.";
      setProgressDetail(friendly);
      setIsRunning(false);
    }
  }, [baseCheckpoints, buildRunConfig, datasets, isRunning, mode, monitorRun, recipe, runId, runOptions, runStatus, selectedAttacks, selectedDataset, selectedModelFamily, selectedModelVersion]);

  const runDefence = useCallback(async (checkpointId) => {
    if (!runId || !checkpointId) return;
    setDefenceBaselineRunId(runId); defenceBaselineRef.current = runId; setDefenceComparison(null);
    finalizedRef.current = false;
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

  const resetSession = useCallback(() => {
    const key = normalizeDatasetKey(selectedDataset);
    if (datasetReportsMapRef.current) {
      delete datasetReportsMapRef.current[key];
    }
    try {
      localStorage.setItem("advertest_dataset_reports_map", JSON.stringify(datasetReportsMapRef.current || {}));
      localStorage.removeItem("advertest_last_session");
    } catch {}
    setReport(null);
    setSamples([]);
    setRunId(null);
    setRunStatus(null);
    setProgress(0);
    setProgressDetail("");
    setActiveTabState("evidence");
  }, [selectedDataset]);

  useEffect(() => () => { wsRef.current?.close(); if (pollRef.current) clearInterval(pollRef.current); }, []);

  const version = baseCheckpoints.find((item) => item.id === selectedModelVersion);

  return {
    state: {
      attacks, models, datasets, modelVersions, modelFamilies, baseCheckpoints, defenceCheckpoints, modes, recipePresets, recipe, loading,
      selectedDataset, selectedAttacks, mode, selectedModelFamily, selectedModelVersion, runOptions,
      runId, runStatus, progress, progressDetail, resultLoadStatus, resultLoadError, report, samples, isRunning,
      backlog, backlogError, isCreatingBacklog, trainingBlockedReason: version?.runnable ? "" : "WAITING_FOR_ARTIFACTS",
      activeTab, defenceBaselineRunId, defenceComparison, activeRunMeta,
    },
    actions: {
      setSelectedDataset: handleSetSelectedDataset, addDataset, setMode, setModelFamily, setSelectedModelVersion, setRunOptions, refreshBaseCheckpoints, refreshDefenceCheckpoints,
      toggleAttack, updateAttackSeverity, handleRun, runDefence, createBacklog, setActiveTab, loadPreset, randomizeRecipe, sweepRecipe, previewRecipe,
      retryEvidence, cancelRun: cancelRunAction, resetSession, clearReportHistory: resetSession,
    },
  };
}
