import { useState, useEffect, useCallback, useRef } from "react";
import {
  getCatalogAttacks, getCatalogModels, getCatalogDatasets, getModelVersions, getPerceptionModes,
  createRun, createInferenceExperiment, getRun, getRunReport, getRunSamples, connectRunWebSocket, triggerAutoFlag, cancelRun,
  createRetrainingBacklog, addRetrainingBacklogItem, approveRetrainingBacklog,
  estimateRun, preflightRun, randomizeRecipe as randomizeRecipeRequest,
  sweepRecipe as sweepRecipeRequest, previewRecipe as previewRecipeRequest,
} from "@/lib/api";

const TERMINAL_STATES = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

function artifactUrl(value) {
  if (!value) return null;
  if (/^https?:\/\//i.test(value)) return value;
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
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
  const [modes, setModes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedAttacks, setSelectedAttacks] = useState([]);
  const [severity, setSeverityState] = useState(3);
  const [selectedSeverities, setSelectedSeverities] = useState([3]);
  const [mode, setModeState] = useState("detection2d");
  const [selectedModelVersion, setSelectedModelVersion] = useState("");
  const [runOptions, setRunOptionsState] = useState({ seed: 42, limit: 8, confidence: 0.25, iou: 0.5 });
  const [runId, setRunId] = useState(null);
  const [runStatus, setRunStatus] = useState(null);
  const [progress, setProgress] = useState(0);
  const [progressDetail, setProgressDetail] = useState("");
  const [report, setReport] = useState(null);
  const [samples, setSamples] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [backlog, setBacklog] = useState(null);
  const [backlogError, setBacklogError] = useState("");
  const [isCreatingBacklog, setIsCreatingBacklog] = useState(false);
  const [activeTab, setActiveTab] = useState("evidence");
  const wsRef = useRef(null);
  const pollRef = useRef(null);
  const finalizedRef = useRef(false);

  useEffect(() => {
    Promise.all([getCatalogAttacks(), getCatalogModels(), getCatalogDatasets(), getModelVersions(), getPerceptionModes()])
      .then(([a, m, d, versions, availableModes]) => {
        setAttacks(a); setModels(m); setDatasets(d); setModelVersions(versions); setModes(availableModes);
        const initial = versions.find((item) => item.task === "detection2d" && item.runnable);
        if (initial) setSelectedModelVersion(initial.id);
        if (d.length) {
          const defaultDataset = d.find((item) => item.name === "synthetic_shapes") || d.find((item) => item.anonymized) || d[0];
          setSelectedDataset(defaultDataset.id || defaultDataset.name);
        }
      })
      .catch((error) => console.error("Failed to load catalog:", error))
      .finally(() => setLoading(false));
  }, []);

  const setSeverity = useCallback((value) => {
    setSeverityState(value);
    setSelectedSeverities([value]);
  }, []);
  const setMode = useCallback((nextMode) => {
    setModeState(nextMode);
    const valid = modelVersions.find((item) => item.id === selectedModelVersion && item.task === nextMode && item.runnable);
    if (!valid) setSelectedModelVersion(modelVersions.find((item) => item.task === nextMode && item.runnable)?.id ?? "");
  }, [modelVersions, selectedModelVersion]);
  const setRunOptions = useCallback((changes) => setRunOptionsState((current) => ({ ...current, ...changes })), []);
  const toggleAttack = useCallback((name) => setSelectedAttacks((current) => current.includes(name) ? current.filter((item) => item !== name) : [...current, name]), []);
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
    const version = modelVersions.find((item) => item.id === selectedModelVersion);
    const dataset = datasets.find((item) => (item.id || item.name) === selectedDataset) || { name: selectedDataset };
    return {
      model: "yolo11",
      adapter_params: { weights: version?.checkpoint_path },
      dataset: dataset.dataset || dataset.name || selectedDataset,
      dataset_params: dataset.dataset_params || {},
      attacks: selectedAttacks,
      severities: selectedSeverities,
      limit: runOptions.limit,
      seed: runOptions.seed,
      iou_threshold: runOptions.iou,
      confidence_threshold: runOptions.confidence,
    };
  }, [datasets, modelVersions, runOptions, selectedAttacks, selectedDataset, selectedModelVersion, selectedSeverities]);

  const finalizeRun = useCallback(async (id, job = null) => {
    if (finalizedRef.current) return;
    finalizedRef.current = true;
    try {
      const [completedReport, rawSamples] = await Promise.all([getRunReport(id), getRunSamples(id)]);
      setReport(completedReport);
      setSamples(normalizeSamples(rawSamples));
      setProgressDetail("Done!");
      triggerAutoFlag(id, 30).catch(console.warn);
    } catch (error) {
      setProgressDetail(error.message || "Run completed but evidence could not be loaded.");
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
      if (job.status === "PREPARING") setProgressDetail("Loading model & dataset...");
      if (job.status === "GENERATING") setProgressDetail("Generating attack variants...");
      if (job.status === "INFERENCING") setProgressDetail("Running inference...");
      if (job.status === "EVALUATING") setProgressDetail("Computing metrics...");
      if (job.status === "COMPLETED") finalizeRun(id, job);
      if (job.status && job.status !== "COMPLETED" && TERMINAL_STATES.has(job.status)) {
        if (pollRef.current) clearInterval(pollRef.current);
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
    const version = modelVersions.find((item) => item.id === selectedModelVersion);
    if (!selectedDataset || !selectedAttacks.length) {
      setProgressDetail(!selectedDataset ? "Select a dataset before running." : "Select at least one attack before running.");
      setRunStatus("BLOCKED"); return;
    }
    if (mode !== "detection2d" || !version?.runnable || !version.checkpoint_path) {
      setProgressDetail(mode !== "detection2d" ? "Segmentation is not yet runnable from this workflow." : version?.blocked_reason ? `Model version unavailable: ${version.blocked_reason}` : "YOLO checkpoint unavailable. Check /api/v1/model-versions and RUNS_ROOT.");
      setRunStatus("BLOCKED"); return;
    }
    const dataset = datasets.find((item) => (item.id || item.name) === selectedDataset);
    const isQuickInference = dataset?.benchmark_ready === false && dataset?.status === "RAW";
    const config = buildRunConfig();
    setIsRunning(true); setRunStatus("PREFLIGHT"); setProgress(0); setReport(null); setSamples([]); setBacklog(null); setBacklogError(""); setActiveTab("evidence"); setProgressDetail("Checking dataset, model and recipe..."); finalizedRef.current = false;
    try {
      let job;
      if (isQuickInference) {
        job = await createInferenceExperiment({
          upload_batch_id: dataset.id,
          model_version_id: version.id,
          attacks: selectedAttacks,
          severities: selectedSeverities,
          seed: runOptions.seed,
          limit: runOptions.limit,
          iou_threshold: runOptions.iou,
          confidence_threshold: runOptions.confidence,
        });
      } else {
        const preflight = await preflightRun(config);
        if (preflight.fatal_errors?.length) {
          setRunStatus("BLOCKED"); setProgressDetail(preflight.fatal_errors.join(" ")); setIsRunning(false); return;
        }
        job = await createRun(config);
      }
      setRunId(job.run_id); setRunStatus(job.status); setProgressDetail("Queueing..."); monitorRun(job.run_id);
    } catch (error) {
      setRunStatus("FAILED"); setProgressDetail(error.message || "Run failed due to an API error."); setIsRunning(false);
    }
  }, [buildRunConfig, datasets, mode, modelVersions, monitorRun, runOptions, selectedAttacks, selectedDataset, selectedModelVersion, selectedSeverities]);

  const createBacklog = useCallback(async () => {
    const failures = (report?.cells ?? []).filter((cell) => cell.degradation > 0);
    if (!runId || !failures.length || isCreatingBacklog || backlog) return;
    setIsCreatingBacklog(true); setBacklogError("");
    try {
      const created = await createRetrainingBacklog(`Measured failures from ${runId}`);
      let updated = created;
      for (const cell of failures) updated = await addRetrainingBacklogItem(created.id, `${runId}:${cell.attack}:severity-${cell.severity}`);
      setBacklog(await approveRetrainingBacklog(updated.id));
    } catch (error) { setBacklogError(error.message || "Could not create the retraining backlog."); }
    finally { setIsCreatingBacklog(false); }
  }, [backlog, isCreatingBacklog, report, runId]);

  const loadPreset = useCallback((presetId) => {
    const recipes = { weather_robustness: ["fog", "rain", "snow", "brightness"], sensor_fault_suite: ["gaussian_noise", "shot_noise", "impulse_noise"] };
    if (recipes[presetId]) { setSelectedAttacks(recipes[presetId]); setSeverity(3); }
  }, [setSeverity]);
  const randomizeRecipe = useCallback(async (nSteps = 3) => {
    const recipe = await randomizeRecipeRequest(nSteps, null, runOptions.seed);
    setSelectedAttacks(recipe.steps.map((step) => step.attack_id));
    setSelectedSeverities([...new Set(recipe.steps.map((step) => step.severity))]);
  }, [runOptions.seed]);
  const sweepRecipe = useCallback(async (attack) => {
    const recipe = await sweepRecipeRequest(attack);
    setSelectedAttacks([attack]); setSelectedSeverities(recipe.steps.map((step) => step.severity));
  }, []);
  const previewRecipe = useCallback(async () => estimateRun(buildRunConfig()), [buildRunConfig]);
  const cancelRunAction = useCallback(async () => { if (!runId) return; setRunStatus("CANCEL_REQUESTED"); setProgressDetail("Cancelling job..."); const job = await cancelRun(runId); setRunStatus(job.status); }, [runId]);

  useEffect(() => () => { wsRef.current?.close(); if (pollRef.current) clearInterval(pollRef.current); }, []);
  const version = modelVersions.find((item) => item.id === selectedModelVersion);
  return { state: { attacks, models, datasets, modelVersions, modes, loading, selectedDataset, selectedAttacks, severity, selectedSeverities, mode, selectedModelVersion, runOptions, runId, runStatus, progress, progressDetail, report, samples, isRunning, backlog, backlogError, isCreatingBacklog, trainingBlockedReason: version?.runnable ? "" : "WAITING_FOR_ARTIFACTS", activeTab }, actions: { setSelectedDataset, addDataset, setSeverity, setMode, setSelectedModelVersion, setRunOptions, toggleAttack, handleRun, createBacklog, setActiveTab, loadPreset, randomizeRecipe, sweepRecipe, previewRecipe, cancelRun: cancelRunAction } };
}
