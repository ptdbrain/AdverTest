import { useState, useEffect, useCallback, useRef } from "react";
import {
  getCatalogAttacks, getCatalogModels, getCatalogDatasets, getModelVersions, getPerceptionModes,
  createRun, createInferenceExperiment, getRun, getRunReport, getRunSamples, connectRunWebSocket, triggerAutoFlag, cancelRun,
  createRetrainingBacklog, addRetrainingBacklogItem, approveRetrainingBacklog,
  estimateRun, preflightRun, randomizeRecipe as randomizeRecipeRequest,
  sweepRecipe as sweepRecipeRequest, previewRecipe as previewRecipeRequest, getRecipePresets,
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
  const [mode, setModeState] = useState("detection2d");
  const [selectedModelVersion, setSelectedModelVersion] = useState("");
  const [runOptions, setRunOptionsState] = useState({ seed: 42, limit: 8, confidence: 0.25, iou: 0.5 });
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
  const [recipe, setRecipe] = useState({ name: "manual", steps: [] });
  const [recipePresets, setRecipePresets] = useState([]);
  const wsRef = useRef(null);
  const pollRef = useRef(null);
  const finalizedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let attempts = 0;
    const loadCatalog = async () => {
      while (!cancelled && attempts < 3) {
        attempts += 1;
        const results = await Promise.allSettled([
          getCatalogAttacks(), getCatalogModels(), getCatalogDatasets(),
          getModelVersions(), getPerceptionModes(), getRecipePresets(),
        ]);
        if (cancelled) return;
        const value = (index, fallback) => results[index].status === "fulfilled" ? results[index].value : fallback;
        const [a, m, d, versions, availableModes, presets] = [
          value(0, null), value(1, null), value(2, null), value(3, null), value(4, null), value(5, null),
        ];
        if (a) setAttacks(a);
        if (m) setModels(m);
        if (d) setDatasets(d);
        if (versions) {
          setModelVersions(versions);
          const initial = versions.find((item) => item.task === "detection2d" && item.runnable);
          if (initial) setSelectedModelVersion(initial.id);
        }
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
        console.error("Failed to load AdverTest catalogs after retries.");
        setLoading(false);
      }
    };
    loadCatalog();
    return () => { cancelled = true; };
  }, []);

  const updateAttackSeverity = useCallback((position, severity) => {
    setRecipe((current) => ({ ...current, steps: current.steps.map((step) => step.position === position ? { ...step, severity } : step) }));
  }, []);
  const setMode = useCallback((nextMode) => {
    setModeState(nextMode);
    const valid = modelVersions.find((item) => item.id === selectedModelVersion && item.task === nextMode && item.runnable);
    if (!valid) setSelectedModelVersion(modelVersions.find((item) => item.task === nextMode && item.runnable)?.id ?? "");
  }, [modelVersions, selectedModelVersion]);
  const setRunOptions = useCallback((changes) => setRunOptionsState((current) => ({ ...current, ...changes })), []);
  const toggleAttack = useCallback((name) => {
    const metadata = attacks.find((item) => item.name === name);
    setRecipe((current) => {
      const remaining = current.steps.filter((step) => step.attack_name !== name);
      const steps = remaining.length === current.steps.length
        ? [...remaining, { position: remaining.length, attack_name: name, implementation_version: metadata?.version || "1.0.0", severity: 3, parameters: {}, seed: runOptions.seed, expected_cost: 1 }]
        : remaining;
      return { ...current, steps: steps.map((step, position) => ({ ...step, position })) };
    });
    setSelectedAttacks((current) => current.includes(name) ? current.filter((item) => item !== name) : [...current, name]);
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
    return {
      model_version_id: selectedModelVersion,
      task_id: mode,
      dataset: dataset.dataset || dataset.name || selectedDataset,
      dataset_params: dataset.dataset_params || {},
      recipe,
      limit: runOptions.limit,
      seed: runOptions.seed,
      iou_threshold: runOptions.iou,
      confidence_threshold: runOptions.confidence,
    };
  }, [datasets, mode, recipe, runOptions, selectedDataset, selectedModelVersion]);

  const finalizeRun = useCallback(async (id, job = null) => {
    if (finalizedRef.current) return;
    finalizedRef.current = true;
    setResultLoadStatus("loading");
    setResultLoadError(null);
    try {
      const [completedReport, rawSamples] = await Promise.all([getRunReport(id), getRunSamples(id)]);
      setReport(completedReport);
      setSamples(normalizeSamples(rawSamples));
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
    const version = modelVersions.find((item) => item.id === selectedModelVersion);
    if (!selectedDataset || !recipe.steps.length) {
      setProgressDetail(!selectedDataset ? "Select a dataset before running." : "Select at least one recipe step before running.");
      setRunStatus("BLOCKED"); return;
    }
    if (mode !== "detection2d" || !version?.runnable) {
      setProgressDetail(mode !== "detection2d" ? "This task needs a compatible, validated task runtime and checkpoint before it can be run." : version?.blocked_reason ? `Model version unavailable: ${version.blocked_reason}` : "YOLO checkpoint unavailable. Check /api/v1/model-versions and RUNS_ROOT.");
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
          model_version_id: version.id,
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
          setRunStatus("BLOCKED"); setProgressDetail(preflight.fatal_errors.join(" ")); setIsRunning(false); return;
        }
        job = await createRun(config);
      }
      setRunId(job.run_id); setRunStatus(job.status); setProgressDetail("Queueing..."); monitorRun(job.run_id);
    } catch (error) {
      setRunStatus("FAILED"); setProgressDetail(error.message || "Run failed due to an API error."); setIsRunning(false);
    }
  }, [buildRunConfig, datasets, mode, modelVersions, monitorRun, recipe, runOptions, selectedDataset, selectedModelVersion]);

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

  const setCanonicalRecipe = useCallback((source) => {
    const steps = (source.steps || []).map((step, position) => {
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
  const previewRecipe = useCallback(async (payload) => previewRecipeRequest(payload), []);
  const cancelRunAction = useCallback(async () => { if (!runId) return; setRunStatus("CANCEL_REQUESTED"); setProgressDetail("Cancelling job..."); const job = await cancelRun(runId); setRunStatus(job.status); }, [runId]);
  const retryEvidence = useCallback(() => { if (!runId) return; finalizedRef.current = false; finalizeRun(runId, { status: "COMPLETED" }); }, [finalizeRun, runId]);

  useEffect(() => () => { wsRef.current?.close(); if (pollRef.current) clearInterval(pollRef.current); }, []);
  const version = modelVersions.find((item) => item.id === selectedModelVersion);
  return { state: { attacks, models, datasets, modelVersions, modes, recipePresets, recipe, loading, selectedDataset, selectedAttacks, mode, selectedModelVersion, runOptions, runId, runStatus, progress, progressDetail, resultLoadStatus, resultLoadError, report, samples, isRunning, backlog, backlogError, isCreatingBacklog, trainingBlockedReason: version?.runnable ? "" : "WAITING_FOR_ARTIFACTS", activeTab }, actions: { setSelectedDataset, addDataset, setMode, setSelectedModelVersion, setRunOptions, toggleAttack, updateAttackSeverity, handleRun, createBacklog, setActiveTab, loadPreset, randomizeRecipe, sweepRecipe, previewRecipe, retryEvidence, cancelRun: cancelRunAction } };
}
