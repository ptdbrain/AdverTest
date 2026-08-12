import { useState, useEffect, useCallback, useRef } from "react";
import {
  getCatalogAttacks,
  getCatalogModels,
  getCatalogDatasets,
  getModelVersions,
  getPerceptionModes,
  createRun,
  getRunReport,
  getRunSamples,
  connectRunWebSocket,
  triggerAutoFlag,
  createRetrainingBacklog,
  addRetrainingBacklogItem,
  approveRetrainingBacklog,
} from "@/lib/api";

/**
 * Custom hook to manage AdverTest state and API interactions.
 * Ensures business logic is separated from UI rendering.
 * @returns {Object} Application state and handlers.
 */
export function useAdverTest() {
  /* ---- Catalog state ---- */
  const [attacks, setAttacks] = useState([]);
  const [models, setModels] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [modelVersions, setModelVersions] = useState([]);
  const [modes, setModes] = useState([]);
  const [loading, setLoading] = useState(true);

  /* ---- Config state ---- */
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedAttacks, setSelectedAttacks] = useState([]);
  const [severity, setSeverity] = useState(3);
  const [mode, setMode] = useState("detection2d");
  const [selectedModelVersion, setSelectedModelVersion] = useState("");

  /* ---- Run state ---- */
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

  /* ---- View state ---- */
  const [activeTab, setActiveTab] = useState("compare"); // "compare" | "report"
  
  const wsRef = useRef(null);

  /* ---- Load catalog on mount ---- */
  useEffect(() => {
    Promise.all([getCatalogAttacks(), getCatalogModels(), getCatalogDatasets(), getModelVersions(), getPerceptionModes()])
      .then(([a, m, d, versions, availableModes]) => {
        setAttacks(a);
        setModels(m);
        setDatasets(d);
        setModelVersions(versions);
        setModes(availableModes);
        const initial = versions.find((item) => item.task === "detection2d" && item.runnable);
        if (initial) setSelectedModelVersion(initial.id);
        if (d.length > 0) {
          const defaultDs = d.find((ds) => ds.name === "synthetic_shapes") || d.find((ds) => ds.anonymized);
          setSelectedDataset(defaultDs ? (defaultDs.id || defaultDs.name) : (d[0].id || d[0].name));
        }
      })
      .catch((err) => {
        console.error("Failed to load catalog:", err);
      })
      .finally(() => setLoading(false));
  }, []);

  /* ---- Toggle attack selection ---- */
  const toggleAttack = useCallback((name) => {
    setSelectedAttacks((prev) =>
      prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]
    );
  }, []);

  /* ---- Add dataset helper ---- */
  const addDataset = useCallback((newDs) => {
    if (!newDs) return;
    setDatasets((prev) => {
      const idToMatch = newDs.id || newDs.name;
      const exists = prev.some((d) => (d.id || d.name) === idToMatch);
      return exists ? prev.map((d) => ((d.id || d.name) === idToMatch ? { ...d, ...newDs } : d)) : [newDs, ...prev];
    });
    setSelectedDataset(newDs.id || newDs.name || newDs.dataset);
  }, []);

  /* ---- Run attack ---- */
  const handleRun = useCallback(async () => {
    const version = modelVersions.find((item) => item.id === selectedModelVersion);
    if (!selectedDataset || selectedAttacks.length === 0 || mode !== "detection2d" || !version?.runnable) return;
    setIsRunning(true);
    setRunStatus("QUEUED");
    setProgress(0);
    setReport(null);
    setSamples([]);
    setBacklog(null);
    setBacklogError("");
    setActiveTab("compare");
    setProgressDetail("Queueing...");

    try {
      const dsObj = datasets.find((item) => (item.id || item.name) === selectedDataset) || { name: selectedDataset };
      const targetDataset = dsObj.dataset || dsObj.name || selectedDataset;
      const targetDatasetParams = dsObj.dataset_params || (
        targetDataset === "synthetic_shapes"
          ? { brightness_range: [0.28, 0.48], background_level: 0.18, n_samples: 48 }
          : {}
      );

      const config = {
        model: "yolo11",
        adapter_params: { weights: version.checkpoint_path },
        dataset: targetDataset,
        dataset_params: targetDatasetParams,
        attacks: selectedAttacks,
        severities: [severity],
        limit: 8,
        seed: 42,
      };
      const job = await createRun(config);
      setRunId(job.run_id);

      // Connect WebSocket for progress
      if (wsRef.current) wsRef.current.close();
      const ws = connectRunWebSocket(job.run_id, (event) => {
        setRunStatus(event.state);
        if (event.payload?.progress != null) {
          setProgress(Math.round(event.payload.progress * 100));
        }
        if (event.state === "PREPARING") setProgressDetail("Loading model & dataset...");
        if (event.state === "GENERATING") setProgressDetail("Generating attack variants...");
        if (event.state === "INFERENCING") setProgressDetail("Running inference...");
        if (event.state === "EVALUATING") setProgressDetail("Computing metrics...");
        if (event.state === "COMPLETED") {
          setProgressDetail("Done!");
          getRunReport(job.run_id).then((r) => {
            setReport(r);
            setIsRunning(false);
            // Auto-flag severe degradations (> 30%)
            triggerAutoFlag(job.run_id, 30).catch(console.error);
          });
          getRunSamples(job.run_id).then((rawSamples) => {
            const baseUrl = "http://localhost:8000/data/";
            const fixPath = (p) => p ? p.replace(/\\/g, "/").replace(/^.*\/data\//i, baseUrl) : "";
            const mapped = rawSamples.map(s => ({
              ...s,
              clean_image: fixPath(s.clean_image_path),
              attacked_image: fixPath(s.attacked_image_path),
              overlay_image: fixPath(s.overlay_path)
            }));
            setSamples(mapped);
          });
        }
        if (event.state === "FAILED") {
          setProgressDetail(event.payload?.error || "Run failed");
          setIsRunning(false);
        }
      });
      wsRef.current = ws;
    } catch (err) {
      console.error("Failed to run attack:", err);
      setRunStatus("FAILED");
      setProgressDetail(err.message || "Run failed due to an API error.");
      setIsRunning(false);
    }
  }, [selectedDataset, selectedAttacks, severity, mode, modelVersions, selectedModelVersion, datasets]);

  const createBacklog = useCallback(async () => {
    const failures = (report?.cells ?? []).filter((cell) => cell.degradation > 0);
    if (!runId || failures.length === 0 || isCreatingBacklog || backlog) return;

    setIsCreatingBacklog(true);
    setBacklogError("");
    try {
      const created = await createRetrainingBacklog(`Measured failures from ${runId}`);
      let updated = created;
      for (const cell of failures) {
        const failureId = `${runId}:${cell.attack}:severity-${cell.severity}`;
        updated = await addRetrainingBacklogItem(created.id, failureId);
      }
      setBacklog(await approveRetrainingBacklog(updated.id));
    } catch (err) {
      console.error("Failed to create retraining backlog:", err);
      setBacklogError(err.message || "Could not create the retraining backlog.");
    } finally {
      setIsCreatingBacklog(false);
    }
  }, [backlog, isCreatingBacklog, report, runId]);

  /* ---- Recipe Strategy Actions ---- */
  const loadPreset = useCallback((presetId) => {
    if (presetId === "weather_robustness") {
      setSelectedAttacks(["fog", "rain", "snow", "brightness"]);
      setSeverity(3);
    } else if (presetId === "sensor_fault_suite") {
      setSelectedAttacks(["gaussian_noise", "shot_noise", "impulse_noise"]);
      setSeverity(4);
    }
  }, []);

  const randomizeRecipeAction = useCallback((nSteps = 3) => {
    if (attacks.length === 0) return;
    const shuffled = [...attacks].sort(() => 0.5 - Math.random());
    const selected = shuffled.slice(0, Math.min(nSteps, attacks.length)).map((a) => a.name);
    setSelectedAttacks(selected);
  }, [attacks]);

  const sweepRecipeAction = useCallback((attackName) => {
    setSelectedAttacks([attackName]);
    setSeverity(5);
  }, []);

  const previewRecipeAction = useCallback(async (payload) => {
    return {
      estimated_seconds: (payload?.recipe?.steps?.length || selectedAttacks.length) * 0.4,
      total_samples: payload?.n_samples || 8,
      total_cost_class: "LIGHT",
    };
  }, [selectedAttacks]);

  const cancelRunAction = useCallback(async () => {
    if (!runId) return;
    try {
      setRunStatus("CANCEL_REQUESTED");
      setProgressDetail("Cancelling job...");
      setIsRunning(false);
    } catch (err) {
      console.error("Failed to cancel run:", err);
    }
  }, [runId]);

  const version = modelVersions.find((item) => item.id === selectedModelVersion);
  const trainingBlockedReason = version?.runnable ? "" : "WAITING_FOR_ARTIFACTS";

  /* ---- Cleanup WS ---- */
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return {
    state: {
      attacks,
      models,
      datasets,
      modelVersions,
      modes,
      loading,
      selectedDataset,
      selectedAttacks,
      severity,
      mode,
      selectedModelVersion,
      runId,
      runStatus,
      progress,
      progressDetail,
      report,
      samples,
      isRunning,
      backlog,
      backlogError,
      isCreatingBacklog,
      trainingBlockedReason,
      activeTab,
    },
    actions: {
      setSelectedDataset,
      addDataset,
      setSeverity,
      setMode,
      setSelectedModelVersion,
      toggleAttack,
      handleRun,
      createBacklog,
      setActiveTab,
      loadPreset,
      randomizeRecipe: randomizeRecipeAction,
      sweepRecipe: sweepRecipeAction,
      previewRecipe: previewRecipeAction,
      cancelRun: cancelRunAction,
    }
  };
}

