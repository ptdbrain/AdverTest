import { useState, useEffect, useCallback, useRef } from "react";
import {
  getCatalogAttacks,
  getCatalogModels,
  getCatalogDatasets,
  createRun,
  getRunReport,
  getRunSamples,
  connectRunWebSocket,
  triggerAutoFlag,
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
  const [loading, setLoading] = useState(true);

  /* ---- Config state ---- */
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedAttacks, setSelectedAttacks] = useState([]);
  const [severity, setSeverity] = useState(3);

  /* ---- Run state ---- */
  const [runId, setRunId] = useState(null);
  const [runStatus, setRunStatus] = useState(null);
  const [progress, setProgress] = useState(0);
  const [progressDetail, setProgressDetail] = useState("");
  const [report, setReport] = useState(null);
  const [samples, setSamples] = useState([]);
  const [isRunning, setIsRunning] = useState(false);

  /* ---- View state ---- */
  const [activeTab, setActiveTab] = useState("compare"); // "compare" | "report"
  
  const wsRef = useRef(null);

  /* ---- Load catalog on mount ---- */
  useEffect(() => {
    Promise.all([getCatalogAttacks(), getCatalogModels(), getCatalogDatasets()])
      .then(([a, m, d]) => {
        setAttacks(a);
        setModels(m);
        setDatasets(d);
        if (d.length > 0) {
          const defaultDs = d.find((ds) => ds.name === "synthetic_shapes") || d.find((ds) => ds.anonymized);
          setSelectedDataset(defaultDs ? defaultDs.name : d[0].name);
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

  /* ---- Run attack ---- */
  const handleRun = useCallback(async () => {
    if (!selectedDataset || selectedAttacks.length === 0) return;
    setIsRunning(true);
    setRunStatus("QUEUED");
    setProgress(0);
    setReport(null);
    setSamples([]);
    setActiveTab("compare");
    setProgressDetail("Queueing...");

    try {
      const config = {
        model: "blob_detector",
        dataset: selectedDataset,
        dataset_params:
          selectedDataset === "synthetic_shapes"
            ? { brightness_range: [0.28, 0.48], background_level: 0.18, n_samples: 48 }
            : {},
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
  }, [selectedDataset, selectedAttacks, severity]);

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
      loading,
      selectedDataset,
      selectedAttacks,
      severity,
      runId,
      runStatus,
      progress,
      progressDetail,
      report,
      samples,
      isRunning,
      activeTab,
    },
    actions: {
      setSelectedDataset,
      setSeverity,
      toggleAttack,
      handleRun,
      setActiveTab,
    }
  };
}
