"use client";

import React, { useEffect, useRef, useState } from "react";
import { createUploadBatch, uploadImage, startFolderDatasetImport, getFolderDatasetImportJob } from "@/lib/api";
import ManualAnnotationPanel from "@/components/ManualAnnotationPanel.jsx";

export default function UploadModal({ isOpen, onClose, onDatasetCreated, datasets = [], taskId = "detection2d" }) {
  const [activeTab, setActiveTab] = useState("files");
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [filePreviews, setFilePreviews] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [batchClassMap, setBatchClassMap] = useState("car:Car");
  const [batchAnonymized, setBatchAnonymized] = useState(false);
  const [uploadedBatch, setUploadedBatch] = useState(null);
  const [datasetKind, setDatasetKind] = useState("clean");
  const [pairedManifest, setPairedManifest] = useState('{"clean_dataset_version_id":"","pairs":[],"attack_name":"","attack_version":"","severity":3,"seed":42,"source_hash":"","ground_truth_hash":""}');
  const previewUrls = useRef(new Set());

  // Folder Import State
  const [folderPath, setFolderPath] = useState("data/anonymized/kitti-de");
  const [datasetName, setDatasetName] = useState("Custom Folder Import");
  const [logicalSourceId, setLogicalSourceId] = useState("folder-import");
  const [inputFormat, setInputFormat] = useState("advertest");
  const [maxSamples, setMaxSamples] = useState(50);
  const [selectedPretrainedModel, setSelectedPretrainedModel] = useState("yolo11n");

  useEffect(() => () => previewUrls.current.forEach((url) => URL.revokeObjectURL(url)), []);

  if (!isOpen) return null;

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    const supported = new Set(["image/jpeg", "image/png", "image/webp", "image/bmp"]);
    const invalid = files.find((file) => !supported.has(file.type) || file.size > 25 * 1024 * 1024);
    if (invalid) {
      setErrorMessage(`${invalid.name} must be a JPG, PNG, WEBP, or BMP no larger than 25 MB.`);
      return;
    }
    setErrorMessage("");
    setSelectedFiles((prev) => [...prev, ...files]);

    const newPreviews = files.map((file) => {
      const url = URL.createObjectURL(file);
      previewUrls.current.add(url);
      return { name: file.name, size: (file.size / 1024).toFixed(1) + " KB", url, file };
    });
    setFilePreviews((prev) => [...prev, ...newPreviews]);
  };

  const removeFile = (index) => {
    if (filePreviews[index]?.url) {
      URL.revokeObjectURL(filePreviews[index].url);
      previewUrls.current.delete(filePreviews[index].url);
    }
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setFilePreviews((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUploadFiles = async () => {
    if (!selectedFiles.length) {
      setErrorMessage("Please select at least one image file to upload.");
      return;
    }
    setIsUploading(true);
    setErrorMessage("");
    setStatusMessage("Creating task-bound upload batch...");
    setUploadProgress(0);

    try {
      const classMap = Object.fromEntries(batchClassMap.split(/[\n,]+/).map((value) => value.trim()).filter(Boolean).map((value) => {
        const [id, label] = value.split(":", 2).map((part) => part.trim());
        return [id, label || id];
      }));
      let attackedManifest = null;
      if (datasetKind === "attacked_paired") {
        try { attackedManifest = JSON.parse(pairedManifest); }
        catch { throw new Error("Paired manifest must be valid JSON."); }
      }
      const batch = await createUploadBatch({
        display_name: datasetName || "Browser upload",
        task_id: taskId,
        class_map: classMap,
        anonymized: batchAnonymized,
        dataset_kind: datasetKind,
        attacked_manifest: attackedManifest,
      });
      const batchId = batch.batch_id;
      setStatusMessage("Uploading images to server...");
      const results = [];
      const totalBytes = selectedFiles.reduce((sum, file) => sum + file.size, 0);
      let completedBytes = 0;
      for (const [index, file] of selectedFiles.entries()) {
        const sampleId = `sample-${String(index + 1).padStart(4, "0")}`;
        const uploaded = await uploadImage(file, batchId, taskId, sampleId, (ratio) => {
          setUploadProgress(Math.round(((completedBytes + file.size * ratio) / totalBytes) * 100));
        });
        results.push(uploaded);
        completedBytes += file.size;
        setUploadProgress(Math.round((completedBytes / totalBytes) * 100));
      }
      setUploadedBatch({ ...batch, samples: Object.fromEntries(results.map((item) => [item.sample_id, item])) });
      setStatusMessage(`Uploaded ${results.length} image${results.length === 1 ? "" : "s"}. Add labels to enable a benchmark.`);
      const upload = results[0];
      const customDataset = {
        id: upload.batch_id,
        name: upload.batch_id,
        dataset: upload.dataset,
        title: `📤 Uploaded Raw Images (${results.length})`,
        dataset_params: upload.dataset_params,
        status: upload.status,
        sample_count: results.length,
        anonymized: upload.anonymized,
        annotation_status: upload.annotation_status,
        benchmark_ready: upload.benchmark_ready,
      };
      onDatasetCreated(customDataset);
      setIsUploading(false);
    } catch (err) {
      setIsUploading(false);
      setErrorMessage(err.message || "Failed to upload image files.");
    }
  };

  const handleImportFolder = async () => {
    if (!folderPath) {
      setErrorMessage("Please provide a valid folder path.");
      return;
    }
    setIsUploading(true);
    setErrorMessage("");
    setStatusMessage("Validating & importing local folder...");
    setUploadProgress(0);

    try {
      const queued = await startFolderDatasetImport({
        root: folderPath,
        name: datasetName || "Custom Import",
        logicalSourceId: logicalSourceId.trim() || `folder-${folderPath.trim().replace(/[^a-zA-Z0-9_-]+/g, "-")}`,
        inputFormat,
        maxSamples,
        taskId,
      });
      let res;
      for (let attempt = 0; attempt < 240; attempt += 1) {
        const job = await getFolderDatasetImportJob(queued.job_id);
        setUploadProgress(Math.round((job.progress_ratio || 0) * 100));
        setStatusMessage(job.detail || `Importing dataset: ${Math.round((job.progress_ratio || 0) * 100)}%`);
        if (job.state === "COMPLETED") { res = job.result; break; }
        if (job.state === "FAILED" || job.state === "CANCELLED") throw new Error(job.error || "Dataset import did not complete.");
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      if (!res) throw new Error("Dataset import is still running. Reopen this upload manager to reconnect to its job.");
      const folderId = res.version_id || res.id;
      setStatusMessage(`Folder imported successfully as dataset '${folderId}'!`);
      const customDataset = {
        id: folderId,
        name: folderId,
        dataset: "folder_dataset",
        title: `📂 ${datasetName || res.name || folderPath}`,
        dataset_params: res.dataset_params,
        anonymized: res.anonymized,
        annotation_status: res.annotation_status,
        benchmark_ready: res.benchmark_ready,
      };
      onDatasetCreated(customDataset);
      setTimeout(() => {
        setIsUploading(false);
        onClose();
      }, 1200);
    } catch (err) {
      setIsUploading(false);
      setErrorMessage(err.message || "Failed to import folder directory.");
    }
  };

  const handleSelectCatalog = (ds) => {
    onDatasetCreated({
      ...ds,
      dataset: ds.name || ds.id,
      title: ds.title || ds.name || ds.id,
      dataset_params: ds.dataset_params || {},
    });
    onClose();
  };

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.75)",
        backdropFilter: "blur(6px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 9999,
      }}
      onClick={onClose}
    >
      <div
        className="glass-panel"
        style={{
          width: "650px",
          maxWidth: "92vw",
          maxHeight: "85vh",
          display: "flex",
          flexDirection: "column",
          borderRadius: "var(--radius-lg)",
          border: "1px solid var(--border-subtle)",
          padding: "var(--space-lg)",
          backgroundColor: "#12141c",
          boxShadow: "0 20px 50px rgba(0,0,0,0.8)",
          overflow: "hidden",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
          <div>
            <h2 style={{ fontSize: "1.2rem", margin: 0, fontWeight: 700, color: "var(--text-primary)" }}>
              Data & Image Ingestion Manager
            </h2>
            <p className="text-xs text-secondary mt-1">Task contract: {taskId === "detection2d" ? "image + 2D boxes" : taskId === "segmentation" ? "image + instance masks/polygons" : "calibration + LiDAR + 3D boxes"}</p>
          </div>
          <button
            type="button"
            className="tab-bar__item"
            onClick={onClose}
            style={{ padding: "4px 10px", fontSize: "1rem", cursor: "pointer" }}
          >
            ✕
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="tab-bar" style={{ marginBottom: "16px", gap: "6px" }}>
          {[
            { id: "files", label: "📁 Batch Image Upload" },
            { id: "attacked", label: "🛡️ Attacked Dataset" },
            { id: "folder", label: "📂 Folder Path Import" },
            { id: "catalog", label: "📊 Preset Catalog" },
            { id: "ultralytics", label: "⚡ Import Pretrained (YOLO)" },
          ].map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`tab-bar__item ${activeTab === tab.id ? "tab-bar__item--active" : ""}`}
              onClick={() => {
                setActiveTab(tab.id);
                setErrorMessage("");
                setStatusMessage("");
              }}
              style={{ fontSize: "0.8rem", padding: "8px 12px" }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Status & Errors */}
        {errorMessage && (
          <div className="empty-state" role="alert" style={{ padding: "8px 12px", marginBottom: "12px", background: "rgba(239, 68, 68, 0.15)" }}>
            <span className="text-xs text-danger">{errorMessage}</span>
          </div>
        )}
        {statusMessage && (
          <div className="empty-state" style={{ padding: "8px 12px", marginBottom: "12px", background: "rgba(16, 185, 129, 0.15)" }}>
            <span className="text-xs text-success">{statusMessage}</span>
          </div>
        )}
        {isUploading && <progress aria-label="Upload progress" max="100" value={uploadProgress} style={{ width: "100%", marginBottom: "12px" }} />}

        {/* TAB 1: Batch Image File Upload */}
        {activeTab === "files" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "12px", overflowY: "auto" }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "10px", alignItems: "end" }}>
              <label className="config-panel__label">Classes (id:display name)
                <input aria-label="Class map" className="select-field" value={batchClassMap} onChange={(event) => setBatchClassMap(event.target.value)} />
              </label>
              <label className="text-xs text-secondary" style={{ display: "flex", gap: "6px", alignItems: "center", paddingBottom: "8px" }}>
                <input type="checkbox" checked={batchAnonymized} onChange={(event) => setBatchAnonymized(event.target.checked)} />
                Anonymised
              </label>
            </div>
            <label
              htmlFor="batch-file-input"
              style={{
                border: "2px dashed var(--border-subtle)",
                borderRadius: "var(--radius-md)",
                padding: "24px",
                textAlign: "center",
                cursor: "pointer",
                background: "rgba(255, 255, 255, 0.02)",
                transition: "border-color 0.2s",
              }}
            >
              <div style={{ fontSize: "1.8rem", marginBottom: "8px" }}>📸</div>
              <div className="font-bold text-sm text-primary">Click or Drag & Drop Images Here</div>
              <div className="text-xs text-tertiary mt-1">Supports JPG, PNG, WEBP, BMP up to 25 MB per file</div>
              <input
                id="batch-file-input"
                type="file"
                multiple
                accept="image/jpeg,image/png,image/webp,image/bmp"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />
            </label>

            {filePreviews.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: "6px", maxHeight: "180px", overflowY: "auto" }}>
                <div className="text-xs font-bold text-secondary">Selected Files ({filePreviews.length}):</div>
                {filePreviews.map((item, idx) => (
                  <div
                    key={item.url || item.name || idx}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "6px 10px",
                      background: "var(--bg-primary)",
                      borderRadius: "var(--radius-sm)",
                      fontSize: "0.8rem",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", overflow: "hidden" }}>
                      {/* eslint-disable-next-line @next/next/no-img-element -- object URLs are local, short-lived upload previews. */}
                      <img src={item.url} alt="thumb" style={{ width: 28, height: 28, objectFit: "cover", borderRadius: 4 }} />
                      <span className="text-primary truncate" style={{ maxWidth: "240px" }}>{item.name}</span>
                      <span className="text-tertiary text-xs">({item.size})</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => removeFile(idx)}
                      style={{ background: "none", border: "none", color: "var(--danger)", cursor: "pointer" }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}

            <button
              type="button"
              className="run-button"
              disabled={isUploading || selectedFiles.length === 0}
              onClick={handleUploadFiles}
              style={{ marginTop: "auto" }}
            >
              {isUploading ? `Uploading ${uploadProgress}%` : `Upload ${selectedFiles.length} Files`}
            </button>
            {uploadedBatch && <p className="text-xs text-secondary" role="status">
              {taskId === "detection3d" ? "3D labels must be supplied with calibration and LiDAR; a cuboid editor is intentionally unavailable." : "Uploaded without labels remains quick-inference only. Choose Label manually to enable benchmark finalization."}
            </p>}
            {uploadedBatch && taskId !== "detection3d" && <ManualAnnotationPanel
              batch={uploadedBatch}
              previews={Object.fromEntries(Object.keys(uploadedBatch.samples).map((sampleId, current) => [sampleId, filePreviews[current]]))}
              onFinalized={(result) => {
                const version = result.dataset_version;
                onDatasetCreated({
                  id: version.version_id,
                  name: version.name,
                  dataset: "folder_dataset",
                  title: version.name,
                  dataset_params: version.dataset_params,
                  benchmark_ready: true,
                  anonymized: version.anonymized,
                  task_id: version.task_id,
                  dataset_kind: version.dataset_kind,
                });
                setStatusMessage("Dataset finalized and benchmark-ready.");
              }}
            />}
          </div>
        )}

        {activeTab === "attacked" && (
          <div style={{ flex: 1, display: "grid", gap: "12px", overflowY: "auto" }}>
            <p className="text-xs text-secondary">Choose a browser directory or individual files; no path on the API server is required for this flow.</p>
            <label className="attack-card" style={{ cursor: "pointer" }}>
              Choose attacked-data directory
              <input type="file" multiple webkitdirectory="" directory="" style={{ display: "none" }} onChange={(event) => { handleFileChange(event); setActiveTab("files"); }} />
            </label>
            <label className="config-panel__label">Attack dataset kind
              <select aria-label="Attack dataset kind" className="select-field" value={datasetKind} onChange={(event) => setDatasetKind(event.target.value)}>
                <option value="attacked_standalone">Standalone — inference/training only</option>
                <option value="attacked_paired">Paired — clean versus attacked comparison</option>
              </select>
            </label>
            {datasetKind === "attacked_paired" && <label className="config-panel__label">Paired manifest JSON
              <textarea aria-label="Paired manifest" className="select-field" rows="9" value={pairedManifest} onChange={(event) => setPairedManifest(event.target.value)} />
              <span className="text-xs text-tertiary">Include clean dataset version, clean/attacked sample mapping, attack provenance, source hash and ground-truth hash.</span>
            </label>}
            <button type="button" className="run-button" onClick={() => setActiveTab("files")}>Continue to upload files</button>
          </div>
        )}

        {/* TAB 2: Folder Path Import */}
        {activeTab === "folder" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "14px", overflowY: "auto" }}>
            <div>
              <label className="config-panel__label">Dataset Name</label>
              <input
                type="text"
                className="select-field"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                placeholder="e.g. KITTI Local Import v1"
              />
            </div>

            <div>
              <label className="config-panel__label">Logical Source ID</label>
              <input
                type="text"
                className="select-field"
                value={logicalSourceId}
                onChange={(e) => setLogicalSourceId(e.target.value)}
                placeholder="Keep this stable when importing a revision"
              />
            </div>

            <div>
              <label className="config-panel__label">Folder Directory Path</label>
              <input
                type="text"
                className="select-field"
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                placeholder="e.g. data/anonymized/kitti-de or C:\Dataset\KITTI"
              />
              <span className="text-xs text-tertiary mt-1 block">Specify the local workspace relative or absolute directory path</span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
              <div>
                <label className="config-panel__label">Annotation Format</label>
                <select className="select-field" value={inputFormat} onChange={(e) => setInputFormat(e.target.value)}>
                  <option value="advertest">AdverTest Annotated Folder</option>
                  <option value="kitti">KITTI Format</option>
                </select>
              </div>

              <div>
                <label className="config-panel__label">Sample Limit ({maxSamples})</label>
                <input
                  type="range"
                  min={5}
                  max={500}
                  step={5}
                  value={maxSamples}
                  onChange={(e) => setMaxSamples(Number(e.target.value))}
                  style={{ width: "100%", marginTop: "8px" }}
                />
              </div>
            </div>

            <button
              type="button"
              className="run-button"
              disabled={isUploading || !folderPath || taskId !== "detection2d"}
              onClick={handleImportFolder}
              style={{ marginTop: "auto" }}
            >
              {isUploading ? "Importing..." : "Import Directory & Register Dataset"}
            </button>
          </div>
        )}

        {/* TAB 3: Catalog Presets */}
        {activeTab === "catalog" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "10px", overflowY: "auto" }}>
            <div className="text-xs text-secondary mb-1">Select from pre-indexed anonymized datasets in the catalog:</div>
            {datasets.map((ds) => (
              <div
                key={ds.id}
                className="glass-panel"
                style={{
                  padding: "12px 14px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  cursor: "pointer",
                  border: "1px solid var(--border-subtle)",
                }}
                onClick={() => handleSelectCatalog(ds)}
              >
                <div>
                  <div className="font-bold text-sm text-primary">{ds.name || ds.id}</div>
                  <div className="text-xs text-tertiary mt-1">
                    Format: {ds.format || "anonymized"} • Samples: {ds.sample_count || 100} • License: {ds.license || "CC-BY"}
                  </div>
                </div>
                <button type="button" className="attack-card" style={{ padding: "6px 12px" }}>
                  Select
                </button>
              </div>
            ))}
          </div>
        )}

        {/* TAB 4: Ultralytics Pretrained */}
        {activeTab === "ultralytics" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "14px" }}>
            <div className="text-xs text-secondary">
              Directly download, SHA-256 hash, and sandbox-validate an official pretrained YOLO checkpoint from Ultralytics:
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
              {[
                { id: "yolo11n", name: "YOLO11 Nano", size: "2.6M params", speed: "Ultra Fast" },
                { id: "yolo11s", name: "YOLO11 Small", size: "9.4M params", speed: "Balanced" },
                { id: "yolo11m", name: "YOLO11 Medium", size: "20.1M params", speed: "Accurate" },
                { id: "yolo11l", name: "YOLO11 Large", size: "25.3M params", speed: "High Capacity" },
                { id: "yolo11x", name: "YOLO11 XLarge", size: "56.9M params", speed: "Max Precision" },
              ].map((m) => (
                <div
                  key={m.id}
                  onClick={() => setSelectedPretrainedModel(m.id)}
                  style={{
                    padding: "12px",
                    borderRadius: "var(--radius-md)",
                    border: `1px solid ${selectedPretrainedModel === m.id ? "var(--primary)" : "var(--border-subtle)"}`,
                    background: selectedPretrainedModel === m.id ? "rgba(99, 102, 241, 0.15)" : "rgba(15, 23, 42, 0.4)",
                    cursor: "pointer",
                    transition: "all 0.15s ease",
                  }}
                >
                  <div className="font-bold text-sm text-primary">{m.name}</div>
                  <div className="text-xs text-tertiary mt-0.5">{m.size} • {m.speed}</div>
                </div>
              ))}
            </div>

            <button
              type="button"
              className="run-button"
              disabled={isUploading}
              onClick={async () => {
                setIsUploading(true);
                setErrorMessage("");
                setStatusMessage(`Importing and validating ${selectedPretrainedModel}...`);
                try {
                  const res = await fetch("/api/v1/checkpoints/import/ultralytics", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ model_id: selectedPretrainedModel }),
                  });
                  if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Failed to import checkpoint");
                  }
                  const data = await res.json();
                  setStatusMessage(`Checkpoint ${data.checkpoint_id || selectedPretrainedModel} registered and READY!`);
                  if (onDatasetCreated) onDatasetCreated();
                  setTimeout(() => onClose(), 1200);
                } catch (err) {
                  setErrorMessage(err.message);
                } finally {
                  setIsUploading(false);
                }
              }}
              style={{ marginTop: "auto" }}
            >
              {isUploading ? "Importing Checkpoint..." : `Import ${selectedPretrainedModel} & Validate`}
            </button>
          </div>
        )}

      </div>
    </div>
  );
}
