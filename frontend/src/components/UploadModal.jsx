"use client";

import React, { useState } from "react";
import { uploadImage, importFolderDataset } from "@/lib/api";

export default function UploadModal({ isOpen, onClose, onDatasetCreated, datasets = [] }) {
  const [activeTab, setActiveTab] = useState("files"); // "files" | "folder" | "catalog" | "url"
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [filePreviews, setFilePreviews] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  // Folder Import State
  const [folderPath, setFolderPath] = useState("data/anonymized/kitti");
  const [datasetName, setDatasetName] = useState("Custom Folder Import");
  const [inputFormat, setInputFormat] = useState("advertest");
  const [maxSamples, setMaxSamples] = useState(50);

  // URL Ingestion State
  const [imageUrl, setImageUrl] = useState("");

  if (!isOpen) return null;

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setSelectedFiles((prev) => [...prev, ...files]);

    const newPreviews = files.map((file) => ({
      name: file.name,
      size: (file.size / 1024).toFixed(1) + " KB",
      url: URL.createObjectURL(file),
      file,
    }));
    setFilePreviews((prev) => [...prev, ...newPreviews]);
  };

  const removeFile = (index) => {
    if (filePreviews[index]?.url) {
      URL.revokeObjectURL(filePreviews[index].url);
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
    setStatusMessage("Uploading images to server...");

    try {
      const batchId = `batch-${Date.now()}`;
      const results = [];
      for (const file of selectedFiles) {
        const uploaded = await uploadImage(file, batchId);
        results.push(uploaded);
      }
      setStatusMessage(`Successfully uploaded ${results.length} images!`);
      const uploadId = `upload-${Date.now()}`;
      const customDataset = {
        id: uploadId,
        name: uploadId,
        dataset: "folder_dataset",
        title: `📤 Uploaded Raw Images (${results.length})`,
        dataset_params: results[0]?.dataset_params || {
          root: `data/uploads/${batchId}`,
          input_format: "advertest",
        },
        anonymized: false,
        benchmark_ready: false,
      };
      onDatasetCreated(customDataset);
      setTimeout(() => {
        setIsUploading(false);
        onClose();
      }, 1000);
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

    try {
      const res = await importFolderDataset({
        root: folderPath,
        name: datasetName || "Custom Import",
        inputFormat,
        maxSamples,
      });
      const folderId = res.version_id || res.id || `folder-${Date.now()}`;
      setStatusMessage(`Folder imported successfully as dataset '${folderId}'!`);
      const customDataset = {
        id: folderId,
        name: folderId,
        dataset: "folder_dataset",
        title: `📂 ${datasetName || res.name || folderPath}`,
        dataset_params: {
          root: folderPath,
          input_format: inputFormat,
          anonymization_manifest: "dataset.json",
        },
        anonymized: true,
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
            <p className="text-xs text-secondary mt-1">Select or upload images/datasets for robustness attack testing</p>
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
            { id: "folder", label: "📂 Folder Path Import" },
            { id: "catalog", label: "📊 Preset Catalog" },
            { id: "url", label: "🌐 Direct Image URL" },
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

        {/* TAB 1: Batch Image File Upload */}
        {activeTab === "files" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "12px", overflowY: "auto" }}>
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
                accept="image/*"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />
            </label>

            {filePreviews.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: "6px", maxHeight: "180px", overflowY: "auto" }}>
                <div className="text-xs font-bold text-secondary">Selected Files ({filePreviews.length}):</div>
                {filePreviews.map((item, idx) => (
                  <div
                    key={idx}
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
              {isUploading ? "Uploading..." : `Upload & Attack ${selectedFiles.length} Images`}
            </button>
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
              <label className="config-panel__label">Folder Directory Path</label>
              <input
                type="text"
                className="select-field"
                value={folderPath}
                onChange={(e) => setFolderPath(e.target.value)}
                placeholder="e.g. data/anonymized/kitti or C:\Dataset\KITTI"
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
              disabled={isUploading || !folderPath}
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

        {/* TAB 4: Direct URL Ingestion */}
        {activeTab === "url" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "14px" }}>
            <div>
              <label className="config-panel__label">Direct Image URL</label>
              <input
                type="url"
                className="select-field"
                placeholder="https://example.com/sample_image.jpg"
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
              />
            </div>

            {imageUrl && (
              <div className="glass-panel" style={{ padding: "12px", textAlign: "center" }}>
                <div className="text-xs text-secondary mb-2">Image Preview</div>
                <img
                  src={imageUrl}
                  alt="preview"
                  style={{ maxHeight: "160px", maxWidth: "100%", objectFit: "contain", borderRadius: "var(--radius-sm)" }}
                  onError={() => setErrorMessage("Could not load image preview from URL")}
                />
              </div>
            )}

            <button
              type="button"
              className="run-button"
              disabled={!imageUrl}
              onClick={() => {
                onDatasetCreated({ id: `url-${Date.now()}`, name: "Direct URL Image", url: imageUrl });
                onClose();
              }}
              style={{ marginTop: "auto" }}
            >
              Use Image URL
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
