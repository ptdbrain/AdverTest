"use client";

import { useCallback, useEffect, useState } from "react";
import {
  downloadProjectArtifact,
  listProjectCheckpoints,
  listProjectDatasetVersions,
  registerProjectCheckpoint,
  registerProjectDataset,
  uploadProjectArtifact,
} from "@/lib/api";

export default function ProjectAssetPicker({
  projectId,
  taskId,
  kind,
  modelFamilyId,
  selectedId,
  onSelect,
  onAssetsChange,
  onComplete,
}) {
  const [message, setMessage] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [assets, setAssets] = useState([]);

  const refreshAssets = useCallback(async () => {
    if (!projectId) {
      setAssets([]);
      onAssetsChange?.([]);
      return [];
    }
    const items =
      kind === "model" ? await listProjectCheckpoints(projectId) : await listProjectDatasetVersions(projectId);
    const next = Array.isArray(items) ? items : [];
    setAssets(next);
    onAssetsChange?.(next);
    return next;
  }, [kind, onAssetsChange, projectId]);

  useEffect(() => {
    refreshAssets().catch((error) => setMessage(error.message || "Không thể tải danh sách asset"));
  }, [refreshAssets]);

  useEffect(() => {
    if (!assets.some((asset) => !["READY", "REJECTED", "FAILED_VALIDATION", "UNSUPPORTED"].includes(asset.status))) {
      return undefined;
    }
    const refreshId = window.setTimeout(() => {
      refreshAssets().catch((error) => setMessage(error.message || "Không thể cập nhật trạng thái asset"));
    }, 2000);
    return () => window.clearTimeout(refreshId);
  }, [assets, refreshAssets]);

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !projectId) return;
    setIsUploading(true);
    setMessage("");
    try {
      const artifact = await uploadProjectArtifact(projectId, file, kind === "model" ? "checkpoint" : "dataset_bundle");
      const registered =
        kind === "model"
          ? await registerProjectCheckpoint(projectId, artifact.id, taskId, modelFamilyId)
          : await registerProjectDataset(projectId, artifact.id, file.name, taskId);
      await refreshAssets();
      setMessage(kind === "model" ? "Đang chờ xác thực" : "Dataset đã được đăng ký");
      onComplete?.(registered);
    } catch (error) {
      setMessage(error.message || "Tải tệp thất bại");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div>
      <label className="text-sm font-medium" htmlFor={`project-${kind}-file`}>
        {kind === "model" ? "Tệp mô hình" : "Gói dữ liệu"}
      </label>
      <input
        id={`project-${kind}-file`}
        className="mt-2 block"
        type="file"
        disabled={isUploading || !projectId}
        onChange={handleFileChange}
      />
      {message && (
        <p role="alert" className="mt-2 text-sm">
          {message}
        </p>
      )}
      {assets.length > 0 && (
        <ul className="mt-3 space-y-2" aria-label={kind === "model" ? "Mô hình của project" : "Dataset của project"}>
          {assets.map((asset) => {
            const name = asset.display_name || asset.id;
            const runnable = asset.status === "READY" && (asset.runnable ?? true);
            return (
              <li key={asset.id} className="flex items-center gap-2 rounded border border-slate-200 p-2 text-xs">
                <button
                  type="button"
                  className="min-w-0 flex-1 text-left disabled:cursor-not-allowed disabled:text-slate-400"
                  disabled={!runnable}
                  aria-pressed={selectedId === asset.id}
                  title={runnable ? `Chọn ${name}` : asset.blocked_reason || asset.status}
                  onClick={() => onSelect?.(asset)}
                >
                  <span className="block truncate font-semibold">{name}</span>
                  <span className="block text-[10px]">{runnable ? "Sẵn sàng" : asset.blocked_reason || asset.status}</span>
                </button>
                {asset.artifact_id && (
                  <button
                    type="button"
                    className="rounded border border-slate-300 px-2 py-1 font-semibold disabled:cursor-not-allowed disabled:opacity-50"
                    disabled={asset.status !== "READY"}
                    aria-label={`Tải ${name}`}
                    onClick={() => downloadProjectArtifact(projectId, asset.artifact_id)}
                  >
                    Tải xuống
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
