"use client";

import { useState } from "react";
import { registerProjectCheckpoint, registerProjectDataset, uploadProjectArtifact } from "@/lib/api";

export default function ProjectAssetPicker({ projectId, taskId, kind, modelFamilyId, onComplete }) {
  const [message, setMessage] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !projectId) return;
    setIsUploading(true); setMessage("");
    try {
      const artifact = await uploadProjectArtifact(projectId, file, kind === "model" ? "CHECKPOINT" : "DATASET_BUNDLE");
      const registered = kind === "model" ? await registerProjectCheckpoint(projectId, artifact.id, taskId, modelFamilyId) : await registerProjectDataset(projectId, artifact.id, file.name, taskId);
      setMessage(kind === "model" ? "Đang chờ xác thực" : "Dataset đã được đăng ký");
      onComplete?.(registered);
    } catch (error) { setMessage(error.message || "Tải tệp thất bại"); }
    finally { setIsUploading(false); }
  };
  return <div><label className="text-sm font-medium" htmlFor={`project-${kind}-file`}>{kind === "model" ? "Tệp mô hình" : "Gói dữ liệu"}</label><input id={`project-${kind}-file`} className="mt-2 block" type="file" disabled={isUploading || !projectId} onChange={handleFileChange} />{message && <p role="alert" className="mt-2 text-sm">{message}</p>}</div>;
}
