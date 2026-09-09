"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import Button from "@/components/common/Button";
import ProjectAssetPicker from "@/components/ProjectAssetPicker";
import { getModelFamilies } from "@/lib/api";

const TASK_LABELS = [
  { id: "detection2d", label: "Object Detection (2D)" },
  { id: "segmentation", label: "Segmentation" },
  { id: "detection3d", label: "3D Detection" },
];

/**
 * Project-scoped quick upload used by the dashboard quick actions.
 *
 * Reuses the same contract as the experiments page's asset pickers: the file
 * goes through the resumable artifact-upload-session API and is then
 * registered as a project checkpoint / dataset version.  Uploads are
 * project-scoped, so without an active project only a hint is shown.
 */
export default function QuickUploadDialog({ kind, project, onClose, onSuccess }) {
  const [taskId, setTaskId] = useState(project?.task_type || "detection2d");
  const [families, setFamilies] = useState([]);
  const [familyId, setFamilyId] = useState("");
  const [done, setDone] = useState(false);
  const isModel = kind === "model";

  useEffect(() => {
    if (!isModel || !project) return undefined;
    let active = true;
    getModelFamilies(taskId)
      .then((rows) => {
        if (!active) return;
        setFamilies(rows);
        setFamilyId((prev) => (rows.some((row) => row.id === prev) ? prev : rows[0]?.id || ""));
      })
      .catch(() => setFamilies([]));
    return () => {
      active = false;
    };
  }, [isModel, project, taskId]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Clickable backdrop closes the dialog; the panel sits above it, so
          clicks on the panel never reach this button. */}
      <button
        type="button"
        aria-label="Đóng"
        className="absolute inset-0 bg-slate-900/70 backdrop-blur-sm"
        onClick={onClose}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={isModel ? "Nạp mô hình" : "Nạp dataset"}
        className="relative z-10 max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-xl border border-slate-200 bg-white p-5 shadow-xl"
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-bold text-slate-900">{isModel ? "Nạp mô hình" : "Nạp dataset"}</h2>
            <p className="mt-0.5 text-xs text-slate-500">
              Upload dùng API project-scoped; server quyết định validation và trạng thái READY.
            </p>
          </div>
          <button type="button" aria-label="Đóng" onClick={onClose} className="text-slate-400 hover:text-slate-600">
            ✕
          </button>
        </div>

        {!project ? (
          <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-600">
            Chưa chọn project. Hãy{" "}
            <Link href="/experiments/new" className="font-semibold text-blue-600">
              tạo project
            </Link>{" "}
            trước khi nạp asset — mọi upload đều thuộc về một project.
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-xs text-slate-500">
              Project: <span className="font-semibold text-slate-700">{project.name}</span>
            </p>
            <label className="block text-xs font-medium text-slate-700">
              Bài toán
              <select
                className="mt-1 w-full rounded-md border border-slate-300 p-2 text-sm"
                value={taskId}
                onChange={(event) => {
                  setTaskId(event.target.value);
                  setDone(false);
                }}
              >
                {TASK_LABELS.map((task) => (
                  <option key={task.id} value={task.id}>
                    {task.label}
                  </option>
                ))}
              </select>
            </label>
            {isModel && (
              <label className="block text-xs font-medium text-slate-700">
                Kiến trúc mô hình (model family)
                <select
                  className="mt-1 w-full rounded-md border border-slate-300 p-2 text-sm"
                  value={familyId}
                  onChange={(event) => {
                    setFamilyId(event.target.value);
                    setDone(false);
                  }}
                >
                  {families.length === 0 && <option value="">Đang tải family…</option>}
                  {families.map((family) => (
                    <option key={family.id} value={family.id}>
                      {family.display_name}
                      {family.runnable ? "" : " — chưa chạy được"}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <div className="text-xs font-medium text-slate-700">
              {isModel ? "Tệp mô hình (.pt / .pth / .onnx / .ckpt)" : "Gói dữ liệu (.zip / .tar.gz)"}
              <ProjectAssetPicker
                projectId={project.id}
                taskId={taskId}
                kind={kind}
                modelFamilyId={isModel ? familyId : undefined}
                onComplete={() => {
                  setDone(true);
                  onSuccess?.();
                }}
              />
            </div>
            {done && (
              <p
                role="status"
                className="rounded-md border border-emerald-200 bg-emerald-50 p-2 text-xs text-emerald-700"
              >
                {isModel
                  ? "Đã đăng ký checkpoint — server đang xác thực. Xem trạng thái trong trang thí nghiệm."
                  : "Dataset đã được đăng ký vào project."}
              </p>
            )}
            <Button variant="outline" size="sm" className="w-full justify-center" onClick={onClose}>
              Đóng
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
