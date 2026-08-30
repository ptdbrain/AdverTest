"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Check,
  CheckCircle2,
  Copy,
  Cpu,
  Download,
  Lock,
  RefreshCw,
  ShieldCheck,
  Upload,
} from "lucide-react";

import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import Card from "@/components/common/Card";
import DefenseTargetSelector from "@/components/DefenseTargetSelector";
import PageHeader from "@/components/layout/PageHeader";
import {
  createDefenceRun,
  getCheckpoint,
  getRun,
  getRunDefenceCandidates,
  listSessions,
  uploadCheckpoint,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const strategies = [
  ["adversarial_training", "Adversarial Training", "Đưa mẫu tấn công đã đo vào từng epoch."],
  ["augmentation_mix", "Augmentation Mix", "Trộn corruption phù hợp với run mục tiêu."],
  ["robust_finetune", "Robust Fine-tuning", "Giữ hiệu năng clean trong quá trình fine-tune."],
];

function messageOf(error, fallback) {
  return error?.message || fallback;
}

export default function DefensePage() {
  const [sessions, setSessions] = useState([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [sessionsError, setSessionsError] = useState("");
  const [selectedSessionId, setSelectedSessionId] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [baselineRun, setBaselineRun] = useState(null);
  const [baselineLoading, setBaselineLoading] = useState(false);
  const [baselineError, setBaselineError] = useState("");
  const [candidates, setCandidates] = useState([]);

  const [strategy, setStrategy] = useState("adversarial_training");
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);
  const [device, setDevice] = useState("cuda:0");
  const [isCopied, setIsCopied] = useState(false);

  const [candidateId, setCandidateId] = useState("");
  const [candidateFileName, setCandidateFileName] = useState("");
  const [uploadState, setUploadState] = useState("idle");
  const [uploadMessage, setUploadMessage] = useState("");
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationJob, setEvaluationJob] = useState(null);
  const [evaluationError, setEvaluationError] = useState("");

  const selectedSession = useMemo(
    () => sessions.find((item) => item.id === selectedSessionId) ?? null,
    [sessions, selectedSessionId]
  );
  const selectedRun = useMemo(
    () => selectedSession?.runs?.find((item) => item.id === selectedRunId) ?? null,
    [selectedRunId, selectedSession]
  );
  const baselineConfig = baselineRun?.config
    ?? baselineRun?.report?.provenance?.run_config
    ?? null;
  const compatibleCandidates = useMemo(() => {
    const familyId = baselineConfig?.model_family_id;
    if (!familyId) return [];
    return candidates.filter((candidate) => candidate.model_family_id === familyId);
  }, [baselineConfig, candidates]);

  const backendBaselineId = selectedRun?.backend_run_id?.trim() ?? "";
  const lockedModelName = selectedSession?.model_name ?? "";
  const lockedModelId = baselineConfig?.checkpoint_id
    ?? baselineConfig?.model_version_id
    ?? selectedSession?.model_id
    ?? "";
  const lockedDatasetName = selectedSession?.dataset_name ?? "";
  const lockedDatasetId = baselineConfig?.dataset ?? selectedSession?.dataset_id ?? "";
  const lockedRecipe = selectedRun?.attack_components?.length
    ? selectedRun.attack_components.join(",")
    : selectedRun?.attack_type ?? "";
  const hasTarget = Boolean(selectedSession && selectedRun);
  const hasBackendBaseline = Boolean(backendBaselineId);
  const canUpload = Boolean(
    hasBackendBaseline
      && baselineConfig?.task_id
      && baselineConfig?.model_family_id
      && lockedModelId
  );
  const canEvaluate = Boolean(
    hasBackendBaseline
      && baselineRun?.status === "COMPLETED"
      && candidateId
      && !isEvaluating
      && uploadState !== "uploading"
  );

  const generatedCommand = hasTarget
    ? `python scripts/train_defence.py --model ${lockedModelId} --dataset ${lockedDatasetId} --recipe ${lockedRecipe} --strategy ${strategy} --epochs ${epochs} --batch-size ${batchSize} --lr ${learningRate} --output-dir weights/defended --device ${device}`
    : "Chọn phiên và attack run để sinh lệnh huấn luyện.";

  const loadSessions = useCallback(async () => {
    setSessionsLoading(true);
    setSessionsError("");
    try {
      const items = await listSessions();
      setSessions(Array.isArray(items) ? items : []);
    } catch (error) {
      setSessions([]);
      setSessionsError(messageOf(error, "Không thể tải danh sách phiên thử nghiệm."));
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    let cancelled = false;
    setBaselineRun(null);
    setCandidates([]);
    setBaselineError("");

    if (!backendBaselineId) {
      setBaselineLoading(false);
      return () => {
        cancelled = true;
      };
    }

    setBaselineLoading(true);
    Promise.all([getRun(backendBaselineId), getRunDefenceCandidates(backendBaselineId)])
      .then(([run, items]) => {
        if (cancelled) return;
        setBaselineRun(run);
        setCandidates(Array.isArray(items) ? items : []);
      })
      .catch((error) => {
        if (!cancelled) {
          setBaselineError(messageOf(error, "Không thể tải locked protocol của baseline run."));
        }
      })
      .finally(() => {
        if (!cancelled) setBaselineLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [backendBaselineId]);

  const clearDownstreamState = () => {
    setCandidateId("");
    setCandidateFileName("");
    setUploadState("idle");
    setUploadMessage("");
    setEvaluationJob(null);
    setEvaluationError("");
  };

  const handleSessionChange = (sessionId) => {
    setSelectedSessionId(sessionId);
    setSelectedRunId("");
    setBaselineRun(null);
    setCandidates([]);
    clearDownstreamState();
  };

  const handleRunChange = (runId) => {
    setSelectedRunId(runId);
    setBaselineRun(null);
    setCandidates([]);
    clearDownstreamState();
  };

  const handleCopyCommand = async () => {
    if (!hasTarget) return;
    await navigator.clipboard.writeText(generatedCommand);
    setIsCopied(true);
    window.setTimeout(() => setIsCopied(false), 2500);
  };

  const handleDownloadScript = (format) => {
    if (!hasTarget) return;
    const windows = format === "bat";
    const content = windows
      ? `@echo off\r\n${generatedCommand}\r\npause\r\n`
      : `#!/usr/bin/env bash\nset -euo pipefail\n${generatedCommand}\n`;
    const url = URL.createObjectURL(new Blob([content], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = windows ? "run_train_defence.bat" : "run_train_defence.sh";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleCandidateUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || !canUpload) return;

    setCandidateId("");
    setCandidateFileName(file.name);
    setUploadState("uploading");
    setUploadMessage("Đang tải checkpoint: 0%");
    setEvaluationJob(null);
    setEvaluationError("");

    try {
      const created = await uploadCheckpoint(
        file,
        {
          taskId: baselineConfig.task_id,
          familyId: baselineConfig.model_family_id,
          displayName: file.name,
          role: "fine_tuned",
          parentCheckpointId: lockedModelId,
        },
        (ratio) => setUploadMessage(`Đang tải checkpoint: ${Math.round(ratio * 100)}%`)
      );

      for (let attempt = 0; attempt < 60; attempt += 1) {
        const checkpoint = await getCheckpoint(created.checkpoint_id);
        if (checkpoint.status === "READY") {
          setCandidateId(created.checkpoint_id);
          setUploadState("ready");
          setUploadMessage(`Checkpoint đã xác thực: ${created.checkpoint_id}`);
          const refreshed = await getRunDefenceCandidates(backendBaselineId);
          setCandidates(Array.isArray(refreshed) ? refreshed : []);
          return;
        }
        if (["REJECTED", "FAILED_VALIDATION"].includes(checkpoint.status)) {
          setUploadState("error");
          setUploadMessage(`Checkpoint bị từ chối: ${checkpoint.validation_reason || checkpoint.status}`);
          return;
        }
        setUploadMessage("Đang quét cách ly và kiểm tra checkpoint...");
        await new Promise((resolve) => window.setTimeout(resolve, 600));
      }
      setUploadState("pending");
      setUploadMessage("Checkpoint vẫn đang được xác thực trong hàng đợi.");
    } catch (error) {
      setUploadState("error");
      setUploadMessage(messageOf(error, "Tải checkpoint thất bại."));
    }
  };

  const handleRunLockedEvaluation = async () => {
    if (!canEvaluate) return;
    setIsEvaluating(true);
    setEvaluationJob(null);
    setEvaluationError("");
    try {
      setEvaluationJob(await createDefenceRun(backendBaselineId, candidateId));
    } catch (error) {
      setEvaluationError(messageOf(error, "Không thể bắt đầu đánh giá phòng thủ."));
    } finally {
      setIsEvaluating(false);
    }
  };

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Huấn luyện phòng thủ & Đánh giá đối kháng"
        subtitle="Chọn đúng phiên và attack run trước khi cấu hình phòng thủ, sau đó đánh giá checkpoint mới trên cùng locked protocol."
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Huấn luyện phòng thủ" },
        ]}
      />

      <DefenseTargetSelector
        sessions={sessions}
        selectedSessionId={selectedSessionId}
        selectedRunId={selectedRunId}
        loading={sessionsLoading}
        error={sessionsError}
        onSessionChange={handleSessionChange}
        onRunChange={handleRunChange}
        onRetry={loadSessions}
      />

      <div className="flex flex-col gap-3 rounded-xl border border-blue-200 bg-blue-50/80 p-4 text-xs text-blue-900 md:flex-row md:items-center">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-700 text-blue-50">
          <Cpu className="h-4 w-4" aria-hidden="true" />
        </span>
        <div>
          <p className="font-bold">Huấn luyện cục bộ, đánh giá tập trung</p>
          <p className="mt-0.5 max-w-3xl text-[11px] leading-5 text-blue-800">
            Lệnh training dùng đúng model, dataset và attack của run đã chọn. Checkpoint phải qua validation trước khi đánh giá.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 items-start gap-5 xl:grid-cols-12">
        <div className="space-y-5 xl:col-span-7">
          <Card title="1. Cấu hình chiến lược phòng thủ" subtitle="Model, dataset và attack được khóa theo run đã chọn">
            <fieldset disabled={!hasTarget} className="space-y-4 text-xs disabled:opacity-60">
              <legend className="sr-only">Cấu hình chiến lược phòng thủ</legend>
              <div>
                <p className="mb-1.5 font-bold text-slate-700">Phương pháp phòng thủ</p>
                <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-3">
                  {strategies.map(([id, name, description]) => (
                    <label
                      key={id}
                      className={cn(
                        "cursor-pointer rounded-lg border p-3 transition-colors",
                        strategy === id
                          ? "border-blue-600 bg-blue-50 ring-1 ring-blue-500"
                          : "border-slate-200 bg-white hover:border-slate-300"
                      )}
                    >
                      <input
                        type="radio"
                        name="defense-strategy"
                        className="sr-only"
                        value={id}
                        checked={strategy === id}
                        onChange={(event) => setStrategy(event.target.value)}
                      />
                      <span className="block font-bold text-slate-800">{name}</span>
                      <span className="mt-1 block text-[11px] leading-4 text-slate-500">{description}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 gap-3 border-t border-slate-100 pt-3 sm:grid-cols-2 lg:grid-cols-3">
                <label className="font-medium text-slate-600">
                  Mô hình gốc
                  <input aria-label="Mô hình gốc" readOnly value={lockedModelName} className="mt-1 w-full rounded-lg border border-slate-300 bg-slate-100 p-2 font-mono text-xs text-slate-800" />
                </label>
                <label className="font-medium text-slate-600">
                  Dataset
                  <input aria-label="Dataset phòng thủ" readOnly value={lockedDatasetName} className="mt-1 w-full rounded-lg border border-slate-300 bg-slate-100 p-2 font-mono text-xs text-slate-800" />
                </label>
                <label className="font-medium text-slate-600">
                  Attack cần phòng thủ
                  <input aria-label="Attack cần phòng thủ" readOnly value={lockedRecipe} className="mt-1 w-full rounded-lg border border-slate-300 bg-slate-100 p-2 font-mono text-xs text-slate-800" />
                </label>
                <label className="font-medium text-slate-600">
                  Số epochs
                  <input type="number" min="1" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} className="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2 font-mono text-xs text-slate-800" />
                </label>
                <label className="font-medium text-slate-600">
                  Batch size
                  <input type="number" min="1" value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} className="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2 font-mono text-xs text-slate-800" />
                </label>
                <label className="font-medium text-slate-600">
                  Learning rate
                  <input type="number" min="0" step="0.0001" value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} className="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2 font-mono text-xs text-slate-800" />
                </label>
                <label className="font-medium text-slate-600">
                  Thiết bị
                  <select value={device} onChange={(event) => setDevice(event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 bg-white p-2 font-mono text-xs text-slate-800">
                    <option value="cuda:0">cuda:0 (NVIDIA GPU)</option>
                    <option value="cpu">cpu</option>
                  </select>
                </label>
              </div>
            </fieldset>
          </Card>

          <Card title="2. Lệnh huấn luyện theo run mục tiêu" subtitle="Các tham số provenance không thể sửa độc lập" headerAction={<Badge variant="primary">Locked Target</Badge>}>
            <div className="space-y-3 text-xs">
              <div className="relative rounded-xl border border-slate-800 bg-slate-950 p-3.5 pr-28 font-mono text-xs leading-relaxed text-emerald-400">
                <span className="select-none text-slate-500">$ </span>
                <span className="break-all">{generatedCommand}</span>
                <button type="button" disabled={!hasTarget} onClick={handleCopyCommand} className="absolute right-2.5 top-2.5 flex items-center gap-1 rounded border border-slate-700 bg-slate-800 px-2.5 py-1 font-sans text-xs text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50">
                  {isCopied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                  {isCopied ? "Đã copy" : "Copy lệnh"}
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant="secondary" size="sm" icon={Download} disabled={!hasTarget} onClick={() => handleDownloadScript("sh")}>Tải script Linux/WSL</Button>
                <Button variant="secondary" size="sm" icon={Download} disabled={!hasTarget} onClick={() => handleDownloadScript("bat")}>Tải script Windows</Button>
              </div>
            </div>
          </Card>
        </div>

        <div className="space-y-5 xl:col-span-5">
          <Card title="3. Checkpoint phòng thủ & đánh giá lại" subtitle="Chỉ chạy trên backend baseline đã hoàn tất">
            <div className="space-y-4 text-xs">
              <div className={cn("flex items-start gap-2 rounded-lg border p-3", hasBackendBaseline ? "border-amber-200 bg-amber-50 text-amber-900" : "border-slate-200 bg-slate-50 text-slate-700")}>
                <Lock className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                <div>
                  <p className="font-bold">Locked protocol</p>
                  {selectedRun ? (
                    hasBackendBaseline ? (
                      <p className="mt-1 leading-5">Baseline <code className="font-mono">{backendBaselineId}</code>, seed {selectedRun.seed ?? "không có dữ liệu"}, attack {lockedRecipe || "không có dữ liệu"}.</p>
                    ) : (
                      <p className="mt-1 font-semibold leading-5 text-red-700">Không có baseline backend có thể đối chiếu. Run hiển thị này chưa gắn evidence thực.</p>
                    )
                  ) : (
                    <p className="mt-1 leading-5">Chọn attack run để khóa protocol.</p>
                  )}
                </div>
              </div>

              {baselineLoading && <p role="status" className="flex items-center gap-2 text-slate-600"><RefreshCw className="h-4 w-4 animate-spin" />Đang tải baseline backend...</p>}
              {baselineError && <p role="alert" className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-red-800"><AlertCircle className="h-4 w-4 shrink-0" />{baselineError}</p>}

              {compatibleCandidates.length > 0 && (
                <label className="block font-semibold text-slate-700">
                  Checkpoint phòng thủ đã xác thực
                  <select
                    aria-label="Checkpoint phòng thủ đã xác thực"
                    value={candidateId}
                    onChange={(event) => {
                      setCandidateId(event.target.value);
                      setCandidateFileName("");
                      setUploadState(event.target.value ? "ready" : "idle");
                      setUploadMessage(event.target.value ? `Checkpoint đã xác thực: ${event.target.value}` : "");
                      setEvaluationJob(null);
                      setEvaluationError("");
                    }}
                    className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-xs text-slate-800"
                  >
                    <option value="">Chọn checkpoint có sẵn</option>
                    {compatibleCandidates.map((candidate) => (
                      <option key={candidate.id} value={candidate.id} disabled={!candidate.runnable}>
                        {candidate.model_name} ({candidate.id}){candidate.blocked_reason ? `, ${candidate.blocked_reason}` : ""}
                      </option>
                    ))}
                  </select>
                </label>
              )}

              <label className={cn("flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-5 text-center", canUpload ? "cursor-pointer border-slate-300 bg-slate-50 hover:border-blue-500" : "cursor-not-allowed border-slate-200 bg-slate-100 opacity-60")}>
                <Upload className="h-6 w-6 text-slate-500" aria-hidden="true" />
                <span className="font-bold text-slate-700">{candidateFileName || "Tải lên checkpoint đã phòng thủ (.pt / .pth)"}</span>
                <span className="text-[11px] text-slate-500">{canUpload ? "Checkpoint sẽ được gắn với model gốc của run." : "Cần baseline backend có task, model family và parent checkpoint."}</span>
                <input aria-label="Tải lên checkpoint đã phòng thủ" type="file" accept=".pt,.pth" disabled={!canUpload || uploadState === "uploading"} className="sr-only" onChange={handleCandidateUpload} />
              </label>

              {uploadMessage && (
                <p role={uploadState === "error" ? "alert" : "status"} className={cn("rounded-lg border px-3 py-2.5 font-semibold", uploadState === "ready" && "border-emerald-200 bg-emerald-50 text-emerald-800", uploadState === "error" && "border-red-200 bg-red-50 text-red-800", !["ready", "error"].includes(uploadState) && "border-blue-200 bg-blue-50 text-blue-800")}>
                  {uploadMessage}
                </p>
              )}

              <Button variant="primary" onClick={handleRunLockedEvaluation} disabled={!canEvaluate} icon={isEvaluating ? RefreshCw : ShieldCheck} className={cn("w-full justify-center py-2.5 text-xs font-bold", canEvaluate && "bg-emerald-700 hover:bg-emerald-800")}>
                {isEvaluating ? "Đang tạo đánh giá phòng thủ..." : "Chạy Đánh Giá Đối Chiếu (Locked Protocol)"}
              </Button>
              {!candidateId && <p className="text-center text-[11px] leading-4 text-slate-500">Nút chạy chỉ mở khi checkpoint đã qua validation.</p>}
              {evaluationError && <p role="alert" className="rounded-lg border border-red-200 bg-red-50 px-3 py-2.5 text-red-800">{evaluationError}</p>}

              {evaluationJob ? (
                <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-emerald-900">
                  <p className="flex items-center gap-2 font-bold"><CheckCircle2 className="h-4 w-4" />Đã tạo evaluation job thật</p>
                  <dl className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
                    <div><dt className="text-[11px] text-emerald-700">Job ID</dt><dd className="mt-0.5 break-all font-mono font-semibold">{evaluationJob.run_id}</dd></div>
                    <div><dt className="text-[11px] text-emerald-700">Trạng thái</dt><dd className="mt-0.5 font-mono font-semibold">{evaluationJob.status}</dd></div>
                  </dl>
                </div>
              ) : (
                <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-4 text-center text-slate-600">Chưa có dữ liệu đánh giá phòng thủ.</div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
