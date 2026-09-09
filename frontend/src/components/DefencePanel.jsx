"use client";

import React, { useRef, useState } from "react";
import DefenceVisualComparison from "@/components/DefenceVisualComparison";
import { useLanguage } from "@/context/LanguageContext";
import { getCheckpoint, uploadCheckpoint } from "@/lib/api";

/**
 * DefencePanel — Đánh giá mô hình phòng thủ & so sánh hiệu năng cải tiến
 * Luồng hoạt động:
 * 1. Khóa giao thức kiểm thử từ bài test gốc (Locked Protocol: cùng dataset, seed, recipe).
 * 2. Tải lên (hoặc chọn) trọng số checkpoint đã qua huấn luyện phòng thủ (Fine-tuned / Repaired).
 * 3. Chạy đánh giá đối đầu trực diện giữa Base Model và Defended Model.
 * 4. Kết xuất báo cáo Tỷ lệ phục hồi (Recovery Rate) và độ lệch metric (Metric Deltas).
 */

export default function DefencePanel({
  baselineRunId,
  baseCheckpoints = [],
  candidates = [],
  comparison,
  running,
  progress,
  onEvaluate,
  onRefreshCandidates,
}) {
  const { t } = useLanguage();
  const [candidateId, setCandidateId] = useState("");
  const [parentId, setParentId] = useState("");
  const [uploadState, setUploadState] = useState("");
  const fileInputRef = useRef(null);

  const uploadCandidate = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !parentId) return;
    const parent = baseCheckpoints.find((item) => item.id === parentId);
    if (!parent) return;

    setUploadState(t("defence.upload.start", { pct: 0 }));
    try {
      const created = await uploadCheckpoint(
        file,
        {
          taskId: parent.task,
          familyId: parent.model_family_id,
          displayName: file.name,
          role: "fine_tuned",
          parentCheckpointId: parentId,
        },
        (ratio) => setUploadState(t("defence.upload.start", { pct: Math.round(ratio * 100) })),
      );

      for (let attempt = 0; attempt < 60; attempt += 1) {
        const status = await getCheckpoint(created.checkpoint_id);
        if (status.status === "READY") {
          setUploadState(t("defence.upload.ready"));
          await onRefreshCandidates?.();
          setCandidateId(created.checkpoint_id);
          return;
        }
        if (status.status === "REJECTED") {
          setUploadState(
            t("defence.upload.rejected", { reason: status.validation_reason || "Integrity check failed" }),
          );
          return;
        }
        setUploadState(t("defence.upload.scanning"));
        await new Promise((resolve) => setTimeout(resolve, 600));
      }
      setUploadState(t("defence.upload.queued"));
    } catch (error) {
      setUploadState(t("defence.upload.failed", { error: error.message || "Network error" }));
    } finally {
      event.target.value = "";
    }
  };

  return (
    <div
      style={{
        padding: "20px",
        margin: "var(--space-sm, 10px)",
        borderRadius: "var(--radius-md, 8px)",
        background: "var(--bg-elevated, #101623)",
        border: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.15))",
        display: "flex",
        flexDirection: "column",
        gap: "18px",
        flex: "1 1 auto",
        minHeight: 0,
        overflowY: "auto",
        overscrollBehavior: "contain",
      }}
    >
      {/* Header Info */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          flexWrap: "wrap",
          gap: "10px",
        }}
      >
        <div>
          <div
            style={{
              fontSize: "0.68rem",
              fontWeight: 800,
              color: "var(--accent, #38BDF8)",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            {t("defence.eyebrow")}
          </div>
          <h2 style={{ fontSize: "1.05rem", fontWeight: 800, margin: "4px 0 2px", color: "var(--text-primary)" }}>
            {t("defence.title")}
          </h2>
          {/* eslint-disable-next-line react/no-danger */}
          <p
            style={{ fontSize: "0.75rem", color: "var(--text-muted)", margin: 0, lineHeight: 1.5 }}
            dangerouslySetInnerHTML={{ __html: t("defence.desc") }}
          />
        </div>

        {baselineRunId && (
          <div
            style={{
              padding: "4px 10px",
              borderRadius: "6px",
              background: "rgba(56, 189, 248, 0.1)",
              border: "1px solid rgba(56, 189, 248, 0.25)",
              fontSize: "0.72rem",
              color: "var(--accent, #38BDF8)",
              fontWeight: 600,
            }}
          >
            {t("defence.baseline")}{" "}
            <code style={{ fontFamily: "var(--font-mono)" }}>{baselineRunId.slice(0, 10)}...</code>
          </div>
        )}
      </div>

      {!baselineRunId ? (
        <div
          style={{
            padding: "32px 20px",
            textAlign: "center",
            background: "var(--bg-primary, #0B0F17)",
            borderRadius: "8px",
            border: "1px dashed var(--border-subtle)",
          }}
        >
          <span style={{ fontSize: "2rem", display: "block", marginBottom: "8px" }}>⚡</span>
          <strong style={{ fontSize: "0.85rem", color: "var(--text-primary)" }}>{t("defence.noBaseline.title")}</strong>
          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", maxWidth: "380px", margin: "6px auto 0" }}>
            {t("defence.noBaseline.desc")}
          </p>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {/* Step 1 & 2: Select Parent & Upload */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "14px" }}>
            {/* Parent Base Checkpoint */}
            <div
              style={{
                background: "var(--bg-primary)",
                padding: "14px",
                borderRadius: "8px",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <label
                className="config-panel__label"
                style={{ fontSize: "0.72rem", marginBottom: "6px", display: "block" }}
              >
                {t("defence.step1")}
              </label>
              <select
                aria-label="Parent checkpoint"
                className="select-field"
                value={parentId}
                onChange={(event) => setParentId(event.target.value)}
                style={{ width: "100%", padding: "7px 10px", fontSize: "0.78rem" }}
              >
                <option value="">{t("defence.step1.placeholder")}</option>
                {baseCheckpoints.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.model_name} ({item.id})
                  </option>
                ))}
              </select>
            </div>

            {/* Upload Checkpoint */}
            <div
              style={{
                background: "var(--bg-primary)",
                padding: "14px",
                borderRadius: "8px",
                border: "1px solid var(--border-subtle)",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
              }}
            >
              <div>
                <label
                  className="config-panel__label"
                  style={{ fontSize: "0.72rem", marginBottom: "6px", display: "block" }}
                >
                  {t("defence.step2")}
                </label>
                <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                  <button
                    type="button"
                    className="action-button action-button--secondary"
                    disabled={!parentId}
                    onClick={() => fileInputRef.current?.click()}
                    style={{ padding: "7px 14px", fontSize: "0.72rem", fontWeight: 700, whiteSpace: "nowrap" }}
                  >
                    {t("defence.step2.button")}
                  </button>
                  <span style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>
                    {parentId ? t("defence.step2.ready") : t("defence.step2.needParent")}
                  </span>
                </div>
                <input
                  ref={fileInputRef}
                  aria-label="Upload defence candidate"
                  type="file"
                  accept=".pt,.pth"
                  style={{ display: "none" }}
                  onChange={uploadCandidate}
                />
              </div>

              {uploadState && (
                <div
                  style={{
                    marginTop: "8px",
                    fontSize: "0.72rem",
                    color: uploadState.startsWith("❌")
                      ? "#EF4444"
                      : uploadState.startsWith("✅")
                        ? "#10B981"
                        : "var(--accent)",
                    fontWeight: 600,
                  }}
                >
                  {uploadState}
                </div>
              )}
            </div>
          </div>

          {/* Step 3: Select Candidate & Run */}
          <div
            style={{
              background: "var(--bg-primary)",
              padding: "16px",
              borderRadius: "8px",
              border: "1px solid var(--border-subtle)",
              display: "flex",
              flexDirection: "column",
              gap: "12px",
            }}
          >
            <label className="config-panel__label" style={{ fontSize: "0.75rem", margin: 0 }}>
              {t("defence.step3")}
            </label>
            <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "center" }}>
              <select
                aria-label="Defence candidate"
                className="select-field"
                value={candidateId}
                onChange={(event) => setCandidateId(event.target.value)}
                style={{ flex: 1, minWidth: "240px", padding: "8px 12px", fontSize: "0.82rem" }}
              >
                <option value="">{t("defence.step3.placeholder")}</option>
                {candidates.map((item) => (
                  <option key={item.id} value={item.id} disabled={!item.runnable}>
                    {item.model_name} — [{item.checkpoint_role}]{" "}
                    {item.blocked_reason ? `(${item.blocked_reason})` : t("defence.readyBadge")}
                  </option>
                ))}
              </select>

              <button
                type="button"
                className="run-button"
                disabled={running || !candidateId}
                onClick={() => onEvaluate(candidateId)}
                style={{ minWidth: "180px", padding: "8px 18px", fontSize: "0.82rem", fontWeight: 700 }}
              >
                {running ? t("defence.runningButton", { progress }) : t("defence.runButton")}
              </button>
            </div>
          </div>

          {/* Comparison Delta Results */}
          {comparison && (
            <div
              style={{
                background: "var(--bg-primary)",
                borderRadius: "8px",
                border: "1px solid rgba(56, 189, 248, 0.25)",
                padding: "16px",
                display: "flex",
                flexDirection: "column",
                gap: "14px",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <strong
                  style={{
                    fontSize: "0.85rem",
                    color: "var(--text-primary)",
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                  }}
                >
                  <span>{t("defence.compareTitle")}</span>
                </strong>
                <span
                  style={{
                    fontSize: "0.68rem",
                    fontWeight: 700,
                    padding: "2px 8px",
                    borderRadius: "4px",
                    background: "rgba(16, 185, 129, 0.15)",
                    color: "#10B981",
                  }}
                >
                  {t("defence.pairedBadge")}
                </span>
              </div>

              {!comparison.paired ? (
                <div
                  style={{
                    color: "var(--danger)",
                    fontSize: "0.78rem",
                    padding: "8px 12px",
                    background: "rgba(239, 68, 68, 0.1)",
                    borderRadius: "6px",
                  }}
                >
                  {t("defence.notPaired", { reason: comparison.incompatibilities?.join(", ") })}
                </div>
              ) : (
                <>
                  {/* Summary Metric Cards */}
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                      gap: "10px",
                    }}
                  >
                    <div
                      style={{
                        padding: "12px",
                        background: "var(--bg-elevated)",
                        borderRadius: "6px",
                        border: "1px solid var(--border-subtle)",
                      }}
                    >
                      <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 600 }}>
                        {t("defence.baseCleanAP")}
                      </span>
                      <p
                        style={{
                          fontSize: "1.1rem",
                          fontWeight: 800,
                          fontFamily: "var(--font-mono)",
                          margin: "4px 0 0",
                          color: "var(--text-primary)",
                        }}
                      >
                        {comparison.recovery_report?.baseline_clean?.toFixed?.(3) ?? "—"}
                      </p>
                    </div>

                    <div
                      style={{
                        padding: "12px",
                        background: "var(--bg-elevated)",
                        borderRadius: "6px",
                        border: "1px solid var(--border-subtle)",
                      }}
                    >
                      <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 600 }}>
                        {t("defence.finetunedCleanAP")}
                      </span>
                      <p
                        style={{
                          fontSize: "1.1rem",
                          fontWeight: 800,
                          fontFamily: "var(--font-mono)",
                          margin: "4px 0 0",
                          color: "var(--accent)",
                        }}
                      >
                        {comparison.recovery_report?.candidate_clean?.toFixed?.(3) ?? "—"}
                      </p>
                    </div>

                    <div
                      style={{
                        padding: "12px",
                        background: "rgba(16, 185, 129, 0.1)",
                        borderRadius: "6px",
                        border: "1px solid rgba(16, 185, 129, 0.3)",
                      }}
                    >
                      <span style={{ fontSize: "0.68rem", color: "#10B981", fontWeight: 700 }}>
                        {t("defence.recovery")}
                      </span>
                      <p
                        style={{
                          fontSize: "1.1rem",
                          fontWeight: 800,
                          fontFamily: "var(--font-mono)",
                          margin: "4px 0 0",
                          color: "#10B981",
                        }}
                      >
                        {comparison.recovery_report?.recovery_rate?.percent_value == null
                          ? "—"
                          : `${Number(comparison.recovery_report.recovery_rate.percent_value).toFixed(1)}%`}
                      </p>
                    </div>
                  </div>

                  {/* Detailed Metric Deltas */}
                  {comparison.metric_deltas && Object.keys(comparison.metric_deltas).length > 0 && (
                    <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "12px" }}>
                      <div
                        style={{
                          fontSize: "0.72rem",
                          fontWeight: 700,
                          color: "var(--text-secondary)",
                          marginBottom: "8px",
                        }}
                      >
                        {t("defence.metricDeltas")}
                      </div>
                      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                        {Object.entries(comparison.metric_deltas)
                          .filter(([name]) => name !== "clean_detection_score")
                          .map(([name, metric]) => {
                            const val = Number(metric?.value ?? 0);
                            const isPositive = val >= 0;
                            return (
                              <span
                                key={name}
                                style={{
                                  padding: "4px 10px",
                                  borderRadius: "4px",
                                  fontSize: "0.72rem",
                                  fontWeight: 600,
                                  background: isPositive ? "rgba(16, 185, 129, 0.12)" : "rgba(239, 68, 68, 0.12)",
                                  color: isPositive ? "#10B981" : "#EF4444",
                                  border: `1px solid ${isPositive ? "rgba(16, 185, 129, 0.25)" : "rgba(239, 68, 68, 0.25)"}`,
                                  fontFamily: "var(--font-mono)",
                                }}
                              >
                                {name.replaceAll("_", " ")}: {isPositive ? "+" : ""}
                                {val.toFixed(3)}
                              </span>
                            );
                          })}
                      </div>
                    </div>
                  )}

                  <DefenceVisualComparison
                    baselineRunId={comparison.baseline_run_id}
                    candidateRunId={comparison.candidate_run_id}
                  />
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
