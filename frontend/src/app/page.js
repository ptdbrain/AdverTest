"use client";

import React, { useState } from "react";
import ConfigPanel from "@/components/ConfigPanel.jsx";
import ClosedLoopPanel from "@/components/ClosedLoopPanel";
import DefencePanel from "@/components/DefencePanel";
import ComparisonView from "@/components/ComparisonView";
import ReportView from "@/components/ReportView";
import ImageGrid from "@/components/ImageGrid";
import ThemeToggle from "@/components/ThemeToggle";
import AdvisorPanel from "@/components/AdvisorPanel.jsx";
import UserMenu from "@/components/UserMenu.jsx";
import { useAdverTest } from "@/hooks/useAdverTest";
import { useLanguage } from "@/context/LanguageContext";

const MODE_LABELS = {
  detection2d: "2D Object Detection",
  segmentation: "Instance Segmentation",
  detection3d: "3D Object Detection",
};

function HeaderBar({ mode, runStatus, isRunning }) {
  const { t } = useLanguage();

  const statusClass = isRunning
    ? "app-header__status--running"
    : runStatus === "FAILED"
    ? "app-header__status--failed"
    : "app-header__status--ready";

  const statusLabel = isRunning
    ? t("common.running")
    : runStatus === "FAILED"
    ? t("common.failed")
    : runStatus === "COMPLETED"
    ? t("common.complete")
    : t("common.ready");

  return (
    <header className="app-header" style={{ backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)" }}>
      <div className="app-header__left" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        <span className="app-header__brand">AdverTest</span>
        <div className="app-header__divider" />
        <span className="app-header__info">
          {t("common.task")} <strong>{MODE_LABELS[mode] || mode}</strong>
        </span>
      </div>

      <div className="app-header__right" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        {/* Review Queue (Human-in-the-Loop) Link */}
        <a
          href="/reviews"
          className="rounded-md border border-amber-600/40 bg-amber-500/10 px-2.5 py-1 text-xs font-semibold text-amber-300 hover:bg-amber-500/20 hover:text-amber-200 transition-colors"
          style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: "4px" }}
          title={t("workspace.reviewQueue.title")}
        >
          <span>{t("workspace.reviewQueue")}</span>
        </a>

        {/* Status Beacon */}
        <div className={`app-header__status ${statusClass}`} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span className={`status-beacon ${isRunning ? "status-beacon--running" : runStatus === "FAILED" ? "status-beacon--failed" : "status-beacon--ready"}`} />
          {statusLabel}
        </div>

        {/* User Authentication Menu (with Settings access) */}
        <UserMenu />
      </div>
    </header>
  );
}

export default function HomePage() {
  const { state, actions } = useAdverTest();
  const { t } = useLanguage();
  const [gridIndex, setGridIndex] = useState(0);
  const [configWidth, setConfigWidth] = useState(25); // percentage (15% to 30%)
  const isDraggingRef = React.useRef(false);

  React.useEffect(() => {
    let updateTimer;
    try {
      const saved = localStorage.getItem("advertest-config-width");
      if (saved) {
        const parsed = Number(saved);
        if (parsed >= 15 && parsed <= 30) {
          updateTimer = window.setTimeout(() => setConfigWidth(parsed), 0);
        }
      }
    } catch {}
    return () => window.clearTimeout(updateTimer);
  }, []);

  const handleMouseDown = (e) => {
    e.preventDefault();
    isDraggingRef.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const handleMouseMove = (moveEvent) => {
      if (!isDraggingRef.current) return;
      const totalWidth = window.innerWidth - 52;
      const relativeX = moveEvent.clientX - 52;
      const percentage = (relativeX / totalWidth) * 100;
      const clamped = Math.min(30, Math.max(15, percentage));
      setConfigWidth(clamped);
    };

    const handleMouseUp = () => {
      isDraggingRef.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      setConfigWidth((current) => {
        try {
          localStorage.setItem("advertest-config-width", current.toString());
        } catch {}
        return current;
      });
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
  };

  if (state.loading) {
    return (
      <>
        <HeaderBar mode={state.mode} runStatus={null} isRunning={false} />
        <main
          className="main-content"
          style={{
            display: "grid",
            gridTemplateColumns: `${configWidth}% 4px minmax(0, 1fr)`,
            gap: 0,
          }}
        >
          <aside className="config-panel" style={{ gap: "var(--space-lg)" }}>
            {/* Skeleton task selector */}
            <div style={{ padding: "0 var(--space-md)" }}>
              <div className="skeleton" style={{ width: "40px", height: "10px", marginBottom: "10px" }} />
              <div className="skeleton" style={{ width: "100%", height: "36px", borderRadius: "var(--radius-md)" }} />
            </div>
            {/* Skeleton model */}
            <div style={{ padding: "0 var(--space-md)" }}>
              <div className="skeleton" style={{ width: "50px", height: "10px", marginBottom: "10px" }} />
              <div className="skeleton" style={{ width: "100%", height: "36px", borderRadius: "var(--radius-sm)" }} />
            </div>
            {/* Skeleton dataset */}
            <div style={{ padding: "0 var(--space-md)" }}>
              <div className="skeleton" style={{ width: "60px", height: "10px", marginBottom: "10px" }} />
              <div className="skeleton" style={{ width: "100%", height: "36px", borderRadius: "var(--radius-sm)" }} />
            </div>
            {/* Skeleton attacks */}
            <div style={{ padding: "0 var(--space-md)" }}>
              <div className="skeleton" style={{ width: "70px", height: "10px", marginBottom: "10px" }} />
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="skeleton" style={{ width: "100%", height: "44px", borderRadius: "var(--radius-sm)", marginBottom: "6px" }} />
              ))}
            </div>
          </aside>
          <div className="resize-divider" />
          <div className="center-view flex-center" style={{ flexDirection: "column", gap: "var(--space-lg)" }}>
            <div className="skeleton" style={{ width: "48px", height: "48px", borderRadius: "50%" }} />
            <div className="skeleton" style={{ width: "160px", height: "12px" }} />
            <div className="skeleton" style={{ width: "220px", height: "10px" }} />
          </div>
        </main>
      </>
    );
  }

  return (
    <>
      <HeaderBar mode={state.mode} runStatus={state.runStatus} isRunning={state.isRunning} />
      <main
        className="main-content"
        style={{
          display: "grid",
          gridTemplateColumns: `${configWidth}% 4px minmax(0, 1fr)`,
          gap: 0,
        }}
      >
        <ConfigPanel
          datasets={state.datasets}
          attacks={state.attacks}
          modes={state.modes}
          modelFamilies={state.modelFamilies}
          baseCheckpoints={state.baseCheckpoints}
          recipePresets={state.recipePresets}
          recipe={state.recipe}
          mode={state.mode}
          selectedModelFamily={state.selectedModelFamily}
          selectedModelVersion={state.selectedModelVersion}
          selectedDataset={state.selectedDataset}
          selectedAttacks={state.selectedAttacks}
          runOptions={state.runOptions}
          runId={state.runId}
          activeRunMeta={state.activeRunMeta}
          isRunning={state.isRunning}
          progress={state.progress}
          progressDetail={state.progressDetail}
          runStatus={state.runStatus}
          actions={actions}
        />

        {/* Resizable Divider Handle */}
        <div
          className="resize-divider"
          onMouseDown={handleMouseDown}
          role="separator"
          aria-orientation="vertical"
          aria-valuenow={Math.round(configWidth)}
          aria-valuemin={15}
          aria-valuemax={30}
          title={t("workspace.resizeHint", { pct: Math.round(configWidth) })}
        >
          <div className="resize-divider__handle" />
        </div>

        <section className="center-view" aria-label={t("workspace.evidenceWorkspace")}>
          {state.isRunning ? (
            <div
              className="progress-overlay"
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "16px",
                padding: "32px",
                textAlign: "center",
              }}
            >
              <div style={{ display: "inline-flex", alignItems: "center", gap: "8px", padding: "6px 14px", borderRadius: "20px", background: "rgba(56, 189, 248, 0.12)", border: "1px solid rgba(56, 189, 248, 0.3)" }}>
                <span className="status-beacon status-beacon--running" />
                <span style={{ fontSize: "0.78rem", fontWeight: 800, color: "var(--accent)" }}>
                  {t("workspace.runningSession", { id: state.runId ? state.runId.slice(0, 10) : "INITIALIZING" })}
                </span>
              </div>

              <div style={{ maxWidth: "460px" }}>
                <h3 style={{ fontSize: "1.1rem", fontWeight: 800, margin: "0 0 6px", color: "var(--text-primary)" }}>
                  {state.progressDetail || t("workspace.processing")}
                </h3>
                <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                  🎯 <strong>{t("common.task")}</strong> {state.mode} &nbsp;|&nbsp; 🤖 <strong>{t("common.model")}</strong> {state.selectedModelVersion || "YOLO11"} &nbsp;|&nbsp; 📦 <strong>{t("common.data")}</strong> {state.selectedDataset}
                  <br />
                  ⚡ <strong>{t("common.attackChain")}</strong> {state.selectedAttacks?.length ? state.selectedAttacks.join(", ") : "Attack Recipe"} (Cấp {state.recipe.steps?.[0]?.severity || 3})
                </div>
              </div>

              <div style={{ width: "100%", maxWidth: "380px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", fontWeight: 700, color: "var(--text-muted)", marginBottom: "4px" }}>
                  <span>{state.runStatus || "PROCESSING"}</span>
                  <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent)" }}>{state.progress}%</span>
                </div>
                <div className="progress-inline__bar-track" style={{ height: "8px", borderRadius: "4px", background: "var(--bg-elevated)" }}>
                  <div className="progress-inline__bar-fill" style={{ width: `${state.progress}%`, background: "var(--accent)", borderRadius: "4px", transition: "width 0.3s ease" }} />
                </div>
              </div>

              <div style={{ display: "flex", gap: "10px", marginTop: "8px" }}>
                <button
                  type="button"
                  onClick={actions.cancelRun}
                  className="action-button action-button--secondary"
                  style={{
                    padding: "8px 18px",
                    fontWeight: 800,
                    fontSize: "0.8rem",
                    color: "#EF4444",
                    borderColor: "rgba(239, 68, 68, 0.4)",
                    background: "rgba(239, 68, 68, 0.1)",
                  }}
                >
                  {t("workspace.cancelSession")}
                </button>
              </div>
            </div>
          ) : (
            <>
              <nav className="tab-bar" aria-label="Run results" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex" }}>
                  <button className={`tab-bar__item ${state.activeTab === "evidence" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("evidence")}>Evidence</button>
                  <button className={`tab-bar__item ${state.activeTab === "comparison" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("comparison")}>Comparison</button>
                  <button className={`tab-bar__item ${state.activeTab === "report" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("report")}>Report</button>
                  <button className={`tab-bar__item ${state.activeTab === "defence" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("defence")}>Defence</button>
                  <button className={`tab-bar__item ${state.activeTab === "advisor" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("advisor")}>AI Advisor</button>
                </div>
                {state.report && (
                  <button
                    type="button"
                    onClick={actions.resetSession}
                    style={{
                      background: "none",
                      border: "none",
                      color: "var(--text-muted)",
                      fontSize: "0.72rem",
                      cursor: "pointer",
                      padding: "4px 10px",
                      marginRight: "10px",
                      display: "flex",
                      alignItems: "center",
                      gap: "4px",
                    }}
                    title={t("workspace.clearSession.title")}
                  >
                    <span>{t("workspace.clearSession")}</span>
                  </button>
                )}
              </nav>

              {state.runStatus === "FAILED" && (
                <p className="empty-state" role="alert" style={{ padding: "var(--space-md)", color: "var(--danger)", fontSize: "0.75rem" }}>
                  {t("workspace.runFailed", { detail: state.progressDetail })}
                </p>
              )}

              {state.resultLoadStatus === "failed" && (
                <p className="empty-state" role="alert" style={{ padding: "var(--space-md)", color: "var(--danger)", fontSize: "0.75rem" }}>
                  {t("workspace.evidenceLoadFailed", { error: state.resultLoadError })}{" "}
                  <button type="button" onClick={actions.retryEvidence} className="action-button action-button--secondary" style={{ display: "inline", padding: "2px 8px", marginLeft: 6 }}>{t("common.retry")}</button>
                </p>
              )}

              {state.activeTab === "advisor" ? (
                <div style={{ padding: "var(--space-md)" }}>
                  <AdvisorPanel
                    runId={state.runId}
                    onTriggerAction={(actionType, params) => {
                      if (actionType === "RUN_CLEAN_BASELINE" && actions.runBenchmark) {
                        actions.runBenchmark();
                      } else if (actionType === "GENERATE_DEFENCE_DATASET") {
                        actions.setActiveTab("defence");
                      }
                    }}
                  />
                </div>
              ) : state.activeTab === "defence" ? (
                <DefencePanel
                  baselineRunId={state.defenceBaselineRunId || state.runId}
                  baseCheckpoints={state.baseCheckpoints}
                  candidates={state.defenceCheckpoints}
                  comparison={state.defenceComparison}
                  running={state.isRunning}
                  progress={state.progress}
                  onEvaluate={actions.runDefence}
                  onRefreshCandidates={actions.refreshDefenceCheckpoints}
                />
              ) : state.activeTab === "comparison" ? (
                <ComparisonView
                  report={state.report}
                  samples={state.samples}
                  severity={state.recipe.steps.at(-1)?.severity}
                />
              ) : state.activeTab === "report" ? (
                <ReportView report={state.report} onClearHistory={actions.clearReportHistory} />
              ) : (
                <ImageGrid
                  key={state.runId || "empty"}
                  report={state.report}
                  samples={state.samples}
                  selectedIndex={gridIndex}
                  onSelectSample={setGridIndex}
                />
              )}

              {state.mode === "detection2d" && state.report && (
                <ClosedLoopPanel
                  report={state.report}
                  backlog={state.backlog}
                  trainingBlockedReason={state.trainingBlockedReason}
                  isCreatingBacklog={state.isCreatingBacklog}
                  error={state.backlogError}
                  onCreateBacklog={actions.createBacklog}
                />
              )}
            </>
          )}
        </section>
      </main>
    </>
  );
}
