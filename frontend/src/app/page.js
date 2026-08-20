"use client";

import React, { useState } from "react";
import ConfigPanel from "@/components/ConfigPanel.jsx";
import ClosedLoopPanel from "@/components/ClosedLoopPanel";
import DefencePanel from "@/components/DefencePanel";
import ComparisonView from "@/components/ComparisonView";
import ReportView from "@/components/ReportView";
import ImageGrid from "@/components/ImageGrid";
import ThemeToggle from "@/components/ThemeToggle";
import { useAdverTest } from "@/hooks/useAdverTest";

const MODE_LABELS = {
  detection2d: "2D Object Detection",
  segmentation: "Instance Segmentation",
  detection3d: "3D Object Detection",
};

function HeaderBar({ mode, runStatus, isRunning }) {
  const statusClass = isRunning
    ? "app-header__status--running"
    : runStatus === "FAILED"
    ? "app-header__status--failed"
    : "app-header__status--ready";

  const statusLabel = isRunning
    ? "Running"
    : runStatus === "FAILED"
    ? "Failed"
    : runStatus === "COMPLETED"
    ? "Complete"
    : "Ready";

  return (
    <header className="app-header" style={{ backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)" }}>
      <div className="app-header__left">
        <span className="app-header__brand">AdverTest</span>
        <div className="app-header__divider" />
        <span className="app-header__info">
          Task: <strong>{MODE_LABELS[mode] || mode}</strong>
        </span>
      </div>
      <div className="app-header__right">
        <ThemeToggle />
        <div className={`app-header__status ${statusClass}`} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span className={`status-beacon ${isRunning ? "status-beacon--running" : runStatus === "FAILED" ? "status-beacon--failed" : "status-beacon--ready"}`} />
          {statusLabel}
        </div>
        <span className="simulation-tag">Simulation Only</span>
      </div>
    </header>
  );
}

export default function HomePage() {
  const { state, actions } = useAdverTest();
  const [gridIndex, setGridIndex] = useState(0);
  const [configWidth, setConfigWidth] = useState(25); // percentage (15% to 30%)
  const isDraggingRef = React.useRef(false);

  React.useEffect(() => {
    try {
      const saved = localStorage.getItem("advertest-config-width");
      if (saved) {
        const parsed = Number(saved);
        if (parsed >= 15 && parsed <= 30) {
          // eslint-disable-next-line react-hooks/set-state-in-effect
          setConfigWidth(parsed);
        }
      }
    } catch {}
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
          title={`Kéo để thay đổi độ rộng cột cài đặt (${Math.round(configWidth)}%, tối đa 30%)`}
        >
          <div className="resize-divider__handle" />
        </div>

        <section className="center-view" aria-label="Robustness evidence workspace">
          {state.isRunning ? (
            <div className="progress-overlay">
              <strong>{state.runStatus}</strong>
              <progress max="100" value={state.progress} aria-label="Run progress" />
              <span>{state.progress}%</span>
              <p>{state.progressDetail}</p>
              <button type="button" onClick={actions.cancelRun}>Cancel</button>
            </div>
          ) : (
            <>
              <nav className="tab-bar" aria-label="Run results">
                <button className={`tab-bar__item ${state.activeTab === "evidence" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("evidence")}>Evidence</button>
                <button className={`tab-bar__item ${state.activeTab === "comparison" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("comparison")}>Comparison</button>
                <button className={`tab-bar__item ${state.activeTab === "report" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("report")}>Report</button>
                <button className={`tab-bar__item ${state.activeTab === "defence" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("defence")}>Defence</button>
              </nav>

              {state.runStatus === "FAILED" && (
                <p className="empty-state" role="alert" style={{ padding: "var(--space-md)", color: "var(--danger)", fontSize: "0.75rem" }}>
                  Run failed: {state.progressDetail}
                </p>
              )}

              {state.resultLoadStatus === "failed" && (
                <p className="empty-state" role="alert" style={{ padding: "var(--space-md)", color: "var(--danger)", fontSize: "0.75rem" }}>
                  Evidence could not be loaded: {state.resultLoadError}{" "}
                  <button type="button" onClick={actions.retryEvidence} className="action-button action-button--secondary" style={{ display: "inline", padding: "2px 8px", marginLeft: 6 }}>Retry</button>
                </p>
              )}

              {state.activeTab === "defence" ? (
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
                <ReportView report={state.report} />
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
