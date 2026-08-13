"use client";

import React from "react";
import ConfigPanel from "@/components/ConfigPanel.jsx";
import ClosedLoopPanel from "@/components/ClosedLoopPanel";
import ComparisonView from "@/components/ComparisonView";
import ReportView from "@/components/ReportView";
import Workspace from "@/components/Workspace";
import { useAdverTest } from "@/hooks/useAdverTest";

export default function HomePage() {
  const { state, actions } = useAdverTest();
  if (state.loading) return <main className="main-content"><aside className="config-panel" /><div className="center-view flex-center">Loading verified catalogs…</div></main>;
  return (
    <main className="main-content">
      <ConfigPanel datasets={state.datasets} attacks={state.attacks} modes={state.modes} modelVersions={state.modelVersions} recipePresets={state.recipePresets} recipe={state.recipe} mode={state.mode} selectedModelVersion={state.selectedModelVersion} selectedDataset={state.selectedDataset} selectedAttacks={state.selectedAttacks} runOptions={state.runOptions} isRunning={state.isRunning} actions={actions} />
      <section className="center-view" aria-label="Robustness evidence workspace">
        {state.isRunning ? <div className="progress-overlay"><strong>{state.runStatus}</strong><progress max="100" value={state.progress} aria-label="Run progress" /><span>{state.progress}%</span><p>{state.progressDetail}</p><button type="button" onClick={actions.cancelRun}>Cancel</button></div> : <>
          {state.report && <nav className="tab-bar" aria-label="Run results"><button className={`tab-bar__item ${state.activeTab === "evidence" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("evidence")}>Evidence</button><button className={`tab-bar__item ${state.activeTab === "comparison" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("comparison")}>Prediction comparison</button><button className={`tab-bar__item ${state.activeTab === "report" ? "tab-bar__item--active" : ""}`} onClick={() => actions.setActiveTab("report")}>Report</button></nav>}
          {state.runStatus === "FAILED" && <p className="empty-state" role="alert">Run failed: {state.progressDetail}</p>}
          {state.resultLoadStatus === "failed" && <p className="empty-state" role="alert">Run completed, but evidence could not be loaded: {state.resultLoadError} <button type="button" onClick={actions.retryEvidence}>Retry evidence</button></p>}
          {state.activeTab === "comparison" ? <ComparisonView report={state.report} samples={state.samples} severity={state.recipe.steps.at(-1)?.severity} /> : state.activeTab === "report" ? <ReportView report={state.report} /> : <Workspace key={state.runId || "empty"} report={state.report} samples={state.samples} mode={state.mode} />}
          {state.mode === "detection2d" && state.report && <ClosedLoopPanel
            report={state.report}
            backlog={state.backlog}
            trainingBlockedReason={state.trainingBlockedReason}
            isCreatingBacklog={state.isCreatingBacklog}
            error={state.backlogError}
            onCreateBacklog={actions.createBacklog}
          />}
        </>}
      </section>
    </main>
  );
}
