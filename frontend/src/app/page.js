"use client";

import React from "react";
import ConfigPanel from "@/components/ConfigPanel.jsx";
import ClosedLoopPanel from "@/components/ClosedLoopPanel";
import Workspace from "@/components/Workspace";
import { useAdverTest } from "@/hooks/useAdverTest";

export default function HomePage() {
  const { state, actions } = useAdverTest();
  if (state.loading) return <main className="main-content"><aside className="config-panel" /><div className="center-view flex-center">Loading verified catalogs…</div></main>;
  return (
    <main className="main-content">
      <ConfigPanel datasets={state.datasets} attacks={state.attacks} modes={state.modes} modelVersions={state.modelVersions} mode={state.mode} selectedModelVersion={state.selectedModelVersion} selectedDataset={state.selectedDataset} selectedAttacks={state.selectedAttacks} severity={state.severity} isRunning={state.isRunning} actions={actions} />
      <section className="center-view" aria-label="Robustness evidence workspace">
        {state.isRunning ? <div className="progress-overlay"><strong>{state.runStatus}</strong><p>{state.progressDetail}</p></div> : <>
          <Workspace report={state.report} samples={state.samples} mode={state.mode} />
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
