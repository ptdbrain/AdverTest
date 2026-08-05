"use client";

import React from "react";
import { useAdverTest } from "@/hooks/useAdverTest";
import ConfigPanel from "@/components/ConfigPanel";
import ComparisonView from "@/components/ComparisonView";
import ReportView from "@/components/ReportView";
import ModelResultsPanel from "@/components/ModelResultsPanel";

/**
 * HomePage Component
 * Container component that orchestrates state via useAdverTest and renders sub-components.
 * Adheres to Single Responsibility and Separation of Concerns.
 */
export default function HomePage() {
  const { state, actions } = useAdverTest();
  
  if (state.loading) {
    return (
      <div className="main-content">
        <div className="config-panel flex-center">
          <div className="empty-state">
            <div className="empty-state__icon">⏳</div>
            <div className="empty-state__message">Loading catalog...</div>
          </div>
        </div>
        <div className="center-view flex-center" />
        <div className="results-panel" />
      </div>
    );
  }

  return (
    <div className="main-content">
      {/* LEFT — Configuration Panel */}
      <ConfigPanel 
        datasets={state.datasets}
        attacks={state.attacks}
        selectedDataset={state.selectedDataset}
        selectedAttacks={state.selectedAttacks}
        severity={state.severity}
        isRunning={state.isRunning}
        actions={actions}
      />

      {/* CENTER — Image Comparison / Report */}
      <main className="center-view">
        {/* Tab bar */}
        <div className="tab-bar">
          <button
            className={`tab-bar__item ${state.activeTab === "compare" ? "tab-bar__item--active" : ""}`}
            onClick={() => actions.setActiveTab("compare")}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="3" width="20" height="14" rx="2" /><path d="M12 3v14" /><path d="M2 20h20" /></svg>
            Image Comparison
          </button>
          <button
            className={`tab-bar__item ${state.activeTab === "report" ? "tab-bar__item--active" : ""}`}
            onClick={() => actions.setActiveTab("report")}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18" /><path d="M9 3v18" /></svg>
            Robustness Report
          </button>
        </div>

        {/* Progress overlay */}
        {state.isRunning && (
          <div className="progress-overlay animate-fade-in">
            <svg width="0" height="0">
              <defs>
                <linearGradient id="progress-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="var(--cyan)" />
                  <stop offset="100%" stopColor="var(--purple)" />
                </linearGradient>
              </defs>
            </svg>
            <div className="progress-ring">
              <svg className="progress-ring__circle" width="120" height="120" viewBox="0 0 120 120">
                <circle className="progress-ring__bg" cx="60" cy="60" r="52" />
                <circle
                  className="progress-ring__fill"
                  cx="60" cy="60" r="52"
                  strokeDasharray={`${2 * Math.PI * 52}`}
                  strokeDashoffset={`${2 * Math.PI * 52 * (1 - state.progress / 100)}`}
                />
              </svg>
              <div className="progress-ring__text">
                <span className="progress-ring__value">{state.progress}%</span>
                <span className="progress-ring__label">{state.runStatus}</span>
              </div>
            </div>
            <div className="progress-status">{state.progressDetail}</div>
          </div>
        )}

        {/* View switching based on active tab */}
        <div className={state.activeTab === "compare" && !state.isRunning ? "comparison-view" : "report-view"} style={{ display: state.isRunning ? "none" : "flex" }}>
          {state.activeTab === "compare" ? (
            <ComparisonView report={state.report} samples={state.samples} severity={state.severity} />
          ) : (
            <ReportView report={state.report} />
          )}
        </div>
      </main>

      {/* RIGHT — Model Results Panel */}
      <ModelResultsPanel report={state.report} models={state.models} />
    </div>
  );
}
