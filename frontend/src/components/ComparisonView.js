import React, { useState } from "react";

function degradationColor(d) {
  if (d < 15) return "var(--success)";
  if (d < 40) return "var(--warning)";
  return "var(--danger)";
}

/**
 * ComparisonView Component
 * Renders the before/after image split view with a refined floating toolbar.
 */
export default function ComparisonView({ report, samples, severity }) {
  const [selectedSampleIdx, setSelectedSampleIdx] = useState(0);

  /* ---- Overlay state ---- */
  const [showBbox, setShowBbox] = useState(true);
  const [showMasks, setShowMasks] = useState(false);
  const [showGT, setShowGT] = useState(true);

  if (!report) {
    return (
      <div className="placeholder-view">
        <div className="placeholder-view__title">AdverTest</div>
        <div className="placeholder-view__subtitle">
          Select a dataset and attack type on the left, then press
          <strong style={{ color: "var(--cyan)" }}> Run Attack </strong>
          to view clean vs attacked image comparisons and measure model degradation.
        </div>
      </div>
    );
  }

  const currentSample = samples[selectedSampleIdx] || null;

  return (
    <>
      <div className="comparison-container tesla-grid-bg">
        <div className="comparison-container__side comparison-container__side--clean">
          <span className="comparison-container__label comparison-container__label--clean">
            Clean — Original
          </span>
          {currentSample?.clean_image ? (
            <img className="comparison-image" src={currentSample.clean_image} alt="Clean" />
          ) : (
            <div className="flex-center" style={{ flex: 1, color: "var(--text-muted)" }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "0.75rem" }}>Clean image</div>
                <div style={{ fontSize: "0.6rem", marginTop: "4px", fontFamily: "var(--font-mono)", color: "var(--text-tertiary)" }}>
                  AP = {report.ap_clean?.toFixed(3) || "\u2014"}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="comparison-container__divider" />

        <div className="comparison-container__side">
          <span className="comparison-container__label comparison-container__label--attacked">
            Attacked — {currentSample?.attack || "N/A"} (Sev.{currentSample?.severity || severity})
          </span>
          {currentSample?.attacked_image ? (
            <img className="comparison-image" src={currentSample.attacked_image} alt="Attacked" />
          ) : (
            <div className="flex-center" style={{ flex: 1, color: "var(--text-muted)" }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: "0.75rem" }}>Attacked image</div>
                {currentSample && (
                  <div style={{ fontSize: "0.6rem", marginTop: "4px", fontFamily: "var(--font-mono)", color: "var(--text-tertiary)" }}>
                    AP = {currentSample.ap?.toFixed(3) || "\u2014"} | D = {currentSample.degradation?.toFixed(1)}%
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Floating Toolbar */}
        <div className="floating-toolbar">
          <button
            className={`floating-toolbar__btn ${showBbox ? "floating-toolbar__btn--active" : ""}`}
            onClick={() => setShowBbox(!showBbox)}
          >
            <span className="floating-toolbar__checkbox" /> Boxes
          </button>
          <button
            className={`floating-toolbar__btn ${showMasks ? "floating-toolbar__btn--active" : ""}`}
            onClick={() => setShowMasks(!showMasks)}
          >
            <span className="floating-toolbar__checkbox" /> Masks
          </button>
          <button
            className={`floating-toolbar__btn ${showGT ? "floating-toolbar__btn--active" : ""}`}
            onClick={() => setShowGT(!showGT)}
          >
            <span className="floating-toolbar__checkbox" /> GT
          </button>
        </div>
      </div>

      {/* Sample navigation */}
      {samples.length > 0 && (
        <div className="sample-nav">
          <button
            className="sample-nav__arrow"
            onClick={() => setSelectedSampleIdx(Math.max(0, selectedSampleIdx - 1))}
            disabled={selectedSampleIdx === 0}
          >
            ←
          </button>
          <div className="sample-nav__thumbnails">
            {samples.map((s, i) => (
              <button
                key={i}
                className={`sample-nav__thumb ${i === selectedSampleIdx ? "sample-nav__thumb--active" : ""}`}
                onClick={() => setSelectedSampleIdx(i)}
                title={`${s.attack} sev.${s.severity} | D=${s.degradation?.toFixed(1)}%`}
              >
                <div
                  style={{
                    width: "100%",
                    height: "100%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: "0.5rem",
                    fontFamily: "var(--font-mono)",
                    color: degradationColor(s.degradation || 0),
                    fontWeight: 600,
                  }}
                >
                  {s.degradation?.toFixed(0)}%
                </div>
              </button>
            ))}
          </div>
          <button
            className="sample-nav__arrow"
            onClick={() => setSelectedSampleIdx(Math.min(samples.length - 1, selectedSampleIdx + 1))}
            disabled={selectedSampleIdx >= samples.length - 1}
          >
            →
          </button>
          <span className="text-xs text-secondary" style={{ marginLeft: "4px", whiteSpace: "nowrap", fontFamily: "var(--font-mono)" }}>
            {selectedSampleIdx + 1}/{samples.length}
          </span>
        </div>
      )}
    </>
  );
}
