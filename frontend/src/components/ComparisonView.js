import React, { useState, useRef, useEffect } from "react";
import Image from "next/image";

function degradationColor(d) {
  if (d < 15) return "var(--success)";
  if (d < 40) return "var(--warning)";
  return "var(--danger)";
}

export default function ComparisonView({ report, samples, severity }) {
  const [selectedSampleIdx, setSelectedSampleIdx] = useState(0);
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef(null);

  if (!report) {
    return (
      <div className="placeholder-view">
        <div className="placeholder-view__title">Prediction Comparison</div>
        <div className="placeholder-view__subtitle">
          Run a test to compare clean vs attacked model predictions side by side.
        </div>
      </div>
    );
  }

  const currentSample = samples[selectedSampleIdx] || null;
  const cleanUrl = currentSample?.artifacts?.clean_prediction_url;
  const attackedUrl = currentSample?.artifacts?.attacked_prediction_url;

  const handleSliderChange = (e) => {
    setSliderPos(e.target.value);
  };

  return (
    <>
      <div className="comparison-container tesla-grid-bg" style={{ margin: "var(--space-sm)", borderRadius: "var(--radius-md)", overflow: "hidden", position: "relative" }} ref={containerRef}>
        {/* Clean Image (Background) */}
        <div className="comparison-container__side" style={{ position: "absolute", inset: 0, zIndex: 1 }}>
          <span className="comparison-container__label comparison-container__label--clean">
            Clean prediction
          </span>
          {cleanUrl ? (
            <Image className="comparison-image animate-fade-in" src={cleanUrl} alt="Clean prediction" fill style={{ objectFit: "contain" }} unoptimized />
          ) : (
            <div className="flex-center skeleton" style={{ width: "100%", height: "100%" }} />
          )}
        </div>

        {/* Attacked Image (Foreground with Clip-path) */}
        <div 
          className="comparison-container__side" 
          style={{ 
            position: "absolute", 
            inset: 0, 
            zIndex: 2, 
            clipPath: `polygon(${sliderPos}% 0, 100% 0, 100% 100%, ${sliderPos}% 100%)` 
          }}
        >
          <span className="comparison-container__label comparison-container__label--attacked" style={{ right: "var(--space-sm)", left: "auto" }}>
            Attacked — {currentSample?.attack || "N/A"} (Sev.{currentSample?.severity || severity})
          </span>
          {attackedUrl ? (
            <Image className="comparison-image animate-fade-in" src={attackedUrl} alt="Attacked prediction" fill style={{ objectFit: "contain" }} unoptimized />
          ) : (
            <div className="flex-center skeleton" style={{ width: "100%", height: "100%" }} />
          )}
        </div>

        {/* Slider Input */}
        <input
          type="range"
          min="0"
          max="100"
          value={sliderPos}
          onChange={handleSliderChange}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            opacity: 0,
            zIndex: 10,
            cursor: "ew-resize",
          }}
          aria-label="Image comparison slider"
        />

        {/* Custom Slider Handle */}
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: `${sliderPos}%`,
            width: "2px",
            background: "var(--accent)",
            zIndex: 5,
            pointerEvents: "none",
            transform: "translateX(-50%)",
            boxShadow: "0 0 10px rgba(0,0,0,0.5)",
          }}
        >
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              width: "28px",
              height: "28px",
              borderRadius: "50%",
              background: "var(--accent)",
              border: "2px solid white",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 2px 6px rgba(0,0,0,0.4)",
            }}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ width: 14, height: 14 }}>
              <polyline points="15 18 9 12 15 6" />
              <polyline points="9 18 15 12 9 6" style={{ transform: "rotate(180deg)", transformOrigin: "center" }} />
            </svg>
          </div>
        </div>
      </div>

      {/* Sample navigation */}
      {samples.length > 0 && (
        <div className="sample-nav animate-fade-in-up">
          <button className="sample-nav__arrow" onClick={() => setSelectedSampleIdx(Math.max(0, selectedSampleIdx - 1))} disabled={selectedSampleIdx === 0}>
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
                <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.48rem", fontFamily: "var(--font-mono)", color: degradationColor(s.degradation || 0), fontWeight: 600 }}>
                  {s.degradation?.toFixed(0)}%
                </div>
              </button>
            ))}
          </div>
          <button className="sample-nav__arrow" onClick={() => setSelectedSampleIdx(Math.min(samples.length - 1, selectedSampleIdx + 1))} disabled={selectedSampleIdx >= samples.length - 1}>
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
