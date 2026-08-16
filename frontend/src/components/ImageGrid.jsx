"use client";

import { useState } from "react";
import Image from "next/image";

/**
 * ImageGrid — Lưới 2×2 hiển thị 4 ảnh đối chiếu evidence
 * 1. Ảnh gốc + Ground Truth BBox
 * 2. Ảnh bị tấn công (raw)
 * 3. Ảnh gốc + YOLO Prediction (+ metrics chip)
 * 4. Ảnh tấn công + YOLO Prediction (+ metrics chip)
 */

function GroundTruthOverlay({ groundTruth }) {
  if (groundTruth?.type !== "boxes" || !groundTruth.objects?.length) return null;
  const width = Number(groundTruth.image_width) || 1;
  const height = Number(groundTruth.image_height) || 1;
  const fontSize = Math.max(8, Math.min(width, height) * 0.024);
  const strokeWidth = Math.max(1.5, Math.min(width, height) * 0.004);

  return (
    <svg className="evidence-media__ground-truth" aria-label="Ground truth overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
      {groundTruth.objects.map((object, index) => {
        const [x1, y1, x2, y2] = object.xyxy ?? [];
        if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
        const label = object.label ?? "object";
        const labelWidth = label.length * fontSize * 0.58 + fontSize * 0.45;
        const labelHeight = fontSize * 1.25;
        const tagY = y1 - labelHeight >= 0 ? y1 - labelHeight : y1;

        return (
          <g key={object.object_id ?? index}>
            <rect
              x={x1}
              y={y1}
              width={Math.max(0, x2 - x1)}
              height={Math.max(0, y2 - y1)}
              stroke="var(--gt-color, #22c55e)"
              strokeWidth={strokeWidth}
              fill="rgba(34, 197, 94, 0.08)"
            />
            <rect
              x={x1}
              y={tagY}
              width={labelWidth}
              height={labelHeight}
              fill="rgba(15, 23, 42, 0.85)"
              rx={Math.max(1, fontSize * 0.15)}
            />
            <text
              x={x1 + fontSize * 0.22}
              y={tagY + fontSize * 0.9}
              fontSize={fontSize}
              fill="#22c55e"
              fontWeight="600"
              fontFamily="var(--font-mono, monospace)"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function GridImage({ src, alt, label, labelClass, groundTruth, children }) {
  if (!src) {
    return (
      <div className="image-grid__empty">
        <div className="image-grid__empty-title">{alt}</div>
        <div className="image-grid__empty-detail">Run a test to see results</div>
      </div>
    );
  }
  const width = Number(groundTruth?.image_width) || 1;
  const height = Number(groundTruth?.image_height) || 1;
  const aspect = groundTruth?.image_width && groundTruth?.image_height ? `${width} / ${height}` : "auto";

  return (
    <div className="image-grid__image-wrapper">
      <div className="image-frame" style={{ position: "relative", aspectRatio: aspect }}>
        {label && <span className={`image-grid__label ${labelClass}`}>{label}</span>}
        <Image src={src} alt={alt} fill sizes="(max-width: 900px) 100vw, 35vw" unoptimized style={{ objectFit: "contain" }} />
        {children}
      </div>
    </div>
  );
}

function MetricChips({ metrics, variant = "clean" }) {
  if (!metrics) return null;
  const isDegraded = variant === "attacked";

  return (
    <div className="metric-chips animate-fade-in-up">
      {metrics.ap50 != null && (
        <span className={`metric-chip ${isDegraded ? "metric-chip--danger" : "metric-chip--success"}`}>
          <span className="metric-chip__label">AP50</span>
          {Number(metrics.ap50).toFixed(2)}
        </span>
      )}
      {metrics.map50_95 != null && (
        <span className={`metric-chip ${isDegraded ? "metric-chip--danger" : "metric-chip--success"}`}>
          <span className="metric-chip__label">mAP50-95</span>
          {Number(metrics.map50_95).toFixed(2)}
        </span>
      )}
      {metrics.iou != null && (
        <span className="metric-chip">
          <span className="metric-chip__label">IoU</span>
          {Number(metrics.iou).toFixed(2)}
        </span>
      )}
      {metrics.detected_boxes != null && (
        <span className="metric-chip">
          <span className="metric-chip__label">Boxes</span>
          {metrics.detected_boxes}
        </span>
      )}
      {metrics.degradation_percent != null && (
        <span className="metric-chip metric-chip--danger">
          <span className="metric-chip__label">D</span>
          -{Number(metrics.degradation_percent).toFixed(1)}%
        </span>
      )}
      {metrics.lost_boxes != null && metrics.lost_boxes > 0 && (
        <span className="metric-chip metric-chip--danger">
          <span className="metric-chip__label">Lost</span>
          {metrics.lost_boxes}
        </span>
      )}
    </div>
  );
}

export default function ImageGrid({ report, samples, selectedIndex = 0, onSelectSample }) {
  const sample = samples?.[selectedIndex] || null;
  const artifacts = sample?.artifacts ?? {};
  const attack = report?.cells?.find(
    (cell) => cell.attack === sample?.attack && cell.severity === sample?.severity
  ) ?? report?.cells?.at(-1) ?? null;
  const benchmarkAvailable = report?.benchmark_metrics_available !== false;

  if (!report) {
    return (
      <div className="placeholder-view">
        <div className="placeholder-view__title">AdverTest</div>
        <div className="placeholder-view__subtitle">
          Configure task, model, dataset and attack on the left panel. Then press Run to compare clean vs attacked predictions.
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="image-grid">
        {/* Cell 1: Clean + Ground Truth */}
        <div className="image-grid__cell">
          <GridImage
            src={artifacts.clean_input_url}
            alt="Clean input with ground truth"
            label="Clean + Ground Truth"
            labelClass="image-grid__label--gt"
            groundTruth={sample?.ground_truth}
          >
            <GroundTruthOverlay groundTruth={sample?.ground_truth} />
          </GridImage>
        </div>

        {/* Cell 2: Attacked Raw */}
        <div className="image-grid__cell">
          <GridImage
            src={artifacts.attacked_input_url}
            alt="Attacked input"
            label="Attacked Input"
            labelClass="image-grid__label--attacked"
            groundTruth={sample?.ground_truth}
          />
        </div>

        {/* Cell 3: Clean + YOLO Prediction + Metrics */}
        <div className="image-grid__cell">
          <GridImage
            src={artifacts.clean_prediction_url}
            alt="Clean model prediction"
            label="Clean Prediction"
            labelClass="image-grid__label--pred"
            groundTruth={sample?.ground_truth}
          />
          {benchmarkAvailable && <MetricChips metrics={report?.metrics?.clean} variant="clean" />}
        </div>

        {/* Cell 4: Attacked + YOLO Prediction + Metrics */}
        <div className="image-grid__cell">
          <GridImage
            src={artifacts.attacked_prediction_url}
            alt="Attacked model prediction"
            label="Attacked Prediction"
            labelClass="image-grid__label--attacked-pred"
            groundTruth={sample?.ground_truth}
          />
          {benchmarkAvailable && <MetricChips metrics={attack?.metrics ?? report?.metrics?.attacked} variant="attacked" />}
        </div>
      </div>

      {/* Sample Navigation */}
      {samples && samples.length > 1 && (
        <div className="sample-nav">
          <button className="sample-nav__arrow" onClick={() => onSelectSample?.(Math.max(0, selectedIndex - 1))} disabled={selectedIndex === 0}>
            ←
          </button>
          <div className="sample-nav__thumbnails">
            {samples.map((s, i) => (
              <button
                key={i}
                className={`sample-nav__thumb ${i === selectedIndex ? "sample-nav__thumb--active" : ""}`}
                onClick={() => onSelectSample?.(i)}
                title={`${s.attack} sev.${s.severity}`}
              >
                <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.48rem", fontFamily: "var(--font-mono)", color: "var(--text-tertiary)", fontWeight: 600 }}>
                  {i + 1}
                </div>
              </button>
            ))}
          </div>
          <button className="sample-nav__arrow" onClick={() => onSelectSample?.(Math.min(samples.length - 1, selectedIndex + 1))} disabled={selectedIndex >= samples.length - 1}>
            →
          </button>
          <span className="text-xs text-secondary" style={{ marginLeft: "4px", whiteSpace: "nowrap", fontFamily: "var(--font-mono)" }}>
            {selectedIndex + 1}/{samples.length}
          </span>
        </div>
      )}
    </>
  );
}
