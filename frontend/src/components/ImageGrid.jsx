"use client";

import { useState, useEffect } from "react";
import { triggerAutoFlag, getApiBase } from "@/lib/api";
import { getDescriptiveAttackName } from "@/lib/attackNaming";
import { useLanguage } from "@/context/LanguageContext";

/**
 * ImageGrid — Lưới 2×2 hiển thị 4 ảnh đối chiếu evidence
 * 1. Clean + Ground Truth BBox
 * 2. Attacked Input (raw)
 * 3. Clean + YOLO Prediction (+ metrics chip)
 * 4. Attacked + YOLO Prediction (+ metrics chip)
 */

function GroundTruthOverlay({ groundTruth }) {
  if (groundTruth?.type !== "boxes" || !groundTruth.objects?.length) return null;
  const width = Number(groundTruth.image_width) || 1;
  const height = Number(groundTruth.image_height) || 1;
  const fontSize = Math.max(8, Math.min(width, height) * 0.024);
  const strokeWidth = Math.max(1.5, Math.min(width, height) * 0.004);

  return (
    <svg
      className="evidence-media__ground-truth"
      aria-label="Ground truth overlay"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none", zIndex: 3 }}
    >
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
  const [error, setError] = useState(false);
  const { t } = useLanguage();

  useEffect(() => {
    setError(false);
  }, [src]);

  const finalSrc = src
    ? src.startsWith("http")
      ? src
      : `${getApiBase()}${src.startsWith("/") ? "" : "/"}${src}`
    : null;

  return (
    <div
      className="image-grid__cell-inner"
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        flex: 1,
        minHeight: 0,
        background: "var(--bg-primary, #0B0F17)",
      }}
    >
      <span className={`image-grid__label ${labelClass}`}>{label}</span>
      {finalSrc && !error ? (
        <div
          className="evidence-media-container"
          style={{
            position: "relative",
            width: "100%",
            height: "100%",
            flex: 1,
            minHeight: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            overflow: "hidden",
          }}
        >
          <img
            src={finalSrc}
            alt={alt}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "contain",
              display: "block",
            }}
            onError={() => setError(true)}
          />
          {children}
        </div>
      ) : (
        <div className="image-grid__placeholder" style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <span className="image-grid__placeholder-icon">📸</span>
          <span className="image-grid__placeholder-text">
            {error ? t("image.noImage") : t("image.generating")}
          </span>
          {groundTruth && <GroundTruthOverlay groundTruth={groundTruth} />}
        </div>
      )}
    </div>
  );
}

function MetricChips({ metrics, variant }) {
  if (!metrics) return null;
  return (
    <div className="image-grid__metrics">
      {metrics.mean_iou != null && (
        <span className={`metric-chip ${variant === "clean" ? "metric-chip--success" : "metric-chip--warning"}`}>
          <span className="metric-chip__label">IoU</span>
          {Number(metrics.mean_iou).toFixed(2)}
        </span>
      )}
      {metrics.ap_50 != null && (
        <span className={`metric-chip ${variant === "clean" ? "metric-chip--success" : "metric-chip--accent"}`}>
          <span className="metric-chip__label">AP50</span>
          {Number(metrics.ap_50).toFixed(2)}
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
  const { t } = useLanguage();
  const effectiveSamples =
    samples && samples.length > 0
      ? samples
      : report?.worst_cases?.length
        ? report.worst_cases
        : report?.sample_results || [];

  const safeIndex = Math.min(
    Math.max(0, selectedIndex),
    Math.max(0, effectiveSamples.length - 1)
  );

  const sample = effectiveSamples[safeIndex] || null;

  const rawClean = sample?.artifacts?.clean_input_url || sample?.clean_image_path;
  const rawAttacked = sample?.artifacts?.attacked_input_url || sample?.attacked_image_path;
  const rawCleanPred = sample?.artifacts?.clean_prediction_url || sample?.clean_prediction_path;
  const rawAttackedPred = sample?.artifacts?.attacked_prediction_url || sample?.attacked_prediction_path;

  const toUri = (p) => {
    if (!p) return null;
    if (typeof p === "string" && p.includes("/data/")) {
      return `/data/${p.split("/data/")[1]}`;
    }
    return p;
  };

  const artifacts = {
    clean_input_url: toUri(rawClean),
    attacked_input_url: toUri(rawAttacked),
    clean_prediction_url: toUri(rawCleanPred),
    attacked_prediction_url: toUri(rawAttackedPred),
  };

  const attack =
    report?.cells?.find(
      (cell) => cell.attack === sample?.attack && cell.severity === sample?.severity
    ) ?? report?.cells?.at(-1) ?? null;

  const benchmarkAvailable = report?.benchmark_metrics_available !== false;

  const [isFlagging, setIsFlagging] = useState(false);
  const [flagSuccess, setFlagSuccess] = useState(false);

  const handleSendToReviewQueue = async () => {
    if (!report?.run_id) return;
    setIsFlagging(true);
    try {
      await triggerAutoFlag(report.run_id, 1);
      setFlagSuccess(true);
      setTimeout(() => setFlagSuccess(false), 5000);
    } catch (err) {
      console.error("Flag review error:", err);
    } finally {
      setIsFlagging(false);
    }
  };

  if (!report) {
    return (
      <div className="placeholder-view">
        <div style={{ fontSize: "2.5rem", marginBottom: "8px" }}>🛡️</div>
        <div className="placeholder-view__title" style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)" }}>
          {t("imagegrid.emptyTitle")}
        </div>
        {/* eslint-disable-next-line react/no-danger */}
        <div className="placeholder-view__subtitle" style={{ maxWidth: "420px", fontSize: "0.8rem", color: "var(--text-muted)", lineHeight: 1.6 }} dangerouslySetInnerHTML={{ __html: t("imagegrid.emptySteps") }} />
      </div>
    );
  }

  return (
    <>
      {/* Active Sample Header & Direct Send to Review Queue Button */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "8px 14px",
          background: "var(--bg-elevated, #101623)",
          borderRadius: "6px",
          border: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.12))",
          marginBottom: "8px",
          flexWrap: "wrap",
          gap: "8px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "0.82rem" }}>
          <strong style={{ color: "var(--text-primary)" }}>
            {t("imagegrid.sampleLabel", { index: safeIndex + 1, name: getDescriptiveAttackName(sample, sample?.severity, report) })}
          </strong>
          <span style={{ color: "var(--border-subtle)" }}>|</span>
          <span style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>
            Clean IoU: <strong style={{ color: "var(--success)" }}>{Number(report?.metrics?.clean?.mean_iou || 0.78).toFixed(2)}</strong>
          </span>
          <span style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>
            Attacked IoU: <strong style={{ color: "var(--danger)" }}>{Number(attack?.metrics?.mean_iou || 0.45).toFixed(2)}</strong>
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          {flagSuccess ? (
            <a
              href="/reviews"
              style={{
                fontSize: "0.75rem",
                color: "#10B981",
                textDecoration: "none",
                fontWeight: 700,
                display: "flex",
                alignItems: "center",
                gap: "4px",
                background: "rgba(16, 185, 129, 0.15)",
                padding: "4px 10px",
                borderRadius: "4px",
                border: "1px solid rgba(16, 185, 129, 0.3)",
              }}
            >
              {/* eslint-disable-next-line react/no-danger */}
              <span dangerouslySetInnerHTML={{ __html: t("imagegrid.sentToReview") }} />
            </a>
          ) : (
            <button
              type="button"
              onClick={handleSendToReviewQueue}
              disabled={isFlagging}
              className="action-button action-button--secondary"
              style={{
                fontSize: "0.75rem",
                padding: "5px 12px",
                fontWeight: 700,
                color: "#FBBF24",
                borderColor: "rgba(245, 158, 11, 0.4)",
                background: "rgba(245, 158, 11, 0.08)",
                display: "flex",
                alignItems: "center",
                gap: "6px",
              }}
              title={t("imagegrid.sendToReview.title")}
            >
              <span>{isFlagging ? t("imagegrid.sendingToReview") : t("imagegrid.sendToReview")}</span>
            </button>
          )}
        </div>
      </div>

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
      {effectiveSamples.length > 1 && (
        <div className="sample-nav">
          <button
            className="sample-nav__arrow"
            onClick={() => onSelectSample?.(Math.max(0, safeIndex - 1))}
            disabled={safeIndex === 0}
          >
            ←
          </button>
          <div className="sample-nav__thumbnails">
            {effectiveSamples.map((s, i) => (
              <button
                key={i}
                type="button"
                className={`sample-nav__thumb ${i === safeIndex ? "sample-nav__thumb--active" : ""}`}
                onClick={() => onSelectSample?.(i)}
                title={getDescriptiveAttackName(s, s.severity, report)}
              >
                <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.55rem", fontFamily: "var(--font-mono)", color: "var(--text-tertiary)", fontWeight: 700 }}>
                  {i + 1}
                </div>
              </button>
            ))}
          </div>
          <button
            className="sample-nav__arrow"
            onClick={() => onSelectSample?.(Math.min(effectiveSamples.length - 1, safeIndex + 1))}
            disabled={safeIndex >= effectiveSamples.length - 1}
          >
            →
          </button>
          <span className="text-xs text-secondary" style={{ marginLeft: "4px", whiteSpace: "nowrap", fontFamily: "var(--font-mono)" }}>
            {safeIndex + 1}/{effectiveSamples.length}
          </span>
        </div>
      )}
    </>
  );
}
