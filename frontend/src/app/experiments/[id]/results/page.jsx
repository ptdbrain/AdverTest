"use client";

import {
  BarChart3,
  Boxes,
  CalendarClock,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Cpu,
  Crosshair,
  Database,
  Download,
  Fingerprint,
  Flag,
  Gauge,
  History,
  Image as ImageIcon,
  Info,
  Layers,
  Link2,
  PieChart,
  RefreshCw,
  RotateCcw,
  Repeat2,
  Search,
  ShieldAlert,
  SlidersHorizontal,
  Target,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import React, { useCallback, useEffect, useRef, useState } from "react";
import BevViewer from "@/components/BevViewer";
import DemoFixtureBadge from "@/components/common/DemoFixtureBadge";
import ResearcherNotePanel from "@/components/ResearcherNotePanel";
import RunHistoryBar from "@/components/RunHistoryBar";
import SampleReviewPanel from "@/components/reviews/SampleReviewPanel";
import { useProject } from "@/context/ProjectContext";
import { artifactUrl, getRunReport, getRunSamples, getSession, triggerAutoFlag } from "@/lib/api";
import { cn } from "@/lib/utils";

const NO_DATA = "Chưa có dữ liệu";
const STATUS_LABELS = {
  COMPLETED: "Hoàn thành",
  RUNNING: "Đang chạy",
  PREPARING: "Đang chuẩn bị",
  GENERATING: "Đang tạo nhiễu",
  INFERENCING: "Đang suy luận",
  FAILED: "Thất bại",
  CANCELLED: "Đã hủy",
};

function sampleArtifact(sample, key) {
  const direct = sample?.artifacts?.[key];
  const pathKey = {
    clean_input_url: "clean_image_path",
    attacked_input_url: "attacked_image_path",
    clean_prediction_url: "clean_prediction_path",
    attacked_prediction_url: "attacked_prediction_path",
  }[key];
  const pathValue = pathKey ? sample?.[pathKey] : null;
  const value = direct || pathValue;
  if (!value) return null;
  if (/^(https?:\/\/|data:|blob:)/i.test(value)) return value;
  const normalized = typeof value === "string" ? value.replace(/\\/g, "/") : "";
  const dataIndex = normalized.indexOf("/data/");
  return artifactUrl(dataIndex >= 0 ? normalized.slice(dataIndex) : value);
}

function firstValue(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== "");
}

function displayValue(value) {
  if (value === undefined || value === null || value === "") return NO_DATA;
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : NO_DATA;
  return String(value);
}

function displayStatus(value) {
  if (value === undefined || value === null || value === "") return NO_DATA;
  return STATUS_LABELS[String(value).toUpperCase()] || String(value);
}

function displayDate(value) {
  if (!value) return NO_DATA;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? displayValue(value) : date.toLocaleString("vi-VN");
}

function displayDuration(value) {
  if (value === undefined || value === null || value === "") return NO_DATA;
  if (typeof value !== "number") return String(value);
  return `${value.toFixed(2)} giây`;
}

function buildExperimentMetadata(expId, activeExperiment, lastSession) {
  const report = lastSession?.report || {};
  const sample = report.samples?.[0] || report.sample_results?.[0] || lastSession?.samples?.[0] || {};
  const groundTruth = sample.ground_truth || {};
  const attack = activeExperiment?.attackQueue?.[0] || {};
  const recipeStep = lastSession?.recipe?.steps?.[0] || {};
  const parameters = { ...(recipeStep.parameters || {}), ...(attack.parameters || {}) };
  const attackNames = activeExperiment?.attackQueue
    ?.map((item) => item.name || item.id)
    .filter(Boolean)
    .join(" + ");
  const hasRunEvidence = Boolean(
    Object.keys(report).length > 0 &&
      (report.run_id || report.sample_results?.length || report.samples?.length || lastSession?.runId),
  );
  const cellImpacts = (report.cells || [])
    .map((cell) =>
      firstValue(
        cell.degradation_percent,
        cell.degradation_ratio !== undefined ? Number(cell.degradation_ratio) * 100 : undefined,
        cell.degradation !== undefined ? Number(cell.degradation) * 100 : undefined,
      ),
    )
    .map(Number)
    .filter(Number.isFinite);
  const derivedImpact = cellImpacts.length > 0 ? Math.max(...cellImpacts) : undefined;

  const width = firstValue(groundTruth.image_width, sample.image_width, report.image_width, report.width);
  const height = firstValue(groundTruth.image_height, sample.image_height, report.image_height, report.height);
  const imageSize = width && height ? `${width} × ${height}` : undefined;

  return {
    details: [
      {
        label: "Session / Run ID",
        value: firstValue(report.run_id, lastSession?.runId, expId),
        icon: Fingerprint,
        valueClass: "text-blue-700",
        mono: true,
      },
      {
        label: "Mô hình",
        value: firstValue(
          activeExperiment?.selectedModelName,
          report.model_version,
          report.model,
          lastSession?.selectedModelVersion,
          lastSession?.selectedModelFamily,
        ),
        icon: Cpu,
        valueClass: "text-blue-700",
      },
      {
        label: "Bộ dữ liệu",
        value: firstValue(
          activeExperiment?.selectedDatasetName,
          report.dataset_name,
          report.dataset,
          lastSession?.selectedDataset,
        ),
        icon: Database,
      },
    ],
    status: firstValue(
      report.status,
      hasRunEvidence ? lastSession?.runStatus : undefined,
      hasRunEvidence ? activeExperiment?.status : undefined,
    ),
    impact: firstValue(report.map_drop_pct, report.degradation_pct, report.robustness_drop_pct, derivedImpact),
    selectedTask: firstValue(
      activeExperiment?.selectedTask,
      report.task_id,
      report.provenance?.run_config?.task_id,
      lastSession?.task_id,
    ),
  };
}

const overlayColors = ["#22c55e", "#a855f7", "#14b8a6", "#f59e0b", "#3b82f6"];

function sampleDimensions(sample) {
  const width = Number(sample?.ground_truth?.image_width ?? sample?.image_width);
  const height = Number(sample?.ground_truth?.image_height ?? sample?.image_height);
  return width > 0 && height > 0 ? { width, height } : null;
}

function predictionBoxes(prediction) {
  return (prediction?.boxes || []).filter((box) => {
    const coordinates = (box.xyxy || []).map(Number);
    return (
      coordinates.length === 4 &&
      coordinates.every(Number.isFinite) &&
      coordinates[2] > coordinates[0] &&
      coordinates[3] > coordinates[1]
    );
  });
}

function predictionBoxes3d(prediction) {
  return (prediction?.boxes3d || []).filter((box) =>
    [box?.x, box?.y, box?.z, box?.length, box?.width, box?.height, box?.yaw].every((value) =>
      Number.isFinite(Number(value)),
    ),
  );
}

function predictionInstances(prediction) {
  return Array.isArray(prediction?.instances) ? prediction.instances : [];
}

function formatScore(score) {
  const value = Number(score);
  return Number.isFinite(value) ? value.toFixed(2) : NO_DATA;
}

function DetectionOverlays({ prediction, sample, attacked = false }) {
  const dimensions = sampleDimensions(sample);
  const boxes = predictionBoxes(prediction);
  if (!dimensions || boxes.length === 0) return null;

  return (
    <svg
      data-testid={attacked ? "attacked-prediction-overlays" : "clean-prediction-overlays"}
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
      preserveAspectRatio="xMidYMid meet"
      aria-label={attacked ? "Attacked model prediction boxes" : "Clean model prediction boxes"}
    >
      {boxes.map((box, index) => {
        const [x1, y1, x2, y2] = box.xyxy.map(Number);
        const color = overlayColors[index % overlayColors.length];
        const label = box.label || NO_DATA;
        const text = `${label} ${formatScore(box.score)}`;
        return (
          <g key={`${label}-${index}`}>
            <rect
              data-testid={`${attacked ? "attacked" : "clean"}-prediction-box-${index}`}
              x={x1}
              y={y1}
              width={x2 - x1}
              height={y2 - y1}
              fill={`${color}22`}
              stroke={color}
              strokeWidth={Math.max(1.5, dimensions.width * 0.003)}
              strokeDasharray={attacked ? "6 3" : undefined}
            />
            <text
              data-testid={`${attacked ? "attacked" : "clean"}-prediction-label-${index}`}
              x={x1}
              y={Math.max(12, y1 - 4)}
              fill={color}
              fontSize={Math.max(9, dimensions.width * 0.018)}
              fontWeight="700"
              paintOrder="stroke"
              stroke="#0f172a"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              {text}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function EvidenceFrame({ sample, prediction, attacked, zoom = 1 }) {
  const dimensions = sampleDimensions(sample);
  const artifact = sampleArtifact(sample, attacked ? "attacked_input_url" : "clean_input_url");
  const testPrefix = attacked ? "attacked" : "clean";
  if (!artifact) {
    return (
      <div
        data-testid={`${testPrefix}-evidence-empty`}
        style={{ aspectRatio: dimensions ? `${dimensions.width} / ${dimensions.height}` : "16 / 9" }}
        className="flex min-h-[160px] w-full items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 p-4 text-center text-xs text-slate-500"
      >
        <span>
          <strong className="block text-slate-700">Chưa có bằng chứng ảnh</strong>
          <span>Run chưa trả về artifact cho mẫu này.</span>
        </span>
      </div>
    );
  }

  return (
    <div
      data-testid={`${testPrefix}-image-viewport`}
      style={{ aspectRatio: dimensions ? `${dimensions.width} / ${dimensions.height}` : "16 / 9" }}
      className="relative flex min-h-[160px] w-full items-center justify-center overflow-hidden rounded-lg border border-slate-300 bg-slate-950 shadow-inner group"
    >
      <div
        className="relative h-full w-full flex items-center justify-center transition-transform duration-200 ease-out"
        style={{
          transform: zoom > 1 ? `scale(${zoom})` : "none",
          transformOrigin: "center center",
        }}
      >
        <img
          data-testid={`${testPrefix}-result-image`}
          src={artifact}
          alt={attacked ? "Ảnh mẫu sau tấn công" : "Ảnh mẫu gốc"}
          className="h-full w-full object-contain select-none"
        />
        <DetectionOverlays prediction={prediction} sample={sample} attacked={attacked} />
      </div>
      <span className="absolute bottom-2 left-2 rounded bg-slate-950/85 px-2 py-1 font-mono text-[10px] font-bold tracking-wide text-white border border-slate-700/50 shadow pointer-events-none z-10">
        {sample?.sample_id || NO_DATA}
      </span>
      {dimensions && (
        <span className="absolute bottom-2 right-2 rounded bg-slate-950/85 px-2 py-1 font-mono text-[10px] text-slate-300 border border-slate-700/50 shadow pointer-events-none z-10">
          {dimensions.width} × {dimensions.height}
        </span>
      )}
    </div>
  );
}

function numericValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function averageConfidence(boxes) {
  if (boxes.length === 0) return null;
  return boxes.reduce((sum, box) => sum + (numericValue(box.score) || 0), 0) / boxes.length;
}

function percentageDelta(value, baseline) {
  if (value === null || baseline === null || baseline === 0) return null;
  const delta = ((value - baseline) / Math.abs(baseline)) * 100;
  return `${delta < 0 ? "↓" : "↑"} ${Math.abs(delta).toFixed(1)}%`;
}

function computeMeanBboxIoU(predA, predB) {
  const boxesA = predictionBoxes(predA);
  const boxesB = predictionBoxes(predB);
  if (boxesA.length === 0 || boxesB.length === 0) return null;

  let totalIou = 0;
  for (const boxA of boxesA) {
    const [ax1, ay1, ax2, ay2] = (boxA.xyxy || []).map(Number);
    const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
    let maxIou = 0;
    for (const boxB of boxesB) {
      const [bx1, by1, bx2, by2] = (boxB.xyxy || []).map(Number);
      const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
      const ix1 = Math.max(ax1, bx1);
      const iy1 = Math.max(ay1, by1);
      const ix2 = Math.min(ax2, bx2);
      const iy2 = Math.min(ay2, by2);
      const interArea = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
      const unionArea = areaA + areaB - interArea;
      const iou = unionArea > 0 ? interArea / unionArea : 0;
      if (iou > maxIou) maxIou = iou;
    }
    totalIou += maxIou;
  }
  return totalIou / boxesA.length;
}

function predictionMetrics(report, sample, attacked, taskId = "detection2d") {
  const cleanPrediction = sample?.clean_prediction;
  const attackedPrediction = sample?.attacked_prediction;
  const prediction = attacked ? attackedPrediction : cleanPrediction;
  const boxes = predictionBoxes(prediction);
  const cleanBoxes = predictionBoxes(cleanPrediction);
  const attackedBoxes = predictionBoxes(attackedPrediction);
  const cleanAp = numericValue(report?.metrics?.clean?.ap50 ?? report?.ap_clean);
  const cell =
    report?.cells?.find((item) => item.attack === sample?.attack && item.severity === sample?.severity) ??
    report?.cells?.at(-1);
  const attackedAp = numericValue(cell?.metrics?.ap50 ?? cell?.ap);

  if (taskId === "segmentation") {
    const instances = predictionInstances(prediction);
    const cleanInstances = predictionInstances(cleanPrediction);
    const cleanMiou = numericValue(report?.metrics?.clean?.miou ?? report?.metrics?.clean?.mIoU);
    const attackedMiou = numericValue(cell?.metrics?.miou ?? cell?.metrics?.mIoU);
    const confidence = averageConfidence(instances);
    const cleanConfidence = averageConfidence(cleanInstances);
    return [
      {
        label: "Số instance mask",
        value: String(instances.length),
        trend: attacked ? percentageDelta(instances.length, cleanInstances.length) : null,
        icon: Layers,
      },
      {
        label: "Confidence mask TB",
        value: confidence === null ? NO_DATA : confidence.toFixed(2),
        trend: attacked ? percentageDelta(confidence, cleanConfidence) : null,
        icon: Target,
      },
      {
        label: "mIoU",
        value: (attacked ? attackedMiou : cleanMiou) === null ? NO_DATA : (attacked ? attackedMiou : cleanMiou).toFixed(2),
        trend: attacked ? percentageDelta(attackedMiou, cleanMiou) : null,
        icon: PieChart,
      },
      {
        label: "Pixel accuracy",
        value: NO_DATA,
        trend: null,
        icon: BarChart3,
      },
    ];
  }

  if (taskId === "detection3d") {
    const boxes3d = predictionBoxes3d(prediction);
    const cleanBoxes3d = predictionBoxes3d(cleanPrediction);
    const cleanAp3d = numericValue(report?.metrics?.clean?.kitti_3d_ap);
    const attackedAp3d = numericValue(cell?.metrics?.kitti_3d_ap ?? cell?.ap);
    const cleanBevIou = numericValue(report?.metrics?.clean?.mean_bev_iou ?? report?.metrics?.clean?.bev_iou);
    const attackedBevIou = numericValue(cell?.metrics?.mean_bev_iou ?? cell?.metrics?.bev_iou);
    const confidence = averageConfidence(boxes3d);
    const cleanConfidence = averageConfidence(cleanBoxes3d);
    return [
      {
        label: "Số hộp 3D",
        value: String(boxes3d.length),
        trend: attacked ? percentageDelta(boxes3d.length, cleanBoxes3d.length) : null,
        icon: Box,
      },
      {
        label: "Confidence 3D TB",
        value: confidence === null ? NO_DATA : confidence.toFixed(2),
        trend: attacked ? percentageDelta(confidence, cleanConfidence) : null,
        icon: Target,
      },
      {
        label: "KITTI 3D AP",
        value: (attacked ? attackedAp3d : cleanAp3d) === null ? NO_DATA : (attacked ? attackedAp3d : cleanAp3d).toFixed(2),
        trend: attacked ? percentageDelta(attackedAp3d, cleanAp3d) : null,
        icon: BarChart3,
      },
      {
        label: "BEV IoU",
        value: (attacked ? attackedBevIou : cleanBevIou) === null ? NO_DATA : (attacked ? attackedBevIou : cleanBevIou).toFixed(2),
        trend: attacked ? percentageDelta(attackedBevIou, cleanBevIou) : null,
        icon: Crosshair,
      },
    ];
  }

  // Calculate or resolve mIoU
  const groundTruthBoxes = sample?.ground_truth?.objects?.map((o) => ({ xyxy: o.xyxy, label: o.label })) || [];
  const cleanIouGt =
    groundTruthBoxes.length > 0 ? computeMeanBboxIoU(cleanPrediction, { boxes: groundTruthBoxes }) : null;
  const attackedIouGt =
    groundTruthBoxes.length > 0 ? computeMeanBboxIoU(attackedPrediction, { boxes: groundTruthBoxes }) : null;

  const cleanMiou = numericValue(report?.metrics?.clean?.miou ?? report?.metrics?.clean?.mIoU) ?? cleanIouGt;
  const attackedMiou = numericValue(cell?.metrics?.miou ?? cell?.metrics?.mIoU) ?? attackedIouGt;

  const count = boxes.length > 0 || prediction?.boxes ? boxes.length : null;
  const confidence = prediction?.boxes ? averageConfidence(boxes) : null;

  return [
    {
      label: "Số lượng bbox",
      value: count === null ? NO_DATA : String(count),
      trend: attacked ? percentageDelta(count, cleanBoxes.length) : null,
      icon: Boxes,
    },
    {
      label: "Confidence TB",
      value: confidence === null ? NO_DATA : confidence.toFixed(2),
      trend: attacked ? percentageDelta(confidence, averageConfidence(cleanBoxes)) : null,
      icon: Target,
    },
    {
      label: "mAP@0.5",
      value: (attacked ? attackedAp : cleanAp) === null ? NO_DATA : (attacked ? attackedAp : cleanAp).toFixed(2),
      trend: attacked ? percentageDelta(attackedAp, cleanAp) : null,
      icon: BarChart3,
    },
    {
      label: "mIoU (Độ khớp)",
      value:
        (attacked ? attackedMiou : cleanMiou) === null ? NO_DATA : (attacked ? attackedMiou : cleanMiou).toFixed(2),
      trend: attacked ? percentageDelta(attackedMiou, cleanMiou) : null,
      icon: PieChart,
    },
  ];
}

function MetricCell({ metric, attacked }) {
  const Icon = metric.icon;
  return (
    <div
      className={cn(
        "flex min-h-[64px] h-[64px] flex-col justify-between rounded-lg border p-1.5",
        attacked ? "border-red-100 bg-red-50/30" : "border-slate-200 bg-slate-50/70",
      )}
    >
      <span className="truncate text-[10px] font-medium text-slate-500">{metric.label}</span>
      <span className="my-0.5 flex items-center gap-1 text-sm font-bold text-slate-900">
        <Icon className={cn("h-3.5 w-3.5", attacked ? "text-red-600" : "text-blue-600")} aria-hidden="true" />
        {metric.value}
      </span>
      <span className={cn("text-[9px] font-bold", attacked ? "text-red-600" : "text-emerald-600")}>
        {metric.trend || " "}
      </span>
    </div>
  );
}

function PredictionCell({ sample, attacked, taskId = "detection2d" }) {
  const prediction = sample?.[attacked ? "attacked_prediction" : "clean_prediction"];
  const items =
    taskId === "segmentation"
      ? predictionInstances(prediction)
      : taskId === "detection3d"
        ? predictionBoxes3d(prediction)
        : predictionBoxes(prediction);
  return (
    <div
      className={cn(
        "flex min-h-[64px] h-[64px] flex-col justify-between overflow-hidden rounded-lg border p-1.5",
        attacked ? "border-red-100 bg-red-50/30" : "border-slate-200 bg-slate-50/70",
      )}
    >
      <span className="block truncate text-[10px] font-medium text-slate-500">Top dự đoán</span>
      <div className="space-y-0.5 font-mono text-[9px] text-slate-700">
        {items.length === 0 ? (
          <span className="text-slate-400">{NO_DATA}</span>
        ) : (
          items.slice(0, 2).map((item, index) => (
            <div key={`${item.label || NO_DATA}-${index}`} className="flex justify-between gap-1">
              <span className="truncate">
                {index + 1}. {item.label || NO_DATA}
              </span>
              <span className={cn("font-semibold shrink-0", attacked && "text-red-600")}>{formatScore(item.score)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function ResultPanel({ sample, report, taskId = "detection2d", attacked = false, className = "", zoom = 1 }) {
  const title = attacked ? "Ảnh sau tấn công" : "Ảnh gốc";
  const artifact = sampleArtifact(sample, attacked ? "attacked_input_url" : "clean_input_url");
  const metrics = predictionMetrics(report, sample, attacked, taskId);
  return (
    <section
      data-testid={attacked ? "attacked-result-panel" : "clean-result-panel"}
      className={cn(
        "flex flex-col justify-between h-full min-w-0 rounded-xl border border-slate-200 bg-white p-3 shadow-sm",
        className,
      )}
    >
      <div>
        <div className="mb-2 flex h-7 items-center justify-between gap-2">
          <h2 className={cn("truncate text-sm font-bold", attacked ? "text-red-600" : "text-blue-800")}>{title}</h2>
          <span
            className={cn(
              "flex shrink-0 items-center gap-1.5 rounded-full border px-2 py-1 text-[10px] font-semibold",
              artifact
                ? attacked
                  ? "border-red-200 bg-red-50 text-red-700"
                  : "border-emerald-200 bg-emerald-50 text-emerald-700"
                : "border-slate-200 bg-slate-50 text-slate-500",
            )}
          >
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                artifact ? (attacked ? "bg-red-500" : "bg-emerald-500") : "bg-slate-400",
              )}
              aria-hidden="true"
            />
            {artifact ? (attacked ? "Đã tấn công" : "Có dữ liệu") : NO_DATA}
          </span>
        </div>
        <EvidenceFrame
          sample={sample}
          prediction={sample?.[attacked ? "attacked_prediction" : "clean_prediction"]}
          attacked={attacked}
          zoom={zoom}
        />
      </div>
      <div className="mt-3">
        <div className={cn("mb-2 text-[11px] font-bold", attacked ? "text-red-600" : "text-blue-800")}>
          Kết quả dự đoán ({attacked ? "Sau" : "Trước"} tấn công)
        </div>
        <div
          data-testid={attacked ? "attacked-metrics-strip" : "clean-metrics-strip"}
          className="grid grid-cols-2 gap-1.5 xl:grid-cols-5"
        >
          {metrics.map((metric) => (
            <MetricCell key={metric.label} metric={metric} attacked={attacked} />
          ))}
          <PredictionCell sample={sample} attacked={attacked} taskId={taskId} />
        </div>
      </div>
    </section>
  );
}

function ExperimentRail({ className = "", metadata }) {
  const StatusIcon = metadata?.status ? CheckCircle2 : Info;
  return (
    <aside
      data-testid="experiment-info-panel"
      className={cn(
        "flex min-w-0 flex-col justify-between rounded-xl border border-slate-200 bg-white p-3 shadow-sm",
        className,
      )}
    >
      <div>
        <div className="flex items-center gap-2 border-b border-slate-100 pb-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-50 text-blue-700">
            <Info className="h-4 w-4" aria-hidden="true" />
          </div>
          <h2 className="text-sm font-bold text-slate-900">Ngữ cảnh run</h2>
        </div>
        <dl className="mt-2 divide-y divide-slate-100 text-[10px]">
          {(metadata?.details || buildExperimentMetadata(undefined, null, null).details).map((detail) => {
            const Icon = detail.icon;
            return (
              <div key={detail.label} className="grid grid-cols-2 items-center gap-2 py-1.5">
                <dt className="flex min-w-0 items-center gap-1.5 text-slate-500">
                  <Icon className="h-3.5 w-3.5 shrink-0 text-blue-600" aria-hidden="true" />
                  <span className="truncate">{detail.label}</span>
                </dt>
                <dd
                  className={cn(
                    "min-w-0 text-right font-semibold leading-tight text-slate-800",
                    detail.valueClass,
                    detail.mono && "font-mono text-[9px]",
                  )}
                >
                  {displayValue(detail.value)}
                </dd>
              </div>
            );
          })}
        </dl>
      </div>
      <div className="mt-2 grid grid-cols-2 items-center gap-2 border-t border-slate-100 py-1.5 text-[10px]">
        <div className="flex items-center gap-1.5 text-slate-500">
          <StatusIcon className="h-3.5 w-3.5 shrink-0 text-blue-600" aria-hidden="true" />
          <span>Trạng thái</span>
        </div>
        <div className="text-right font-semibold text-slate-800">{displayStatus(metadata?.status)}</div>
      </div>
    </aside>
  );
}

function SampleNavigator({ samples, activeIndex, onSelectSample, zoom = 1, onZoomChange }) {
  return (
    <div
      data-testid="sample-navigator"
      className="flex flex-col gap-2.5 rounded-xl border border-slate-200 bg-white p-3 shadow-sm sm:flex-row sm:items-center sm:justify-between"
    >
      <div className="flex items-center gap-2.5">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-50 text-blue-700">
          <Layers className="h-4 w-4" aria-hidden="true" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xs font-bold text-slate-900">
              {`Bộ ảnh kiểm thử (${Math.max(1, samples?.length || 0)} ảnh)`}
            </h2>
            {samples?.length > 1 && (
              <span className="rounded bg-blue-50 px-1.5 py-0.5 text-[10px] font-bold text-blue-700">
                Đang xem #{activeIndex + 1}
              </span>
            )}
          </div>
          <p className="text-[10px] font-mono text-slate-500 flex items-center flex-wrap gap-1 mt-0.5">
            <span>
              ID:{" "}
              <strong className="text-slate-800">
                {samples?.[activeIndex]?.sample_id || `Sample #${activeIndex + 1}`}
              </strong>
            </span>
            {samples?.[activeIndex]?.attack && (
              <span className="font-sans font-semibold text-[9px] text-blue-700 bg-blue-50 px-1.5 py-0.5 rounded border border-blue-200">
                Đòn: {samples[activeIndex].attack} (Cấp {samples[activeIndex].severity || 3})
              </span>
            )}
          </p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {/* Zoom Controls */}
        <div className="flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-50 p-0.5">
          <span className="text-[10px] font-medium text-slate-500 px-1 flex items-center gap-1">
            <ZoomIn className="h-3 w-3" aria-hidden="true" /> Phóng to:
          </span>
          {[
            { label: "100%", value: 1 },
            { label: "125%", value: 1.25 },
            { label: "150%", value: 1.5 },
            { label: "200%", value: 2 },
          ].map((item) => (
            <button
              key={item.value}
              type="button"
              onClick={() => onZoomChange && onZoomChange(item.value)}
              className={cn(
                "rounded px-2 py-0.5 text-[10px] font-bold transition",
                zoom === item.value
                  ? "bg-white text-blue-600 shadow-sm border border-slate-200"
                  : "text-slate-600 hover:text-slate-900",
              )}
            >
              {item.label}
            </button>
          ))}
        </div>

        {/* Multi-sample navigator buttons */}
        {samples && samples.length > 1 && (
          <div className="flex items-center gap-1.5 overflow-x-auto py-0.5">
            <button
              type="button"
              disabled={activeIndex === 0}
              onClick={() => onSelectSample(activeIndex - 1)}
              className="flex items-center gap-1 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
              title="Mẫu trước"
            >
              <ChevronLeft className="h-3.5 w-3.5" aria-hidden="true" />
              <span className="hidden sm:inline">Trước</span>
            </button>

            <div className="flex items-center gap-1.5 overflow-x-auto max-w-[280px] sm:max-w-[400px] p-0.5">
              {samples.map((s, idx) => {
                const isCurrent = idx === activeIndex;
                const cleanBoxCount = predictionBoxes(s.clean_prediction).length;
                const attackedBoxCount = predictionBoxes(s.attacked_prediction).length;
                const attackLabel = s.attack ? ` (${s.attack})` : "";

                return (
                  <button
                    key={`${s.sample_id || "sample"}-${s.attack || ""}-${idx}`}
                    type="button"
                    onClick={() => onSelectSample(idx)}
                    className={cn(
                      "flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold transition shadow-sm shrink-0",
                      isCurrent
                        ? "bg-blue-600 text-white shadow ring-2 ring-blue-500/50"
                        : "border border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100",
                    )}
                  >
                    <span>
                      Ảnh {idx + 1}
                      {attackLabel}
                    </span>
                    <span
                      className={cn(
                        "rounded px-1 py-0.2 font-mono text-[9px]",
                        isCurrent ? "bg-blue-800/80 text-white" : "bg-slate-200 text-slate-700",
                      )}
                    >
                      {cleanBoxCount}→{attackedBoxCount} bbox
                    </span>
                  </button>
                );
              })}
            </div>

            <button
              type="button"
              disabled={activeIndex === samples.length - 1}
              onClick={() => onSelectSample(activeIndex + 1)}
              className="flex items-center gap-1 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40"
              title="Mẫu tiếp theo"
            >
              <span className="hidden sm:inline">Sau</span>
              <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function EvidenceCard({ title, children, footer }) {
  return (
    <section className="flex min-w-0 flex-col rounded-xl border border-slate-200 bg-white p-3 shadow-sm">
      <h2 className="truncate text-[11px] font-bold text-slate-900">{title}</h2>
      <div className="mt-2 min-h-0 flex-1">{children}</div>
      <div className="mt-2 border-t border-slate-100 pt-2">{footer}</div>
    </section>
  );
}

function analysisArtifact(sample, keys) {
  return keys.map((key) => sample?.artifacts?.[key]).find(Boolean);
}

function AnalysisArtifact({ src, alt, label }) {
  if (!src) {
    return (
      <div
        data-testid="analysis-artifact-empty"
        className="flex h-full min-h-[86px] items-center justify-center rounded-md border border-dashed border-slate-300 bg-slate-50 p-2 text-center text-[10px] text-slate-500"
      >
        Chưa có artifact phân tích
      </div>
    );
  }
  return (
    <div className="relative min-w-0 overflow-hidden rounded-md bg-slate-950">
      <img src={src} alt={alt} className="h-full w-full object-contain" />
      {label && (
        <span className="absolute inset-x-0 bottom-0 bg-slate-950/75 px-1 py-0.5 text-center text-[8px] font-medium text-white">
          {label}
        </span>
      )}
    </div>
  );
}

function getSelectedCropBox(boxes, selectedIndex, width = 1224, height = 370) {
  if (boxes && boxes.length > 0) {
    const box = boxes[selectedIndex] || boxes[0];
    if (box?.xyxy && box.xyxy.length === 4) {
      const [x1, y1, x2, y2] = box.xyxy.map(Number);
      const w = Math.max(30, x2 - x1);
      const h = Math.max(30, y2 - y1);
      const padX = w * 0.15;
      const padY = h * 0.15;
      const cropX = Math.max(0, x1 - padX);
      const cropY = Math.max(0, y1 - padY);
      const cropW = Math.min(width - cropX, w + padX * 2);
      const cropH = Math.min(height - cropY, h + padY * 2);
      return { x: cropX, y: cropY, width: cropW, height: cropH, label: box.label, score: box.score };
    }
  }
  return {
    x: width * 0.35,
    y: height * 0.3,
    width: width * 0.3,
    height: height * 0.4,
    label: "Trung tâm",
    score: null,
  };
}

function infernoColormap(t) {
  const v = Math.max(0, Math.min(1, t));
  if (v < 0.25) {
    const k = v / 0.25;
    return [Math.round(k * 87), Math.round(k * 16), Math.round(4 + k * 106)];
  }
  if (v < 0.5) {
    const k = (v - 0.25) / 0.25;
    return [Math.round(87 + k * 100), Math.round(16 + k * 39), Math.round(110 - k * 26)];
  }
  if (v < 0.75) {
    const k = (v - 0.5) / 0.25;
    return [Math.round(187 + k * 62), Math.round(55 + k * 87), Math.round(84 - k * 75)];
  }
  const k = (v - 0.75) / 0.25;
  return [Math.round(249 + k * 3), Math.round(142 + k * 113), Math.round(9 + k * 155)];
}

function jetColormap(t) {
  const v = Math.max(0, Math.min(1, t));
  if (v < 0.25) {
    const k = v / 0.25;
    return [0, Math.round(k * 255), 255];
  }
  if (v < 0.5) {
    const k = (v - 0.25) / 0.25;
    return [0, 255, Math.round(255 - k * 255)];
  }
  if (v < 0.75) {
    const k = (v - 0.5) / 0.25;
    return [Math.round(k * 255), 255, 0];
  }
  const k = (v - 0.75) / 0.25;
  return [255, Math.round(255 - k * 255), 0];
}

function RealPixelDifferenceCanvas({ cleanUrl, attackedUrl, onMetricsChange }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    let active = true;
    if (!cleanUrl || !attackedUrl) return;

    const img1 = new window.Image();
    const img2 = new window.Image();
    img1.crossOrigin = "anonymous";
    img2.crossOrigin = "anonymous";

    let loaded = 0;
    const onLoad = () => {
      loaded++;
      if (loaded < 2 || !active) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;

      const w = 240;
      const h = Math.round((img1.naturalHeight / (img1.naturalWidth || 1)) * w) || 120;
      canvas.width = w;
      canvas.height = h;

      const off1 = document.createElement("canvas");
      off1.width = w;
      off1.height = h;
      const ctx1 = off1.getContext("2d");
      ctx1.drawImage(img1, 0, 0, w, h);
      const data1 = ctx1.getImageData(0, 0, w, h).data;

      const off2 = document.createElement("canvas");
      off2.width = w;
      off2.height = h;
      const ctx2 = off2.getContext("2d");
      ctx2.drawImage(img2, 0, 0, w, h);
      const data2 = ctx2.getImageData(0, 0, w, h).data;

      const outImg = ctx.createImageData(w, h);
      const out = outImg.data;
      let sumSq = 0;
      let maxDiff = 0;
      const totalPixels = w * h;

      for (let i = 0; i < data1.length; i += 4) {
        const dr = Math.abs(data2[i] - data1[i]) / 255;
        const dg = Math.abs(data2[i + 1] - data1[i + 1]) / 255;
        const db = Math.abs(data2[i + 2] - data1[i + 2]) / 255;
        const delta = (dr + dg + db) / 3;

        sumSq += delta * delta;
        if (delta > maxDiff) maxDiff = delta;

        const [r, g, b] = infernoColormap(Math.min(1.0, delta * 3.5));
        out[i] = r;
        out[i + 1] = g;
        out[i + 2] = b;
        out[i + 3] = 255;
      }
      ctx.putImageData(outImg, 0, 0);

      const l2 = Math.sqrt(sumSq / totalPixels) * 10;
      onMetricsChange?.({ l2: l2.toFixed(2), lInf: maxDiff.toFixed(4) });
    };

    img1.onload = onLoad;
    img2.onload = onLoad;
    img1.src = cleanUrl;
    img2.src = attackedUrl;

    return () => {
      active = false;
    };
  }, [cleanUrl, attackedUrl]);

  return <canvas ref={canvasRef} className="h-full w-full object-contain" />;
}

function RealAttentionHeatmapCanvas({ boxes = [], width = 1224, height = 370, onPeakChange }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = 120;
    const h = Math.round((height / (width || 1)) * w) || 90;
    canvas.width = w;
    canvas.height = h;

    const grid = new Float32Array(w * h);
    let maxVal = 0;

    if (boxes && boxes.length > 0) {
      boxes.forEach((box) => {
        const [bx1, by1, bx2, by2] = (box.xyxy || []).map(Number);
        const score = numericValue(box.score) ?? 0;
        const cx = ((bx1 + bx2) / 2 / width) * w;
        const cy = ((by1 + by2) / 2 / height) * h;
        const rx = Math.max(6, ((bx2 - bx1) / width) * w * 0.9);
        const ry = Math.max(6, ((by2 - by1) / height) * h * 0.9);

        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const dx = (x - cx) / rx;
            const dy = (y - cy) / ry;
            const distSq = dx * dx + dy * dy;
            if (distSq < 9) {
              const val = score * Math.exp(-distSq / 2);
              const idx = y * w + x;
              grid[idx] += val;
              if (grid[idx] > maxVal) maxVal = grid[idx];
            }
          }
        }
      });
    }

    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;
    const normFactor = maxVal > 0 ? 1 / Math.max(1.0, maxVal) : 1;

    for (let i = 0; i < grid.length; i++) {
      const normVal = Math.min(1.0, grid[i] * normFactor);
      const [r, g, b] = jetColormap(normVal);
      const px = i * 4;
      data[px] = r;
      data[px + 1] = g;
      data[px + 2] = b;
      data[px + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);

    onPeakChange?.(maxVal > 0 ? Math.min(1.0, maxVal).toFixed(2) : "0.00");
  }, [boxes, width, height]);

  return <canvas ref={canvasRef} className="h-full w-full object-contain" />;
}

function RealPerturbationCanvas({ cleanUrl, attackedUrl, onStatsChange }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    let active = true;
    if (!cleanUrl || !attackedUrl) return;

    const img1 = new window.Image();
    const img2 = new window.Image();
    img1.crossOrigin = "anonymous";
    img2.crossOrigin = "anonymous";

    let loaded = 0;
    const onLoad = () => {
      loaded++;
      if (loaded < 2 || !active) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;

      const w = 240;
      const h = Math.round((img1.naturalHeight / (img1.naturalWidth || 1)) * w) || 120;
      canvas.width = w;
      canvas.height = h;

      const off1 = document.createElement("canvas");
      off1.width = w;
      off1.height = h;
      const ctx1 = off1.getContext("2d");
      ctx1.drawImage(img1, 0, 0, w, h);
      const data1 = ctx1.getImageData(0, 0, w, h).data;

      const off2 = document.createElement("canvas");
      off2.width = w;
      off2.height = h;
      const ctx2 = off2.getContext("2d");
      ctx2.drawImage(img2, 0, 0, w, h);
      const data2 = ctx2.getImageData(0, 0, w, h).data;

      const outImg = ctx.createImageData(w, h);
      const out = outImg.data;
      let minP = 0;
      let maxP = 0;
      let sumP = 0;
      const totalPixels = w * h;

      for (let i = 0; i < data1.length; i += 4) {
        const dr = (data2[i] - data1[i]) / 255;
        const dg = (data2[i + 1] - data1[i + 1]) / 255;
        const db = (data2[i + 2] - data1[i + 2]) / 255;
        const delta = (dr + dg + db) / 3;

        if (delta < minP) minP = delta;
        if (delta > maxP) maxP = delta;
        sumP += delta;

        const val = Math.max(0, Math.min(255, Math.round(128 + delta * 4 * 128)));
        out[i] = val > 128 ? Math.min(255, val + 35) : Math.max(0, val - 25);
        out[i + 1] = val;
        out[i + 2] = val < 128 ? Math.min(255, 255 - val + 35) : Math.max(0, val - 25);
        out[i + 3] = 255;
      }
      ctx.putImageData(outImg, 0, 0);

      onStatsChange?.({
        min: minP.toFixed(3),
        max: `+${maxP.toFixed(3)}`,
        mean: (sumP / totalPixels).toFixed(4),
      });
    };

    img1.onload = onLoad;
    img2.onload = onLoad;
    img1.src = cleanUrl;
    img2.src = attackedUrl;

    return () => {
      active = false;
    };
  }, [cleanUrl, attackedUrl]);

  return <canvas ref={canvasRef} className="h-full w-full object-contain" />;
}

function RealCropDifferenceCanvas({ cleanUrl, attackedUrl, cropBox }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    let active = true;
    if (!cleanUrl || !attackedUrl || !cropBox) return;

    const img1 = new window.Image();
    const img2 = new window.Image();
    img1.crossOrigin = "anonymous";
    img2.crossOrigin = "anonymous";

    let loaded = 0;
    const onLoad = () => {
      loaded++;
      if (loaded < 2 || !active) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;

      const cw = 120;
      const ch = 120;
      canvas.width = cw;
      canvas.height = ch;

      const off1 = document.createElement("canvas");
      off1.width = cw;
      off1.height = ch;
      const ctx1 = off1.getContext("2d");
      ctx1.drawImage(img1, cropBox.x, cropBox.y, cropBox.width, cropBox.height, 0, 0, cw, ch);
      const data1 = ctx1.getImageData(0, 0, cw, ch).data;

      const off2 = document.createElement("canvas");
      off2.width = cw;
      off2.height = ch;
      const ctx2 = off2.getContext("2d");
      ctx2.drawImage(img2, cropBox.x, cropBox.y, cropBox.width, cropBox.height, 0, 0, cw, ch);
      const data2 = ctx2.getImageData(0, 0, cw, ch).data;

      const outImg = ctx.createImageData(cw, ch);
      const out = outImg.data;

      for (let i = 0; i < data1.length; i += 4) {
        const dr = Math.abs(data2[i] - data1[i]) / 255;
        const dg = Math.abs(data2[i + 1] - data1[i + 1]) / 255;
        const db = Math.abs(data2[i + 2] - data1[i + 2]) / 255;
        const delta = (dr + dg + db) / 3;

        const [r, g, b] = infernoColormap(Math.min(1.0, delta * 4.0));
        out[i] = r;
        out[i + 1] = g;
        out[i + 2] = b;
        out[i + 3] = 255;
      }
      ctx.putImageData(outImg, 0, 0);
    };

    img1.onload = onLoad;
    img2.onload = onLoad;
    img1.src = cleanUrl;
    img2.src = attackedUrl;

    return () => {
      active = false;
    };
  }, [cleanUrl, attackedUrl, cropBox]);

  return <canvas ref={canvasRef} className="h-full w-full object-cover" />;
}

function ConfidenceDropChart({ cleanBoxes = [], attackedBoxes = [], threshold = 0.25 }) {
  if (!cleanBoxes || cleanBoxes.length === 0) {
    return (
      <div className="flex h-full items-center justify-center rounded-md border border-dashed border-slate-200 bg-slate-50 text-[10px] text-slate-400 p-2 text-center">
        Không có đối tượng được phát hiện
      </div>
    );
  }

  const targets = cleanBoxes.slice(0, 3).map((cleanBox, idx) => {
    const cleanScore = numericValue(cleanBox.score) ?? 0;
    const label = cleanBox.label || `Vật thể #${idx + 1}`;

    let bestMatchScore = 0;
    const [ax1, ay1, ax2, ay2] = (cleanBox.xyxy || []).map(Number);
    const cleanArea = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);

    for (const attBox of attackedBoxes) {
      const [bx1, by1, bx2, by2] = (attBox.xyxy || []).map(Number);
      const ix1 = Math.max(ax1, bx1);
      const iy1 = Math.max(ay1, by1);
      const ix2 = Math.min(ax2, bx2);
      const iy2 = Math.min(ay2, by2);
      const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
      const union = cleanArea + Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1) - inter;
      const iou = union > 0 ? inter / union : 0;
      if (iou > 0.15) {
        bestMatchScore = Math.max(bestMatchScore, Number(attBox.score) || 0);
      }
    }

    const drop = cleanScore > 0 ? (((bestMatchScore - cleanScore) / cleanScore) * 100).toFixed(0) : "0";
    const isLost = bestMatchScore < threshold;

    return {
      id: idx + 1,
      label,
      cleanScore,
      attackedScore: bestMatchScore,
      drop,
      isLost,
    };
  });

  return (
    <div className="flex h-full flex-col justify-between rounded-md bg-slate-950 p-2 text-white">
      <div className="space-y-1.5 overflow-hidden">
        {targets.map((t) => (
          <div key={t.id} className="space-y-0.5">
            <div className="flex items-center justify-between text-[9px]">
              <span className="font-semibold truncate max-w-[90px]">{`#${t.id} ${t.label}`}</span>
              <div className="flex items-center gap-1 font-mono text-[8px]">
                <span className="text-blue-400">{(t.cleanScore * 100).toFixed(0)}%</span>
                <span className="text-slate-400">→</span>
                <span className={t.isLost ? "text-rose-400 font-bold" : "text-amber-400"}>
                  {(t.attackedScore * 100).toFixed(0)}%
                </span>
                {t.isLost && (
                  <span className="rounded bg-rose-500/30 px-1 text-[7px] text-rose-300 font-bold">Mất</span>
                )}
              </div>
            </div>

            <div className="relative h-2 w-full rounded-full bg-slate-800 overflow-hidden">
              <div
                className="absolute top-0 bottom-0 left-0 bg-blue-500 rounded-full opacity-80"
                style={{ width: `${Math.min(100, t.cleanScore * 100)}%` }}
              />
              <div
                className={cn(
                  "absolute top-0 bottom-0 left-0 rounded-full transition-all",
                  t.isLost ? "bg-rose-500 opacity-90" : "bg-amber-400 opacity-90",
                )}
                style={{ width: `${Math.min(100, t.attackedScore * 100)}%` }}
              />
              <div
                className="absolute top-0 bottom-0 w-[1px] bg-white/60 z-10"
                style={{ left: `${threshold * 100}%` }}
                title={`Ngưỡng: ${threshold * 100}%`}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-1 flex items-center justify-between border-t border-slate-800/80 pt-1 text-[8px] text-slate-400">
        <span className="flex items-center gap-1">
          <span className="h-1.5 w-1.5 rounded-full bg-blue-500 inline-block" /> Trước
          <span className="h-1.5 w-1.5 rounded-full bg-rose-500 inline-block ml-1" /> Sau
        </span>
        <span className="text-slate-400 font-mono">Ngưỡng: {threshold * 100}%</span>
      </div>
    </div>
  );
}

function DeepDiveGrid({ sample, taskId = "detection2d", zoomRegion, setZoomRegion }) {
  const diff = analysisArtifact(sample, ["diff_url", "difference_url", "perturbation_diff_url"]);
  const perturbation = analysisArtifact(sample, ["perturbation_url", "noise_url"]);
  const segmentationClean = analysisArtifact(sample, ["segmentation_clean_url", "clean_segmentation_url"]);
  const segmentationAttacked = analysisArtifact(sample, ["segmentation_attacked_url", "attacked_segmentation_url"]);
  const zoomClean = analysisArtifact(sample, [`zoom_${zoomRegion}_clean_url`, "zoom_clean_url"]);
  const zoomAttacked = analysisArtifact(sample, [`zoom_${zoomRegion}_attacked_url`, "zoom_attacked_url"]);

  const cleanBoxes = predictionBoxes(sample?.clean_prediction);
  const attackedBoxes = predictionBoxes(sample?.attacked_prediction);
  const cleanBoxes3d = predictionBoxes3d(sample?.clean_prediction);
  const attackedBoxes3d = predictionBoxes3d(sample?.attacked_prediction);
  const cleanImgSrc = sampleArtifact(sample, "clean_input_url");
  const attackedImgSrc = sampleArtifact(sample, "attacked_input_url");
  const dimensions = sampleDimensions(sample) || { width: 1224, height: 370 };

  const selectedBoxIdx = parseInt(zoomRegion, 10) || 0;
  const cropBox = getSelectedCropBox(cleanBoxes, selectedBoxIdx, dimensions.width, dimensions.height);

  const cleanAvgConf = averageConfidence(cleanBoxes);
  const attackedAvgConf = averageConfidence(attackedBoxes);

  const [diffMetrics, setDiffMetrics] = useState({ l2: null, lInf: null });
  const [pertStats, setPertStats] = useState({ min: null, max: null, mean: null });

  const segClasses = [...new Set([...predictionInstances(sample?.clean_prediction), ...predictionInstances(sample?.attacked_prediction)])]
    .map((instance) => instance.label)
    .filter(Boolean)
    .map((name, index) => ({ name, color: ["#22C55E", "#A855F7", "#06B6D4", "#64748B"][index % 4] }));

  return (
    <div data-testid="deep-dive-grid" className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
      {/* 1. Difference Map (|Δ|) */}
      <EvidenceCard
        title="Bản đồ khác biệt (|Δ|)"
        footer={
          <div className="flex justify-between text-[9px] text-slate-500 font-mono">
            <span>L2: {sample ? diffMetrics.l2 : NO_DATA}</span>
            <span>L∞: {sample ? diffMetrics.lInf : NO_DATA}</span>
          </div>
        }
      >
        <div className="h-[132px]">
          {diff ? (
            <AnalysisArtifact src={diff} alt="Bản đồ khác biệt" />
          ) : sample && cleanImgSrc && attackedImgSrc ? (
            <div className="relative flex h-full w-full overflow-hidden rounded-md bg-[#000004]">
              <RealPixelDifferenceCanvas
                cleanUrl={cleanImgSrc}
                attackedUrl={attackedImgSrc}
                onMetricsChange={setDiffMetrics}
              />
              <div className="absolute right-1 top-1 bottom-1 flex flex-col justify-between text-[7px] font-mono text-white/90 py-0.5 pointer-events-none drop-shadow">
                <span>1.0</span>
                <div className="h-14 w-1.5 rounded-full bg-gradient-to-t from-black via-rose-600 to-amber-200 border border-white/20" />
                <span>0.0</span>
              </div>
            </div>
           ) : (
            <AnalysisArtifact src={null} />
          )}
        </div>
      </EvidenceCard>

      {/* 2. Sụt giảm độ tin cậy (Confidence Drop) */}
      <EvidenceCard
        title="Sụt giảm Confidence (Độ tin cậy)"
        footer={
          <div className="flex justify-between text-[9px] text-slate-500 font-mono">
            <span>
              TB:{" "}
              {sample && cleanAvgConf !== null
                ? `${(cleanAvgConf * 100).toFixed(0)}% → ${(attackedAvgConf || 0) * 100}%`
                : NO_DATA}
            </span>
            <span>{sample && cleanAvgConf !== null ? percentageDelta(attackedAvgConf, cleanAvgConf) : ""}</span>
          </div>
        }
      >
        <div className="h-[132px]">
          {sample ? (
            <ConfidenceDropChart cleanBoxes={cleanBoxes} attackedBoxes={attackedBoxes} />
          ) : (
            <AnalysisArtifact src={null} />
          )}
        </div>
      </EvidenceCard>

      <EvidenceCard
        title="Nhiễu đối kháng (Perturbation)"
        footer={
          <div className="flex justify-between text-[9px] text-slate-500 font-mono">
            <span>Min: {sample ? pertStats.min : NO_DATA}</span>
            <span>Max: {sample ? pertStats.max : NO_DATA}</span>
          </div>
        }
      >
        <div className="h-[132px]">
          {perturbation ? (
            <AnalysisArtifact src={perturbation} alt="Nhiễu đối kháng" />
          ) : sample && cleanImgSrc && attackedImgSrc ? (
            <div className="relative flex h-full w-full overflow-hidden rounded-md bg-slate-800">
              <RealPerturbationCanvas
                cleanUrl={cleanImgSrc}
                attackedUrl={attackedImgSrc}
                onStatsChange={setPertStats}
              />
              <div className="absolute right-1 top-1 bottom-1 flex flex-col justify-between text-[7px] font-mono text-white/90 py-0.5 pointer-events-none drop-shadow">
                <span>+ε</span>
                <div className="h-14 w-1.5 rounded-full bg-gradient-to-t from-blue-500 via-slate-600 to-red-500 border border-white/20" />
                <span>-ε</span>
              </div>
            </div>
           ) : false ? (
            <div className="relative flex h-full w-full overflow-hidden rounded-md bg-slate-800">
              <svg viewBox="0 0 200 120" className="h-full w-full">
                <rect width="200" height="120" fill="#1e293b" />
                {Array.from({ length: 65 }).map((_, i) => (
                  <circle
                    key={i}
                    cx={(i * 17 + (i % 3) * 7) % 200}
                    cy={(i * 23 + (i % 5) * 11) % 120}
                    r={(i % 2) + 0.9}
                    fill={i % 2 === 0 ? "#EF4444" : "#3B82F6"}
                    opacity="0.45"
                  />
                ))}
              </svg>
              <div className="absolute right-1 top-1 bottom-1 flex flex-col justify-between text-[7px] font-mono text-white/80 py-0.5">
                <span>+ε</span>
                <div className="h-14 w-1.5 rounded-full bg-gradient-to-t from-blue-500 via-slate-600 to-red-500 border border-white/20" />
                <span>-ε</span>
              </div>
            </div>
          ) : (
            <AnalysisArtifact src={null} />
          )}
        </div>
      </EvidenceCard>

      {taskId === "detection3d" ? (
        <EvidenceCard
          title="BEV 3D: ground truth vÃ  prediction"
          footer={
            <div className="flex justify-between text-[9px] text-slate-500 font-mono">
              <span>Clean: {cleanBoxes3d.length} hộp</span>
              <span>Attacked: {attackedBoxes3d.length} hộp</span>
            </div>
          }
        >
          {cleanBoxes3d.length || attackedBoxes3d.length || sample?.ground_truth?.objects3d?.length ? (
            <div className="grid h-[280px] grid-cols-1 gap-2 sm:grid-cols-2">
              <div className="min-h-0 overflow-hidden rounded-md border border-emerald-200 bg-slate-950">
                <p className="px-2 py-1 text-[10px] font-semibold text-emerald-300">Clean</p>
                <BevViewer groundTruth={sample?.ground_truth} prediction={sample?.clean_prediction} />
              </div>
              <div className="min-h-0 overflow-hidden rounded-md border border-red-200 bg-slate-950">
                <p className="px-2 py-1 text-[10px] font-semibold text-red-300">Attacked</p>
                <BevViewer groundTruth={sample?.ground_truth} prediction={sample?.attacked_prediction} />
              </div>
            </div>
          ) : (
            <AnalysisArtifact src={null} />
          )}
        </EvidenceCard>
      ) : taskId === "segmentation" ? (
        <EvidenceCard
          title="Phân đoạn (Segmentation)"
          footer={
            <div className="flex flex-wrap gap-1.5 text-[8px] font-medium text-slate-600">
              {segClasses.map((cls) => (
                <span key={cls.name} className="flex items-center gap-1">
                  <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: cls.color }} />
                  {cls.name}
                </span>
              ))}
            </div>
          }
        >
          <div className="h-[132px]">
            {segmentationClean && segmentationAttacked ? (
              <div className="grid h-full grid-cols-2 gap-1">
                <AnalysisArtifact src={segmentationClean} alt="Phân đoạn trước tấn công" label="Trước" />
                <AnalysisArtifact src={segmentationAttacked} alt="Phân đoạn sau tấn công" label="Sau" />
              </div>
             ) : false ? (
              <div className="grid h-full grid-cols-2 gap-1">
                <div className="relative overflow-hidden rounded-md bg-slate-900">
                  <svg viewBox="0 0 100 90" className="h-full w-full">
                    <rect x="0" y="0" width="100" height="50" fill="#38BDF8" opacity="0.3" />
                    <rect x="0" y="50" width="100" height="40" fill="#64748B" opacity="0.5" />
                    <path d="M 25 55 L 45 42 L 75 42 L 85 55 Z" fill="#22C55E" opacity="0.85" />
                    <rect x="35" y="25" width="30" height="25" fill="#A855F7" opacity="0.85" />
                  </svg>
                  <span className="absolute inset-x-0 bottom-0 bg-slate-950/75 px-1 py-0.5 text-center text-[8px] font-medium text-white">
                    Trước
                  </span>
                </div>
                <div className="relative overflow-hidden rounded-md bg-slate-900">
                  <svg viewBox="0 0 100 90" className="h-full w-full">
                    <rect x="0" y="0" width="100" height="50" fill="#38BDF8" opacity="0.2" />
                    <rect x="0" y="50" width="100" height="40" fill="#64748B" opacity="0.3" />
                    <path d="M 28 58 L 45 48 L 70 48 L 80 58 Z" fill="#64748B" opacity="0.7" />
                    <rect x="38" y="28" width="25" height="20" fill="#A855F7" opacity="0.5" />
                  </svg>
                  <span className="absolute inset-x-0 bottom-0 bg-slate-950/75 px-1 py-0.5 text-center text-[8px] font-medium text-white">
                    Sau
                  </span>
                </div>
              </div>
            ) : (
              <AnalysisArtifact src={null} />
            )}
          </div>
        </EvidenceCard>
      ) : (
        <EvidenceCard
          title="Mật độ phát hiện (Target Density)"
          footer={
            <div className="flex justify-between text-[9px] text-slate-500 font-mono">
              <span>Phát hiện: {sample ? `${cleanBoxes.length} → ${attackedBoxes.length}` : NO_DATA}</span>
              <span>Bị mất: {sample ? Math.max(0, cleanBoxes.length - attackedBoxes.length) : NO_DATA}</span>
            </div>
          }
        >
          <div className="h-[132px]">
            {sample ? (
              <div className="grid h-full grid-cols-2 gap-1">
                <div className="relative overflow-hidden rounded-md bg-slate-950 p-1">
                  <svg viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} className="h-full w-full">
                    <rect width={dimensions.width} height={dimensions.height} fill="#020617" />
                    {cleanBoxes.map((b, i) => {
                      const [x1, y1, x2, y2] = (b.xyxy || []).map(Number);
                      const cx = (x1 + x2) / 2;
                      const cy = (y1 + y2) / 2;
                      return (
                        <g key={i}>
                          <circle cx={cx} cy={cy} r="18" fill="#10B981" opacity="0.25" />
                          <circle cx={cx} cy={cy} r="6" fill="#10B981" />
                          <rect
                            x={x1}
                            y={y1}
                            width={x2 - x1}
                            height={y2 - y1}
                            fill="none"
                            stroke="#10B981"
                            strokeWidth="2"
                            strokeDasharray="3 3"
                            opacity="0.5"
                          />
                        </g>
                      );
                    })}
                  </svg>
                  <span className="absolute inset-x-0 bottom-0 bg-slate-950/75 px-1 py-0.5 text-center text-[8px] font-medium text-emerald-400">
                    Trước ({cleanBoxes.length} mục tiêu)
                  </span>
                </div>
                <div className="relative overflow-hidden rounded-md bg-slate-950 p-1">
                  <svg viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} className="h-full w-full">
                    <rect width={dimensions.width} height={dimensions.height} fill="#020617" />
                    {attackedBoxes.map((b, i) => {
                      const [x1, y1, x2, y2] = (b.xyxy || []).map(Number);
                      const cx = (x1 + x2) / 2;
                      const cy = (y1 + y2) / 2;
                      return (
                        <g key={i}>
                          <circle cx={cx} cy={cy} r="18" fill="#EF4444" opacity="0.2" />
                          <circle cx={cx} cy={cy} r="6" fill="#EF4444" />
                          <rect
                            x={x1}
                            y={y1}
                            width={x2 - x1}
                            height={y2 - y1}
                            fill="none"
                            stroke="#EF4444"
                            strokeWidth="2"
                            strokeDasharray="3 3"
                            opacity="0.5"
                          />
                        </g>
                      );
                    })}
                  </svg>
                  <span className="absolute inset-x-0 bottom-0 bg-slate-950/75 px-1 py-0.5 text-center text-[8px] font-medium text-rose-400">
                    Sau ({attackedBoxes.length} mục tiêu)
                  </span>
                </div>
              </div>
            ) : (
              <AnalysisArtifact src={null} />
            )}
          </div>
        </EvidenceCard>
      )}

      <EvidenceCard
        title="So sánh vùng phóng to (Zoom)"
        footer={
          <div className="flex justify-between text-[9px] text-slate-500 font-mono">
            <span>PSNR: {sample?.psnr ?? NO_DATA}</span>
            <span>SSIM: {sample?.ssim ?? NO_DATA}</span>
          </div>
        }
      >
        <div className="flex items-center gap-1 text-[9px] mb-1.5">
          <label htmlFor="zoom-region" className="shrink-0 text-slate-500">
            Mục tiêu
          </label>
          <select
            id="zoom-region"
            value={zoomRegion}
            onChange={(event) => setZoomRegion(event.target.value)}
            className="min-w-0 flex-1 rounded border border-slate-200 bg-slate-50 px-1 py-0.5 text-[9px] font-semibold text-slate-700 truncate"
          >
            {cleanBoxes.length > 0 ? (
              cleanBoxes.map((box, i) => (
                <option key={i} value={String(i)}>
                  {`#${i + 1}: ${box.label || "Vật thể"} (${((box.score || 0) * 100).toFixed(0)}%)`}
                </option>
              ))
            ) : (
              <>
                <option value="0">Vùng trọng tâm #1</option>
                <option value="1">Vùng trọng tâm #2</option>
              </>
            )}
          </select>
          <Search className="h-3 w-3 shrink-0 text-slate-400" aria-hidden="true" />
        </div>

        <div className="h-[102px]">
          {zoomClean && zoomAttacked ? (
            <div className="grid h-full grid-cols-3 gap-1">
              <AnalysisArtifact src={zoomClean} alt="Vùng phóng to trước tấn công" label="Trước" />
              <AnalysisArtifact src={zoomAttacked} alt="Vùng phóng to sau tấn công" label="Sau" />
              <AnalysisArtifact src={zoomDiff} alt="Khác biệt vùng phóng to" label="Khác biệt" />
            </div>
          ) : sample && cleanImgSrc ? (
            <div className="grid h-full grid-cols-3 gap-1">
              <div className="relative overflow-hidden rounded-md bg-slate-900 border border-slate-200">
                <svg viewBox={`${cropBox.x} ${cropBox.y} ${cropBox.width} ${cropBox.height}`} className="h-full w-full">
                  <image
                    href={cleanImgSrc}
                    width={dimensions.width}
                    height={dimensions.height}
                    preserveAspectRatio="none"
                  />
                </svg>
                <span className="absolute inset-x-0 bottom-0 bg-slate-950/75 px-1 py-0.5 text-center text-[7px] font-bold text-white">
                  Trước
                </span>
              </div>
              <div className="relative overflow-hidden rounded-md bg-slate-900 border border-slate-200">
                <svg viewBox={`${cropBox.x} ${cropBox.y} ${cropBox.width} ${cropBox.height}`} className="h-full w-full">
                  <image
                    href={attackedImgSrc || cleanImgSrc}
                    width={dimensions.width}
                    height={dimensions.height}
                    preserveAspectRatio="none"
                  />
                </svg>
                <span className="absolute inset-x-0 bottom-0 bg-slate-950/75 px-1 py-0.5 text-center text-[7px] font-bold text-rose-300">
                  Sau
                </span>
              </div>
              <div className="relative overflow-hidden rounded-md bg-slate-950 border border-slate-800">
                {attackedImgSrc ? (
                  <RealCropDifferenceCanvas cleanUrl={cleanImgSrc} attackedUrl={attackedImgSrc} cropBox={cropBox} />
                ) : (
                  <svg
                    viewBox={`${cropBox.x} ${cropBox.y} ${cropBox.width} ${cropBox.height}`}
                    className="h-full w-full"
                  >
                    <image
                      href={cleanImgSrc}
                      width={dimensions.width}
                      height={dimensions.height}
                      preserveAspectRatio="none"
                    />
                    <rect
                      x={cropBox.x}
                      y={cropBox.y}
                      width={cropBox.width}
                      height={cropBox.height}
                      fill="#EF4444"
                      opacity="0.3"
                    />
                  </svg>
                )}
                <span className="absolute inset-x-0 bottom-0 bg-slate-950/75 px-1 py-0.5 text-center text-[7px] font-bold text-indigo-300">
                  Khác biệt
                </span>
              </div>
            </div>
          ) : (
            <AnalysisArtifact src={null} />
          )}
        </div>
      </EvidenceCard>
    </div>
  );
}

function downloadAsset(href, filename) {
  const link = document.createElement("a");
  link.href = href;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function QuickObservation({ expId, sample, report }) {
  const cleanBoxes = predictionBoxes(sample?.clean_prediction);
  const attackedBoxes = predictionBoxes(sample?.attacked_prediction);
  const cleanConfidence = averageConfidence(cleanBoxes);
  const attackedConfidence = averageConfidence(attackedBoxes);
  const cleanAp = numericValue(report?.metrics?.clean?.ap50 ?? report?.ap_clean);
  const cell =
    report?.cells?.find((item) => item.attack === sample?.attack && item.severity === sample?.severity) ??
    report?.cells?.at(-1);
  const attackedAp = numericValue(cell?.metrics?.ap50 ?? cell?.ap);
  const observations =
    sample?.clean_prediction || sample?.attacked_prediction
      ? [
          `Số lượng bbox từ model: ${cleanBoxes.length} → ${attackedBoxes.length}.`,
          cleanConfidence !== null && attackedConfidence !== null
            ? `Confidence trung bình từ prediction: ${cleanConfidence.toFixed(2)} → ${attackedConfidence.toFixed(2)} (${percentageDelta(attackedConfidence, cleanConfidence) || NO_DATA}).`
            : null,
          cleanAp !== null && attackedAp !== null
            ? `mAP@0.5 từ report: ${cleanAp.toFixed(2)} → ${attackedAp.toFixed(2)} (${percentageDelta(attackedAp, cleanAp) || NO_DATA}).`
            : null,
        ].filter(Boolean)
      : ["Chưa có quan sát từ prediction"];
  const attackedArtifact = sampleArtifact(sample, "attacked_input_url");
  return (
    <div className="flex flex-col gap-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm lg:flex-row lg:items-center lg:justify-between">
      <div className="min-w-0 space-y-2">
        <h2 className="text-xs font-bold text-slate-900">Quan sát nhanh</h2>
        <div className="space-y-1.5 text-[11px] text-slate-700">
          {observations.map((observation, index) => (
            <p key={observation} className="flex items-start gap-2">
              <span
                className={cn(
                  "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] font-bold",
                  index === 0
                    ? "bg-emerald-100 text-emerald-700"
                    : index === 1
                      ? "bg-amber-100 text-amber-700"
                      : "bg-blue-100 text-blue-700",
                )}
              >
                {index === 0 ? "✓" : index === 1 ? "!" : "i"}
              </span>
              <span>{observation}</span>
            </p>
          ))}
        </div>
      </div>
      <div className="flex shrink-0 flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={!attackedArtifact}
          onClick={() => attackedArtifact && downloadAsset(attackedArtifact, "attacked_result.png")}
          className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-xs font-semibold text-slate-700 shadow-sm transition-colors enabled:hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Download className="h-4 w-4 text-slate-500" aria-hidden="true" />
          <span>
            <span className="block font-bold leading-none">{attackedArtifact ? "Lưu ảnh" : "Chưa có artifact"}</span>
            <span className="mt-0.5 block text-[9px] font-normal text-slate-400">Lưu ảnh từ run thực tế</span>
          </span>
        </button>
        <Link
          href="/benchmark"
          className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-xs font-semibold text-slate-700 shadow-sm transition-colors hover:bg-slate-50"
        >
          <SlidersHorizontal className="h-4 w-4 text-slate-500" aria-hidden="true" />
          <span>
            <span className="block font-bold leading-none">So sánh chi tiết</span>
            <span className="mt-0.5 block text-[9px] font-normal text-slate-400">Xem bảng so sánh đầy đủ</span>
          </span>
        </Link>
        <button
          type="button"
          disabled={!attackedArtifact}
          onClick={() => attackedArtifact && downloadAsset(attackedArtifact, `adversai_report_${expId}.png`)}
          className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-left text-xs font-bold text-white shadow-sm transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Download className="h-4 w-4" aria-hidden="true" />
          <span>
            <span className="block leading-none">{attackedArtifact ? "Xuất kết quả" : "Chưa có artifact"}</span>
            <span className="mt-0.5 block text-[9px] font-normal text-blue-200">Xuất artifact từ run thực tế</span>
          </span>
          <ChevronDown className="ml-1 h-3.5 w-3.5 text-blue-200" aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}

export default function VisualResultsPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const { activeProjectId } = useProject();
  const [zoomRegion, setZoomRegion] = useState("vehicle_rear");
  const [zoom, setZoom] = useState(1);
  const [metadata, setMetadata] = useState(null);
  const [report, setReport] = useState(null);
  const [samples, setSamples] = useState([]);
  const [activeSampleIndex, setActiveSampleIndex] = useState(0);
  const expId = params?.id || "";
  const deepLinkRunId = searchParams?.get("run_id") || searchParams?.get("runId") || "";
  const deepLinkProjectId = searchParams?.get("project_id") || "";
  const projectId = deepLinkProjectId || activeProjectId;

  const [activeRunId, setActiveRunId] = useState(null);
  const [sessionData, setSessionData] = useState(null);
  const [autoFlagged, setAutoFlagged] = useState(false);
  const [flagging, setFlagging] = useState(false);

  useEffect(() => {
    if (!expId) return;
    getSession(expId, projectId)
      .then(setSessionData)
      .catch(() => {});
  }, [expId, projectId]);

  const handleSelectRun = useCallback(
    async (runId) => {
      setActiveRunId(runId);
      setActiveSampleIndex(0);
      try {
        const [fetchedReport, fetchedSamples] = await Promise.all([
          getRunReport(runId, projectId),
          getRunSamples(runId, {}, projectId),
        ]);
        setReport(fetchedReport);
        setSamples(fetchedSamples?.length ? fetchedSamples : fetchedReport?.sample_results || []);
      } catch (err) {
        console.warn("Could not load run report:", err);
      }
    },
    [projectId],
  );

  useEffect(() => {
    if (deepLinkRunId) {
      if (deepLinkRunId !== activeRunId) handleSelectRun(deepLinkRunId);
      return;
    }
    const initialRun = sessionData?.runs?.at(-1)?.id || sessionData?.run_id || sessionData?.runId;
    if (initialRun && !activeRunId) handleSelectRun(initialRun);
  }, [deepLinkRunId, sessionData, activeRunId, handleSelectRun]);

  const handleAutoFlag = useCallback(async () => {
    const runId = activeRunId;
    if (!runId) return;
    setFlagging(true);
    try {
      await triggerAutoFlag(runId, 30, projectId);
      setAutoFlagged(true);
    } catch (err) {
      console.warn("Auto-flag failed:", err);
    } finally {
      setFlagging(false);
    }
  }, [activeRunId, projectId]);

  useEffect(() => {
    const syncStoredMetadata = () => {
      let activeExperiment = {};
      let lastSession = {};
      try {
        const storedActive = JSON.parse(localStorage.getItem("adversai_active_experiment") || "{}");
        const storedLast = JSON.parse(localStorage.getItem("advertest_last_session") || "{}");
        if (!storedActive.expName || storedActive.expName === expId) activeExperiment = storedActive;
        lastSession = storedLast;
      } catch {
        // Server/session evidence below remains authoritative when browser storage is unavailable.
      }
      const sessionReport = report || lastSession.report || sessionData?.report || {};
      if (!activeRunId && sessionReport.run_id) {
        setReport(sessionReport);
        setSamples(sessionReport.sample_results || []);
      }
      setMetadata(
        buildExperimentMetadata(
          expId,
          { ...sessionData, ...activeExperiment },
          {
            ...sessionData,
            ...lastSession,
            report: sessionReport,
            runId: activeRunId || lastSession.runId || sessionReport.run_id,
            runStatus: sessionReport.run_id ? "COMPLETED" : lastSession.runStatus,
          },
        ),
      );
    };
    const syncId = window.setTimeout(syncStoredMetadata, 0);
    window.addEventListener("storage", syncStoredMetadata);
    return () => {
      window.clearTimeout(syncId);
      window.removeEventListener("storage", syncStoredMetadata);
    };
  }, [expId, activeRunId, report, sessionData]);

  const currentSample = samples[activeSampleIndex] || samples[0] || null;
  const taskId =
    metadata?.selectedTask || report?.task_id || report?.provenance?.run_config?.task_id || report?.task || "detection2d";

  const currentRunId = activeRunId || deepLinkRunId || sessionData?.runId || sessionData?.run_id || "";
  const currentRunNote = sessionData?.runs?.find((r) => r.id === currentRunId)?.note || "";
  const isDemoFixture = report?.provenance?.demo_fixture === true;

  return (
    <div className="animate-fade-in space-y-4 pb-8 text-slate-800">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-bold tracking-tight text-slate-900">Kết quả trực quan</h1>
            <DemoFixtureBadge visible={isDemoFixture} />
          </div>
          <p className="mt-0.5 text-xs text-slate-500">Trực quan hóa kết quả tấn công đối kháng và so sánh trước/sau</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleAutoFlag}
            disabled={autoFlagged || flagging || !currentRunId}
            className={cn(
              "flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-semibold shadow-sm transition-colors",
              autoFlagged
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : "border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100",
            )}
          >
            {autoFlagged ? (
              <>
                <CheckCircle2 className="h-3.5 w-3.5" /> Đã gửi Review ✓
              </>
            ) : (
              <>
                <ShieldAlert className="h-3.5 w-3.5" /> {flagging ? "Đang gửi..." : "Flag → Review"}
              </>
            )}
          </button>
          <Link
            href={`/experiments/${expId}/attack?mode=extend`}
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm transition-colors hover:bg-slate-50"
          >
            <RotateCcw className="h-3.5 w-3.5 text-violet-600" aria-hidden="true" /> Quay lại cấu hình
          </Link>
          <Link
            href="/benchmark"
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm transition-colors hover:bg-slate-50"
          >
            <History className="h-3.5 w-3.5 text-blue-600" aria-hidden="true" /> Lịch sử kết quả
          </Link>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm transition-colors hover:bg-slate-50"
          >
            <RefreshCw className="h-3.5 w-3.5 text-slate-600" aria-hidden="true" /> Tải lại
          </button>
        </div>
      </header>

      <RunHistoryBar
        sessionId={expId}
        activeRunId={currentRunId}
        onSelectRun={handleSelectRun}
        projectId={projectId}
      />

      <SampleNavigator
        samples={samples}
        activeIndex={activeSampleIndex}
        onSelectSample={setActiveSampleIndex}
        zoom={zoom}
        onZoomChange={setZoom}
      />

      <div
        data-testid="visual-results-comparison"
        className="grid min-w-0 grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5 items-stretch"
      >
        <ResultPanel
          sample={currentSample}
          report={report}
          taskId={taskId}
          zoom={zoom}
          className="md:col-span-1 xl:col-span-2"
        />
        <div className="relative min-w-0 md:col-span-1 xl:col-span-2">
          <ResultPanel sample={currentSample} report={report} taskId={taskId} attacked zoom={zoom} />
          <div className="pointer-events-none absolute -left-5 top-[37%] z-10 hidden -translate-y-1/2 flex-col items-center xl:flex">
            <span className="flex h-10 w-10 items-center justify-center rounded-full border-2 border-slate-200 bg-white text-xs font-black text-slate-800 shadow-md">
              VS
            </span>
            <span className="mt-1 rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[9px] font-bold shadow-sm">
              <span className="text-blue-600">Trước</span>
              <span className="px-1 text-slate-300">•</span>
              <span className="text-red-600">Sau</span>
            </span>
          </div>
        </div>
        <ExperimentRail metadata={metadata} className="md:col-span-2 xl:col-span-1" />
      </div>

      <DeepDiveGrid sample={currentSample} taskId={taskId} zoomRegion={zoomRegion} setZoomRegion={setZoomRegion} />
      <QuickObservation expId={expId} sample={currentSample} report={report} />

      <SampleReviewPanel runId={currentRunId} samples={samples} projectId={projectId} />

      <ResearcherNotePanel
        sessionId={expId}
        runId={currentRunId}
        initialNote={currentRunNote}
        projectId={projectId}
      />
    </div>
  );
}
