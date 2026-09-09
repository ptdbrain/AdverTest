"use client";

import React, { useMemo, useRef, useState } from "react";
import { finalizeUploadBatch, saveBatchAnnotation } from "@/lib/api";

function toPoint(event, svg, width, height) {
  const bounds = svg.getBoundingClientRect();
  return [
    Math.max(0, Math.min(width, ((event.clientX - bounds.left) / bounds.width) * width)),
    Math.max(0, Math.min(height, ((event.clientY - bounds.top) / bounds.height) * height)),
  ];
}

function normalizedBox([x1, y1], [x2, y2]) {
  return [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)];
}

/** Browser-native SVG editor; canonical values stay in original image pixels. */
export default function ManualAnnotationPanel({ batch, previews, onFinalized }) {
  const sampleIds = useMemo(() => Object.keys(batch?.samples || {}), [batch]);
  const [index, setIndex] = useState(0);
  const [classId, setClassId] = useState(Object.keys(batch?.class_map || {})[0] || "");
  const [items, setItems] = useState({});
  const [polygon, setPolygon] = useState([]);
  const [drag, setDrag] = useState(null);
  const [history, setHistory] = useState([]);
  const [future, setFuture] = useState([]);
  const [message, setMessage] = useState("");
  const svgRef = useRef(null);
  const sampleId = sampleIds[index];
  const sample = batch?.samples?.[sampleId];
  const annotations = items[sampleId] || [];

  if (!sample) return null;
  const mutate = (next) => {
    setHistory((value) => [...value, items]);
    setFuture([]);
    setItems(next);
  };
  const point = (event) => toPoint(event, svgRef.current, sample.width, sample.height);
  const isDetection = batch.task_id === "detection2d";

  const save = async () => {
    try {
      const document = {
        task_id: batch.task_id,
        annotations: annotations.map((annotation) =>
          isDetection
            ? { class_id: annotation.class_id, bbox_xyxy: annotation.bbox_xyxy }
            : { class_id: annotation.class_id, polygon: annotation.polygon },
        ),
      };
      await saveBatchAnnotation(batch.batch_id, sampleId, document);
      setMessage(`${sampleId} saved.`);
    } catch (error) {
      setMessage(error.message || "Annotation is invalid.");
    }
  };
  const finalize = async () => {
    try {
      onFinalized(await finalizeUploadBatch(batch.batch_id));
    } catch (error) {
      setMessage(error.message || "Every benchmark sample needs a valid annotation.");
    }
  };
  const preview = previews?.[sampleId]?.url;

  return (
    <section
      aria-label="Manual annotation"
      style={{ display: "grid", gap: 10, borderTop: "1px solid var(--border-subtle)", paddingTop: 12 }}
    >
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <strong>Label manually</strong>
        <span className="text-xs text-secondary">
          {batch.task_id === "segmentation"
            ? "Click polygon points, then close polygon."
            : "Drag to create a box; drag an existing box to move it."}
        </span>
        <select
          aria-label="Annotation class"
          className="select-field"
          style={{ width: "auto" }}
          value={classId}
          onChange={(event) => setClassId(event.target.value)}
        >
          {Object.entries(batch.class_map || {}).map(([id, name]) => (
            <option key={id} value={id}>
              {name}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() =>
            history.length &&
            (setFuture((value) => [items, ...value]),
            setItems(history.at(-1)),
            setHistory((value) => value.slice(0, -1)))
          }
        >
          Undo
        </button>
        <button
          type="button"
          onClick={() =>
            future.length &&
            (setHistory((value) => [...value, items]), setItems(future[0]), setFuture((value) => value.slice(1)))
          }
        >
          Redo
        </button>
      </div>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button type="button" disabled={index === 0} onClick={() => setIndex(index - 1)}>
          Previous
        </button>
        <span className="text-xs">
          {index + 1}/{sampleIds.length}: {sample.filename}
        </span>
        <button type="button" disabled={index >= sampleIds.length - 1} onClick={() => setIndex(index + 1)}>
          Next
        </button>
        {!isDetection && (
          <button
            type="button"
            disabled={polygon.length < 3}
            onClick={() => {
              mutate({ ...items, [sampleId]: [...annotations, { class_id: classId, polygon }] });
              setPolygon([]);
            }}
          >
            Close polygon
          </button>
        )}
      </div>
      <div style={{ position: "relative", width: "100%", maxHeight: 360, overflow: "auto", background: "#06070a" }}>
        {preview ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element -- this is a temporary local Blob preview, never a product image URL. */}
            <img
              src={preview}
              alt={`Upload ${sampleId}`}
              style={{ display: "block", maxWidth: "100%", maxHeight: 350, margin: "auto" }}
            />
          </>
        ) : (
          <div className="empty-state">
            Image preview unavailable; labels are still stored in original pixel coordinates.
          </div>
        )}
        <svg
          ref={svgRef}
          viewBox={`0 0 ${sample.width} ${sample.height}`}
          preserveAspectRatio="xMidYMid meet"
          style={{ position: "absolute", inset: 0, width: "100%", height: preview ? "100%" : 350, cursor: "crosshair" }}
          onPointerDown={(event) => {
            const p = point(event);
            if (isDetection) setDrag({ mode: "create", start: p });
            else setPolygon([...polygon, p]);
          }}
          onPointerUp={(event) => {
            if (!drag?.start) return;
            const end = point(event);
            if (!isDetection) {
              if (drag.mode === "polygon-point")
                mutate({
                  ...items,
                  [sampleId]: annotations.map((item, current) =>
                    current === drag.index
                      ? {
                          ...item,
                          polygon: item.polygon.map((value, pointIndex) =>
                            pointIndex === drag.pointIndex ? end : value,
                          ),
                        }
                      : item,
                  ),
                });
              setDrag(null);
              return;
            }
            if (drag.mode === "create") {
              const bbox_xyxy = normalizedBox(drag.start, end);
              if (bbox_xyxy[2] - bbox_xyxy[0] >= 2 && bbox_xyxy[3] - bbox_xyxy[1] >= 2)
                mutate({ ...items, [sampleId]: [...annotations, { class_id: classId, bbox_xyxy }] });
            } else if (drag.mode === "move") {
              const dx = end[0] - drag.start[0];
              const dy = end[1] - drag.start[1];
              const [x1, y1, x2, y2] = drag.original;
              const width = x2 - x1;
              const height = y2 - y1;
              const nx1 = Math.max(0, Math.min(sample.width - width, x1 + dx));
              const ny1 = Math.max(0, Math.min(sample.height - height, y1 + dy));
              mutate({
                ...items,
                [sampleId]: annotations.map((item, current) =>
                  current === drag.index ? { ...item, bbox_xyxy: [nx1, ny1, nx1 + width, ny1 + height] } : item,
                ),
              });
            } else if (drag.mode === "resize") {
              const [x1, y1] = drag.original;
              const bbox_xyxy = normalizedBox([x1, y1], end);
              if (bbox_xyxy[2] - bbox_xyxy[0] >= 2 && bbox_xyxy[3] - bbox_xyxy[1] >= 2)
                mutate({
                  ...items,
                  [sampleId]: annotations.map((item, current) =>
                    current === drag.index ? { ...item, bbox_xyxy } : item,
                  ),
                });
            }
            setDrag(null);
          }}
        >
          {annotations.map((annotation, itemIndex) =>
            isDetection ? (
              <g key={itemIndex}>
                <rect
                  x={annotation.bbox_xyxy[0]}
                  y={annotation.bbox_xyxy[1]}
                  width={annotation.bbox_xyxy[2] - annotation.bbox_xyxy[0]}
                  height={annotation.bbox_xyxy[3] - annotation.bbox_xyxy[1]}
                  fill="rgba(56,189,248,.18)"
                  stroke="#38bdf8"
                  strokeWidth="2"
                  onPointerDown={(event) => {
                    event.stopPropagation();
                    setDrag({ mode: "move", index: itemIndex, start: point(event), original: annotation.bbox_xyxy });
                  }}
                  onDoubleClick={(event) => {
                    event.stopPropagation();
                    mutate({ ...items, [sampleId]: annotations.filter((_, current) => current !== itemIndex) });
                  }}
                />
                <circle
                  cx={annotation.bbox_xyxy[2]}
                  cy={annotation.bbox_xyxy[3]}
                  r="5"
                  fill="#38bdf8"
                  onPointerDown={(event) => {
                    event.stopPropagation();
                    setDrag({ mode: "resize", index: itemIndex, start: point(event), original: annotation.bbox_xyxy });
                  }}
                />
              </g>
            ) : (
              <g key={itemIndex}>
                <polygon
                  points={annotation.polygon.map((p) => p.join(",")).join(" ")}
                  fill="rgba(56,189,248,.18)"
                  stroke="#38bdf8"
                  strokeWidth="2"
                />
                {annotation.polygon.map((vertex, pointIndex) => (
                  <circle
                    key={pointIndex}
                    cx={vertex[0]}
                    cy={vertex[1]}
                    r="5"
                    fill="#38bdf8"
                    onPointerDown={(event) => {
                      event.stopPropagation();
                      setDrag({ mode: "polygon-point", index: itemIndex, pointIndex, start: point(event) });
                    }}
                  />
                ))}
              </g>
            ),
          )}
          {polygon.length > 0 && (
            <polyline points={polygon.map((p) => p.join(",")).join(" ")} fill="none" stroke="#fbbf24" strokeWidth="2" />
          )}
        </svg>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button type="button" className="run-button" onClick={save}>
          Save this annotation
        </button>
        <button type="button" className="run-button" onClick={finalize}>
          Finalize benchmark dataset
        </button>
      </div>
      {message && (
        <p className="text-xs text-secondary" role="status">
          {message}
        </p>
      )}
    </section>
  );
}
