"use client";

import { useState } from "react";
import Image from "next/image";

import EvidenceStage from "@/components/EvidenceStage";
import BaseModelMetrics from "@/components/FiveMetrics";

function GroundTruthOverlay({ groundTruth }) {
  if (groundTruth?.type !== "boxes" || !groundTruth.objects?.length) return null;
  const width = Number(groundTruth.image_width) || 1;
  const height = Number(groundTruth.image_height) || 1;
  return (
    <svg className="evidence-media__ground-truth" aria-label="Ground truth overlay" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      {groundTruth.objects.map((object, index) => {
        const [x1, y1, x2, y2] = object.xyxy ?? [];
        if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
        return <g key={object.object_id ?? index}><rect x={x1} y={y1} width={Math.max(0, x2 - x1)} height={Math.max(0, y2 - y1)} /><text x={x1} y={Math.max(14, y1 - 4)}>{object.label ?? "object"}</text></g>;
      })}
    </svg>
  );
}

function EvidenceImage({ src, alt, unavailableTitle, unavailableDetail, groundTruth, aspectRatio }) {
  if (!src) {
    return (
      <div className="evidence-media__empty">
        <strong>{unavailableTitle}</strong>
        <span>{unavailableDetail}</span>
      </div>
    );
  }

  return (
    <div className="evidence-media" style={{ "--evidence-aspect": aspectRatio ?? "16 / 9" }}>
      <Image
        src={src}
        alt={alt}
        fill
        sizes="(max-width: 1100px) 100vw, 50vw"
        unoptimized
        className="evidence-media__image"
      />
      <GroundTruthOverlay groundTruth={groundTruth} />
    </div>
  );
}

function EvidenceUnavailable({ title, detail }) {
  return <div className="evidence-media__empty"><strong>{title}</strong><span>{detail}</span></div>;
}

function imageAspectRatio(sample) {
  const width = Number(sample?.ground_truth?.image_width ?? sample?.image_width);
  const height = Number(sample?.ground_truth?.image_height ?? sample?.image_height);
  return width > 0 && height > 0 ? `${width} / ${height}` : null;
}

export default function Workspace({ report, samples, mode }) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const normalizedSamples = samples ?? [];
  const activeIndex = Math.min(selectedIndex, Math.max(0, normalizedSamples.length - 1));
  const sample = normalizedSamples[activeIndex];
  const attack = report?.cells?.find((cell) => cell.attack === sample?.attack && cell.severity === sample?.severity)
    ?? report?.cells?.at(-1)
    ?? null;
  const artifacts = sample?.artifacts ?? {};
  const perObjectFailures = sample?.per_object_failures ?? sample?.object_evidence ?? [];
  const config = report?.provenance?.run_config ?? {};
  const benchmarkAvailable = report?.benchmark_metrics_available !== false;
  const aspectRatio = imageAspectRatio(sample);

  return (
    <div className="workspace">
      {normalizedSamples.length > 1 && (
        <div className="workspace__evidence_selector">
          <label htmlFor="evidence-sample">Evidence sample</label>
          <select id="evidence-sample" value={String(activeIndex)} onChange={(event) => setSelectedIndex(Number(event.target.value))}>
            {normalizedSamples.map((item, index) => (
              <option key={`${item.sample_id}-${item.attack}-${item.severity}`} value={index}>
                {item.sample_id} · {item.attack} · severity {item.severity}
              </option>
            ))}
          </select>
        </div>
      )}
      <EvidenceStage className="evidence-stage--source" title="Input and ground truth" description="Source image and reviewed bounding-box labels.">
        <EvidenceImage src={artifacts.clean_input_url} alt="Clean input" unavailableTitle="Source input unavailable" unavailableDetail="The backend did not return a source artifact for this sample." groundTruth={sample?.ground_truth} aspectRatio={aspectRatio} />
      </EvidenceStage>
      <EvidenceStage className="evidence-stage--attacked" title="Attacked input" description={attack ? `${attack.attack}, severity ${attack.severity}, paired with the same source sample.` : "Recipe evidence will appear after a completed run."}>
        <EvidenceImage src={artifacts.attacked_input_url} alt="Attacked input" unavailableTitle="Attacked input unavailable" unavailableDetail="The backend did not return an attacked input artifact." aspectRatio={aspectRatio} />
      </EvidenceStage>
      <EvidenceStage className="evidence-stage--clean-pred" title="Clean model prediction" description="Prediction before perturbation, evaluated against the locked ground truth.">
        <div className="prediction-evidence">
          <EvidenceImage src={artifacts.clean_prediction_url} alt="Clean prediction" unavailableTitle="Clean prediction unavailable" unavailableDetail="The backend did not return a rendered clean prediction artifact." aspectRatio={aspectRatio} />
          <BaseModelMetrics ariaLabel="Base model metrics on clean input" title="Base model on clean input" metrics={report?.metrics?.clean} fallbackAp={report?.ap_clean} benchmarkAvailable={benchmarkAvailable} />
        </div>
      </EvidenceStage>
      <EvidenceStage className="evidence-stage--attacked-pred" title="Attacked model prediction" description="Paired prediction after the final ordered recipe step.">
        <div className="prediction-evidence">
          <EvidenceImage src={artifacts.attacked_prediction_url} alt="Attacked prediction" unavailableTitle="Attacked prediction unavailable" unavailableDetail="The backend did not return a rendered attacked prediction artifact." aspectRatio={aspectRatio} />
          <BaseModelMetrics ariaLabel="Base model metrics on attacked input" title="Base model on attacked input" metrics={attack?.metrics} fallbackAp={attack?.ap} benchmarkAvailable={benchmarkAvailable} />
        </div>
      </EvidenceStage>

      <section className="workspace__per_object_failures" aria-label="Per-object failure evidence">
        <header><p className="evidence-stage__eyebrow">Granular Evidence</p><h2>Per-Object Failure Breakdown</h2></header>
        {perObjectFailures.length === 0 ? <EvidenceUnavailable title="No per-object evidence was produced for this sample." detail="Verify that model inference and object matching completed successfully." /> : (
          <div className="per-object-table-wrapper"><table className="data-table"><thead><tr><th>Object ID</th><th>Label</th><th>Clean Conf.</th><th>Attacked Conf.</th><th>Degradation</th><th>Status</th></tr></thead><tbody>
            {perObjectFailures.map((obj) => {
              const cleanConfidence = obj.clean_conf ?? obj.clean_confidence ?? 0;
              const attackedConfidence = obj.attacked_conf ?? obj.attacked_confidence ?? 0;
              const degradation = obj.degradation ?? Math.max(0, (cleanConfidence - attackedConfidence) * 100);
              const status = obj.status ?? (obj.failure_reason ? "FAILED" : "PASSED");
              return <tr key={obj.id ?? obj.object_id}><td>{obj.id ?? obj.object_id}</td><td>{obj.label ?? obj.gt_box?.label ?? "unavailable"}</td><td>{(cleanConfidence * 100).toFixed(1)}%</td><td>{(attackedConfidence * 100).toFixed(1)}%</td><td>↓ {degradation.toFixed(1)}%</td><td><span className={`status-badge ${status === "FAILED" ? "status-badge--failed" : "status-badge--queued"}`}>{status}</span></td></tr>;
            })}
          </tbody></table></div>
        )}
      </section>

      <section className="workspace__provenance" aria-label="Evidence provenance">
        <strong>Evidence provenance</strong>
        <span>Model: {report?.model_version || report?.model || "unavailable"}</span>
        <span>Checkpoint hash: {sample?.model_checkpoint_hash || report?.provenance?.model?.checkpoint_hash || "unavailable"}</span>
        <span>Dataset: {report?.dataset || "unavailable"}</span>
        <span>Sample: {sample?.sample_id || "unavailable"}</span>
        <span>Attack: {sample?.attack || "unavailable"}</span>
        <span>Severity: {sample?.severity ?? "unavailable"}</span>
        <span>Seed: {config.seed ?? "unavailable"}</span>
        <span>Run ID: {report?.run_id || "unavailable"}</span>
      </section>
    </div>
  );
}
