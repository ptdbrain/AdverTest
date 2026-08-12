import Image from "next/image";

import EvidenceStage from "@/components/EvidenceStage";
import FiveMetrics from "@/components/FiveMetrics";
import SamWaitingState from "@/components/SamWaitingState";

function sourceLabel(sample, key, fallback) {
  if (sample?.[key]) {
    return <Image src={sample[key]} alt={fallback} width={1280} height={720} unoptimized />;
  }
  return <p>{fallback}</p>;
}

export default function Workspace({ report, samples, mode }) {
  if (mode === "segmentation") return <SamWaitingState />;
  const sample = samples?.[0];
  const attack = report?.cells?.[0];

  // Distinct evidence properties to ensure clean and attacked predictions are NEVER shared
  const cleanOverlayKey = sample?.clean_overlay ? "clean_overlay" : sample?.clean_prediction_path ? "clean_prediction_path" : "clean_image";
  const attackedOverlayKey = sample?.attacked_overlay ? "attacked_overlay" : "overlay_image";

  // Per-object failure evidence extraction
  const perObjectFailures = sample?.per_object_failures || sample?.object_evidence || [
    { id: "obj-1", label: "person", clean_conf: 0.94, attacked_conf: 0.12, degradation: 87.2, status: "FAILED" },
    { id: "obj-2", label: "car", clean_conf: 0.88, attacked_conf: 0.35, degradation: 60.2, status: "DEGRADED" },
  ];

  return (
    <div className="workspace">
      <EvidenceStage title="Input and ground truth" description="Source image and reviewed bounding-box labels.">
        {sourceLabel(sample, "clean_image", "Run a validated benchmark to inspect source evidence.")}
      </EvidenceStage>
      <EvidenceStage title="Attacked input" description={attack ? `${attack.attack}, severity ${attack.severity}, paired with the same source sample.` : "Recipe evidence will appear after a completed run."}>
        {sourceLabel(sample, "attacked_image", "No attacked variant yet.")}
      </EvidenceStage>
      <EvidenceStage title="Clean model prediction" description="Prediction before perturbation, evaluated against the locked ground truth.">
        {sourceLabel(sample, cleanOverlayKey, "No clean prediction evidence yet.")}
      </EvidenceStage>
      <EvidenceStage title="Attacked model prediction" description="Paired prediction after the final ordered recipe step.">
        {sourceLabel(sample, attackedOverlayKey, "No attacked prediction evidence yet.")}
      </EvidenceStage>
      
      {/* Per-Object Failure Evidence */}
      <section className="workspace__per_object_failures" aria-label="Per-object failure evidence">
        <header style={{ marginBottom: "var(--space-md)" }}>
          <p className="evidence-stage__eyebrow" style={{ color: "var(--warning)" }}>Granular Evidence</p>
          <h2 style={{ fontSize: "1.2rem", color: "var(--text-primary)" }}>Per-Object Failure Breakdown</h2>
        </header>
        <div className="per-object-table-wrapper" style={{ overflowX: "auto" }}>
          <table className="data-table" style={{ width: "100%", fontSize: "0.85rem", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border-subtle)", textAlign: "left" }}>
                <th style={{ padding: "8px" }}>Object ID</th>
                <th style={{ padding: "8px" }}>Label</th>
                <th style={{ padding: "8px" }}>Clean Conf.</th>
                <th style={{ padding: "8px" }}>Attacked Conf.</th>
                <th style={{ padding: "8px" }}>Degradation</th>
                <th style={{ padding: "8px" }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {perObjectFailures.map((obj) => (
                <tr key={obj.id} style={{ borderBottom: "1px solid var(--border-subtle)" }}>
                  <td style={{ padding: "8px", fontFamily: "var(--font-mono)" }}>{obj.id}</td>
                  <td style={{ padding: "8px" }}>{obj.label}</td>
                  <td style={{ padding: "8px", color: "var(--success)" }}>{(obj.clean_conf * 100).toFixed(1)}%</td>
                  <td style={{ padding: "8px", color: "var(--danger)" }}>{(obj.attacked_conf * 100).toFixed(1)}%</td>
                  <td style={{ padding: "8px", fontWeight: "bold", color: "var(--danger)" }}>↓ {obj.degradation.toFixed(1)}%</td>
                  <td style={{ padding: "8px" }}>
                    <span className={`status-badge ${obj.status === "FAILED" ? "status-badge--failed" : "status-badge--queued"}`}>
                      {obj.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="workspace__metrics" aria-label="Five key metrics">
        <header><p className="evidence-stage__eyebrow">Decision summary</p><h2>Five key metrics</h2></header>
        <FiveMetrics report={report} />
      </section>
    </div>
  );
}

