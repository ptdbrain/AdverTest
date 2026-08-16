export default function SamWaitingState() {
  const missingArtifacts = [
    "SAM2 Checkpoint (checkpoints/sam2/sam2_hiera_large.pt)",
    "SAM2 Model Config (configs/sam2/sam2.1_hiera_l.yaml)",
    "Reviewed Mask Ground Truth Annotations (data/segmentation/sa1b_val.json)",
    "GT Box Prompting Protocol Specification (eval/sam_prompt_protocol.json)",
  ];

  return (
    <section className="sam-waiting" aria-label="SAM2 handoff status" style={{ padding: "var(--space-xl)", borderRadius: "var(--radius-lg)", border: "1px solid var(--warning)", background: "rgba(234, 179, 8, 0.05)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
        <span className="status-badge status-badge--queued" style={{ background: "var(--warning)", color: "#000", fontWeight: "bold" }}>
          WAITING_FOR_ARTIFACTS
        </span>
        <p className="evidence-stage__eyebrow" style={{ margin: 0, color: "var(--warning)" }}>Segmentation Mode Handoff Gate</p>
      </div>
      <h2 style={{ fontSize: "1.4rem", margin: "8px 0", color: "var(--text-primary)" }}>SAM2 Handoff Gated</h2>
      <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginBottom: "16px" }}>
        SAM2 evaluation remains locked until the following real model artifacts are delivered and validated by Person C.
        Simulated masks are strictly forbidden under the AdverTest scientific protocol.
      </p>
      <div className="missing-artifacts-list" style={{ background: "var(--bg-primary)", padding: "16px", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
        <div style={{ fontSize: "0.8rem", fontWeight: "bold", textTransform: "uppercase", color: "var(--text-tertiary)", marginBottom: "8px" }}>
          Required Artifact Manifest
        </div>
        <ul style={{ margin: 0, paddingLeft: "20px", fontSize: "0.85rem", color: "var(--text-primary)" }}>
          {missingArtifacts.map((item, idx) => (
            <li key={idx} style={{ marginBottom: "4px", fontFamily: "var(--font-mono)" }}>
              {item}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

