import EvidenceStage from "@/components/EvidenceStage";
import FiveMetrics from "@/components/FiveMetrics";
import SamWaitingState from "@/components/SamWaitingState";

function sourceLabel(sample, key, fallback) {
  return sample?.[key] ? <img src={sample[key]} alt={fallback} /> : <p>{fallback}</p>;
}

export default function Workspace({ report, samples, mode }) {
  if (mode === "segmentation") return <SamWaitingState />;
  const sample = samples?.[0];
  const attack = report?.cells?.[0];
  return (
    <div className="workspace">
      <EvidenceStage title="Input and ground truth" description="Source image and reviewed bounding-box labels.">
        {sourceLabel(sample, "clean_image", "Run a validated benchmark to inspect source evidence.")}
      </EvidenceStage>
      <EvidenceStage title="Attacked input" description={attack ? `${attack.attack}, severity ${attack.severity}, paired with the same source sample.` : "Recipe evidence will appear after a completed run."}>
        {sourceLabel(sample, "attacked_image", "No attacked variant yet.")}
      </EvidenceStage>
      <EvidenceStage title="Clean model prediction" description="Prediction before perturbation, evaluated against the locked ground truth.">
        {sourceLabel(sample, "overlay_image", "No clean prediction evidence yet.")}
      </EvidenceStage>
      <EvidenceStage title="Attacked model prediction" description="Paired prediction after the final ordered recipe step.">
        {sourceLabel(sample, "overlay_image", "No attacked prediction evidence yet.")}
      </EvidenceStage>
      <section className="workspace__metrics" aria-label="Five key metrics">
        <header><p className="evidence-stage__eyebrow">Decision summary</p><h2>Five key metrics</h2></header>
        <FiveMetrics report={report} />
      </section>
    </div>
  );
}
