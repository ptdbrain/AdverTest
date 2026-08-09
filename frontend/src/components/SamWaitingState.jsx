export default function SamWaitingState() {
  return (
    <section className="sam-waiting" aria-label="SAM2 handoff status">
      <p className="evidence-stage__eyebrow">Segmentation mode</p>
      <h2>Waiting for Person C artefacts</h2>
      <p>
        SAM2 remains unavailable until its runnable adapter, real masks, fixed prompt protocol,
        evaluator, and checkpoint metadata are delivered and validated.
      </p>
    </section>
  );
}
