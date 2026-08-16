export default function ClosedLoopPanel({ report, backlog, trainingBlockedReason, isCreatingBacklog, error, onCreateBacklog }) {
  const measuredFailures = report?.cells?.filter((cell) => cell.degradation > 0) ?? [];
  const canCreateBacklog = measuredFailures.length > 0 && !backlog;

  return (
    <section className="closed-loop-panel" aria-label="Recovery workflow">
      <p className="evidence-stage__eyebrow">Recovery</p>
      <h2 style={{ fontSize: "0.85rem", fontWeight: 600, margin: "4px 0 8px" }}>Retraining Workflow</h2>
      <p>{measuredFailures.length} attack cells eligible for retraining.</p>
      {backlog ? (
        <p><strong>{backlog.name}</strong>: {backlog.status} ({backlog.failure_ids.length} failures)</p>
      ) : (
        <button type="button" className="action-button action-button--primary" disabled={!canCreateBacklog || isCreatingBacklog} onClick={onCreateBacklog} style={{ marginTop: 8 }}>
          {isCreatingBacklog ? "Creating backlog..." : "Create retraining backlog"}
        </button>
      )}
      {error && <p role="alert" className="text-xs" style={{ color: "var(--danger)", marginTop: 6 }}>{error}</p>}
      <p role="status" className="text-xs text-tertiary" style={{ marginTop: 6 }}>
        {trainingBlockedReason ? "Training is waiting for verified model artefacts." : "Approved backlog can start training."}
      </p>
    </section>
  );
}
