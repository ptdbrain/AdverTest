export default function ClosedLoopPanel({ report, backlog, trainingBlockedReason, isCreatingBacklog, error, onCreateBacklog }) {
  const measuredFailures = report?.cells?.filter((cell) => cell.degradation > 0) ?? [];
  const canCreateBacklog = measuredFailures.length > 0 && !backlog;

  return (
    <section className="closed-loop-panel" aria-label="Recovery workflow">
      <p className="evidence-stage__eyebrow">Recovery workflow</p>
      <h2>Measured failures become retraining evidence</h2>
      <p>{measuredFailures.length} measured attack cells are eligible for reviewer selection.</p>
      {backlog ? (
        <p><strong>{backlog.name}</strong>: {backlog.status} ({backlog.failure_ids.length} failures)</p>
      ) : (
        <button type="button" className="action-button action-button--primary" disabled={!canCreateBacklog || isCreatingBacklog} onClick={onCreateBacklog}>
          {isCreatingBacklog ? "Creating retraining backlog…" : "Create retraining backlog"}
        </button>
      )}
      {error && <p role="alert">{error}</p>}
      {trainingBlockedReason ? (
        <p role="status">Training is waiting for verified model artefacts.</p>
      ) : (
        <p role="status">Approved backlog can start a queued training run.</p>
      )}
    </section>
  );
}
