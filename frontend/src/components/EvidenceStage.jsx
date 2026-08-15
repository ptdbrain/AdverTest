export default function EvidenceStage({ title, description, children, className = "" }) {
  return (
    <section className={`evidence-stage ${className}`} aria-label={title}>
      <header>
        <p className="evidence-stage__eyebrow">Evidence</p>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </header>
      <div className="evidence-stage__content">{children}</div>
    </section>
  );
}
