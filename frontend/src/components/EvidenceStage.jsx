export default function EvidenceStage({ title, description, children }) {
  return (
    <section className="evidence-stage" aria-label={title}>
      <header>
        <p className="evidence-stage__eyebrow">Evidence stage</p>
        <h2>{title}</h2>
        <p>{description}</p>
      </header>
      <div className="evidence-stage__content">{children}</div>
    </section>
  );
}
