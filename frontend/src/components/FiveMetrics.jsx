const DETECTION_METRICS = [
  ["AP@50", "ap50"],
  ["AP@75", "ap75"],
  ["mAP@50-95", "map50_95"],
];

function formatMetric(value) {
  return value == null ? "Unavailable" : Number(value).toFixed(3);
}

export default function BaseModelMetrics({ ariaLabel, title, metrics, fallbackAp, benchmarkAvailable }) {
  const values = {
    ...metrics,
    ap50: metrics?.ap50 ?? fallbackAp,
  };

  return (
    <section className="prediction-metrics" aria-label={ariaLabel}>
      <header>
        <p className="evidence-stage__eyebrow">Base model metrics</p>
        <h3>{title}</h3>
      </header>
      {!benchmarkAvailable ? (
        <p className="prediction-metrics__unavailable">No ground truth supplied. Benchmark AP/mAP metrics are unavailable.</p>
      ) : (
        <div className="prediction-metrics__grid">
          {DETECTION_METRICS.map(([label, key]) => (
            <div className="prediction-metrics__item" key={key}>
              <strong>{formatMetric(values[key])}</strong>
              <span>{label}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
