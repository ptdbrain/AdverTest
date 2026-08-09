const METRICS = [
  ["Clean detection score", "ap_clean", "score"],
  ["Score after attack", "attacked", "score"],
  ["Performance lost", "degradation", "percent"],
  ["Objects broken", "broken", "count"],
  ["Overall robustness", "robust_score_normalized", "points"],
];

function number(value, unit) {
  if (value == null) return "Unavailable";
  if (unit === "percent") return `${(value * 100).toFixed(1)}%`;
  return Number(value).toFixed(unit === "count" ? 0 : 3);
}

export default function FiveMetrics({ report }) {
  const firstCell = report?.cells?.[0];
  const values = {
    ap_clean: report?.ap_clean,
    attacked: firstCell?.ap,
    degradation: firstCell?.degradation,
    broken: firstCell?.metrics?.objects_broken,
    robust_score_normalized: report?.metrics?.robust_score_normalized,
  };

  return (
    <div className="five-metrics">
      {METRICS.map(([label, key, unit], index) => (
        <div className="five-metrics__item" key={key}>
          <span>{String(index + 1).padStart(2, "0")}</span>
          <strong>{number(values[key], unit)}</strong>
          <p>{label}</p>
          <small>{unit}</small>
        </div>
      ))}
    </div>
  );
}
