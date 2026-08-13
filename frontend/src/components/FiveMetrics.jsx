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

export default function FiveMetrics({ report, selectedResult }) {
  const finalRecipeResult = selectedResult ?? report?.cells?.at(-1) ?? null;
  const benchmarkAvailable = report?.benchmark_metrics_available !== false;
  const values = {
    ap_clean: benchmarkAvailable ? report?.ap_clean : null,
    attacked: benchmarkAvailable ? finalRecipeResult?.ap : null,
    degradation: benchmarkAvailable ? finalRecipeResult?.degradation_ratio : null,
    broken: benchmarkAvailable ? finalRecipeResult?.metrics?.objects_broken : null,
    robust_score_normalized: benchmarkAvailable ? report?.metrics?.robustness?.robust_score_normalized : null,
  };

  return (
    <div className="five-metrics">
      {!benchmarkAvailable && <p className="five-metrics__unavailable">No ground truth supplied. Benchmark AP/mAP metrics are unavailable.</p>}
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
