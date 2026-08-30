const MISSING_ELIGIBILITY = {
  status: "NOT_ELIGIBLE",
  reasons: ["REPORT_MISSING"],
};

export function displayMetric(value, suffix = "") {
  return Number.isFinite(value) ? `${value}${suffix}` : "—";
}

export function normalizeDefenseReport(payload) {
  const eligibility = payload?.eligibility ?? MISSING_ELIGIBILITY;
  return {
    raw: payload ?? null,
    eligibility,
    isEligible: eligibility.status === "ELIGIBLE",
    metricDeltas: Array.isArray(payload?.metric_deltas) ? payload.metric_deltas : [],
    recovery: payload?.recovery ?? { reason: "REPORT_MISSING" },
    failures: payload?.failures ?? null,
    nextAction: payload?.decision?.next_action || eligibility.next_action || "Create a paired benchmark with complete evidence.",
    displayMetric,
  };
}
