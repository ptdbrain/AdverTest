import { describe, expect, it } from "vitest";

import { normalizeDefenseReport } from "@/lib/defenseReport";

describe("normalizeDefenseReport", () => {
  it("keeps missing evidence as no data instead of replacing it with zero", () => {
    const report = normalizeDefenseReport(null);

    expect(report.eligibility.status).toBe("NOT_ELIGIBLE");
    expect(report.metricDeltas).toEqual([]);
    expect(report.displayMetric(null)).toBe("—");
  });

  it("normalizes the canonical metric array and exposes evidence remediation", () => {
    const report = normalizeDefenseReport({
      eligibility: { status: "NOT_ELIGIBLE", reasons: ["GROUND_TRUTH_HASH_MISSING"] },
      metric_deltas: [{ key: "ap50", attacked_delta: 0.1, unit: "ratio" }],
      decision: { next_action: "Rerun the locked benchmark." },
    });

    expect(report.metricDeltas).toHaveLength(1);
    expect(report.nextAction).toBe("Rerun the locked benchmark.");
    expect(report.isEligible).toBe(false);
  });
});
