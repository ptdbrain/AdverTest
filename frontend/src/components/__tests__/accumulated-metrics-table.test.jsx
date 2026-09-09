import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import AccumulatedMetricsTable from "@/components/AccumulatedMetricsTable";

afterEach(cleanup);

// ─── helpers ───────────────────────────────────────────────────────────────

/**
 * Builds a minimal RunReport cell object for testing.
 *
 * @param {string} attack - Attack name.
 * @param {number} severity - Severity level.
 * @param {number} ap - AP@50 value for this cell.
 * @param {number|null} map50_95 - mAP@50-95 value for this cell.
 * @returns {object}
 */
function makeCell(attack, severity, ap, map50_95 = ap * 0.72) {
  return {
    attack,
    severity,
    ap,
    group: "A",
    metrics: { ap50: ap, map50_95 },
  };
}

/**
 * Builds a minimal accumulated RunReport.
 *
 * @param {number} apClean - Clean baseline AP.
 * @param {number|null} cleanMap - Clean baseline mAP (defaults to apClean * 0.72).
 * @param {object[]} cells - Array of CellResult objects.
 * @returns {object}
 */
function makeReport(apClean, cells, { cleanMap = apClean * 0.72, dataset = "kitti" } = {}) {
  return {
    ap_clean: apClean,
    dataset,
    cells,
    sample_results: [],
    worst_cases: [],
    provenance: {},
    metrics: { clean: { ap50: apClean, map50_95: cleanMap } },
  };
}

// ─── tests ────────────────────────────────────────────────────────────────

describe("AccumulatedMetricsTable", () => {
  it("renders empty state when report has no cells", () => {
    render(<AccumulatedMetricsTable report={makeReport(0.92, [])} />);
    expect(screen.getByText(/chưa có dữ liệu/i)).toBeDefined();
  });

  it("renders empty state when report is null", () => {
    render(<AccumulatedMetricsTable report={null} />);
    expect(screen.getByText(/chưa có dữ liệu/i)).toBeDefined();
  });

  it("shows the three metric rows (AP@50, mAP@50-95, Robustness)", () => {
    render(<AccumulatedMetricsTable report={makeReport(0.92, [makeCell("gaussian_noise", 3, 0.62)])} />);

    expect(screen.getByText("AP@50")).toBeDefined();
    expect(screen.getByText("mAP@50-95")).toBeDefined();
    expect(screen.getByText("Robustness (giữ lại)")).toBeDefined();
  });

  it("renders a baseline column with the clean values", () => {
    render(
      <AccumulatedMetricsTable
        report={makeReport(0.92, [makeCell("gaussian_noise", 3, 0.62)], {
          cleanMap: 0.68,
        })}
      />,
    );

    // Baseline column header + clean AP/mAP + 100% robustness
    expect(screen.getByText("baseline")).toBeDefined();
    expect(screen.getByText("0.920")).toBeDefined();
    expect(screen.getByText("0.680")).toBeDefined();
    expect(screen.getByText("100.0%")).toBeDefined();
  });

  it("adds one column per accumulated attack combination", () => {
    // Simulates useAdverTest output after three runs on the same dataset:
    //   Run 1 → gaussian_noise @3
    //   Run 2 → fgsm @3
    //   Run 3 → motion_blur @5
    render(
      <AccumulatedMetricsTable
        report={makeReport(0.92, [
          makeCell("gaussian_noise", 3, 0.62, 0.45),
          makeCell("fgsm", 3, 0.55, 0.41),
          makeCell("motion_blur", 5, 0.7, 0.52),
        ])}
      />,
    );

    // Canonical column keys from getCanonicalAttackKey
    expect(screen.getByText("gaussian_noise_3")).toBeDefined();
    expect(screen.getByText("fgsm_3")).toBeDefined();
    expect(screen.getByText("motion_blur_5")).toBeDefined();

    // Combo AP values render in the AP@50 row
    expect(screen.getByText("0.620")).toBeDefined();
    expect(screen.getByText("0.550")).toBeDefined();
    expect(screen.getByText("0.700")).toBeDefined();

    // Combo mAP values render in the mAP@50-95 row
    expect(screen.getByText("0.450")).toBeDefined();
    expect(screen.getByText("0.410")).toBeDefined();
    expect(screen.getByText("0.520")).toBeDefined();
  });

  it("computes robustness as retained AP (AP_combo / AP_clean)", () => {
    // ap=0.62 vs ap_clean=0.92 → 67.4% retained
    render(
      <AccumulatedMetricsTable
        report={makeReport(0.92, [makeCell("gaussian_noise", 3, 0.62)], {
          cleanMap: 0.68,
        })}
      />,
    );

    expect(screen.getByText("67.4%")).toBeDefined();
  });

  it("counts the number of accumulated combinations in the header", () => {
    render(
      <AccumulatedMetricsTable
        report={makeReport(0.92, [makeCell("gaussian_noise", 3, 0.62), makeCell("fgsm", 3, 0.55)])}
      />,
    );

    // Header subtitle references the 2 accumulated combinations
    expect(screen.getByText(/2 tổ hợp tấn công tích lũy/i)).toBeDefined();
  });

  it("renders an em-dash for a missing clean mAP baseline", () => {
    render(
      <AccumulatedMetricsTable
        report={makeReport(0.92, [makeCell("gaussian_noise", 3, 0.62)], {
          cleanMap: null,
        })}
      />,
    );

    // mAP baseline is absent → "—" (metric row uses formatter's fallback)
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(1);
  });
});
