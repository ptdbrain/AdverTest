import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import MethodComparisonTable from "@/components/MethodComparisonTable";

afterEach(cleanup);

// ─── helpers ───────────────────────────────────────────────────────────────

/**
 * Builds a minimal RunReport cell object for testing.
 *
 * @param {string} attack - Attack name.
 * @param {number} severity - Severity level.
 * @param {number} ap - AP value for this cell.
 * @returns {object}
 */
function makeCell(attack, severity, ap) {
  return { attack, severity, ap, group: "A", metrics: { ap50: ap } };
}

/**
 * Builds a minimal accumulated RunReport.
 *
 * @param {number} apClean - Clean baseline AP.
 * @param {object[]} cells - Array of CellResult objects.
 * @returns {object}
 */
function makeReport(apClean, cells) {
  return {
    ap_clean: apClean,
    cells,
    sample_results: [],
    worst_cases: [],
    provenance: {},
  };
}

// ─── tests ────────────────────────────────────────────────────────────────

describe("MethodComparisonTable", () => {
  it("renders empty state when report has no cells", () => {
    render(<MethodComparisonTable report={makeReport(0.9, [])} />);
    expect(screen.getByText(/chưa có dữ liệu/i)).toBeDefined();
  });

  it("renders empty state when report is null", () => {
    render(<MethodComparisonTable report={null} />);
    expect(screen.getByText(/chưa có dữ liệu/i)).toBeDefined();
  });

  // ── Run 1: single method with multiple severities ──────────────────────

  it("run 1 — shows baseline column + method1_sev columns", () => {
    const report = makeReport(0.92, [
      makeCell("gaussian_noise", 1, 0.81),
      makeCell("gaussian_noise", 3, 0.62),
      makeCell("gaussian_noise", 5, 0.44),
    ]);

    render(<MethodComparisonTable report={report} />);

    // Baseline column header must exist
    expect(screen.getByText("baseline")).toBeDefined();

    // Severity columns: canonical keys from getCanonicalAttackKey
    expect(screen.getByText("gaussian_noise_1")).toBeDefined();
    expect(screen.getByText("gaussian_noise_3")).toBeDefined();
    expect(screen.getByText("gaussian_noise_5")).toBeDefined();

    // Row label for the attack (without severity suffix) — may appear in th subtitle too
    expect(screen.getAllByText(/nhiễu.*gaussian/i).length).toBeGreaterThanOrEqual(1);

    // Baseline value displayed in the baseline cell
    expect(screen.getByText("0.920")).toBeDefined();

    // AP values for each severity
    expect(screen.getByText("0.810")).toBeDefined();
    expect(screen.getByText("0.620")).toBeDefined();
    expect(screen.getByText("0.440")).toBeDefined();
  });

  // ── Run 2+3: accumulated — new columns appear alongside existing ones ──

  it("run 1 then run 2+3 — accumulation adds method2 and method3 columns", () => {
    // Simulates what useAdverTest produces after accumulating three runs:
    //   Run 1 → gaussian_noise at severities 1,3,5
    //   Run 2 → fgsm at severity 3
    //   Run 3 → motion_blur at severity 5
    const report = makeReport(0.92, [
      makeCell("gaussian_noise", 1, 0.81),
      makeCell("gaussian_noise", 3, 0.62),
      makeCell("gaussian_noise", 5, 0.44),
      makeCell("fgsm", 3, 0.55),
      makeCell("motion_blur", 5, 0.7),
    ]);

    render(<MethodComparisonTable report={report} />);

    // All columns must be present
    expect(screen.getByText("baseline")).toBeDefined();
    expect(screen.getByText("gaussian_noise_1")).toBeDefined();
    expect(screen.getByText("gaussian_noise_3")).toBeDefined();
    expect(screen.getByText("gaussian_noise_5")).toBeDefined();
    expect(screen.getByText("fgsm_3")).toBeDefined();
    expect(screen.getByText("motion_blur_5")).toBeDefined();

    // Row labels: three distinct attacks (labels may appear in th subtitles too)
    expect(screen.getAllByText(/nhiễu.*gaussian/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/FGSM/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/mờ|blur/i).length).toBeGreaterThanOrEqual(1);
  });

  // ── Cell deduplication: latest run wins per canonical key ─────────────

  it("deduplicates cells with the same attack+severity key, keeping the latest", () => {
    // First run: ap=0.60, second run (same key): ap=0.75 — second must win
    const report = makeReport(0.92, [makeCell("gaussian_noise", 3, 0.6), makeCell("gaussian_noise", 3, 0.75)]);

    render(<MethodComparisonTable report={report} />);

    // Only ONE column for gaussian_noise_3
    const colHeaders = screen.getAllByText("gaussian_noise_3");
    expect(colHeaders).toHaveLength(1);

    // The winning AP value (last entry in the cells array wins via Map.set)
    expect(screen.getByText("0.750")).toBeDefined();
  });

  // ── Empty cell renders "—" ─────────────────────────────────────────────

  it("renders an em-dash for missing cells in the grid", () => {
    // gaussian_noise only at severity 1; fgsm only at severity 3
    // The gaussian_noise row has no fgsm_3 cell → should render "—"
    const report = makeReport(0.92, [makeCell("gaussian_noise", 1, 0.81), makeCell("fgsm", 3, 0.55)]);

    render(<MethodComparisonTable report={report} />);

    // Both attack rows present (may appear in th + td)
    expect(screen.getAllByText(/nhiễu.*gaussian/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/FGSM/i).length).toBeGreaterThanOrEqual(1);

    // Expect at least two "—" placeholders (one per missing cross-cell)
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThanOrEqual(2);
  });

  // ── Legend always rendered when data present ───────────────────────────

  it("renders the colour-scale legend", () => {
    const report = makeReport(0.92, [makeCell("gaussian_noise", 3, 0.62)]);
    render(<MethodComparisonTable report={report} />);
    expect(screen.getByText(/< 5% suy giảm/i)).toBeDefined();
  });
});
