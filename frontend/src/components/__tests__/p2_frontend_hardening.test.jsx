import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";

import EvidenceBadge, { getEvidenceState } from "@/components/common/EvidenceBadge";
import { projectKitti3dTo2d } from "@/components/PointCloudViewer";
import { exportReportAsCsv, exportReportAsJson } from "@/lib/exportReport";

describe("P2.7 Evidence Badges", () => {
  it("uses the backend evidence decision and never infers real evidence from partial provenance", () => {
    expect(getEvidenceState({ checkpoint_sha256: "abc12345" }, false)).toBe("NO_DATA");
    expect(
      getEvidenceState(
        { checkpoint_sha256: "abc12345" },
        false,
        undefined,
        { status: "NOT_ELIGIBLE", missing: ["ground_truth_hash"] },
      ),
    ).toBe("NOT_ELIGIBLE");
    expect(getEvidenceState({}, false, undefined, { status: "VERIFIED" })).toBe("VERIFIED");
  });

  it("renders with proper accessibility role and label", () => {
    render(<EvidenceBadge state="VERIFIED" />);
    const badge = screen.getByRole("status");
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent("VERIFIED");
    expect(badge).toHaveAttribute("aria-label", "Trạng thái nguồn: VERIFIED");
  });
});

describe("P2.10 3D to 2D Camera Projection (KITTI Calibration)", () => {
  const dummyCalib = {
    P2: [
      [718.856, 0.0, 607.1928, 45.38225],
      [0.0, 718.856, 185.2157, -0.1130887],
      [0.0, 0.0, 1.0, 0.00378997],
    ],
    R_rect: [
      [0.9999, 0.0105, -0.0076, 0.0],
      [-0.0105, 0.9999, -0.0023, 0.0],
      [0.0076, 0.0022, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ],
    Tr_velo_to_cam: [
      [0.007533745, -0.9999714, -0.0006166, -0.0040697],
      [0.01480249, 0.000728, -0.9998902, -0.0763161],
      [0.9998621, 0.00752379, 0.0148075, -0.2717806],
      [0.0, 0.0, 0.0, 1.0],
    ],
  };

  it("projects a forward 3D point into valid positive pixel coordinates", () => {
    // 3D point 10 meters ahead in Velodyne frame: [10, 0, 0]
    const proj = projectKitti3dTo2d([10.0, 0.0, 0.0], dummyCalib);
    expect(proj).not.toBeNull();
    expect(proj.u).toBeGreaterThan(0);
    expect(proj.v).toBeGreaterThan(0);
    expect(proj.depth).toBeGreaterThan(0);
  });

  it("returns null for points behind the camera lens (z <= 0)", () => {
    // Point behind LiDAR / camera
    const proj = projectKitti3dTo2d([-10.0, 0.0, 0.0], dummyCalib);
    expect(proj).toBeNull();
  });

  it("returns null safely when calibration is missing", () => {
    const proj = projectKitti3dTo2d([10.0, 0.0, 0.0], null);
    expect(proj).toBeNull();
  });
});

describe("P2.11 Export Report Helpers", () => {
  it("formats CSV with versioned header and handles missing metrics honestly", () => {
    const mockReport = {
      run_id: "test-run-123",
      model: "pointpillars",
      model_version: "v1.0",
      dataset: "kitti3d",
      n_samples: 50,
      ap_clean: 0.85,
      cells: [
        { attack: "lidar_beam_drop", severity: 2, ap: 0.72, degradation_ratio: 0.1529 },
      ],
      provenance: { seed: 2026, split_manifest_hash: "abcd" },
      simulation_only: false,
    };

    // Spy URL.createObjectURL and document.createElement
    const createObjectURL = vi.fn().mockReturnValue("blob:dummy");
    global.URL.createObjectURL = createObjectURL;
    global.URL.revokeObjectURL = vi.fn();

    exportReportAsCsv(mockReport);
    expect(createObjectURL).toHaveBeenCalled();
  });
});
