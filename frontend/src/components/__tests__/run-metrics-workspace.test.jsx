import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RunMetricsContent } from "@/components/analytics/RunMetricsWorkspace";

const report3d = {
  run_id: "run-real-3d",
  model: "pointpillars",
  dataset: "kitti3d",
  n_samples: 42,
  ap_clean: 0.62,
  simulation_only: false,
  benchmark_metrics_available: true,
  evidence: { status: "VERIFIED", missing: [] },
  metrics: { clean: { kitti_3d_ap: 0.62, mean_bev_iou: 0.71 } },
  cells: [{ attack: "lidar_fog", severity: 3, ap: 0.39, degradation_percent: 37.1, metrics: { kitti_3d_ap: 0.39, mean_bev_iou: 0.48 } }],
  provenance: { run_config: { task_id: "detection3d", benchmark_protocol_id: "kitti-val-v1", split: "val", seed: 42 } },
};

describe("RunMetricsContent", () => {
  it("shows an honest no-data state without rendering zero as evidence", () => {
    render(<RunMetricsContent report={null} mode="benchmark" analytics={{}} />);
    expect(screen.getByText(/Chưa có benchmark hoàn tất/i)).toBeVisible();
    expect(screen.queryByText("0.0%")).toBeNull();
  });

  it("uses task-aware 3D metrics and protocol provenance", () => {
    render(<RunMetricsContent report={report3d} mode="benchmark" analytics={{}} />);
    expect(screen.getAllByText("KITTI 3D AP").length).toBeGreaterThan(0);
    expect(screen.getAllByText("62.0%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("39.0%").length).toBeGreaterThan(0);
    expect(screen.getByText("kitti-val-v1")).toBeVisible();
    expect(screen.queryByText(/mAP@0.5 qua các lần/i)).toBeNull();
  });

  it("shows measured findings and limitations instead of generated advice", () => {
    render(
      <RunMetricsContent
        report={{ ...report3d, simulation_only: true, n_samples: 8, evidence: { status: "NOT_ELIGIBLE", missing: ["simulation_only"] } }}
        mode="analysis"
        analytics={{
          summary: { worst_attack: { attack: "lidar_fog", mean_degradation_percent: 37.1 } },
          classes: [{ class_name: "Car", detection_drop_percent: 22.0, total_ground_truth_objects: 12 }],
          distance: { most_vulnerable_distance: "far", buckets: {} },
        }}
      />,
    );
    expect(screen.getAllByText(/lidar_fog/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Car/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/NOT ELIGIBLE/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Không đủ dữ liệu benchmark/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/L_adv = 0.4/i)).toBeNull();
  });
});
