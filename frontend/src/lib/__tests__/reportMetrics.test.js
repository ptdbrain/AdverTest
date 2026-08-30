import { describe, expect, it } from "vitest";

import { buildRunDecisionView, filterCompletedRuns, metricCatalogForTask, primaryMetricForCell } from "@/lib/reportMetrics";

describe("task-aware report metrics", () => {
  it.each([
    ["detection2d", "mAP@50-95", "map50_95"],
    ["segmentation", "mIoU", "miou"],
    ["detection3d", "KITTI 3D AP", "kitti_3d_ap"],
  ])("uses the meaningful headline for %s", (task, label, key) => {
    const catalog = metricCatalogForTask(task);
    expect(catalog[0]).toMatchObject({ label, key });
  });

  it("never turns missing benchmark evidence into zero or an optimistic score", () => {
    const view = buildRunDecisionView({
      run_id: "quick-1",
      benchmark_metrics_available: false,
      metrics: { benchmark_metrics_available: false },
      cells: [],
      provenance: { run_config: { task_id: "detection2d" } },
    });

    expect(view.dataState).toBe("NO_GROUND_TRUTH");
    expect(view.primary.clean).toBeNull();
    expect(view.primary.attacked).toBeNull();
    expect(view.robustnessRetained).toBeNull();
  });

  it("does not relabel legacy AP as mAP when attacked mAP is missing", () => {
    const report = {
      ap_clean: 0.9,
      metrics: { clean: { map50_95: 0.55 } },
      cells: [{ ap: 0.4, metrics: { ap50: 0.4 } }],
      provenance: { run_config: { task_id: "detection2d" } },
      benchmark_metrics_available: true,
    };
    const view = buildRunDecisionView(report);

    expect(view.dataState).toBe("NO_GROUND_TRUTH");
    expect(view.primary.clean).toBeNull();
    expect(view.primary.attacked).toBeNull();
    expect(primaryMetricForCell(report, report.cells[0], view.taskId, view.primary.key)).toBeNull();
  });

  it("uses measured cell metrics and carries protocol provenance", () => {
    const view = buildRunDecisionView({
      run_id: "run-3d",
      n_samples: 42,
      ap_clean: 0.62,
      metrics: { clean: { kitti_3d_ap: 0.62, mean_bev_iou: 0.71 } },
      cells: [
        { attack: "lidar_fog", severity: 3, ap: 0.39, metrics: { kitti_3d_ap: 0.39, mean_bev_iou: 0.48 } },
      ],
      provenance: {
        run_config: { task_id: "detection3d", benchmark_protocol_id: "kitti-val-v1" },
        dataset_version_id: "sha256:data",
      },
      simulation_only: false,
      benchmark_metrics_available: true,
      evidence: { status: "VERIFIED", missing: [] },
    });

    expect(view.primary).toMatchObject({ label: "KITTI 3D AP", clean: 0.62, attacked: 0.39 });
    expect(view.robustnessRetained).toBeCloseTo(62.9032, 3);
    expect(view.protocol.benchmarkProtocolId).toBe("kitti-val-v1");
    expect(view.protocol.datasetVersionId).toBe("sha256:data");
  });

  it("filters a long run history by run, model, dataset, or task", () => {
    const runs = [
      { run_id: "run-yolo", report: { model: "yolo11", dataset: "coco", task_id: "detection2d" } },
      { run_id: "run-kitti", report: { model: "pointpillars", dataset: "kitti3d", task_id: "detection3d" } },
    ];

    expect(filterCompletedRuns(runs, "POINTPILLARS")).toEqual([runs[1]]);
    expect(filterCompletedRuns(runs, "kitti3d")).toEqual([runs[1]]);
    expect(filterCompletedRuns(runs, "run-yolo")).toEqual([runs[0]]);
    expect(filterCompletedRuns(runs, "detection3d")).toEqual([runs[1]]);
  });
});
