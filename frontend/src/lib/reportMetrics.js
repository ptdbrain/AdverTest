const TASK_METRICS = {
  detection2d: [
    { key: "map50_95", label: "mAP@50-95", help: "Độ chính xác phát hiện trung bình trên IoU 0.50–0.95" },
    { key: "ap50", label: "AP@50", help: "AP tại ngưỡng IoU 0.50" },
    { key: "ap75", label: "AP@75", help: "AP tại ngưỡng IoU 0.75" },
    { key: "precision", label: "Precision", help: "Tỷ lệ dự đoán dương đúng" },
    { key: "recall", label: "Recall", help: "Tỷ lệ đối tượng thật được phát hiện" },
    { key: "detection_rate", label: "Detection Rate", help: "Tỷ lệ ground truth được khớp ở IoU 0.5" },
    { key: "miss_rate", label: "Miss Rate", help: "Tỷ lệ ground truth bị bỏ sót (1 − Detection Rate)" },
  ],
  segmentation: [
    { key: "miou", label: "mIoU", help: "IoU trung bình theo lớp" },
    { key: "boundary_iou", label: "Boundary IoU", help: "Độ khớp tại biên mask" },
    { key: "pixel_accuracy", label: "Pixel Accuracy", help: "Tỷ lệ pixel phân loại đúng trên toàn ảnh" },
    { key: "pixel_precision", label: "Pixel precision", help: "Precision ở mức pixel" },
    { key: "pixel_recall", label: "Pixel recall", help: "Recall ở mức pixel" },
  ],
  detection3d: [
    { key: "kitti_3d_ap", label: "KITTI 3D AP", help: "Average Precision 3D theo protocol KITTI" },
    { key: "mean_bev_iou", label: "BEV IoU", help: "IoU trung bình trên mặt phẳng nhìn từ trên" },
    { key: "detection3d_precision", label: "3D precision", help: "Precision của khung 3D" },
    { key: "detection3d_recall", label: "3D recall", help: "Recall của khung 3D" },
    { key: "kitti_3d_ap_far", label: "Far-range AP", help: "AP cho vật thể cách từ 40 m" },
  ],
};

function finite(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function resolveReportTask(report = {}) {
  const explicit = report?.provenance?.run_config?.task_id || report.task_id || report.task;
  if (TASK_METRICS[explicit]) return explicit;
  const clean = report?.metrics?.clean || {};
  if (finite(clean.kitti_3d_ap) != null) return "detection3d";
  if (finite(clean.miou) != null) return "segmentation";
  return "detection2d";
}

export function metricCatalogForTask(taskId) {
  return TASK_METRICS[taskId] || TASK_METRICS.detection2d;
}

export function filterCompletedRuns(runs = [], query = "") {
  const needle = query.trim().toLocaleLowerCase();
  if (!needle) return runs;
  return runs.filter((run) => {
    const report = run.report || {};
    const haystack = [
      run.run_id,
      report.model,
      report.dataset,
      resolveReportTask(report),
      report.provenance?.benchmark_protocol_id,
      report.provenance?.run_config?.benchmark_protocol_id,
    ]
      .filter(Boolean)
      .join(" ")
      .toLocaleLowerCase();
    return haystack.includes(needle);
  });
}

export function primaryMetricForCell(report = {}, cell = {}, taskId = resolveReportTask(report), metricKey = null) {
  const key = metricKey || metricCatalogForTask(taskId)[0].key;
  const measured = finite(cell?.metrics?.[key]);
  if (measured != null) return measured;
  const hasExplicitCleanMetric = finite(report?.metrics?.clean?.[key]) != null;
  return !hasExplicitCleanMetric && taskId !== "segmentation" ? finite(cell?.ap) : null;
}

export function buildRunDecisionView(report = {}) {
  const taskId = resolveReportTask(report);
  const catalog = metricCatalogForTask(taskId);
  const primary = catalog[0];
  const metrics = report.metrics || {};
  const benchmarkAvailable =
    report.benchmark_metrics_available !== false && metrics.benchmark_metrics_available !== false;
  const cleanMetrics = metrics.clean || {};
  const explicitPrimary = finite(cleanMetrics[primary.key]);
  const allowLegacyPrimary = explicitPrimary == null && taskId !== "segmentation";
  const legacyPrimary = allowLegacyPrimary ? finite(report.ap_clean) : null;
  const clean = benchmarkAvailable ? (explicitPrimary ?? legacyPrimary) : null;
  const cellValues = (report.cells || [])
    .map((cell) => primaryMetricForCell(report, cell, taskId, primary.key))
    .filter((value) => value != null);
  const attacked =
    benchmarkAvailable && cellValues.length
      ? cellValues.reduce((sum, value) => sum + value, 0) / cellValues.length
      : null;
  const robustnessRetained =
    clean != null && clean > 0 && attacked != null ? Math.max(0, Math.min(100, (attacked / clean) * 100)) : null;
  const provenance = report.provenance || {};
  const config = provenance.run_config || {};
  const dataState = !benchmarkAvailable
    ? "NO_GROUND_TRUTH"
    : clean == null || attacked == null
      ? "NO_DATA"
      : "MEASURED";
  const limitations = [];
  if (!benchmarkAvailable) limitations.push("Không có ground truth; không thể tính benchmark metric.");
  const implementation = cleanMetrics.metric_implementation || null;
  if (implementation === "advertest-greedy-interpolated-v1") {
    limitations.push(
      "AP/mAP được tính bằng matcher nội bộ AdverTest; không được trình bày là COCO API/pycocotools chính thức.",
    );
  }
  if (report.simulation_only !== false) limitations.push("SIMULATION: chưa phải bằng chứng an toàn production.");
  if (benchmarkAvailable && report.n_samples > 0 && report.n_samples < 100)
    limitations.push("Cỡ mẫu dưới 100; chỉ phù hợp smoke test, chưa phù hợp làm deployment gate.");

  return {
    taskId,
    dataState,
    primary: { ...primary, clean, attacked },
    robustnessRetained,
    degradationPercent: robustnessRetained == null ? null : Math.max(0, 100 - robustnessRetained),
    metrics: catalog.map((definition) => {
      const cleanValue = benchmarkAvailable ? finite(cleanMetrics[definition.key]) : null;
      const values = (report.cells || [])
        .map((cell) => finite(cell?.metrics?.[definition.key]))
        .filter((value) => value != null);
      return {
        ...definition,
        clean: cleanValue,
        attacked: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null,
      };
    }),
    protocol: {
      benchmarkProtocolId: provenance.benchmark_protocol_id || config.benchmark_protocol_id || null,
      datasetVersionId: provenance.dataset_version_id || config.dataset_version_id || null,
      split: provenance.split || config.split || config.dataset_params?.split || null,
      checkpointHash: provenance.model?.checkpoint_hash || null,
      seed: config.seed ?? null,
      iouThreshold: config.iou_threshold ?? null,
    },
    implementation,
    limitations,
  };
}

export function formatRatio(value, digits = 1) {
  return value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value, digits = 1, suffix = "") {
  return value == null ? "—" : `${value.toFixed(digits)}${suffix}`;
}
