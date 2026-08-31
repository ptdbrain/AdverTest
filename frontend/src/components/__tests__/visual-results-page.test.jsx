import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import VisualResultsPage from "@/app/experiments/[id]/results/page";

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "EXP-2025-0512-001" }),
}));

vi.mock("@/lib/api", () => ({
  artifactUrl: (path) => `http://127.0.0.1:8000${path}`,
  getSession: async () => {
    const experiment = JSON.parse(localStorage.getItem("adversai_active_experiment") || "{}");
    const lastSession = JSON.parse(localStorage.getItem("advertest_last_session") || "{}");
    return { ...lastSession, ...experiment, report: lastSession.report, samples: lastSession.samples };
  },
  getRunReport: async () => JSON.parse(localStorage.getItem("advertest_last_session") || "{}").report || {},
  getRunSamples: async () => JSON.parse(localStorage.getItem("advertest_last_session") || "{}").samples || [],
  triggerAutoFlag: async () => ({}),
}));

afterEach(() => {
  cleanup();
  localStorage.clear();
});

describe("VisualResultsPage layout contract", () => {
  it("renders the paired evidence bands and experiment rail", () => {
    render(<VisualResultsPage />);

    expect(screen.getByTestId("visual-results-comparison")).toBeDefined();
    expect(screen.getByTestId("clean-result-panel")).toBeDefined();
    expect(screen.getByTestId("attacked-result-panel")).toBeDefined();
    expect(screen.getByTestId("experiment-info-panel")).toBeDefined();
    expect(screen.getByTestId("clean-evidence-empty")).toBeDefined();
    expect(screen.getByTestId("clean-metrics-strip").className).toContain("xl:grid-cols-5");
    expect(screen.getByText("Bản đồ khác biệt (|Δ|)")).toBeDefined();
    expect(screen.getByText("Sụt giảm Confidence (Độ tin cậy)")).toBeDefined();
    expect(screen.getByText("Nhiễu đối kháng (Perturbation)")).toBeDefined();
    expect(screen.getByText("Quan sát nhanh")).toBeDefined();
  });

  it("renders experiment metadata from the saved configuration and run session", async () => {
    localStorage.setItem("adversai_active_experiment", JSON.stringify({
      expName: "EXP-2025-0512-001",
      selectedTask: "classification",
      taskName: "Phân loại ảnh thực tế",
      selectedModelName: "ResNet50 checkpoint thật",
      selectedDatasetName: "Dataset xác minh 2026",
      selectedModelFormat: ".pt",
      selectedDatasetId: "dataset-real",
      updatedAt: "2026-08-27T10:20:00.000Z",
      attackQueue: [{ id: "fgsm", name: "FGSM thực tế", severity: 4, eps: "4 / 255", norm: "Linf" }],
      status: "COMPLETED",
      activeRunId: "RUN-REAL-42",
      executedAt: "2026-08-27T10:21:00.000Z",
    }));
    localStorage.setItem("advertest_last_session", JSON.stringify({
      runId: "API-RUN-42",
      selectedDataset: "dataset-real",
      selectedModelFamily: "family-real",
      selectedModelVersion: "version-real",
    recipe: {
      name: "fgsm-real",
      steps: [{ attack_name: "fgsm", severity: 4, parameters: { epsilon: 0.02, steps: 7, alpha: 0.003 } }],
    },
    report: { run_id: "API-RUN-42" },
    runStatus: "COMPLETED",
  }));

    render(<VisualResultsPage />);

    await waitFor(() => expect(screen.getByText("ResNet50 checkpoint thật")).toBeDefined());
    expect(screen.getByText("Dataset xác minh 2026")).toBeDefined();
    expect(screen.getByText("FGSM thực tế")).toBeDefined();
    expect(screen.getAllByText("Hoàn thành").length).toBe(2);
    expect(screen.queryByText("YOLOv8n")).toBeNull();
    expect(screen.queryByText("PGD (L∞)")).toBeNull();
  });

  it("does not invent metadata when previous-tab data is unavailable", async () => {
    localStorage.setItem("adversai_active_experiment", JSON.stringify({ status: "COMPLETED" }));
    render(<VisualResultsPage />);

    await waitFor(() => expect(screen.getAllByText("Chưa có dữ liệu").length).toBeGreaterThan(0));
    expect(screen.queryByText("Hoàn thành")).toBeNull();
    expect(screen.getAllByText("Chưa có bằng chứng ảnh").length).toBe(2);
    expect(screen.queryByTestId("clean-prediction-box-0")).toBeNull();
    expect(screen.getAllByText("Chưa có artifact phân tích").length).toBeGreaterThan(0);
    expect(screen.getByText("Chưa có quan sát từ prediction")).toBeDefined();
    expect(screen.queryByAltText("Bản đồ khác biệt")).toBeNull();
    expect(screen.queryByText("YOLOv8n")).toBeNull();
    expect(screen.queryByText("PGD (L∞)")).toBeNull();
  });

  it("renders image artifacts and prediction boxes from the completed sample result", async () => {
    const sample = {
      sample_id: "sample-real-001",
      attack: "fgsm",
      severity: 4,
      artifacts: {
        clean_input_url: "/data/runs/real/clean.png",
        attacked_input_url: "/data/runs/real/attacked.png",
      },
      ground_truth: {
        type: "boxes",
        image_width: 200,
        image_height: 100,
        objects: [{ object_id: "gt-1", label: "truck-ground-truth", xyxy: [8, 10, 90, 92] }],
      },
      clean_prediction: {
        prediction_type: "detection",
        boxes: [{ xyxy: [10, 12, 80, 88], label: "truck-clean-real", score: 0.73 }],
      },
      attacked_prediction: {
        prediction_type: "detection",
        boxes: [{ xyxy: [12, 20, 60, 70], label: "truck-attacked-real", score: 0.21 }],
      },
    };
    localStorage.setItem("advertest_last_session", JSON.stringify({
      runId: "API-RUN-REAL-001",
      selectedDataset: "dataset-real",
      selectedModelFamily: "family-real",
      selectedModelVersion: "version-real",
      report: {
        model: "model-real",
        model_version: "version-real",
        dataset: "dataset-real",
        ap_clean: 0.73,
        metrics: { clean: { ap50: 0.73 } },
        cells: [{ attack: "fgsm", severity: 4, ap: 0.21, degradation_percent: 71.23, metrics: { ap50: 0.21 } }],
        samples: [sample],
      },
      samples: [sample],
      runStatus: "COMPLETED",
    }));

    render(<VisualResultsPage />);

    await waitFor(() => expect(screen.getByTestId("clean-prediction-box-0")).toBeDefined());
    expect(screen.getByTestId("clean-result-image").getAttribute("src")).toBe("http://127.0.0.1:8000/data/runs/real/clean.png");
    expect(screen.getByTestId("attacked-result-image").getAttribute("src")).toBe("http://127.0.0.1:8000/data/runs/real/attacked.png");
    expect(screen.getByTestId("clean-prediction-label-0").textContent).toContain("truck-clean-real");
    expect(screen.getByTestId("attacked-prediction-label-0").textContent).toContain("truck-attacked-real");
    expect(screen.getByTestId("clean-prediction-box-0").getAttribute("x")).toBe("10");
    expect(screen.getByTestId("clean-prediction-box-0").getAttribute("y")).toBe("12");
    expect(screen.getByText("71.23%")).toBeDefined();
    expect(screen.queryByText("car 0.96")).toBeNull();
    expect(screen.queryByText("1024 × 768")).toBeNull();
    expect(screen.queryByAltText("Bản đồ khác biệt")).toBeNull();
  });

  it("renders sample navigator and switches between multiple samples", async () => {
    const sample1 = {
      sample_id: "kitti_000000",
      artifacts: { clean_input_url: "/data/runs/real/clean_0.png", attacked_input_url: "/data/runs/real/attacked_0.png" },
      clean_prediction: { boxes: [{ xyxy: [1, 2, 3, 4], label: "car", score: 0.9 }] },
      attacked_prediction: { boxes: [] },
    };
    const sample2 = {
      sample_id: "kitti_000001",
      artifacts: { clean_input_url: "/data/runs/real/clean_1.png", attacked_input_url: "/data/runs/real/attacked_1.png" },
      clean_prediction: { boxes: [{ xyxy: [5, 6, 7, 8], label: "pedestrian", score: 0.85 }] },
      attacked_prediction: { boxes: [{ xyxy: [5, 6, 7, 8], label: "pedestrian", score: 0.3 }] },
    };

    localStorage.setItem("advertest_last_session", JSON.stringify({
      runId: "API-RUN-MULTI",
      report: { samples: [sample1, sample2] },
      samples: [sample1, sample2],
      runStatus: "COMPLETED",
    }));

    render(<VisualResultsPage />);

    await waitFor(() => expect(screen.getByText("Bộ ảnh kiểm thử (2 ảnh)")).toBeDefined());
    expect(screen.getByText("Ảnh 1")).toBeDefined();
    expect(screen.getByText("Ảnh 2")).toBeDefined();
    expect(screen.getAllByText("kitti_000000").length).toBeGreaterThan(0);

    // Click on Sample 2
    const sample2Btn = screen.getByText("Ảnh 2");
    sample2Btn.click();

    await waitFor(() => expect(screen.getAllByText("kitti_000001").length).toBeGreaterThan(0));
  });
});
