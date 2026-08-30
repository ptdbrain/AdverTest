import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";

import VisualResultsPage from "@/app/experiments/[id]/results/page";

const { replace, getRunReport, getRunSamples, getSession, scopedHref } = vi.hoisted(() => ({
  replace: vi.fn(), getRunReport: vi.fn(), getRunSamples: vi.fn(), getSession: vi.fn(),
  scopedHref: (path, selectedRunId) => `${path}?project_id=project-a${selectedRunId ? `&run_id=${selectedRunId}` : ""}`,
}));
let runId = "";

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "EXP-2025-0512-001" }),
  useRouter: () => ({ replace }),
  useSearchParams: () => new URLSearchParams(runId ? `project_id=project-a&run_id=${runId}` : "project_id=project-a"),
}));
vi.mock("@/context/ProjectContext", () => ({
  useProjectContext: () => ({ projectId: "project-a", scopedHref }),
}));
vi.mock("@/lib/api", () => ({
  artifactUrl: (path) => `http://127.0.0.1:8000${path}`,
  getRunReport,
  getRunSamples,
  getSession,
  triggerAutoFlag: vi.fn(),
}));

afterEach(() => {
  cleanup();
  runId = "";
  vi.clearAllMocks();
});

describe("VisualResultsPage evidence routing", () => {
  it("shows no-data evidence state without a canonical run id", () => {
    getSession.mockResolvedValue({ runs: [] });
    render(<VisualResultsPage />);
    expect(screen.getByTestId("clean-evidence-empty")).toBeInTheDocument();
    expect(screen.getByText("Chưa có quan sát từ prediction")).toBeInTheDocument();
  });

  it("loads paired artifacts and predictions from the run id in the URL", async () => {
    runId = "run-real-001";
    getSession.mockResolvedValue({ runs: [] });
    const sample = {
      sample_id: "sample-real-001",
      attack: "fgsm",
      severity: 4,
      artifacts: { clean_input_url: "/data/runs/real/clean.png", attacked_input_url: "/data/runs/real/attacked.png" },
      ground_truth: { image_width: 200, image_height: 100, objects: [{ label: "truck", xyxy: [8, 10, 90, 92] }] },
      clean_prediction: { boxes: [{ xyxy: [10, 12, 80, 88], label: "truck-clean", score: 0.73 }] },
      attacked_prediction: { boxes: [{ xyxy: [12, 20, 60, 70], label: "truck-attacked", score: 0.21 }] },
    };
    getRunReport.mockResolvedValue({ run_id: runId, evidence: { status: "NOT_ELIGIBLE" }, sample_results: [sample] });
    getRunSamples.mockResolvedValue([sample]);

    render(<VisualResultsPage />);

    await waitFor(() => expect(getRunReport).toHaveBeenCalledWith("run-real-001"));
    expect(screen.getByTestId("clean-result-image")).toHaveAttribute("src", "http://127.0.0.1:8000/data/runs/real/clean.png");
    expect(screen.getByTestId("attacked-prediction-label-0")).toHaveTextContent("truck-attacked");
  });

  it("does not display benchmark scores for non-verified evidence", async () => {
    runId = "run-ineligible";
    getSession.mockResolvedValue({ runs: [] });
    getRunReport.mockResolvedValue({ run_id: runId, ap_clean: 0.99, evidence: { status: "NOT_ELIGIBLE" }, cells: [{ ap: 0.01 }], sample_results: [] });
    getRunSamples.mockResolvedValue([]);

    render(<VisualResultsPage />);

    await waitFor(() => expect(getRunReport).toHaveBeenCalledWith("run-ineligible"));
    expect(screen.getAllByText("Chưa có dữ liệu").length).toBeGreaterThan(0);
    expect(screen.queryByText("0.99")).toBeNull();
  });
});
