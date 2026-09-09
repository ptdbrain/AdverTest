import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import SampleReviewPanel from "@/components/reviews/SampleReviewPanel";

const api = vi.hoisted(() => ({
  artifactUrl: vi.fn((value) => value),
  createAdversarialDataset: vi.fn(),
  getApiBase: vi.fn(() => ""),
  getSampleReviews: vi.fn(),
  upsertSampleReviews: vi.fn(),
}));

vi.mock("@/lib/api", () => api);

const samples = [
  {
    sample_id: "scene-1",
    attack: "fog",
    severity: 2,
    artifacts: { clean_input_url: "/clean-1.png", attacked_input_url: "/attack-1.png" },
    ground_truth: { objects: [{ label: "car" }] },
    clean_prediction: { boxes: [{ label: "car", score: 0.91 }] },
    attacked_prediction: { boxes: [{ label: "truck", score: 0.31 }] },
    degradation_hint: 65.9,
  },
  {
    sample_id: "scene-1",
    attack: "rain",
    severity: 4,
    artifacts: { clean_input_url: "/clean-2.png", attacked_input_url: "/attack-2.png" },
    ground_truth: { objects: [{ label: "pedestrian" }] },
    clean_prediction: { boxes: [{ label: "pedestrian", score: 0.88 }] },
    attacked_prediction: { boxes: [] },
    metric: { name: "AP50", clean: 0.8, attacked: 0.2 },
    degradation: 75,
  },
];

describe("SampleReviewPanel evidence and decisions", () => {
  afterEach(cleanup);

  beforeEach(() => {
    api.getSampleReviews.mockReset().mockResolvedValue([
      { sample_id: "scene-1", attack: "rain", severity: 4, decision: "REJECTED" },
    ]);
    api.upsertSampleReviews.mockReset().mockImplementation(async (_runId, reviews) => reviews);
    api.createAdversarialDataset.mockReset();
  });

  it("opens evidence for the exact composite sample key", async () => {
    const user = userEvent.setup();
    render(<SampleReviewPanel runId="run-1" samples={samples} projectId="project-1" />);

    await user.click(await screen.findByRole("button", { name: "Xem scene-1 · rain · mức 4" }));

    expect(screen.getByRole("img", { name: "Ảnh gốc scene-1" })).toHaveAttribute("src", "/clean-2.png");
    expect(screen.getByRole("img", { name: "Ảnh attacked scene-1" })).toHaveAttribute("src", "/attack-2.png");
    expect(screen.getByText(/GT: pedestrian/)).toBeVisible();
    expect(screen.getByText(/Prediction gốc: pedestrian 0.88/)).toBeVisible();
    expect(screen.getByText(/Prediction attacked: —/)).toBeVisible();
    expect(screen.getByText(/AP50.*0.8.*0.2/)).toBeVisible();
    expect(screen.getByText(/Suy giảm: 75%/)).toBeVisible();
  });

  it("offers Pending, Approve and Reject and persists the composite key", async () => {
    const user = userEvent.setup();
    render(<SampleReviewPanel runId="run-1" samples={samples} projectId="project-1" />);
    await screen.findByRole("button", { name: "Duyệt" });

    await user.click(screen.getByRole("button", { name: "Từ chối" }));

    await waitFor(() =>
      expect(api.upsertSampleReviews).toHaveBeenCalledWith(
        "run-1",
        [expect.objectContaining({ sample_id: "scene-1", attack: "fog", severity: 2, decision: "REJECTED" })],
        "project-1",
      ),
    );
    expect(screen.getByRole("button", { name: "Chờ duyệt" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Duyệt" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Từ chối" })).toBeVisible();
  });

  it("shows the task-native export format and skipped sample reasons", async () => {
    api.getSampleReviews.mockResolvedValueOnce([
      { sample_id: "scene-1", attack: "fog", severity: 2, decision: "APPROVED" },
    ]);
    api.createAdversarialDataset.mockResolvedValueOnce({
      dataset_id: "ads-seg",
      sample_count: 1,
      format: "segmentation-mask",
      class_map: { 1: "road" },
      skipped: [{ sample_id: "scene-2", reason: "mask missing" }],
      download_url: "/download/ads-seg",
    });
    const user = userEvent.setup();
    render(<SampleReviewPanel runId="run-1" samples={samples} projectId="project-1" />);

    await user.click(await screen.findByRole("button", { name: /Tạo Adversarial Dataset/i }));

    expect(await screen.findByText(/Định dạng segmentation-mask/)).toHaveTextContent("images/ + masks/");
    expect(screen.getByText(/scene-2 \(mask missing\)/)).toBeVisible();
    expect(api.createAdversarialDataset).toHaveBeenCalledWith("run-1", "project-1");
  });
});
