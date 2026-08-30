import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import DefensePage from "@/app/defense/page.jsx";
import {
  createModelComparison,
  createDefenceRun,
  getCheckpoint,
  getRun,
  getRunDefenceCandidates,
  listSessions,
  uploadCheckpoint,
} from "@/lib/api";

vi.mock("@/lib/api", () => ({
  createModelComparison: vi.fn(),
  createDefenceRun: vi.fn(),
  getCheckpoint: vi.fn(),
  getRun: vi.fn(),
  getRunDefenceCandidates: vi.fn(),
  listSessions: vi.fn(),
  uploadCheckpoint: vi.fn(),
}));

const sessions = [
  {
    id: "session-a",
    name: "Phiên YOLO11s trên KITTI",
    description: "Fog robustness",
    task_id: "detection2d",
    task_name: "Object Detection 2D",
    model_id: "base-checkpoint-a",
    model_name: "YOLO11s Production",
    dataset_id: "kitti-val",
    dataset_name: "KITTI Validation",
    created_at: "29/08/2026 20:00:00",
    updated_at: "29/08/2026 20:10:00",
    status: "active",
    runs: [
      {
        id: "display-run-a",
        name: "Depth Fog cấp 3",
        timestamp: "29/08/2026 20:05:00",
        attack_type: "depth_fog",
        attack_name: "Depth Fog",
        severity: 3,
        clean_map: 0.82,
        attacked_map: 0.45,
        map_drop_pct: 45.1,
        clean_conf: 0.92,
        attacked_conf: 0.68,
        psnr: "24.21 dB",
        ssim: "0.781",
        inference_ms: 56.9,
        robustness_score: 54.9,
        clean_bbox_count: 10,
        attacked_bbox_count: 5,
        sample_id: "000000",
        is_combined: false,
        attack_components: [],
        note: "",
        seed: 42,
        run_config_hash: "protocol-a",
        backend_run_id: "backend-run-a",
      },
    ],
  },
  {
    id: "session-b",
    name: "Phiên cũ chưa có evidence backend",
    description: "Legacy display run",
    task_id: "detection2d",
    task_name: "Object Detection 2D",
    model_id: "base-checkpoint-b",
    model_name: "YOLO11n Legacy",
    dataset_id: "cityscapes-val",
    dataset_name: "Cityscapes Validation",
    created_at: "28/08/2026 20:00:00",
    updated_at: "28/08/2026 20:10:00",
    status: "ended",
    runs: [
      {
        id: "display-run-b",
        name: "Gaussian Noise cấp 2",
        timestamp: "28/08/2026 20:05:00",
        attack_type: "gaussian_noise",
        attack_name: "Gaussian Noise",
        severity: 2,
        clean_map: 0.71,
        attacked_map: 0.51,
        map_drop_pct: 28.2,
        clean_conf: 0.84,
        attacked_conf: 0.61,
        psnr: "22.00 dB",
        ssim: "0.700",
        inference_ms: 44,
        robustness_score: 71.8,
        clean_bbox_count: 8,
        attacked_bbox_count: 6,
        sample_id: "000001",
        is_combined: false,
        attack_components: [],
        note: "",
        seed: 7,
        run_config_hash: "",
        backend_run_id: "",
      },
    ],
  },
];

const baselineRun = {
  run_id: "backend-run-a",
  status: "COMPLETED",
  progress: 1,
  report: {
    run_id: "backend-run-a",
    simulation_only: false,
    provenance: {
      run_config: {
        task_id: "detection2d",
        model: "yolo11",
        model_family_id: "yolo11",
        checkpoint_id: "base-checkpoint-a",
        model_version_id: "base-checkpoint-a",
        dataset: "kitti_anonymized",
        seed: 42,
        recipe: { name: "depth-fog-3", steps: [{ attack: "depth_fog", severity: 3 }] },
      },
    },
  },
};

afterEach(cleanup);

beforeEach(() => {
  vi.clearAllMocks();
  listSessions.mockResolvedValue(sessions);
  getRun.mockResolvedValue(baselineRun);
  getRunDefenceCandidates.mockResolvedValue([]);
  uploadCheckpoint.mockResolvedValue({ checkpoint_id: "candidate-ready" });
  getCheckpoint.mockResolvedValue({
    checkpoint_id: "candidate-ready",
    status: "READY",
    validation_reason: null,
  });
  createDefenceRun.mockResolvedValue({ run_id: "defence-job-1", status: "QUEUED", progress: 0 });
  createModelComparison.mockResolvedValue({
    comparison_id: "comparison-1",
    eligibility: { status: "NOT_ELIGIBLE", reasons: ["GROUND_TRUTH_HASH_MISSING"] },
    metric_deltas: [],
    recovery: { reason: "NOT_ELIGIBLE" },
    decision: { next_action: "Rerun the locked benchmark." },
  });
});

describe("DefensePage session and run targeting", () => {
  it("derives locked defense context and clears the run when the session changes", async () => {
    const user = userEvent.setup();
    render(<DefensePage />);

    expect(screen.getByText("— / No verified data")).toBeInTheDocument();
    const sessionSelect = await screen.findByRole("combobox", { name: /phiên thử nghiệm/i });
    const runSelect = screen.getByRole("combobox", { name: /attack run/i });
    expect(runSelect).toBeDisabled();
    expect(screen.getByRole("button", { name: /chạy đánh giá đối chiếu/i })).toBeDisabled();

    await user.selectOptions(sessionSelect, "session-a");
    await user.selectOptions(runSelect, "display-run-a");

    expect(await screen.findByDisplayValue("YOLO11s Production")).toHaveAttribute("readOnly");
    expect(screen.getByDisplayValue("KITTI Validation")).toHaveAttribute("readOnly");
    expect(screen.getByDisplayValue("depth_fog")).toHaveAttribute("readOnly");
    expect(screen.queryByText("Phục hồi +68.4%")).not.toBeInTheDocument();

    await user.selectOptions(sessionSelect, "session-b");
    expect(runSelect).toHaveValue("");
    expect(screen.queryByDisplayValue("YOLO11s Production")).not.toBeInTheDocument();

    await user.selectOptions(runSelect, "display-run-b");
    expect(screen.getByText(/không có baseline backend có thể đối chiếu/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /chạy đánh giá đối chiếu/i })).toBeDisabled();
  });

  it("uploads a defense candidate and evaluates it against the backend baseline id", async () => {
    const user = userEvent.setup();
    render(<DefensePage />);

    await user.selectOptions(
      await screen.findByRole("combobox", { name: /phiên thử nghiệm/i }),
      "session-a"
    );
    await user.selectOptions(screen.getByRole("combobox", { name: /attack run/i }), "display-run-a");
    await waitFor(() => {
      expect(screen.getByLabelText(/tải lên checkpoint đã phòng thủ/i)).not.toBeDisabled();
    });

    const file = new File(["checkpoint"], "yolo11s-defended.pt", { type: "application/octet-stream" });
    await user.upload(screen.getByLabelText(/tải lên checkpoint đã phòng thủ/i), file);

    await waitFor(() => expect(screen.getByText(/checkpoint đã xác thực/i)).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: /chạy đánh giá đối chiếu/i }));

    await waitFor(() => expect(screen.getByText("defence-job-1")).toBeInTheDocument());
    expect(createDefenceRun).toHaveBeenCalledWith("backend-run-a", "candidate-ready");
  });

  it("offers only validated candidates from the baseline model family", async () => {
    getRunDefenceCandidates.mockResolvedValue([
      {
        id: "yolo-defended",
        model_name: "YOLO11s Defended",
        model_family_id: "yolo11",
        runnable: true,
      },
      {
        id: "sam-defended",
        model_name: "SAM2 Defended",
        model_family_id: "sam2",
        runnable: true,
      },
    ]);
    const user = userEvent.setup();
    render(<DefensePage />);

    await user.selectOptions(
      await screen.findByRole("combobox", { name: /phiên thử nghiệm/i }),
      "session-a"
    );
    await user.selectOptions(screen.getByRole("combobox", { name: /attack run/i }), "display-run-a");

    const candidateSelect = await screen.findByRole("combobox", {
      name: /checkpoint phòng thủ đã xác thực/i,
    });
    expect(candidateSelect).toHaveTextContent("YOLO11s Defended");
    expect(candidateSelect).not.toHaveTextContent("SAM2 Defended");
  });

  it("renders the canonical backend decision instead of a client-calculated recovery score", async () => {
    createDefenceRun.mockResolvedValue({ run_id: "defence-job-1", status: "COMPLETED", progress: 1 });
    getRunDefenceCandidates.mockResolvedValue([
      { id: "yolo-defended", model_name: "YOLO11s Defended", model_family_id: "yolo11", runnable: true },
    ]);
    const user = userEvent.setup();
    render(<DefensePage />);

    await user.selectOptions(await screen.findByRole("combobox", { name: /phiên thử nghiệm/i }), "session-a");
    await user.selectOptions(screen.getByRole("combobox", { name: /attack run/i }), "display-run-a");
    await user.selectOptions(
      await screen.findByRole("combobox", { name: /checkpoint phòng thủ đã xác thực/i }),
      "yolo-defended"
    );
    await user.click(screen.getByRole("button", { name: /chạy đánh giá đối chiếu/i }));

    expect(await screen.findByText(/không đủ điều kiện kết luận benchmark/i)).toBeInTheDocument();
    expect(screen.getByText("GROUND_TRUTH_HASH_MISSING")).toBeInTheDocument();
    expect(createModelComparison).toHaveBeenCalledWith("backend-run-a", "defence-job-1");
  });
});
