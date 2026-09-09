import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ConfigureAttackPage, { normalizeAttackQueue, resolveBackendDataset, resolveBackendModel } from "../page";

const {
  push,
  createRun,
  getCatalogAttacks,
  getModelVersions,
  getCatalogDatasets,
  preflightRun,
  getRun,
  getRunReport,
  getRunSamples,
  addRunToSession,
} = vi.hoisted(() => ({
  push: vi.fn(),
  createRun: vi.fn(),
  getCatalogAttacks: vi.fn(),
  getModelVersions: vi.fn(),
  getCatalogDatasets: vi.fn(),
  preflightRun: vi.fn(),
  getRun: vi.fn(),
  getRunReport: vi.fn(),
  getRunSamples: vi.fn(),
  addRunToSession: vi.fn().mockResolvedValue({}),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "EXP-REAL-001" }),
  useRouter: () => ({ push }),
  useSearchParams: () => ({ get: () => null }),
}));

vi.mock("@/lib/api", () => ({
  createRun,
  getCatalogAttacks,
  getCatalogDatasets,
  getModelVersions,
  getRun,
  getRunReport,
  getRunSamples,
  preflightRun,
  addRunToSession,
}));

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.clearAllMocks();
});

describe("ConfigureAttackPage real run handoff", () => {
  it("normalizes legacy UI attack ids to executable backend ids", () => {
    expect(
      normalizeAttackQueue([
        { id: "low_light", name: "Low Light / Night" },
        { id: "lidar_jitter", name: "Point Jitter" },
      ]),
    ).toMatchObject([{ id: "brightness" }, { id: "lidar_xyz_noise" }]);
  });

  it("never substitutes a global asset for an explicitly selected project asset", () => {
    const context = {
      selectedTask: "detection2d",
      selectedModelId: "project-checkpoint",
      selectedModelSource: "project",
      selectedDatasetId: "project-dataset",
      selectedDatasetSource: "project",
    };
    const globalModels = [{ id: "global-model", runnable: true, checkpoint_role: "base", task: "detection2d" }];
    const globalDatasets = [{ name: "kitti", anonymized: true }];

    expect(resolveBackendModel(context, globalModels, [])).toBeNull();
    expect(resolveBackendDataset(context, globalDatasets, [])).toBeNull();
  });

  it("persists backend report and samples before opening visual results", async () => {
    getModelVersions.mockResolvedValueOnce([
      {
        id: "yolo11s-kitti-clean-b0",
        model_name: "yolo11s",
        checkpoint_path: "data/checkpoints/uploaded/yolo11s-clean-b0_best.pt",
        model_family_id: "yolo11",
        task: "detection2d",
        runnable: true,
        checkpoint_role: "defence_baseline",
      },
      {
        id: "yolo11s-base",
        model_name: "yolo11s",
        checkpoint_path: "data/checkpoints/surrogates/yolo11s.pt",
        model_family_id: "yolo11",
        task: "detection2d",
        runnable: true,
        checkpoint_role: "base",
      },
    ]);
    getCatalogDatasets.mockResolvedValueOnce([
      {
        name: "kitti",
        title: "KITTI thật từ backend",
        task_id: "detection2d",
      },
    ]);
    getCatalogAttacks.mockResolvedValue([
      { name: "depth_fog", available: true },
      { name: "motion_blur", available: true },
    ]);
    preflightRun.mockResolvedValueOnce({ fatal_errors: [] });
    createRun.mockResolvedValueOnce({ run_id: "run-real-001", status: "QUEUED" });
    getRun.mockResolvedValue({ run_id: "run-real-001", status: "COMPLETED", progress: 1 });
    const sample = {
      sample_id: "000001",
      attack: "depth_fog",
      severity: 3,
      clean_prediction: { boxes: [{ xyxy: [1, 2, 10, 20], label: "Car", score: 0.8 }] },
      attacked_prediction: { boxes: [{ xyxy: [2, 3, 8, 12], label: "Car", score: 0.4 }] },
      artifacts: {
        clean_input_url: "/data/runs/run-real-001/clean.png",
        attacked_input_url: "/data/runs/run-real-001/attacked.png",
      },
      ground_truth: { image_width: 100, image_height: 50, objects: [] },
    };
    const report = {
      run_id: "run-real-001",
      model: "yolo11s",
      model_version: "yolo11s-base",
      dataset: "kitti",
      n_samples: 1,
      ap_clean: 0.8,
      sample_results: [sample],
      cells: [],
      metrics: {},
    };
    getRunReport.mockResolvedValueOnce(report);
    getRunSamples.mockResolvedValueOnce([sample]);

    render(<ConfigureAttackPage />);
    await waitFor(() => expect(getCatalogAttacks).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: /Bắt đầu chạy suy luận/i }));

    await waitFor(() => expect(createRun).toHaveBeenCalledTimes(1));
    expect(getCatalogAttacks).toHaveBeenCalledWith(
      expect.objectContaining({
        task_id: "detection2d",
        model_family_id: "yolo11",
        checkpoint_id: "yolo11s-kitti-clean-b0",
        dataset: "kitti",
      }),
    );
    expect(createRun.mock.calls[0][0]).toMatchObject({
      checkpoint_id: "yolo11s-kitti-clean-b0",
      model_family_id: "yolo11",
      task_id: "detection2d",
      dataset: "kitti",
      recipe: {
        steps: [
          expect.objectContaining({ attack_name: "depth_fog", position: 0 }),
          expect.objectContaining({ attack_name: "motion_blur", position: 1 }),
        ],
      },
    });
    await waitFor(() => expect(push).toHaveBeenCalledWith("/experiments/EXP-REAL-001/results"));
    const saved = JSON.parse(localStorage.getItem("advertest_last_session"));
    expect(saved.runId).toBe("run-real-001");
    expect(saved.report).toEqual(report);
    expect(saved.samples).toEqual([sample]);
    expect(saved.recipe).toBeDefined();
  });

  it("submits individual attacks config when Tách riêng lẻ mode is selected", async () => {
    getModelVersions.mockResolvedValueOnce([
      {
        id: "yolo11s-base",
        model_name: "yolo11s",
        checkpoint_path: "data/checkpoints/surrogates/yolo11s.pt",
        model_family_id: "yolo11",
        task: "detection2d",
        runnable: true,
        checkpoint_role: "base",
      },
    ]);
    getCatalogDatasets.mockResolvedValueOnce([
      {
        name: "kitti",
        title: "KITTI",
        task_id: "detection2d",
      },
    ]);
    getCatalogAttacks.mockResolvedValue([
      { name: "depth_fog", available: true },
      { name: "motion_blur", available: true },
    ]);
    preflightRun.mockResolvedValueOnce({ fatal_errors: [] });
    createRun.mockResolvedValueOnce({ run_id: "run-individual-001", status: "QUEUED" });
    getRun.mockResolvedValue({ run_id: "run-individual-001", status: "COMPLETED", progress: 1 });
    const sample = { sample_id: "000001", clean_prediction: { boxes: [] }, attacked_prediction: { boxes: [] } };
    getRunReport.mockResolvedValueOnce({ run_id: "run-individual-001", sample_results: [sample] });
    getRunSamples.mockResolvedValueOnce([sample]);

    render(<ConfigureAttackPage />);
    await waitFor(() => expect(getCatalogAttacks).toHaveBeenCalledTimes(1));

    // Switch to Individual Mode
    fireEvent.click(screen.getByTestId("mode-individual-btn"));
    fireEvent.click(screen.getByRole("button", { name: /Bắt đầu chạy suy luận/i }));

    await waitFor(() => expect(createRun).toHaveBeenCalledTimes(1));
    expect(createRun.mock.calls[0][0]).toMatchObject({
      checkpoint_id: "yolo11s-base",
      attacks: ["depth_fog", "motion_blur"],
    });
    expect(createRun.mock.calls[0][0].recipe).toBeUndefined();
  });
});
