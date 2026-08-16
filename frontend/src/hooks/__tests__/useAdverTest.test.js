import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", () => ({
  getCatalogAttacks: vi.fn().mockRejectedValue(new Error("temporary attack catalog failure")),
  getCatalogModels: vi.fn().mockResolvedValue([{ name: "yolo11" }]),
  getCatalogDatasets: vi.fn().mockResolvedValue([{ id: "synthetic_shapes", name: "synthetic_shapes" }]),
  getModelVersions: vi.fn().mockResolvedValue([{ id: "yolo-b0", task: "detection2d", runnable: true }]),
  getModelFamilies: vi.fn().mockResolvedValue([{ id: "yolo11", display_name: "YOLO11", runnable: true }]),
  getBaseCheckpoints: vi.fn().mockResolvedValue([{ id: "yolo11s-base", task: "detection2d", model_family_id: "yolo11", runnable: true }]),
  getDefenceCheckpoints: vi.fn().mockResolvedValue([]),
  createDefenceRun: vi.fn(),
  createModelComparison: vi.fn(),
  getPerceptionModes: vi.fn().mockResolvedValue([{ id: "detection2d", title: "2D Object Detection", runnable: true }]),
  getRecipePresets: vi.fn().mockResolvedValue([]),
  createRetrainingBacklog: vi.fn().mockResolvedValue({ id: "backlog-1" }),
  addRetrainingBacklogItem: vi.fn().mockResolvedValue({ id: "backlog-1", status: "DRAFT", failure_ids: ["failure-1"] }),
  approveRetrainingBacklog: vi.fn(),
}));

import { useAdverTest } from "@/hooks/useAdverTest";
import { approveRetrainingBacklog } from "@/lib/api";

describe("useAdverTest catalog loading", () => {
  it("keeps successful selectors available when one catalog request fails", async () => {
    const { result } = renderHook(() => useAdverTest());

    await waitFor(() => expect(result.current.state.loading).toBe(false));

    expect(result.current.state.models).toHaveLength(1);
    expect(result.current.state.datasets).toHaveLength(1);
    expect(result.current.state.modelVersions).toHaveLength(1);
    expect(result.current.state.modes).toHaveLength(1);
  });

  it("keeps a newly created retraining backlog in reviewer draft state", async () => {
    const { result } = renderHook(() => useAdverTest());
    await waitFor(() => expect(result.current.state.loading).toBe(false));

    // A reviewer-visible measured failure is the only condition for drafting.
    result.current.actions.createBacklog();
    await waitFor(() => expect(approveRetrainingBacklog).not.toHaveBeenCalled());
  });
});
