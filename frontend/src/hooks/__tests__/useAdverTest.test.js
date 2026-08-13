import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", () => ({
  getCatalogAttacks: vi.fn().mockRejectedValue(new Error("temporary attack catalog failure")),
  getCatalogModels: vi.fn().mockResolvedValue([{ name: "yolo11" }]),
  getCatalogDatasets: vi.fn().mockResolvedValue([{ id: "synthetic_shapes", name: "synthetic_shapes" }]),
  getModelVersions: vi.fn().mockResolvedValue([{ id: "yolo-b0", task: "detection2d", runnable: true }]),
  getPerceptionModes: vi.fn().mockResolvedValue([{ id: "detection2d", title: "2D Object Detection", runnable: true }]),
  getRecipePresets: vi.fn().mockResolvedValue([]),
}));

import { useAdverTest } from "@/hooks/useAdverTest";

describe("useAdverTest catalog loading", () => {
  it("keeps successful selectors available when one catalog request fails", async () => {
    const { result } = renderHook(() => useAdverTest());

    await waitFor(() => expect(result.current.state.loading).toBe(false));

    expect(result.current.state.models).toHaveLength(1);
    expect(result.current.state.datasets).toHaveLength(1);
    expect(result.current.state.modelVersions).toHaveLength(1);
    expect(result.current.state.modes).toHaveLength(1);
  });
});
