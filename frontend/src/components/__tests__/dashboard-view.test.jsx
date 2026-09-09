import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import DashboardView from "@/components/DashboardView";

vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

vi.mock("@/context/ProjectContext", () => ({
  useProject: () => ({
    activeProject: null,
    activeProjectId: null,
    projects: [],
    isLoadingProjects: false,
  }),
}));

const { getCatalogAttacks, getCatalogDatasets, getModelVersions, listRuns } = vi.hoisted(() => ({
  getCatalogAttacks: vi.fn().mockResolvedValue([]),
  getCatalogDatasets: vi.fn().mockResolvedValue([]),
  getModelVersions: vi.fn().mockResolvedValue([]),
  listRuns: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  getCatalogAttacks,
  getCatalogDatasets,
  getModelVersions,
  listRuns,
}));

describe("DashboardView", () => {
  afterEach(cleanup);

  beforeEach(() => {
    getCatalogAttacks.mockResolvedValue([]);
    getCatalogDatasets.mockResolvedValue([]);
    getModelVersions.mockResolvedValue([]);
    listRuns.mockResolvedValue([]);
  });

  it("shows the human model name as the primary label and keeps the id as metadata", async () => {
    getModelVersions.mockResolvedValueOnce([
      {
        id: "checkpoint-opaque-id",
        model_name: "YOLO11 Small",
        model_family_id: "yolo11",
        task: "detection2d",
        runnable: true,
      },
    ]);

    render(<DashboardView />);

    expect(await screen.findByText("YOLO11 Small", { selector: "p" })).toBeVisible();
    expect(screen.getByText("ID: checkpoint-opaque-id")).toBeVisible();
  });

  it("does not render the removed workflow architecture and results overview cards", async () => {
    render(<DashboardView />);
    await waitFor(() => expect(getModelVersions).toHaveBeenCalled());

    expect(screen.queryAllByText("Quy trình đánh giá & phòng thủ AI đối kháng")).toHaveLength(0);
    expect(screen.queryAllByText("Kết quả đã xác minh")).toHaveLength(0);
  });

  it("never links to the default demo experiment id", async () => {
    render(<DashboardView />);
    await waitFor(() => expect(getModelVersions).toHaveBeenCalled());

    expect(screen.getAllByRole("link").map((link) => link.getAttribute("href"))).not.toContain(
      "/experiments/EXP-2025-0512-001/attack",
    );
  });
});
