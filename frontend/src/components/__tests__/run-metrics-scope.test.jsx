import { cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const { listRuns } = vi.hoisted(() => ({
  listRuns: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/context/ProjectContext", () => ({
  useProject: () => ({ activeProjectId: "project-a" }),
}));

vi.mock("@/lib/api", () => ({
  getApiBase: vi.fn(() => "http://localhost:8000"),
  getRunAnalyticsAttacks: vi.fn(),
  getRunAnalyticsClasses: vi.fn(),
  getRunAnalyticsDistance: vi.fn(),
  getRunAnalyticsSummary: vi.fn(),
  listRuns,
}));

vi.mock("@/lib/exportReport", () => ({
  exportReportAsCsv: vi.fn(),
  exportReportAsJson: vi.fn(),
}));

import RunMetricsWorkspace from "@/components/analytics/RunMetricsWorkspace";

describe("RunMetricsWorkspace project scope", () => {
  afterEach(() => {
    cleanup();
    listRuns.mockClear();
  });

  it("loads completed runs for the active project only", async () => {
    render(<RunMetricsWorkspace />);

    await waitFor(() => expect(listRuns).toHaveBeenCalledWith("project-a"));
  });
});
