import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";

import { ProjectProvider, useProjectContext } from "@/context/ProjectContext";

const { replace, listProjects } = vi.hoisted(() => ({ replace: vi.fn(), listProjects: vi.fn() }));

vi.mock("next/navigation", () => ({
  usePathname: () => "/experiments/exp-1/results",
  useRouter: () => ({ replace }),
  useSearchParams: () => new URLSearchParams("project_id=forbidden&run_id=run-old"),
}));

vi.mock("@/context/AuthContext", () => ({ useAuth: () => ({ isAuthenticated: true }) }));
vi.mock("@/lib/api", () => ({
  getActiveProjectId: () => null,
  setActiveProjectId: vi.fn(),
  listProjects,
}));

function Harness() {
  const { projectId, scopedHref } = useProjectContext();
  return <output data-testid="project">{projectId ? scopedHref("/benchmark", "run-new") : "loading"}</output>;
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("ProjectProvider", () => {
  it("replaces an unavailable deep-link project and clears its stale run", async () => {
    listProjects.mockResolvedValue([{ id: "project-a", name: "Project A" }]);

    render(<ProjectProvider><Harness /></ProjectProvider>);

    await waitFor(() => expect(screen.getByTestId("project")).toHaveTextContent("project_id=project-a&run_id=run-new"));
    expect(replace).toHaveBeenCalledWith("/experiments/exp-1/results?project_id=project-a");
  });
});
