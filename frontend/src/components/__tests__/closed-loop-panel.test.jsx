import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ClosedLoopPanel from "@/components/ClosedLoopPanel";

describe("ClosedLoopPanel", () => {
  it("offers a measured failure backlog and keeps training blocked without a checkpoint", () => {
    const createBacklog = vi.fn();
    render(
      <ClosedLoopPanel
        report={{ cells: [{ attack: "fog", severity: 3, degradation: 0.4 }] }}
        backlog={null}
        trainingBlockedReason="WAITING_FOR_ARTIFACTS"
        onCreateBacklog={createBacklog}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Create retraining backlog" }));
    expect(createBacklog).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Training is waiting for verified model artefacts.")).toBeVisible();
  });
});
