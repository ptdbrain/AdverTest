import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Workspace from "@/components/Workspace";

describe("Workspace", () => {
  it("renders the five ordered investigation stages for a completed YOLO run", () => {
    render(<Workspace report={{ ap_clean: 0.72, cells: [], metrics: {} }} samples={[]} mode="detection2d" />);

    expect(screen.getByRole("region", { name: "Input and ground truth" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Attacked input" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Clean model prediction" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Attacked model prediction" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Five key metrics" })).toBeVisible();
  });

  it("renders the explicit SAM handoff state instead of metrics", () => {
    render(<Workspace report={null} samples={[]} mode="segmentation" />);

    expect(screen.getByText("Waiting for Person C artefacts")).toBeVisible();
  });
});
