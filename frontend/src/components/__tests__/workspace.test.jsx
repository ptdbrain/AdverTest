import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Workspace from "@/components/Workspace";

describe("Workspace", () => {
  it("renders the ordered investigation stages and per-object failure evidence for a completed YOLO run", () => {
    const samples = [
      {
        clean_image: "http://localhost:8000/data/clean.jpg",
        attacked_image: "http://localhost:8000/data/attacked.jpg",
        clean_overlay: "http://localhost:8000/data/clean_overlay.jpg",
        attacked_overlay: "http://localhost:8000/data/attacked_overlay.jpg",
        per_object_failures: [
          { id: "obj-1", label: "person", clean_conf: 0.95, attacked_conf: 0.10, degradation: 89.5, status: "FAILED" },
        ],
      },
    ];

    render(<Workspace report={{ ap_clean: 0.72, cells: [], metrics: {} }} samples={samples} mode="detection2d" />);

    expect(screen.getByRole("region", { name: "Input and ground truth" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Attacked input" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Clean model prediction" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Attacked model prediction" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Per-object failure evidence" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Five key metrics" })).toBeVisible();
    expect(screen.getByText("person")).toBeVisible();
    expect(screen.getByText("obj-1")).toBeVisible();
  });

  it("renders the explicit SAM WAITING_FOR_ARTIFACTS handoff state", () => {
    render(<Workspace report={null} samples={[]} mode="segmentation" />);

    expect(screen.getByText("SAM2 Handoff Gated")).toBeVisible();
    expect(screen.getByText("WAITING_FOR_ARTIFACTS")).toBeVisible();
  });
});
