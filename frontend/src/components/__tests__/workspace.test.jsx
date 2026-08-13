import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import Workspace from "@/components/Workspace";

afterEach(cleanup);

describe("Workspace", () => {
  it("renders the ordered investigation stages and per-object failure evidence for a completed YOLO run", () => {
    const samples = [
      {
        artifacts: {
          clean_input_url: "http://localhost:8000/data/clean.jpg",
          attacked_input_url: "http://localhost:8000/data/attacked.jpg",
          clean_prediction_url: "http://localhost:8000/data/clean_overlay.jpg",
          attacked_prediction_url: "http://localhost:8000/data/attacked_overlay.jpg",
        },
        per_object_failures: [
          { id: "obj-1", label: "person", clean_conf: 0.95, attacked_conf: 0.10, degradation: 89.5, status: "FAILED" },
        ],
        ground_truth: {
          type: "boxes",
          image_width: 640,
          image_height: 480,
          objects: [{ object_id: "obj-1", label: "person", xyxy: [10, 20, 100, 220] }],
        },
      },
    ];

    render(<Workspace report={{ ap_clean: 0.72, cells: [], metrics: {} }} samples={samples} mode="detection2d" />);

    expect(screen.getByRole("region", { name: "Input and ground truth" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Attacked input" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Clean model prediction" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Attacked model prediction" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Per-object failure evidence" })).toBeVisible();
    expect(screen.getByRole("region", { name: "Five key metrics" })).toBeVisible();
    expect(screen.getAllByText("person")).toHaveLength(2);
    expect(screen.getByText("obj-1")).toBeVisible();
    expect(screen.getByLabelText("Ground truth overlay")).toBeVisible();
  });

  it("does not replace a task workspace with a model-family waiting screen", () => {
    render(<Workspace report={null} samples={[]} mode="segmentation" />);

    expect(screen.queryByText("SAM2 Handoff Gated")).toBeNull();
    expect(screen.getByText("Source input unavailable")).toBeVisible();
  });

  it("does not fabricate object evidence or substitute an input for a prediction", () => {
    render(
      <Workspace
        report={{ cells: [] }}
        samples={[{ clean_image: "/data/clean.jpg", attacked_image: "/data/attacked.jpg" }]}
        mode="detection2d"
      />,
    );

    expect(screen.getByText(/No per-object evidence was produced/)).toBeVisible();
    expect(screen.getByText("Clean prediction unavailable")).toBeVisible();
    expect(screen.getByText("Attacked prediction unavailable")).toBeVisible();
  });

  it("uses the selected final recipe result and nested robustness score", () => {
    render(<Workspace report={{ ap_clean: 0.72, cells: [{ ap: 0.45, degradation_ratio: 0.375, metrics: { objects_broken: 7 } }], metrics: { robustness: { robust_score_normalized: 63.4 } } }} samples={[]} mode="detection2d" />);

    expect(screen.getAllByText("0.720")).toHaveLength(1);
    expect(screen.getByText("0.450")).toBeVisible();
    expect(screen.getByText("37.5%")).toBeVisible();
    expect(screen.getByText("7")).toBeVisible();
    expect(screen.getByText("63.400")).toBeVisible();
  });

  it("lets the reviewer choose an evidence sample and keeps its report cell paired by attack and severity", () => {
    const samples = [
      { sample_id: "first", attack: "fog", severity: 1, clean_image: "/data/one.jpg" },
      { sample_id: "second", attack: "fog", severity: 3, clean_image: "/data/two.jpg" },
    ];
    render(
      <Workspace
        report={{ cells: [{ attack: "fog", severity: 1 }, { attack: "fog", severity: 3 }] }}
        samples={samples}
        mode="detection2d"
      />,
    );

    fireEvent.change(screen.getByLabelText("Evidence sample"), { target: { value: "1" } });
    expect(screen.getByText(/fog, severity 3/)).toBeVisible();
  });
});
