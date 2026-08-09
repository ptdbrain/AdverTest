import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ConfigPanel from "@/components/ConfigPanel.jsx";

describe("ConfigPanel", () => {
  it("binds the selected model version and blocks a non-runnable SAM mode", () => {
    const actions = {
      setSelectedDataset: vi.fn(), setMode: vi.fn(), setSelectedModelVersion: vi.fn(),
      toggleAttack: vi.fn(), setSeverity: vi.fn(), handleRun: vi.fn(),
    };
    render(<ConfigPanel datasets={[{ name: "synthetic_shapes", anonymized: true }]} attacks={[]}
      modes={[{ id: "detection2d", title: "YOLO", runnable: true }, { id: "segmentation", title: "SAM2", runnable: false, blocked_reason: "WAITING_FOR_ARTIFACTS" }]}
      modelVersions={[{ id: "yolo-b0", model_name: "yolo11s", task: "detection2d", runnable: true }]}
      mode="detection2d" selectedModelVersion="yolo-b0" selectedDataset="synthetic_shapes"
      selectedAttacks={["gaussian_noise"]} severity={3} isRunning={false} actions={actions} />);

    fireEvent.change(screen.getByLabelText("Model version"), { target: { value: "yolo-b0" } });
    expect(actions.setSelectedModelVersion).toHaveBeenCalledWith("yolo-b0");
    fireEvent.change(screen.getByLabelText("Perception mode"), { target: { value: "segmentation" } });
    expect(actions.setMode).toHaveBeenCalledWith("segmentation");
  });
});
