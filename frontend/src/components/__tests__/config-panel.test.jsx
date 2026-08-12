import { fireEvent, render, screen, cleanup } from "@testing-library/react";
import { describe, expect, it, vi, afterEach } from "vitest";
import ConfigPanel from "@/components/ConfigPanel.jsx";

afterEach(() => {
  cleanup();
});


describe("ConfigPanel", () => {
  it("binds the selected model version and blocks a non-runnable SAM mode", () => {
    const actions = {
      setSelectedDataset: vi.fn(),
      setMode: vi.fn(),
      setSelectedModelVersion: vi.fn(),
      toggleAttack: vi.fn(),
      setSeverity: vi.fn(),
      handleRun: vi.fn(),
    };
    render(
      <ConfigPanel
        datasets={[{ name: "synthetic_shapes", anonymized: true }]}
        attacks={[{ name: "gaussian_noise", group: "A", cost_class: "LIGHT" }]}
        modes={[
          { id: "detection2d", title: "YOLO", runnable: true },
          { id: "segmentation", title: "SAM2", runnable: false, blocked_reason: "WAITING_FOR_ARTIFACTS" },
        ]}
        modelVersions={[{ id: "yolo-b0", model_name: "yolo11s", task: "detection2d", runnable: true }]}
        mode="detection2d"
        selectedModelVersion="yolo-b0"
        selectedDataset="synthetic_shapes"
        selectedAttacks={["gaussian_noise"]}
        severity={3}
        isRunning={false}
        actions={actions}
      />
    );

    fireEvent.change(screen.getByLabelText("Model Version"), { target: { value: "yolo-b0" } });
    expect(actions.setSelectedModelVersion).toHaveBeenCalledWith("yolo-b0");
    fireEvent.change(screen.getByLabelText("Perception Mode"), { target: { value: "segmentation" } });
    expect(actions.setMode).toHaveBeenCalledWith("segmentation");
  });

  it("supports recipe strategy modes and cost preview button", () => {
    const actions = {
      setSelectedDataset: vi.fn(),
      setMode: vi.fn(),
      setSelectedModelVersion: vi.fn(),
      toggleAttack: vi.fn(),
      setSeverity: vi.fn(),
      handleRun: vi.fn(),
      loadPreset: vi.fn(),
      randomizeRecipe: vi.fn(),
      sweepRecipe: vi.fn(),
      previewRecipe: vi.fn().mockResolvedValue({ estimated_seconds: 1.2, total_samples: 8 }),
    };

    render(
      <ConfigPanel
        datasets={[{ name: "synthetic_shapes", anonymized: true }]}
        attacks={[{ name: "gaussian_noise", group: "A", cost_class: "LIGHT" }]}
        modes={[{ id: "detection2d", title: "YOLO", runnable: true }]}
        modelVersions={[{ id: "yolo-b0", model_name: "yolo11s", task: "detection2d", runnable: true }]}
        mode="detection2d"
        selectedModelVersion="yolo-b0"
        selectedDataset="synthetic_shapes"
        selectedAttacks={["gaussian_noise"]}
        severity={3}
        isRunning={false}
        actions={actions}
      />
    );

    // Click PRESET tab
    fireEvent.click(screen.getByText("PRESET"));
    fireEvent.click(screen.getByRole("button", { name: "Weather Robustness" }));
    expect(actions.loadPreset).toHaveBeenCalledWith("weather_robustness");

    // Click RANDOM N tab
    fireEvent.click(screen.getByText("RANDOM N"));
    fireEvent.click(screen.getByRole("button", { name: /Randomize Recipe/ }));
    expect(actions.randomizeRecipe).toHaveBeenCalledWith(3);

    // Click Preview Cost button
    fireEvent.click(screen.getByRole("button", { name: /Preview Resource Cost/ }));
    expect(actions.previewRecipe).toHaveBeenCalled();
  });

  it("toggles the advanced settings drawer", () => {
    render(
      <ConfigPanel
        datasets={[{ name: "synthetic_shapes", anonymized: true }]}
        attacks={[]}
        modes={[{ id: "detection2d", title: "YOLO", runnable: true }]}
        modelVersions={[{ id: "yolo-b0", runnable: true }]}
        mode="detection2d"
        selectedModelVersion="yolo-b0"
        selectedDataset="synthetic_shapes"
        selectedAttacks={[]}
        severity={3}
        isRunning={false}
        actions={{}}
      />
    );

    expect(screen.queryByText(/Seed: 42/)).toBeNull();
    fireEvent.click(screen.getByText(/Show Advanced Settings Drawer/));
    expect(screen.getByText(/Seed: 42/)).toBeVisible();
  });
});

