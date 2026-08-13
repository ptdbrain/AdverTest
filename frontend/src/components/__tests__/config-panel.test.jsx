import { fireEvent, render, screen, cleanup } from "@testing-library/react";
import { describe, expect, it, vi, afterEach } from "vitest";
import ConfigPanel from "@/components/ConfigPanel.jsx";

afterEach(() => {
  cleanup();
});


describe("ConfigPanel", () => {
  it("binds the selected checkpoint while keeping a task selectable before its artifacts arrive", () => {
    const actions = {
      setSelectedDataset: vi.fn(),
      setMode: vi.fn(),
      setSelectedModelVersion: vi.fn(),
      toggleAttack: vi.fn(),
      updateAttackSeverity: vi.fn(),
      handleRun: vi.fn(),
    };
    render(
      <ConfigPanel
        datasets={[{ name: "synthetic_shapes", anonymized: true }]}
        attacks={[{ name: "gaussian_noise", group: "A", cost_class: "LIGHT" }]}
        modes={[
          { id: "detection2d", title: "2D Object Detection", runnable: true },
          { id: "segmentation", title: "Instance Segmentation", runnable: false, status: "ready", blocked_reason: "WAITING_FOR_ARTIFACTS" },
        ]}
        modelVersions={[{ id: "yolo-b0", model_name: "yolo11s", task: "detection2d", runnable: true }]}
        recipePresets={[{ preset_id: "weather_robustness", name: "Weather Robustness" }]}
        mode="detection2d"
        selectedModelVersion="yolo-b0"
        selectedDataset="synthetic_shapes"
        selectedAttacks={["gaussian_noise"]}
        recipe={{ steps: [{ position: 0, attack_name: "gaussian_noise", severity: 3 }] }}
        isRunning={false}
        actions={actions}
      />
    );

    fireEvent.change(screen.getByLabelText("Checkpoint"), { target: { value: "yolo-b0" } });
    expect(actions.setSelectedModelVersion).toHaveBeenCalledWith("yolo-b0");
    expect(screen.getByRole("option", { name: "Instance Segmentation" })).not.toBeDisabled();
  });

  it("supports recipe strategy modes and cost preview button", () => {
    const actions = {
      setSelectedDataset: vi.fn(),
      setMode: vi.fn(),
      setSelectedModelVersion: vi.fn(),
      toggleAttack: vi.fn(),
      updateAttackSeverity: vi.fn(),
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
        recipePresets={[{ preset_id: "weather_robustness", name: "Weather Robustness" }]}
        mode="detection2d"
        selectedModelVersion="yolo-b0"
        selectedDataset="synthetic_shapes"
        selectedAttacks={["gaussian_noise"]}
        recipe={{ steps: [{ position: 0, attack_name: "gaussian_noise", severity: 3 }] }}
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
        recipe={{ steps: [] }}
        isRunning={false}
        actions={{}}
      />
    );

    expect(screen.queryByLabelText("Seed")).toBeNull();
    fireEvent.click(screen.getByText(/Show Advanced Settings Drawer/));
    expect(screen.getByLabelText("Seed")).toHaveValue(42);
    expect(screen.getByLabelText("Sample Limit")).toHaveValue(8);
  });

  it("changes only the severity of the selected recipe step", () => {
    const updateAttackSeverity = vi.fn();
    render(<ConfigPanel datasets={[]} attacks={[]} modes={[]} modelVersions={[]} mode="detection2d" selectedDataset="" selectedModelVersion="" selectedAttacks={["brightness", "fgsm"]} recipe={{ steps: [{ position: 0, attack_name: "brightness", severity: 1 }, { position: 1, attack_name: "fgsm", severity: 4 }] }} isRunning={false} actions={{ updateAttackSeverity }} />);

    fireEvent.change(screen.getByLabelText("brightness severity"), { target: { value: "2" } });

    expect(updateAttackSeverity).toHaveBeenCalledWith(0, 2);
    expect(updateAttackSeverity).not.toHaveBeenCalledWith(1, expect.anything());
  });
});
