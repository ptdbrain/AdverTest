import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ProjectAssetPicker from "@/components/ProjectAssetPicker";

const api = vi.hoisted(() => ({
  downloadProjectArtifact: vi.fn(),
  listProjectCheckpoints: vi.fn(),
  listProjectDatasetVersions: vi.fn(),
  registerProjectCheckpoint: vi.fn(),
  registerProjectDataset: vi.fn(),
  uploadProjectArtifact: vi.fn(),
}));

vi.mock("@/lib/api", () => api);

describe("ProjectAssetPicker", () => {
  afterEach(cleanup);

  beforeEach(() => {
    Object.values(api).forEach((fn) => fn.mockReset());
    api.downloadProjectArtifact.mockResolvedValue(undefined);
    api.uploadProjectArtifact.mockResolvedValue({ id: "artifact-new" });
    api.registerProjectCheckpoint.mockResolvedValue({ checkpoint: { id: "checkpoint-new" } });
    api.registerProjectDataset.mockResolvedValue({ id: "dataset-new" });
    api.listProjectDatasetVersions.mockResolvedValue([]);
  });

  it("refreshes after upload and only allows a READY runnable checkpoint to be selected", async () => {
    const waiting = {
      id: "checkpoint-waiting",
      artifact_id: "artifact-waiting",
      display_name: "waiting.pt",
      status: "QUARANTINED",
      runnable: false,
      blocked_reason: "QUARANTINED",
    };
    const ready = {
      id: "checkpoint-new",
      artifact_id: "artifact-new",
      display_name: "new.pt",
      status: "READY",
      runnable: true,
      blocked_reason: null,
    };
    api.listProjectCheckpoints.mockResolvedValueOnce([waiting]).mockResolvedValueOnce([waiting, ready]);
    const onSelect = vi.fn();
    const onAssetsChange = vi.fn();
    const user = userEvent.setup();

    render(
      <ProjectAssetPicker
        projectId="project-1"
        taskId="detection2d"
        kind="model"
        modelFamilyId="yolo11"
        onSelect={onSelect}
        onAssetsChange={onAssetsChange}
      />,
    );

    expect(await screen.findByTitle("QUARANTINED")).toBeDisabled();
    const file = new File(["checkpoint"], "new.pt", { type: "application/octet-stream" });
    await user.upload(screen.getByLabelText(/Tệp mô hình/i), file);

    const readyButton = await screen.findByRole("button", { name: /^new\.pt/i });
    await user.click(readyButton);
    expect(onSelect).toHaveBeenCalledWith(ready);
    expect(onAssetsChange).toHaveBeenLastCalledWith([waiting, ready]);
  });

  it.each([
    ["model", "listProjectCheckpoints"],
    ["dataset", "listProjectDatasetVersions"],
  ])("downloads a registered %s by its project artifact", async (kind, listMethod) => {
    const asset = {
      id: `${kind}-1`,
      artifact_id: `${kind}-artifact`,
      display_name: `${kind}.zip`,
      status: "READY",
      runnable: true,
    };
    api[listMethod].mockResolvedValue([asset]);
    if (kind === "model") api.listProjectDatasetVersions.mockResolvedValue([]);
    else api.listProjectCheckpoints.mockResolvedValue([]);
    const user = userEvent.setup();

    render(<ProjectAssetPicker projectId="project-1" taskId="detection2d" kind={kind} modelFamilyId="yolo11" />);
    await user.click(await screen.findByRole("button", { name: `Tải ${kind}.zip` }));

    await waitFor(() =>
      expect(api.downloadProjectArtifact).toHaveBeenCalledWith("project-1", `${kind}-artifact`),
    );
  });
});
