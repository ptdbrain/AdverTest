import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import QuickUploadDialog from "@/components/dashboard/QuickUploadDialog";

const { getModelFamilies } = vi.hoisted(() => ({ getModelFamilies: vi.fn() }));

vi.mock("@/lib/api", () => ({ getModelFamilies }));

vi.mock("@/components/ProjectAssetPicker", () => ({
  default: ({ kind, onComplete }) => (
    <button type="button" data-testid="picker" onClick={() => onComplete({ id: "artifact-1" })}>
      picker:{kind}
    </button>
  ),
}));

const PROJECT = { id: "p-1", name: "Demo", task_type: "detection2d" };

describe("QuickUploadDialog", () => {
  afterEach(cleanup);

  beforeEach(() => {
    getModelFamilies.mockReset().mockResolvedValue([{ id: "yolo11", display_name: "YOLO11", runnable: true }]);
  });

  it("shows a no-project hint and no picker when there is no active project", () => {
    render(<QuickUploadDialog kind="model" project={null} onClose={() => {}} />);
    expect(screen.getByRole("dialog")).toHaveTextContent("Chưa chọn project");
    expect(screen.queryByTestId("picker")).not.toBeInTheDocument();
  });

  it("loads model families and reports success after registration", async () => {
    const onSuccess = vi.fn();
    const user = userEvent.setup();
    render(<QuickUploadDialog kind="model" project={PROJECT} onClose={() => {}} onSuccess={onSuccess} />);

    await waitFor(() => expect(getModelFamilies).toHaveBeenCalledWith("detection2d"));
    expect(await screen.findByRole("combobox", { name: /Kiến trúc mô hình/i })).toBeVisible();

    await user.click(screen.getByTestId("picker"));

    expect(screen.getByRole("status")).toHaveTextContent("Đã đăng ký checkpoint");
    expect(onSuccess).toHaveBeenCalledTimes(1);
  });

  it("renders the dataset variant without a family selector", () => {
    render(<QuickUploadDialog kind="dataset" project={PROJECT} onClose={() => {}} />);
    expect(screen.getByTestId("picker")).toHaveTextContent("picker:dataset");
    expect(screen.queryByRole("combobox", { name: /Kiến trúc mô hình/i })).not.toBeInTheDocument();
  });
});
