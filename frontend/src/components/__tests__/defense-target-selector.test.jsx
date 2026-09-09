import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import DefenseTargetSelector from "@/components/DefenseTargetSelector";

afterEach(cleanup);

const sessions = [
  {
    id: "session-a",
    name: "Phiên đánh giá thời tiết",
    description: "Đánh giá model production trên KITTI",
    task_id: "detection2d",
    task_name: "Object Detection 2D",
    model_id: "yolo11s-prod",
    model_name: "YOLO11s Production",
    dataset_id: "kitti-val",
    dataset_name: "KITTI Validation",
    created_at: "29/08/2026 20:00:00",
    updated_at: "29/08/2026 20:10:00",
    status: "active",
    runs: [
      {
        id: "run-visible-1",
        name: "Depth Fog cấp 3",
        timestamp: "29/08/2026 20:05:00",
        attack_type: "depth_fog",
        attack_name: "Depth Fog",
        severity: 3,
        clean_map: 0.82,
        attacked_map: 0.45,
        map_drop_pct: 45.1,
        clean_conf: 0.92,
        attacked_conf: 0.68,
        psnr: "24.21 dB",
        ssim: "0.781",
        inference_ms: 56.9,
        robustness_score: 54.9,
        clean_bbox_count: 10,
        attacked_bbox_count: 5,
        sample_id: "000000",
        is_combined: false,
        attack_components: [],
        note: "",
        seed: 42,
        run_config_hash: "protocol-hash-a",
        backend_run_id: "backend-run-a",
      },
    ],
  },
];

function SelectorHarness() {
  const [sessionId, setSessionId] = useState("");
  const [runId, setRunId] = useState("");

  return (
    <DefenseTargetSelector
      sessions={sessions}
      selectedSessionId={sessionId}
      selectedRunId={runId}
      loading={false}
      error=""
      onSessionChange={(nextSessionId) => {
        setSessionId(nextSessionId);
        setRunId("");
      }}
      onRunChange={setRunId}
      onRetry={vi.fn()}
    />
  );
}

describe("DefenseTargetSelector", () => {
  it("requires a session before an attack run can be selected", () => {
    render(<SelectorHarness />);

    expect(screen.getByRole("combobox", { name: /phiên thử nghiệm/i })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: /attack run/i })).toBeDisabled();
    expect(screen.getByText(/chọn phiên và attack run/i)).toBeInTheDocument();
  });

  it("shows measured model, dataset, attack and protocol provenance for the selected run", async () => {
    const user = userEvent.setup();
    render(<SelectorHarness />);

    await user.selectOptions(screen.getByRole("combobox", { name: /phiên thử nghiệm/i }), "session-a");
    await user.selectOptions(screen.getByRole("combobox", { name: /attack run/i }), "run-visible-1");

    expect(screen.getByText("YOLO11s Production")).toBeInTheDocument();
    expect(screen.getByText("KITTI Validation")).toBeInTheDocument();
    expect(screen.getByText("Depth Fog")).toBeInTheDocument();
    expect(screen.getByText("Cấp 3")).toBeInTheDocument();
    expect(screen.getByText("45.1%")).toBeInTheDocument();
    expect(screen.getByText("backend-run-a")).toBeInTheDocument();
    expect(screen.getByText("protocol-hash-a")).toBeInTheDocument();
  });
});
