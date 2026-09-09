import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import DemoFixtureBadge from "@/components/common/DemoFixtureBadge";
import DemoDefenceResult from "@/components/DemoDefenceResult";

afterEach(cleanup);

describe("DemoFixtureBadge", () => {
  it("labels prepared fixture evidence", () => {
    render(<DemoFixtureBadge visible />);

    expect(screen.getByText("DEMO / SIMULATED")).toBeInTheDocument();
  });

  it("does not label a real run when hidden", () => {
    render(<DemoFixtureBadge visible={false} />);

    expect(screen.queryByText("DEMO / SIMULATED")).toBeNull();
  });

  it("renders prepared defence recovery metrics", () => {
    render(
      <DemoDefenceResult
        job={{
          run_id: "demo-defence-robust-001",
          status: "COMPLETED",
          report: {
            provenance: { demo_fixture: true },
            metrics: { clean: { ap50: 0.91 } },
            cells: [{ ap: 0.84, metrics: { ap50: 0.84 } }],
          },
        }}
      />,
    );

    expect(screen.getByText("Kết quả phòng thủ")).toBeInTheDocument();
    expect(screen.getByText("0.840")).toBeInTheDocument();
    expect(screen.getByText("DEMO / SIMULATED")).toBeInTheDocument();
  });
});
