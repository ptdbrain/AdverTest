import { render, screen, waitFor, act } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ReviewPage from "@/app/reviews/page.jsx";

// Mock API functions
vi.mock("@/lib/api", () => ({
  getReviews: vi.fn().mockResolvedValue([
    {
      review_id: "REV-001",
      run_id: "run-100",
      attack: "fog",
      severity: 3,
      dataset: "synthetic_shapes",
      model: "yolo11s",
      degradation: 35.5,
      status: "PENDING",
      flagged_by: "system_auto",
      created_at: "2026-08-12T00:00:00Z",
    },
  ]),
  resolveReview: vi.fn().mockResolvedValue({ review_id: "REV-001", status: "RESOLVED" }),
  getRunSamples: vi.fn().mockResolvedValue([]),
  getFailureClusters: vi.fn().mockResolvedValue([{ id: "cluster-1", name: "Weather Failures" }]),
  createFailureCluster: vi.fn().mockResolvedValue({ id: "cluster-2", name: "New Cluster" }),
  createDefenseProfile: vi.fn().mockResolvedValue({ id: "dp-1", name: "Spatial Filter" }),
}));

describe("ReviewPage", () => {
  it("renders pending review queue and decision options without placeholders", async () => {
    await act(async () => {
      render(<ReviewPage />);
    });

    await waitFor(
      () => {
        expect(screen.getAllByText("REV-001")[0]).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    expect(screen.getByText(/FOG/i)).toBeInTheDocument();
    expect(screen.getByText("Accept Risk")).toBeInTheDocument();
    expect(screen.getByText("Request Retrain")).toBeInTheDocument();
    expect(screen.getByText("Reject Sample")).toBeInTheDocument();
  });
});


