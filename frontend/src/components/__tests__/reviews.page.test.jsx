import { act, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ReviewPage from "@/app/reviews/page.jsx";
import { AuthProvider } from "@/context/AuthContext";
import { ProjectProvider } from "@/context/ProjectContext";

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
      degradation_percent: 35.5,
      status: "PENDING",
      flagged_by: "system_auto",
      created_at: "2026-08-12T00:00:00Z",
    },
  ]),
  resolveReview: vi.fn().mockResolvedValue({ review_id: "REV-001", status: "RESOLVED" }),
  getRunSamples: vi.fn().mockResolvedValue([]),
  getFailureClusters: vi
    .fn()
    .mockResolvedValue([
      { cluster_id: "cluster-1", name: "Weather Failures", member_count: 1, representative_review_id: "REV-001" },
    ]),
  createFailureCluster: vi.fn().mockResolvedValue({ id: "cluster-2", name: "New Cluster" }),
  autoGroupFailureClusters: vi
    .fn()
    .mockResolvedValue([
      { cluster_id: "cluster-1", name: "Weather Failures", member_count: 1, representative_review_id: "REV-001" },
    ]),
  assessReviewRisk: vi.fn().mockResolvedValue({
    risk_level: "HIGH",
    risk_category: "weather_pedestrian",
    recommended_decision: "REQUEST_RETRAIN",
    recommended_decision_label: "Yêu cầu tôi luyện đối kháng",
    justification: "Mất nhận diện trong sương mù cấp 3",
    downstream_actions: ["Tạo Retraining Backlog"],
    confidence: 0.85,
    contributing_factors: ["Suy giảm 35.5%"],
  }),
  getRiskSessionSummary: vi.fn().mockResolvedValue({
    total_reviews: 1,
    critical_count: 0,
    high_count: 1,
    medium_count: 0,
    low_count: 0,
    auto_pass_count: 0,
    dominant_risk_level: "HIGH",
  }),
  getRiskRubric: vi.fn().mockResolvedValue([]),
  createDefenseProfile: vi.fn().mockResolvedValue({ id: "dp-1", name: "Spatial Filter" }),
  getGoogleAuthConfig: vi.fn().mockResolvedValue({ client_id: "", configured: false, demo_profiles: [] }),
  getAuthMe: vi.fn().mockResolvedValue({
    id: "user-1",
    email: "reviewer@example.test",
    display_name: "Reviewer",
    role: "RESEARCHER",
    status: "ACTIVE",
  }),
  getApiBase: vi.fn().mockReturnValue("http://127.0.0.1:8000"),
  getSampleReviews: vi.fn().mockResolvedValue([]),
  upsertSampleReviews: vi.fn().mockResolvedValue([]),
  createAdversarialDataset: vi.fn().mockResolvedValue(null),
  getActiveProjectId: vi.fn().mockReturnValue("project-1"),
  listProjects: vi.fn().mockResolvedValue([{ id: "project-1", name: "Test project" }]),
}));

describe("ReviewPage", () => {
  it("renders pending review queue and 5-decision options without placeholders", async () => {
    await act(async () => {
      render(
        <AuthProvider>
          <ProjectProvider>
            <ReviewPage />
          </ProjectProvider>
        </AuthProvider>,
      );
    });

    await waitFor(
      () => {
        expect(screen.getAllByText(/REV-001/i)[0]).toBeInTheDocument();
      },
      { timeout: 3000 },
    );

    expect(screen.getAllByText(/FOG/i)[0]).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Chặn triển khai/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Yêu cầu retrain/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Giới hạn ODD/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Sửa nhãn dữ liệu/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Chấp nhận rủi ro/i })).toBeInTheDocument();
  });
});
