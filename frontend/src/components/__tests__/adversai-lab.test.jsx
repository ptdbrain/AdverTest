import React from "react";
import { describe, it, expect, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import MetricCard from "@/components/metrics/MetricCard";
import Sidebar from "@/components/layout/Sidebar";
import TopNavigation from "@/components/layout/TopNavigation";

describe("AdversAI Lab Design System & Common Components", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders standard Card with title and subtitle", () => {
    render(<Card title="Tổng quan hệ thống" subtitle="Kiểm thử độ bền vững" />);
    expect(screen.getByText("Tổng quan hệ thống")).toBeDefined();
    expect(screen.getByText("Kiểm thử độ bền vững")).toBeDefined();
  });

  it("renders Badge with various variants", () => {
    render(<Badge variant="success">Hoàn thành</Badge>);
    expect(screen.getByText("Hoàn thành")).toBeDefined();
  });

  it("renders Button with primary variant", () => {
    render(<Button variant="primary">Bắt đầu tấn công</Button>);
    expect(screen.getByText("Bắt đầu tấn công")).toBeDefined();
  });

  it("renders MetricCard with value, trend, and sparkline", () => {
    render(
      <MetricCard
        title="Số thí nghiệm"
        value="128"
        trend="↑ 18%"
        trendLabel="so với tuần trước"
        sparkline={[80, 90, 100, 110, 128]}
      />
    );
    expect(screen.getByText("Số thí nghiệm")).toBeDefined();
    expect(screen.getByText("128")).toBeDefined();
    expect(screen.getByText("↑ 18%")).toBeDefined();
  });

  it("renders Sidebar brand and menu links", () => {
    render(<Sidebar />);
    expect(screen.getByText("AdversAI Lab")).toBeDefined();
    expect(screen.getAllByText("Tổng quan hệ thống").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Cấu hình bài toán").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Phòng thủ").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Quản trị").length).toBeGreaterThan(0);
  });

  it("renders compact TopNavigation controls", () => {
    render(<TopNavigation />);
    expect(screen.getByRole("button", { name: "Toggle Sidebar" })).toBeDefined();
    expect(screen.getByTitle("Thông báo hệ thống")).toBeDefined();
    expect(screen.getByTitle("Cài đặt & Tích hợp")).toBeDefined();
    expect(screen.queryByText("Metrics & Benchmark")).toBeNull();
  });
});
