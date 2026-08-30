"use client";

import RunMetricsWorkspace from "@/components/analytics/RunMetricsWorkspace";
import PageHeader from "@/components/layout/PageHeader";

export default function AnalysisPage() {
  return (
    <div className="space-y-5">
      <PageHeader
        title="Phân tích & Báo cáo"
        subtitle="Ưu tiên failure evidence, provenance và hành động tái kiểm chứng; không sinh kết luận từ số mock."
        breadcrumb={[{ label: "Trang chủ", href: "/dashboard" }, { label: "Phân tích & Báo cáo" }]}
      />
      <RunMetricsWorkspace mode="analysis" />
    </div>
  );
}
