"use client";

import RunMetricsWorkspace from "@/components/analytics/RunMetricsWorkspace";
import PageHeader from "@/components/layout/PageHeader";

export default function BenchmarkPage() {
  return (
    <div className="space-y-5">
      <PageHeader
        title="Metrics & Benchmark"
        subtitle="Đọc metric theo đúng bài toán từ completed run thật; mọi số liệu thiếu được giữ là No data."
        breadcrumb={[{ label: "Trang chủ", href: "/dashboard" }, { label: "Metrics & Benchmark" }]}
      />
      <RunMetricsWorkspace mode="benchmark" />
    </div>
  );
}
