import React from "react";

import DemoFixtureBadge from "@/components/common/DemoFixtureBadge";

function metric(report, key) {
  return Number(report?.metrics?.clean?.[key] ?? report?.ap_clean ?? 0);
}

export default function DemoDefenceResult({ job }) {
  const report = job?.report;
  if (!report) return null;
  const clean = metric(report, report.metrics?.clean?.miou != null ? "miou" : "ap50");
  const cell = report.cells?.at(-1) || {};
  const attacked = Number(cell.metrics?.miou ?? cell.metrics?.ap50 ?? cell.ap ?? 0);
  const recovery = clean > 0 ? Math.max(0, Math.min(100, (attacked / clean) * 100)) : 0;
  const isSegmentation = report.metrics?.clean?.miou != null;
  const metricLabel = isSegmentation ? "mIoU sau defence" : "mAP@0.5 sau defence";

  return (
    <section className="mt-4 rounded-xl border border-emerald-200 bg-gradient-to-br from-emerald-50 to-white p-4" aria-label="Kết quả phòng thủ">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-bold text-emerald-950">Kết quả phòng thủ</h3>
          <p className="mt-0.5 text-[11px] text-emerald-800">Đã chuẩn bị sẵn trên cùng locked protocol</p>
        </div>
        <DemoFixtureBadge visible={report.provenance?.demo_fixture === true} />
      </div>
      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-3">
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <span className="text-[10px] font-semibold text-slate-500">Clean baseline</span>
          <strong className="mt-1 block font-mono text-lg text-slate-900">{clean.toFixed(3)}</strong>
        </div>
        <div className="rounded-lg border border-emerald-200 bg-emerald-100/70 p-3">
          <span className="text-[10px] font-semibold text-emerald-800">{metricLabel}</span>
          <strong className="mt-1 block font-mono text-lg text-emerald-900">{attacked.toFixed(3)}</strong>
        </div>
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-3">
          <span className="text-[10px] font-semibold text-blue-800">Recovery score</span>
          <strong className="mt-1 block font-mono text-lg text-blue-900">{recovery.toFixed(1)}%</strong>
        </div>
      </div>
    </section>
  );
}
