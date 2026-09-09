"use client";

import { AlertCircle, Database, LockKeyhole, RefreshCw, ShieldCheck } from "lucide-react";

import Button from "@/components/common/Button";

const missingValue = "Không có dữ liệu";

function formatMetric(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : missingValue;
}

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)}%` : missingValue;
}

function ProvenanceItem({ label, value, mono = false }) {
  return (
    <div className="min-w-0 border-b border-slate-100 pb-2 last:border-b-0 sm:border-b-0 sm:pb-0">
      <dt className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className={`mt-1 break-words text-xs font-semibold text-slate-800 ${mono ? "font-mono" : ""}`}>
        {value || missingValue}
      </dd>
    </div>
  );
}

export default function DefenseTargetSelector({
  sessions = [],
  selectedSessionId,
  selectedRunId,
  loading = false,
  error = "",
  onSessionChange,
  onRunChange,
  onRetry,
}) {
  const selectedSession = sessions.find((item) => item.id === selectedSessionId) ?? null;
  const selectedRun = selectedSession?.runs?.find((item) => item.id === selectedRunId) ?? null;
  const runs = selectedSession?.runs ?? [];
  const attackName = selectedRun?.attack_components?.length
    ? selectedRun.attack_components.join(" + ")
    : selectedRun?.attack_name || selectedRun?.attack_type;
  const cleanMetric = selectedRun?.clean_miou ?? selectedRun?.clean_map;
  const attackedMetric = selectedRun?.attacked_miou ?? selectedRun?.attacked_map;
  const degradation = selectedRun?.miou_drop_pct ?? selectedRun?.map_drop_pct;

  return (
    <section
      aria-labelledby="defense-target-title"
      className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-xs"
    >
      <div className="flex flex-col gap-3 border-b border-slate-200 bg-slate-50 px-4 py-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-slate-900 text-slate-100">
            <ShieldCheck className="h-4 w-4" aria-hidden="true" />
          </span>
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-blue-700">Bước bắt buộc</p>
            <h2 id="defense-target-title" className="mt-0.5 text-sm font-bold text-slate-900">
              Chọn mục tiêu phòng thủ
            </h2>
            <p className="mt-1 max-w-2xl text-xs leading-5 text-slate-600">
              Chọn phiên và một attack run cụ thể để khóa đúng model, dataset và giao thức đánh giá.
            </p>
          </div>
        </div>
        {selectedRun && (
          <span className="inline-flex w-fit items-center gap-1.5 rounded-md border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-[11px] font-bold text-emerald-800">
            <LockKeyhole className="h-3.5 w-3.5" aria-hidden="true" />
            Đã khóa mục tiêu
          </span>
        )}
      </div>

      <div className="space-y-4 p-4">
        {error && (
          <div
            role="alert"
            className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2.5 text-xs text-red-800"
          >
            <span className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 shrink-0" aria-hidden="true" />
              {error}
            </span>
            <Button variant="secondary" size="sm" icon={RefreshCw} onClick={onRetry}>
              Thử lại
            </Button>
          </div>
        )}

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <label className="text-xs font-semibold text-slate-700">
            Phiên thử nghiệm
            <select
              aria-label="Phiên thử nghiệm"
              className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-xs text-slate-800 outline-none transition-colors focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:cursor-not-allowed disabled:bg-slate-100"
              value={selectedSessionId}
              disabled={loading || Boolean(error)}
              onChange={(event) => onSessionChange(event.target.value)}
            >
              <option value="">{loading ? "Đang tải phiên..." : "Chọn một phiên thử nghiệm"}</option>
              {sessions.map((session) => (
                <option key={session.id} value={session.id}>
                  {session.name} ({session.id})
                </option>
              ))}
            </select>
          </label>

          <label className="text-xs font-semibold text-slate-700">
            Attack run
            <select
              aria-label="Attack run"
              className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-xs text-slate-800 outline-none transition-colors focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:cursor-not-allowed disabled:bg-slate-100"
              value={selectedRunId}
              disabled={!selectedSession || runs.length === 0}
              onChange={(event) => onRunChange(event.target.value)}
            >
              <option value="">
                {!selectedSession
                  ? "Chọn phiên trước"
                  : runs.length === 0
                    ? "Phiên này chưa có attack run"
                    : "Chọn một attack run"}
              </option>
              {runs.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.name || run.attack_name} ({run.id})
                </option>
              ))}
            </select>
          </label>
        </div>

        {!selectedRun ? (
          <div className="flex items-start gap-3 rounded-lg border border-dashed border-slate-300 bg-slate-50 px-4 py-4 text-xs text-slate-600">
            <Database className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" aria-hidden="true" />
            <div>
              <p className="font-semibold text-slate-800">Chọn phiên và attack run để xem evidence phòng thủ.</p>
              <p className="mt-1 leading-5">Hệ thống sẽ không tự điền model hoặc attack khi chưa có run được đo.</p>
            </div>
          </div>
        ) : (
          <dl className="grid grid-cols-1 gap-x-5 gap-y-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 sm:grid-cols-2 xl:grid-cols-4">
            <ProvenanceItem label="Model gốc" value={selectedSession.model_name} />
            <ProvenanceItem label="Dataset" value={selectedSession.dataset_name} />
            <ProvenanceItem label="Attack" value={attackName} />
            <ProvenanceItem
              label="Mức tấn công"
              value={selectedRun.severity == null ? missingValue : `Cấp ${selectedRun.severity}`}
            />
            <ProvenanceItem label="Metric clean" value={formatMetric(cleanMetric)} mono />
            <ProvenanceItem label="Metric attacked" value={formatMetric(attackedMetric)} mono />
            <ProvenanceItem label="Suy giảm" value={formatPercent(degradation)} mono />
            <ProvenanceItem
              label="Seed"
              value={selectedRun.seed == null ? missingValue : String(selectedRun.seed)}
              mono
            />
            <ProvenanceItem label="Backend run ID" value={selectedRun.backend_run_id} mono />
            <ProvenanceItem label="Protocol hash" value={selectedRun.run_config_hash} mono />
          </dl>
        )}
      </div>
    </section>
  );
}
