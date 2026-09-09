"use client";

import { Database, RefreshCw, TrendingDown, TrendingUp } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import ReportView from "@/components/ReportView";
import Button from "@/components/common/Button";
import Card from "@/components/common/Card";
import MetricCard from "@/components/metrics/MetricCard";
import TrendLineChart from "@/components/metrics/TrendLineChart";
import { useProject } from "@/context/ProjectContext";
import { getProjectAnalytics, getProjectRunsComparison, getRunReport } from "@/lib/api";
import { formatNumber } from "@/lib/reportMetrics";

function degTokens(degradation) {
  if (degradation == null) return { bg: "transparent", text: "var(--text-muted, #94a3b8)" };
  const d = Math.max(0, Math.min(100, degradation));
  if (d < 5) return { bg: "rgba(104,211,145,0.12)", text: "#3f9142" };
  if (d < 15) return { bg: "rgba(104,211,145,0.08)", text: "#4f8f4f" };
  if (d < 30) return { bg: "rgba(246,173,85,0.14)", text: "#b45309" };
  if (d < 50) return { bg: "rgba(246,173,85,0.22)", text: "#9a3412" };
  return { bg: "rgba(252,129,129,0.16)", text: "#b91c1c" };
}

function shortRunLabel(run) {
  const label = run?.name || run?.run_id || "";
  return label.length > 22 ? `${label.slice(0, 21)}…` : label;
}

export default function ProjectAnalyticsPanel() {
  const { activeProject, activeProjectId } = useProject();
  const [analytics, setAnalytics] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [selectedReport, setSelectedReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadAnalytics = useCallback(async () => {
    if (!activeProjectId) {
      setAnalytics(null);
      setComparison(null);
      return;
    }
    setLoading(true);
    setError("");
    setSelectedReport(null);
    setSelectedRunId("");
    try {
      const data = await getProjectAnalytics(activeProjectId);
      setAnalytics(data);
      const runIds = (data?.runs || []).map((run) => run.run_id);
      setComparison(
        runIds.length >= 2 ? await getProjectRunsComparison(activeProjectId, runIds).catch(() => null) : null,
      );
    } catch (requestError) {
      setAnalytics(null);
      setError(requestError.message || "Không tải được analytics của project.");
    } finally {
      setLoading(false);
    }
  }, [activeProjectId]);

  useEffect(() => {
    loadAnalytics();
  }, [loadAnalytics]);

  useEffect(() => {
    if (!selectedRunId) {
      setSelectedReport(null);
      return;
    }
    let cancelled = false;
    getRunReport(selectedRunId, activeProjectId)
      .then((report) => {
        if (!cancelled) setSelectedReport(report);
      })
      .catch(() => {
        if (!cancelled) setSelectedReport(null);
      });
    return () => {
      cancelled = true;
    };
  }, [activeProjectId, selectedRunId]);

  if (!activeProjectId) return null;
  if (loading)
    return (
      <div role="status" className="rounded-xl border border-slate-200 bg-white p-8 text-sm text-slate-600">
        Đang tổng hợp analytics cho project “{activeProject?.name}”…
      </div>
    );
  if (error)
    return (
      <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900">
        {error}
      </div>
    );
  const runs = analytics?.runs || [];
  if (!runs.length)
    return (
      <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 px-6 py-10 text-center">
        <Database className="mx-auto h-8 w-8 text-slate-400" aria-hidden="true" />
        <h3 className="mt-3 text-base font-semibold text-slate-900">Project chưa có run hoàn tất</h3>
        <p className="mx-auto mt-1 max-w-lg text-sm text-slate-600">
          Lưu kết quả benchmark vào session của project (bước 8 của workflow) để thấy xu hướng robustness và
          vulnerability tổng hợp tại đây.
        </p>
      </div>
    );

  const overall = analytics.overall || {};
  const trend = analytics.trend || {};
  const worst = analytics.worst_attack_overall;
  const sensitive = analytics.most_sensitive_severity;
  const vulnerable = analytics.most_vulnerable_class;
  const TrendIcon = trend.direction === "declining" ? TrendingDown : TrendingUp;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            Phân tích project: {activeProject?.name}
            <span className="ml-2 rounded-md border border-slate-200 bg-slate-50 px-2 py-0.5 font-mono text-xs text-slate-500">
              {activeProject?.task_type}
            </span>
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Tổng hợp {runs.length} run đã lưu — chỉ tính run có bằng chứng đo được; ô thiếu giữ nguyên No data.
          </p>
        </div>
        <Button variant="secondary" icon={RefreshCw} className="min-h-11" onClick={loadAnalytics}>
          Làm mới
        </Button>
      </div>

      <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label="Project overall">
        <MetricCard
          title="Robust score trung bình"
          value={overall.mean_robust_score == null ? "—" : `${overall.mean_robust_score}/100`}
          trend={
            trend.delta_first_last == null ? undefined : `${trend.delta_first_last > 0 ? "+" : ""}${trend.delta_first_last}`
          }
          trendLabel="first → last run"
          trendType={trend.direction === "declining" ? "down" : "up"}
          color="blue"
          sparkline={trend.robust_scores}
        />
        <MetricCard
          title="Suy giảm trung bình"
          value={formatNumber(overall.mean_degradation_percent, 1, "%")}
          note={`Trên ${overall.measured_run_count ?? 0} run có đo được`}
          color="red"
        />
        <MetricCard
          title="Số run / số cell"
          value={`${overall.run_count ?? 0} / ${overall.total_cells ?? 0}`}
          color="purple"
        />
        <MetricCard
          title="Attack nguy hiểm nhất"
          value={worst?.attack ?? "No data"}
          trend={worst ? `${worst.mean_degradation_percent}%` : undefined}
          trendLabel="suy giảm TB xuyên run"
          trendType="down-positive"
          color="teal"
        />
      </section>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        <Card
          className="xl:col-span-7"
          title="Xu hướng robustness theo run"
          subtitle="Run xếp theo thời gian lưu; trục % để so sánh clean vs attacked xuyên các phiên."
        >
          {trend.robust_scores?.length >= 2 ? (
            <TrendLineChart
              data={runs
                .filter((run) => run.clean != null && run.attacked_mean != null)
                .map((run) => ({
                  exp: shortRunLabel(run),
                  Clean_mAP: Number((run.clean * 100).toFixed(2)),
                  Attacked_mAP: Number((run.attacked_mean * 100).toFixed(2)),
                }))}
            />
          ) : (
            <p className="p-6 text-center text-sm text-slate-500">
              Cần tối thiểu 2 run có metric đo được để vẽ xu hướng.
            </p>
          )}
          {trend.direction && trend.direction !== "insufficient_data" && (
            <p className="mt-2 text-xs text-slate-600">
              Xu hướng: <strong>{trend.direction}</strong>
              {trend.delta_first_last != null && (
                <span className="ml-1">
                  ({trend.delta_first_last > 0 ? "+" : ""}
                  {trend.delta_first_last} điểm robust score từ run đầu đến run mới nhất)
                </span>
              )}
            </p>
          )}
        </Card>
        <Card
          className="xl:col-span-5"
          title="Bằng chứng tổng hợp đa run"
          subtitle="Chỉ kết luận từ object-level evidence; thiếu dữ liệu thì ghi rõ No data."
        >
          <div className="space-y-3">
            <div className="rounded-lg border border-red-200 bg-red-50 p-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-red-700">
                Attack nguy hiểm nhất xuyên run
              </div>
              <div className="mt-1 text-base font-bold text-red-950">{worst?.attack ?? "No data"}</div>
              <p className="mt-1 text-xs text-red-800">
                {worst
                  ? `Suy giảm TB ${worst.mean_degradation_percent}% trên ${worst.runs_affected} run (tệ nhất ${worst.max_degradation_percent}% tại severity ${worst.worst_severity ?? "—"}).`
                  : "Chưa có breakdown đo được."}
              </p>
            </div>
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-amber-700">
                Severity nhạy cảm nhất
              </div>
              <div className="mt-1 text-base font-bold text-amber-950">
                {sensitive ? `Cấp ${sensitive.severity}` : "No data"}
              </div>
              <p className="mt-1 text-xs text-amber-800">
                {sensitive
                  ? `Suy giảm TB ${sensitive.mean_degradation_percent}% trên ${sensitive.runs_measured} run.`
                  : "Chưa có severity ladder đo được."}
              </p>
            </div>
            <div className="rounded-lg border border-blue-200 bg-blue-50 p-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-blue-700">Lớp dễ tổn thương nhất</div>
              <div className="mt-1 text-base font-bold text-blue-950">{vulnerable?.class_name ?? "No data"}</div>
              <p className="mt-1 text-xs text-blue-800">
                {vulnerable
                  ? `Mất ${vulnerable.detection_drop_percent}% trên ${vulnerable.total_ground_truth_objects} GT objects xuyên run.`
                  : "Cần object-level evidence."}
              </p>
            </div>
          </div>
        </Card>
      </div>

      {comparison && (
        <Card
          title="Bảng so sánh đa run"
          subtitle={`Suy giảm trung bình theo attack trên ${comparison.runs?.length ?? 0} run${
            comparison.comparable ? "" : " — cảnh báo: các run khác task, so sánh chỉ mang tham khảo"
          }.`}
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-xs">
              <thead className="border-b border-slate-200 bg-slate-50 text-slate-600">
                <tr>
                  <th className="p-3">Attack</th>
                  {(comparison.runs || []).map((run) => (
                    <th key={run.run_id} className="p-3 text-right">
                      {shortRunLabel(run)}
                      {run.robust_score != null && (
                        <span className="block text-[10px] font-normal text-slate-500">
                          robust {run.robust_score}
                        </span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {Object.entries(comparison.attack_pivot || {}).map(([attack, byRun]) => (
                  <tr key={attack}>
                    <th className="p-3 font-semibold text-slate-900">{attack}</th>
                    {(comparison.runs || []).map((run) => {
                      const entry = byRun[run.run_id];
                      const tokens = degTokens(entry?.mean_degradation_percent);
                      return (
                        <td
                          key={run.run_id}
                          className="p-3 text-right font-mono"
                          style={{ background: tokens.bg, color: tokens.text }}
                        >
                          {entry?.mean_degradation_percent == null
                            ? "—"
                            : `${entry.mean_degradation_percent.toFixed(1)}%`}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card
        title="Báo cáo run chi tiết"
        subtitle="Chọn một run của project để xem đầy đủ bảng, biểu đồ và heatmap degradation."
      >
        <label className="block max-w-md text-xs font-semibold text-slate-700">
          Run của project
          <select
            value={selectedRunId}
            onChange={(event) => setSelectedRunId(event.target.value)}
            className="mt-1 min-h-11 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
          >
            <option value="">— Chọn run —</option>
            {runs.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {shortRunLabel(run)} · {run.metric_label ?? "?"} · robust {run.robust_score ?? "—"}
              </option>
            ))}
          </select>
        </label>
        <div className="mt-4">
          {selectedReport ? <ReportView report={selectedReport} /> : null}
        </div>
      </Card>
    </div>
  );
}
