"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Database, Download, RefreshCw, Search, ShieldCheck } from "lucide-react";

import Button from "@/components/common/Button";
import Card from "@/components/common/Card";
import EvidenceBadge from "@/components/common/EvidenceBadge";
import {
  downloadRunExport,
  getRunAnalyticsAttacks,
  getRunAnalyticsClasses,
  getRunAnalyticsDistance,
  getRunAnalyticsSummary,
  listRuns,
} from "@/lib/api";
import { buildRunDecisionView, filterCompletedRuns, formatNumber, formatRatio, primaryMetricForCell } from "@/lib/reportMetrics";

function valueDelta(clean, attacked) {
  if (clean == null || attacked == null || clean <= 0) return null;
  return ((attacked - clean) / clean) * 100;
}

function NoData({ message = "Chạy một benchmark có ground truth để nhận metric, provenance và failure evidence." }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50 px-6 py-12 text-center">
      <Database className="mx-auto h-8 w-8 text-slate-400" aria-hidden="true" />
      <h2 className="mt-3 text-base font-semibold text-slate-900">Chưa có benchmark hoàn tất</h2>
      <p className="mx-auto mt-1 max-w-xl text-sm text-slate-600">{message}</p>
    </div>
  );
}

function ProtocolValue({ label, value }) {
  return <div><dt className="text-xs font-medium text-slate-500">{label}</dt><dd className="mt-1 break-all font-mono text-xs text-slate-900">{value ?? "—"}</dd></div>;
}

export function RunMetricsContent({ report, mode = "benchmark", analytics = {} }) {
  const view = useMemo(() => (report ? buildRunDecisionView(report) : null), [report]);
  const [exportError, setExportError] = useState("");
  if (!report || !view) return <NoData />;
  const exportEligible = report?.evidence?.status === "VERIFIED";

  const handleServerExport = async (format) => {
    setExportError("");
    try {
      await downloadRunExport(report.run_id, format);
    } catch (error) {
      setExportError(error?.message || "Không thể xuất report evidence.");
    }
  };

  const cells = report.cells || [];
  const worstAttack = analytics.summary?.worst_attack || analytics.attacks?.[0] || null;
  const vulnerableClass = [...(analytics.classes || [])].sort((a, b) => (b.detection_drop_percent ?? -1) - (a.detection_drop_percent ?? -1))[0] || null;
  const nextAction = view.dataState !== "MEASURED"
    ? "Chạy benchmark trên dataset có ground truth và split khóa trước khi kết luận."
    : report.simulation_only !== false
      ? "Lặp lại cùng protocol bằng checkpoint và dataset artifact đã xác thực; giữ nguyên seed, split và ngưỡng IoU."
      : view.degradationPercent >= 20
        ? "Mở failure samples của đòn tệ nhất, duyệt lỗi theo lớp rồi retest checkpoint phòng thủ trên đúng locked protocol."
        : "Kiểm tra worst cases và khoảng tin cậy trước khi đưa kết quả vào deployment gate.";

  return (
    <div className="space-y-5">
      <section className="rounded-xl border border-slate-200 bg-slate-950 p-4 text-white" aria-label="Run evidence context">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <EvidenceBadge provenance={report.provenance} simulationOnly={report.simulation_only} evidence={report.evidence} />
              <span className="rounded-md border border-slate-700 px-2 py-1 font-mono text-xs text-slate-300">{report.run_id}</span>
            </div>
            <h2 className="mt-3 text-lg font-semibold">{report.model || "Unknown model"} · {report.dataset || "Unknown dataset"}</h2>
            <p className="mt-1 text-sm text-slate-300">{view.taskId} · {report.n_samples ?? "—"} samples · {cells.length} attack cells · {formatNumber(report.seconds, 2, " s")}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {['json', 'csv', 'pdf', 'zip'].map((format) => <Button key={format} className="min-h-11" variant="secondary" icon={Download} disabled={!exportEligible} title={exportEligible ? undefined : "Cần benchmark evidence VERIFIED trước khi export"} onClick={() => handleServerExport(format)}>{format.toUpperCase()}</Button>)}
          </div>
        </div>
        {exportError && <p role="alert" className="mt-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-950">{exportError}</p>}
      </section>

      {view.dataState !== "MEASURED" && <div role="alert" className="flex gap-3 rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950"><AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" /><div><strong>Không đủ dữ liệu benchmark.</strong> Các giá trị thiếu được giữ là “—”, không quy đổi thành 0 hoặc điểm tốt giả.</div></div>}
      {(report.skipped || []).length > 0 && <div role="alert" className="flex gap-3 rounded-xl border border-orange-300 bg-orange-50 p-4 text-sm text-orange-950"><AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" /><div><strong>{report.skipped.length} attack đã bị bỏ qua.</strong><ul className="mt-1 list-disc pl-5 text-xs leading-5">{report.skipped.map((item, index) => <li key={`${item.attack}-${index}`}>{item.attack || "Unknown attack"}: {item.reason || "Không có lý do"}</li>)}</ul></div></div>}

      <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4" aria-label="Decision metrics">
        {[
          { label: `${view.primary.label} clean`, value: formatRatio(view.primary.clean), note: view.primary.help },
          { label: `${view.primary.label} attacked`, value: formatRatio(view.primary.attacked), note: `Trung bình ${cells.length} attack cells` },
          { label: "Hiệu năng giữ lại", value: formatNumber(view.robustnessRetained, 1, "%"), note: "attacked ÷ clean; không phải chứng nhận an toàn" },
          { label: "Độ suy giảm", value: formatNumber(view.degradationPercent, 1, "%"), note: "Mức mất tương đối so với clean" },
        ].map((item) => <div key={item.label} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"><div className="text-xs font-semibold text-slate-600">{item.label}</div><div className="mt-2 text-2xl font-bold text-slate-950">{item.value}</div><p className="mt-2 text-xs leading-5 text-slate-500">{item.note}</p></div>)}
      </section>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
        <Card className="xl:col-span-8" title="Metric theo đúng bài toán" subtitle="Chỉ hiển thị metric backend thực sự trả về; ô thiếu giữ nguyên No data.">
          <div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left text-xs"><thead className="border-b border-slate-200 bg-slate-50 text-slate-600"><tr><th className="p-3">Metric</th><th className="p-3 text-right">Clean</th><th className="p-3 text-right">Attacked mean</th><th className="p-3 text-right">Thay đổi</th><th className="p-3">Ý nghĩa</th></tr></thead><tbody className="divide-y divide-slate-100">
            {view.metrics.map((metric) => { const delta = valueDelta(metric.clean, metric.attacked); return <tr key={metric.key}><th className="p-3 font-semibold text-slate-900">{metric.label}</th><td className="p-3 text-right font-mono">{formatRatio(metric.clean)}</td><td className="p-3 text-right font-mono">{formatRatio(metric.attacked)}</td><td className={`p-3 text-right font-mono font-semibold ${delta != null && delta < 0 ? "text-red-700" : "text-slate-500"}`}>{delta == null ? "—" : `${delta.toFixed(1)}%`}</td><td className="p-3 text-slate-600">{metric.help}</td></tr>; })}
          </tbody></table></div>
        </Card>
        <Card className="xl:col-span-4" title="Protocol & provenance" subtitle="Điều kiện tối thiểu để so sánh hoặc tái lập."><dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-1"><ProtocolValue label="Benchmark protocol" value={view.protocol.benchmarkProtocolId} /><ProtocolValue label="Dataset version" value={view.protocol.datasetVersionId} /><ProtocolValue label="Split" value={view.protocol.split} /><ProtocolValue label="Checkpoint hash" value={view.protocol.checkpointHash} /><ProtocolValue label="Seed / IoU" value={view.protocol.seed == null && view.protocol.iouThreshold == null ? null : `${view.protocol.seed ?? "—"} / ${view.protocol.iouThreshold ?? "—"}`} /></dl></Card>
      </div>

      <Card title="Attack × severity" subtitle={`Giá trị ${view.primary.label} đo được cho từng cell; không trộn task hoặc protocol khác nhau.`}>
        {cells.length === 0 ? <NoData message="Run này chưa có attack cell đo được." /> : <div className="overflow-x-auto"><table className="w-full min-w-[660px] text-left text-xs"><thead className="border-b border-slate-200 bg-slate-50 text-slate-600"><tr><th className="p-3">Attack</th><th className="p-3">Severity</th><th className="p-3 text-right">Clean</th><th className="p-3 text-right">Attacked</th><th className="p-3 text-right">Degradation</th><th className="p-3 text-right">Samples</th></tr></thead><tbody className="divide-y divide-slate-100">
          {cells.map((cell, index) => { const attacked = primaryMetricForCell(report, cell, view.taskId, view.primary.key); const derived = valueDelta(view.primary.clean, attacked); const degradation = typeof cell.degradation_percent === "number" && attacked != null ? cell.degradation_percent : derived == null ? null : Math.max(0, -derived); return <tr key={`${cell.attack}-${cell.severity}-${index}`}><th className="p-3 font-semibold text-slate-900">{cell.attack || "—"}</th><td className="p-3">{cell.severity ?? "—"}</td><td className="p-3 text-right font-mono">{formatRatio(view.primary.clean)}</td><td className="p-3 text-right font-mono">{formatRatio(attacked)}</td><td className="p-3 text-right font-mono font-semibold text-red-700">{formatNumber(degradation, 1, "%")}</td><td className="p-3 text-right font-mono">{cell.n_samples ?? report.n_samples ?? "—"}</td></tr>; })}
        </tbody></table></div>}
      </Card>

      {mode === "analysis" && <>
        <section className="grid grid-cols-1 gap-3 lg:grid-cols-3" aria-label="Measured findings">
          <div className="rounded-xl border border-red-200 bg-red-50 p-4"><div className="text-xs font-semibold uppercase tracking-wide text-red-700">Attack ưu tiên điều tra</div><div className="mt-2 text-base font-bold text-red-950">{worstAttack?.attack ?? "No data"}</div><p className="mt-1 text-xs text-red-800">{worstAttack?.mean_degradation_percent == null ? "Không có breakdown đo được." : `Suy giảm trung bình ${worstAttack.mean_degradation_percent.toFixed(1)}%.`}</p></div>
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-4"><div className="text-xs font-semibold uppercase tracking-wide text-amber-700">Lớp dễ tổn thương</div><div className="mt-2 text-base font-bold text-amber-950">{vulnerableClass?.class_name ?? "No data"}</div><p className="mt-1 text-xs text-amber-800">{vulnerableClass ? `Mất ${vulnerableClass.detection_drop_percent.toFixed(1)}% trên ${vulnerableClass.total_ground_truth_objects} GT objects.` : "Cần object-level evidence để kết luận."}</p></div>
          <div className="rounded-xl border border-blue-200 bg-blue-50 p-4"><div className="text-xs font-semibold uppercase tracking-wide text-blue-700">Khoảng cách dễ tổn thương</div><div className="mt-2 text-base font-bold text-blue-950">{view.taskId === "detection3d" ? analytics.distance?.most_vulnerable_distance ?? "No data" : "Không áp dụng"}</div><p className="mt-1 text-xs text-blue-800">Chỉ có ý nghĩa cho detection3d khi report chứa metric near/medium/far.</p></div>
        </section>
        {(analytics.classes || []).length > 0 && <Card title="Failure theo lớp" subtitle="Chỉ tính từ object-level ground-truth evidence; không suy ra từ prediction label."><div className="overflow-x-auto"><table className="w-full min-w-[680px] text-left text-xs"><thead className="border-b border-slate-200 bg-slate-50 text-slate-600"><tr><th className="p-3">Lớp</th><th className="p-3 text-right">GT objects</th><th className="p-3 text-right">Clean detection</th><th className="p-3 text-right">Attacked detection</th><th className="p-3 text-right">Mất sau attack</th><th className="p-3 text-right">Hallucinated</th></tr></thead><tbody className="divide-y divide-slate-100">{analytics.classes.map((item) => <tr key={item.class_name}><th className="p-3 font-semibold text-slate-900">{item.class_name}</th><td className="p-3 text-right font-mono">{item.total_ground_truth_objects ?? "—"}</td><td className="p-3 text-right font-mono">{formatRatio(item.clean_detection_rate)}</td><td className="p-3 text-right font-mono">{formatRatio(item.attacked_detection_rate)}</td><td className="p-3 text-right font-mono font-semibold text-red-700">{item.lost_objects_count ?? "—"}</td><td className="p-3 text-right font-mono">{item.hallucinated_objects_count ?? "—"}</td></tr>)}</tbody></table></div></Card>}
        {view.taskId === "detection3d" && analytics.distance?.data_state === "MEASURED" && <Card title="Độ bền theo khoảng cách" subtitle="Near/medium/far chỉ xuất hiện khi report có metric 3D riêng cho từng bucket."><div className="overflow-x-auto"><table className="w-full min-w-[620px] text-left text-xs"><thead className="border-b border-slate-200 bg-slate-50 text-slate-600"><tr><th className="p-3">Khoảng</th><th className="p-3">Cự ly</th><th className="p-3 text-right">Clean AP</th><th className="p-3 text-right">Attacked AP</th><th className="p-3 text-right">Suy giảm</th><th className="p-3 text-right">Failure cases</th></tr></thead><tbody className="divide-y divide-slate-100">{Object.entries(analytics.distance.buckets || {}).map(([name, bucket]) => <tr key={name}><th className="p-3 font-semibold capitalize text-slate-900">{name}</th><td className="p-3">{bucket.range_meters || "—"}</td><td className="p-3 text-right font-mono">{formatRatio(bucket.clean_ap)}</td><td className="p-3 text-right font-mono">{formatRatio(bucket.attacked_ap)}</td><td className="p-3 text-right font-mono font-semibold text-red-700">{formatNumber(bucket.degradation_percent, 1, "%")}</td><td className="p-3 text-right font-mono">{bucket.failure_count ?? "—"}</td></tr>)}</tbody></table></div></Card>}
        <Card title="Kết luận có thể hành động" subtitle="Khuyến nghị theo trạng thái bằng chứng, không phải nội dung AI tạo giả."><div className="flex gap-3 rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-950"><ShieldCheck className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" /><p>{nextAction}</p></div><div className="mt-4"><h3 className="text-sm font-semibold text-slate-900">Giới hạn cần đọc trước khi quyết định</h3>{view.limitations.length ? <ul className="mt-2 list-disc space-y-1 pl-5 text-xs leading-5 text-slate-700">{view.limitations.map((item) => <li key={item}>{item}</li>)}</ul> : <p className="mt-2 text-xs text-slate-600">Không có limitation tự động; vẫn cần review protocol và confidence interval.</p>}</div></Card>
      </>}
    </div>
  );
}

export default function RunMetricsWorkspace({ mode = "benchmark" }) {
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [runQuery, setRunQuery] = useState("");
  const [analytics, setAnalytics] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const loadRuns = async () => { setLoading(true); setError(""); try { const items = await listRuns(); const completed = (Array.isArray(items) ? items : []).filter((item) => item.status === "COMPLETED" && item.report); setRuns(completed); setSelectedRunId((current) => completed.some((item) => item.run_id === current) ? current : completed[0]?.run_id || ""); } catch (requestError) { setError(requestError.message || "Không tải được benchmark runs."); } finally { setLoading(false); } };
  useEffect(() => { loadRuns(); }, []);
  const filteredRuns = useMemo(() => filterCompletedRuns(runs, runQuery), [runs, runQuery]);
  const active = runs.find((item) => item.run_id === selectedRunId) || null;
  useEffect(() => { if (!selectedRunId) { setAnalytics({}); return; } let cancelled = false; Promise.allSettled([getRunAnalyticsSummary(selectedRunId), getRunAnalyticsAttacks(selectedRunId), getRunAnalyticsClasses(selectedRunId), getRunAnalyticsDistance(selectedRunId)]).then(([summary, attacks, classes, distance]) => { if (!cancelled) setAnalytics({ summary: summary.status === "fulfilled" ? summary.value : null, attacks: attacks.status === "fulfilled" ? attacks.value : [], classes: classes.status === "fulfilled" ? classes.value : [], distance: distance.status === "fulfilled" ? distance.value : null }); }); return () => { cancelled = true; }; }, [selectedRunId]);
  if (loading) return <div role="status" className="rounded-xl border border-slate-200 bg-white p-8 text-sm text-slate-600">Đang tải completed runs và report evidence…</div>;
  if (error) return <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900">{error}</div>;
  if (!runs.length) return <RunMetricsContent report={null} mode={mode} analytics={{}} />;
  return <div className="space-y-4"><div className="grid grid-cols-1 gap-3 rounded-xl border border-slate-200 bg-white p-4 lg:grid-cols-[minmax(220px,0.8fr)_minmax(320px,1.2fr)_auto] lg:items-end"><label className="block min-w-0 text-xs font-semibold text-slate-700">Tìm phiên benchmark<div className="relative mt-1"><Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" aria-hidden="true" /><input type="search" value={runQuery} onChange={(event) => setRunQuery(event.target.value)} placeholder="Run ID, model, dataset, task…" className="min-h-11 w-full rounded-lg border border-slate-300 bg-white pl-9 pr-3 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200" /></div></label><label className="block min-w-0 text-xs font-semibold text-slate-700">Completed benchmark run<select value={filteredRuns.some((item) => item.run_id === selectedRunId) ? selectedRunId : ""} onChange={(event) => setSelectedRunId(event.target.value)} className="mt-1 min-h-11 w-full rounded-lg border border-slate-300 bg-white px-3 text-sm text-slate-900 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"><option value="" disabled>{filteredRuns.length ? `Chọn 1 trong ${filteredRuns.length} phiên` : "Không tìm thấy phiên phù hợp"}</option>{filteredRuns.map((item) => <option key={item.run_id} value={item.run_id}>{item.run_id} · {item.report?.model} · {item.report?.dataset}</option>)}</select></label><Button variant="secondary" icon={RefreshCw} className="min-h-11" onClick={loadRuns}>Làm mới</Button><p className="text-xs text-slate-500 lg:col-span-3">Hiển thị {filteredRuns.length}/{runs.length} phiên hoàn tất. Kết quả bên dưới luôn thuộc đúng run đang chọn.</p></div><RunMetricsContent report={active?.report || null} mode={mode} analytics={analytics} /></div>;
}
