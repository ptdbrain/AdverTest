"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { RefreshCw } from "lucide-react";
import { getReviews, getRunSamples, resolveReview } from "@/lib/api";
import { useProject } from "@/context/ProjectContext";
import ReviewQueue from "@/components/reviews/ReviewQueue";
import ReviewDetail from "@/components/reviews/ReviewDetail";
import ReviewDecisionPanel from "@/components/reviews/ReviewDecisionPanel";

/** A compact, responsive review queue. AppShell supplies global navigation. */
export default function ReviewPage() {
  const { activeProject, activeProjectId, isLoadingProjects } = useProject();
  const [status, setStatus] = useState("PENDING");
  const [reviews, setReviews] = useState([]);
  const [selected, setSelected] = useState(null);
  const [evidence, setEvidence] = useState([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    if (!activeProjectId) return;
    setLoading(true); setError("");
    try {
      const items = await getReviews({ status, project_id: activeProjectId });
      const next = Array.isArray(items) ? items : [];
      setReviews(next);
      setSelected((current) => next.find((item) => item.review_id === current?.review_id) || next[0] || null);
    } catch (cause) { setReviews([]); setSelected(null); setError(cause.message || "Không thể tải hàng đợi review."); }
    finally { setLoading(false); }
  }, [activeProjectId, status]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    let cancelled = false;
    if (!selected?.run_id || !activeProjectId) { setEvidence([]); return undefined; }
    getRunSamples(selected.run_id, {}, activeProjectId).then((items) => { if (!cancelled) setEvidence(Array.isArray(items) ? items : []); }).catch(() => { if (!cancelled) setEvidence([]); });
    return () => { cancelled = true; };
  }, [activeProjectId, selected?.run_id]);

  const submit = async ({ decision, decision_note }) => {
    if (!selected) return;
    setSubmitting(true); setError("");
    try { await resolveReview(selected.review_id, decision, decision_note, "Current reviewer"); await load(); }
    catch (cause) { setError(cause.message || "Không thể lưu quyết định review."); }
    finally { setSubmitting(false); }
  };

  if (!isLoadingProjects && !activeProjectId) return <main className="mx-auto max-w-3xl p-6"><h1 className="text-xl font-bold">Thẩm định</h1><p className="mt-3 text-sm text-slate-500">Hãy chọn hoặc tạo project trước khi mở hàng đợi review.</p><Link className="mt-4 inline-block text-sm text-blue-600 underline" href="/dashboard">Đi tới tổng quan project</Link></main>;

  return <main className="mx-auto max-w-7xl space-y-4 p-4 sm:p-6"><div className="flex flex-wrap items-center justify-between gap-3"><div><h1 className="text-xl font-bold">Thẩm định kết quả</h1><p className="text-sm text-slate-500">{activeProject?.name || "Đang tải project..."}</p></div><div className="flex gap-2"><select aria-label="Review status" value={status} onChange={(event) => setStatus(event.target.value)} className="rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)] px-3 py-2 text-sm"><option value="PENDING">Chờ duyệt</option><option value="RESOLVED">Đã duyệt</option></select><button type="button" onClick={load} className="rounded-lg border border-[var(--app-border)] p-2" aria-label="Tải lại review"><RefreshCw className={loading ? "h-4 w-4 animate-spin" : "h-4 w-4"} /></button></div></div>{error && <p role="alert" className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700">{error}</p>}<div className="grid gap-4 lg:grid-cols-[minmax(15rem,0.8fr)_minmax(0,1.4fr)_minmax(16rem,0.8fr)]"><ReviewQueue reviews={reviews} selectedId={selected?.review_id} onSelect={setSelected} loading={loading} error="" /><div className="space-y-4"><ReviewDetail review={selected} evidence={evidence.length > 0} /><section className="rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4"><h2 className="font-semibold">Bằng chứng</h2><p className="mt-2 text-sm text-slate-500">{evidence.length ? `${evidence.length} mẫu bằng chứng đã xác minh cho run này.` : "— / No verified data"}</p></section></div><ReviewDecisionPanel review={selected} onSubmit={submit} submitting={submitting} /></div></main>;
}
