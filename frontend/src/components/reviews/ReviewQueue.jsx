"use client";

import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";

export default function ReviewQueue({ reviews = [], selectedId, onSelect, loading = false, error = "" }) {
  return <Card title="Review queue" data-testid="review-queue-card">
    {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
    {loading ? <p className="py-6 text-center text-sm text-slate-500">Đang tải...</p> : reviews.length === 0 ? <p className="py-6 text-center text-sm text-slate-500">Hàng đợi rỗng.</p> : <div className="space-y-2">{reviews.map((review) => <button key={review.review_id} type="button" onClick={() => onSelect?.(review)} className={`flex w-full items-center justify-between rounded-lg border p-3 text-left ${selectedId === review.review_id ? "border-blue-500 bg-blue-50" : "border-slate-200 bg-[var(--app-surface)]"}`}><span className="min-w-0 truncate text-sm font-semibold">{review.attack || review.review_id}</span><Badge variant="secondary">{review.risk_level || "—"}</Badge></button>)}</div>}
  </Card>;
}
