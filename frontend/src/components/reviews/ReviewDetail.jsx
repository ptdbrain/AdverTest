"use client";

import Card from "@/components/common/Card";

export default function ReviewDetail({ review, cluster, evidence }) {
  if (!review) return <Card title="Bằng chứng"><p className="py-8 text-center text-sm text-slate-500">Chưa chọn ca lỗi nào.</p></Card>;
  return <Card title={review.attack || "Review detail"} subtitle={review.review_id}><dl className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2"><div><dt className="text-slate-500">Model</dt><dd className="font-semibold">{review.model || "— / No verified data"}</dd></div><div><dt className="text-slate-500">Dataset</dt><dd className="font-semibold">{review.dataset || "— / No verified data"}</dd></div><div><dt className="text-slate-500">Cluster</dt><dd className="font-semibold">{cluster?.name || "—"}</dd></div><div><dt className="text-slate-500">Evidence</dt><dd className="font-semibold">{evidence ? "Available" : "— / No verified data"}</dd></div></dl></Card>;
}
