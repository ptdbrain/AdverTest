"use client";

import { useState } from "react";
import Card from "@/components/common/Card";
import Button from "@/components/common/Button";

export default function ReviewDecisionPanel({ review, cluster, onSubmit, submitting = false }) {
  const [decision, setDecision] = useState("");
  const [decisionNote, setDecisionNote] = useState("");
  return <Card title="Quyết định thẩm định" data-testid="review-decision-panel-card"><div className="space-y-3"><select aria-label="Decision" value={decision} onChange={(event) => setDecision(event.target.value)} className="w-full rounded-lg border border-slate-300 bg-[var(--app-surface)] p-2 text-sm"><option value="">Chọn quyết định</option><option value="BLOCK_DEPLOY">Chặn triển khai</option><option value="REQUEST_RETRAIN">Yêu cầu retrain</option><option value="RESTRICT_ODD">Giới hạn ODD</option><option value="RELABEL_DATA">Sửa nhãn dữ liệu</option><option value="ACCEPT_RISK">Chấp nhận rủi ro</option></select><textarea aria-label="Decision note" value={decisionNote} onChange={(event) => setDecisionNote(event.target.value)} className="min-h-20 w-full rounded-lg border border-slate-300 bg-[var(--app-surface)] p-2 text-sm" placeholder="Ghi chú quyết định" /><Button disabled={!review || !decision || submitting} onClick={() => onSubmit?.({ decision, decision_note: decisionNote, ...(cluster?.cluster_id ? { batch_cluster_id: cluster.cluster_id } : {}) })} className="w-full justify-center">{submitting ? "Đang lưu..." : "Lưu quyết định"}</Button></div></Card>;
}
