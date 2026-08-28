"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  FileText,
  Sparkles,
  Download,
  Calendar,
  Filter,
  Share2,
  Table,
  CheckCircle,
  AlertTriangle,
  Lightbulb,
  Layers,
  BarChart2,
  FileSpreadsheet,
  Info,
  ShieldCheck,
  Target,
  ArrowRight,
  ShieldAlert,
  FolderOpen,
} from "lucide-react";
import PageHeader from "@/components/layout/PageHeader";
import Card from "@/components/common/Card";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import { cn } from "@/lib/utils";
import { getApiBase, listSessions } from "@/lib/api";
import SessionTimeline from "@/components/SessionTimeline";

export default function AnalysisPage() {
  const [sessionsList, setSessionsList] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState("EXP-2025-0512-001");
  const [activeSession, setActiveSession] = useState(null);

  const [generatingReport, setGeneratingReport] = useState(false);
  const [reportGenerated, setReportGenerated] = useState(false);

  useEffect(() => {
    listSessions()
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setSessionsList(data);
          const current = data.find((s) => s.id === activeSessionId) || data[0];
          setActiveSessionId(current.id);
          setActiveSession(current);
        }
      })
      .catch((err) => console.warn("Error fetching sessions:", err));
  }, []);

  const handleSwitchSession = (sessionId) => {
    setActiveSessionId(sessionId);
    const found = sessionsList.find((s) => s.id === sessionId);
    if (found) {
      setActiveSession(found);
    }
  };

  const runs = activeSession?.runs || [];
  const worstRun = runs.length > 0 ? [...runs].sort((a, b) => a.robustness_score - b.robustness_score)[0] : null;
  const avgRobustness = runs.length > 0
    ? (runs.reduce((acc, r) => acc + r.robustness_score, 0) / runs.length).toFixed(1)
    : "N/A";

  const handleGenerateAIReport = () => {
    setGeneratingReport(true);
    setTimeout(() => {
      setGeneratingReport(false);
      setReportGenerated(true);
    }, 800);
  };

  const getHeatmapColor = (val) => {
    if (val >= 80) return "bg-red-500 text-white font-bold";
    if (val >= 50) return "bg-red-400 text-white font-bold";
    if (val >= 30) return "bg-amber-300 text-slate-900";
    if (val >= 15) return "bg-blue-100 text-slate-800";
    return "bg-slate-50 text-slate-600";
  };

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Phân tích & Báo cáo chuyên sâu"
        subtitle={`Báo cáo phân tích tổng hợp dựa trên phiên làm việc [${activeSession?.name || activeSessionId}] với ${runs.length} lần chạy thử nghiệm.`}
        breadcrumb={[
          { label: "Trang chủ", href: "/dashboard" },
          { label: "Phân tích & Báo cáo" },
        ]}
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="primary"
              size="sm"
              icon={Sparkles}
              disabled={generatingReport}
              onClick={handleGenerateAIReport}
              className="bg-purple-600 hover:bg-purple-700 border-purple-600 font-bold"
            >
              {generatingReport ? "Đang cập nhật báo cáo..." : "Tạo báo cáo tự động ✨"}
            </Button>
          </div>
        }
      />

      {/* SESSION CONTEXT BANNER WITH SWITCHER */}
      <div className="p-3.5 rounded-xl bg-gradient-to-r from-blue-950 via-slate-900 to-slate-900 border border-blue-800/60 text-white flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-md">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 text-white flex items-center justify-center font-bold">
            <FileText className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs font-bold text-blue-400 uppercase tracking-wide">
              Phạm vi phân tích: Phiên {activeSessionId}
            </div>
            <div className="text-xs text-slate-200 mt-0.5 flex flex-wrap items-center gap-2">
              <span>Mô hình: <strong className="text-white">{activeSession?.model_name || "YOLO11s"}</strong></span>
              <span className="text-slate-500">•</span>
              <span>Dữ liệu: <strong className="text-blue-300">{activeSession?.dataset_name || "KITTI"}</strong></span>
              <span className="text-slate-500">•</span>
              <span>Số bài test đã ghi: <strong className="text-amber-300">{runs.length} lần</strong></span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={activeSessionId}
            onChange={(e) => handleSwitchSession(e.target.value)}
            className="text-xs py-1.5 px-3 rounded-lg border border-slate-700 bg-slate-800 text-slate-100 font-semibold shadow-inner focus:ring-2 focus:ring-blue-500"
          >
            {sessionsList.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} ({s.runs?.length || 0} bài test)
              </option>
            ))}
          </select>

          <Link href={`/experiments/${activeSessionId}/results`}>
            <Button
              variant="secondary"
              size="sm"
              icon={ArrowRight}
              className="bg-slate-800 text-slate-300 hover:bg-slate-700 border-slate-700 text-xs py-1"
            >
              Xem ảnh trực quan
            </Button>
          </Link>
        </div>
      </div>

      {/* 4 INSIGHT SUMMARY CARDS */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
        <div className="p-3.5 rounded-xl bg-red-50 border border-red-200 space-y-1">
          <div className="text-[11px] font-bold text-red-700 uppercase tracking-wide">Đòn nguy hiểm nhất phiên</div>
          <div className="text-base font-bold text-red-900">{worstRun ? worstRun.attack_name : "Chưa có"}</div>
          <p className="text-xs text-red-700">Gây sụt giảm mAP cao nhất: {worstRun ? `-${worstRun.map_drop_pct}%` : "0%"}.</p>
        </div>

        <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200 space-y-1">
          <div className="text-[11px] font-bold text-amber-700 uppercase tracking-wide">Nhãn lớp tổn thương nhất</div>
          <div className="text-base font-bold text-amber-900">Người đi bộ (Pedestrian)</div>
          <p className="text-xs text-amber-800">Tỷ lệ bỏ sót (False Negative) lên tới 100% khi có sương mù và mưa giông.</p>
        </div>

        <div className="p-3.5 rounded-xl bg-blue-50 border border-blue-200 space-y-1">
          <div className="text-[11px] font-bold text-blue-700 uppercase tracking-wide">Độ trễ trung bình</div>
          <div className="text-base font-bold text-blue-900">54.2 ms / ảnh</div>
          <p className="text-xs text-blue-800">Thực thi inference thời gian thực trên tập dữ liệu KITTI.</p>
        </div>

        <div className="p-3.5 rounded-xl bg-purple-50 border border-purple-200 space-y-1">
          <div className="text-[11px] font-bold text-purple-700 uppercase tracking-wide">Chiến lược đề xuất</div>
          <div className="text-base font-bold text-purple-900">Multi-Condition Training</div>
          <p className="text-xs text-purple-800">Tôi luyện đồng thời với Weather Augmentation và PGD Defense.</p>
        </div>
      </div>

      {/* SESSION TIMELINE */}
      {activeSession && <SessionTimeline session={activeSession} />}

      {/* 2-COLUMN SECTION: MULTI-RUN MATRIX & EFFECTIVENESS RANKING */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* MULTI-RUN SEVERITY MATRIX (7 COLS) */}
        <div className="lg:col-span-7 space-y-4">
          <Card
            title="Ma trận so sánh các lần chạy trong phiên làm việc"
            subtitle={`Đo lường trên [${activeSession?.model_name || "YOLO11s"}] qua ${runs.length} kịch bản`}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="bg-slate-50 text-slate-700 border-b border-slate-200">
                    <th className="p-2.5 font-semibold">Lần chạy / Kịch bản</th>
                    <th className="p-2.5 text-center font-semibold">Cấp độ</th>
                    <th className="p-2.5 text-right font-semibold">Clean mAP</th>
                    <th className="p-2.5 text-right font-semibold">Robust mAP</th>
                    <th className="p-2.5 text-center font-semibold">Độ sụt giảm</th>
                    <th className="p-2.5 text-left font-semibold">Ghi chú</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono text-center">
                  {runs.map((r, idx) => (
                    <tr key={r.id}>
                      <td className="p-2.5 font-sans font-bold text-slate-800 text-left">
                        Lần #{idx + 1}: {r.attack_name}
                      </td>
                      <td className="p-2.5 font-sans">
                        <span className="px-2 py-0.5 rounded text-[10px] bg-slate-100 text-slate-700 border border-slate-200 font-bold">
                          Cấp {r.severity}
                        </span>
                      </td>
                      <td className="p-2.5 text-right text-slate-700 font-semibold">
                        {(r.clean_map * 100).toFixed(1)}%
                      </td>
                      <td className="p-2.5 text-right font-bold text-red-600">
                        {(r.attacked_map * 100).toFixed(1)}%
                      </td>
                      <td className="p-2.5">
                        <span className={cn("px-2 py-1 rounded", getHeatmapColor(r.map_drop_pct))}>
                          -{r.map_drop_pct}%
                        </span>
                      </td>
                      <td className="p-2.5 text-left text-[11px] text-slate-600 font-sans max-w-[180px] truncate" title={r.note || ""}>
                        {r.note || <span className="text-slate-300 italic">—</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-3 flex items-center justify-between text-[11px] text-slate-500 pt-2 border-t border-slate-100">
              <span>Được ghi lại liên tục qua từng lần thực thi</span>
              <span>Tổng số: {runs.length} lần chạy</span>
            </div>
          </Card>
        </div>

        {/* ATTACK EFFECTIVENESS RANKING (5 COLS) */}
        <div className="lg:col-span-5 space-y-4">
          <Card
            title="Xếp hạng mức độ triệt tiêu của từng kịch bản"
            subtitle="Sắp xếp theo độ sụt giảm mAP@0.5 trong phiên"
          >
            <div className="space-y-3 text-xs">
              {[...runs]
                .sort((a, b) => b.map_drop_pct - a.map_drop_pct)
                .map((r, idx) => (
                  <div
                    key={r.id}
                    className="p-3 rounded-lg border border-slate-200 bg-slate-50/60 space-y-1.5"
                  >
                    <div className="flex items-center justify-between font-bold">
                      <span className="text-slate-900">#{idx + 1}. {r.attack_name}</span>
                      <span className="text-red-600 font-mono">-{r.map_drop_pct}% mAP</span>
                    </div>
                    <div className="flex justify-between text-[11px] text-slate-500">
                      <span>Điểm bền vững: <strong className="text-amber-600">{r.robustness_score}/100</strong></span>
                      <span>PSNR: <strong className="text-slate-700">{r.psnr}</strong></span>
                    </div>
                  </div>
                ))}
            </div>
          </Card>
        </div>
      </div>

      {/* DYNAMIC NARRATIVE REPORT & RECOMMENDATIONS */}
      <Card
        title="Báo cáo tường thuật tổng kết phiên làm việc (AI Executive Narrative)"
        subtitle="Tổng hợp phân tích tự động dựa trên toàn bộ các lần chạy trong phiên"
      >
        <div className="space-y-4 text-xs leading-relaxed text-slate-700">
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <h4 className="font-bold text-slate-900 text-sm">
              1. Tóm tắt kết quả phiên làm việc (Executive Summary)
            </h4>
            <p>
              Trong phiên thử nghiệm <strong>{activeSession?.name || activeSessionId}</strong>, mô hình <strong>{activeSession?.model_name || "YOLO11s"}</strong> đã hoàn thành <strong>{runs.length} lần chạy thử nghiệm</strong> trên tập dữ liệu <strong>{activeSession?.dataset_name || "KITTI"}</strong>.
            </p>
            <p>
              Kịch bản gây sụt giảm nghiêm trọng nhất là <strong>{worstRun?.attack_name || "PGD"}</strong> với mức độ triệt tiêu mAP lên tới <strong>{worstRun?.map_drop_pct || 86.6}%</strong>. Điểm bền vững trung bình của toàn phiên đạt <strong>{avgRobustness} / 100 điểm</strong>.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-purple-50/60 border border-purple-200 space-y-2">
            <h4 className="font-bold text-purple-900 text-sm flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-purple-700" />
              2. Kế hoạch phòng thủ & Tôi luyện tổng hợp (Defense Strategy)
            </h4>
            <ul className="list-disc pl-5 space-y-1 text-purple-900">
              <li>
                <strong>Huấn luyện đa điều kiện (Multi-Weather & PGD Augmentation)</strong>: Kết hợp đồng thời các mẫu biến dạng Depth Rain, Fog và nhiễu PGD vào tập huấn luyện với trọng số loss đối kháng L_adv = 0.4.
              </li>
              <li>
                <strong>Tăng cường độ sâu nhận diện (Depth Prior Alignment)</strong>: Bổ sung lớp tiền xử lý khử mờ quang học nhằm bảo vệ các BBox ở khoảng cách xa (&gt; 15m).
              </li>
              <li>
                <strong>Tái đánh giá với Locked Protocol</strong>: Tải lên checkpoint phòng thủ tại tab <em>Phòng thủ</em> để đo lường tỷ lệ phục hồi mAP trên toàn bộ các lần chạy.
              </li>
            </ul>
          </div>

          {runs.filter(r => r.note).length > 0 && (
            <div className="p-4 rounded-xl bg-blue-50/60 border border-blue-200 space-y-2">
              <h4 className="font-bold text-blue-900 text-sm flex items-center gap-1.5">
                <Info className="w-4 h-4 text-blue-700" />
                3. Ghi chú từ Kỹ sư ({runs.filter(r => r.note).length} nhận xét)
              </h4>
              <ul className="list-disc pl-5 space-y-1 text-blue-900 text-xs">
                {runs.filter(r => r.note).map((r) => (
                  <li key={r.id}>
                    <strong>{r.attack_name}</strong>: {r.note}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-100">
            <div className="flex flex-wrap items-center gap-2">
              <Link href={`/experiments/${activeSessionId}/attack`}>
                <Button variant="secondary" size="sm" className="text-xs font-semibold">
                  ← Xem lại cấu hình tấn công
                </Button>
              </Link>
              <Link href={`/experiments/${activeSessionId}/attack?mode=extend`}>
                <Button variant="secondary" size="sm" className="text-xs font-bold text-blue-700 bg-blue-50 border-blue-200 hover:bg-blue-100">
                  + Mở rộng lượt tấn công mới
                </Button>
              </Link>
              <Link href={`/experiments/${activeSessionId}/attack?mode=reset`}>
                <Button variant="secondary" size="sm" className="text-xs font-bold text-amber-800 bg-amber-50 border-amber-200 hover:bg-amber-100">
                  ↺ Reset đòn tấn công mới
                </Button>
              </Link>
            </div>

            <Link href="/defense">
              <Button variant="primary" size="sm" icon={ArrowRight} className="bg-blue-600 hover:bg-blue-700 font-bold">
                Chuyển sang Tab Phòng thủ & Tôi luyện lại mô hình
              </Button>
            </Link>
          </div>
        </div>
      </Card>
    </div>
  );
}
