"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import Link from "next/link";
import UserMenu from "@/components/UserMenu.jsx";
import RiskBadge from "@/components/RiskBadge.jsx";
import { useLanguage } from "@/context/LanguageContext";
import {
  getReviews,
  resolveReview,
  getRunSamples,
  getFailureClusters,
  createFailureCluster,
  autoGroupFailureClusters,
  assessReviewRisk,
  getRiskSessionSummary,
  triggerAutoFlag,
  createRetrainingBacklog,
  addRetrainingBacklogItem,
  getApiBase,
} from "@/lib/api";
import {
  ShieldAlert,
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
  Tag,
  Sparkles,
  Layers,
  ListFilter,
  ArrowRight,
  RefreshCw,
  Eye,
  Sliders,
  FileText,
  ExternalLink,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

const DEFAULT_PRESET_TAGS = [
  "🌫️ Sương mù & Khói",
  "🌧️ Mưa & Tuyết rơi",
  "⚡ Nhiễu cảm biến ISO",
  "🌓 Mất tương phản",
  "🛡️ Tấn công Gradient",
  "🎯 Vật thể bị che khuất",
];

const DECISION_OPTIONS = [
  {
    key: "BLOCK_DEPLOY",
    label: "Chặn triển khai",
    sublabel: "Bắt buộc retrain và khóa CI/CD gate",
    icon: ShieldAlert,
    color: "border-red-300 bg-red-50/80 text-red-900 hover:bg-red-100/80 dark:bg-red-950/40 dark:text-red-200 dark:border-red-800",
    activeColor: "ring-2 ring-red-500 border-red-500 bg-red-100 dark:bg-red-950/70",
    badgeColor: "bg-red-500 text-white",
  },
  {
    key: "REQUEST_RETRAIN",
    label: "Yêu cầu retrain",
    sublabel: "Đưa vào Retraining Backlog & tôi luyện",
    icon: RefreshCw,
    color: "border-orange-300 bg-orange-50/80 text-orange-900 hover:bg-orange-100/80 dark:bg-orange-950/40 dark:text-orange-200 dark:border-orange-800",
    activeColor: "ring-2 ring-orange-500 border-orange-500 bg-orange-100 dark:bg-orange-950/70",
    badgeColor: "bg-orange-500 text-white",
  },
  {
    key: "RESTRICT_ODD",
    label: "Giới hạn ODD",
    sublabel: "Hạn chế vùng vận hành an toàn",
    icon: Sliders,
    color: "border-amber-300 bg-amber-50/80 text-amber-900 hover:bg-amber-100/80 dark:bg-amber-950/40 dark:text-amber-200 dark:border-amber-800",
    activeColor: "ring-2 ring-amber-500 border-amber-500 bg-amber-100 dark:bg-amber-950/70",
    badgeColor: "bg-amber-500 text-white",
  },
  {
    key: "RELABEL_DATA",
    label: "Sửa nhãn dữ liệu",
    sublabel: "Chuyển sang hàng đợi chỉnh sửa nhãn",
    icon: Tag,
    color: "border-purple-300 bg-purple-50/80 text-purple-900 hover:bg-purple-100/80 dark:bg-purple-950/40 dark:text-purple-200 dark:border-purple-800",
    activeColor: "ring-2 ring-purple-500 border-purple-500 bg-purple-100 dark:bg-purple-950/70",
    badgeColor: "bg-purple-500 text-white",
  },
  {
    key: "ACCEPT_RISK",
    label: "Chấp nhận rủi ro",
    sublabel: "Ghi nhận audit trail và cho phép tiếp tục",
    icon: CheckCircle2,
    color: "border-emerald-300 bg-emerald-50/80 text-emerald-900 hover:bg-emerald-100/80 dark:bg-emerald-950/40 dark:text-emerald-200 dark:border-emerald-800",
    activeColor: "ring-2 ring-emerald-500 border-emerald-500 bg-emerald-100 dark:bg-emerald-950/70",
    badgeColor: "bg-emerald-500 text-white",
  },
];

export default function ReviewPage() {
  const { t } = useLanguage();
  const [activeStatusTab, setActiveStatusTab] = useState("PENDING");
  const [viewMode, setViewMode] = useState("clusters"); // "clusters" | "flat"

  const [reviews, setReviews] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [selectedReview, setSelectedReview] = useState(null);
  const [reviewSamples, setReviewSamples] = useState([]);
  const [riskAssessment, setRiskAssessment] = useState(null);
  const [riskSummary, setRiskSummary] = useState(null);

  // Tagging
  const [tagList, setTagList] = useState(DEFAULT_PRESET_TAGS);
  const [selectedTag, setSelectedTag] = useState("🌫️ Sương mù & Khói");
  const [newTagInput, setNewTagInput] = useState("");

  // Decision state
  const [decision, setDecision] = useState("");
  const [decisionNote, setDecisionNote] = useState("");
  const [applyToEntireCluster, setApplyToEntireCluster] = useState(false);

  // Loading & feedback states
  const [isLoading, setIsLoading] = useState(true);
  const [isClustering, setIsClustering] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isAutoScanning, setIsAutoScanning] = useState(false);
  const [actionSuccessMsg, setActionSuccessMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  // Load reviews list
  const loadReviewsData = useCallback(async (status = activeStatusTab) => {
    setIsLoading(true);
    setErrorMsg("");
    try {
      const [fetchedReviews, summaryData] = await Promise.all([
        getReviews({ status }),
        getRiskSessionSummary({ status }).catch(() => null),
      ]);

      const reviewList = fetchedReviews || [];
      setReviews(reviewList);
      setRiskSummary(summaryData);

      if (reviewList.length > 0) {
        setSelectedReview((prev) => {
          const exists = reviewList.find((r) => r.review_id === prev?.review_id);
          return exists || reviewList[0];
        });
      } else {
        setSelectedReview(null);
      }
    } catch (err) {
      console.error("Failed to load reviews:", err);
      setErrorMsg("Không thể tải danh sách review queue.");
      setReviews([]);
    } finally {
      setIsLoading(false);
    }
  }, [activeStatusTab]);

  // Load existing clusters
  const loadClustersData = useCallback(async () => {
    try {
      const existingClusters = await getFailureClusters();
      if (Array.isArray(existingClusters) && existingClusters.length > 0) {
        setClusters(existingClusters);
        setSelectedCluster(existingClusters[0]);
      } else {
        setClusters([]);
        setSelectedCluster(null);
      }
    } catch (err) {
      console.warn("Could not load failure clusters:", err);
      setClusters([]);
    }
  }, []);

  useEffect(() => {
    loadReviewsData(activeStatusTab);
    loadClustersData();
  }, [activeStatusTab, loadReviewsData, loadClustersData]);

  // Load samples and risk assessment for selected review
  useEffect(() => {
    if (!selectedReview?.run_id) {
      setReviewSamples([]);
      setRiskAssessment(null);
      return;
    }

    let ignore = false;

    // Load sample images
    getRunSamples(selectedReview.run_id)
      .then((samples) => {
        if (ignore) return;
        const apiBase = getApiBase();
        const artifactUrl = (value) => (value ? new URL(value, `${apiBase.replace(/\/$/, "")}/`).href : null);
        setReviewSamples(
          (samples || []).map((s) => ({
            ...s,
            clean_overlay: artifactUrl(s.artifacts?.clean_prediction_url),
            attacked_overlay: artifactUrl(s.artifacts?.attacked_prediction_url),
          }))
        );
      })
      .catch(() => {
        if (!ignore) setReviewSamples([]);
      });

    // Assess risk via API
    assessReviewRisk(selectedReview.review_id)
      .then((assessment) => {
        if (ignore) return;
        setRiskAssessment(assessment);
        if (assessment?.recommended_decision && !decision) {
          setDecision(assessment.recommended_decision);
          setDecisionNote(assessment.justification || "");
        }
      })
      .catch(() => {
        if (!ignore) setRiskAssessment(null);
      });

    // Auto-match tag
    const atk = (selectedReview.attack || "").toLowerCase();
    if (atk.includes("fog")) setSelectedTag("🌫️ Sương mù & Khói");
    else if (atk.includes("rain") || atk.includes("snow")) setSelectedTag("🌧️ Mưa & Tuyết rơi");
    else if (atk.includes("noise") || atk.includes("blur")) setSelectedTag("⚡ Nhiễu cảm biến ISO");
    else if (atk.includes("contrast") || atk.includes("brightness")) setSelectedTag("🌓 Mất tương phản");
    else if (atk.includes("fgsm") || atk.includes("pgd") || atk.includes("cw") || atk.includes("tog")) setSelectedTag("🛡️ Tấn công Gradient");
    else if (atk.includes("occlusion") || atk.includes("erasing")) setSelectedTag("🎯 Vật thể bị che khuất");

    return () => {
      ignore = true;
    };
  }, [selectedReview]);

  // Handle manual auto-grouping trigger
  const handleTriggerAutoGroup = async () => {
    setIsClustering(true);
    try {
      const grouped = await autoGroupFailureClusters();
      if (Array.isArray(grouped) && grouped.length > 0) {
        setClusters(grouped);
        setSelectedCluster(grouped[0]);
        // Select the representative review of the first cluster
        const repId = grouped[0].representative_review_id;
        const repReview = reviews.find((r) => r.review_id === repId);
        if (repReview) setSelectedReview(repReview);
        setActionSuccessMsg(`✨ Đã gom thành công ${grouped.length} cụm lỗi thông minh!`);
      }
    } catch (err) {
      setErrorMsg(`Lỗi gom cụm: ${err.message}`);
    } finally {
      setIsClustering(false);
    }
  };

  // Auto scan DB for degraded runs
  const handleAutoScanDB = async () => {
    setIsAutoScanning(true);
    setActionSuccessMsg("");
    try {
      const apiBase = getApiBase();
      const runsRes = await fetch(`${apiBase}/api/v1/runs?limit=10`);
      const runs = await runsRes.json();
      let totalCreated = 0;

      if (Array.isArray(runs) && runs.length > 0) {
        for (const r of runs.slice(0, 5)) {
          try {
            const res = await triggerAutoFlag(r.run_id || r.id, 20);
            totalCreated += res?.count || (Array.isArray(res) ? res.length : 0);
          } catch {}
        }
      }

      setActionSuccessMsg(`Đã quét DB và nạp ${totalCreated} ca lỗi mới vào hàng đợi.`);
      await loadReviewsData("PENDING");
      await loadClustersData();
    } catch (err) {
      setErrorMsg(`Lỗi quét DB: ${err.message}`);
    } finally {
      setIsAutoScanning(false);
    }
  };

  // Submit review resolution
  const handleSubmitDecision = async () => {
    if (!decision || !selectedReview) return;
    setIsSubmitting(true);
    setErrorMsg("");

    try {
      const batchClusterId = applyToEntireCluster && selectedCluster ? selectedCluster.cluster_id : null;
      const noteContent = `${decisionNote.trim()}${selectedTag ? ` [Tag: ${selectedTag}]` : ""}`;

      const updated = await resolveReview(
        selectedReview.review_id,
        decision,
        noteContent,
        "Staff Reviewer",
        batchClusterId
      );

      // Remove resolved review(s) from local list
      if (batchClusterId && selectedCluster) {
        const memberSet = new Set(selectedCluster.member_ids || selectedCluster.member_review_ids || []);
        setReviews((prev) => prev.filter((r) => !memberSet.has(r.review_id)));
        setClusters((prev) => prev.filter((c) => c.cluster_id !== batchClusterId));
        setSelectedCluster(null);
      } else {
        setReviews((prev) => prev.filter((r) => r.review_id !== updated.review_id));
      }

      setSelectedReview(null);
      setDecision("");
      setDecisionNote("");
      setApplyToEntireCluster(false);
      setActionSuccessMsg("✅ Đã lưu quyết định và kích hoạt hành động downstream thành công!");
    } catch (err) {
      setErrorMsg(`Lỗi thẩm định: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Add custom tag
  const handleAddNewTag = () => {
    const trimmed = newTagInput.trim();
    if (!trimmed) return;
    if (!tagList.includes(trimmed)) {
      setTagList([...tagList, trimmed]);
    }
    setSelectedTag(trimmed);
    setNewTagInput("");
  };

  const currentSample = reviewSamples[0];

  return (
    <div className="flex flex-col h-screen max-h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* ── Top Navigation Bar ── */}
      <header className="h-14 min-h-14 px-5 flex items-center justify-between border-b border-slate-800/80 bg-slate-900/90 backdrop-blur-md shrink-0">
        <div className="flex items-center gap-3">
          <Link
            href="/dashboard"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-sky-400 bg-sky-500/10 border border-sky-500/20 hover:bg-sky-500/20 transition-colors"
          >
            ← Tổng quan
          </Link>
          <div className="flex items-center gap-2">
            <span className="font-bold text-sm text-amber-400 flex items-center gap-1.5">
              <ShieldAlert className="w-4 h-4 text-amber-400" />
              Thẩm Định & Review Queue (HITL)
            </span>
            <span className="text-[11px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700 font-mono">
              Phễu 5 Tầng
            </span>
          </div>
        </div>

        {/* Global Risk Summary Chips */}
        {riskSummary && (
          <div className="hidden lg:flex items-center gap-2 text-xs">
            <span className="text-slate-400 text-[11px]">Tổng quan rủi ro:</span>
            {riskSummary.critical_count > 0 && (
              <span className="px-2 py-0.5 rounded-full bg-red-950/80 text-red-400 border border-red-800/60 font-semibold font-mono text-[11px]">
                {riskSummary.critical_count} CRITICAL
              </span>
            )}
            {riskSummary.high_count > 0 && (
              <span className="px-2 py-0.5 rounded-full bg-orange-950/80 text-orange-400 border border-orange-800/60 font-semibold font-mono text-[11px]">
                {riskSummary.high_count} HIGH
              </span>
            )}
            {riskSummary.medium_count > 0 && (
              <span className="px-2 py-0.5 rounded-full bg-amber-950/80 text-amber-400 border border-amber-800/60 font-semibold font-mono text-[11px]">
                {riskSummary.medium_count} MEDIUM
              </span>
            )}
          </div>
        )}

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleAutoScanDB}
            disabled={isAutoScanning}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-amber-300 bg-amber-500/10 border border-amber-500/30 hover:bg-amber-500/20 disabled:opacity-50 transition-colors"
          >
            {isAutoScanning ? (
              <><RefreshCw className="w-3.5 h-3.5 animate-spin" /> Đang quét DB...</>
            ) : (
              <><Sparkles className="w-3.5 h-3.5" /> Quét ca lỗi từ DB</>
            )}
          </button>
          <UserMenu />
        </div>
      </header>

      {/* ── Status Banner Messages ── */}
      {actionSuccessMsg && (
        <div className="px-5 py-2 bg-emerald-950/70 border-b border-emerald-800/60 text-emerald-300 text-xs flex items-center justify-between animate-fadeIn shrink-0">
          <span>{actionSuccessMsg}</span>
          <button type="button" onClick={() => setActionSuccessMsg("")} className="text-emerald-400 hover:text-emerald-200">✕</button>
        </div>
      )}
      {errorMsg && (
        <div className="px-5 py-2 bg-red-950/70 border-b border-red-800/60 text-red-300 text-xs flex items-center justify-between animate-fadeIn shrink-0">
          <span>{errorMsg}</span>
          <button type="button" onClick={() => setErrorMsg("")} className="text-red-400 hover:text-red-200">✕</button>
        </div>
      )}

      {/* ── 3-Column Triage Console ── */}
      <div className="flex flex-1 overflow-hidden">
        {/* ════ COLUMN 1: FAILURE CLUSTER & CASE NAVIGATOR (w-80) ════ */}
        <div className="w-80 shrink-0 border-r border-slate-800/80 bg-slate-900/50 flex flex-col overflow-hidden">
          {/* Status & View Switcher */}
          <div className="p-3 border-b border-slate-800/80 space-y-2.5 shrink-0">
            <div className="flex items-center justify-between">
              {/* Pending vs Resolved switcher */}
              <div className="inline-flex rounded-lg bg-slate-800/80 p-0.5 border border-slate-700/60">
                <button
                  type="button"
                  onClick={() => setActiveStatusTab("PENDING")}
                  className={cn(
                    "px-3 py-1 text-xs font-semibold rounded-md transition-colors",
                    activeStatusTab === "PENDING"
                      ? "bg-amber-500/20 text-amber-300 shadow-xs"
                      : "text-slate-400 hover:text-slate-200"
                  )}
                >
                  Chờ duyệt ({reviews.length})
                </button>
                <button
                  type="button"
                  onClick={() => setActiveStatusTab("RESOLVED")}
                  className={cn(
                    "px-3 py-1 text-xs font-semibold rounded-md transition-colors",
                    activeStatusTab === "RESOLVED"
                      ? "bg-emerald-500/20 text-emerald-300 shadow-xs"
                      : "text-slate-400 hover:text-slate-200"
                  )}
                >
                  Đã duyệt
                </button>
              </div>

              {/* Cluster vs Flat view toggle */}
              <button
                type="button"
                onClick={() => setViewMode((m) => (m === "clusters" ? "flat" : "clusters"))}
                className="flex items-center gap-1 text-[11px] font-medium text-slate-400 hover:text-slate-200 px-2 py-1 rounded bg-slate-800/60 border border-slate-700/50"
                title="Chuyển đổi giữa chế độ Gom Cụm và Danh sách Phẳng"
              >
                {viewMode === "clusters" ? (
                  <><Layers className="w-3 h-3 text-indigo-400" /> Theo Cụm</>
                ) : (
                  <><ListFilter className="w-3 h-3 text-sky-400" /> Danh Sách</>
                )}
              </button>
            </div>

            {/* Smart Auto-Group Trigger */}
            {viewMode === "clusters" && activeStatusTab === "PENDING" && (
              <button
                type="button"
                onClick={handleTriggerAutoGroup}
                disabled={isClustering || reviews.length === 0}
                className="w-full py-1.5 px-2.5 rounded-lg text-xs font-semibold text-indigo-300 bg-indigo-500/10 border border-indigo-500/25 hover:bg-indigo-500/20 disabled:opacity-50 flex items-center justify-center gap-1.5 transition-colors"
              >
                {isClustering ? (
                  <><RefreshCw className="w-3.5 h-3.5 animate-spin" /> Đang gom cụm thông minh...</>
                ) : (
                  <><Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Tự động gom cụm (Smart Cluster)</>
                )}
              </button>
            )}
          </div>

          {/* List Content */}
          <div className="flex-1 overflow-y-auto p-2 space-y-1.5 divide-y divide-slate-800/40">
            {isLoading ? (
              <div className="p-8 text-center text-slate-500 text-xs">
                <RefreshCw className="w-5 h-5 animate-spin mx-auto mb-2 text-slate-400" />
                Đang tải dữ liệu ca lỗi...
              </div>
            ) : viewMode === "clusters" && clusters.length > 0 ? (
              /* Cluster Cards */
              clusters.map((cluster) => {
                const isSelected = selectedCluster?.cluster_id === cluster.cluster_id;
                return (
                  <div
                    key={cluster.cluster_id || cluster.id}
                    onClick={() => {
                      setSelectedCluster(cluster);
                      // Select representative review
                      const rep = reviews.find((r) => r.review_id === cluster.representative_review_id) || reviews[0];
                      if (rep) setSelectedReview(rep);
                    }}
                    className={cn(
                      "p-3 rounded-xl cursor-pointer border transition-all text-left space-y-1.5",
                      isSelected
                        ? "bg-slate-800/90 border-indigo-500/60 ring-1 ring-indigo-500/40 shadow-sm"
                        : "bg-slate-900/60 border-slate-800 hover:bg-slate-800/50 hover:border-slate-700"
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <h4 className="font-semibold text-xs text-slate-200 line-clamp-2 leading-snug">
                        {cluster.name || cluster.label || "Cụm lỗi"}
                      </h4>
                      <span className="shrink-0 px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-indigo-950 text-indigo-300 border border-indigo-800/60">
                        {cluster.member_count || (cluster.member_ids?.length ?? 1)} ca
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-[11px] pt-1">
                      <RiskBadge level={cluster.dominant_risk_level || "MEDIUM"} size="sm" />
                      <span className="font-mono text-rose-400 text-[11px]">
                        ↓{Number(cluster.mean_degradation || cluster.degradation || 35).toFixed(1)}% mAP
                      </span>
                    </div>
                  </div>
                );
              })
            ) : reviews.length > 0 ? (
              /* Flat Review Cards */
              reviews.map((rev) => {
                const isSelected = selectedReview?.review_id === rev.review_id;
                const dropPct = rev.degradation_percent ?? (rev.degradation ? rev.degradation * 100 : 0);

                return (
                  <div
                    key={rev.review_id}
                    onClick={() => setSelectedReview(rev)}
                    className={cn(
                      "p-2.5 rounded-xl cursor-pointer border transition-all text-left space-y-1",
                      isSelected
                        ? "bg-slate-800/90 border-amber-500/60 ring-1 ring-amber-500/30"
                        : "bg-slate-900/50 border-slate-800/70 hover:bg-slate-800/40 hover:border-slate-700"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[11px] text-slate-400 font-semibold truncate max-w-[120px]">
                        {rev.review_id}
                      </span>
                      <RiskBadge level={rev.risk_level || (dropPct >= 50 ? "HIGH" : "MEDIUM")} size="sm" />
                    </div>
                    <div className="flex items-center justify-between text-xs pt-0.5">
                      <span className="font-semibold text-slate-200 truncate max-w-[160px]">
                        {rev.attack} <span className="text-slate-400 font-normal">Cấp {rev.severity}</span>
                      </span>
                      <span className="font-mono text-rose-400 font-bold text-[11px]">
                        ↓{dropPct.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="p-8 text-center text-slate-500 text-xs">
                Hàng đợi rỗng. Chưa có ca lỗi nào cần duyệt.
              </div>
            )}
          </div>
        </div>

        {/* ════ COLUMN 2: XAI VISUAL EVIDENCE & CONTEXT (flex-1) ════ */}
        <div className="flex-1 border-r border-slate-800/80 bg-slate-950 flex flex-col overflow-y-auto">
          {selectedReview ? (
            <div className="p-6 space-y-5">
              {/* Evidence Header */}
              <div className="flex items-start justify-between pb-4 border-b border-slate-800/80">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                      <span>{selectedReview.attack}</span>
                      <span className="px-2 py-0.5 rounded-md text-xs font-semibold bg-slate-800 text-slate-300 border border-slate-700">
                        Cấp {selectedReview.severity}
                      </span>
                    </h2>
                    <RiskBadge level={selectedReview.risk_level || "MEDIUM"} size="md" />
                  </div>
                  <p className="text-xs text-slate-400 mt-1 font-mono">
                    ID: {selectedReview.review_id} · Model: {selectedReview.model || "YOLO11s"} · Dataset: {selectedReview.dataset || "KITTI"}
                  </p>
                </div>

                <div className="text-right">
                  <span className="text-[11px] text-slate-400 block uppercase tracking-wider">Mức độ sụt giảm</span>
                  <span className="text-xl font-mono font-extrabold text-rose-400">
                    ↓{Number(selectedReview.degradation_percent ?? (selectedReview.degradation ? selectedReview.degradation * 100 : 0)).toFixed(1)}% mAP
                  </span>
                </div>
              </div>

              {/* ── System Risk Assessment Card (Intelligence Layer) ── */}
              {riskAssessment && (
                <div className="p-4 rounded-xl border border-indigo-900/60 bg-gradient-to-br from-indigo-950/40 via-slate-900/60 to-slate-900/80 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-xs text-indigo-300 flex items-center gap-1.5 uppercase tracking-wide">
                      <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                      Phân tích rủi ro & Khuyến nghị hệ thống
                    </h3>
                    <span className="text-[11px] px-2 py-0.5 rounded-full bg-indigo-900/60 text-indigo-200 border border-indigo-700/50 font-medium">
                      Độ tin cậy {(riskAssessment.confidence * 100).toFixed(0)}%
                    </span>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed font-sans">
                    {riskAssessment.justification}
                  </p>

                  {/* Contributing factors */}
                  {riskAssessment.contributing_factors?.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {riskAssessment.contributing_factors.map((factor, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 rounded-md text-[11px] bg-slate-800/80 text-slate-300 border border-slate-700/60"
                        >
                          • {factor}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Downstream chain preview */}
                  {riskAssessment.downstream_actions?.length > 0 && (
                    <div className="pt-2 border-t border-indigo-900/40 text-[11px] text-indigo-200/90 flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-slate-400">Chuỗi hành động tiếp theo:</span>
                      {riskAssessment.downstream_actions.map((act, idx) => (
                        <span key={idx} className="flex items-center gap-1">
                          <span className="text-indigo-300 font-medium">{act}</span>
                          {idx < riskAssessment.downstream_actions.length - 1 && (
                            <ChevronRight className="w-3 h-3 text-slate-600" />
                          )}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* ── Side-by-Side Visual Prediction Viewport ── */}
              <div className="space-y-2">
                <h3 className="font-semibold text-xs text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <Eye className="w-3.5 h-3.5 text-sky-400" />
                  Đối chiếu ảnh trước & sau tấn công
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Clean Prediction */}
                  <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-3 space-y-2">
                    <div className="flex items-center justify-between text-xs font-semibold text-emerald-400">
                      <span>1. Clean Model Prediction</span>
                      <span className="text-[10px] text-slate-400 font-mono">Chuẩn Baseline</span>
                    </div>
                    <div className="relative aspect-video rounded-lg overflow-hidden bg-slate-950 border border-slate-800/80 flex items-center justify-center">
                      {currentSample?.clean_overlay ? (
                        <img
                          src={currentSample.clean_overlay}
                          alt="Clean Prediction"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <span className="text-slate-600 text-xs">Clean Model Prediction</span>
                      )}
                    </div>
                  </div>

                  {/* Attacked Prediction */}
                  <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-3 space-y-2">
                    <div className="flex items-center justify-between text-xs font-semibold text-rose-400">
                      <span>2. Attacked Model Prediction</span>
                      <span className="text-[10px] text-rose-300 font-mono">Phát hiện Lỗi / Mất BBox</span>
                    </div>
                    <div className="relative aspect-video rounded-lg overflow-hidden bg-slate-950 border border-slate-800/80 flex items-center justify-center">
                      {currentSample?.attacked_overlay ? (
                        <img
                          src={currentSample.attacked_overlay}
                          alt="Attacked Prediction"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <span className="text-slate-600 text-xs">Attacked Model Prediction</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* ── Failure Category Tagging ── */}
              <div className="space-y-2 pt-2 border-t border-slate-800/80">
                <h3 className="font-semibold text-xs text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <Tag className="w-3.5 h-3.5 text-amber-400" />
                  Gán Nhóm Lỗi (Failure Category Tag)
                </h3>

                <div className="flex flex-wrap gap-2">
                  {tagList.map((tag) => (
                    <button
                      key={tag}
                      type="button"
                      onClick={() => setSelectedTag(tag)}
                      className={cn(
                        "px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors",
                        selectedTag === tag
                          ? "bg-amber-500/20 text-amber-300 border-amber-500/50 shadow-xs"
                          : "bg-slate-900 text-slate-400 border-slate-800 hover:border-slate-700 hover:text-slate-200"
                      )}
                    >
                      {tag}
                    </button>
                  ))}
                </div>

                <div className="flex items-center gap-2 pt-1">
                  <input
                    type="text"
                    value={newTagInput}
                    onChange={(e) => setNewTagInput(e.target.value)}
                    placeholder="Nhập nhóm lỗi tùy chỉnh khác..."
                    className="flex-1 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-amber-500/50"
                  />
                  <button
                    type="button"
                    onClick={handleAddNewTag}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700"
                  >
                    + Thêm tag
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-16 text-center text-slate-500 space-y-2 my-auto">
              <ShieldAlert className="w-10 h-10 mx-auto text-slate-600 mb-3" />
              <h3 className="font-bold text-sm text-slate-300">Chưa chọn ca lỗi nào</h3>
              <p className="text-xs max-w-sm mx-auto text-slate-500">
                Chọn một cụm lỗi hoặc ca lỗi từ cột bên trái để kiểm tra đối chiếu visual và ra quyết định.
              </p>
            </div>
          )}
        </div>

        {/* ════ COLUMN 3: 5-DECISION RUBRIC & DOWNSTREAM ACTION (w-88) ════ */}
        <div className="w-88 shrink-0 bg-slate-900/70 p-5 flex flex-col justify-between overflow-y-auto space-y-5">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="font-bold text-sm text-slate-100 flex items-center gap-1.5">
                <FileText className="w-4 h-4 text-amber-400" />
                Quyết Định Thẩm Định (HITL)
              </h3>
              <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                5-Option Rubric
              </span>
            </div>

            {/* Decision Cards */}
            <div className="space-y-2">
              {DECISION_OPTIONS.map((opt) => {
                const Icon = opt.icon;
                const isChosen = decision === opt.key;
                return (
                  <button
                    key={opt.key}
                    type="button"
                    onClick={() => {
                      setDecision(opt.key);
                      if (!decisionNote || decisionNote.startsWith("Chấp nhận") || decisionNote.startsWith("Yêu cầu")) {
                        setDecisionNote(
                          opt.key === "BLOCK_DEPLOY"
                            ? `Chặn triển khai do suy giảm nghiêm trọng (-${Number(selectedReview?.degradation_percent || 30).toFixed(1)}% mAP).`
                            : opt.key === "REQUEST_RETRAIN"
                            ? `Yêu cầu tôi luyện đối kháng cho nhóm ${selectedReview?.attack || "attack"}.`
                            : opt.key === "RESTRICT_ODD"
                            ? `Giới hạn vận hành khi ${selectedReview?.attack || "nhiễu"} vượt cấp ${selectedReview?.severity || 3}.`
                            : opt.key === "RELABEL_DATA"
                            ? "Chuyển sang hàng đợi chỉnh sửa nhãn dữ liệu."
                            : "Chấp nhận mức suy giảm hiệu năng có giải trình."
                        );
                      }
                    }}
                    className={cn(
                      "w-full p-2.5 rounded-xl border text-left flex items-start gap-3 transition-all",
                      isChosen
                        ? opt.activeColor
                        : "border-slate-800/80 bg-slate-900/60 hover:bg-slate-800/50 hover:border-slate-700 text-slate-300"
                    )}
                  >
                    <div className={cn("p-1.5 rounded-lg shrink-0 mt-0.5", isChosen ? opt.badgeColor : "bg-slate-800 text-slate-400")}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-xs text-slate-100 flex items-center justify-between">
                        <span>{opt.label}</span>
                        {isChosen && <span className="text-[10px] font-bold text-amber-400">✓ Đã chọn</span>}
                      </div>
                      <p className="text-[11px] text-slate-400 line-clamp-1 mt-0.5">{opt.sublabel}</p>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Justification Note Textarea */}
            <div className="space-y-1.5">
              <label className="block text-xs font-semibold text-slate-300">
                Ghi chú giải trình (Justification):
              </label>
              <textarea
                rows={3}
                value={decisionNote}
                onChange={(e) => setDecisionNote(e.target.value)}
                placeholder="Nhập lý do thẩm định hoặc giải trình an toàn..."
                className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-amber-500/50 resize-none font-sans"
              />
            </div>

            {/* Batch Cluster Resolve Checkbox */}
            {selectedCluster && (
              <label className="flex items-center gap-2.5 p-2.5 rounded-xl bg-indigo-950/30 border border-indigo-900/50 cursor-pointer">
                <input
                  type="checkbox"
                  checked={applyToEntireCluster}
                  onChange={(e) => setApplyToEntireCluster(e.target.checked)}
                  className="rounded text-indigo-600 focus:ring-indigo-500 w-4 h-4"
                />
                <div className="text-xs">
                  <span className="font-semibold text-indigo-300">Áp dụng cho toàn bộ cụm</span>
                  <span className="block text-[11px] text-slate-400">
                    Resolve cùng lúc {selectedCluster.member_count || selectedCluster.member_ids?.length || 1} ca tương tự
                  </span>
                </div>
              </label>
            )}
          </div>

          {/* Action Footer */}
          <div className="space-y-3 pt-4 border-t border-slate-800">
            <button
              type="button"
              onClick={handleSubmitDecision}
              disabled={!decision || !selectedReview || isSubmitting}
              className="w-full py-2.5 px-4 rounded-xl font-bold text-xs bg-amber-500 hover:bg-amber-400 text-slate-950 disabled:opacity-40 disabled:cursor-not-allowed shadow-md transition-all flex items-center justify-center gap-2"
            >
              {isSubmitting ? (
                <><RefreshCw className="w-4 h-4 animate-spin" /> Đang lưu quyết định...</>
              ) : (
                <><CheckCircle2 className="w-4 h-4" /> Phê duyệt & Kích hoạt Downstream</>
              )}
            </button>

            {/* Downstream Quick Links */}
            <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1">
              <Link href="/defense" className="hover:text-amber-400 flex items-center gap-1">
                Phòng thủ & Retrain <ExternalLink className="w-3 h-3" />
              </Link>
              <Link href="/analysis" className="hover:text-amber-400 flex items-center gap-1">
                Báo cáo phân tích <ExternalLink className="w-3 h-3" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
