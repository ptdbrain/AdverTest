"use client";

import React, { useState, useEffect, useCallback } from "react";
import Image from "next/image";
import Link from "next/link";
import UserMenu from "@/components/UserMenu.jsx";
import { useLanguage } from "@/context/LanguageContext";
import {
  getReviews,
  resolveReview,
  getRunSamples,
  getFailureClusters,
  createFailureCluster,
  triggerAutoFlag,
  createRetrainingBacklog,
  addRetrainingBacklogItem,
  getApiBase,
} from "@/lib/api";

const DEFAULT_PRESET_TAGS = [
  "🌫️ Sương mù & Khói",
  "🌧️ Mưa & Tuyết rơi",
  "⚡ Nhiễu cảm biến ISO",
  "🌓 Mất tương phản",
  "🛡️ Tấn công Gradient",
  "🎯 Vật thể bị che khuất",
];

export default function ReviewPage() {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState("PENDING");
  const [reviews, setReviews] = useState(null);
  const [selectedReview, setSelectedReview] = useState(null);
  const [reviewSamples, setReviewSamples] = useState([]);
  
  // Custom & Preset failure tags list
  const [tagList, setTagList] = useState(DEFAULT_PRESET_TAGS);
  const [selectedTag, setSelectedTag] = useState("🌫️ Sương mù & Khói");
  const [newTagInput, setNewTagInput] = useState("");

  const [decision, setDecision] = useState("");
  const [decisionNote, setDecisionNote] = useState("");

  const [isAutoScanning, setIsAutoScanning] = useState(false);
  const [actionSuccessMsg, setActionSuccessMsg] = useState("");
  const [error, setError] = useState("");

  const loading = reviews === null;

  // Load saved custom tags from localStorage + Backend
  const loadSavedTags = useCallback(() => {
    let savedLocal = [];
    try {
      savedLocal = JSON.parse(localStorage.getItem("advertest_custom_tags") || "[]");
    } catch {}

    getFailureClusters()
      .then((cls) => {
        const backendNames = (cls || []).map((c) => c.name || c.id).filter(Boolean);
        const merged = Array.from(new Set([...DEFAULT_PRESET_TAGS, ...savedLocal, ...backendNames]));
        setTagList(merged);
      })
      .catch(() => {
        const merged = Array.from(new Set([...DEFAULT_PRESET_TAGS, ...savedLocal]));
        setTagList(merged);
      });
  }, []);

  // Load reviews list
  const loadReviews = useCallback((status = activeTab) => {
    setReviews(null);
    setError("");
    getReviews({ status })
      .then((data) => {
        setReviews(data || []);
        if (data && data.length > 0) {
          setSelectedReview((prev) => {
            const exists = data.find((r) => r.review_id === prev?.review_id);
            return exists || data[0];
          });
        } else {
          setSelectedReview(null);
        }
      })
      .catch((err) => {
        console.error("Failed to load reviews:", err);
        setReviews([]);
        setError(t("review.loadFailed"));
      });
  }, [activeTab]);

  useEffect(() => {
    loadReviews(activeTab);
    loadSavedTags();
  }, [activeTab, loadReviews, loadSavedTags]);

  // Load samples for selected review
  useEffect(() => {
    if (!selectedReview?.run_id) {
      setReviewSamples([]);
      return;
    }
    let ignore = false;
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

    // Auto-match simple tag based on attack name
    const atk = (selectedReview.attack || "").toLowerCase();
    if (atk.includes("fog")) setSelectedTag("🌫️ Sương mù & Khói");
    else if (atk.includes("rain") || atk.includes("snow")) setSelectedTag("🌧️ Mưa & Tuyết rơi");
    else if (atk.includes("noise") || atk.includes("blur")) setSelectedTag("⚡ Nhiễu cảm biến ISO");
    else if (atk.includes("contrast") || atk.includes("brightness")) setSelectedTag("🌓 Mất tương phản");
    else if (atk.includes("fgsm") || atk.includes("pgd") || atk.includes("cw") || atk.includes("tog")) setSelectedTag("🛡️ Tấn công Đối kháng");
    else if (atk.includes("occlusion") || atk.includes("erasing")) setSelectedTag("🎯 Vật thể bị che khuất");

    return () => {
      ignore = true;
    };
  }, [selectedReview]);

  // Thêm một nhóm lỗi tùy chỉnh mới
  const handleAddNewTag = () => {
    const trimmed = newTagInput.trim();
    if (!trimmed) return;
    if (!tagList.includes(trimmed)) {
      const updated = [...tagList, trimmed];
      setTagList(updated);
      try {
        const saved = JSON.parse(localStorage.getItem("advertest_custom_tags") || "[]");
        localStorage.setItem("advertest_custom_tags", JSON.stringify([...saved, trimmed]));
      } catch {}
    }
    setSelectedTag(trimmed);
    setNewTagInput("");
    setActionSuccessMsg(t("review.tagAdded", { tag: trimmed }));
  };

  const handleSelectDecision = (chosenDecision) => {
    setDecision(chosenDecision);
    if (!decisionNote || decisionNote.startsWith("Yêu cầu") || decisionNote.startsWith("Chấp nhận") || decisionNote.startsWith("Loại bỏ")) {
      if (chosenDecision === "REQUEST_RETRAIN") {
        setDecisionNote(`Yêu cầu huấn luyện lại (-${Number(selectedReview?.degradation || 0).toFixed(1)}% mAP).`);
      } else if (chosenDecision === "ACCEPT_RISK") {
        setDecisionNote("Chấp nhận mức suy giảm hiệu năng.");
      } else if (chosenDecision === "REJECT_SAMPLE") {
        setDecisionNote("Loại bỏ mẫu dữ liệu không hợp lệ.");
      }
    }
  };

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
            totalCreated += res?.count || 0;
          } catch {}
        }
      }

      setActionSuccessMsg(t("review.scanned", { count: totalCreated }));
      loadReviews("PENDING");
      setActiveTab("PENDING");
    } catch (err) {
      setError(t("review.error", { message: err.message }));
    } finally {
      setIsAutoScanning(false);
    }
  };

  const handleAddToRetrainingBacklog = async () => {
    if (!selectedReview) return;
    try {
      const backlog = await createRetrainingBacklog(`Backlog-${selectedReview.attack}-${new Date().toLocaleDateString()}`);
      if (backlog?.id) {
        await addRetrainingBacklogItem(backlog.id, selectedReview.review_id);
      }
      setActionSuccessMsg(t("review.backlogAdded"));
    } catch (err) {
      alert(t("review.error", { message: err.message }));
    }
  };

  const handleSubmitDecision = async () => {
    if (!decision || !selectedReview) return;

    try {
      if (selectedTag) {
        try {
          await createFailureCluster(selectedTag, [selectedReview.review_id]);
        } catch {}
      }

      const noteText = decisionNote.trim() || decision;
      const noteContent = `${noteText}${selectedTag ? ` [Tag: ${selectedTag}]` : ""}`;

      const updated = await resolveReview(
        selectedReview.review_id,
        decision,
        noteContent,
        "Reviewer"
      );

      setReviews((prev) => prev.filter((r) => r.review_id !== updated.review_id));
      setSelectedReview(null);
      setDecision("");
      setDecisionNote("");
      setActionSuccessMsg(t("review.decisionSaved"));
    } catch (err) {
      alert(t("review.error", { message: err.message }));
    }
  };

  const currentSample = reviewSamples[0];

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        maxHeight: "100vh",
        background: "var(--bg-primary, #0B0F17)",
        color: "var(--text-primary, #E2E8F0)",
        overflow: "hidden",
      }}
    >
      {/* Header - Generous 56px height */}
      <header
        style={{
          height: "56px",
          minHeight: "56px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "0 20px",
          background: "var(--bg-elevated, #131B2A)",
          borderBottom: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.12))",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <Link
            href="/"
            style={{
              color: "var(--accent, #38BDF8)",
              textDecoration: "none",
              fontSize: "0.85rem",
              fontWeight: 700,
              padding: "6px 12px",
              borderRadius: "6px",
              background: "rgba(56, 189, 248, 0.1)",
              border: "1px solid rgba(56, 189, 248, 0.25)",
            }}
          >
            {t("review.backDashboard")}
          </Link>
          <span style={{ fontSize: "1.05rem", fontWeight: 800, color: "#FBBF24" }}>
            {t("review.title")}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <button
            type="button"
            onClick={handleAutoScanDB}
            disabled={isAutoScanning}
            className="action-button action-button--secondary"
            style={{ fontSize: "0.78rem", padding: "6px 14px", color: "#FBBF24", fontWeight: 700 }}
          >
            {isAutoScanning ? t("review.scanBtn.running") : t("review.scanBtn")}
          </button>
          <UserMenu />
        </div>
      </header>

      {/* Main Split Content */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "clamp(280px, 23vw, 340px) 1fr",
          flex: 1,
          minHeight: 0,
          overflow: "hidden",
        }}
      >
        {/* Left Sidebar */}
        <aside
          style={{
            borderRight: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.12))",
            display: "flex",
            flexDirection: "column",
            background: "var(--bg-elevated, #101623)",
            overflow: "hidden",
          }}
        >
          {/* Tabs */}
          <div style={{ display: "flex", borderBottom: "1px solid var(--border-subtle)", background: "var(--bg-primary)" }}>
            <button
              type="button"
              onClick={() => setActiveTab("PENDING")}
              style={{
                flex: 1,
                padding: "14px 8px",
                border: "none",
                background: activeTab === "PENDING" ? "var(--bg-elevated)" : "transparent",
                color: activeTab === "PENDING" ? "var(--accent)" : "var(--text-secondary)",
                fontWeight: 700,
                fontSize: "0.85rem",
                cursor: "pointer",
                borderBottom: activeTab === "PENDING" ? "2px solid var(--accent)" : "2px solid transparent",
              }}
            >
              {t("review.tab.pending", { count: reviews && activeTab === "PENDING" ? `(${reviews.length})` : "" })}
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("RESOLVED")}
              style={{
                flex: 1,
                padding: "14px 8px",
                border: "none",
                background: activeTab === "RESOLVED" ? "var(--bg-elevated)" : "transparent",
                color: activeTab === "RESOLVED" ? "var(--success)" : "var(--text-secondary)",
                fontWeight: 700,
                fontSize: "0.85rem",
                cursor: "pointer",
                borderBottom: activeTab === "RESOLVED" ? "2px solid var(--success)" : "2px solid transparent",
              }}
            >
              {t("review.tab.resolved", { count: reviews && activeTab === "RESOLVED" ? `(${reviews.length})` : "" })}
            </button>
          </div>

          {/* List */}
          <div style={{ flex: 1, overflowY: "auto", padding: "10px", display: "flex", flexDirection: "column", gap: "8px" }}>
            {loading ? (
              <div style={{ padding: "30px", textAlign: "center", color: "var(--text-muted)", fontSize: "0.82rem" }}>
                ⏳ {t("common.loading")}
              </div>
            ) : error ? (
              <div style={{ padding: "14px", color: "var(--danger)", fontSize: "0.78rem", textAlign: "center" }}>
                {error}
              </div>
            ) : reviews.length === 0 ? (
              <div style={{ padding: "30px 16px", textAlign: "center", color: "var(--text-muted)", fontSize: "0.82rem" }}>
                {t("review.emptyQueue")}
              </div>
            ) : (
              reviews.map((review) => {
                const isSelected = selectedReview?.review_id === review.review_id;
                return (
                  <div
                    key={review.review_id}
                    onClick={() => {
                      setSelectedReview(review);
                      setActionSuccessMsg("");
                    }}
                    style={{
                      padding: "12px 14px",
                      borderRadius: "6px",
                      background: isSelected ? "rgba(56, 189, 248, 0.14)" : "var(--bg-primary)",
                      border: isSelected ? "1px solid var(--accent)" : "1px solid var(--border-subtle)",
                      cursor: "pointer",
                      transition: "all 0.15s ease",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div>
                        <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                          {review.review_id}
                        </div>
                        <span style={{ fontWeight: 700, fontSize: "0.85rem", color: "var(--text-primary)" }}>
                          {t("review.levelInList", { attack: review.attack?.replace(/_/g, " "), severity: review.severity })}
                        </span>
                      </div>
                      <span style={{ color: "var(--danger)", fontWeight: 800, fontSize: "0.85rem", fontFamily: "var(--font-mono)" }}>
                        ↓{Number(review.degradation || 0).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>

        {/* Right Main Area */}
        <main
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "18px 20px",
            display: "flex",
            flexDirection: "column",
            gap: "16px",
            background: "var(--bg-primary, #0B0F17)",
          }}
        >
          {actionSuccessMsg && (
            <div
              style={{
                padding: "10px 16px",
                background: "rgba(16, 185, 129, 0.15)",
                color: "#10B981",
                borderRadius: "6px",
                fontSize: "0.82rem",
                fontWeight: 600,
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>{actionSuccessMsg}</span>
              <button
                type="button"
                onClick={() => setActionSuccessMsg("")}
                style={{ background: "none", border: "none", color: "#10B981", cursor: "pointer", fontWeight: 700 }}
              >
                ✕
              </button>
            </div>
          )}

          {!selectedReview ? (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", fontSize: "0.95rem" }}>
              {t("review.selectHint")}
            </div>
          ) : (
            <>
              {/* Header Info - Generous padding */}
              <div
                style={{
                  background: "var(--bg-elevated)",
                  padding: "14px 18px",
                  borderRadius: "8px",
                  border: "1px solid var(--border-subtle)",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <strong style={{ fontSize: "1.1rem", color: "var(--text-primary)" }}>
                    {t("review.levelHeader", { attack: selectedReview.attack?.toUpperCase(), severity: selectedReview.severity })}
                  </strong>
                  <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                    ID: {selectedReview.review_id}
                  </span>
                </div>
                <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--danger)", fontFamily: "var(--font-mono)" }}>
                  ↓ {Number(selectedReview.degradation || 0).toFixed(1)}% mAP
                </div>
              </div>

              {/* 2 LARGE EXPANDED SIDE-BY-SIDE MODEL PREDICTION IMAGES */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "16px",
                }}
              >
                {/* 1. Clean Model Prediction */}
                <div style={{ background: "var(--bg-elevated)", borderRadius: "8px", padding: "14px", border: "1px solid var(--border-subtle)" }}>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#38BDF8", marginBottom: "8px" }}>
                    {t("review.cleanPred")}
                  </div>
                  <div
                    style={{
                      position: "relative",
                      width: "100%",
                      minHeight: "360px",
                      height: "clamp(340px, 44vh, 520px)",
                      background: "#05080E",
                      borderRadius: "6px",
                      overflow: "hidden",
                    }}
                  >
                    {currentSample?.clean_overlay ? (
                      <Image src={currentSample.clean_overlay} alt="Clean Model Prediction" fill unoptimized style={{ objectFit: "contain" }} />
                    ) : (
                      <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", fontSize: "0.85rem" }}>
                        {t("review.cleanPredEmpty")}
                      </div>
                    )}
                  </div>
                </div>

                {/* 2. Attacked Model Prediction */}
                <div style={{ background: "var(--bg-elevated)", borderRadius: "8px", padding: "14px", border: "1px solid var(--border-subtle)" }}>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#F59E0B", marginBottom: "8px" }}>
                    {t("review.attackedPred")}
                  </div>
                  <div
                    style={{
                      position: "relative",
                      width: "100%",
                      minHeight: "360px",
                      height: "clamp(340px, 44vh, 520px)",
                      background: "#05080E",
                      borderRadius: "6px",
                      overflow: "hidden",
                    }}
                  >
                    {currentSample?.attacked_overlay ? (
                      <Image src={currentSample.attacked_overlay} alt="Attacked Model Prediction" fill unoptimized style={{ objectFit: "contain" }} />
                    ) : (
                      <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", fontSize: "0.85rem" }}>
                        {t("review.attackedPredEmpty")}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Action Area - Generous Height and Padding */}
              {selectedReview.status === "PENDING" ? (
                <div
                  style={{
                    background: "var(--bg-elevated)",
                    padding: "16px 18px",
                    borderRadius: "8px",
                    border: "1px solid var(--border-subtle)",
                    display: "flex",
                    flexDirection: "column",
                    gap: "14px",
                  }}
                >
                  {/* 1. Failure Category Tagging with Expanded Chips */}
                  <div>
                    <div style={{ fontSize: "0.85rem", fontWeight: 700, marginBottom: "8px", color: "var(--text-primary)" }}>
                      {t("review.failureTag")}
                    </div>
                    
                    {/* Chips suggestions - Taller & more comfortable */}
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginBottom: "10px" }}>
                      {tagList.map((tag) => {
                        const isSelected = selectedTag === tag;
                        return (
                          <button
                            key={tag}
                            type="button"
                            onClick={() => setSelectedTag(tag)}
                            style={{
                              padding: "6px 14px",
                              borderRadius: "6px",
                              fontSize: "0.8rem",
                              fontWeight: 600,
                              border: isSelected ? "1px solid var(--accent)" : "1px solid var(--border-subtle)",
                              background: isSelected ? "rgba(56, 189, 248, 0.16)" : "var(--bg-primary)",
                              color: isSelected ? "var(--accent)" : "var(--text-secondary)",
                              cursor: "pointer",
                              transition: "all 0.15s ease",
                            }}
                          >
                            {tag}
                          </button>
                        );
                      })}
                    </div>

                    {/* Inline custom tag addition - 38px Height */}
                    <div style={{ display: "flex", gap: "8px", maxWidth: "460px" }}>
                      <input
                        type="text"
                        className="select-field"
                        placeholder={t("review.tagPlaceholder")}
                        value={newTagInput}
                        onChange={(e) => setNewTagInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            handleAddNewTag();
                          }
                        }}
                        style={{ flex: 1, padding: "8px 12px", fontSize: "0.8rem", height: "38px" }}
                      />
                      <button
                        type="button"
                        onClick={handleAddNewTag}
                        className="action-button action-button--secondary"
                        style={{ fontSize: "0.8rem", padding: "8px 16px", height: "38px", whiteSpace: "nowrap", fontWeight: 700 }}
                      >
                        {t("review.add")}
                      </button>
                    </div>
                  </div>

                  {/* 2. Decision & Note - Taller Cards & Inputs */}
                  <div>
                    <div style={{ fontSize: "0.85rem", fontWeight: 700, marginBottom: "8px", color: "var(--text-primary)" }}>
                      {t("review.decisionStep")}
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "10px" }}>
                      <button
                        type="button"
                        onClick={() => handleSelectDecision("REQUEST_RETRAIN")}
                        className={`attack-card ${decision === "REQUEST_RETRAIN" ? "attack-card--selected" : ""}`}
                        style={{
                          padding: "14px 16px",
                          minHeight: "52px",
                          borderRadius: "8px",
                          borderColor: decision === "REQUEST_RETRAIN" ? "var(--danger)" : "var(--border-subtle)",
                          background: decision === "REQUEST_RETRAIN" ? "rgba(239, 68, 68, 0.15)" : "transparent",
                          cursor: "pointer",
                        }}
                      >
                        <div style={{ fontWeight: 700, color: "var(--danger)", fontSize: "0.88rem" }}>Request Retrain</div>
                      </button>

                      <button
                        type="button"
                        onClick={() => handleSelectDecision("ACCEPT_RISK")}
                        className={`attack-card ${decision === "ACCEPT_RISK" ? "attack-card--selected" : ""}`}
                        style={{
                          padding: "14px 16px",
                          minHeight: "52px",
                          borderRadius: "8px",
                          borderColor: decision === "ACCEPT_RISK" ? "var(--success)" : "var(--border-subtle)",
                          background: decision === "ACCEPT_RISK" ? "rgba(16, 185, 129, 0.15)" : "transparent",
                          cursor: "pointer",
                        }}
                      >
                        <div style={{ fontWeight: 700, color: "var(--success)", fontSize: "0.88rem" }}>Accept Risk</div>
                      </button>

                      <button
                        type="button"
                        onClick={() => handleSelectDecision("REJECT_SAMPLE")}
                        className={`attack-card ${decision === "REJECT_SAMPLE" ? "attack-card--selected" : ""}`}
                        style={{
                          padding: "14px 16px",
                          minHeight: "52px",
                          borderRadius: "8px",
                          borderColor: decision === "REJECT_SAMPLE" ? "var(--warning)" : "var(--border-subtle)",
                          background: decision === "REJECT_SAMPLE" ? "rgba(245, 158, 11, 0.15)" : "transparent",
                          cursor: "pointer",
                        }}
                      >
                        <div style={{ fontWeight: 700, color: "var(--warning)", fontSize: "0.88rem" }}>Reject Sample</div>
                      </button>
                    </div>

                    <div style={{ display: "flex", gap: "10px", marginTop: "12px" }}>
                      <input
                        type="text"
                        className="select-field"
                        placeholder={t("review.notePlaceholder")}
                        value={decisionNote}
                        onChange={(e) => setDecisionNote(e.target.value)}
                        style={{ flex: 1, padding: "10px 14px", fontSize: "0.82rem", height: "42px" }}
                      />
                      <button
                        type="button"
                        onClick={handleSubmitDecision}
                        disabled={!decision}
                        className="run-button"
                        style={{
                          padding: "0 24px",
                          height: "42px",
                          fontSize: "0.85rem",
                          fontWeight: 700,
                          whiteSpace: "nowrap",
                        }}
                      >
                        {t("review.approve")}
                      </button>
                    </div>
                  </div>

                  {/* Optional Retraining Backlog Push */}
                  <div style={{ display: "flex", justifyContent: "flex-end", paddingTop: "4px" }}>
                    <button
                      type="button"
                      onClick={handleAddToRetrainingBacklog}
                      style={{
                        background: "none",
                        border: "none",
                        color: "var(--accent)",
                        fontSize: "0.78rem",
                        cursor: "pointer",
                        fontWeight: 600,
                        padding: "4px 8px",
                      }}
                    >
                      {t("review.toBacklog")}
                    </button>
                  </div>
                </div>
              ) : (
                <div
                  style={{
                    background: "rgba(16, 185, 129, 0.1)",
                    border: "1px solid rgba(16, 185, 129, 0.3)",
                    borderRadius: "8px",
                    padding: "16px 20px",
                    fontSize: "0.85rem",
                  }}
                >
                  <strong style={{ color: "#10B981" }}>{t("review.resolved", { decision: selectedReview.decision })}</strong>
                  <div style={{ color: "var(--text-muted)", marginTop: "4px" }}>
                    {selectedReview.decision_note}
                  </div>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
