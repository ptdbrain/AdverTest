"use client";

import { useState, useEffect } from "react";
import { getReviews, resolveReview } from "@/lib/api";

export default function ReviewPage() {
  const [activeTab, setActiveTab] = useState("PENDING");
  const [reviews, setReviews] = useState([]);
  const [selectedReview, setSelectedReview] = useState(null);
  const [decision, setDecision] = useState("");
  const [decisionNote, setDecisionNote] = useState("");
  const [loading, setLoading] = useState(true);

  // Fetch reviews based on active tab
  useEffect(() => {
    let mounted = true;
    setLoading(true);
    getReviews({ status: activeTab })
      .then((data) => {
        if (mounted) {
          setReviews(data);
          // Select the first item if nothing is selected or if selected item isn't in this list
          if (data.length > 0) {
            setSelectedReview(data[0]);
          } else {
            setSelectedReview(null);
          }
        }
      })
      .catch((err) => console.error("Failed to load reviews:", err))
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [activeTab]);

  const handleSubmitDecision = async () => {
    if (!decision || !decisionNote || !selectedReview) return;
    
    try {
      const updated = await resolveReview(
        selectedReview.review_id,
        decision,
        decisionNote,
        "reviewer_admin" // Hardcoded for now
      );
      
      // Remove from current list since its status changed
      setReviews((prev) => prev.filter((r) => r.review_id !== updated.review_id));
      setSelectedReview(null);
      setDecision("");
      setDecisionNote("");
    } catch (err) {
      alert(`Failed to submit decision: ${err.message}`);
    }
  };

  return (
    <div className="main-content" style={{ gridTemplateColumns: "350px 1fr" }}>
      {/* LEFT - Queue List */}
      <aside className="config-panel" style={{ padding: 0 }}>
        <div style={{ padding: "var(--space-lg)", borderBottom: "1px solid var(--border-subtle)" }}>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700, margin: 0, color: "var(--text-primary)" }}>Review Queue</h2>
          <p className="text-xs text-secondary mt-1">Human-in-the-loop validation</p>
        </div>
        
        <div className="tab-bar" style={{ padding: "0 var(--space-md)" }}>
          <button 
            className={`tab-bar__item ${activeTab === "PENDING" ? "tab-bar__item--active" : ""}`}
            onClick={() => setActiveTab("PENDING")}
            style={{ padding: "10px 14px" }}
          >
            Pending
          </button>
          <button 
            className={`tab-bar__item ${activeTab === "RESOLVED" ? "tab-bar__item--active" : ""}`}
            onClick={() => setActiveTab("RESOLVED")}
            style={{ padding: "10px 14px" }}
          >
            Resolved
          </button>
        </div>

        <div className="review-list" style={{ padding: "var(--space-md)", flex: 1 }}>
          {loading ? (
            <div className="empty-state">
              <div className="empty-state__icon">⏳</div>
              <div className="empty-state__message">Loading queue...</div>
            </div>
          ) : reviews.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state__message">No reviews in this queue.</div>
            </div>
          ) : (
            reviews.map(review => (
              <div 
                key={review.review_id}
                className={`review-item ${selectedReview?.review_id === review.review_id ? "review-item--active" : ""}`}
                onClick={() => setSelectedReview(review)}
              >
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                  <span className="text-xs font-bold text-secondary">{review.review_id}</span>
                  <span className={`status-badge ${review.status === "PENDING" ? "status-badge--queued" : "status-badge--completed"}`}>
                    {review.status}
                  </span>
                </div>
                <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-primary)" }}>
                  {review.attack.replace(/_/g, " ").toUpperCase()} (Sev. {review.severity})
                </div>
                <div className="text-xs text-tertiary mt-1" style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>{review.dataset || "Unknown"}</span>
                  <span className="text-danger font-mono font-bold">↓ {review.degradation.toFixed(1)}%</span>
                </div>
              </div>
            ))
          )}
        </div>
      </aside>

      {/* RIGHT - Review Details & Decision */}
      <main className="center-view" style={{ overflowY: "auto", padding: "var(--space-xl)", gap: "var(--space-xl)" }}>
        {!selectedReview ? (
          <div className="placeholder-view">
            <div className="placeholder-view__title">Select a review case</div>
          </div>
        ) : (
          <>
            {/* Header info */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "var(--space-sm)", marginBottom: "4px" }}>
                  <h1 style={{ fontSize: "1.5rem", fontWeight: 800, margin: 0, color: "var(--text-primary)" }}>
                    {selectedReview.review_id}
                  </h1>
                  <span className={`status-badge ${selectedReview.status === "PENDING" ? "status-badge--queued" : "status-badge--completed"}`}>
                    {selectedReview.status}
                  </span>
                </div>
                <div className="text-sm text-secondary">
                  Flagged by <strong className="text-primary">{selectedReview.flagged_by}</strong> on {new Date(selectedReview.created_at).toLocaleString()}
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div className="text-xs text-tertiary uppercase tracking-wider">Degradation</div>
                <div className="text-danger font-mono font-bold" style={{ fontSize: "1.8rem" }}>
                  {selectedReview.degradation.toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Engineer Notes */}
            <div className="glass-panel" style={{ padding: "var(--space-md)", borderRadius: "var(--radius-md)", borderLeft: "4px solid var(--warning)" }}>
              <div className="text-xs font-bold text-warning uppercase" style={{ marginBottom: "8px" }}>Model</div>
              <p className="text-sm" style={{ color: "var(--text-primary)" }}>{selectedReview.model}</p>
            </div>

            {/* Image Comparison - Placeholder since we don't store image URLs in review table yet */}
            <div className="comparison-container" style={{ minHeight: "350px", borderRadius: "var(--radius-lg)" }}>
              <div className="comparison-container__side comparison-container__side--clean">
                <span className="comparison-container__label comparison-container__label--clean">Clean</span>
                <div className="flex-center" style={{ flex: 1, color: "var(--text-tertiary)" }}>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: "0.75rem" }}>Run ID: {selectedReview.run_id}</div>
                  </div>
                </div>
              </div>
              <div className="comparison-container__divider" />
              <div className="comparison-container__side">
                <span className="comparison-container__label comparison-container__label--attacked">Attacked</span>
                <div className="flex-center" style={{ flex: 1, color: "var(--text-tertiary)" }}>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: "0.75rem" }}>Load details from report API to see images</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Decision Panel */}
            {selectedReview.status === "PENDING" ? (
              <div className="config-panel" style={{ animation: "none", marginTop: "auto", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-lg)" }}>
                <div className="config-panel__label" style={{ fontSize: "0.9rem", color: "var(--text-primary)" }}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                  Reviewer Decision
                </div>
                
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-sm)", marginTop: "var(--space-md)" }}>
                  <button 
                    className={`attack-card ${decision === "ACCEPT_RISK" ? "attack-card--selected" : ""}`}
                    onClick={() => setDecision("ACCEPT_RISK")}
                    style={{ borderColor: decision === "ACCEPT_RISK" ? "var(--success)" : "var(--border-subtle)" }}
                  >
                    <span className="attack-card__name">Accept Risk</span>
                    <span className="attack-card__group">Proceed with caution</span>
                  </button>
                  <button 
                    className={`attack-card ${decision === "REQUEST_RETRAIN" ? "attack-card--selected" : ""}`}
                    onClick={() => setDecision("REQUEST_RETRAIN")}
                    style={{ borderColor: decision === "REQUEST_RETRAIN" ? "var(--danger)" : "var(--border-subtle)" }}
                  >
                    <span className="attack-card__name">Request Retrain</span>
                    <span className="attack-card__group">Model fails requirements</span>
                  </button>
                </div>

                <div className="config-panel__section mt-4">
                  <label className="config-panel__label">Mandatory Note</label>
                  <textarea 
                    className="select-field" 
                    style={{ minHeight: "100px", padding: "12px", background: "var(--bg-primary)", resize: "vertical" }}
                    placeholder="Explain the reasoning for your decision..."
                    value={decisionNote}
                    onChange={(e) => setDecisionNote(e.target.value)}
                  />
                </div>

                <button 
                  className="run-button mt-4" 
                  disabled={!decision || !decisionNote}
                  onClick={handleSubmitDecision}
                >
                  Submit Decision
                </button>
              </div>
            ) : (
              <div className="glass-panel" style={{ padding: "var(--space-lg)", borderRadius: "var(--radius-lg)", borderTop: "4px solid var(--success)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
                  <h3 style={{ margin: 0 }}>Review Resolved</h3>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: "12px", fontSize: "0.9rem" }}>
                  <span className="text-secondary">Decision:</span>
                  <span className="font-bold text-primary">{selectedReview.decision}</span>
                  <span className="text-secondary">Resolved by:</span>
                  <span className="text-primary">{selectedReview.resolved_by}</span>
                  <span className="text-secondary">Note:</span>
                  <span className="text-primary">{selectedReview.decision_note}</span>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
