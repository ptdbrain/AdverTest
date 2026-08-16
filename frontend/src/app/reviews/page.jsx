"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import {
  getReviews,
  resolveReview,
  getRunSamples,
  getFailureClusters,
  createFailureCluster,
  createDefenseProfile,
} from "@/lib/api";

export default function ReviewPage() {
  const [activeTab, setActiveTab] = useState("PENDING");
  const [reviews, setReviews] = useState(null);
  const [selectedReview, setSelectedReview] = useState(null);
  const [reviewSamples, setReviewSamples] = useState([]);
  const [clusters, setClusters] = useState([]);
  const [selectedClusterId, setSelectedClusterId] = useState("");
  const [newClusterName, setNewClusterName] = useState("");

  const [decision, setDecision] = useState("");
  const [decisionNote, setDecisionNote] = useState("");
  const [defenseName, setDefenseName] = useState("");
  const [defenseType, setDefenseType] = useState("spatial_filter");

  const [error, setError] = useState("");
  const loading = reviews === null;

  // Load reviews and failure clusters
  useEffect(() => {
    let mounted = true;
    getReviews({ status: activeTab })
      .then((data) => {
        if (mounted) {
          setReviews(data);
          if (data.length > 0) {
            setSelectedReview(data[0]);
          } else {
            setSelectedReview(null);
          }
        }
      })
      .catch((err) => {
        console.error("Failed to load reviews:", err);
        if (mounted) {
          setReviews([]);
          setError("Could not load review queue.");
        }
      });

    getFailureClusters()
      .then((cls) => {
        if (mounted) setClusters(cls);
      })
      .catch(console.error);

    return () => {
      mounted = false;
    };
  }, [activeTab]);

  // Load samples when a review is selected
  useEffect(() => {
    if (!selectedReview?.run_id) return;
    let ignore = false;
    getRunSamples(selectedReview.run_id)
      .then((samples) => {
        if (ignore) return;
        const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
        const artifactUrl = (value) => value ? new URL(value, `${apiBase.replace(/\/$/, "")}/`).href : null;
        setReviewSamples(
          samples.map((s) => ({
            ...s,
            clean_image: artifactUrl(s.artifacts?.clean_input_url),
            attacked_image: artifactUrl(s.artifacts?.attacked_input_url),
            clean_overlay: artifactUrl(s.artifacts?.clean_prediction_url),
            attacked_overlay: artifactUrl(s.artifacts?.attacked_prediction_url),
          }))
        );
      })
      .catch(() => {
        if (!ignore) setReviewSamples([]);
      });
    return () => {
      ignore = true;
    };
  }, [selectedReview]);

  const selectTab = (status) => {
    setReviews(null);
    setError("");
    setActiveTab(status);
  };

  const handleCreateCluster = async () => {
    if (!newClusterName || !selectedReview) return;
    try {
      const cluster = await createFailureCluster(newClusterName, [selectedReview.review_id]);
      setClusters((prev) => [...prev, cluster]);
      setSelectedClusterId(cluster.cluster_id || cluster.id);
      setNewClusterName("");
    } catch (err) {
      alert(`Failed to create cluster: ${err.message}`);
    }
  };

  const handleSubmitDecision = async () => {
    if (!decision || !decisionNote || !selectedReview) return;

    try {
      // Optional defense profile creation if specified
      let defenseProfileId = null;
      if (defenseName) {
        const dp = await createDefenseProfile({
          name: defenseName,
          type: defenseType,
          params: { threshold: 0.5 },
        });
        defenseProfileId = dp.id;
      }

      const updated = await resolveReview(
        selectedReview.review_id,
        decision,
        `${decisionNote}${selectedClusterId ? ` [Cluster: ${selectedClusterId}]` : ""}${
          defenseProfileId ? ` [Defense: ${defenseProfileId}]` : ""
        }`,
        "reviewer_admin"
      );

      setReviews((prev) => prev.filter((r) => r.review_id !== updated.review_id));
      setSelectedReview(null);
      setDecision("");
      setDecisionNote("");
      setDefenseName("");
    } catch (err) {
      alert(`Failed to submit decision: ${err.message}`);
    }
  };

  const currentSample = reviewSamples[0];

  return (
    <div className="main-content" style={{ gridTemplateColumns: "350px 1fr" }}>
      {/* LEFT - Queue List */}
      <aside className="config-panel" style={{ padding: 0 }}>
        <div style={{ padding: "var(--space-lg)", borderBottom: "1px solid var(--border-subtle)" }}>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700, margin: 0, color: "var(--text-primary)" }}>
            Review Queue
          </h2>
          <p className="text-xs text-secondary mt-1">Human-in-the-loop audit & defense profiling</p>
        </div>

        <div className="tab-bar" style={{ padding: "0 var(--space-md)" }}>
          <button
            type="button"
            className={`tab-bar__item ${activeTab === "PENDING" ? "tab-bar__item--active" : ""}`}
            onClick={() => selectTab("PENDING")}
            style={{ padding: "10px 14px" }}
          >
            Pending
          </button>
          <button
            type="button"
            className={`tab-bar__item ${activeTab === "RESOLVED" ? "tab-bar__item--active" : ""}`}
            onClick={() => selectTab("RESOLVED")}
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
          ) : error ? (
            <div className="empty-state" role="alert">
              <div className="empty-state__message">{error}</div>
            </div>
          ) : reviews.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state__message">No reviews in this queue.</div>
            </div>
          ) : (
            reviews.map((review) => (
              <div
                key={review.review_id}
                className={`review-item ${selectedReview?.review_id === review.review_id ? "review-item--active" : ""}`}
                onClick={() => setSelectedReview(review)}
              >
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                  <span className="text-xs font-bold text-secondary">{review.review_id}</span>
                  <span
                    className={`status-badge ${
                      review.status === "PENDING" ? "status-badge--queued" : "status-badge--completed"
                    }`}
                  >
                    {review.status}
                  </span>
                </div>
                <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-primary)" }}>
                  {review.attack.replace(/_/g, " ").toUpperCase()} (Sev. {review.severity})
                </div>
                <div className="text-xs text-tertiary mt-1" style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>{review.dataset || "Synthetic"}</span>
                  <span className="text-danger font-mono font-bold">↓ {review.degradation.toFixed(1)}%</span>
                </div>
              </div>
            ))
          )}
        </div>
      </aside>

      {/* RIGHT - Detailed Audit & Image Evidence */}
      <main className="center-view" style={{ overflowY: "auto", padding: "var(--space-xl)", gap: "var(--space-xl)" }}>
        {!selectedReview ? (
          <div className="placeholder-view">
            <div className="placeholder-view__title">Select a review case to inspect evidence</div>
          </div>
        ) : (
          <>
            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "var(--space-sm)", marginBottom: "4px" }}>
                  <h1 style={{ fontSize: "1.5rem", fontWeight: 800, margin: 0, color: "var(--text-primary)" }}>
                    {selectedReview.review_id}
                  </h1>
                  <span
                    className={`status-badge ${
                      selectedReview.status === "PENDING" ? "status-badge--queued" : "status-badge--completed"
                    }`}
                  >
                    {selectedReview.status}
                  </span>
                </div>
                <div className="text-sm text-secondary">
                  Flagged by <strong className="text-primary">{selectedReview.flagged_by}</strong> on{" "}
                  {new Date(selectedReview.created_at).toLocaleString()}
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div className="text-xs text-tertiary uppercase tracking-wider">Degradation</div>
                <div className="text-danger font-mono font-bold" style={{ fontSize: "1.8rem" }}>
                  ↓ {selectedReview.degradation.toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Real Evidence Overlays (Distinct Clean vs Attacked) */}
            <div className="grid-2x2" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--space-md)" }}>
              <div className="glass-panel" style={{ padding: "12px" }}>
                <div style={{ fontSize: "0.8rem", fontWeight: "bold", color: "var(--success)", marginBottom: "8px" }}>
                  Clean Image & GT
                </div>
                {currentSample?.clean_image ? (
                  <Image src={currentSample.clean_image} alt="Clean Input" width={640} height={360} unoptimized />
                ) : (
                  <div style={{ padding: "20px", textAlign: "center", color: "var(--text-tertiary)" }}>Clean Sample Available</div>
                )}
              </div>

              <div className="glass-panel" style={{ padding: "12px" }}>
                <div style={{ fontSize: "0.8rem", fontWeight: "bold", color: "var(--danger)", marginBottom: "8px" }}>
                  Attacked Image Variant
                </div>
                {currentSample?.attacked_image ? (
                  <Image src={currentSample.attacked_image} alt="Attacked Input" width={640} height={360} unoptimized />
                ) : (
                  <div style={{ padding: "20px", textAlign: "center", color: "var(--text-tertiary)" }}>Attacked Variant Available</div>
                )}
              </div>

              <div className="glass-panel" style={{ padding: "12px" }}>
                <div style={{ fontSize: "0.8rem", fontWeight: "bold", color: "var(--info)", marginBottom: "8px" }}>
                  Clean Prediction Overlay
                </div>
                {currentSample?.clean_overlay ? (
                  <Image src={currentSample.clean_overlay} alt="Clean Overlay" width={640} height={360} unoptimized />
                ) : (
                  <div style={{ padding: "20px", textAlign: "center", color: "var(--text-tertiary)" }}>Clean Prediction Overlay</div>
                )}
              </div>

              <div className="glass-panel" style={{ padding: "12px" }}>
                <div style={{ fontSize: "0.8rem", fontWeight: "bold", color: "var(--warning)", marginBottom: "8px" }}>
                  Attacked Prediction Overlay
                </div>
                {currentSample?.attacked_overlay ? (
                  <Image src={currentSample.attacked_overlay} alt="Attacked Overlay" width={640} height={360} unoptimized />
                ) : (
                  <div style={{ padding: "20px", textAlign: "center", color: "var(--text-tertiary)" }}>Attacked Prediction Overlay</div>
                )}
              </div>
            </div>

            {/* Failure Cluster & Defense Profile Assignment */}
            <div className="glass-panel" style={{ padding: "var(--space-md)" }}>
              <h3 style={{ fontSize: "1rem", margin: "0 0 12px 0" }}>Failure Cluster & Defense Profile Linking</h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                <div>
                  <label className="config-panel__label">Assign to Failure Cluster</label>
                  <select
                    className="select-field"
                    value={selectedClusterId}
                    onChange={(e) => setSelectedClusterId(e.target.value)}
                  >
                    <option value="">-- None --</option>
                    {clusters.map((c) => (
                      <option key={c.id || c.cluster_id} value={c.id || c.cluster_id}>
                        {c.name || c.id}
                      </option>
                    ))}
                  </select>
                  <div style={{ display: "flex", gap: "6px", marginTop: "8px" }}>
                    <input
                      type="text"
                      className="select-field"
                      placeholder="New cluster name..."
                      value={newClusterName}
                      onChange={(e) => setNewClusterName(e.target.value)}
                    />
                    <button type="button" className="attack-card" onClick={handleCreateCluster}>
                      Create
                    </button>
                  </div>
                </div>

                <div>
                  <label className="config-panel__label">Link New Defense Profile</label>
                  <input
                    type="text"
                    className="select-field"
                    placeholder="Defense Profile Name..."
                    value={defenseName}
                    onChange={(e) => setDefenseName(e.target.value)}
                    style={{ marginBottom: "8px" }}
                  />
                  <select
                    className="select-field"
                    value={defenseType}
                    onChange={(e) => setDefenseType(e.target.value)}
                  >
                    <option value="spatial_filter">Spatial Filtering</option>
                    <option value="adversarial_training">Adversarial Fine-Tuning</option>
                    <option value="input_sanitization">Input Sanitization</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Reviewer Decision Form */}
            {selectedReview.status === "PENDING" ? (
              <div className="config-panel" style={{ animation: "none", marginTop: "auto", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-lg)" }}>
                <div className="config-panel__label" style={{ fontSize: "0.9rem", color: "var(--text-primary)" }}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                  Reviewer Decision
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "var(--space-sm)", marginTop: "var(--space-md)" }}>
                  <button
                    type="button"
                    className={`attack-card ${decision === "ACCEPT_RISK" ? "attack-card--selected" : ""}`}
                    onClick={() => setDecision("ACCEPT_RISK")}
                    style={{ borderColor: decision === "ACCEPT_RISK" ? "var(--success)" : "var(--border-subtle)" }}
                  >
                    <span className="attack-card__name">Accept Risk</span>
                    <span className="attack-card__group">Proceed with caution</span>
                  </button>
                  <button
                    type="button"
                    className={`attack-card ${decision === "REQUEST_RETRAIN" ? "attack-card--selected" : ""}`}
                    onClick={() => setDecision("REQUEST_RETRAIN")}
                    style={{ borderColor: decision === "REQUEST_RETRAIN" ? "var(--danger)" : "var(--border-subtle)" }}
                  >
                    <span className="attack-card__name">Request Retrain</span>
                    <span className="attack-card__group">Model fails requirement</span>
                  </button>
                  <button
                    type="button"
                    className={`attack-card ${decision === "REJECT_SAMPLE" ? "attack-card--selected" : ""}`}
                    onClick={() => setDecision("REJECT_SAMPLE")}
                    style={{ borderColor: decision === "REJECT_SAMPLE" ? "var(--warning)" : "var(--border-subtle)" }}
                  >
                    <span className="attack-card__name">Reject Sample</span>
                    <span className="attack-card__group">Invalid ground truth</span>
                  </button>
                </div>

                <div className="config-panel__section mt-4">
                  <label className="config-panel__label">Mandatory Audit Note</label>
                  <textarea
                    className="select-field"
                    style={{ minHeight: "80px", padding: "12px", background: "var(--bg-primary)", resize: "vertical" }}
                    placeholder="Explain the technical reasoning for your decision..."
                    value={decisionNote}
                    onChange={(e) => setDecisionNote(e.target.value)}
                  />
                </div>

                <button
                  type="button"
                  className="run-button mt-4"
                  disabled={!decision || !decisionNote}
                  onClick={handleSubmitDecision}
                >
                  Submit Audit Decision
                </button>
              </div>
            ) : (
              <div className="glass-panel" style={{ padding: "var(--space-lg)", borderRadius: "var(--radius-lg)", borderTop: "4px solid var(--success)" }}>
                <h3 style={{ margin: "0 0 12px 0" }}>Review Resolved</h3>
                <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: "8px", fontSize: "0.9rem" }}>
                  <span className="text-secondary">Decision:</span>
                  <span className="font-bold text-primary">{selectedReview.decision}</span>
                  <span className="text-secondary">Resolved by:</span>
                  <span className="text-primary">{selectedReview.resolved_by}</span>
                  <span className="text-secondary">Audit Note:</span>
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
