"use client";

import React, { useEffect, useState } from "react";

const PRIORITY_STYLES = {
  CRITICAL: "bg-red-900/40 border-red-500/50 text-red-300",
  HIGH: "bg-amber-900/40 border-amber-500/50 text-amber-300",
  MEDIUM: "bg-blue-900/40 border-blue-500/50 text-blue-300",
  LOW: "bg-slate-800/40 border-slate-600/50 text-slate-300",
};

const PRIORITY_BADGES = {
  CRITICAL: "bg-red-500/20 text-red-400 border-red-500/30",
  HIGH: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  MEDIUM: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  LOW: "bg-slate-500/20 text-slate-400 border-slate-500/30",
};

export default function AdvisorPanel({ runId, projectId, onTriggerAction }) {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dismissedIds, setDismissedIds] = useState(new Set());

  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      const url = new URL("/api/v1/advisor/recommendations", window.location.origin);
      if (runId) url.searchParams.set("run_id", runId);
      if (projectId) url.searchParams.set("project_id", projectId);

      const res = await fetch(url.toString());
      if (res.ok) {
        const data = await res.json();
        setRecommendations(data.recommendations || []);
      }
    } catch (err) {
      console.error("Failed to load AI recommendations:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendations();
  }, [runId, projectId]);

  const handleDismiss = async (recId) => {
    setDismissedIds((prev) => new Set([...prev, recId]));
    try {
      await fetch("/api/v1/advisor/dismiss", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recommendation_id: recId }),
      });
    } catch (err) {
      console.error("Failed to dismiss recommendation:", err);
    }
  };

  const activeRecs = recommendations.filter((r) => !dismissedIds.has(r.id));

  if (loading) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
          <span className="text-sm font-medium text-slate-400">Evaluating AI Advisor rules...</span>
        </div>
      </div>
    );
  }

  if (activeRecs.length === 0) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 text-center">
        <span className="text-xs text-slate-500">✨ No active AI Advisor recommendations. All evaluated tasks are aligned.</span>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="flex h-2 w-2 rounded-full bg-indigo-500 animate-pulse" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-indigo-400">
            AI Next-Action Advisor ({activeRecs.length})
          </h3>
        </div>
        <button
          onClick={fetchRecommendations}
          className="text-xs text-slate-400 hover:text-slate-200 transition-colors"
        >
          Refresh Rules
        </button>
      </div>

      <div className="grid gap-3">
        {activeRecs.map((rec) => {
          const cardStyle = PRIORITY_STYLES[rec.priority] || PRIORITY_STYLES.LOW;
          const badgeStyle = PRIORITY_BADGES[rec.priority] || PRIORITY_BADGES.LOW;

          return (
            <div
              key={rec.id}
              className={`rounded-xl border p-4 transition-all duration-200 shadow-sm ${cardStyle}`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className={`rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider border ${badgeStyle}`}>
                      {rec.priority}
                    </span>
                    <h4 className="text-sm font-semibold text-slate-100">{rec.title}</h4>
                  </div>
                  <p className="text-xs leading-relaxed text-slate-300">{rec.reason}</p>
                </div>

                <button
                  onClick={() => handleDismiss(rec.id)}
                  className="rounded p-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-colors"
                  title="Dismiss recommendation"
                >
                  ✕
                </button>
              </div>

              {rec.evidence?.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {rec.evidence.map((ev, idx) => (
                    <span
                      key={idx}
                      className="rounded bg-slate-950/60 px-2 py-0.5 font-mono text-[10px] text-slate-400 border border-slate-800"
                    >
                      {ev}
                    </span>
                  ))}
                </div>
              )}

              <div className="mt-3 flex items-center justify-end gap-2 pt-2 border-t border-slate-800/40">
                <button
                  onClick={() => onTriggerAction && onTriggerAction(rec.action_type, rec.suggested_parameters)}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-indigo-500 transition-colors"
                >
                  <span>Execute Next Action</span>
                  <span>→</span>
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
