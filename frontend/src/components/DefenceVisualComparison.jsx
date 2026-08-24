"use client";

import { useEffect, useState } from "react";
import { getRunSamples, artifactUrl } from "@/lib/api";
import { useLanguage } from "@/context/LanguageContext";

/**
 * DefenceVisualComparison — Ghép cặp mẫu giữa base run và candidate run (cùng Locked Protocol)
 * rồi hiển thị ảnh prediction side-by-side: Base Model vs Fine-tuned Model trên cùng một ảnh bị tấn công.
 */

function sampleKey(sample) {
  return `${sample?.sample_id || ""}::${sample?.attack || ""}::${sample?.severity ?? 0}`;
}

function PaneImage({ src, alt, label, accent }) {
  const { t } = useLanguage();
  const finalSrc = src ? artifactUrl(src) : null;
  return (
    <div
      style={{
        flex: 1,
        minWidth: 0,
        display: "flex",
        flexDirection: "column",
        background: "var(--bg-elevated, #101623)",
        borderRadius: "6px",
        border: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.15))",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          fontSize: "0.66rem",
          fontWeight: 800,
          padding: "6px 10px",
          color: accent || "var(--accent, #38BDF8)",
          background: "var(--bg-primary, #0B0F17)",
          borderBottom: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.12))",
          letterSpacing: "0.04em",
        }}
      >
        {label}
      </div>
      <div
        style={{
          position: "relative",
          flex: 1,
          minHeight: 220,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "var(--bg-primary, #0B0F17)",
        }}
      >
        {finalSrc ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={finalSrc}
            alt={alt}
            style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
          />
        ) : (
          <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{t("visual.noImage")}</span>
        )}
      </div>
    </div>
  );
}

export default function DefenceVisualComparison({ baselineRunId, candidateRunId }) {
  const { t } = useLanguage();
  const [pairs, setPairs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [index, setIndex] = useState(0);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [base, cand] = await Promise.all([
          getRunSamples(baselineRunId),
          getRunSamples(candidateRunId),
        ]);
        if (cancelled) return;
        const baseArr = Array.isArray(base) ? base : base?.items || [];
        const candArr = Array.isArray(cand) ? cand : cand?.items || [];
        const candMap = new Map(candArr.map((s) => [sampleKey(s), s]));
        const matched = baseArr
          .map((b) => ({ base: b, candidate: candMap.get(sampleKey(b)) || null }))
          .filter((p) => p.candidate);
        setPairs(matched);
        setIndex(0);
      } catch (e) {
        if (!cancelled) setError(e.message || "Could not load comparison images.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [baselineRunId, candidateRunId]);

  if (loading) {
    return (
      <div style={{ padding: "14px", fontSize: "0.75rem", color: "var(--text-muted)" }}>
        {t("visual.loading")}
      </div>
    );
  }
  if (error) {
    return (
      <div style={{ padding: "10px 12px", background: "rgba(239, 68, 68, 0.1)", borderRadius: "6px", fontSize: "0.75rem", color: "var(--danger)" }}>
        {t("visual.loadError", { error })}
      </div>
    );
  }
  if (!pairs.length) {
    return (
      <div style={{ padding: "12px", fontSize: "0.75rem", color: "var(--text-muted)" }}>
        {t("visual.noMatch")}
      </div>
    );
  }

  const safeIndex = Math.min(Math.max(0, index), pairs.length - 1);
  const current = pairs[safeIndex];
  const baseAttacked =
    current.base?.artifacts?.attacked_prediction_url || current.base?.artifacts?.attacked_input_url;
  const candAttacked =
    current.candidate?.artifacts?.attacked_prediction_url || current.candidate?.artifacts?.attacked_input_url;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "8px",
        }}
      >
        <strong style={{ fontSize: "0.8rem", color: "var(--text-primary)" }}>
          {t("visual.title")}
        </strong>
        <span style={{ fontSize: "0.68rem", color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>
          {t("visual.sampleLabel", {
            id: current.base?.sample_id?.slice(0, 12),
            sev: current.base?.severity ?? "—",
            index: safeIndex + 1,
            total: pairs.length,
          })}
        </span>
      </div>

      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
        <PaneImage
          src={baseAttacked}
          alt="Base model attacked prediction"
          label="🔵 BASE MODEL"
          accent="var(--accent, #38BDF8)"
        />
        <PaneImage
          src={candAttacked}
          alt="Fine-tuned model attacked prediction"
          label="🟢 FINE-TUNED / REPAIRED"
          accent="#10B981"
        />
      </div>

      {pairs.length > 1 && (
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <button
            type="button"
            className="sample-nav__arrow"
            onClick={() => setIndex(Math.max(0, safeIndex - 1))}
            disabled={safeIndex === 0}
          >
            ←
          </button>
          <div style={{ display: "flex", gap: "5px", flexWrap: "wrap", flex: 1 }}>
            {pairs.map((pair, i) => (
              <button
                key={`${pair.base?.sample_id}-${i}`}
                type="button"
                onClick={() => setIndex(i)}
                style={{
                  width: "26px",
                  height: "26px",
                  borderRadius: "4px",
                  border: `1px solid ${i === safeIndex ? "var(--accent)" : "var(--border-subtle)"}`,
                  background: i === safeIndex ? "rgba(56, 189, 248, 0.15)" : "transparent",
                  color: "var(--text-secondary)",
                  fontSize: "0.62rem",
                  fontWeight: 700,
                  cursor: "pointer",
                }}
              >
                {i + 1}
              </button>
            ))}
          </div>
          <button
            type="button"
            className="sample-nav__arrow"
            onClick={() => setIndex(Math.min(pairs.length - 1, safeIndex + 1))}
            disabled={safeIndex >= pairs.length - 1}
          >
            →
          </button>
        </div>
      )}
    </div>
  );
}