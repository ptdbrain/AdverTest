"use client";

import { CheckCircle2, Download, Loader2, PackagePlus, ShieldCheck, XCircle } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import Button from "@/components/common/Button";
import { artifactUrl, createAdversarialDataset, getApiBase, getSampleReviews, upsertSampleReviews } from "@/lib/api";

/** Mirrors the results-page artifact resolution: API-served evidence, cut at /data/. */
function sampleImageSrc(sample, attacked) {
  const raw = attacked
    ? sample?.artifacts?.attacked_input_url || sample?.attacked_image_path || sample?.attacked_input_url || ""
    : sample?.artifacts?.clean_input_url || sample?.clean_image_path || sample?.clean_input_url || "";
  if (!raw) return "";
  const normalized = String(raw).replaceAll("\\", "/");
  const dataIndex = normalized.indexOf("/data/");
  return artifactUrl(dataIndex >= 0 ? normalized.slice(dataIndex) : normalized);
}

function attackedImageSrc(sample) {
  return sampleImageSrc(sample, true);
}

function labelsFromGroundTruth(groundTruth) {
  const items = groundTruth?.objects || groundTruth?.boxes || [];
  return items.map((item) => item.label).filter(Boolean);
}

function predictionText(prediction) {
  const boxes = prediction?.boxes || [];
  if (!boxes.length) return "—";
  return boxes
    .map((box) => {
      const score = Number(box.score);
      return `${box.label || "—"}${Number.isFinite(score) ? ` ${score.toFixed(2)}` : ""}`;
    })
    .join(", ");
}

function sampleDegradation(sample) {
  const value = sample?.degradation ?? sample?.degradation_percent ?? sample?.degradation_hint;
  const number = Number(value);
  return Number.isFinite(number) ? `${number}%` : "— / No verified data";
}

function sampleKey(sample) {
  return `${sample.sample_id}|${sample.attack}|${sample.severity}`;
}

function shortLabel(value, max = 18) {
  const label = String(value || "");
  return label.length > max ? `${label.slice(0, max - 1)}…` : label;
}

/** Per-image approve/reject grid + "create adversarial dataset" flow (workflow steps 12-13). */
export default function SampleReviewPanel({ runId, samples = [], projectId, className = "" }) {
  const [decisions, setDecisions] = useState({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [dataset, setDataset] = useState(null);
  const [creating, setCreating] = useState(false);
  const [selectedKey, setSelectedKey] = useState("");

  const reviewable = useMemo(
    () => samples.filter((sample) => sample?.sample_id && attackedImageSrc(sample)),
    [samples],
  );

  useEffect(() => {
    if (!reviewable.length) {
      setSelectedKey("");
      return;
    }
    if (!reviewable.some((sample) => sampleKey(sample) === selectedKey)) {
      setSelectedKey(sampleKey(reviewable[0]));
    }
  }, [reviewable, selectedKey]);

  useEffect(() => {
    setDecisions({});
    setDataset(null);
    setError("");
    if (!runId) return undefined;
    let cancelled = false;
    setLoading(true);
    getSampleReviews(runId, undefined, projectId)
      .then((rows) => {
        if (cancelled) return;
        const next = {};
        for (const row of Array.isArray(rows) ? rows : []) {
          next[`${row.sample_id}|${row.attack}|${row.severity}`] = row.decision;
        }
        setDecisions(next);
      })
      .catch(() => {
        if (!cancelled) setError("Không tải được quyết định duyệt ảnh đã lưu.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, runId]);

  const persist = useCallback(
    async (entries) => {
      if (!runId || !entries.length) return;
      setSaving(true);
      setError("");
      try {
        const rows = await upsertSampleReviews(
          runId,
          entries.map(({ sample, decision }) => ({
            sample_id: sample.sample_id,
            attack: sample.attack,
            severity: sample.severity,
            decision,
            reviewed_by: "reviewer",
          })),
          projectId,
        );
        setDecisions((current) => {
          const next = { ...current };
          for (const row of rows) next[`${row.sample_id}|${row.attack}|${row.severity}`] = row.decision;
          return next;
        });
        return rows;
      } catch (cause) {
        setError(cause.message || "Không lưu được quyết định duyệt ảnh.");
        return null;
      } finally {
        setSaving(false);
      }
    },
    [projectId, runId],
  );

  const decide = useCallback(
    (sample, decision) => {
      const key = sampleKey(sample);
      setDecisions((current) => ({ ...current, [key]: decision }));
      persist([{ sample, decision }]);
    },
    [persist],
  );

  const approveAll = useCallback(() => {
    const pending = reviewable.filter((sample) => decisions[sampleKey(sample)] !== "APPROVED");
    if (!pending.length) return;
    setDecisions((current) => {
      const next = { ...current };
      for (const sample of pending) next[sampleKey(sample)] = "APPROVED";
      return next;
    });
    persist(pending.map((sample) => ({ sample, decision: "APPROVED" })));
  }, [decisions, persist, reviewable]);

  const clearAll = useCallback(() => {
    const reviewed = reviewable.filter((sample) => decisions[sampleKey(sample)] !== "PENDING");
    if (!reviewed.length) return;
    setDecisions((current) => {
      const next = { ...current };
      for (const sample of reviewed) next[sampleKey(sample)] = "PENDING";
      return next;
    });
    persist(reviewed.map((sample) => ({ sample, decision: "PENDING" })));
  }, [decisions, persist, reviewable]);

  const createDataset = useCallback(async () => {
    setCreating(true);
    setError("");
    try {
      setDataset(await createAdversarialDataset(runId, projectId));
    } catch (cause) {
      setDataset(null);
      setError(cause.message || "Không tạo được adversarial dataset.");
    } finally {
      setCreating(false);
    }
  }, [projectId, runId]);

  if (!runId || loading) {
    if (!runId) return null;
    return (
      <div
        className={`rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4 text-sm text-slate-500 ${className}`}
      >
        <Loader2 className="mr-2 inline h-4 w-4 animate-spin" aria-hidden="true" />
        Đang tải trạng thái duyệt ảnh…
      </div>
    );
  }
  if (!reviewable.length) return null;

  const approvedCount = reviewable.filter((sample) => decisions[sampleKey(sample)] === "APPROVED").length;
  const selectedSample = reviewable.find((sample) => sampleKey(sample) === selectedKey) || reviewable[0];
  const selectedDecision = decisions[sampleKey(selectedSample)] || "PENDING";

  return (
    <section
      data-testid="sample-review-panel"
      aria-label="Duyệt từng ảnh"
      className={`rounded-xl border border-[var(--app-border)] bg-[var(--app-surface)] p-4 ${className}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="flex items-center gap-2 text-base font-semibold text-slate-900">
            <ShieldCheck className="h-4 w-4 text-emerald-600" aria-hidden="true" />
            Duyệt ảnh cho adversarial dataset
          </h2>
          <p className="mt-0.5 text-xs text-slate-500">
            Chọn mẫu cần giữ lại; hệ thống sẽ đóng gói theo đúng loại bài toán của run.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-md bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-700">
            {approvedCount}/{reviewable.length} ảnh đã duyệt
          </span>
          <Button variant="secondary" className="min-h-9 px-3 text-xs" onClick={approveAll}>
            Duyệt tất cả
          </Button>
          <Button variant="secondary" className="min-h-9 px-3 text-xs" onClick={clearAll}>
            Bỏ chọn
          </Button>
          <Button
            className="min-h-9 px-3 text-xs"
            icon={creating ? Loader2 : PackagePlus}
            disabled={approvedCount === 0 || creating || saving}
            onClick={createDataset}
          >
            {creating ? "Đang tạo dataset…" : "Tạo Adversarial Dataset từ ảnh đã duyệt"}
          </Button>
        </div>
      </div>

      {error && (
        <p role="alert" className="mt-3 rounded-lg border border-red-200 bg-red-50 p-2 text-xs text-red-800">
          {error}
        </p>
      )}

      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
        {reviewable.map((sample) => {
          const key = sampleKey(sample);
          const decision = decisions[key] || "PENDING";
          const selected = key === sampleKey(selectedSample);
          return (
            <button
              key={key}
              type="button"
              onClick={() => setSelectedKey(key)}
              aria-label={`Xem ${sample.sample_id} · ${sample.attack} · mức ${sample.severity}`}
              aria-pressed={selected}
              className={`group relative overflow-hidden rounded-lg border text-left transition-all ${
                selected ? "border-blue-500 ring-2 ring-blue-500/20" : "border-[var(--app-border)] hover:border-slate-400"
              }`}
            >
              {/* biome-ignore lint/performance/noImgElement: artifact URLs are served by the API host, not the Next.js asset pipeline */}
              <img
                src={attackedImageSrc(sample)}
                alt={`Ảnh bị tấn công ${sample.sample_id}`}
                className="h-24 w-full bg-slate-100 object-cover"
                loading="lazy"
              />
              <div className="p-2">
                <div className="truncate text-[11px] font-medium text-slate-900" title={sample.sample_id}>
                  {shortLabel(sample.sample_id)}
                </div>
                <div className="mt-0.5 flex items-center justify-between gap-1 text-[10px] text-slate-500">
                  <span className="truncate">
                    {sample.attack} · s{sample.severity}
                  </span>
                  {decision === "APPROVED" ? (
                    <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-600" aria-hidden="true" />
                  ) : decision === "REJECTED" ? (
                    <XCircle className="h-3.5 w-3.5 shrink-0 text-red-600" aria-hidden="true" />
                  ) : (
                    <XCircle className="h-3.5 w-3.5 shrink-0 text-slate-300" aria-hidden="true" />
                  )}
                </div>
              </div>
              {decision !== "PENDING" && (
                <span
                  className={`absolute top-1 right-1 rounded px-1.5 py-0.5 text-[9px] font-bold text-white ${
                    decision === "APPROVED" ? "bg-emerald-600" : "bg-red-600"
                  }`}
                >
                  {decision === "APPROVED" ? "Đã duyệt" : "Từ chối"}
                </span>
              )}
            </button>
          );
        })}
      </div>

      <div className="mt-4 rounded-lg border border-[var(--app-border)] bg-[var(--app-surface)] p-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">
              {selectedSample.sample_id} · {selectedSample.attack} · mức {selectedSample.severity}
            </h3>
            <p className="mt-1 text-xs text-slate-500">Quyết định hiện tại: {selectedDecision}</p>
          </div>
          <div className="flex flex-wrap gap-2" aria-label="Quyết định cho ảnh đang chọn">
            <Button
              variant="secondary"
              size="sm"
              aria-pressed={selectedDecision === "PENDING"}
              disabled={saving}
              onClick={() => decide(selectedSample, "PENDING")}
            >
              Chờ duyệt
            </Button>
            <Button
              variant="success"
              size="sm"
              aria-pressed={selectedDecision === "APPROVED"}
              disabled={saving}
              onClick={() => decide(selectedSample, "APPROVED")}
            >
              Duyệt
            </Button>
            <Button
              variant="danger"
              size="sm"
              aria-pressed={selectedDecision === "REJECTED"}
              disabled={saving}
              onClick={() => decide(selectedSample, "REJECTED")}
            >
              Từ chối
            </Button>
          </div>
        </div>

        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <div>
            {sampleImageSrc(selectedSample, false) ? (
              // biome-ignore lint/performance/noImgElement: authenticated evidence is served by the API host
              <img
                src={sampleImageSrc(selectedSample, false)}
                alt={`Ảnh gốc ${selectedSample.sample_id}`}
                className="h-48 w-full rounded border border-[var(--app-border)] bg-slate-100 object-contain"
              />
            ) : (
              <div className="flex h-48 items-center justify-center rounded border border-[var(--app-border)] text-xs text-slate-500">
                — / No verified data
              </div>
            )}
          </div>
          <div>
            {sampleImageSrc(selectedSample, true) ? (
              // biome-ignore lint/performance/noImgElement: authenticated evidence is served by the API host
              <img
                src={sampleImageSrc(selectedSample, true)}
                alt={`Ảnh attacked ${selectedSample.sample_id}`}
                className="h-48 w-full rounded border border-[var(--app-border)] bg-slate-100 object-contain"
              />
            ) : (
              <div className="flex h-48 items-center justify-center rounded border border-[var(--app-border)] text-xs text-slate-500">
                — / No verified data
              </div>
            )}
          </div>
        </div>

        <dl className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
          <div>
            <dt className="font-semibold text-slate-600">Ground truth</dt>
            <dd>GT: {labelsFromGroundTruth(selectedSample.ground_truth).join(", ") || "—"}</dd>
          </div>
          <div>
            <dt className="font-semibold text-slate-600">Prediction</dt>
            <dd>Prediction gốc: {predictionText(selectedSample.clean_prediction)}</dd>
            <dd>Prediction attacked: {predictionText(selectedSample.attacked_prediction)}</dd>
          </div>
          <div>
            <dt className="font-semibold text-slate-600">Metric</dt>
            <dd>
              {selectedSample.metric?.name
                ? `${selectedSample.metric.name}: ${selectedSample.metric.clean ?? "—"} → ${selectedSample.metric.attacked ?? "—"}`
                : "— / No verified data"}
            </dd>
          </div>
          <div>
            <dt className="font-semibold text-slate-600">Degradation</dt>
            <dd>Suy giảm: {sampleDegradation(selectedSample)}</dd>
          </div>
        </dl>
      </div>

      {dataset && (
        <div
          data-testid="adversarial-dataset-result"
          className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm"
        >
          <div className="font-semibold text-emerald-900">
            Đã tạo dataset “{dataset.dataset_id}” với {dataset.sample_count} ảnh đã duyệt.
          </div>
          <p className="mt-1 text-xs text-emerald-800">
            Định dạng {dataset.format || "—"}: {dataset.format === "segmentation-mask"
              ? "images/ + masks/ + manifest.json"
              : dataset.format === "kitti3d"
                ? "velodyne/ + label_2/ + calib/ + ImageSets/ + manifest.json"
                : "images/ + labels/ + manifest.json"}. Lớp:{" "}
            {Object.values(dataset.class_map || {}).join(", ") || "—"}.{" "}
            {dataset.skipped?.length
              ? `${dataset.skipped.length} mẫu bị bỏ qua: ${dataset.skipped
                  .map((item) => `${item.sample_id} (${item.reason})`)
                  .join(", ")}.`
              : ""}
          </p>
          <Button
            variant="secondary"
            className="mt-2 min-h-9 px-3 text-xs"
            icon={Download}
            onClick={() =>
              window.open(
                dataset.download_url.startsWith("http")
                  ? dataset.download_url
                  : `${getApiBase()}${dataset.download_url}`,
                "_blank",
              )
            }
          >
            Tải ZIP dataset
          </Button>
        </div>
      )}
    </section>
  );
}
