"use client";

import { useEffect, useMemo, useState } from "react";

const normalize = (value) => value.trim().toLocaleLowerCase();
const IGNORE = "ignore";
const IGNORE_OPTION = "__ignore__";

export default function ClassMappingCard({ modelClasses, datasetClasses, source, onChange }) {
  const [mapping, setMapping] = useState({});
  // Parents construct these arrays while rendering, so their references change
  // often even when their contents do not.  Depend on stable content keys to
  // avoid overwriting a user-selected mapping on every parent render.
  const datasetKey = useMemo(() => datasetClasses.map(normalize).join("\u001f"), [datasetClasses]);
  const modelKey = useMemo(() => modelClasses.map(normalize).join("\u001f"), [modelClasses]);
  const stableDatasetClasses = useMemo(() => datasetClasses, [datasetKey]);
  const stableModelClasses = useMemo(() => modelClasses, [modelKey]);

  useEffect(() => {
    setMapping((previous) => {
      const next = Object.fromEntries(stableDatasetClasses.map((label) => [
        label,
        previous[label] ?? stableModelClasses.find((modelLabel) => normalize(modelLabel) === normalize(label)) ?? null,
      ]));
      onChange(next);
      return next;
    });
  }, [datasetKey, modelKey, stableDatasetClasses, stableModelClasses, onChange]);

  const changeMapping = (label, modelLabel) => {
    const next = { ...mapping, [label]: modelLabel === IGNORE_OPTION ? IGNORE : modelLabel || null };
    setMapping(next);
    onChange(next);
  };
  return <section className="rounded-xl border border-slate-200 bg-white p-4">
    <h2 className="font-semibold text-slate-900">Ánh xạ nhãn lớp</h2>
    {source === "demo" && <p className="mt-2 text-sm text-amber-700">DEMO MAPPING — không dùng để chạy benchmark.</p>}
    <div className="mt-3 space-y-2">{datasetClasses.map((label) => <label key={label} className="flex items-center justify-between gap-3 text-sm"><span>{label}</span><select aria-label={`Mapping ${label}`} value={mapping[label] === IGNORE ? IGNORE_OPTION : mapping[label] || ""} onChange={(event) => changeMapping(label, event.target.value)}><option value="">Không ánh xạ</option><option value={IGNORE_OPTION}>Bỏ qua có chủ đích</option>{modelClasses.map((modelLabel) => <option key={modelLabel} value={modelLabel}>{modelLabel}</option>)}</select></label>)}</div>
  </section>;
}
