"use client";

import { useEffect, useState } from "react";

const normalize = (value) => value.trim().toLocaleLowerCase();

export default function ClassMappingCard({ modelClasses, datasetClasses, source, onChange }) {
  const [mapping, setMapping] = useState({});
  useEffect(() => {
    const initial = Object.fromEntries(datasetClasses.map((label) => [label, modelClasses.find((modelLabel) => normalize(modelLabel) === normalize(label)) || null]));
    setMapping(initial);
    onChange(initial);
  }, [datasetClasses, modelClasses, onChange]);
  const changeMapping = (label, modelLabel) => {
    const next = { ...mapping, [label]: modelLabel || null };
    setMapping(next);
    onChange(next);
  };
  return <section className="rounded-xl border border-slate-200 bg-white p-4">
    <h2 className="font-semibold text-slate-900">Ánh xạ nhãn lớp</h2>
    {source === "demo" && <p className="mt-2 text-sm text-amber-700">DEMO MAPPING — không dùng để chạy benchmark.</p>}
    <div className="mt-3 space-y-2">{datasetClasses.map((label) => <label key={label} className="flex items-center justify-between gap-3 text-sm"><span>{label}</span><select aria-label={`Mapping ${label}`} value={mapping[label] || ""} onChange={(event) => changeMapping(label, event.target.value)}><option value="">Không ánh xạ</option>{modelClasses.map((modelLabel) => <option key={modelLabel} value={modelLabel}>{modelLabel}</option>)}</select></label>)}</div>
  </section>;
}
