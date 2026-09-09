"use client";

import { useEffect, useMemo } from "react";

const normalize = (value) => value.trim().toLocaleLowerCase();
const IGNORE = "ignore";
const IGNORE_OPTION = "__ignore__";

export default function ClassMappingCard({ modelClasses, datasetClasses, source, value = {}, onChange }) {
  const datasetKey = datasetClasses.map(normalize).join("\u001f");
  const modelKey = modelClasses.map(normalize).join("\u001f");
  const mapping = useMemo(
    () =>
      Object.fromEntries(
        datasetClasses.map((label) => {
          const hasPersistedValue = Object.prototype.hasOwnProperty.call(value, label);
          const exactMatch = modelClasses.find((modelLabel) => normalize(modelLabel) === normalize(label));
          return [label, hasPersistedValue ? value[label] : exactMatch || null];
        }),
      ),
    // Content keys keep parent-created arrays from resetting the controlled map.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [datasetKey, modelKey, value],
  );

  useEffect(() => {
    if (JSON.stringify(mapping) !== JSON.stringify(value)) onChange?.(mapping);
  }, [mapping, onChange, value]);

  const missing = datasetClasses.filter((label) => !mapping[label]);
  const changeMapping = (label, modelLabel) => {
    onChange?.({ ...mapping, [label]: modelLabel === IGNORE_OPTION ? IGNORE : modelLabel || null });
  };

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4">
      <h2 className="font-semibold text-slate-900">Ánh xạ nhãn lớp</h2>
      {source === "demo" && <p className="mt-2 text-sm text-amber-700">DEMO MAPPING — không dùng để chạy benchmark.</p>}
      <div className="mt-3 space-y-2">
        {datasetClasses.map((label) => (
          <label key={label} className="flex items-center justify-between gap-3 text-sm">
            <span>{label}</span>
            <select
              aria-label={`Mapping ${label}`}
              value={mapping[label] === IGNORE ? IGNORE_OPTION : mapping[label] || ""}
              onChange={(event) => changeMapping(label, event.target.value)}
            >
              <option value="">Không ánh xạ</option>
              <option value={IGNORE_OPTION}>Bỏ qua có chủ đích</option>
              {modelClasses.map((modelLabel) => (
                <option key={modelLabel} value={modelLabel}>
                  {modelLabel}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      {missing.length > 0 && (
        <p role="alert" className="mt-3 text-sm text-amber-700">
          Chưa ánh xạ: {missing.join(", ")}
        </p>
      )}
    </section>
  );
}
