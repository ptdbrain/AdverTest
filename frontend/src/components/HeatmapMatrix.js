"use client";

import { useMemo } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { getDescriptiveAttackName } from "@/lib/attackNaming";

function cellColor(degradation) {
  const d = Math.max(0, Math.min(100, degradation));
  if (d < 5)  return { bg: "rgba(104, 211, 145, 0.18)", text: "#68d391" };
  if (d < 15) return { bg: "rgba(104, 211, 145, 0.12)", text: "#9ae6b4" };
  if (d < 30) return { bg: "rgba(246, 173, 85, 0.14)",  text: "#fbd38d" };
  if (d < 50) return { bg: "rgba(246, 173, 85, 0.2)",   text: "#f6ad55" };
  if (d < 75) return { bg: "rgba(252, 129, 129, 0.16)", text: "#feb2b2" };
  return       { bg: "rgba(252, 129, 129, 0.25)", text: "#fc8181" };
}

export default function HeatmapMatrix({ cells = [], heatmap = {}, report = null }) {
  const { t } = useLanguage();
  const groupLabels = {
    A: t("heatmap.catA"),
    B: t("heatmap.catB"),
    C: t("heatmap.catC"),
    D: t("heatmap.catD"),
    E: t("heatmap.catE"),
    F: t("heatmap.catF"),
  };
  const { groupedRows, severities } = useMemo(() => {
    const sevSet = new Set();
    cells.forEach((c) => sevSet.add(c.severity));
    const sevs = [...sevSet].sort((a, b) => a - b);
    if (sevs.length === 0) sevs.push(1, 2, 3, 4, 5);

    const byAttack = {};
    cells.forEach((c) => {
      // Extract clean human-friendly attack name without (Cấp X) for row header
      const displayName = getDescriptiveAttackName(c, null, report).replace(/\s*\(Cấp\s*\d+\)/i, "").trim();
      const groupKey = c.group || "?";
      const key = `${groupKey}::${displayName}`;

      if (!byAttack[key]) byAttack[key] = { group: groupKey, name: displayName, cells: {} };
      byAttack[key].cells[c.severity] = c;
    });

    const grouped = {};
    Object.entries(byAttack).forEach(([key, data]) => {
      const g = data.group || "?";
      if (!grouped[g]) grouped[g] = [];
      grouped[g].push(data);
    });

    return { groupedRows: grouped, severities: sevs };
  }, [cells, report]);

  if (cells.length === 0) {
    return (
      <div className="heatmap">
        <div className="heatmap__title">Degradation Heatmap</div>
        <div className="empty-state">
          <div className="empty-state__message">No data yet — run an attack first</div>
        </div>
      </div>
    );
  }

  return (
    <div className="heatmap">
      <div className="heatmap__title">
        {t("heatmap.title")}
        <span style={{ fontSize: "0.55rem", color: "var(--text-muted)", fontWeight: 400, marginLeft: "auto" }}>
          {t("heatmap.cellD")}
        </span>
      </div>
      <table className="heatmap__table">
        <thead>
          <tr>
            <th>{t("heatmap.attackCol")}</th>
            {severities.map((s) => (
              <th key={s}>{t("heatmap.levelCol", { level: s })}</th>
            ))}
          </tr>
        </thead>
          {Object.entries(groupedRows)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([group, attacks]) => (
              <tbody key={group}>
                <tr className="heatmap__group-header">
                  <td colSpan={severities.length + 1}>
                    {groupLabels[group] || group}
                  </td>
                </tr>
                {attacks.map((atk) => (
                  <tr key={atk.name}>
                    <td style={{ fontWeight: 600 }}>{atk.name}</td>
                    {severities.map((s) => {
                      const cell = atk.cells[s];
                      const d = cell?.degradation ?? null;
                      const color = d != null ? cellColor(d) : { bg: "var(--bg-elevated)", text: "var(--text-muted)" };
                      return (
                        <td
                          key={s}
                          style={{
                            background: color.bg,
                            color: color.text,
                          }}
                          title={d != null ? `AP: ${cell.ap?.toFixed(3)} | D: ${d.toFixed(1)}%` : "N/A"}
                        >
                          {d != null ? `${d.toFixed(1)}%` : "\u2014"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            ))}
      </table>
    </div>
  );
}
