"use client";

import { ChevronRight, Crosshair, History } from "lucide-react";
import React, { useEffect, useState } from "react";
import { useProject } from "@/context/ProjectContext";
import { getSession } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function RunHistoryBar({ sessionId, activeRunId, onSelectRun, projectId }) {
  const { activeProjectId } = useProject();
  const scopedProjectId = projectId || activeProjectId;
  const [runs, setRuns] = useState([]);

  useEffect(() => {
    if (!sessionId) return;
    getSession(sessionId, scopedProjectId)
      .then((data) => {
        if (data && data.runs && Array.isArray(data.runs)) {
          setRuns(data.runs);
        }
      })
      .catch((err) => {
        console.warn("Could not fetch session runs for history bar:", err);
      });
  }, [activeProjectId, scopedProjectId, sessionId, activeRunId]);

  if (!runs || runs.length === 0) return null;

  return (
    <div className="w-full border-b border-slate-200 bg-white sticky top-0 z-10 overflow-x-auto custom-scrollbar">
      <div className="flex items-center p-2 min-w-max gap-2">
        <div className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold text-slate-500 border-r border-slate-200 mr-1">
          <History className="w-3.5 h-3.5" />
          <span>Lịch sử phiên ({runs.length})</span>
        </div>

        {runs.map((run, idx) => {
          const isActive = run.id === activeRunId;
          const drop = run.map_drop_pct || 0;
          let colorClass = "bg-emerald-100 text-emerald-700 border-emerald-200";
          if (drop >= 50) colorClass = "bg-red-100 text-red-700 border-red-200";
          else if (drop >= 15) colorClass = "bg-amber-100 text-amber-700 border-amber-200";

          return (
            <React.Fragment key={run.id}>
              {idx > 0 && <ChevronRight className="w-3 h-3 text-slate-300 shrink-0" />}
              <button
                type="button"
                onClick={() => onSelectRun(run.id)}
                className={cn(
                  "flex flex-col items-start gap-1 rounded-lg border px-3 py-1.5 transition-all min-w-[140px] max-w-[200px] text-left shrink-0",
                  isActive
                    ? "border-blue-500 bg-blue-50 shadow-sm border-b-2"
                    : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50",
                )}
              >
                <div className="flex items-center justify-between w-full gap-2">
                  <span className="text-[10px] font-bold text-slate-700 truncate" title={run.attack_name}>
                    {run.attack_name || `Lần ${idx + 1}`}
                  </span>
                  <span
                    className={cn("text-[9px] font-semibold px-1.5 py-0.5 rounded-full whitespace-nowrap", colorClass)}
                  >
                    -{drop.toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center gap-1 w-full text-[9px] text-slate-400 justify-between">
                  <span className="flex items-center gap-0.5">
                    <Crosshair className="w-2.5 h-2.5" /> Sev {run.severity || 3}
                  </span>
                  <span className="truncate">{run.timestamp?.split(" ")[1] || "—"}</span>
                </div>
              </button>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
