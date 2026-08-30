"use client";

import React from "react";
import Card from "@/components/common/Card";
import { Play, Square, Circle, ShieldAlert, FileText, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

export default function SessionTimeline({ session }) {
  if (!session) return null;

  const runs = session.runs || [];
  
  const formatDate = (dateStr) => {
    if (!dateStr) return "—";
    try {
      const d = new Date(dateStr);
      return d.toLocaleString("vi-VN", {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute:'2-digit', second:'2-digit'
      });
    } catch {
      return dateStr;
    }
  };

  return (
    <Card 
      title="Dòng thời gian Phiên kiểm thử" 
      subtitle={`Lịch sử ${runs.length} lần chạy trong phiên: ${session.name || session.id}`}
    >
      <div className="relative pl-6 py-2 before:content-[''] before:absolute before:left-3 before:top-4 before:bottom-4 before:w-0.5 before:bg-slate-200">
        
        {/* Session Start Event */}
        <div className="relative mb-6">
          <div className="absolute -left-[27px] bg-white p-1 rounded-full border-2 border-emerald-500 text-emerald-500 z-10">
            <Play className="w-3 h-3 ml-0.5" />
          </div>
          <div className="pl-4">
            <div className="text-xs font-bold text-slate-800">Bắt đầu phiên</div>
            <div className="text-[10px] text-slate-500">{formatDate(session.created_at)}</div>
          </div>
        </div>

        {/* Runs Events */}
        {runs.map((run, idx) => {
          const drop = run.map_drop_pct || 0;
          let dotColor = "border-emerald-400 bg-emerald-100";
          let badgeColor = "bg-emerald-50 text-emerald-700 border-emerald-200";
          
          if (drop >= 50) {
            dotColor = "border-red-500 bg-red-100";
            badgeColor = "bg-red-50 text-red-700 border-red-200";
          } else if (drop >= 15) {
            dotColor = "border-amber-400 bg-amber-100";
            badgeColor = "bg-amber-50 text-amber-700 border-amber-200";
          }

          return (
            <div key={run.id} className="relative mb-6 group">
              <div className={cn("absolute -left-[23px] w-3 h-3 rounded-full border-2 z-10 mt-1.5 transition-transform group-hover:scale-125", dotColor)} />
              <div className="pl-4">
                <div className="flex flex-wrap items-center gap-2 mb-1">
                  <span className="text-sm font-bold text-slate-800">
                    {run.attack_name || `Lần chạy ${idx + 1}`}
                  </span>
                  <span className={cn("text-[10px] font-bold px-2 py-0.5 rounded border", badgeColor)}>
                    -{drop.toFixed(1)}% mAP
                  </span>
                  <span className="text-[10px] text-slate-500">
                    {formatDate(run.timestamp)}
                  </span>
                </div>
                
                <div className="flex items-center gap-4 text-[11px] text-slate-600 mb-2">
                  <span>Mức độ: <strong>{run.severity || 3}</strong></span>
                  <span>Sạch: {run.clean_map}</span>
                  <span>Bị tấn công: {run.attacked_map}</span>
                </div>

                {run.note && (
                  <div className="bg-blue-50/50 border border-blue-100 rounded-lg p-2.5 text-xs text-blue-800 flex items-start gap-2 max-w-2xl">
                    <FileText className="w-4 h-4 shrink-0 text-blue-500 mt-0.5" />
                    <div>
                      <span className="font-semibold text-blue-700 mr-1">Ghi chú:</span>
                      {run.note}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {/* Session End Event (if completed) */}
        {session.status === "completed" && (
          <div className="relative mt-2">
            <div className="absolute -left-[27px] bg-white p-1 rounded-full border-2 border-slate-700 text-slate-700 z-10">
              <CheckCircle2 className="w-3 h-3" />
            </div>
            <div className="pl-4">
              <div className="text-xs font-bold text-slate-800">Khóa phiên (Hoàn tất)</div>
              <div className="text-[10px] text-slate-500">{formatDate(session.ended_at || session.updated_at)}</div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
