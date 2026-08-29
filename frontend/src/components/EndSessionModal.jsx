"use client";

import React, { useState } from "react";
import { CheckCircle, AlertTriangle, X } from "lucide-react";
import { endSession } from "@/lib/api";

export default function EndSessionModal({ isOpen, onClose, session, onConfirm }) {
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!isOpen || !session) return null;

  const runsCount = session.runs?.length || 0;
  const notesCount = session.runs?.filter(r => r.note && r.note.trim().length > 0).length || 0;
  
  let worstRun = null;
  if (runsCount > 0) {
    worstRun = [...session.runs].sort((a, b) => (b.map_drop_pct || 0) - (a.map_drop_pct || 0))[0];
  }

  const handleConfirm = async () => {
    setIsSubmitting(true);
    try {
      await endSession(session.id);
      if (onConfirm) onConfirm();
      onClose();
    } catch (err) {
      console.error("Failed to end session:", err);
      alert("Không thể hoàn tất phiên, vui lòng thử lại.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md overflow-hidden flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-slate-100">
          <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-emerald-500" />
            Hoàn tất Phiên kiểm thử
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-sm space-y-2">
            <div className="flex justify-between">
              <span className="text-slate-500">Phiên:</span>
              <span className="font-semibold text-slate-700">{session.name || session.id}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Mô hình:</span>
              <span className="font-semibold text-slate-700 truncate max-w-[200px] text-right" title={session.model}>{session.model}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Dữ liệu:</span>
              <span className="font-semibold text-slate-700 truncate max-w-[200px] text-right" title={session.dataset}>{session.dataset}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="bg-blue-50 border border-blue-100 rounded-lg p-3 text-center">
              <div className="text-2xl font-black text-blue-700">{runsCount}</div>
              <div className="text-[11px] font-bold text-blue-600 uppercase tracking-wide">Tổng số lần chạy</div>
            </div>
            <div className="bg-purple-50 border border-purple-100 rounded-lg p-3 text-center">
              <div className="text-2xl font-black text-purple-700">{notesCount}</div>
              <div className="text-[11px] font-bold text-purple-600 uppercase tracking-wide">Ghi chú đã tạo</div>
            </div>
          </div>

          {worstRun && (
            <div className="bg-red-50 border border-red-100 rounded-lg p-3 flex items-center justify-between">
              <div className="flex flex-col">
                <span className="text-[10px] font-bold text-red-500 uppercase">Điểm yếu lớn nhất</span>
                <span className="text-sm font-semibold text-red-800">{worstRun.attack_name || 'Không xác định'}</span>
              </div>
              <div className="text-xl font-black text-red-600">
                -{worstRun.map_drop_pct?.toFixed(1) || 0}%
              </div>
            </div>
          )}

          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex gap-3 items-start text-amber-800 text-xs">
            <AlertTriangle className="w-5 h-5 shrink-0 text-amber-500" />
            <p><strong>Sau khi hoàn tất, phiên sẽ được khóa</strong> và chuyển sang trạng thái chỉ đọc. Bạn không thể thêm lần chạy mới hay sửa ghi chú.</p>
          </div>
        </div>

        <div className="p-4 border-t border-slate-100 bg-slate-50 flex gap-3 justify-end">
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="px-4 py-2 rounded-lg text-sm font-semibold text-slate-600 hover:bg-slate-200 transition-colors"
          >
            Hủy
          </button>
          <button
            onClick={handleConfirm}
            disabled={isSubmitting}
            className="px-4 py-2 rounded-lg text-sm font-bold bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm flex items-center gap-1.5 transition-colors disabled:opacity-70"
          >
            {isSubmitting ? "Đang xử lý..." : "🏁 Xác nhận hoàn tất"}
          </button>
        </div>
      </div>
    </div>
  );
}
