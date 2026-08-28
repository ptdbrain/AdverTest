"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { updateRunNote } from "@/lib/api";

export default function ResearcherNotePanel({ sessionId, runId, initialNote = "" }) {
  const [note, setNote] = useState(initialNote);
  const [saveStatus, setSaveStatus] = useState(""); // "saving", "saved", "error", ""
  const timeoutRef = useRef(null);

  useEffect(() => {
    setNote(initialNote);
    setSaveStatus("");
  }, [runId, initialNote]);

  const saveNote = useCallback(async (content) => {
    if (!sessionId || !runId) return;
    setSaveStatus("saving");
    try {
      await updateRunNote(sessionId, runId, content);
      setSaveStatus("saved");
      setTimeout(() => setSaveStatus(""), 2000);
    } catch (err) {
      console.error("Failed to save note:", err);
      setSaveStatus("error");
    }
  }, [sessionId, runId]);

  const handleChange = (e) => {
    const val = e.target.value;
    setNote(val);
    
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    
    timeoutRef.current = setTimeout(() => {
      saveNote(val);
    }, 1500);
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col mt-5">
      <div className="bg-slate-50 border-b border-slate-200 px-4 py-2 flex items-center justify-between">
        <h3 className="text-xs font-bold text-slate-700">Ghi chú & Nhận xét khoa học</h3>
        <div className="text-[10px] font-semibold h-4 flex items-center">
          {saveStatus === "saving" && <span className="text-blue-600">Đang lưu...</span>}
          {saveStatus === "saved" && <span className="text-emerald-600">Đã lưu ✓</span>}
          {saveStatus === "error" && <span className="text-red-600">Lỗi ✗</span>}
        </div>
      </div>
      <textarea
        className="w-full p-4 text-xs text-slate-800 focus:outline-none resize-none min-h-[100px] placeholder-slate-400"
        placeholder="Ghi chú nhận xét khoa học cho lần chạy này..."
        value={note}
        onChange={handleChange}
        disabled={!runId}
      />
    </div>
  );
}
