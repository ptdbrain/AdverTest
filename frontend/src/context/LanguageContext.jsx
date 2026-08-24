"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { translate } from "@/lib/i18n";

const LanguageContext = createContext(null);

/**
 * Ngôn ngữ hệ thống (vi/en) cho toàn bộ UI. Persist client-side ở
 * `localStorage.advertest_lang` (đúng pattern theme/auth). Tự set `<html lang>`
 * để giữ đồng bộ với trình duyệt, và cung cấp hàm `t(key, params)`.
 */
export function LanguageProvider({ children }) {
  const [lang, setLangState] = useState("vi");

  // Nạp lựa chọn đã lưu trên mount (tránh lệch với bootstrap script trong layout).
  useEffect(() => {
    try {
      const saved = localStorage.getItem("advertest_lang") || "vi";
      const next = saved === "en" ? "en" : "vi";
      setLangState(next);
      document.documentElement.setAttribute("lang", next);
    } catch {}
  }, []);

  const setLang = useCallback((l) => {
    const next = l === "en" ? "en" : "vi";
    setLangState(next);
    try {
      localStorage.setItem("advertest_lang", next);
      document.documentElement.setAttribute("lang", next);
    } catch {}
  }, []);

  const t = useCallback((key, params) => translate(lang, key, params), [lang]);

  return (
    <LanguageContext.Provider value={{ lang, setLang, t }}>{children}</LanguageContext.Provider>
  );
}

export function useLanguage() {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error("useLanguage phải được dùng bên trong <LanguageProvider>");
  }
  return ctx;
}