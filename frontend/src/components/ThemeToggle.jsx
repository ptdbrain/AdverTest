"use client";

import React, { useEffect, useState } from "react";

export default function ThemeToggle({ className = "", compact = false }) {
  const [theme, setTheme] = useState("dark");

  useEffect(() => {
    let updateTimer;
    try {
      const saved = localStorage.getItem("theme");
      const initial = saved || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
      document.documentElement.setAttribute("data-theme", initial);
      // Defer the UI state sync: React's effect rule intentionally rejects
      // a synchronous setState here because it causes an extra render pass.
      updateTimer = window.setTimeout(() => setTheme(initial), 0);
    } catch {
      // ignore
    }
    return () => window.clearTimeout(updateTimer);
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    try {
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("theme", next);
    } catch {
      // ignore
    }
  };

  const isLight = theme === "light";

  return (
    <button
      type="button"
      className={`theme-toggle-btn ${className}`}
      onClick={toggleTheme}
      title={isLight ? "Chuyển sang Dark mode" : "Chuyển sang Light mode"}
      aria-label={isLight ? "Switch to Dark mode" : "Switch to Light mode"}
      suppressHydrationWarning
    >
      {isLight ? (
        // Moon Icon (for switching to dark)
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          width="16"
          height="16"
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      ) : (
        // Sun Icon (for switching to light)
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          width="16"
          height="16"
        >
          <circle cx="12" cy="12" r="5" />
          <line x1="12" y1="1" x2="12" y2="3" />
          <line x1="12" y1="21" x2="12" y2="23" />
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
          <line x1="1" y1="12" x2="3" y2="12" />
          <line x1="21" y1="12" x2="23" y2="12" />
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
        </svg>
      )}
      {!compact && <span className="theme-toggle-btn__text">{isLight ? "Dark" : "Light"}</span>}
    </button>
  );
}
