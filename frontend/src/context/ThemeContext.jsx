"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

const ThemeContext = createContext(null);
const STORAGE_KEY = "advertest_theme";

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState("system");
  const [systemTheme, setSystemTheme] = useState("light");
  const resolvedTheme = theme === "system" ? systemTheme : theme;

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const updateSystemTheme = () => setSystemTheme(media.matches ? "dark" : "light");
    const storedTheme = localStorage.getItem(STORAGE_KEY);
    if (["light", "dark", "system"].includes(storedTheme)) setTheme(storedTheme);
    updateSystemTheme();
    media.addEventListener("change", updateSystemTheme);
    return () => media.removeEventListener("change", updateSystemTheme);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = resolvedTheme;
  }, [resolvedTheme]);

  const value = useMemo(() => ({ theme, resolvedTheme, setTheme: (nextTheme) => {
    if (!["light", "dark", "system"].includes(nextTheme)) return;
    localStorage.setItem(STORAGE_KEY, nextTheme);
    setTheme(nextTheme);
  } }), [resolvedTheme, theme]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context) return context;
  // Keep utility components renderable in isolated tests/embedders while the
  // application itself always mounts ThemeProvider through AppProviders.
  return { theme: "light", resolvedTheme: "light", setTheme: () => {} };
}
