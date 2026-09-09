"use client";

import React, { createContext, useContext, useEffect, useState } from "react";

const SidebarContext = createContext({
  isCollapsed: false,
  toggleSidebar: () => {},
  setCollapsed: () => {},
});

export function SidebarProvider({ children }) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem("adversai_sidebar_collapsed");
      if (saved !== null) {
        setIsCollapsed(JSON.parse(saved));
      }
    } catch {}
  }, []);

  const toggleSidebar = () => {
    setIsCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem("adversai_sidebar_collapsed", JSON.stringify(next));
      } catch {}
      return next;
    });
  };

  const setCollapsed = (val) => {
    setIsCollapsed(val);
    try {
      localStorage.setItem("adversai_sidebar_collapsed", JSON.stringify(val));
    } catch {}
  };

  return (
    <SidebarContext.Provider value={{ isCollapsed, toggleSidebar, setCollapsed }}>{children}</SidebarContext.Provider>
  );
}

export function useSidebar() {
  return useContext(SidebarContext);
}
