"use client";

import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { getActiveProjectId, listProjects } from "@/lib/api";

const EMPTY = {
  projects: [],
  activeProjectId: "",
  activeProject: null,
  isLoadingProjects: false,
  setActiveProjectId: () => {},
  refreshProjects: async () => {},
};
const ProjectContext = createContext(EMPTY);

export function ProjectProvider({ children }) {
  const { isAuthenticated, isLoading } = useAuth();
  const [projects, setProjects] = useState([]);
  const [activeProjectId, setActiveProjectIdState] = useState("");
  const [isLoadingProjects, setIsLoadingProjects] = useState(false);

  const setActiveProjectId = useCallback((projectId) => {
    const value = projectId || "";
    setActiveProjectIdState(value);
    if (typeof window !== "undefined") {
      if (value) window.localStorage.setItem("advertest.activeProjectId", value);
      else window.localStorage.removeItem("advertest.activeProjectId");
      window.dispatchEvent(new Event("advertest-project-changed"));
    }
  }, []);

  const refreshProjects = useCallback(async () => {
    if (!isAuthenticated) {
      setProjects([]);
      setActiveProjectIdState("");
      return [];
    }
    setIsLoadingProjects(true);
    try {
      const items = await listProjects();
      const available = Array.isArray(items) ? items : [];
      setProjects(available);
      const saved = getActiveProjectId();
      const next = available.some((project) => project.id === saved) ? saved : available[0]?.id || "";
      setActiveProjectIdState(next);
      if (next && next !== saved && typeof window !== "undefined")
        window.localStorage.setItem("advertest.activeProjectId", next);
      return available;
    } finally {
      setIsLoadingProjects(false);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    if (!isLoading)
      refreshProjects().catch(() => {
        setProjects([]);
      });
  }, [isLoading, refreshProjects]);
  useEffect(() => {
    const sync = () => {
      refreshProjects().catch(() => {});
    };
    window.addEventListener("advertest-project-changed", sync);
    return () => window.removeEventListener("advertest-project-changed", sync);
  }, [refreshProjects]);

  const activeProject = projects.find((project) => project.id === activeProjectId) || null;
  return (
    <ProjectContext.Provider
      value={{ projects, activeProjectId, activeProject, isLoadingProjects, setActiveProjectId, refreshProjects }}
    >
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject() {
  return useContext(ProjectContext);
}
