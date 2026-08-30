"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import { getActiveProjectId, listProjects, setActiveProjectId } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

const ProjectContext = createContext(null);

export function ProjectProvider({ children }) {
  const { isAuthenticated } = useAuth();
  const pathname = usePathname() || "/";
  const router = useRouter();
  const replaceRoute = router.replace;
  const params = useSearchParams();
  const requestedProjectId = params?.get("project_id") || null;
  const requestedRunId = params?.get("run_id") || null;
  const [projects, setProjects] = useState([]);
  const [projectId, setProjectId] = useState(null);

  useEffect(() => {
    if (!isAuthenticated) { setProjects([]); setProjectId(null); return; }
    listProjects().then((items) => {
      const available = Array.isArray(items) ? items : [];
      setProjects(available);
      const selected = available.some((item) => item.id === requestedProjectId)
        ? requestedProjectId
        : available.some((item) => item.id === getActiveProjectId())
          ? getActiveProjectId()
          : available[0]?.id || null;
      setProjectId(selected);
      if (selected) setActiveProjectId(selected);
      if (selected && requestedProjectId !== selected) {
        const next = new URLSearchParams(params?.toString() || "");
        next.set("project_id", selected);
        if (requestedProjectId || requestedRunId) next.delete("run_id");
        replaceRoute(`${pathname}?${next.toString()}`);
      }
    }).catch(() => { setProjects([]); setProjectId(null); });
  }, [isAuthenticated, pathname, requestedProjectId, requestedRunId, replaceRoute]);

  const scopedHref = useCallback((path, runId = null) => {
    const url = new URL(path, window.location.origin);
    if (projectId) url.searchParams.set("project_id", projectId);
    if (runId) url.searchParams.set("run_id", runId);
    return `${url.pathname}${url.search}`;
  }, [projectId]);

  const selectProject = useCallback((nextProjectId) => {
    setProjectId(nextProjectId || null);
    setActiveProjectId(nextProjectId || null);
    const url = new URL(window.location.href);
    if (nextProjectId) url.searchParams.set("project_id", nextProjectId);
    else url.searchParams.delete("project_id");
    url.searchParams.delete("run_id");
    replaceRoute(`${pathname}${url.search}`);
  }, [pathname, replaceRoute]);

  const value = useMemo(() => ({ projectId, projects, selectProject, scopedHref }), [projectId, projects, selectProject, scopedHref]);
  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>;
}

export function useProjectContext() {
  const value = useContext(ProjectContext);
  return value || { projectId: null, projects: [], selectProject: () => {}, scopedHref: (path) => path };
}
