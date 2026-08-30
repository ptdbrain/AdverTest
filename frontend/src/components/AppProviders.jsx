"use client";

import React from "react";
import { AuthProvider } from "@/context/AuthContext";
import { LanguageProvider } from "@/context/LanguageContext";
import { SidebarProvider } from "@/context/SidebarContext";
import { ProjectProvider } from "@/context/ProjectContext";
import AuthModal from "@/components/AuthModal";

export default function AppProviders({ children }) {
  return (
    <LanguageProvider>
      <AuthProvider>
        <React.Suspense fallback={<div role="status" className="min-h-screen bg-slate-50" />}>
          <ProjectProvider>
            <SidebarProvider>
              {children}
              <AuthModal />
            </SidebarProvider>
          </ProjectProvider>
        </React.Suspense>
      </AuthProvider>
    </LanguageProvider>
  );
}
