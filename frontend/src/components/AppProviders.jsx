"use client";

import React from "react";
import { AuthProvider } from "@/context/AuthContext";
import { LanguageProvider } from "@/context/LanguageContext";
import { SidebarProvider } from "@/context/SidebarContext";
import { ThemeProvider } from "@/context/ThemeContext";
import { ProjectProvider } from "@/context/ProjectContext";
import AuthModal from "@/components/AuthModal";

export default function AppProviders({ children }) {
  return (
    <ThemeProvider><LanguageProvider>
      <AuthProvider>
        <ProjectProvider>
          <SidebarProvider>
            {children}
            <AuthModal />
          </SidebarProvider>
        </ProjectProvider>
      </AuthProvider>
    </LanguageProvider></ThemeProvider>
  );
}
