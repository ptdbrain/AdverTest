"use client";

import React from "react";
import AuthModal from "@/components/AuthModal";
import { AuthProvider } from "@/context/AuthContext";
import { LanguageProvider } from "@/context/LanguageContext";
import { ProjectProvider } from "@/context/ProjectContext";
import { SidebarProvider } from "@/context/SidebarContext";
import { ThemeProvider } from "@/context/ThemeContext";

export default function AppProviders({ children }) {
  return (
    <ThemeProvider>
      <LanguageProvider>
        <AuthProvider>
          <ProjectProvider>
            <SidebarProvider>
              {children}
              <AuthModal />
            </SidebarProvider>
          </ProjectProvider>
        </AuthProvider>
      </LanguageProvider>
    </ThemeProvider>
  );
}
