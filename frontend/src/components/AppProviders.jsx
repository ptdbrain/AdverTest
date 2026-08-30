"use client";

import React from "react";
import { AuthProvider } from "@/context/AuthContext";
import { LanguageProvider } from "@/context/LanguageContext";
import { SidebarProvider } from "@/context/SidebarContext";
import { ThemeProvider } from "@/context/ThemeContext";
import AuthModal from "@/components/AuthModal";

export default function AppProviders({ children }) {
  return (
    <ThemeProvider><LanguageProvider>
      <AuthProvider>
        <SidebarProvider>
          {children}
          <AuthModal />
        </SidebarProvider>
      </AuthProvider>
    </LanguageProvider></ThemeProvider>
  );
}
