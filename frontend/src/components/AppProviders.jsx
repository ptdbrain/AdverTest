"use client";

import React from "react";
import { AuthProvider } from "@/context/AuthContext";
import { LanguageProvider } from "@/context/LanguageContext";
import { SidebarProvider } from "@/context/SidebarContext";
import AuthModal from "@/components/AuthModal";

export default function AppProviders({ children }) {
  return (
    <LanguageProvider>
      <AuthProvider>
        <SidebarProvider>
          {children}
          <AuthModal />
        </SidebarProvider>
      </AuthProvider>
    </LanguageProvider>
  );
}