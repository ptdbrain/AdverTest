"use client";

import React from "react";
import { AuthProvider } from "@/context/AuthContext";
import { LanguageProvider } from "@/context/LanguageContext";
import AuthModal from "@/components/AuthModal";

export default function AppProviders({ children }) {
  return (
    <LanguageProvider>
      <AuthProvider>
        {children}
        <AuthModal />
      </AuthProvider>
    </LanguageProvider>
  );
}