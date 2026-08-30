"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import * as api from "@/lib/api";

const DEFAULT_AUTH_FALLBACK = {
  user: null,
  isAuthenticated: false,
  role: "ENGINEER",
  isLoading: false,
  googleConfig: { client_id: "", configured: false, demo_profiles: [] },
  isAuthModalOpen: false,
  openAuthModal: () => {},
  closeAuthModal: () => {},
  loginWithGoogle: async () => {},
  loginWithDemoProfile: async () => {},
  loginWithCredentials: async () => {},
  registerWithCredentials: async () => {},
  logout: () => {},
  switchRole: () => {},
};

const AuthContext = createContext(DEFAULT_AUTH_FALLBACK);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [googleConfig, setGoogleConfig] = useState({ client_id: "", configured: false, demo_profiles: [] });
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  // Load active user session on mount
  useEffect(() => {
    let active = true;
    let readSession = () => Promise.resolve(null);
    let hasSessionReader = false;
    let readGoogleConfig = () => Promise.resolve(null);
    try { if (typeof api.getAuthMe === "function") { readSession = api.getAuthMe; hasSessionReader = true; } } catch {}
    if (!hasSessionReader) {
      try { if (typeof api.getCurrentUser === "function") readSession = api.getCurrentUser; } catch {}
    }
    try { if (typeof api.getGoogleAuthConfig === "function") readGoogleConfig = api.getGoogleAuthConfig; } catch {}
    Promise.all([readSession().catch(() => null), readGoogleConfig().catch(() => null)]).then(([sessionUser, config]) => {
      if (!active) return;
      setUser(sessionUser);
      if (config) setGoogleConfig(config);
      setIsLoading(false);
    });

    return () => {
      active = false;
    };
  }, []);

  const handleAuthSuccess = useCallback((payload) => {
    if (payload?.user) {
      setUser(payload.user);
      setIsAuthModalOpen(false);
    }
  }, []);

  const loginWithGoogle = useCallback(async (googleCredential) => {
    setIsLoading(true);
    try {
      const res = await api.loginGoogleSSO({ credential: googleCredential });
      handleAuthSuccess(res);
      return res;
    } finally {
      setIsLoading(false);
    }
  }, [handleAuthSuccess]);

  const loginWithDemoProfile = useCallback(async (profile) => {
    setIsLoading(true);
    try {
      const res = await api.loginGoogleSSO({
        email: profile.email,
        display_name: profile.display_name,
        avatar_url: profile.avatar_url,
        role: profile.role,
      });
      handleAuthSuccess(res);
      return res;
    } finally {
      setIsLoading(false);
    }
  }, [handleAuthSuccess]);

  const loginWithCredentials = useCallback(async (email, password) => {
    setIsLoading(true);
    try {
      const res = await api.loginUser(email, password);
      handleAuthSuccess(res);
      return res;
    } finally {
      setIsLoading(false);
    }
  }, [handleAuthSuccess]);

  const registerWithCredentials = useCallback(async (payload) => {
    setIsLoading(true);
    try {
      const res = await api.registerUser(payload);
      handleAuthSuccess(res);
      return res;
    } finally {
      setIsLoading(false);
    }
  }, [handleAuthSuccess]);

  const logout = useCallback(async () => {
    await (api.logoutUser ? api.logoutUser() : Promise.resolve());
    setUser(null);
  }, []);

  const switchRole = useCallback((newRole) => {
    setUser((prev) => {
      if (!prev) return prev;
      return { ...prev, role: newRole };
    });
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        role: user?.role || "ENGINEER",
        isLoading,
        googleConfig,
        isAuthModalOpen,
        openAuthModal: () => setIsAuthModalOpen(true),
        closeAuthModal: () => setIsAuthModalOpen(false),
        loginWithGoogle,
        loginWithDemoProfile,
        loginWithCredentials,
        registerWithCredentials,
        logout,
        switchRole,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  return context || DEFAULT_AUTH_FALLBACK;
}
