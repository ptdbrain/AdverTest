"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { loginGoogleSSO, getGoogleAuthConfig, getAuthMe, loginUser, registerUser } from "@/lib/api";

const DEFAULT_AUTH_FALLBACK = {
  user: null,
  token: null,
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
  const [token, setToken] = useState(null);
  const [googleConfig, setGoogleConfig] = useState({ client_id: "", configured: false, demo_profiles: [] });
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  // Load active user session on mount
  useEffect(() => {
    let active = true;
    try {
      const savedToken = localStorage.getItem("advertest_auth_token");
      const savedUser = localStorage.getItem("advertest_auth_user");
      if (savedToken && savedUser) {
        setToken(savedToken);
        setUser(JSON.parse(savedUser));
      }
    } catch {}

    getGoogleAuthConfig()
      .then((cfg) => {
        if (active && cfg) setGoogleConfig(cfg);
      })
      .catch(console.warn)
      .finally(() => {
        if (active) setIsLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const handleAuthSuccess = useCallback((payload) => {
    if (payload?.access_token && payload?.user) {
      setToken(payload.access_token);
      setUser(payload.user);
      try {
        localStorage.setItem("advertest_auth_token", payload.access_token);
        localStorage.setItem("advertest_auth_user", JSON.stringify(payload.user));
      } catch {}
      setIsAuthModalOpen(false);
    }
  }, []);

  const loginWithGoogle = useCallback(async (googleCredential) => {
    setIsLoading(true);
    try {
      const res = await loginGoogleSSO({ credential: googleCredential });
      handleAuthSuccess(res);
      return res;
    } finally {
      setIsLoading(false);
    }
  }, [handleAuthSuccess]);

  const loginWithDemoProfile = useCallback(async (profile) => {
    setIsLoading(true);
    try {
      const res = await loginGoogleSSO({
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
      const res = await loginUser(email, password);
      handleAuthSuccess(res);
      return res;
    } finally {
      setIsLoading(false);
    }
  }, [handleAuthSuccess]);

  const registerWithCredentials = useCallback(async (payload) => {
    setIsLoading(true);
    try {
      const res = await registerUser(payload);
      handleAuthSuccess(res);
      return res;
    } finally {
      setIsLoading(false);
    }
  }, [handleAuthSuccess]);

  const logout = useCallback(() => {
    setUser(null);
    setToken(null);
    try {
      localStorage.removeItem("advertest_auth_token");
      localStorage.removeItem("advertest_auth_user");
    } catch {}
  }, []);

  const switchRole = useCallback((newRole) => {
    setUser((prev) => {
      if (!prev) return prev;
      const updated = { ...prev, role: newRole };
      try {
        localStorage.setItem("advertest_auth_user", JSON.stringify(updated));
      } catch {}
      return updated;
    });
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
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
