"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import { useLanguage } from "@/context/LanguageContext";

export default function AuthModal() {
  const {
    isAuthModalOpen,
    closeAuthModal,
    googleConfig,
    loginWithGoogle,
    loginWithDemoProfile,
    loginWithCredentials,
    isLoading,
  } = useAuth();
  const { t } = useLanguage();

  const [activeTab, setActiveTab] = useState("google");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  // Load Google Identity Services SDK if Client ID is configured
  useEffect(() => {
    if (!isAuthModalOpen || !googleConfig.client_id) return;

    const script = document.createElement("script");
    script.src = "https://accounts.google.com/gsi/client";
    script.async = true;
    script.defer = true;
    script.onload = () => {
      if (window.google?.accounts?.id) {
        window.google.accounts.id.initialize({
          client_id: googleConfig.client_id,
          callback: (response) => {
            if (response.credential) {
              loginWithGoogle(response.credential).catch((err) => setError(err.message));
            }
          },
        });
        const buttonEl = document.getElementById("google-signin-btn-container");
        if (buttonEl) {
          window.google.accounts.id.renderButton(buttonEl, {
            theme: "filled_blue",
            size: "large",
            width: 320,
            text: "signin_with",
            shape: "rectangular",
          });
        }
      }
    };
    document.body.appendChild(script);
    return () => {
      if (document.body.contains(script)) {
        document.body.removeChild(script);
      }
    };
  }, [isAuthModalOpen, googleConfig.client_id, loginWithGoogle]);

  if (!isAuthModalOpen) return null;

  const handlePasswordLogin = async (e) => {
    e.preventDefault();
    setError("");
    try {
      await loginWithCredentials(email, password);
    } catch (err) {
      setError(err.message || t("auth.loginFailed"));
    }
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "rgba(3, 7, 18, 0.75)",
        backdropFilter: "blur(6px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "16px",
      }}
      onClick={closeAuthModal}
    >
      <div
        style={{
          background: "var(--bg-elevated, #131B2A)",
          border: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.2))",
          borderRadius: "12px",
          width: "100%",
          maxWidth: "440px",
          padding: "24px 28px",
          boxShadow: "0 20px 40px rgba(0, 0, 0, 0.5)",
          display: "flex",
          flexDirection: "column",
          gap: "18px",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-primary, #FFF)", margin: 0 }}>
              {t("auth.title")}
            </h2>
            <p style={{ fontSize: "0.78rem", color: "var(--text-muted, #94A3B8)", margin: "4px 0 0 0" }}>
              {t("auth.subtitle")}
            </p>
          </div>
          <button
            type="button"
            onClick={closeAuthModal}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-muted)",
              fontSize: "1.2rem",
              cursor: "pointer",
              lineHeight: 1,
            }}
          >
            ✕
          </button>
        </div>

        {/* Tab Buttons */}
        <div
          style={{
            display: "flex",
            background: "var(--bg-primary, #0B0F17)",
            padding: "3px",
            borderRadius: "6px",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <button
            type="button"
            onClick={() => setActiveTab("google")}
            style={{
              flex: 1,
              padding: "8px 10px",
              border: "none",
              borderRadius: "4px",
              background: activeTab === "google" ? "var(--bg-elevated, #1E293B)" : "transparent",
              color: activeTab === "google" ? "var(--text-primary)" : "var(--text-muted)",
              fontWeight: 700,
              fontSize: "0.78rem",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "6px",
            }}
          >
            <span>{t("auth.tab.google")}</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("password")}
            style={{
              flex: 1,
              padding: "8px 10px",
              border: "none",
              borderRadius: "4px",
              background: activeTab === "password" ? "var(--bg-elevated, #1E293B)" : "transparent",
              color: activeTab === "password" ? "var(--text-primary)" : "var(--text-muted)",
              fontWeight: 700,
              fontSize: "0.78rem",
              cursor: "pointer",
            }}
          >
            <span>{t("auth.tab.password")}</span>
          </button>
        </div>

        {error && (
          <div
            style={{
              padding: "8px 12px",
              background: "rgba(239, 68, 68, 0.15)",
              border: "1px solid rgba(239, 68, 68, 0.3)",
              color: "#EF4444",
              borderRadius: "6px",
              fontSize: "0.75rem",
              fontWeight: 600,
            }}
          >
            {error}
          </div>
        )}

        {/* Tab 1: Google SSO */}
        {activeTab === "google" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            {/* Google Sign-in Container for Live Client ID */}
            {googleConfig?.configured ? (
              <div style={{ display: "flex", justifyContent: "center" }} id="google-signin-btn-container" />
            ) : null}

            {/* Quick-Access Google Profiles (One-Click SSO simulation for test review) */}
            <div>
              <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-secondary)", marginBottom: "8px" }}>
                {t("auth.demoProfiles")}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {(googleConfig?.demo_profiles || []).map((prof) => (
                  <button
                    key={prof.email}
                    type="button"
                    onClick={() => loginWithDemoProfile(prof).catch((e) => setError(e.message))}
                    disabled={isLoading}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "12px",
                      padding: "10px 14px",
                      borderRadius: "8px",
                      background: "var(--bg-primary, #0B0F17)",
                      border: "1px solid var(--border-subtle)",
                      cursor: "pointer",
                      textAlign: "left",
                      transition: "all 0.15s ease",
                    }}
                  >
                    <div
                      style={{
                        width: "34px",
                        height: "34px",
                        borderRadius: "50%",
                        background: "#EA4335",
                        color: "#FFF",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontWeight: 800,
                        fontSize: "0.9rem",
                        flexShrink: 0,
                      }}
                    >
                      {prof.display_name.charAt(0)}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: "0.82rem", fontWeight: 700, color: "var(--text-primary)" }}>
                        {prof.display_name}
                      </div>
                      <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {prof.email}
                      </div>
                    </div>
                    <span
                      style={{
                        fontSize: "0.65rem",
                        fontWeight: 700,
                        padding: "2px 6px",
                        borderRadius: "4px",
                        background: prof.role === "ADMIN" ? "rgba(168, 85, 247, 0.15)" : "rgba(56, 189, 248, 0.15)",
                        color: prof.role === "ADMIN" ? "#A855F7" : "#38BDF8",
                      }}
                    >
                      {prof.role}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Email & Password Form */}
        {activeTab === "password" && (
          <form onSubmit={handlePasswordLogin} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div>
              <label className="config-panel__label" style={{ fontSize: "0.72rem" }}>{t("auth.email")}</label>
              <input
                type="email"
                required
                className="select-field"
                placeholder="admin@advertest.ai"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                style={{ width: "100%", padding: "8px 12px", fontSize: "0.8rem" }}
              />
            </div>

            <div>
              <label className="config-panel__label" style={{ fontSize: "0.72rem" }}>{t("auth.password")}</label>
              <input
                type="password"
                required
                className="select-field"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{ width: "100%", padding: "8px 12px", fontSize: "0.8rem" }}
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="run-button"
              style={{ marginTop: "6px", padding: "10px", fontSize: "0.85rem", fontWeight: 700 }}
            >
              {isLoading ? t("auth.authenticating") : t("auth.signIn")}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
