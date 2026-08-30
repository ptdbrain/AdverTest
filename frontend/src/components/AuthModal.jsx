"use client";

import React, { useState, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import { useLanguage } from "@/context/LanguageContext";

export default function AuthModal({ isOpen: propsIsOpen, onClose: propsOnClose }) {
  const {
    isAuthModalOpen: contextIsOpen,
    closeAuthModal: contextClose,
    googleConfig,
    loginWithGoogle,
    loginWithCredentials,
    registerWithCredentials,
    isLoading,
  } = useAuth();
  const { t } = useLanguage();

  const isOpen = propsIsOpen !== undefined ? propsIsOpen : contextIsOpen;
  const handleClose = propsOnClose || contextClose;

  const [activeTab, setActiveTab] = useState("login"); // "google" | "login" | "register"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");

  // Load Google Identity Services SDK if Client ID is configured
  useEffect(() => {
    if (!isOpen || !googleConfig?.client_id) return;

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
  }, [isOpen, googleConfig?.client_id, loginWithGoogle]);

  if (!isOpen) return null;

  const handlePasswordLogin = async (e) => {
    e.preventDefault();
    setError("");
    try {
      await loginWithCredentials(email, password);
    } catch (err) {
      setError(err.message || t("auth.loginFailed"));
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError("");
    try {
      if (registerWithCredentials) {
        await registerWithCredentials({
          email,
          password,
          display_name: displayName || email.split("@")[0],
        });
      }
    } catch (err) {
      setError(err.message || "Đăng ký không thành công. Vui lòng kiểm tra lại thông tin.");
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
      onClick={handleClose}
    >
      <div
        style={{
          background: "var(--bg-elevated, #131B2A)",
          border: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.2))",
          borderRadius: "12px",
          width: "100%",
          maxWidth: "460px",
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
              {activeTab === "register" ? "Đăng ký tài khoản mới" : t("auth.title")}
            </h2>
            <p style={{ fontSize: "0.78rem", color: "var(--text-muted, #94A3B8)", margin: "4px 0 0 0" }}>
              {activeTab === "register" ? "Tạo tài khoản để lưu trữ và quản lý thí nghiệm đối kháng" : t("auth.subtitle")}
            </p>
          </div>
          {googleConfig?.configured && <button
            type="button"
            onClick={handleClose}
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
          </button>}
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
            onClick={() => { setActiveTab("google"); setError(""); }}
            style={{
              flex: 1,
              padding: "8px 8px",
              border: "none",
              borderRadius: "4px",
              background: activeTab === "google" ? "var(--bg-elevated, #1E293B)" : "transparent",
              color: activeTab === "google" ? "var(--text-primary)" : "var(--text-muted)",
              fontWeight: 700,
              fontSize: "0.75rem",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "4px",
            }}
          >
            <span>Google SSO</span>
          </button>
          <button
            type="button"
            onClick={() => { setActiveTab("login"); setError(""); }}
            style={{
              flex: 1,
              padding: "8px 8px",
              border: "none",
              borderRadius: "4px",
              background: activeTab === "login" ? "var(--bg-elevated, #1E293B)" : "transparent",
              color: activeTab === "login" ? "var(--text-primary)" : "var(--text-muted)",
              fontWeight: 700,
              fontSize: "0.75rem",
              cursor: "pointer",
            }}
          >
            <span>Đăng nhập</span>
          </button>
          <button
            type="button"
            onClick={() => { setActiveTab("register"); setError(""); }}
            style={{
              flex: 1,
              padding: "8px 8px",
              border: "none",
              borderRadius: "4px",
              background: activeTab === "register" ? "var(--bg-elevated, #1E293B)" : "transparent",
              color: activeTab === "register" ? "var(--text-primary)" : "var(--text-muted)",
              fontWeight: 700,
              fontSize: "0.75rem",
              cursor: "pointer",
            }}
          >
            <span>Đăng ký</span>
          </button>
        </div>

        {error && (
          <div
            style={{
              background: "rgba(239, 68, 68, 0.15)",
              border: "1px solid rgba(239, 68, 68, 0.4)",
              borderRadius: "6px",
              padding: "10px 12px",
              fontSize: "0.75rem",
              color: "#FCA5A5",
            }}
          >
            {error}
          </div>
        )}

        {/* Tab 1: Google SSO & 1-Click Demo Profiles */}
        {activeTab === "google" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {googleConfig?.configured ? (
              <div style={{ display: "flex", justifyContent: "center", margin: "8px 0" }}>
                <div id="google-signin-btn-container" />
              </div>
            ) : <div role="status" style={{ color: "var(--text-muted)", textAlign: "center", fontSize: "0.75rem" }}>{t("auth.googleNotConfigured")}</div>}
          </div>
        )}

        {/* Tab 2: Email & Password Login */}
        {activeTab === "login" && (
          <form onSubmit={handlePasswordLogin} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div>
              <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
                Email
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="admin@advertest.ai"
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid var(--border-subtle)",
                  background: "var(--bg-primary, #0B0F17)",
                  color: "var(--text-primary, #FFF)",
                  fontSize: "0.8rem",
                  boxSizing: "border-box",
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>
                Mật khẩu
              </label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid var(--border-subtle)",
                  background: "var(--bg-primary, #0B0F17)",
                  color: "var(--text-primary, #FFF)",
                  fontSize: "0.8rem",
                  boxSizing: "border-box",
                }}
              />
            </div>
            <button
              type="submit"
              disabled={isLoading}
              style={{
                marginTop: "6px",
                padding: "10px",
                borderRadius: "6px",
                border: "none",
                background: "var(--primary-accent, #3B82F6)",
                color: "#FFF",
                fontWeight: 700,
                fontSize: "0.8rem",
                cursor: "pointer",
                boxShadow: "0 2px 8px rgba(59, 130, 246, 0.4)",
              }}
            >
              {isLoading ? "Đang đăng nhập..." : t("auth.signIn")}
            </button>
          </form>
        )}

        {/* Tab 3: Register New Account */}
        {activeTab === "register" && (
          <form onSubmit={handleRegister} style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <div>
              <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: "3px" }}>
                Họ và Tên (Display Name)
              </label>
              <input
                type="text"
                required
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Nguyen Van A"
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid var(--border-subtle)",
                  background: "var(--bg-primary, #0B0F17)",
                  color: "var(--text-primary, #FFF)",
                  fontSize: "0.8rem",
                  boxSizing: "border-box",
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: "3px" }}>
                Email
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="user@organization.vn"
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid var(--border-subtle)",
                  background: "var(--bg-primary, #0B0F17)",
                  color: "var(--text-primary, #FFF)",
                  fontSize: "0.8rem",
                  boxSizing: "border-box",
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: "3px" }}>
                Mật khẩu
              </label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                style={{
                  width: "100%",
                  padding: "8px 10px",
                  borderRadius: "6px",
                  border: "1px solid var(--border-subtle)",
                  background: "var(--bg-primary, #0B0F17)",
                  color: "var(--text-primary, #FFF)",
                  fontSize: "0.8rem",
                  boxSizing: "border-box",
                }}
              />
            </div>
            <button
              type="submit"
              disabled={isLoading}
              style={{
                marginTop: "6px",
                padding: "10px",
                borderRadius: "6px",
                border: "none",
                background: "var(--primary-accent, #3B82F6)",
                color: "#FFF",
                fontWeight: 700,
                fontSize: "0.8rem",
                cursor: "pointer",
                boxShadow: "0 2px 8px rgba(59, 130, 246, 0.4)",
              }}
            >
              {isLoading ? "Đang tạo tài khoản..." : "Đăng ký & Đăng nhập ngay"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
