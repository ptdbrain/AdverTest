"use client";

import React, { useEffect, useState } from "react";
import { useAuth } from "@/context/AuthContext";
import { useLanguage } from "@/context/LanguageContext";

function EyeIcon({ off }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {off ? (
        <>
          <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
          <line x1="1" y1="1" x2="23" y2="23" />
        </>
      ) : (
        <>
          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
          <circle cx="12" cy="12" r="3" />
        </>
      )}
    </svg>
  );
}

const AUTH_INPUT_STYLE = {
  width: "100%",
  padding: "8px 10px",
  borderRadius: "6px",
  border: "1px solid var(--border-subtle)",
  background: "var(--app-bg)",
  color: "var(--app-text)",
  fontSize: "0.8rem",
  boxSizing: "border-box",
};

/** Password input with an eye toggle that reveals the text currently being typed. */
function PasswordInput({ value, onChange, placeholder, required, autoComplete, minLength, show, onToggle }) {
  return (
    <div style={{ position: "relative" }}>
      <input
        type={show ? "text" : "password"}
        required={required}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        autoComplete={autoComplete}
        minLength={minLength}
        style={{ ...AUTH_INPUT_STYLE, paddingRight: "38px" }}
      />
      <button
        type="button"
        aria-label={show ? "Ẩn mật khẩu" : "Hiện mật khẩu"}
        aria-pressed={show}
        onClick={onToggle}
        style={{
          position: "absolute",
          right: "10px",
          top: "50%",
          transform: "translateY(-50%)",
          background: "none",
          border: "none",
          padding: "2px",
          cursor: "pointer",
          color: "var(--text-muted)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <EyeIcon off={show} />
      </button>
    </div>
  );
}

function DemoProfilesList({ profiles, isLoading, onSelect }) {
  if (!profiles || profiles.length === 0) return null;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
      <div
        style={{
          fontSize: "0.7rem",
          color: "var(--text-muted)",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.06em",
        }}
      >
        Tài khoản demo sẵn có — 1-Click đăng nhập
      </div>
      {profiles.map((profile) => (
        <button
          key={profile.email}
          type="button"
          disabled={isLoading}
          onClick={() => onSelect(profile)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            padding: "10px 12px",
            borderRadius: "8px",
            border: "1px solid var(--border-subtle)",
            background: "var(--app-bg)",
            cursor: "pointer",
            textAlign: "left",
          }}
        >
          <img
            src={profile.avatar_url}
            alt=""
            width={36}
            height={36}
            style={{ borderRadius: "50%", flexShrink: 0 }}
          />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--app-text)" }}>
              {profile.display_name}
            </div>
            <div style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>{profile.email}</div>
            <div style={{ fontSize: "0.62rem", color: "var(--text-muted)", opacity: 0.85 }}>
              Mật khẩu: {profile.password}
            </div>
          </div>
          <span
            style={{
              fontSize: "0.62rem",
              fontWeight: 800,
              padding: "3px 8px",
              borderRadius: "999px",
              background: profile.role === "ADMIN" ? "rgba(168, 85, 247, 0.15)" : "rgba(59, 130, 246, 0.15)",
              color: profile.role === "ADMIN" ? "#A855F7" : "#3B82F6",
              border: `1px solid ${profile.role === "ADMIN" ? "rgba(168, 85, 247, 0.35)" : "rgba(59, 130, 246, 0.35)"}`,
            }}
          >
            {profile.role}
          </span>
        </button>
      ))}
    </div>
  );
}

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
  const [confirmPassword, setConfirmPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [showLoginPassword, setShowLoginPassword] = useState(false);
  const [showRegisterPassword, setShowRegisterPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

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
    if (password.length < 8) {
      setError("Mật khẩu phải có ít nhất 8 ký tự.");
      return;
    }
    if (password !== confirmPassword) {
      setError("Mật khẩu nhập lại không khớp. Vui lòng kiểm tra lại.");
      return;
    }
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

  const handleDemoLogin = async (profile) => {
    setError("");
    try {
      await loginWithCredentials(profile.email, profile.password);
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
      onClick={handleClose}
    >
      <div
        style={{
          background: "var(--app-surface)",
          border: "1px solid var(--app-border)",
          borderRadius: "12px",
          width: "100%",
          maxWidth: "460px",
          padding: "24px 28px",
          boxShadow: "0 20px 40px rgba(15, 23, 42, 0.16)",
          display: "flex",
          flexDirection: "column",
          gap: "18px",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--app-text)", margin: 0 }}>
              {activeTab === "register" ? "Đăng ký tài khoản mới" : t("auth.title")}
            </h2>
            <p style={{ fontSize: "0.78rem", color: "var(--text-muted, #94A3B8)", margin: "4px 0 0 0" }}>
              {activeTab === "register"
                ? "Tạo tài khoản để lưu trữ và quản lý thí nghiệm đối kháng"
                : t("auth.subtitle")}
            </p>
          </div>
          {googleConfig?.configured && (
            <button
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
            </button>
          )}
        </div>

        {/* Tab Buttons */}
        <div
          style={{
            display: "flex",
            background: "var(--app-bg)",
            padding: "3px",
            borderRadius: "6px",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <button
            type="button"
            onClick={() => {
              setActiveTab("google");
              setError("");
            }}
            style={{
              flex: 1,
              padding: "8px 8px",
              border: "none",
              borderRadius: "4px",
              background: activeTab === "google" ? "var(--app-surface)" : "transparent",
              color: activeTab === "google" ? "var(--app-text)" : "var(--text-muted)",
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
            onClick={() => {
              setActiveTab("login");
              setError("");
            }}
            style={{
              flex: 1,
              padding: "8px 8px",
              border: "none",
              borderRadius: "4px",
              background: activeTab === "login" ? "var(--app-surface)" : "transparent",
              color: activeTab === "login" ? "var(--app-text)" : "var(--text-muted)",
              fontWeight: 700,
              fontSize: "0.75rem",
              cursor: "pointer",
            }}
          >
            <span>Đăng nhập</span>
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveTab("register");
              setError("");
            }}
            style={{
              flex: 1,
              padding: "8px 8px",
              border: "none",
              borderRadius: "4px",
              background: activeTab === "register" ? "var(--app-surface)" : "transparent",
              color: activeTab === "register" ? "var(--app-text)" : "var(--text-muted)",
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
              color: "#B91C1C",
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
            ) : (
              <div
                role="status"
                style={{
                  color: "var(--text-muted)",
                  textAlign: "center",
                  fontSize: "0.75rem",
                  background: "var(--app-bg)",
                  padding: "10px 12px",
                  borderRadius: "6px",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                Google SSO chưa được cấu hình. Bạn có thể sử dụng 1 trong 2 tài khoản demo 1-click dưới đây:
              </div>
            )}
            <DemoProfilesList
              profiles={googleConfig?.demo_profiles}
              isLoading={isLoading}
              onSelect={handleDemoLogin}
            />
          </div>
        )}

        {/* Tab 2: Email & Password Login */}
        {activeTab === "login" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            <DemoProfilesList
              profiles={googleConfig?.demo_profiles}
              isLoading={isLoading}
              onSelect={handleDemoLogin}
            />
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <div style={{ flex: 1, height: 1, background: "var(--border-subtle)" }} />
              <span style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>hoặc đăng nhập bằng email</span>
              <div style={{ flex: 1, height: 1, background: "var(--border-subtle)" }} />
            </div>
            <form onSubmit={handlePasswordLogin} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              <div>
                <label
                  style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}
                >
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
                    background: "var(--app-bg)",
                    color: "var(--app-text)",
                    fontSize: "0.8rem",
                    boxSizing: "border-box",
                  }}
                />
              </div>
              <div>
                <label
                  style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}
                >
                  Mật khẩu
                </label>
                <PasswordInput
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  autoComplete="current-password"
                  show={showLoginPassword}
                  onToggle={() => setShowLoginPassword(!showLoginPassword)}
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
                  background: "var(--primary)",
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
          </div>
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
                  background: "var(--app-bg)",
                  color: "var(--app-text)",
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
                  background: "var(--app-bg)",
                  color: "var(--app-text)",
                  fontSize: "0.8rem",
                  boxSizing: "border-box",
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: "3px" }}>
                Mật khẩu
              </label>
              <PasswordInput
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                autoComplete="new-password"
                minLength={8}
                show={showRegisterPassword}
                onToggle={() => setShowRegisterPassword(!showRegisterPassword)}
              />
              <div style={{ fontSize: "0.66rem", color: "var(--text-muted)", marginTop: "3px" }}>
                Tối thiểu 8 ký tự
              </div>
            </div>
            <div>
              <label style={{ fontSize: "0.72rem", color: "var(--text-muted)", display: "block", marginBottom: "3px" }}>
                Nhập lại mật khẩu
              </label>
              <PasswordInput
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="••••••••"
                autoComplete="new-password"
                show={showConfirmPassword}
                onToggle={() => setShowConfirmPassword(!showConfirmPassword)}
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
                background: "var(--primary)",
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
