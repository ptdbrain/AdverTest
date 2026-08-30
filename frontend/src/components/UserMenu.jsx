"use client";

import React, { useState, useRef, useEffect } from "react";
import { useAuth } from "@/context/AuthContext";
import { useLanguage } from "@/context/LanguageContext";

export default function UserMenu() {
  const { user, isAuthenticated, openAuthModal, logout, switchRole } = useAuth();
  const { t } = useLanguage();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (!isAuthenticated || !user) {
    return (
      <button
        type="button"
        onClick={openAuthModal}
        className="action-button"
        style={{
          background: "linear-gradient(135deg, #4285F4 0%, #1A73E8 100%)",
          color: "#FFF",
          border: "none",
          fontSize: "0.75rem",
          fontWeight: 700,
          padding: "5px 12px",
          borderRadius: "6px",
          display: "flex",
          alignItems: "center",
          gap: "6px",
          cursor: "pointer",
          boxShadow: "0 2px 6px rgba(66, 133, 244, 0.3)",
        }}
      >
        <svg width="14" height="14" viewBox="0 0 24 24">
          <path
            fill="#FFF"
            d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
          />
          <path
            fill="#FFF"
            d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
          />
          <path
            fill="#FFF"
            d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
          />
          <path
            fill="#FFF"
            d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
          />
        </svg>
        <span>{t("usermenu.loginGoogle")}</span>
      </button>
    );
  }

  const roleLabels = {
    ENGINEER: "👨‍💻 ML Engineer",
    ADMIN: "👑 Administrator",
  };

  const roleColors = {
    ENGINEER: "#38BDF8",
    ADMIN: "#A855F7",
  };

  const currentRole = user.role || "ENGINEER";

  return (
    <div style={{ position: "relative" }} ref={menuRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        style={{
          background: "var(--bg-elevated, #101623)",
          border: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.2))",
          borderRadius: "20px",
          padding: "3px 10px 3px 4px",
          display: "flex",
          alignItems: "center",
          gap: "8px",
          cursor: "pointer",
        }}
      >
        <div
          style={{
            width: "26px",
            height: "26px",
            borderRadius: "50%",
            background: "#EA4335",
            color: "#FFF",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 800,
            fontSize: "0.75rem",
            overflow: "hidden",
          }}
        >
          {user.avatar_url ? (
            <img src={user.avatar_url} alt={user.display_name} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          ) : (
            user.display_name?.charAt(0) || "U"
          )}
        </div>
        <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-primary)" }}>
          {user.display_name?.split(" ")[0]}
        </span>
        <span
          style={{
            fontSize: "0.62rem",
            fontWeight: 700,
            padding: "1px 5px",
            borderRadius: "3px",
            background: `${roleColors[currentRole] || "#38BDF8"}20`,
            color: roleColors[currentRole] || "#38BDF8",
          }}
        >
          {currentRole}
        </span>
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 6px)",
            right: 0,
            zIndex: 999,
            background: "var(--bg-elevated, #131B2A)",
            border: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.2))",
            borderRadius: "8px",
            width: "240px",
            padding: "10px",
            boxShadow: "0 10px 25px rgba(0, 0, 0, 0.5)",
            display: "flex",
            flexDirection: "column",
            gap: "10px",
          }}
        >
          {/* User Info Header */}
          <div style={{ borderBottom: "1px solid var(--border-subtle)", paddingBottom: "8px" }}>
            <div style={{ fontSize: "0.82rem", fontWeight: 700, color: "var(--text-primary)" }}>
              {user.display_name}
            </div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", overflow: "hidden", textOverflow: "ellipsis" }}>
              {user.email}
            </div>
            <div style={{ fontSize: "0.65rem", color: "#4285F4", fontWeight: 600, marginTop: "2px" }}>
              {t("usermenu.signedInGoogle")}
            </div>
          </div>

          {/* Account Role Display */}
          <div style={{ background: "var(--bg-primary)", padding: "8px 10px", borderRadius: "6px", border: "1px solid var(--border-subtle)" }}>
            <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", fontWeight: 600 }}>{t("usermenu.role")}</div>
            <div style={{ fontSize: "0.78rem", fontWeight: 700, color: roleColors[currentRole] || "var(--text-primary)", marginTop: "2px" }}>
              {roleLabels[currentRole] || currentRole}
            </div>
          </div>

          {/* Settings, Admin Link & Logout */}
          <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "8px", display: "flex", flexDirection: "column", gap: "4px" }}>
            <a
              href="/settings"
              style={{
                fontSize: "0.72rem",
                color: "var(--text-primary)",
                textDecoration: "none",
                padding: "5px 8px",
                borderRadius: "4px",
                background: "rgba(56, 189, 248, 0.08)",
                display: "flex",
                alignItems: "center",
                gap: "6px",
                fontWeight: 600,
              }}
            >
              <span>{t("usermenu.settings")}</span>
            </a>
            <a
              href="/admin"
              style={{
                fontSize: "0.72rem",
                color: "var(--text-secondary)",
                textDecoration: "none",
                padding: "4px 8px",
                borderRadius: "4px",
              }}
            >
              {t("usermenu.adminDashboard")}
            </a>
            <button
              type="button"
              onClick={() => {
                logout();
                setIsOpen(false);
              }}
              style={{
                background: "none",
                border: "none",
                color: "var(--danger, #EF4444)",
                fontSize: "0.72rem",
                fontWeight: 700,
                textAlign: "left",
                padding: "4px 8px",
                borderRadius: "4px",
                cursor: "pointer",
              }}
            >
              {t("usermenu.logout")}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
