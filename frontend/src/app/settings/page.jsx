"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import { useLanguage } from "@/context/LanguageContext";
import UserMenu from "@/components/UserMenu.jsx";
import { useTheme } from "@/context/ThemeContext";
import {
  getWandbSettings,
  saveWandbSettings,
  testWandbConnection,
  updateUserProfile,
} from "@/lib/api";

const ROLES_INFO = [
  {
    id: "ENGINEER",
    name: "👨‍💻 ML Engineer",
    color: "#38BDF8",
    descKey: "settings.role.engineer.desc",
    permissionKeys: [
      "settings.role.engineer.perm1",
      "settings.role.engineer.perm2",
      "settings.role.engineer.perm3",
      "settings.role.engineer.perm4",
    ],
  },
  {
    id: "ADMIN",
    name: "👑 System Administrator",
    color: "#A855F7",
    descKey: "settings.role.admin.desc",
    permissionKeys: [
      "settings.role.admin.perm1",
      "settings.role.admin.perm2",
      "settings.role.admin.perm3",
      "settings.role.admin.perm4",
    ],
  },
];

export default function SettingsPage() {
  const { user, role, isAuthenticated } = useAuth();
  const { lang, setLang, t } = useLanguage();
  const { theme, setTheme } = useTheme();
  const [activeTab, setActiveTab] = useState("appearance");

  // Profile Form State
  const [displayName, setDisplayName] = useState("");
  const [profileSuccessMsg, setProfileSuccessMsg] = useState("");
  const [profileErrorMsg, setProfileErrorMsg] = useState("");
  const [isSavingProfile, setIsSavingProfile] = useState(false);

  // Appearance & Language State
  const [minimalMode, setMinimalMode] = useState(true);
  const [prefSuccessMsg, setPrefSuccessMsg] = useState("");

  // W&B Integration State
  const [wandbKey, setWandbKey] = useState("");
  const [wandbEntity, setWandbEntity] = useState("");
  const [wandbProject, setWandbProject] = useState("advertest-perception-robustness");
  const [wandbAutoSync, setWandbAutoSync] = useState(true);
  const [wandbConnected, setWandbConnected] = useState(false);
  const [wandbMaskedKey, setWandbMaskedKey] = useState("");
  const [showWandbKey, setShowWandbKey] = useState(false);
  const [wandbStatusMsg, setWandbStatusMsg] = useState("");
  const [isTestingWandb, setIsTestingWandb] = useState(false);
  const [isSavingWandb, setIsSavingWandb] = useState(false);

  // Load user data, theme, and W&B configuration
  useEffect(() => {
    if (user) {
      setDisplayName(user.display_name || "");
    }

    try {
      const savedMin = localStorage.getItem("advertest_minimal_mode");
      if (savedMin !== null) setMinimalMode(savedMin === "true");
    } catch {}

    getWandbSettings()
      .then((res) => {
        if (res) {
          setWandbMaskedKey(res.api_key_masked || "");
          setWandbEntity(res.entity || "");
          setWandbProject(res.project || "advertest-perception-robustness");
          setWandbAutoSync(res.auto_sync ?? true);
          setWandbConnected(res.connected ?? false);
        }
      })
      .catch(console.warn);
  }, [user]);

  // Handle Theme Change
  const handleThemeChange = (newTheme) => {
    setTheme(newTheme);
    try {
      setPrefSuccessMsg(t("settings.theme.saved"));
      setTimeout(() => setPrefSuccessMsg(""), 3000);
    } catch {}
  };

  // Handle Language Change — broadcast to whole UI via context
  const handleLanguageChange = (newLang) => {
    setLang(newLang);
    try {
      setPrefSuccessMsg(t("settings.language.saved"));
      setTimeout(() => setPrefSuccessMsg(""), 3000);
    } catch {}
  };

  // Handle Minimal Mode Toggle
  const handleToggleMinimal = (e) => {
    const val = e.target.checked;
    setMinimalMode(val);
    try {
      localStorage.setItem("advertest_minimal_mode", String(val));
      setPrefSuccessMsg(t("settings.minimal.saved"));
      setTimeout(() => setPrefSuccessMsg(""), 3000);
    } catch {}
  };

  // Handle Profile Update
  const handleSaveProfile = async (e) => {
    e.preventDefault();
    setProfileSuccessMsg("");
    setProfileErrorMsg("");
    setIsSavingProfile(true);

    try {
      await updateUserProfile({
        display_name: displayName,
      });
      setProfileSuccessMsg(t("settings.profile.updated"));
    } catch (err) {
      setProfileErrorMsg(err.message || t("settings.profile.saveError"));
    } finally {
      setIsSavingProfile(false);
    }
  };

  // Handle W&B Save
  const handleSaveWandb = async (e) => {
    e.preventDefault();
    setWandbStatusMsg("");
    setIsSavingWandb(true);

    try {
      const res = await saveWandbSettings({
        api_key: wandbKey || wandbMaskedKey,
        entity: wandbEntity,
        project: wandbProject,
        auto_sync: wandbAutoSync,
      });
      setWandbMaskedKey(res.api_key_masked);
      setWandbConnected(res.connected);
      setWandbKey("");
      setWandbStatusMsg(t("settings.wandb.saved"));
    } catch (err) {
      setWandbStatusMsg(t("settings.wandb.saveError", { message: err.message }));
    } finally {
      setIsSavingWandb(false);
    }
  };

  // Handle W&B Test Connection
  const handleTestWandb = async () => {
    setWandbStatusMsg("");
    setIsTestingWandb(true);

    try {
      const res = await testWandbConnection({
        api_key: wandbKey || wandbMaskedKey,
        entity: wandbEntity,
        project: wandbProject,
        auto_sync: wandbAutoSync,
      });
      setWandbStatusMsg(`✅ ${res.message || t("settings.wandb.testOk")}`);
      setWandbConnected(true);
    } catch (err) {
      setWandbStatusMsg(t("settings.wandb.testError", { message: err.message }));
    } finally {
      setIsTestingWandb(false);
    }
  };

  const currentRole = role || user?.role || "ENGINEER";

  const tabStyle = (tab, withDot) => ({
    padding: "8px 16px",
    borderRadius: "6px",
    border: "none",
    background: activeTab === tab ? "var(--bg-elevated)" : "transparent",
    color: activeTab === tab ? "var(--accent, #38BDF8)" : "var(--text-secondary)",
    fontWeight: 700,
    fontSize: "0.85rem",
    cursor: "pointer",
    display: withDot ? "flex" : undefined,
    alignItems: withDot ? "center" : undefined,
    gap: withDot ? "6px" : undefined,
  });

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
        background: "var(--bg-primary, #0B0F17)",
        color: "var(--text-primary, #E2E8F0)",
      }}
    >
      {/* Header */}
      <header
        style={{
          height: "56px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "0 24px",
          background: "var(--bg-elevated, #131B2A)",
          borderBottom: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.12))",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <Link
            href="/"
            style={{
              color: "var(--accent, #38BDF8)",
              textDecoration: "none",
              fontSize: "0.85rem",
              fontWeight: 700,
              padding: "6px 12px",
              borderRadius: "6px",
              background: "rgba(56, 189, 248, 0.1)",
              border: "1px solid rgba(56, 189, 248, 0.25)",
            }}
          >
            {t("settings.backDashboard")}
          </Link>
          <span style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-primary)" }}>
            {t("settings.headerTitle")}
          </span>
        </div>

        <UserMenu />
      </header>

      {/* Main Content Area */}
      <div style={{ maxWidth: "1050px", width: "100%", margin: "0 auto", padding: "28px 20px" }}>
        {/* Navigation Tabs */}
        <nav
          style={{
            display: "flex",
            gap: "8px",
            borderBottom: "1px solid var(--border-subtle)",
            paddingBottom: "12px",
            marginBottom: "24px",
            flexWrap: "wrap",
          }}
        >
          <button type="button" onClick={() => setActiveTab("appearance")} style={tabStyle("appearance")}>
            {t("settings.tab.appearance")}
          </button>
          <button type="button" onClick={() => setActiveTab("profile")} style={tabStyle("profile")}>
            {t("settings.tab.profile")}
          </button>
          <button type="button" onClick={() => setActiveTab("roles")} style={tabStyle("roles")}>
            {t("settings.tab.roles")}
          </button>
          <button type="button" onClick={() => setActiveTab("wandb")} style={tabStyle("wandb", true)}>
            <span>{t("settings.tab.wandb")}</span>
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: wandbConnected ? "#10B981" : "#64748B",
              }}
            />
          </button>
        </nav>

        {/* Tab: Appearance & Language */}
        {activeTab === "appearance" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {prefSuccessMsg && (
              <div style={{ padding: "10px 14px", background: "rgba(16, 185, 129, 0.15)", color: "#10B981", borderRadius: "6px", fontSize: "0.82rem", fontWeight: 600 }}>
                {prefSuccessMsg}
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "20px" }}>
              {/* Theme Settings Card */}
              <div style={{ background: "var(--bg-elevated)", padding: "22px", borderRadius: "10px", border: "1px solid var(--border-subtle)" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 800, margin: "0 0 8px 0", color: "var(--text-primary)" }}>
                  {t("settings.theme.title")}
                </h3>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", margin: "0 0 16px 0" }}>
                  {t("settings.theme.desc")}
                </p>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px" }}>
                  {[
                    { mode: "dark", icon: "🌙", labelKey: "settings.theme.dark" },
                    { mode: "light", icon: "☀️", labelKey: "settings.theme.light" },
                    { mode: "system", icon: "💻", labelKey: "settings.theme.system" },
                  ].map(({ mode, icon, labelKey }) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => handleThemeChange(mode)}
                      style={{
                        padding: "12px 10px",
                        borderRadius: "8px",
                        background: theme === mode ? "rgba(56, 189, 248, 0.15)" : "var(--bg-primary)",
                        border: theme === mode ? "2px solid var(--accent, #38BDF8)" : "1px solid var(--border-subtle)",
                        color: theme === mode ? "var(--text-primary)" : "var(--text-secondary)",
                        cursor: "pointer",
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: "6px",
                        fontWeight: 700,
                        fontSize: "0.78rem",
                      }}
                    >
                      <span style={{ fontSize: "1.3rem" }}>{icon}</span>
                      <span>{t(labelKey)}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Language Settings Card */}
              <div style={{ background: "var(--bg-elevated)", padding: "22px", borderRadius: "10px", border: "1px solid var(--border-subtle)" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 800, margin: "0 0 8px 0", color: "var(--text-primary)" }}>
                  {t("settings.language.title")}
                </h3>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", margin: "0 0 16px 0" }}>
                  {t("settings.language.desc")}
                </p>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                  <button
                    type="button"
                    onClick={() => handleLanguageChange("vi")}
                    style={{
                      padding: "12px 14px",
                      borderRadius: "8px",
                      background: lang === "vi" ? "rgba(56, 189, 248, 0.15)" : "var(--bg-primary)",
                      border: lang === "vi" ? "2px solid var(--accent, #38BDF8)" : "1px solid var(--border-subtle)",
                      color: lang === "vi" ? "var(--text-primary)" : "var(--text-secondary)",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "10px",
                      fontWeight: 700,
                      fontSize: "0.82rem",
                    }}
                  >
                    <span style={{ fontSize: "1.3rem" }}>🇻🇳</span>
                    <span>{t("settings.language.vi")}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleLanguageChange("en")}
                    style={{
                      padding: "12px 14px",
                      borderRadius: "8px",
                      background: lang === "en" ? "rgba(56, 189, 248, 0.15)" : "var(--bg-primary)",
                      border: lang === "en" ? "2px solid var(--accent, #38BDF8)" : "1px solid var(--border-subtle)",
                      color: lang === "en" ? "var(--text-primary)" : "var(--text-secondary)",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "10px",
                      fontWeight: 700,
                      fontSize: "0.82rem",
                    }}
                  >
                    <span style={{ fontSize: "1.3rem" }}>🇺🇸</span>
                    <span>{t("settings.language.en")}</span>
                  </button>
                </div>

                {/* Clean Workspace Option */}
                <div style={{ marginTop: "18px", paddingTop: "14px", borderTop: "1px solid var(--border-subtle)" }}>
                  <label style={{ display: "flex", alignItems: "center", gap: "10px", cursor: "pointer", fontSize: "0.78rem", fontWeight: 600, color: "var(--text-primary)" }}>
                    <input
                      type="checkbox"
                      checked={minimalMode}
                      onChange={handleToggleMinimal}
                      style={{ width: "16px", height: "16px", cursor: "pointer" }}
                    />
                    <span>{t("settings.minimal.label")}</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab: Profile & Quotas */}
        {activeTab === "profile" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {profileSuccessMsg && (
              <div style={{ padding: "10px 14px", background: "rgba(16, 185, 129, 0.15)", color: "#10B981", borderRadius: "6px", fontSize: "0.82rem", fontWeight: 600 }}>
                {profileSuccessMsg}
              </div>
            )}
            {profileErrorMsg && (
              <div style={{ padding: "10px 14px", background: "rgba(239, 68, 68, 0.15)", color: "#EF4444", borderRadius: "6px", fontSize: "0.82rem", fontWeight: 600 }}>
                {profileErrorMsg}
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "20px" }}>
              {/* Profile Card */}
              <div style={{ background: "var(--bg-elevated)", padding: "20px", borderRadius: "10px", border: "1px solid var(--border-subtle)" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 800, margin: "0 0 16px 0", color: "var(--text-primary)" }}>
                  {t("settings.profile.title")}
                </h3>

                <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "20px" }}>
                  <div
                    style={{
                      width: "64px",
                      height: "64px",
                      borderRadius: "50%",
                      background: "#EA4335",
                      color: "#FFF",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontWeight: 800,
                      fontSize: "1.6rem",
                      overflow: "hidden",
                    }}
                  >
                    {user?.avatar_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={user.avatar_url} alt={user.display_name} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    ) : (
                      user?.display_name?.charAt(0) || "U"
                    )}
                  </div>
                  <div>
                    <div style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-primary)" }}>
                      {user?.display_name || t("settings.profile.unnamed")}
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
                      {user?.email || "developer@advertest.ai"}
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "4px" }}>
                      <span style={{ fontSize: "0.68rem", fontWeight: 700, padding: "2px 8px", borderRadius: "4px", background: "#4285F425", color: "#4285F4" }}>
                        {t("settings.profile.googleAuth")}
                      </span>
                    </div>
                  </div>
                </div>

                <form onSubmit={handleSaveProfile} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                  <div>
                    <label className="config-panel__label" style={{ fontSize: "0.75rem" }}>{t("settings.profile.displayName")}</label>
                    <input
                      type="text"
                      className="select-field"
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                      style={{ width: "100%", padding: "8px 12px", fontSize: "0.85rem" }}
                    />
                  </div>

                  <div>
                    <label className="config-panel__label" style={{ fontSize: "0.75rem" }}>{t("settings.profile.email")}</label>
                    <input
                      type="text"
                      disabled
                      className="select-field"
                      value={user?.email || "alex.engineer@gmail.com"}
                      style={{ width: "100%", padding: "8px 12px", fontSize: "0.85rem", opacity: 0.6 }}
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={isSavingProfile}
                    className="action-button action-button--primary"
                    style={{ padding: "8px 16px", fontSize: "0.82rem", fontWeight: 700, alignSelf: "flex-start", marginTop: "6px" }}
                  >
                    {isSavingProfile ? t("settings.profile.saving") : t("settings.profile.save")}
                  </button>
                </form>
              </div>

              {/* Resource Quotas Card */}
              <div style={{ background: "var(--bg-elevated)", padding: "20px", borderRadius: "10px", border: "1px solid var(--border-subtle)" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 800, margin: "0 0 16px 0", color: "var(--text-primary)" }}>
                  {t("settings.quota.title")}
                </h3>

                <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
                  {/* Storage */}
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", fontWeight: 700, marginBottom: "6px" }}>
                      <span>{t("settings.quota.storage")}</span>
                      <span style={{ color: "var(--accent)" }}>245 MB / 10,240 MB (2.4%)</span>
                    </div>
                    <div style={{ width: "100%", height: "8px", background: "var(--bg-primary)", borderRadius: "4px", overflow: "hidden" }}>
                      <div style={{ width: "2.4%", height: "100%", background: "#38BDF8", borderRadius: "4px" }} />
                    </div>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: "4px" }}>
                      {t("settings.quota.storage.desc")}
                    </div>
                  </div>

                  {/* GPU Compute */}
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", fontWeight: 700, marginBottom: "6px" }}>
                      <span>{t("settings.quota.gpu")}</span>
                      <span style={{ color: "#10B981" }}>12.4 hrs / 100.0 hrs (12.4%)</span>
                    </div>
                    <div style={{ width: "100%", height: "8px", background: "var(--bg-primary)", borderRadius: "4px", overflow: "hidden" }}>
                      <div style={{ width: "12.4%", height: "100%", background: "#10B981", borderRadius: "4px" }} />
                    </div>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: "4px" }}>
                      {t("settings.quota.gpu.desc")}
                    </div>
                  </div>

                  {/* Status Badge */}
                  <div style={{ padding: "12px", borderRadius: "6px", background: "rgba(16, 185, 129, 0.1)", border: "1px solid rgba(16, 185, 129, 0.25)", marginTop: "8px" }}>
                    <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#10B981" }}>
                      {t("settings.quota.statusActive")}
                    </div>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                      {t("settings.quota.statusDesc")}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab: RBAC Roles & Permissions */}
        {activeTab === "roles" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            <div style={{ background: "var(--bg-elevated)", padding: "16px 20px", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
              <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-secondary)" }}>
                {t("settings.roles.current")}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "8px" }}>
                <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--accent)" }}>
                  {ROLES_INFO.find((r) => r.id === currentRole)?.name || currentRole}
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  {t("settings.roles.managedBy")}
                </span>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "16px" }}>
              {ROLES_INFO.map((r) => {
                const isActive = currentRole === r.id;
                return (
                  <div
                    key={r.id}
                    style={{
                      background: isActive ? "rgba(56, 189, 248, 0.08)" : "var(--bg-elevated)",
                      border: isActive ? `2px solid ${r.color}` : "1px solid var(--border-subtle)",
                      borderRadius: "10px",
                      padding: "20px",
                      display: "flex",
                      flexDirection: "column",
                      gap: "12px",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span style={{ fontSize: "1.05rem", fontWeight: 800, color: r.color }}>
                        {r.name}
                      </span>
                      {isActive && (
                        <span style={{ fontSize: "0.7rem", fontWeight: 800, padding: "3px 8px", borderRadius: "4px", background: r.color, color: "#000" }}>
                          {t("settings.roles.yourRole")}
                        </span>
                      )}
                    </div>

                    <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", margin: 0 }}>
                      {t(r.descKey)}
                    </p>

                    <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "10px" }}>
                      <div style={{ fontSize: "0.68rem", fontWeight: 700, color: "var(--text-muted)", marginBottom: "6px" }}>
                        {t("settings.roles.permissions")}
                      </div>
                      <ul style={{ margin: 0, paddingLeft: "16px", fontSize: "0.72rem", color: "var(--text-secondary)", display: "flex", flexDirection: "column", gap: "4px" }}>
                        {r.permissionKeys.map((k, idx) => (
                          <li key={idx}>{t(k)}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Tab: Weights & Biases Integration */}
        {activeTab === "wandb" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {wandbStatusMsg && (
              <div
                style={{
                  padding: "12px 16px",
                  background: wandbStatusMsg.startsWith("✅") ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)",
                  color: wandbStatusMsg.startsWith("✅") ? "#10B981" : "#EF4444",
                  borderRadius: "6px",
                  fontSize: "0.82rem",
                  fontWeight: 600,
                }}
              >
                {wandbStatusMsg}
              </div>
            )}

            <div style={{ background: "var(--bg-elevated)", padding: "24px", borderRadius: "10px", border: "1px solid var(--border-subtle)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "18px" }}>
                <div>
                  <h3 style={{ fontSize: "1.1rem", fontWeight: 800, margin: 0, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: "8px" }}>
                    <span>{t("settings.wandb.title")}</span>
                    <span
                      style={{
                        fontSize: "0.68rem",
                        fontWeight: 700,
                        padding: "2px 8px",
                        borderRadius: "4px",
                        background: wandbConnected ? "rgba(16, 185, 129, 0.15)" : "rgba(100, 116, 139, 0.2)",
                        color: wandbConnected ? "#10B981" : "#94A3B8",
                      }}
                    >
                      {wandbConnected ? t("settings.wandb.connected") : t("settings.wandb.disconnected")}
                    </span>
                  </h3>
                  <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", margin: "4px 0 0 0" }}>
                    {t("settings.wandb.desc")}
                  </p>
                </div>

                <a
                  href="https://wandb.ai/authorize"
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: 700,
                    color: "#F59E0B",
                    textDecoration: "none",
                    padding: "6px 12px",
                    borderRadius: "6px",
                    background: "rgba(245, 158, 11, 0.12)",
                    border: "1px solid rgba(245, 158, 11, 0.3)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {t("settings.wandb.getApiKey")}
                </a>
              </div>

              <form onSubmit={handleSaveWandb} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                {/* API Key */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                    <label className="config-panel__label" style={{ fontSize: "0.75rem" }}>
                      {t("settings.wandb.apiKeyLabel")}
                    </label>
                    <button
                      type="button"
                      onClick={() => setShowWandbKey(!showWandbKey)}
                      style={{ background: "none", border: "none", color: "var(--accent)", fontSize: "0.7rem", cursor: "pointer", fontWeight: 600 }}
                    >
                      {showWandbKey ? t("settings.wandb.toggle.hide") : t("settings.wandb.toggle.show")}
                    </button>
                  </div>
                  <input
                    type={showWandbKey ? "text" : "password"}
                    className="select-field"
                    placeholder={wandbMaskedKey ? `${t("settings.wandb.currentKey")} ${wandbMaskedKey}` : t("settings.wandb.apiKeyPlaceholder")}
                    value={wandbKey}
                    onChange={(e) => setWandbKey(e.target.value)}
                    style={{ width: "100%", padding: "9px 12px", fontSize: "0.85rem", fontFamily: "var(--font-mono)" }}
                  />
                  <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: "4px" }}>
                    {t("settings.wandb.apiKeySecurity")}
                  </div>
                </div>

                {/* Team / Entity & Project */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                  <div>
                    <label className="config-panel__label" style={{ fontSize: "0.75rem" }}>
                      {t("settings.wandb.entityLabel")}
                    </label>
                    <input
                      type="text"
                      className="select-field"
                      placeholder={t("settings.wandb.entityPlaceholder")}
                      value={wandbEntity}
                      onChange={(e) => setWandbEntity(e.target.value)}
                      style={{ width: "100%", padding: "9px 12px", fontSize: "0.85rem" }}
                    />
                  </div>
                  <div>
                    <label className="config-panel__label" style={{ fontSize: "0.75rem" }}>
                      {t("settings.wandb.projectLabel")}
                    </label>
                    <input
                      type="text"
                      className="select-field"
                      placeholder="advertest-perception-robustness"
                      value={wandbProject}
                      onChange={(e) => setWandbProject(e.target.value)}
                      style={{ width: "100%", padding: "9px 12px", fontSize: "0.85rem" }}
                    />
                  </div>
                </div>

                {/* Auto Sync Toggle */}
                <div style={{ display: "flex", alignItems: "center", gap: "10px", padding: "12px", borderRadius: "6px", background: "var(--bg-primary)" }}>
                  <input
                    type="checkbox"
                    id="auto-sync"
                    checked={wandbAutoSync}
                    onChange={(e) => setWandbAutoSync(e.target.checked)}
                    style={{ width: "16px", height: "16px", cursor: "pointer" }}
                  />
                  <label htmlFor="auto-sync" style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-primary)", cursor: "pointer" }}>
                    {t("settings.wandb.autoSync")}
                  </label>
                </div>

                {/* Action Buttons */}
                <div style={{ display: "flex", gap: "12px", marginTop: "8px" }}>
                  <button
                    type="button"
                    onClick={handleTestWandb}
                    disabled={isTestingWandb}
                    className="action-button action-button--secondary"
                    style={{ padding: "9px 18px", fontSize: "0.82rem", fontWeight: 700 }}
                  >
                    {isTestingWandb ? t("settings.wandb.testing") : t("settings.wandb.test")}
                  </button>

                  <button
                    type="submit"
                    disabled={isSavingWandb}
                    className="action-button action-button--primary"
                    style={{ padding: "9px 22px", fontSize: "0.82rem", fontWeight: 700 }}
                  >
                    {isSavingWandb ? t("settings.wandb.saving") : t("settings.wandb.save")}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
