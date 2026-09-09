"use client";

import { useEffect, useState } from "react";
import Button from "@/components/common/Button";
import Card from "@/components/common/Card";
import PageHeader from "@/components/layout/PageHeader";
import { useAuth } from "@/context/AuthContext";
import { useLanguage } from "@/context/LanguageContext";
import { useTheme } from "@/context/ThemeContext";
import { getWandbSettings, saveWandbSettings, testWandbConnection, updateUserProfile } from "@/lib/api";
import { cn } from "@/lib/utils";

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

const THEME_MODES = [
  { mode: "dark", icon: "🌙", labelKey: "settings.theme.dark" },
  { mode: "light", icon: "☀️", labelKey: "settings.theme.light" },
  { mode: "system", icon: "💻", labelKey: "settings.theme.system" },
];

const inputClass =
  "w-full rounded-md border border-[var(--app-border)] bg-[var(--bg-secondary)] px-3 py-2 text-[13px] text-[var(--text-primary)] outline-none focus:border-blue-500 placeholder:text-[var(--text-muted)]";
const labelClass = "block text-[11px] font-semibold text-[var(--text-secondary)] mb-1";

function StatusBanner({ tone, children }) {
  return (
    <div
      className={cn(
        "rounded-md px-4 py-2.5 text-[12px] font-semibold",
        tone === "error" ? "bg-[rgba(239,68,68,0.15)] text-[#EF4444]" : "bg-[rgba(16,185,129,0.15)] text-[#10B981]",
      )}
    >
      {children}
    </div>
  );
}

export default function SettingsPage() {
  const { user, role } = useAuth();
  const { lang, setLang, t } = useLanguage();
  const { theme, setTheme } = useTheme();
  const [activeTab, setActiveTab] = useState("appearance");

  // Profile Form State
  const [displayName, setDisplayName] = useState("");
  const [profileSuccessMsg, setProfileSuccessMsg] = useState("");
  const [profileErrorMsg, setProfileErrorMsg] = useState("");
  const [isSavingProfile, setIsSavingProfile] = useState(false);

  // Appearance & Language State
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

  // Load user data and W&B configuration
  useEffect(() => {
    if (user) {
      setDisplayName(user.display_name || "");
    }

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

  const TABS = [
    { id: "appearance", labelKey: "settings.tab.appearance" },
    { id: "profile", labelKey: "settings.tab.profile" },
    { id: "roles", labelKey: "settings.tab.roles" },
    { id: "wandb", labelKey: "settings.tab.wandb", withDot: true },
  ];

  return (
    <div className="space-y-5">
      <PageHeader title={t("settings.headerTitle")} />

      {/* Navigation Tabs */}
      <nav className="flex flex-wrap gap-2 border-b border-[var(--app-border)] pb-3">
        {TABS.map(({ id, labelKey, withDot }) => (
          <button
            key={id}
            type="button"
            onClick={() => setActiveTab(id)}
            className={cn(
              "flex items-center gap-2 rounded-lg border px-4 py-2 text-[13px] font-bold transition-colors cursor-pointer",
              activeTab === id
                ? "border-[var(--app-border)] bg-[var(--app-surface)] text-blue-600 shadow-sm"
                : "border-transparent text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] hover:text-blue-600",
            )}
          >
            <span>{t(labelKey)}</span>
            {withDot && (
              <span
                className={cn("h-[7px] w-[7px] rounded-full", wandbConnected ? "bg-emerald-500" : "bg-slate-400")}
              />
            )}
          </button>
        ))}
      </nav>

      {/* Tab: Appearance & Language */}
      {activeTab === "appearance" && (
        <div className="space-y-5">
          {prefSuccessMsg && <StatusBanner>{prefSuccessMsg}</StatusBanner>}

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
            {/* Theme Settings Card */}
            <Card title={t("settings.theme.title")} subtitle={t("settings.theme.desc")}>
              <div className="grid grid-cols-3 gap-2.5">
                {THEME_MODES.map(({ mode, icon, labelKey }) => {
                  const selected = theme === mode;
                  return (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => handleThemeChange(mode)}
                      className={cn(
                        "flex cursor-pointer flex-col items-center gap-1.5 rounded-lg border py-3 text-[12px] font-bold transition-colors",
                        selected
                          ? "border-blue-600 bg-[var(--primary-light)] text-[var(--text-primary)]"
                          : "border-[var(--app-border)] bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:border-blue-500",
                      )}
                    >
                      <span className="text-[1.3rem]">{icon}</span>
                      <span>{t(labelKey)}</span>
                    </button>
                  );
                })}
              </div>
            </Card>

            {/* Language Settings Card */}
            <Card title={t("settings.language.title")} subtitle={t("settings.language.desc")}>
              <div className="grid grid-cols-2 gap-2.5">
                {[
                  { code: "vi", flag: "🇻🇳", labelKey: "settings.language.vi" },
                  { code: "en", flag: "🇺🇸", labelKey: "settings.language.en" },
                ].map(({ code, flag, labelKey }) => {
                  const selected = lang === code;
                  return (
                    <button
                      key={code}
                      type="button"
                      onClick={() => handleLanguageChange(code)}
                      className={cn(
                        "flex cursor-pointer items-center gap-2.5 rounded-lg border px-3.5 py-3 text-[13px] font-bold transition-colors",
                        selected
                          ? "border-blue-600 bg-[var(--primary-light)] text-[var(--text-primary)]"
                          : "border-[var(--app-border)] bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:border-blue-500",
                      )}
                    >
                      <span className="text-[1.3rem]">{flag}</span>
                      <span>{t(labelKey)}</span>
                    </button>
                  );
                })}
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* Tab: Profile & Quotas */}
      {activeTab === "profile" && (
        <div className="space-y-5">
          {profileSuccessMsg && <StatusBanner>{profileSuccessMsg}</StatusBanner>}
          {profileErrorMsg && <StatusBanner tone="error">{profileErrorMsg}</StatusBanner>}

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
            {/* Profile Card */}
            <Card title={t("settings.profile.title")}>
              <div className="mb-5">
                <div className="text-[15px] font-bold text-[var(--text-primary)]">
                  {user?.display_name || t("settings.profile.unnamed")}
                </div>
                <div className="text-[12px] text-[var(--text-muted)]">{user?.email || "—"}</div>
                <span className="mt-1.5 inline-block rounded bg-[rgba(66,133,244,0.15)] px-2 py-0.5 text-[11px] font-bold text-[#4285F4]">
                  {t("settings.profile.googleAuth")}
                </span>
              </div>

              <form onSubmit={handleSaveProfile} className="flex flex-col gap-3.5">
                <div>
                  <label htmlFor="settings-display-name" className={labelClass}>
                    {t("settings.profile.displayName")}
                  </label>
                  <input
                    id="settings-display-name"
                    type="text"
                    className={inputClass}
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                  />
                </div>

                <div>
                  <label htmlFor="settings-email" className={labelClass}>
                    {t("settings.profile.email")}
                  </label>
                  <input
                    id="settings-email"
                    type="text"
                    disabled
                    className={cn(inputClass, "opacity-60")}
                    value={user?.email || "—"}
                  />
                </div>

                <Button type="submit" disabled={isSavingProfile} className="self-start">
                  {isSavingProfile ? t("settings.profile.saving") : t("settings.profile.save")}
                </Button>
              </form>
            </Card>

            {/* Resource Quotas Card */}
            <Card title={t("settings.quota.title")}>
              <div className="flex flex-col gap-4">
                {/* Storage */}
                <div>
                  <div className="mb-1.5 flex justify-between text-[12px] font-bold">
                    <span className="text-[var(--text-primary)]">{t("settings.quota.storage")}</span>
                    <span className="text-[#38BDF8]">245 MB / 10,240 MB (2.4%)</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded bg-[var(--bg-tertiary)]">
                    <div className="h-full w-[2.4%] rounded bg-[#38BDF8]" />
                  </div>
                  <div className="mt-1 text-[11px] text-[var(--text-muted)]">{t("settings.quota.storage.desc")}</div>
                </div>

                {/* GPU Compute */}
                <div>
                  <div className="mb-1.5 flex justify-between text-[12px] font-bold">
                    <span className="text-[var(--text-primary)]">{t("settings.quota.gpu")}</span>
                    <span className="text-[#10B981]">12.4 hrs / 100.0 hrs (12.4%)</span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded bg-[var(--bg-tertiary)]">
                    <div className="h-full w-[12.4%] rounded bg-[#10B981]" />
                  </div>
                  <div className="mt-1 text-[11px] text-[var(--text-muted)]">{t("settings.quota.gpu.desc")}</div>
                </div>

                {/* Status Badge */}
                <div className="mt-1 rounded-md border border-[rgba(16,185,129,0.25)] bg-[rgba(16,185,129,0.1)] p-3">
                  <div className="text-[12px] font-bold text-[#10B981]">{t("settings.quota.statusActive")}</div>
                  <div className="mt-0.5 text-[11px] text-[var(--text-secondary)]">
                    {t("settings.quota.statusDesc")}
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* Tab: RBAC Roles & Permissions */}
      {activeTab === "roles" && (
        <div className="space-y-5">
          <Card>
            <div className="text-[13px] font-bold text-[var(--text-secondary)]">{t("settings.roles.current")}</div>
            <div className="mt-2 flex items-center gap-3">
              <span className="text-[19px] font-extrabold text-[var(--text-primary)]">
                {ROLES_INFO.find((r) => r.id === currentRole)?.name || currentRole}
              </span>
              <span className="text-[12px] text-[var(--text-muted)]">{t("settings.roles.managedBy")}</span>
            </div>
          </Card>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {ROLES_INFO.map((r) => {
              const isActive = currentRole === r.id;
              return (
                <div
                  key={r.id}
                  style={isActive ? { borderColor: r.color } : undefined}
                  className={cn(
                    "flex flex-col gap-3 rounded-[10px] border p-5",
                    isActive
                      ? "bg-[var(--primary-light)] border-2"
                      : "bg-[var(--app-surface)] border-[var(--app-border)]",
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[17px] font-extrabold" style={{ color: r.color }}>
                      {r.name}
                    </span>
                    {isActive && (
                      <span
                        className="rounded px-2 py-0.5 text-[11px] font-extrabold text-black"
                        style={{ background: r.color }}
                      >
                        {t("settings.roles.yourRole")}
                      </span>
                    )}
                  </div>

                  <p className="m-0 text-[12px] text-[var(--text-secondary)]">{t(r.descKey)}</p>

                  <div className="border-t border-[var(--app-border)] pt-2.5">
                    <div className="mb-1.5 text-[11px] font-bold text-[var(--text-muted)]">
                      {t("settings.roles.permissions")}
                    </div>
                    <ul className="m-0 flex list-disc flex-col gap-1 pl-4 text-[11px] text-[var(--text-secondary)]">
                      {r.permissionKeys.map((k) => (
                        <li key={k}>{t(k)}</li>
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
        <div className="space-y-5">
          {wandbStatusMsg && (
            <StatusBanner tone={wandbStatusMsg.startsWith("✅") ? "success" : "error"}>{wandbStatusMsg}</StatusBanner>
          )}

          <Card
            title={t("settings.wandb.title")}
            subtitle={t("settings.wandb.desc")}
            headerAction={
              <a
                href="https://wandb.ai/authorize"
                target="_blank"
                rel="noreferrer"
                className="rounded-md border border-[rgba(245,158,11,0.3)] bg-[rgba(245,158,11,0.12)] px-3 py-1.5 text-[12px] font-bold text-[#F59E0B] no-underline whitespace-nowrap"
              >
                {t("settings.wandb.getApiKey")}
              </a>
            }
          >
            <div className="mb-4">
              <span
                className={cn(
                  "inline-block rounded px-2 py-0.5 text-[11px] font-bold",
                  wandbConnected
                    ? "bg-[rgba(16,185,129,0.15)] text-[#10B981]"
                    : "bg-[rgba(100,116,139,0.2)] text-[#94A3B8]",
                )}
              >
                {wandbConnected ? t("settings.wandb.connected") : t("settings.wandb.disconnected")}
              </span>
            </div>

            <form onSubmit={handleSaveWandb} className="flex flex-col gap-4">
              {/* API Key */}
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <label htmlFor="settings-wandb-key" className={labelClass}>
                    {t("settings.wandb.apiKeyLabel")}
                  </label>
                  <button
                    type="button"
                    onClick={() => setShowWandbKey(!showWandbKey)}
                    className="cursor-pointer border-0 bg-transparent p-0 text-[11px] font-semibold text-blue-600"
                  >
                    {showWandbKey ? t("settings.wandb.toggle.hide") : t("settings.wandb.toggle.show")}
                  </button>
                </div>
                <input
                  id="settings-wandb-key"
                  type={showWandbKey ? "text" : "password"}
                  className={cn(inputClass, "font-mono")}
                  placeholder={
                    wandbMaskedKey
                      ? `${t("settings.wandb.currentKey")} ${wandbMaskedKey}`
                      : t("settings.wandb.apiKeyPlaceholder")
                  }
                  value={wandbKey}
                  onChange={(e) => setWandbKey(e.target.value)}
                />
                <div className="mt-1 text-[11px] text-[var(--text-muted)]">{t("settings.wandb.apiKeySecurity")}</div>
              </div>

              {/* Team / Entity & Project */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="settings-wandb-entity" className={labelClass}>
                    {t("settings.wandb.entityLabel")}
                  </label>
                  <input
                    id="settings-wandb-entity"
                    type="text"
                    className={inputClass}
                    placeholder={t("settings.wandb.entityPlaceholder")}
                    value={wandbEntity}
                    onChange={(e) => setWandbEntity(e.target.value)}
                  />
                </div>
                <div>
                  <label htmlFor="settings-wandb-project" className={labelClass}>
                    {t("settings.wandb.projectLabel")}
                  </label>
                  <input
                    id="settings-wandb-project"
                    type="text"
                    className={inputClass}
                    placeholder="advertest-perception-robustness"
                    value={wandbProject}
                    onChange={(e) => setWandbProject(e.target.value)}
                  />
                </div>
              </div>

              {/* Auto Sync Toggle */}
              <div className="flex items-center gap-2.5 rounded-md border border-[var(--app-border)] bg-[var(--bg-secondary)] p-3">
                <input
                  type="checkbox"
                  id="settings-wandb-autosync"
                  checked={wandbAutoSync}
                  onChange={(e) => setWandbAutoSync(e.target.checked)}
                  className="h-4 w-4 cursor-pointer"
                />
                <label
                  htmlFor="settings-wandb-autosync"
                  className="cursor-pointer text-[13px] font-semibold text-[var(--text-primary)]"
                >
                  {t("settings.wandb.autoSync")}
                </label>
              </div>

              {/* Action Buttons */}
              <div className="mt-1 flex gap-3">
                <Button variant="secondary" onClick={handleTestWandb} disabled={isTestingWandb}>
                  {isTestingWandb ? t("settings.wandb.testing") : t("settings.wandb.test")}
                </Button>
                <Button type="submit" disabled={isSavingWandb}>
                  {isSavingWandb ? t("settings.wandb.saving") : t("settings.wandb.save")}
                </Button>
              </div>
            </form>
          </Card>
        </div>
      )}
    </div>
  );
}
