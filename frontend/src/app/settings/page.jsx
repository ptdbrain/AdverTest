"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import UserMenu from "@/components/UserMenu.jsx";
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
    description: "Kỹ sư AI xây dựng mô hình, cấu hình tấn công đối kháng, thẩm định lỗi và tối ưu hóa robustness.",
    permissions: [
      "Khởi chạy và dừng các bài kiểm thử độ bền vững (FGSM, PGD, Fog, Rain, Occlusion...)",
      "Cấu hình các bộ nhận dạng Perception (YOLO, SAM2, MMDetection3D)",
      "Import dữ liệu, thẩm định các ca lỗi khó trong Review Queue (HITL)",
      "Đẩy các ca lỗi khó vào Retraining Backlog để tái huấn luyện mô hình",
    ],
  },
  {
    id: "ADMIN",
    name: "👑 System Administrator",
    color: "#A855F7",
    description: "Quản trị viên toàn quyền quản trị người dùng, hạn ngạch GPU/Storage và tích hợp MLOps.",
    permissions: [
      "Quản lý danh sách tài khoản người dùng, kích hoạt hoặc tạm ngưng tài khoản",
      "Phân bổ hạn ngạch lưu trữ (Storage Quota) và giờ tính toán GPU (Compute Hours)",
      "Cấu hình các khóa API bảo mật và tích hợp Weights & Biases (W&B)",
      "Truy cập toàn bộ nhật ký sự kiện hệ thống (System Audit Trail)",
    ],
  },
];

export default function SettingsPage() {
  const { user, role, isAuthenticated } = useAuth();
  const [activeTab, setActiveTab] = useState("appearance");

  // Profile Form State
  const [displayName, setDisplayName] = useState("");
  const [profileSuccessMsg, setProfileSuccessMsg] = useState("");
  const [profileErrorMsg, setProfileErrorMsg] = useState("");
  const [isSavingProfile, setIsSavingProfile] = useState(false);

  // Appearance & Language State
  const [currentTheme, setCurrentTheme] = useState("dark");
  const [language, setLanguage] = useState("vi");
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
      const savedTheme = localStorage.getItem("theme") || "dark";
      setCurrentTheme(savedTheme);
      const savedLang = localStorage.getItem("advertest_lang") || "vi";
      setLanguage(savedLang);
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
    setCurrentTheme(newTheme);
    try {
      const effectiveTheme =
        newTheme === "system"
          ? window.matchMedia("(prefers-color-scheme: light)").matches
            ? "light"
            : "dark"
          : newTheme;
      document.documentElement.setAttribute("data-theme", effectiveTheme);
      localStorage.setItem("theme", newTheme);
      setPrefSuccessMsg("🎨 Đã áp dụng chủ đề mới!");
      setTimeout(() => setPrefSuccessMsg(""), 3000);
    } catch {}
  };

  // Handle Language Change
  const handleLanguageChange = (newLang) => {
    setLanguage(newLang);
    try {
      localStorage.setItem("advertest_lang", newLang);
      setPrefSuccessMsg("🌐 Đã lưu tùy chọn ngôn ngữ!");
      setTimeout(() => setPrefSuccessMsg(""), 3000);
    } catch {}
  };

  // Handle Minimal Mode Toggle
  const handleToggleMinimal = (e) => {
    const val = e.target.checked;
    setMinimalMode(val);
    try {
      localStorage.setItem("advertest_minimal_mode", String(val));
      setPrefSuccessMsg("✨ Đã cập nhật chế độ hiển thị!");
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
      setProfileSuccessMsg("✅ Đã cập nhật thông tin hồ sơ thành công!");
    } catch (err) {
      setProfileErrorMsg(err.message || "Cập nhật hồ sơ thất bại.");
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
      setWandbStatusMsg("✅ Đã lưu cấu hình Weights & Biases thành công!");
    } catch (err) {
      setWandbStatusMsg(`❌ Lỗi lưu cấu hình: ${err.message}`);
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
      setWandbStatusMsg(`✅ ${res.message || "Kết nối Weights & Biases thành công!"}`);
      setWandbConnected(true);
    } catch (err) {
      setWandbStatusMsg(`❌ Kiểm tra thất bại: ${err.message}`);
    } finally {
      setIsTestingWandb(false);
    }
  };

  const currentRole = role || user?.role || "ENGINEER";

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
            ← Quay lại Dashboard
          </Link>
          <span style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-primary)" }}>
            ⚙️ Cài Đặt Hệ Thống & Tùy Chọn
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
          <button
            type="button"
            onClick={() => setActiveTab("appearance")}
            style={{
              padding: "8px 16px",
              borderRadius: "6px",
              border: "none",
              background: activeTab === "appearance" ? "var(--bg-elevated)" : "transparent",
              color: activeTab === "appearance" ? "var(--accent, #38BDF8)" : "var(--text-secondary)",
              fontWeight: 700,
              fontSize: "0.85rem",
              cursor: "pointer",
            }}
          >
            🎨 Giao Diện & Ngôn Ngữ
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("profile")}
            style={{
              padding: "8px 16px",
              borderRadius: "6px",
              border: "none",
              background: activeTab === "profile" ? "var(--bg-elevated)" : "transparent",
              color: activeTab === "profile" ? "var(--accent, #38BDF8)" : "var(--text-secondary)",
              fontWeight: 700,
              fontSize: "0.85rem",
              cursor: "pointer",
            }}
          >
            👤 Tài Khoản & Hạn Mức
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("roles")}
            style={{
              padding: "8px 16px",
              borderRadius: "6px",
              border: "none",
              background: activeTab === "roles" ? "var(--bg-elevated)" : "transparent",
              color: activeTab === "roles" ? "var(--accent, #38BDF8)" : "var(--text-secondary)",
              fontWeight: 700,
              fontSize: "0.85rem",
              cursor: "pointer",
            }}
          >
            🛡️ Phân Quyền Vai Trò
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("wandb")}
            style={{
              padding: "8px 16px",
              borderRadius: "6px",
              border: "none",
              background: activeTab === "wandb" ? "var(--bg-elevated)" : "transparent",
              color: activeTab === "wandb" ? "var(--accent, #38BDF8)" : "var(--text-secondary)",
              fontWeight: 700,
              fontSize: "0.85rem",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <span>📊 Weights & Biases (W&B)</span>
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
                  🎨 Chủ Đề Hiển Thị (Theme)
                </h3>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", margin: "0 0 16px 0" }}>
                  Chọn tông màu giao diện phù hợp với môi trường làm việc của bạn.
                </p>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px" }}>
                  {/* Dark Mode */}
                  <button
                    type="button"
                    onClick={() => handleThemeChange("dark")}
                    style={{
                      padding: "12px 10px",
                      borderRadius: "8px",
                      background: currentTheme === "dark" ? "rgba(56, 189, 248, 0.15)" : "var(--bg-primary)",
                      border: currentTheme === "dark" ? "2px solid var(--accent, #38BDF8)" : "1px solid var(--border-subtle)",
                      color: currentTheme === "dark" ? "var(--text-primary)" : "var(--text-secondary)",
                      cursor: "pointer",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: "6px",
                      fontWeight: 700,
                      fontSize: "0.78rem",
                    }}
                  >
                    <span style={{ fontSize: "1.3rem" }}>🌙</span>
                    <span>Tối (Dark)</span>
                  </button>

                  {/* Light Mode */}
                  <button
                    type="button"
                    onClick={() => handleThemeChange("light")}
                    style={{
                      padding: "12px 10px",
                      borderRadius: "8px",
                      background: currentTheme === "light" ? "rgba(56, 189, 248, 0.15)" : "var(--bg-primary)",
                      border: currentTheme === "light" ? "2px solid var(--accent, #38BDF8)" : "1px solid var(--border-subtle)",
                      color: currentTheme === "light" ? "var(--text-primary)" : "var(--text-secondary)",
                      cursor: "pointer",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: "6px",
                      fontWeight: 700,
                      fontSize: "0.78rem",
                    }}
                  >
                    <span style={{ fontSize: "1.3rem" }}>☀️</span>
                    <span>Sáng (Light)</span>
                  </button>

                  {/* System Auto */}
                  <button
                    type="button"
                    onClick={() => handleThemeChange("system")}
                    style={{
                      padding: "12px 10px",
                      borderRadius: "8px",
                      background: currentTheme === "system" ? "rgba(56, 189, 248, 0.15)" : "var(--bg-primary)",
                      border: currentTheme === "system" ? "2px solid var(--accent, #38BDF8)" : "1px solid var(--border-subtle)",
                      color: currentTheme === "system" ? "var(--text-primary)" : "var(--text-secondary)",
                      cursor: "pointer",
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: "6px",
                      fontWeight: 700,
                      fontSize: "0.78rem",
                    }}
                  >
                    <span style={{ fontSize: "1.3rem" }}>💻</span>
                    <span>Tự động</span>
                  </button>
                </div>
              </div>

              {/* Language Settings Card */}
              <div style={{ background: "var(--bg-elevated)", padding: "22px", borderRadius: "10px", border: "1px solid var(--border-subtle)" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 800, margin: "0 0 8px 0", color: "var(--text-primary)" }}>
                  🌐 Ngôn Ngữ Hệ Thống (Language)
                </h3>
                <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", margin: "0 0 16px 0" }}>
                  Lựa chọn ngôn ngữ thể hiện giao diện và báo cáo thẩm định.
                </p>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                  {/* Vietnamese */}
                  <button
                    type="button"
                    onClick={() => handleLanguageChange("vi")}
                    style={{
                      padding: "12px 14px",
                      borderRadius: "8px",
                      background: language === "vi" ? "rgba(56, 189, 248, 0.15)" : "var(--bg-primary)",
                      border: language === "vi" ? "2px solid var(--accent, #38BDF8)" : "1px solid var(--border-subtle)",
                      color: language === "vi" ? "var(--text-primary)" : "var(--text-secondary)",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "10px",
                      fontWeight: 700,
                      fontSize: "0.82rem",
                    }}
                  >
                    <span style={{ fontSize: "1.3rem" }}>🇻🇳</span>
                    <span>Tiếng Việt</span>
                  </button>

                  {/* English */}
                  <button
                    type="button"
                    onClick={() => handleLanguageChange("en")}
                    style={{
                      padding: "12px 14px",
                      borderRadius: "8px",
                      background: language === "en" ? "rgba(56, 189, 248, 0.15)" : "var(--bg-primary)",
                      border: language === "en" ? "2px solid var(--accent, #38BDF8)" : "1px solid var(--border-subtle)",
                      color: language === "en" ? "var(--text-primary)" : "var(--text-secondary)",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "10px",
                      fontWeight: 700,
                      fontSize: "0.82rem",
                    }}
                  >
                    <span style={{ fontSize: "1.3rem" }}>🇺🇸</span>
                    <span>English</span>
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
                    <span>✨ Chế độ Tinh gọn (Minimalist Workspace — Ẩn các badge và nút bấm dư thừa trên màn hình chính)</span>
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
                  Thông Tin Cá Nhân
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
                      <img src={user.avatar_url} alt={user.display_name} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    ) : (
                      user?.display_name?.charAt(0) || "U"
                    )}
                  </div>
                  <div>
                    <div style={{ fontSize: "1.05rem", fontWeight: 800, color: "var(--text-primary)" }}>
                      {user?.display_name || "Chưa đặt tên"}
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
                      {user?.email || "developer@advertest.ai"}
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "4px" }}>
                      <span style={{ fontSize: "0.68rem", fontWeight: 700, padding: "2px 8px", borderRadius: "4px", background: "#4285F425", color: "#4285F4" }}>
                        🌐 Google SSO Authenticated
                      </span>
                    </div>
                  </div>
                </div>

                <form onSubmit={handleSaveProfile} style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
                  <div>
                    <label className="config-panel__label" style={{ fontSize: "0.75rem" }}>Tên hiển thị</label>
                    <input
                      type="text"
                      className="select-field"
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                      style={{ width: "100%", padding: "8px 12px", fontSize: "0.85rem" }}
                    />
                  </div>

                  <div>
                    <label className="config-panel__label" style={{ fontSize: "0.75rem" }}>Địa chỉ Email (Google)</label>
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
                    {isSavingProfile ? "Đang lưu..." : "💾 Lưu Thay Đổi"}
                  </button>
                </form>
              </div>

              {/* Resource Quotas Card */}
              <div style={{ background: "var(--bg-elevated)", padding: "20px", borderRadius: "10px", border: "1px solid var(--border-subtle)" }}>
                <h3 style={{ fontSize: "1rem", fontWeight: 800, margin: "0 0 16px 0", color: "var(--text-primary)" }}>
                  Hạn Mức Tài Nguyên (Quotas)
                </h3>

                <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
                  {/* Storage */}
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", fontWeight: 700, marginBottom: "6px" }}>
                      <span>📦 Dung lượng lưu trữ Artifacts</span>
                      <span style={{ color: "var(--accent)" }}>245 MB / 10,240 MB (2.4%)</span>
                    </div>
                    <div style={{ width: "100%", height: "8px", background: "var(--bg-primary)", borderRadius: "4px", overflow: "hidden" }}>
                      <div style={{ width: "2.4%", height: "100%", background: "#38BDF8", borderRadius: "4px" }} />
                    </div>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: "4px" }}>
                      Lưu trữ ảnh Clean, Attacked, Noise Mask, và Checkpoint Retrained.
                    </div>
                  </div>

                  {/* GPU Compute */}
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", fontWeight: 700, marginBottom: "6px" }}>
                      <span>⚡ Giờ tính toán GPU Benchmark</span>
                      <span style={{ color: "#10B981" }}>12.4 hrs / 100.0 hrs (12.4%)</span>
                    </div>
                    <div style={{ width: "100%", height: "8px", background: "var(--bg-primary)", borderRadius: "4px", overflow: "hidden" }}>
                      <div style={{ width: "12.4%", height: "100%", background: "#10B981", borderRadius: "4px" }} />
                    </div>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: "4px" }}>
                      Thời gian GPU NVIDIA dùng để sinh gradient đối kháng và suy luận perception.
                    </div>
                  </div>

                  {/* Status Badge */}
                  <div style={{ padding: "12px", borderRadius: "6px", background: "rgba(16, 185, 129, 0.1)", border: "1px solid rgba(16, 185, 129, 0.25)", marginTop: "8px" }}>
                    <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#10B981" }}>
                      🟢 Trạng thái tài khoản: HOẠT ĐỘNG (ACTIVE)
                    </div>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                      Tài khoản của bạn có đầy đủ quyền truy cập các tính năng kiểm thử an toàn.
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
                Vai trò tài khoản hiện tại của bạn:
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "8px" }}>
                <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--accent)" }}>
                  {ROLES_INFO.find((r) => r.id === currentRole)?.name || currentRole}
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  (Thuộc tính định danh của tài khoản được quản lý bởi Quản trị viên)
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
                          VAI TRÒ CỦA BẠN
                        </span>
                      )}
                    </div>

                    <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", margin: 0 }}>
                      {r.description}
                    </p>

                    <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "10px" }}>
                      <div style={{ fontSize: "0.68rem", fontWeight: 700, color: "var(--text-muted)", marginBottom: "6px" }}>
                        Quyền hạn chính:
                      </div>
                      <ul style={{ margin: 0, paddingLeft: "16px", fontSize: "0.72rem", color: "var(--text-secondary)", display: "flex", flexDirection: "column", gap: "4px" }}>
                        {r.permissions.map((p, idx) => (
                          <li key={idx}>{p}</li>
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
                    <span>📊 Tích Hợp Weights & Biases (W&B)</span>
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
                      {wandbConnected ? "✅ ĐÃ KẾT NỐI" : "⚪ CHƯA KẾT NỐI"}
                    </span>
                  </h3>
                  <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", margin: "4px 0 0 0" }}>
                    Tự động đồng bộ hóa các bài kiểm thử độ bền vững, ma trận IoU và hình ảnh bounding box lên nền tảng MLOps W&B.
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
                  🔑 Lấy API Key tại wandb.ai ↗
                </a>
              </div>

              <form onSubmit={handleSaveWandb} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                {/* API Key */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                    <label className="config-panel__label" style={{ fontSize: "0.75rem" }}>
                      W&B API Key (40 ký tự)
                    </label>
                    <button
                      type="button"
                      onClick={() => setShowWandbKey(!showWandbKey)}
                      style={{ background: "none", border: "none", color: "var(--accent)", fontSize: "0.7rem", cursor: "pointer", fontWeight: 600 }}
                    >
                      {showWandbKey ? "Ẩn khóa" : "Hiện khóa"}
                    </button>
                  </div>
                  <input
                    type={showWandbKey ? "text" : "password"}
                    className="select-field"
                    placeholder={wandbMaskedKey ? `Khóa hiện tại: ${wandbMaskedKey}` : "Nhập API Key từ wandb.ai/authorize..."}
                    value={wandbKey}
                    onChange={(e) => setWandbKey(e.target.value)}
                    style={{ width: "100%", padding: "9px 12px", fontSize: "0.85rem", fontFamily: "var(--font-mono)" }}
                  />
                  <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: "4px" }}>
                    Khóa API được lưu mã hóa an toàn trên hệ thống máy chủ nội bộ.
                  </div>
                </div>

                {/* Team / Entity & Project */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                  <div>
                    <label className="config-panel__label" style={{ fontSize: "0.75rem" }}>
                      W&B Entity / Team (Tùy chọn)
                    </label>
                    <input
                      type="text"
                      className="select-field"
                      placeholder="vd: team-ai-safety hoặc tên tài khoản"
                      value={wandbEntity}
                      onChange={(e) => setWandbEntity(e.target.value)}
                      style={{ width: "100%", padding: "9px 12px", fontSize: "0.85rem" }}
                    />
                  </div>
                  <div>
                    <label className="config-panel__label" style={{ fontSize: "0.75rem" }}>
                      W&B Project Name
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
                    Tự động đồng bộ hóa kết quả mỗi khi bấm "Run Test"
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
                    {isTestingWandb ? "⏳ Đang kiểm tra..." : "⚡ Kiểm Tra Kết Nối"}
                  </button>

                  <button
                    type="submit"
                    disabled={isSavingWandb}
                    className="action-button action-button--primary"
                    style={{ padding: "9px 22px", fontSize: "0.82rem", fontWeight: 700 }}
                  >
                    {isSavingWandb ? "Đang lưu..." : "💾 Lưu Cấu Hình W&B"}
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
