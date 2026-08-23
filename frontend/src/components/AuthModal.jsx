"use client";

import React, { useState, useEffect } from "react";
import { loginUser, registerUser, getCurrentUser } from "@/lib/api";

export default function AuthModal({ isOpen, onClose, onAuthChange }) {
  const [activeTab, setActiveTab] = useState("login"); // "login" | "register" | "profile"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    try {
      const savedUser = localStorage.getItem("advertest_auth_user");
      if (savedUser) {
        setCurrentUser(JSON.parse(savedUser));
      }
    } catch {}
  }, [isOpen]);

  if (!isOpen) return null;

  const handleLogin = async (e, customEmail, customPass) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const loginEmail = customEmail || email;
      const loginPass = customPass || password;
      const res = await loginUser({ email: loginEmail, password: loginPass });
      
      localStorage.setItem("advertest_auth_token", res.access_token);
      localStorage.setItem("advertest_auth_user", JSON.stringify(res.user));
      setCurrentUser(res.user);
      if (onAuthChange) onAuthChange(res.user);
      onClose();
    } catch (err) {
      setError(err.message || "Failed to sign in. Please verify credentials.");
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await registerUser({
        email,
        password,
        display_name: displayName || email.split("@")[0],
      });
      localStorage.setItem("advertest_auth_token", res.access_token);
      localStorage.setItem("advertest_auth_user", JSON.stringify(res.user));
      setCurrentUser(res.user);
      if (onAuthChange) onAuthChange(res.user);
      onClose();
    } catch (err) {
      setError(err.message || "Registration failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("advertest_auth_token");
    localStorage.removeItem("advertest_auth_user");
    setCurrentUser(null);
    if (onAuthChange) onAuthChange(null);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="relative w-full max-w-md rounded-xl border border-slate-700 bg-slate-900 shadow-2xl p-6 text-slate-200">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white text-lg font-bold"
        >
          ✕
        </button>

        {currentUser ? (
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="h-12 w-12 rounded-full bg-gradient-to-tr from-cyan-600 to-indigo-600 flex items-center justify-center font-bold text-lg text-white shadow">
                {currentUser.display_name?.charAt(0) || "U"}
              </div>
              <div>
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  {currentUser.display_name}
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${currentUser.role === "ADMIN" ? "bg-amber-500/20 text-amber-300 border border-amber-500/30" : "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"}`}>
                    {currentUser.role === "ADMIN" ? "👑 ADMIN" : "👤 USER"}
                  </span>
                </h3>
                <p className="text-xs text-slate-400">{currentUser.email}</p>
              </div>
            </div>

            <div className="space-y-2 mb-6 rounded-lg bg-slate-800/60 p-3 text-xs border border-slate-700/60">
              <div className="flex justify-between">
                <span className="text-slate-400">Account Status:</span>
                <span className="font-semibold text-emerald-400">{currentUser.status}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Compute Quota:</span>
                <span className="font-semibold text-slate-300">{currentUser.compute_quota_hours} hrs</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Storage Quota:</span>
                <span className="font-semibold text-slate-300">{Math.round(currentUser.storage_quota_bytes / (1024 * 1024 * 1024))} GB</span>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleLogout}
                className="flex-1 rounded-lg border border-red-500/30 bg-red-500/10 py-2 text-xs font-semibold text-red-400 hover:bg-red-500/20 transition-colors"
              >
                Sign Out
              </button>
              <button
                onClick={onClose}
                className="flex-1 rounded-lg bg-cyan-600 hover:bg-cyan-500 py-2 text-xs font-semibold text-white transition-colors"
              >
                Done
              </button>
            </div>
          </div>
        ) : (
          <div>
            <h2 className="text-xl font-bold text-white mb-1">Account & Role Access</h2>
            <p className="text-xs text-slate-400 mb-5">
              Sign in with an existing role or register to save benchmark runs and access admin controls.
            </p>

            {error && (
              <div className="mb-4 rounded-lg bg-rose-500/10 border border-rose-500/30 p-2.5 text-xs text-rose-300">
                {error}
              </div>
            )}

            {/* Quick 1-click Logins */}
            <div className="mb-4 space-y-2">
              <span className="text-[11px] font-semibold text-slate-400 block uppercase tracking-wider">
                ⚡ Quick Switch Role
              </span>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => handleLogin(null, "admin@advertest.ai", "AdminPassword123!")}
                  disabled={loading}
                  className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-2 text-left hover:bg-amber-500/20 transition-colors"
                >
                  <span className="text-xs font-bold text-amber-300 flex items-center gap-1">
                    👑 Admin
                  </span>
                  <span className="text-[10px] text-slate-400 block">admin@advertest.ai</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleLogin(null, "user@advertest.ai", "UserPassword123!")}
                  disabled={loading}
                  className="rounded-lg border border-cyan-500/30 bg-cyan-500/10 p-2 text-left hover:bg-cyan-500/20 transition-colors"
                >
                  <span className="text-xs font-bold text-cyan-300 flex items-center gap-1">
                    👤 Developer
                  </span>
                  <span className="text-[10px] text-slate-400 block">user@advertest.ai</span>
                </button>
              </div>
            </div>

            <div className="relative my-4">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-slate-700" />
              </div>
              <div className="relative flex justify-center text-xs">
                <span className="bg-slate-900 px-2 text-slate-500">Or use email & password</span>
              </div>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-slate-700 mb-4">
              <button
                type="button"
                onClick={() => setActiveTab("login")}
                className={`flex-1 py-2 text-xs font-bold transition-colors border-b-2 ${activeTab === "login" ? "border-cyan-500 text-cyan-400" : "border-transparent text-slate-400 hover:text-slate-200"}`}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("register")}
                className={`flex-1 py-2 text-xs font-bold transition-colors border-b-2 ${activeTab === "register" ? "border-cyan-500 text-cyan-400" : "border-transparent text-slate-400 hover:text-slate-200"}`}
              >
                Register
              </button>
            </div>

            <form onSubmit={activeTab === "login" ? handleLogin : handleRegister} className="space-y-3">
              {activeTab === "register" && (
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Display Name</label>
                  <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    placeholder="e.g. Alex Nguyen"
                    className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                  />
                </div>
              )}

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Email</label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@company.ai"
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Password</label>
                <input
                  type="password"
                  required
                  minLength={8}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="At least 8 characters"
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full mt-2 rounded-lg bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 py-2.5 text-xs font-bold text-white shadow-lg transition-all disabled:opacity-50"
              >
                {loading ? "Processing..." : activeTab === "login" ? "Sign In" : "Create Account"}
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}
