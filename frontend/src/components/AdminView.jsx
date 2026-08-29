"use client";

import React, { useEffect, useState } from "react";

export default function AdminView() {
  const [users, setUsers] = useState([]);
  const [quotas, setQuotas] = useState(null);
  const [auditLogs, setAuditLogs] = useState([]);
  const [activeTab, setActiveTab] = useState("users");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchData = async () => {
    try {
      setLoading(true);
      setError("");

      const token = localStorage.getItem("advertest_auth_token");
      const headers = token ? { Authorization: `Bearer ${token}` } : {};

      const [usersRes, quotasRes, logsRes] = await Promise.all([
        fetch("/api/v1/admin/users", { headers }),
        fetch("/api/v1/admin/quotas", { headers }),
        fetch("/api/v1/admin/audit-logs", { headers }),
      ]);

      if (!usersRes.ok) {
        throw new Error(usersRes.status === 403 ? "Administrator privileges required to view this panel." : "Failed to load users");
      }

      setUsers(await usersRes.json());
      if (quotasRes.ok) setQuotas(await quotasRes.json());
      if (logsRes.ok) setAuditLogs(await logsRes.json());
    } catch (err) {
      setError(err.message || "Failed to load admin telemetry.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleToggleUserStatus = async (userId, currentStatus) => {
    const nextStatus = currentStatus === "ACTIVE" ? "SUSPENDED" : "ACTIVE";
    try {
      const token = localStorage.getItem("advertest_auth_token");
      const res = await fetch(`/api/v1/admin/users/${userId}/status`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: token ? `Bearer ${token}` : "",
        },
        body: JSON.stringify({ status: nextStatus }),
      });
      if (res.ok) {
        fetchData();
      }
    } catch (err) {
      console.error("Failed to update user status:", err);
    }
  };

  if (loading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <div className="flex items-center gap-3 text-sm text-slate-400">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
          <span>Loading Admin Console...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-500/30 bg-red-950/20 p-6 text-center text-red-300">
        <h3 className="text-base font-semibold">Access Restricted</h3>
        <p className="mt-1 text-sm">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Metric Telemetry Cards */}
      {quotas && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 backdrop-blur">
            <span className="text-xs text-slate-400">Total Registered Users</span>
            <div className="mt-1 text-2xl font-bold text-slate-100">{quotas.total_users}</div>
            <span className="text-[11px] text-emerald-400">{quotas.active_users} active accounts</span>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 backdrop-blur">
            <span className="text-xs text-slate-400">Total Storage Consumed</span>
            <div className="mt-1 text-2xl font-bold text-slate-100">{quotas.total_storage_used_mb} MB</div>
            <span className="text-[11px] text-slate-500">Across all user artifacts</span>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 backdrop-blur">
            <span className="text-xs text-slate-400">Total Benchmarks Run</span>
            <div className="mt-1 text-2xl font-bold text-slate-100">{quotas.total_runs_executed}</div>
            <span className="text-[11px] text-indigo-400">Durable SQLite/Postgres</span>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4 backdrop-blur">
            <span className="text-xs text-slate-400">Compute Time Utilized</span>
            <div className="mt-1 text-2xl font-bold text-slate-100">{quotas.total_compute_hours} hrs</div>
            <span className="text-[11px] text-slate-500">Worker execution</span>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab("users")}
          className={`rounded-lg px-4 py-2 text-xs font-semibold transition-colors ${
            activeTab === "users" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          User Accounts ({users.length})
        </button>
        <button
          onClick={() => setActiveTab("audit")}
          className={`rounded-lg px-4 py-2 text-xs font-semibold transition-colors ${
            activeTab === "audit" ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          Compliance Audit Logs ({auditLogs.length})
        </button>
      </div>

      {/* Users Table */}
      {activeTab === "users" && (
        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/40">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-800 bg-slate-950/60 font-semibold text-slate-400">
              <tr>
                <th className="p-3">User ID</th>
                <th className="p-3">Display Name</th>
                <th className="p-3">Email</th>
                <th className="p-3">Role</th>
                <th className="p-3">Status</th>
                <th className="p-3">Last Login</th>
                <th className="p-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {users.map((u) => (
                <tr key={u.id} className="hover:bg-slate-800/30 transition-colors">
                  <td className="p-3 font-mono text-slate-400">{u.id}</td>
                  <td className="p-3 font-semibold text-slate-100">{u.display_name}</td>
                  <td className="p-3 text-slate-400">{u.email}</td>
                  <td className="p-3">
                    <span
                      className={`rounded px-2 py-0.5 font-bold uppercase text-[10px] ${
                        u.role === "ADMIN" ? "bg-purple-500/20 text-purple-400 border border-purple-500/30" : "bg-slate-800 text-slate-400"
                      }`}
                    >
                      {u.role}
                    </span>
                  </td>
                  <td className="p-3">
                    <span
                      className={`rounded px-2 py-0.5 font-semibold text-[10px] ${
                        u.status === "ACTIVE" ? "bg-emerald-500/20 text-emerald-400" : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {u.status}
                    </span>
                  </td>
                  <td className="p-3 text-slate-400">{u.last_login_at ? new Date(u.last_login_at).toLocaleString() : "Never"}</td>
                  <td className="p-3 text-right">
                    <button
                      onClick={() => handleToggleUserStatus(u.id, u.status)}
                      className={`rounded px-2.5 py-1 text-[11px] font-semibold transition-colors ${
                        u.status === "ACTIVE"
                          ? "bg-red-900/30 text-red-300 hover:bg-red-900/50"
                          : "bg-emerald-900/30 text-emerald-300 hover:bg-emerald-900/50"
                      }`}
                    >
                      {u.status === "ACTIVE" ? "Suspend" : "Activate"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Audit Logs Table */}
      {activeTab === "audit" && (
        <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/40">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-slate-800 bg-slate-950/60 font-semibold text-slate-400">
              <tr>
                <th className="p-3">Timestamp</th>
                <th className="p-3">Actor</th>
                <th className="p-3">Action</th>
                <th className="p-3">Resource</th>
                <th className="p-3">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-[11px] text-slate-300">
              {auditLogs.map((log) => (
                <tr key={log.id} className="hover:bg-slate-800/30 transition-colors">
                  <td className="p-3 text-slate-400">{new Date(log.timestamp).toLocaleString()}</td>
                  <td className="p-3 text-indigo-400">{log.actor_user_id}</td>
                  <td className="p-3 font-semibold text-slate-200">{log.action}</td>
                  <td className="p-3 text-slate-400">{log.resource_type}:{log.resource_id}</td>
                  <td className="p-3 text-slate-400 max-w-xs truncate">{log.detail_json}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
