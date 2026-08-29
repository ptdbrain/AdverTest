"use client";

import Link from "next/link";
import AdminView from "@/components/AdminView.jsx";

export default function AdminPage() {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        <header className="flex items-center justify-between border-b border-slate-800 pb-5">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              <span className="rounded bg-indigo-600/20 px-2 py-0.5 text-xs text-indigo-400 font-mono border border-indigo-500/30">ADMIN</span>
              AdverTest Governance & Operations
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Multi-user management, storage quotas, compute usage accounting, and audit compliance logs.
            </p>
          </div>
          <Link
            href="/"
            className="rounded-lg bg-slate-900 border border-slate-700 px-3 py-1.5 text-xs font-semibold text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
          >
            ← Back to Workbench
          </Link>
        </header>

        <AdminView />
      </div>
    </main>
  );
}
