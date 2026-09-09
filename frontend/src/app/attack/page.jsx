"use client";

import { Suspense } from "react";
import ConfigureAttackPage from "../experiments/[id]/attack/page";

export default function AttackAliasPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-slate-900 flex items-center justify-center text-slate-400">
          Loading attack configuration...
        </div>
      }
    >
      <ConfigureAttackPage />
    </Suspense>
  );
}
