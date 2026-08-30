"use client";

import React from "react";
import {
  ShieldAlert,
  Cpu,
  Database,
  Layers,
  HelpCircle,
  Clock,
} from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Derive the exact provenance evidence state from backend metadata.
 */
export function getEvidenceState(_provenance, _simulationOnly, _status, evidence) {
  return evidence?.status || "NO_DATA";
}

const BADGE_CONFIG = {
  VERIFIED: {
    label: "VERIFIED",
    icon: Database,
    className: "bg-emerald-100 text-emerald-900 border-emerald-300 font-semibold",
    description: "Benchmark evidence đầy đủ và đã xác minh",
  },
  NOT_ELIGIBLE: {
    label: "NOT ELIGIBLE",
    icon: ShieldAlert,
    className: "bg-orange-100 text-orange-900 border-orange-300 font-semibold",
    description: "Thiếu evidence; không được kết luận benchmark hoặc promotion",
  },
  INVALID: {
    label: "INVALID",
    icon: ShieldAlert,
    className: "bg-red-100 text-red-900 border-red-300 font-semibold",
    description: "Evidence không hợp lệ",
  },
  WAITING_FOR_GPU_VALIDATION: {
    label: "WAITING_FOR_GPU_VALIDATION",
    icon: Clock,
    className: "bg-yellow-100 text-yellow-900 border-yellow-300 font-semibold",
    description: "Chờ xác thực phần cứng GPU/CUDA chuyên dụng",
  },
  WAITING_FOR_EXTERNAL_DATA: {
    label: "WAITING_FOR_EXTERNAL_DATA",
    icon: ShieldAlert,
    className: "bg-orange-100 text-orange-900 border-orange-300 font-semibold",
    description: "Chờ nạp dataset ngoại vi chính thức",
  },
  NO_DATA: {
    label: "NO DATA",
    icon: HelpCircle,
    className: "bg-slate-100 text-slate-800 border-slate-300 font-semibold",
    description: "Chưa có dữ liệu provenance",
  },
};

/**
 * P2.7: Production Evidence Badge
 * Renders source evidence state derived directly from backend provenance.
 * Accessibility compliant: minimum 12px font size, icon + text, high contrast.
 */
export default function EvidenceBadge({
  provenance,
  simulationOnly,
  status,
  evidence,
  state: explicitState,
  className,
}) {
  const stateKey = explicitState || getEvidenceState(provenance, simulationOnly, status, evidence);
  const config = BADGE_CONFIG[stateKey] || BADGE_CONFIG.NO_DATA;
  const Icon = config.icon;

  return (
    <span
      role="status"
      aria-label={`Trạng thái nguồn: ${config.label}`}
      title={config.description}
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[12px] leading-tight border transition-colors select-none shadow-xs",
        config.className,
        className
      )}
    >
      <Icon className="w-3.5 h-3.5 flex-shrink-0" aria-hidden="true" />
      <span>{config.label}</span>
    </span>
  );
}
