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
export function getEvidenceState(provenance, simulationOnly, status) {
  if (provenance?.demo_mode || provenance?.is_demo) {
    return "DEMO";
  }
  if (
    status === "WAITING_FOR_GPU_VALIDATION" ||
    provenance?.status === "WAITING_FOR_GPU_VALIDATION"
  ) {
    return "WAITING_FOR_GPU_VALIDATION";
  }
  if (
    status === "WAITING_FOR_EXTERNAL_DATA" ||
    provenance?.status === "WAITING_FOR_EXTERNAL_DATA"
  ) {
    return "WAITING_FOR_EXTERNAL_DATA";
  }
  if (simulationOnly === true || provenance?.simulation_only === true) {
    return "SIMULATION";
  }
  if (
    provenance?.checkpoint_sha256 ||
    provenance?.split_manifest_hash ||
    simulationOnly === false
  ) {
    return "REAL_ARTIFACT";
  }
  return "NO_DATA";
}

const BADGE_CONFIG = {
  DEMO: {
    label: "DEMO",
    icon: Layers,
    className: "bg-amber-100 text-amber-900 border-amber-300 font-semibold",
    description: "Chế độ mô phỏng trực quan minh họa",
  },
  SIMULATION: {
    label: "SIMULATION",
    icon: Cpu,
    className: "bg-purple-100 text-purple-900 border-purple-300 font-semibold",
    description: "Đánh giá trên tập dữ liệu mô phỏng synthetic",
  },
  REAL_ARTIFACT: {
    label: "REAL ARTIFACT",
    icon: Database,
    className: "bg-emerald-100 text-emerald-900 border-emerald-300 font-semibold",
    description: "Dữ liệu và mô hình thực tế đã xác thực provenance",
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
  state: explicitState,
  className,
}) {
  const stateKey = explicitState || getEvidenceState(provenance, simulationOnly, status);
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
