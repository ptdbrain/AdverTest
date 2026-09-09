import { AlertCircle, AlertTriangle, CheckCircle2, Info, ShieldAlert } from "lucide-react";
import React from "react";
import { cn } from "@/lib/utils";

const RISK_CONFIG = {
  CRITICAL: {
    label: "CRITICAL",
    bg: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950/30 dark:text-red-400 dark:border-red-800",
    dot: "bg-red-500",
    icon: ShieldAlert,
  },
  HIGH: {
    label: "HIGH",
    bg: "bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950/30 dark:text-orange-400 dark:border-orange-800",
    dot: "bg-orange-500",
    icon: AlertTriangle,
  },
  MEDIUM: {
    label: "MEDIUM",
    bg: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/30 dark:text-amber-400 dark:border-amber-800",
    dot: "bg-amber-500",
    icon: AlertCircle,
  },
  LOW: {
    label: "LOW",
    bg: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/30 dark:text-blue-400 dark:border-blue-800",
    dot: "bg-blue-500",
    icon: Info,
  },
  AUTO_PASS: {
    label: "AUTO PASS",
    bg: "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/30 dark:text-emerald-400 dark:border-emerald-800",
    dot: "bg-emerald-500",
    icon: CheckCircle2,
  },
};

export default function RiskBadge({ level = "MEDIUM", showIcon = true, size = "md", className = "" }) {
  const normalizedLevel = String(level || "MEDIUM").toUpperCase();
  const config = RISK_CONFIG[normalizedLevel] || RISK_CONFIG.MEDIUM;
  const IconComponent = config.icon;

  const sizeClasses = {
    sm: "text-[10px] px-1.5 py-0.5 gap-1",
    md: "text-xs px-2 py-0.5 gap-1.5",
    lg: "text-sm px-3 py-1 gap-2",
  };

  const iconSizes = {
    sm: "w-3 h-3",
    md: "w-3.5 h-3.5",
    lg: "w-4 h-4",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center font-semibold rounded-full border shadow-xs transition-colors",
        config.bg,
        sizeClasses[size] || sizeClasses.md,
        className,
      )}
    >
      {showIcon && <IconComponent className={cn("shrink-0", iconSizes[size] || iconSizes.md)} />}
      <span>{config.label}</span>
    </span>
  );
}
