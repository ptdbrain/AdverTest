import React from "react";
import { cn } from "@/lib/utils";

const VARIANT_STYLES = {
  default: "bg-slate-100 text-slate-700 border-slate-200",
  primary: "bg-blue-50 text-blue-700 border-blue-200",
  success: "bg-emerald-50 text-emerald-700 border-emerald-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  danger: "bg-red-50 text-red-700 border-red-200",
  purple: "bg-purple-50 text-purple-700 border-purple-200",
  teal: "bg-teal-50 text-teal-700 border-teal-200",
};

export default function Badge({
  children,
  variant = "default",
  className,
  dot = false,
  ...props
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-medium border",
        VARIANT_STYLES[variant] || VARIANT_STYLES.default,
        className
      )}
      {...props}
    >
      {dot && (
        <span
          className={cn(
            "w-1.5 h-1.5 rounded-full",
            variant === "success" && "bg-emerald-500",
            variant === "danger" && "bg-red-500",
            variant === "warning" && "bg-amber-500",
            variant === "primary" && "bg-blue-500",
            variant === "purple" && "bg-purple-500",
            variant === "teal" && "bg-teal-500",
            variant === "default" && "bg-slate-400"
          )}
        />
      )}
      {children}
    </span>
  );
}
