import React from "react";
import { cn } from "@/lib/utils";

const BUTTON_VARIANTS = {
  primary: "bg-blue-600 hover:bg-blue-700 text-white shadow-sm border border-blue-600",
  secondary: "bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 shadow-sm",
  outline: "bg-transparent hover:bg-blue-50 text-blue-600 border border-blue-300",
  ghost: "bg-transparent hover:bg-slate-100 text-slate-600",
  danger: "bg-red-600 hover:bg-red-700 text-white shadow-sm border border-red-600",
  success: "bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm border border-emerald-600",
};

const BUTTON_SIZES = {
  sm: "px-2.5 py-1 text-xs rounded-md",
  md: "px-3.5 py-1.5 text-[13px] rounded-lg",
  lg: "px-5 py-2.5 text-[14px] rounded-lg font-semibold",
};

export default function Button({
  children,
  variant = "primary",
  size = "md",
  icon: Icon,
  className,
  disabled = false,
  ...props
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 font-medium transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer select-none",
        BUTTON_VARIANTS[variant] || BUTTON_VARIANTS.primary,
        BUTTON_SIZES[size] || BUTTON_SIZES.md,
        className
      )}
      {...props}
    >
      {Icon && <Icon className="w-4 h-4 flex-shrink-0" />}
      {children}
    </button>
  );
}
