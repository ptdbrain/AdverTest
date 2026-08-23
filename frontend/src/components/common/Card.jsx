import React from "react";
import { cn } from "@/lib/utils";

export default function Card({
  children,
  className,
  title,
  subtitle,
  headerAction,
  noPadding = false,
  ...props
}) {
  return (
    <div
      className={cn(
        "bg-white border border-[#E2E8F0] rounded-[10px] shadow-[0_1px_2px_rgba(15,23,42,0.03),0_2px_6px_rgba(15,23,42,0.025)] transition-all",
        className
      )}
      {...props}
    >
      {(title || headerAction) && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
          <div>
            {title && (
              <h3 className="text-[14px] font-semibold text-[#0F172A] leading-snug">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-[11px] text-slate-500 mt-0.5">{subtitle}</p>
            )}
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      )}
      <div className={cn(!noPadding && "p-4")}>{children}</div>
    </div>
  );
}
