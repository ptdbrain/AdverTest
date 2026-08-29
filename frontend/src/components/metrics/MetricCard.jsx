import React from "react";
import MetricSparkline from "./MetricSparkline";
import { cn } from "@/lib/utils";

export default function MetricCard({
  title,
  value,
  trend,
  trendLabel,
  trendType = "up",
  icon: Icon,
  color = "blue",
  sparkline = [],
  className,
}) {
  const iconBgMap = {
    blue: "bg-blue-50 text-blue-600 border-blue-100",
    purple: "bg-purple-50 text-purple-600 border-purple-100",
    teal: "bg-teal-50 text-teal-600 border-teal-100",
    green: "bg-emerald-50 text-emerald-600 border-emerald-100",
    red: "bg-red-50 text-red-600 border-red-100",
  };

  const isPositive = trendType === "up" || trendType === "down-positive";

  return (
    <div
      className={cn(
        "bg-white border border-[#E2E8F0] rounded-[10px] p-3.5 shadow-[0_1px_2px_rgba(15,23,42,0.03),0_2px_6px_rgba(15,23,42,0.025)] flex flex-col justify-between hover:border-slate-300 transition-all",
        className
      )}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="space-y-0.5">
          <p className="text-[12px] font-medium text-slate-500 truncate">{title}</p>
          <p className="text-[24px] font-bold text-[#0F172A] leading-tight tracking-tight">
            {value}
          </p>
        </div>
        {Icon && (
          <div
            className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center border flex-shrink-0",
              iconBgMap[color] || iconBgMap.blue
            )}
          >
            <Icon className="w-4 h-4" />
          </div>
        )}
      </div>

      <div className="flex items-end justify-between pt-1 border-t border-slate-50">
        {trend && (
          <div className="text-[11px] font-medium text-slate-500 leading-tight">
            <span
              className={cn(
                "font-semibold mr-1",
                isPositive ? "text-emerald-600" : "text-red-600"
              )}
            >
              {trend}
            </span>
            {trendLabel && <span className="text-slate-400">{trendLabel}</span>}
          </div>
        )}
        {sparkline && sparkline.length > 0 && (
          <div className="ml-auto">
            <MetricSparkline data={sparkline} color={color} height={22} width={64} />
          </div>
        )}
      </div>
    </div>
  );
}
