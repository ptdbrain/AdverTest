import React from "react";
import Link from "next/link";
import { ChevronRight } from "lucide-react";

export default function PageHeader({
  title,
  subtitle,
  breadcrumb = [],
  actions = null,
}) {
  return (
    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 pb-2 border-b border-slate-200">
      <div className="space-y-1">
        {breadcrumb && breadcrumb.length > 0 && (
          <nav className="flex items-center space-x-1.5 text-xs text-slate-500 mb-1">
            {breadcrumb.map((item, idx) => {
              const isLast = idx === breadcrumb.length - 1;
              return (
                <React.Fragment key={item.label || item}>
                  {idx > 0 && <ChevronRight className="w-3 h-3 text-slate-400" />}
                  {item.href && !isLast ? (
                    <Link href={item.href} className="hover:text-blue-600 transition-colors">
                      {item.label}
                    </Link>
                  ) : (
                    <span className={isLast ? "font-semibold text-slate-700" : ""}>
                      {item.label || item}
                    </span>
                  )}
                </React.Fragment>
              );
            })}
          </nav>
        )}
        <h1 className="text-[22px] md:text-[24px] font-bold text-[#0F172A] tracking-tight leading-snug">
          {title}
        </h1>
        {subtitle && (
          <p className="text-[13px] text-slate-500 max-w-3xl leading-normal">
            {subtitle}
          </p>
        )}
      </div>

      {actions && (
        <div className="flex items-center gap-2.5 flex-shrink-0">
          {actions}
        </div>
      )}
    </div>
  );
}
