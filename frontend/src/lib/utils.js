import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Combines and merges Tailwind CSS classes safely.
 * @param {...any} inputs - List of class names or conditional class objects.
 * @returns {string} Merged class names string.
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

/**
 * Format a number as percentage with optional prefix.
 * @param {number} value - Decimal or percentage value.
 * @param {number} decimals - Number of decimal places.
 * @returns {string} Formatted percentage string.
 */
export function formatPercent(value, decimals = 1) {
  if (value === null || value === undefined || isNaN(value)) return "—";
  return `${Number(value).toFixed(decimals)}%`;
}

/**
 * Format an integer with thousands separator.
 * @param {number} value - Raw number.
 * @returns {string} Formatted integer string.
 */
export function formatNumber(value) {
  if (value === null || value === undefined || isNaN(value)) return "0";
  return new Intl.NumberFormat("vi-VN").format(value);
}

/**
 * Map severity or risk level to badge styles.
 * @param {string} level - Risk level (e.g., 'Rất cao', 'Cao', 'Trung bình', 'Thấp').
 * @returns {string} CSS class names.
 */
export function getSeverityBadgeStyle(level) {
  switch (level) {
    case "Rất cao":
    case "VERY_HIGH":
    case "DANGER":
      return "bg-red-50 text-red-700 border-red-200";
    case "Cao":
    case "HIGH":
    case "WARNING":
      return "bg-amber-50 text-amber-700 border-amber-200";
    case "Trung bình":
    case "MEDIUM":
      return "bg-blue-50 text-blue-700 border-blue-200";
    case "Thấp":
    case "LOW":
    case "SUCCESS":
      return "bg-emerald-50 text-emerald-700 border-emerald-200";
    default:
      return "bg-slate-50 text-slate-700 border-slate-200";
  }
}
