import React from "react";

export default function MetricSparkline({
  data = [],
  color = "blue",
  height = 28,
  width = 80,
}) {
  if (!data || data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data
    .map((val, idx) => {
      const x = (idx / (data.length - 1)) * width;
      const y = height - ((val - min) / range) * (height - 6) - 3;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const colorMap = {
    blue: "#2563EB",
    purple: "#7C3AED",
    teal: "#0EA5A8",
    green: "#16A34A",
    red: "#EF4444",
    amber: "#F59E0B",
  };

  const strokeColor = colorMap[color] || colorMap.blue;

  return (
    <svg width={width} height={height} className="overflow-visible inline-block">
      <polyline
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
}
