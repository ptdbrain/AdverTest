import React from "react";

import Badge from "@/components/common/Badge";

export default function DemoFixtureBadge({ visible = false, className = "" }) {
  if (!visible) return null;
  return (
    <Badge variant="purple" dot className={className}>
      DEMO / SIMULATED
    </Badge>
  );
}
