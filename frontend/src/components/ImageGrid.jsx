"use client";

import React from "react";
import VisualResultsView from "./visual/VisualResultsView";

/**
 * ImageGrid serves as the main entry point for the Visual Results workspace.
 * It coordinates the 80/20 comparison view, experiment info sidebar,
 * lower detail maps (diff, attention, perturbation, segmentation, zoom),
 * and quick observations with action triggers.
 */
export default function ImageGrid({
  report,
  samples,
  selectedIndex = 0,
  onSelectSample,
  onReload,
  onViewHistory,
  onDetailedCompare,
}) {
  return (
    <VisualResultsView
      report={report}
      samples={samples}
      selectedIndex={selectedIndex}
      onSelectSample={onSelectSample}
      onReload={onReload}
      onViewHistory={onViewHistory}
      onDetailedCompare={onDetailedCompare}
    />
  );
}
