# Visual Results Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refit `/experiments/[id]/results` to the supplied AdversAI Lab visual-results layout without changing its existing routes, demo data, or actions.

**Architecture:** Keep the page self-contained in `page.jsx`, but make repeated UI data-driven through local arrays and small presentational helpers. The page will use a desktop-first CSS grid with a 3-column primary evidence area, a fixed-width information rail, 16:9 media viewports, and breakpoint fallbacks for stacked narrow layouts.

**Tech Stack:** Next.js 16, React 19, Tailwind CSS 4 utility classes, Lucide React, Vitest, Testing Library.

## Global Constraints

- Modify only the visual-results page and its focused regression test, plus the required design/plan documents.
- Preserve the current route, sample asset paths, refresh/history/download/benchmark/zoom/export actions, and existing worktree changes.
- Use the existing light evidence-led product vocabulary: tinted neutrals, blue active actions, green intact state, and red attacked state.
- Do not add dependencies, API calls, or generated image assets.
- Do not rely on color alone for clean/attacked/status meaning; retain text labels and accessible names.

### Task 1: Add a failing structural regression test

**Files:**
- Create: `frontend/src/components/__tests__/visual-results-page.test.jsx`
- Test: `frontend/src/components/__tests__/visual-results-page.test.jsx`

**Interfaces:**
- Consumes: `VisualResultsPage` from `@/app/experiments/[id]/results/page`.
- Produces: regression coverage for the primary comparison, experiment rail, deep-dive evidence row, and quick-observation section.

- [ ] **Step 1: Write the failing test**

```jsx
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import VisualResultsPage from "@/app/experiments/[id]/results/page";

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "EXP-2025-0512-001" }),
}));

describe("VisualResultsPage layout contract", () => {
  it("renders the paired evidence bands and experiment rail", () => {
    render(<VisualResultsPage />);

    expect(screen.getByTestId("visual-results-comparison")).toBeDefined();
    expect(screen.getByTestId("clean-result-panel")).toBeDefined();
    expect(screen.getByTestId("attacked-result-panel")).toBeDefined();
    expect(screen.getByTestId("experiment-info-panel")).toBeDefined();
    expect(screen.getByText("Bản đồ khác biệt (|Δ|)")).toBeDefined();
    expect(screen.getByText("Bản đồ chú ý (Attention)")).toBeDefined();
    expect(screen.getByText("Nhiễu đối kháng (Perturbation)")).toBeDefined();
    expect(screen.getByText("Quan sát nhanh")).toBeDefined();
  });
});
```

- [ ] **Step 2: Run the focused test and verify it fails for the missing contract**

Run from `frontend`:

```powershell
npm test -- --run src/components/__tests__/visual-results-page.test.jsx
```

Expected: FAIL because the current page does not expose the new structural test ids.

### Task 2: Implement the approved visual refit

**Files:**
- Modify: `frontend/src/app/experiments/[id]/results/page.jsx`

**Interfaces:**
- Consumes: existing sample images and current local page state (`zoomRegion`, `expId`).
- Produces: the same page route and actions with the approved four-band layout.

- [ ] **Step 1: Add local presentation data and reusable helpers**

Define local arrays for clean/attacked detections, metrics, experiment metadata, and deep-dive panels. Render repeated metric cells and metadata rows from those arrays so labels and spacing stay consistent without adding a new dependency or file.

- [ ] **Step 2: Replace the tall primary image layout with the reference grid**

Use a wrapper with `data-testid="visual-results-comparison"`, two equal result panels with `data-testid="clean-result-panel"` and `data-testid="attacked-result-panel"`, and a right rail with `data-testid="experiment-info-panel"`. Set paired image viewports to `aspect-video`, keep detection overlays positioned relative to the image, and retain the VS divider between the panels.

- [ ] **Step 3: Align the metric, deep-dive, and action bands**

Keep the four metric cells plus Top prediction under each image, render the five analysis panels in a responsive grid, and preserve zoom selection and all existing action handlers. Use Lucide icons for the metadata/status rail instead of emoji markers.

- [ ] **Step 4: Add responsive fallbacks and accessible labels**

At narrow widths, collapse the primary grid to one column, stack the experiment rail after the paired panels, allow the deep-dive row to wrap, and retain visible text labels for status and before/after meaning. Add `type="button"` to action buttons that are not form submits and use descriptive image alt text.

### Task 3: Run regression checks and inspect the rendered page

**Files:**
- Verify: `frontend/src/app/experiments/[id]/results/page.jsx`
- Verify: `frontend/src/components/__tests__/visual-results-page.test.jsx`

**Interfaces:**
- Consumes: the completed page and focused test.
- Produces: fresh test, full suite, build, and browser screenshot evidence.

- [ ] **Step 1: Run the focused test and confirm it passes**

```powershell
npm test -- --run src/components/__tests__/visual-results-page.test.jsx
```

- [ ] **Step 2: Run the complete frontend test suite**

```powershell
npm test -- --run
```

- [ ] **Step 3: Build the frontend**

```powershell
npm run build
```

- [ ] **Step 4: Perform desktop and narrow viewport visual QA**

Start the app with `npm run dev`, open `/experiments/EXP-2025-0512-001/results`, and verify at desktop and narrow widths that the two media panels remain paired, the right rail is readable, the five analysis panels do not overflow horizontally, and the existing controls remain usable.

- [ ] **Step 5: Review the final diff and report only scoped changes**

```powershell
git diff -- frontend/src/app/experiments/[id]/results/page.jsx frontend/src/components/__tests__/visual-results-page.test.jsx
git status --short
```

Confirm unrelated pending work remains untouched.
