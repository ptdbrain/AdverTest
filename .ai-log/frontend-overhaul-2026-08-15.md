# AI Log — AdverTest Frontend Overhaul

**Date:** 2026-08-15
**Branch:** `feat/advertest-core-wave0`
**Task:** Hoàn thiện FE theo `docs/FRONTEND_HANDOFF.md`

---

## 1. Analysis Phase

- Pulled latest code from `feat/advertest-core-wave0`
- Read `FRONTEND_HANDOFF.md` — spec yêu cầu:
  - 2-column layout (30% config / 70% image grid)
  - Lưới 4 ảnh 2×2 với metrics chip
  - Attack accordion với 5 ô severity
  - Bảng màu trung tính, KHÔNG neon
  - Progress bar mượt mà
- Analyzed existing codebase: Next.js 16 + React 19 + TailwindCSS v4
- Identified gaps between current UI and spec

## 2. Changes Made

### globals.css — Complete Design System Overhaul
- **Palette**: Chuyển từ purple/neon → Slate/Zinc trung tính (#0B0F14, #141A22)
- **Accent**: Steel Blue (#3884CC) thay vì cyan glow
- **Layout**: `grid-template-columns: minmax(300px, 32%) minmax(0, 1fr)`
- **New components**: `.segmented-control`, `.attack-tabs`, `.severity-squares`, `.severity-expanded`, `.metric-chips`, `.image-grid`, `.progress-inline`, `.app-header`, `.simulation-tag`
- **Animations**: `fadeIn`, `scaleIn`, `slideDown`, `shimmer`, `progressPulse`, `stagger-children`
- **Removed**: Neon glow shadows, gradient backgrounds, progress ring

### layout.js
- Removed `simulation-banner` (moved to header)
- Simplified sidebar (no banner-height offset)

### page.js
- Added `HeaderBar` component: AdverTest | Task | Status | SIMULATION ONLY
- Replaced `Workspace` with `ImageGrid` for evidence tab
- Added `gridIndex` state for sample navigation
- Passed `progress`, `progressDetail`, `runStatus` to ConfigPanel

### ConfigPanel.jsx — Complete Rewrite
- **Task selector**: Dropdown → Segmented control pills
- **Attack section**: 4 sections → 3 tabs (White/Gray/Black-box)
- **Attack cards**: Grid → Accordion list with severity squares
- **Severity**: Click card → expand → 5 buttons → collapse → show 5 squares
- **Progress**: Inline animated progress bar (4px height, shimmer effect)
- **Removed**: Recipe Strategy Mode, Resource Cost Preview, Advanced Settings Drawer
- **Labels**: Shortened to 1-2 words (Task, Model, Checkpoint, Dataset, Attacks)

### ImageGrid.jsx — New Component
- 2x2 grid layout with `grid-template-columns: 1fr 1fr`
- Cell 1: Clean + Ground Truth BBox (green overlay)
- Cell 2: Attacked Input (raw, no boxes)
- Cell 3: Clean Prediction + MetricChips (AP50, mAP50-95, IoU, Boxes)
- Cell 4: Attacked Prediction + MetricChips (AP50, mAP50-95, D%, Lost)
- Stagger animation on cells (60ms delay each)
- Sample navigation bar at bottom

### Other Components Updated
- **FiveMetrics.jsx**: "Unavailable" → em-dash, shorter eyebrow text
- **EvidenceStage.jsx**: Eyebrow "Evidence stage" → "Evidence"
- **ComparisonView.js**: Consistent styling, shorter labels
- **ReportView.js**: Unified label styling with `config-panel__label`
- **DefencePanel.jsx**: Shorter labels, consistent spacing
- **ClosedLoopPanel.jsx**: Unified design language

## 3. Design Principles Applied

1. **Trung tinh tuyet doi** — Slate/Zinc palette, no purple, no neon glow
2. **Toi gian chu** — Labels 1-2 tu, no verbose descriptions
3. **Micro-animations** — fadeIn, scaleIn, stagger children, shimmer progress
4. **Dong bo ngon ngu thiet ke** — All components use same tokens, radii, transitions
5. **High-tech professional** — Inspired by Linear, Vercel, Stripe Dashboard
6. **Han che icon** — No emoji icons, minimal SVG icons in sidebar only

## 4. Verification

- Dev server compiles without errors
- Page loads correctly at localhost:3000
- 2-column layout renders properly
- Header bar with status badges
- Config panel with all sections (Task, Model, Checkpoint, Dataset, Attacks)
- 3-tab attack selector (White/Gray/Black-box)
- No neon/gradient visual elements
- Professional neutral dark theme
- API calls fail without backend (expected behavior)

## 5. Files Modified

| File | Action | Description |
|------|--------|-------------|
| `globals.css` | REWRITE | Complete design system overhaul |
| `layout.js` | MODIFY | Remove banner, simplify sidebar |
| `page.js` | REWRITE | Add header bar, use ImageGrid |
| `ConfigPanel.jsx` | REWRITE | Task pills, attack tabs, accordion |
| `ImageGrid.jsx` | NEW | 2x2 image grid with metric chips |
| `FiveMetrics.jsx` | MODIFY | Shorter labels |
| `EvidenceStage.jsx` | MODIFY | Simplified |
| `ComparisonView.js` | MODIFY | Consistent styling |
| `ReportView.js` | MODIFY | Unified labels |
| `DefencePanel.jsx` | MODIFY | Consistent design |
| `ClosedLoopPanel.jsx` | MODIFY | Unified design |
