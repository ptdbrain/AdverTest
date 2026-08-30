# Visual Results Layout Design

**Date:** 2026-08-27  
**Surface:** `/experiments/[id]/results`  
**Register:** Product UI

## Goal

Refit the visual-results page to the supplied AdversAI Lab reference layout while preserving its current demo data and interactions.

## Design

- Keep the existing light, evidence-led product vocabulary: cool tinted neutrals, blue as the active/action color, green for intact/success state, and red for attacked/degraded state.
- Retain the application shell, route, page title, history link, reload action, zoom selector, image download, benchmark link, and export action.
- Compose the page in four visual bands:
  1. compact page header with title, subtitle, and two utility actions;
  2. primary comparison area with two equal Before/After image panels and a right experiment-information rail;
  3. five equal deep-dive panels for difference, attention, perturbation, segmentation, and zoom evidence;
  4. quick-observation summary paired with the export actions.
- Use a 16:9 media viewport for the paired street images so the top comparison row remains close to the supplied desktop reference instead of expanding into a tall image wall.
- Keep metric content legible at desktop density, with shared metric-card treatment and explicit labels for clean versus attacked state.
- Replace decorative emoji markers in the experiment rail with consistent Lucide icons and preserve text labels so state is not conveyed by color alone.
- At narrower widths, stack the experiment rail below the paired panels, collapse the paired images to one column, and let the deep-dive row wrap without horizontal overflow.

## Boundaries

- Modify only the visual-results page and its focused regression test.
- Do not alter API contracts, sample assets, navigation behavior, or unrelated pending work in the worktree.
- Do not introduce new dependencies or generated raster assets.

## Verification

- Add a focused test that asserts the page exposes the reference section labels and paired image structure.
- Run the focused Vitest test, the full frontend test suite, and `npm run build` from `frontend`.
- Perform a browser screenshot check at desktop and narrow viewport sizes, confirming no horizontal overflow and that the paired images retain the intended aspect ratio.
