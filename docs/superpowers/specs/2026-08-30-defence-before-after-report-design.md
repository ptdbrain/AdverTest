# Defense Before/After Report Design

## Purpose

Turn the Defense flow into one evidence-led, paired comparison of a baseline
run and a defended candidate run. It must show what changed, what evidence is
missing, and whether the result is eligible for a deployment decision.

## Boundaries

The canonical server report is the only source for recovery, effectiveness,
deployment readiness, failure transitions, and exported values. Frontend code
only formats values returned by that report. Visual inference is evidence for
inspection, never a substitute for benchmark metrics.

Every comparison, artifact, export, event stream, and deep link is scoped to
one immutable `project_id`; access requires active membership. A comparison
cannot be loaded, exported, or inferred from an ID belonging to another
project.

## Scientific Contract

A direct defense claim requires both completed reports to have a verified,
identical protocol signature: task, dataset version, split manifest/sample
set, recipe and exact attack/severity keys, seed, thresholds, preprocessing,
class mapping, and metric versions. Missing or mismatched evidence produces an
explicit non-measured state with null metrics, never zeroes.

Metric catalogs are selected by task and dataset protocol:

- Detection 2D uses its declared 2D AP/mAP metrics.
- Segmentation uses mIoU and class metrics only when supplied by the evaluator.
- KITTI detection 3D uses the declared KITTI 3D/BEV AP protocol.
- nuScenes detection 3D uses NDS only after the complete nuScenes protocol is
  supplied. Before that, it remains `NOT_ELIGIBLE` and is not labelled as
  KITTI AP.

Recovery remains direction-aware and unbounded. Effectiveness and deployment
readiness are separate decisions. Simulation-only, insufficient sample size,
missing paired confidence interval, or missing ground truth can never produce
a production deployment pass.

## Demo and External Training

`scripts/train_defence.py` does not create a checkpoint unless a real,
registered trainer does so. Its explicit `--demo` mode writes a JSON artifact
under `demo_display_estimates`, marked `simulation_only: true`; those values
are ineligible for canonical comparison, promotion, or conclusion exports.

The Defense page may offer a collapsed external-training command, labelled as
guidance only. It selects a runtime-supported device rather than assuming
`cuda:0`, and never represents command generation as training completion.

## Report and Export UX

The first viewport shows pairedness/evidence state, baseline-under-attack,
defended-under-attack, robust gain, recovery, clean trade-off, and failure
transitions. A separate deployment block lists passed and failed checks.
Missing data is `—`; color supplements explicit text and icons.

Metric matrices and charts use returned values without defaults. Paired visual
evidence keeps attacked input, baseline prediction, candidate prediction, and
ground truth as distinct artifacts. Missing predictions show an unavailable
state rather than a substituted image.

JSON, CSV, HTML, PDF, and ZIP exports use the same canonical report and its
eligibility gate. When evidence is insufficient they may export diagnostics and
provenance, but cannot contain a benchmark conclusion or deployment promotion.

## Compatibility and Verification

Existing British-spelling API routes remain compatibility aliases for one
release, but delegate to the project-scoped canonical report. Legacy v1
comparisons render as legacy with no new calculated verdict.

Acceptance requires task-aware unit tests, API authorization and project
isolation tests, paired-report/export integration tests, frontend orchestration
and rendering tests, production build, and no mocked report metrics in runtime
UI.
