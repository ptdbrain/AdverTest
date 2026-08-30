# Defense Session and Attack Run Targeting Design

## Goal

The Defense page must not ask users to configure a defense without first identifying the measured failure it targets. Users select one experiment session and then one attack run from that session. That selection becomes the immutable source for the base model, dataset, attack recipe, severity, metrics, and locked benchmark protocol.

## User Flow

1. Load experiment sessions from the existing sessions API.
2. Require the user to select a session.
3. Require the user to select one attack run within that session.
4. Show a read-only Defense Target summary containing the session, model, dataset, attack or attack chain, severity, clean and attacked metrics, degradation, backend run ID, seed, and protocol hash when available.
5. Enable defense strategy and training controls only after the target is selected.
6. Filter the base checkpoint and defense candidates to the selected task and model lineage.
7. Evaluate the selected defense checkpoint against the selected run's `backend_run_id`, preserving the original locked protocol.

## Validity Rules

- Changing the session clears the selected attack run, checkpoint, generated command context, and evaluation result.
- A session with no runs shows an explicit empty state.
- A run without a real `backend_run_id` may be inspected but cannot start a paired defense evaluation.
- A defense candidate with a different task or model family is not selectable.
- No mock comparison is shown. Before a real comparison exists, the page shows a no-data state.
- The evaluation button remains disabled until session, run, and compatible checkpoint evidence are present.

## Components and Data Flow

- `DefensePage` owns session loading and the selected session/run IDs.
- A small target selector/summary component renders the two dependent selects and read-only provenance.
- Existing `getSessions`, checkpoint, and defense-run API helpers remain the data sources; no parallel mock data source is introduced.
- Training parameters remain editable, but model, dataset, and recipe arguments are derived from the selected evidence rather than independent defaults.
- API, empty, and validation failures are shown inline with a retry action where appropriate.

## Accessibility and Responsive Behavior

- Both selects have persistent visible labels and accessible names.
- Locked values are represented by text as well as status badges.
- The selector uses one column on narrow screens and two columns when space permits.
- Disabled actions include an adjacent textual reason instead of relying only on disabled styling.

## Test Contract

- Initial state requires a session and has defense controls disabled.
- Selecting a session lists only its runs and does not infer a run automatically.
- Selecting a run displays its model, dataset, attack, severity, and measured degradation.
- Switching sessions clears the run and downstream state.
- A run without `backend_run_id` blocks evaluation with an evidence-specific reason.
- A valid run invokes defense evaluation with its `backend_run_id`, never the display run ID.
- No `DEFENSE_COMPARISON` mock values render before a real comparison response.

## Scope

This change targets the existing `/defense` workflow and focused frontend tests. It does not redesign the training backend or claim that an unavailable GPU/checkpoint has passed real validation.
