# Person D Platform Execution Decisions

## Status and Scope

These decisions are binding for execution of the Person D platform plan. The
implementation base is the local `origin/main` commit
`8767ea853872371ccc49ed01a2e8809b38a8030f`. A fetch was attempted but GitHub
was unavailable, so this local reference is the approved base for this work.

All work occurs only in the linked P-195 worktree on
`feat/person-d-platform`. Do not access or modify DAY01.

## Ownership Boundaries

Person D owns generic contracts and services. Person B and Person C own the
concrete adapter, evaluator, and trainer implementations for their respective
model domains. Person A owns routes, persistence, and queue integration.

## Mask and Dataset Identity

`MaskWireV1` is the JSON representation for a 2D boolean mask. It uses
row-major run-length encoding, including the mask shape and encoding version.
Canonical persisted mask arrays remain `.npy` files, and runtime masks remain
NumPy arrays.

Source dataset identity includes a logical source ID, source hash,
ground-truth hash, loader version, and schema version. Absolute paths are never
part of the identity, version ID, manifest hash, or provenance hash.

## Execution Gates

Source-level leakage validation is a prerequisite for generator work.
Generated-dataset lineage validation is added after recipe generation.
`HardExampleBank` is completed before `TrainingDatasetBuilder`.

Only Task 17 registers the public `recipe-validate` and `recipe-sample` CLI
commands. Earlier tasks may provide internal validation and sampling services,
but they must not register those public commands.

CPU fake tests establish Person D platform completion. When required real
artifacts from A, B, or C are unavailable, record the affected handoff gate as
`WAITING_FOR_OWNER`. Those missing artifacts block team integration completion
only; they do not weaken the CPU contract or platform-completion evidence.

## Task Ordering

| Order | Tasks or gate | Required outcome |
| --- | --- | --- |
| 1 | Tasks 1-3 | Isolated harness and shared generic contracts are ready. |
| 2 | Task 4 | Dataset identity and deterministic locked splits include the approved non-path inputs. |
| 3 | Task 5 | Source-level leakage validation passes before any generator work. |
| 4 | Tasks 6-10 | Catalog, recipes, transformations, composition, and recipe generation are complete. |
| 5 | Post-Task-10 gate | Generated-dataset lineage validation runs after recipe generation. |
| 6 | Task 12, then Task 11 | HardExampleBank is available before TrainingDatasetBuilder consumes it. |
| 7 | Tasks 13-16 | Generic benchmark, comparison, and training orchestration contracts are ready for supplied adapters. |
| 8 | Task 17 | Public service facade and the `recipe-validate` and `recipe-sample` CLI commands are registered. |
| 9 | Task 18 | CPU fake acceptance establishes platform completion; missing A/B/C artifacts remain `WAITING_FOR_OWNER` for team integration. |
