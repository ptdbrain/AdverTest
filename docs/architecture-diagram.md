# AdverTest architecture diagrams

These diagrams describe the repository's component boundaries and data flow for
a benchmark run. The platform is simulation-only: results are evidence for
human review, not a deployment-safety claim.

## Component architecture

```mermaid
flowchart LR
    U[Engineer or browser UI] --> API[FastAPI API\n/catalog /datasets /runs]
    CLI[CLI\npython -m src.cli] --> API
    API --> JOBS[Run and workflow jobs]
    JOBS --> ORCH[TestRunner / composition]
    ORCH --> DS[Dataset registry\nvalidation + anonymisation gate]
    ORCH --> ATK[Attack catalog\nwhite/gray/black/real-world]
    ORCH --> MOD[Model registry\nadapters + checkpoints]
    ORCH --> MET[Evaluation\nAP / degradation / evidence]
    ORCH <--> CACHE[(Prediction cache)]
    DS --> STORE[(Local datasets and uploads)]
    MOD --> WEIGHTS[(Checkpoint artifacts)]
    MET --> REPORT[RunReport / comparison / export]
    REPORT --> API
    API --> U
```

## Data flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Validator as Dataset/model validators
    participant Runner as TestRunner
    participant Attack as Attack plugin
    participant Adapter as Model adapter
    participant Eval as Metrics/evidence

    Client->>API: submit dataset, checkpoint, recipe, seed
    API->>Validator: validate task, schema, capability and provenance
    Validator-->>API: ready or actionable errors
    API->>Runner: queue immutable run config
    Runner->>Adapter: clean inference
    Adapter-->>Runner: clean predictions
    loop attack x severity
        Runner->>Attack: generate attacked sample
        Attack-->>Runner: attacked sample + provenance
        Runner->>Adapter: attacked inference
        Adapter-->>Runner: attacked predictions
        Runner->>Eval: compare against locked ground truth
    end
    Eval-->>Runner: metrics, per-object evidence and artifacts
    Runner-->>API: durable report and job status
    API-->>Client: progress, report, export links
```

## Boundary rules

- `Perception Mode`, model family, checkpoint and dataset contract remain separate.
- A base checkpoint is used by Attack; fine-tuned/repaired checkpoints belong to Defence.
- Missing ground truth permits quick inference only; it cannot produce benchmark AP/mAP.
- The backend preflight is the final compatibility gate even when the UI disables an option.

## Operational contracts

| Boundary | Input | Output | Failure is reported as |
|---|---|---|---|
| Ingestion -> validation | image/sensor files, task, annotations | validated dataset version | field-level validation issues |
| Catalog -> preflight | family, checkpoint, recipe, capabilities | compatible run plan | incompatible task/capability/annotation |
| Runner -> adapter | canonical sample and locked preprocessing | predictions and provenance | adapter/runtime error |
| Evaluation -> report | clean/attacked predictions and ground truth | metrics, evidence, artifacts | incomplete benchmark evidence |

All long-running boundaries expose a durable job state so the UI can reconnect,
show progress, cancel work when supported, and retain the reason for failure.

## Extension points

New attacks, model adapters, datasets, and metrics are registered behind their
existing contracts. The API and UI consume catalog metadata rather than
hard-coding implementation details, so compatibility remains a backend
decision even when a client is upgraded independently.
