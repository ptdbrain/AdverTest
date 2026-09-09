# GCP execution plane

This directory keeps costly execution outside Render:

`Render API -> Cloud Run dispatcher -> Pub/Sub -> Cloud Run L4 worker -> GCS`.

The Render API is the **only** process that connects to Render PostgreSQL. The
GPU worker claims jobs and reports progress/results over authenticated callback
endpoints; it has neither `PLATFORM_DATABASE_URL` nor a database IP allowlist.

## Resources to create

1. Pub/Sub topic `advertest-jobs` and a push subscription targeting the GPU
   worker endpoint `/pubsub` with OIDC authentication.
2. A Cloud Run service built from `deploy/gcp/dispatcher`, with a service
   account allowed to publish to that topic.
3. A Cloud Run GPU service (NVIDIA L4, `min-instances=0`, `max-instances=1`)
   built from `deploy/gcp/gce-worker/Dockerfile`, with a service account allowed
   to read/write the artifact bucket.
4. A separate Cloud Run service built from `deploy/gcp/sandbox/Dockerfile`.
   It is the only endpoint configured as `CHECKPOINT_SANDBOX_URL`.

## Render API variables

```text
QUEUE_BACKEND=http_dispatcher
RUN_EXECUTION_BACKEND=platform
EXTERNAL_QUEUE_DISPATCH_URL=https://<dispatcher>/dispatch
EXTERNAL_QUEUE_DISPATCH_TOKEN=<long-random-shared-token>
WORKER_CALLBACK_TOKEN=<different-long-random-token>
CHECKPOINT_SANDBOX_URL=https://<sandbox>/inspect
CHECKPOINT_SANDBOX_TOKEN=<different-long-random-token>
```

## Cloud Run L4 worker variables

```text
MODEL_DEVICE=cuda:0
MODEL_HALF_PRECISION=true
WORKER_CALLBACK_API_URL=https://<render-api>
WORKER_CALLBACK_TOKEN=<same callback token as Render API>
OBJECT_STORAGE_BACKEND=s3
OBJECT_STORAGE_BUCKET=advertest-prod-artifacts
OBJECT_STORAGE_ENDPOINT_URL=https://storage.googleapis.com
OBJECT_STORAGE_ACCESS_KEY_ID=<GCS HMAC access key>
OBJECT_STORAGE_SECRET_ACCESS_KEY=<GCS HMAC secret>
```

The dispatcher and sandbox must require authentication and have only the
minimum ingress permitted. Do not expose the GCE worker as a public API.

The Cloud Run service account also needs `roles/storage.objectAdmin` on the
bucket. The worker uses this IAM identity to write evidence, so HMAC storage
credentials are only needed if it must download bootstrap catalog assets.

## On-demand GPU lifecycle

Cloud Run scales from zero when Pub/Sub pushes the first message, and can return
to zero after the request finishes. Do not configure VM start/stop permissions,
Cloud NAT, a static outbound IP, or a PostgreSQL inbound-IP rule for this
worker. Set request timeout to accommodate the longest permitted benchmark.
