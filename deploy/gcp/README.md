# GCP execution plane

This directory keeps costly execution outside Render:

`Render API -> Cloud Run dispatcher -> Pub/Sub -> GCE worker -> GCS/Postgres`.

## Resources to create

1. Pub/Sub topic `advertest-jobs` and a pull subscription, such as
   `advertest-gce-worker`.
2. A Cloud Run service built from `deploy/gcp/dispatcher`, with a service
   account allowed to publish to that topic.
3. A GCE GPU VM built from `deploy/gcp/gce-worker/Dockerfile`, with a service
   account allowed to consume from the subscription and read/write the artifact
   bucket.
4. A separate Cloud Run service built from `deploy/gcp/sandbox/Dockerfile`.
   It is the only endpoint configured as `CHECKPOINT_SANDBOX_URL`.

## Render API variables

```text
QUEUE_BACKEND=http_dispatcher
RUN_EXECUTION_BACKEND=platform
EXTERNAL_QUEUE_DISPATCH_URL=https://<dispatcher>/dispatch
EXTERNAL_QUEUE_DISPATCH_TOKEN=<long-random-shared-token>
EXTERNAL_GPU_WORKER_URL=https://<gce-worker-identity>
CHECKPOINT_SANDBOX_URL=https://<sandbox>/inspect
CHECKPOINT_SANDBOX_TOKEN=<different-long-random-token>
```

## GCE worker variables

```text
GCP_PUBSUB_SUBSCRIPTION=projects/<project>/subscriptions/advertest-gce-worker
MODEL_DEVICE=cuda:0
MODEL_HALF_PRECISION=true
PLATFORM_DATABASE_URL=<Render external PostgreSQL URL>
OBJECT_STORAGE_BACKEND=s3
OBJECT_STORAGE_BUCKET=advertest-prod-artifacts
OBJECT_STORAGE_ENDPOINT_URL=https://storage.googleapis.com
OBJECT_STORAGE_ACCESS_KEY_ID=<GCS HMAC access key>
OBJECT_STORAGE_SECRET_ACCESS_KEY=<GCS HMAC secret>
```

The dispatcher and sandbox must require authentication and have only the
minimum ingress permitted. Do not expose the GCE worker as a public API.
