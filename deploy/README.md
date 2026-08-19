# Platform deployment

`render.yaml` provisions a frontend web service, an API web service, a
Redis-compatible background worker queue, a persistent PostgreSQL database,
and a separate worker process. The API and worker reference the same private
Render Postgres/Key Value URLs.

## Free Render preview

Use `render-free.yaml` for a short-lived, no-card preview. It provisions only
Free-supported services: the frontend, one API instance, and a Free Postgres
database. The API runs platform jobs in a bounded in-process queue, so a job
only runs while the API is awake; it is not a production worker replacement.
It does not provision Redis or a background worker. Free Postgres expires
after 30 days, and Free web services can spin down when idle. Keep artifacts
in the configured object bucket, never on Render's local filesystem.

When creating the Blueprint, select `render-free.yaml` and supply:

```text
OBJECT_STORAGE_BUCKET=<your private GCS bucket>
OBJECT_STORAGE_ENDPOINT_URL=https://storage.googleapis.com
OBJECT_STORAGE_ACCESS_KEY_ID=<GCS HMAC access key>
OBJECT_STORAGE_SECRET_ACCESS_KEY=<GCS HMAC secret>
CORS_ORIGINS=https://<frontend>.onrender.com
NEXT_PUBLIC_API_URL=https://<api>.onrender.com
```

Set the two URL variables after Render assigns the public service URLs, then
redeploy the affected services. Checkpoint uploads remain disabled in this
preview because a separately isolated checkpoint sandbox is deliberately not
included.

Before the first production deploy, supply the S3-compatible bucket values
requested by the Blueprint. The bucket must allow private object access and
presigned PUT/GET URLs. Do not put these credentials in Git.

Set `NEXT_PUBLIC_API_URL` to the API's public HTTPS URL and add the frontend
HTTPS URL to the API's `CORS_ORIGINS` secret.

Production user-upload checkpoint loading is disabled unless
`CHECKPOINT_SANDBOX_URL` points to a separately isolated service. That service
must run untrusted loads in a network-disabled container with a read-only
filesystem, bounded CPU/memory, and a timeout; it returns class metadata and
smoke-test results only. This deliberately prevents the Render API/worker from
deserializing untrusted checkpoint bytes itself.

Run migrations as a release/pre-deploy command only:

```bash
PLATFORM_DATABASE_URL='postgresql+psycopg://...' alembic upgrade head
```

For local multi-service verification, start PostgreSQL, Redis, MinIO, API, and
the worker with `docker compose up --build`. MinIO's development credentials
are intentionally local-only and must never be reused in production.
