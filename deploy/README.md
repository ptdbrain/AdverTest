# Platform deployment

`render.yaml` provisions a frontend web service, an API web service, a
Redis-compatible background worker queue, a persistent PostgreSQL database,
and a separate worker process. The API and worker reference the same private
Render Postgres/Key Value URLs.

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
