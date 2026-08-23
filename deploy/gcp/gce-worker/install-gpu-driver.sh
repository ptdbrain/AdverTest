#!/bin/bash
# COS removes NVIDIA modules on reboot; install the supported driver before
# the current worker container is recreated again.
set -euo pipefail
cos-extensions install gpu -- -version=R580
mount --bind /var/lib/nvidia /var/lib/nvidia
mount -o remount,exec /var/lib/nvidia
until docker info >/dev/null 2>&1; do sleep 2; done
access_token="$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
  | sed -n 's/.*"access_token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
printf '%s' "$access_token" | docker login -u oauth2accesstoken --password-stdin https://asia-southeast1-docker.pkg.dev
docker pull asia-southeast1-docker.pkg.dev/ai20k-build/advertest/gce-worker:gpu-dispatch-v7
docker rm -f advertest-gpu-worker advertest-gpu-worker-v4 || true
docker run -d --name advertest-gpu-worker --restart always --network host \
  -v /var/lib/nvidia/lib64:/usr/local/nvidia/lib64:ro \
  -v /var/lib/nvidia/bin:/usr/local/nvidia/bin:ro \
  -e APP_ENV=production \
  -e GOOGLE_CLOUD_PROJECT=ai20k-build \
  -e GCP_PUBSUB_SUBSCRIPTION=projects/ai20k-build/subscriptions/advertest-gce-worker \
  -e OBJECT_STORAGE_BACKEND=s3 \
  -e OBJECT_STORAGE_BUCKET=advertest-prod-artifacts \
  -e OBJECT_STORAGE_ENDPOINT_URL=https://storage.googleapis.com \
  -e OBJECT_STORAGE_REGION=auto \
  -e MODEL_DEVICE=cuda:0 \
  -e MODEL_HALF_PRECISION=true \
  -e GPU_IDLE_SHUTDOWN_SECONDS=900 \
  asia-southeast1-docker.pkg.dev/ai20k-build/advertest/gce-worker:gpu-dispatch-v7
