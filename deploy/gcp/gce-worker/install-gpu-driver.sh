#!/bin/bash
# COS removes NVIDIA modules on reboot; install the supported driver before
# the worker container is started again.
set -euo pipefail
cos-extensions install gpu -- -version=R580
mount --bind /var/lib/nvidia /var/lib/nvidia
mount -o remount,exec /var/lib/nvidia
until docker info >/dev/null 2>&1; do sleep 2; done
docker start advertest-gpu-worker-v4 || true
