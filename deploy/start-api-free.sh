#!/bin/sh
set -eu

# Free Render instances do not support preDeployCommand. A single API instance
# can safely migrate before it begins accepting preview traffic.
alembic upgrade head
exec uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-10000}"
