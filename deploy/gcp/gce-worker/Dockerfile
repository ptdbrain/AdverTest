FROM python:3.11-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/
RUN apt-get update && apt-get install --no-install-recommends -y git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra models-gpu
COPY deploy/gcp/gce-worker/requirements-gcp.txt /tmp/requirements-gcp.txt
RUN uv pip install --python .venv/bin/python -r /tmp/requirements-gcp.txt
ENV PATH="/app/.venv/bin:$PATH"
COPY . .
CMD ["uvicorn", "src.cloud_run_gpu_worker:app", "--host", "0.0.0.0", "--port", "8080"]
