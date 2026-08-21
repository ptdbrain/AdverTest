FROM python:3.11-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/
RUN apt-get update && apt-get install --no-install-recommends -y git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra models-cpu
ENV PATH="/app/.venv/bin:$PATH"
COPY . .
CMD ["uvicorn", "src.gcp_sandbox:app", "--host", "0.0.0.0", "--port", "8080"]
