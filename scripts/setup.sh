#!/bin/bash
# Setup script cho AI20K project

set -e

echo "=== AI20K Project Setup ==="

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# uv installs the Python version pinned in .python-version and owns .venv.
uv sync --locked

# Create .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env — please edit with your API keys"
fi

# Create data directories
mkdir -p data/chroma

echo "Setup complete! Run: uv run uvicorn src.main:app --reload"
