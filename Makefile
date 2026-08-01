.PHONY: run test lint format typecheck check catalog demo clean

run:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

check: lint format test

# Registered plugins (attacks / adapters / datasets) with their owners.
catalog:
	uv run python -m src.cli attacks
	uv run python -m src.cli models
	uv run python -m src.cli datasets

# Smallest end-to-end run: reference dataset + reference detector.
# fgsm gets a larger epsilon here on purpose: blob_detector is a threshold model,
# so the plan's {1..16}/255 ladder (right for real CNNs) barely moves it.
demo:
	uv run python -m src.cli run --limit 4 --severities 1,3,5 \
	  --params '{"fgsm": {"epsilon_per_severity": [0.02, 0.04, 0.08, 0.16, 0.32]}}'

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
