.PHONY: run test lint format typecheck check catalog demo kitti-data benchmark-kitti clean

run:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

check: lint format test

# Registered plugins (attacks / adapters / datasets) with their owners.
catalog:
	python -m src.cli attacks
	python -m src.cli models
	python -m src.cli datasets

# Smallest end-to-end run: reference dataset + reference detector.
# fgsm gets a larger epsilon here on purpose: blob_detector is a threshold model,
# so the plan's {1..16}/255 ladder (right for real CNNs) barely moves it.
demo:
	python -m src.cli run --limit 4 --severities 1,3,5 \
	  --params '{"fgsm": {"epsilon_per_severity": [0.02, 0.04, 0.08, 0.16, 0.32]}}'

# KITTI 2D object split for the group C benchmark (~12 GB, or ~800 MB with a subset).
kitti-data:
	bash scripts/fetch_kitti.sh --subset 500

# Group C x YOLO11 x KITTI robustness benchmark -> eval/results/.
benchmark-kitti:
	python scripts/benchmark_kitti_yolo11.py --limit 500

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
