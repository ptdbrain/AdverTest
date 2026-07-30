"""Command-line entry point — the fastest way to check your plugin works.

    python -m src.cli attacks                     # catalog + owners
    python -m src.cli models
    python -m src.cli datasets
    python -m src.cli estimate --attacks fgsm     # cost before running
    python -m src.cli run --attacks gaussian_noise,fgsm --severities 1,3,5
    python -m src.cli run --attacks fgsm --params '{"fgsm": {"epsilon_per_severity": [0.05]}}'
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from src.adapters import load_adapters
from src.attacks import load_attacks
from src.config import get_settings
from src.datasets import load_datasets
from src.evaluation.report import RunReport
from src.pipeline import RunConfig, TestRunner


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "attacks": _show_attacks,
        "models": _show_models,
        "datasets": _show_datasets,
        "estimate": _show_estimate,
        "run": _show_run,
    }
    return handlers[args.command](args)


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="advertest", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("attacks", "models", "datasets"):
        subparsers.add_parser(name, help=f"list registered {name}")
    for name in ("estimate", "run"):
        sub = subparsers.add_parser(name, help=f"{name} a test run")
        sub.add_argument("--model", default=settings.default_model)
        sub.add_argument("--dataset", default=settings.default_dataset)
        sub.add_argument("--attacks", default="", help="comma separated; empty = whole catalog")
        sub.add_argument("--severities", default=settings.default_severities)
        sub.add_argument("--limit", type=int, default=settings.default_sample_limit)
        sub.add_argument("--seed", type=int, default=settings.run_seed)
        sub.add_argument(
            "--params",
            default="{}",
            help='per-attack overrides as JSON, e.g. \'{"fgsm": {"epsilon_per_severity": [0.05]}}\'',
        )
        sub.add_argument("--json", action="store_true", help="dump the raw report as JSON")
    return parser


def _config_from(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        model=args.model,
        dataset=args.dataset,
        attacks=[name for name in args.attacks.split(",") if name],
        attack_params=json.loads(args.params),
        severities=[int(value) for value in args.severities.split(",") if value.strip()],
        limit=args.limit,
        seed=args.seed,
    )


def _show_attacks(args: argparse.Namespace) -> int:
    rows = [attack.describe() for attack in load_attacks().values()]
    _print_table(
        ["group", "name", "cost_class", "severity_levels", "needs_model", "owner"],
        rows,
    )
    print(f"\n{len(rows)} attacks registered")
    return 0


def _show_models(args: argparse.Namespace) -> int:
    rows = [adapter.describe() for adapter in load_adapters().values()]
    _print_table(["name", "task", "version", "supports_gradients", "owner"], rows)
    return 0


def _show_datasets(args: argparse.Namespace) -> int:
    rows = [dataset.describe() for dataset in load_datasets().values()]
    _print_table(["name", "modality", "anonymized", "owner"], rows)
    return 0


def _show_estimate(args: argparse.Namespace) -> int:
    estimate = TestRunner().estimate(_config_from(args))
    print(json.dumps(estimate.as_dict(), indent=2))
    return 0


def _show_run(args: argparse.Namespace) -> int:
    report = TestRunner().run(_config_from(args))
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
        return 0
    _print_report(report)
    return 0


def _print_report(report: RunReport) -> None:
    """Human-readable summary: clean AP, then degradation per cell."""
    print(f"run {report.run_id}  model={report.model_version}  dataset={report.dataset}")
    print(f"samples={report.n_samples}  AP_clean={report.ap_clean:.3f}  seconds={report.seconds:.2f}")
    rows = [
        {**cell.as_dict(), "D%": f"{report.degradation(cell) * 100:.1f}"}
        for cell in sorted(report.cells, key=lambda cell: (cell.group, cell.attack, cell.severity))
    ]
    _print_table(["group", "attack", "severity", "ap", "D%", "cache_hits"], rows)
    for skipped in report.skipped:
        print(f"skipped {skipped.attack}: {skipped.reason}")


def _print_table(columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Minimal fixed-width table — no extra dependency."""
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows)) if rows else len(column)
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


if __name__ == "__main__":
    raise SystemExit(main())
