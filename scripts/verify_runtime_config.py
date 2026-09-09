"""CLI validation for portable local configuration and Render preview topology."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.deployment.runtime_config import validate_portable_dotenv, validate_preview_blueprint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-example", type=Path, default=Path(".env.example"))
    parser.add_argument("--blueprint", type=Path, default=Path("render.preview.yaml"))
    args = parser.parse_args()
    errors = validate_portable_dotenv(args.env_example) + validate_preview_blueprint(args.blueprint)
    if errors:
        print("Runtime configuration is invalid:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Runtime configuration contract is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
