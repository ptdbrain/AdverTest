from pathlib import Path


def test_day01_is_outside_repository() -> None:
    """The deployment checkout name is not guaranteed to be ``P-195``."""
    repository_root = Path(__file__).resolve().parents[1]
    assert not any(path.name.startswith("DAY01") for path in repository_root.iterdir())
