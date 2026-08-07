from pathlib import Path


def test_day01_is_outside_p195_repository() -> None:
    worktree = Path(__file__).resolve().parents[1]
    p195_root = next(
        (path for path in (worktree, *worktree.parents) if path.name == "P-195"),
        None,
    )

    assert p195_root is not None
    assert not any(path.name.startswith("DAY01") for path in p195_root.iterdir())
