"""A whole-directory snapshot for proving what a cockpit run wrote."""

from __future__ import annotations

from pathlib import Path


def snapshot_tree(base: Path) -> dict[str, tuple[int, bytes]]:
    """Map every file under *base* (relative path -> (mtime_ns, bytes)).

    Used to prove the cockpit's write discipline by diffing this snapshot
    across an app run: an absent path is not a KeyError, it's just missing
    from the mapping, so a deleted-or-never-written file shows up as a plain
    dict-equality mismatch rather than needing special-casing.
    """
    if not base.is_dir():
        return {}
    return {
        str(path.relative_to(base)): (path.stat().st_mtime_ns, path.read_bytes())
        for path in base.rglob('*')
        if path.is_file()
    }
