"""The docs-only classifier that gates the conditional project_root grant in
``orchestrator/src/orchestrator/agents/write_set.py::compute_write_set``.

It fails closed: a wrong "docs" verdict widens the OS sandbox into the main
checkout, so any input it cannot parse is "not docs".
"""
from __future__ import annotations

from pathlib import PurePosixPath

DOCS_SUFFIXES = frozenset({'.md', '.yaml', '.yml'})
DOCS_TOP_LEVEL_DIRS = frozenset({'docs'})


def is_docs_path(rel: PurePosixPath) -> bool:
    return rel.suffix in DOCS_SUFFIXES or (
        bool(rel.parts) and rel.parts[0] in DOCS_TOP_LEVEL_DIRS
    )


def _relative_path(entry: object) -> PurePosixPath | None:
    if not isinstance(entry, str) or not entry.strip():
        return None
    rel = PurePosixPath(entry)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        return None
    return rel


def docs_only_paths(files: object) -> tuple[PurePosixPath, ...]:
    """The normalized, deduped relative paths of *files* when every entry is
    a docs path; ``()`` for any other shape or content."""
    if not isinstance(files, (list, tuple)) or not files:
        return ()
    paths: dict[PurePosixPath, None] = {}
    for entry in files:
        rel = _relative_path(entry)
        if rel is None or not is_docs_path(rel):
            return ()
        paths[rel] = None
    return tuple(paths)
