"""Module path normalization for scheduler lock keys and task-corpus module matching.

The orchestrator's scheduler uses fixed-depth path prefixes as lock keys so that two
tasks touching `crates/reify-compiler/src/foo.rs` and `crates/reify-compiler/src/bar.rs`
serialize on the same lock. The task curator reuses the same normalization to find
tasks whose module footprint overlaps a candidate.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    'normalize_lock',
    'files_to_modules',
    'modules_conflict',
]


def modules_conflict(a: str, b: str) -> bool:
    """Two module locks conflict if one is a prefix of the other (or exact match).

    This is the canonical hierarchical-conflict rule used by the orchestrator's
    ``ModuleLockTable`` (which delegates to it) and by the dashboard's holder
    lookup. Keeping it here means the prefix semantics live in exactly one place.
    """
    return a == b or a.startswith(b + '/') or b.startswith(a + '/')


def normalize_lock(module: str, depth: int = 2) -> str:
    """Normalize a module path to a fixed depth for lock granularity.

    e.g. normalize_lock('crates/reify-types/src/persistent.rs') -> 'crates/reify-types'
    """
    if not module:
        return module
    parts = module.strip('/').split('/')
    return '/'.join(parts[:depth])


def files_to_modules(files: Iterable[str], depth: int) -> list[str]:
    """Derive unique module locks from a list of file paths.

    Each file path is normalized to ``depth`` components, then deduplicated and
    returned sorted so callers get a stable ordering.
    """
    modules: set[str] = set()
    for f in files:
        normalized = normalize_lock(f, depth)
        if normalized:
            modules.add(normalized)
    return sorted(modules)
