"""Module path normalization for scheduler lock keys and task-corpus module matching.

The orchestrator's scheduler uses fixed-depth path prefixes as lock keys so that two
tasks touching `crates/reify-compiler/src/foo.rs` and `crates/reify-compiler/src/bar.rs`
serialize on the same lock. The task curator reuses the same normalization to find
tasks whose module footprint overlaps a candidate.

Directory/file path classifier (α/γ shared predicate)
------------------------------------------------------
``CODE_EXTENSIONS``, ``is_file_path``, ``directory_locks``, and
``strip_directory_locks`` live here so both the orchestrator scheduler (α
enforcement point) and the fused-memory lock-charter guard (γ enforcement
point) share ONE classifier — eliminating α/γ drift by construction.

``lock_charter_guard.py`` re-exports these names unchanged so its public API
and tests (``server/tools.py``, ``tests/test_lock_charter_guard.py``) remain
green without modification.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = [
    'CODE_EXTENSIONS',
    'directory_locks',
    'files_to_modules',
    'is_file_path',
    'modules_conflict',
    'normalize_lock',
    'strip_directory_locks',
]

# ---------------------------------------------------------------------------
# Canonical extension allowlist — single source of truth for α and γ.
# Copied verbatim from reify's scripts/lock-charter-guard.sh _EXTS (reify:4676).
# α/γ drift guard: fused-memory/tests/test_lock_charter_guard.py::test_extension_drift_guard
# ---------------------------------------------------------------------------

CODE_EXTENSIONS: frozenset[str] = frozenset({
    'c', 'cc', 'cjs', 'cpp', 'css', 'cts', 'cxx', 'gcode',
    'h', 'hh', 'hpp', 'html',
    'js', 'json', 'jsonc', 'jsx',
    'lock', 'md', 'mjs', 'mts', 'png', 'py',
    'ri', 'rs', 'scss', 'service', 'sh', 'step', 'stl', 'svg',
    'toml', 'ts', 'tsx', 'txt',
    'yaml', 'yml',
})


def is_file_path(path: str) -> bool:
    """Return True iff *path* is a file-level declaration (not a directory).

    Mirrors ``_is_file_path`` in reify's ``scripts/lock-charter-guard.sh``
    exactly — pure string, no filesystem access, no model call (C-P3).

    Algorithm (C-P1..C-P3):
    1. Strip ALL trailing ``/`` from *path*.
    2. Final segment = substring after last ``/``.
       Empty segment (path was all slashes) → directory → return False.
    3. If the segment contains no ``.`` → extension-less → return False.
    4. ext = substring after the last ``.`` in the segment.
       Return ``ext in CODE_EXTENSIONS`` (case-sensitive, matching α's
       ``[ "$ext" = "$e" ]`` against a lowercase allowlist).
    """
    # Strip all trailing slashes.
    p = path
    while p.endswith('/'):
        p = p[:-1]

    # Final segment (everything after the last '/').
    seg = p.rsplit('/', 1)[-1] if '/' in p else p

    # Empty segment → path was all slashes → directory.
    if not seg:
        return False

    # No dot in segment → extension-less → treat as directory (REJECT).
    if '.' not in seg:
        return False

    # Extension = substring after the last dot.
    ext = seg.rsplit('.', 1)[1]

    # Dotfiles: seg == '' after split when the dot is the first char.
    # e.g. '.gitignore' → seg='.gitignore', ext='gitignore' — not in allowlist → False.
    # 'f.PY' → ext='PY' — not in (lowercase) allowlist → False.  Correct.
    return ext in CODE_EXTENSIONS


def directory_locks(files: list[Any]) -> list[str]:
    """Return the ordered, de-duplicated list of directory-like entries in *files*.

    Non-string, empty, and whitespace-only tokens are silently skipped
    (mirrors ``_extract_metadata_files`` benign-absent convention).
    Order is preserved from the input; duplicates are collapsed to the first
    occurrence.
    """
    seen: set[str] = set()
    result: list[str] = []
    for f in files:
        if not isinstance(f, str) or not f.strip():
            continue
        if not is_file_path(f) and f not in seen:
            seen.add(f)
            result.append(f)
    return result


def strip_directory_locks(files: list[Any]) -> list[str]:
    """Return only the file-level entries from *files* (inverse of directory_locks).

    Non-string, empty, and whitespace-only tokens are silently skipped.
    Used as a pre-filter before lock derivation so directory charters do not
    produce subtree-wide prefix locks.
    """
    return [f for f in files if isinstance(f, str) and f.strip() and is_file_path(f)]


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
