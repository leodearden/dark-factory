"""Lock-charter directory-vs-file predicate guard (γ).

Backstop at the fused-memory task-creation path (``submit_task`` /
``commit_planning``) that REJECTS a directory string in ``metadata.files``
with a clear error, while ACCEPTING a file-level set or ``[]`` (the
deliberate defer-to-architect value).

This catches every creation path the /prd decompose guard (β) misses —
most importantly human-decompose (``submit_task(planning_mode=True)``  →
``commit_planning``), the dominant origin of over-wide directory locks
migrated verbatim from ``metadata.modules`` by commit fabfa367f5.

## Predicate contract (C-P1..C-P4, PRD §4.1)

The α/γ shared predicate is a *pure string* classification — no filesystem
stat, no model call (C-P3).  C-P3 determinism means the Python
implementation here and reify's Bash ``_is_file_path`` in
``scripts/lock-charter-guard.sh`` (reify task 4676, status=done) are
equivalent *by construction*: same allowlist, same strip/segment/ext
algorithm.

The drift-guard test in ``tests/test_lock_charter_guard.py`` pins
``sorted(CODE_EXTENSIONS)`` to the shared α/γ canonical vector printed by
``lock-charter-guard.sh --list-extensions``, so divergence is caught at CI
time.

Extension allowlist rationale (reify PRD §11 Q2):
- PRD-explicit: rs ri toml cpp c h hpp md json yaml yml lock py sh ts tsx js txt step stl
- Corpus-evidenced: css mjs html jsonc gcode service
- Common source siblings: cc cxx hh mts cts cjs jsx scss svg png
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Canonical extension allowlist — single source of truth for γ.
# Copied verbatim from reify's scripts/lock-charter-guard.sh _EXTS (reify:4676).
# α/γ drift guard: tests/test_lock_charter_guard.py::test_extension_drift_guard
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


# ---------------------------------------------------------------------------
# List-gate helpers
# ---------------------------------------------------------------------------


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


def extract_files(metadata: str | dict[str, Any] | None) -> list[str]:
    """Parse *metadata* (dict, JSON string, or None) and return ``metadata.files``.

    Mirrors the benign-absent logic of ``_extract_metadata_files`` in
    ``task_interceptor.py``: malformed or absent metadata is treated as
    no-files (return []) rather than raising.

    Rules:
    - ``None`` → []
    - ``dict`` → use directly
    - ``str`` → ``json.loads``; on failure or non-dict result → []
    - If the resolved dict has no ``files`` key → []
    - If ``files`` is not a list → []
    - Return only the string entries of the list.
    """
    if metadata is None:
        return []

    parsed: dict[str, Any] | None = None
    if isinstance(metadata, dict):
        parsed = metadata
    elif isinstance(metadata, str):
        try:
            loaded = json.loads(metadata)
        except (ValueError, TypeError):
            return []
        parsed = loaded if isinstance(loaded, dict) else None

    if parsed is None:
        return []

    files = parsed.get('files')
    if not isinstance(files, list):
        return []

    return [f for f in files if isinstance(f, str)]


def lock_charter_error(
    directories: list[str],
    task_id: str | None = None,
) -> dict[str, Any]:
    """Return a structured LockCharterViolation error dict.

    Shape mirrors ``_done_gate_error`` / ``PathGuardVerdict.to_error_dict``
    so MCP callers handle it uniformly.

    Args:
        directories: The offending directory-like path strings.
        task_id: Optional task id to name in the error message (used by
            ``commit_planning`` to name the first offending task).
    """
    task_clause = f' (task {task_id})' if task_id else ''
    dir_list = ', '.join(repr(d) for d in directories)
    return {
        'error': (
            f'metadata.files contains directory declarations{task_clause}: '
            f'{dir_list}. '
            f'Declare individual file paths (e.g. "src/foo.py") or use '
            f'files=[] to defer scope to the architect.'
        ),
        'error_type': 'LockCharterViolation',
        'directory_paths': directories,
        'hint': (
            'Replace each directory with the specific file paths it contains, '
            'or use files=[] (the defer-to-architect value) to leave scope '
            'unspecified. Directory strings in metadata.files are rejected '
            'because they create over-wide lock charters that block unrelated '
            'work in the same directory.'
        ),
    }
