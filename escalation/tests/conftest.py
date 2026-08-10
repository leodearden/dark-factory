"""pytest configuration — ensure local src takes precedence over installed package.

Without this conftest, ``import escalation`` resolves to the editable
install in the prod venv (which points at /home/leo/src/dark-factory/),
not the worktree under test.  Injecting the worktree's ``src`` at the
front of ``sys.path`` mirrors the orchestrator/tests pattern.

Also injects ``orchestrator/src`` and ``shared/src`` so that
TestMergeRequestDedup and TestGetMergeQueue can do inline
``from orchestrator.*`` imports.  Those tests live in the escalation test
suite but exercise code paths that require the neighbouring orchestrator
package (InFlightMergeRegistry, MergeOutcome, MergeResult).

Because the escalation project-scoped venv intentionally excludes
``aiosqlite`` (pulled in by ``shared/__init__`` via
``shared.async_sqlite_base``), we pre-populate ``sys.modules`` with a
minimal stub for the async-sqlite layer before adding the paths.  Only
the stubs that satisfy the orchestrator import chain are provided;
everything else is loaded from the actual source files.
"""

from __future__ import annotations

import collections
import sys
import types
from pathlib import Path

_SRC = Path(__file__).parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Orchestrator/shared path injection for TestMergeRequestDedup /
# TestGetMergeQueue tests.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent
_SHARED_SRC = _REPO_ROOT / 'shared' / 'src'
_ORCH_SRC = _REPO_ROOT / 'orchestrator' / 'src'

if _SHARED_SRC.is_dir() and _ORCH_SRC.is_dir():
    # Stub ``shared`` package before adding shared/src to sys.path so that
    # Python never executes ``shared/__init__.py`` (which imports aiosqlite,
    # not installed in the escalation project-scoped venv).
    if 'shared' not in sys.modules:
        _shared_stub = types.ModuleType('shared')
        _shared_stub.__path__ = [str(_SHARED_SRC / 'shared')]
        _shared_stub.__package__ = 'shared'
        sys.modules['shared'] = _shared_stub

    # ``shared.async_sqlite_base`` imports aiosqlite.
    # ``shared.sqlite_sync_base`` re-exports CheckpointResult from it.
    # Provide minimal stubs so orchestrator modules that import these can load.
    if 'shared.async_sqlite_base' not in sys.modules:
        _async_base_stub = types.ModuleType('shared.async_sqlite_base')
        _CheckpointResult = collections.namedtuple('CheckpointResult', ['log', 'checkpointed'])
        # Populate via __dict__.update: pyright rejects attribute assignment on a
        # bare ModuleType instance (reportAttributeAccessIssue), but a module
        # namespace is a plain dict[str, Any], so this is runtime-equivalent and
        # type-clean.
        _async_base_stub.__dict__.update(
            CheckpointResult=_CheckpointResult,
            AsyncSqliteBase=type('AsyncSqliteBase', (), {}),
            apply_full_durability_pragmas=lambda *a, **kw: None,
            apply_wal_pragmas=lambda *a, **kw: None,
            connect_daemon=lambda *a, **kw: None,
        )
        sys.modules['shared.async_sqlite_base'] = _async_base_stub

    for _p in (str(_SHARED_SRC), str(_ORCH_SRC)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

# Suite-wide git isolation (task 3355, incident esc-3072-3).  The verify lane
# runs `cd escalation && uv run pytest tests/`, which makes rootdir the
# SUBPROJECT — the repo-root conftest.py is never loaded, so each test-root
# conftest wires the defence itself.  APPEND the repo root, never insert(0, ...):
# at sys.path[0] it would make the subproject directories resolve as namespace
# packages pointing at the project folder instead of src/<pkg>/, beating the
# inserts above.  Placed after the stubbing block to keep that contiguous;
# df_pytest_isolation is stdlib + pytest only, so it imports cleanly in this
# venv, which deliberately excludes aiosqlite.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from df_pytest_isolation import (  # noqa: E402
    _df_deploy_clocks_unwritten,  # noqa: F401  — the binding IS the wiring
    _df_git_ceiling_at_basetemp,  # noqa: F401  — the binding IS the wiring
    reject_unsafe_basetemp,
)


def pytest_configure(config):
    """Refuse a --basetemp aimed inside a live task worktree (esc-3072-3)."""
    reject_unsafe_basetemp(config)
