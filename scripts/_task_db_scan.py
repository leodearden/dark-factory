"""Shared plumbing for the READ-ONLY tasks.db sweep scripts.

Home of the tasks.db discovery ladder and the leak-scanner CLI skeleton that
were previously copied into ``scan_task_toolcall_leaks.py``,
``scan_provenance_note_log_leaks.py`` and ``audit_wiped_metadata_files.py``
(task 3336, closing task 3286's "~134 identical lines" finding). Every
function body here was lifted from those copies verbatim, so "pure extraction,
no behaviour change" stays checkable by diff.

This module is NOT a CLI in its own right — the leading underscore marks it as
importable-by-sibling-scripts only. It hosts two tiers:

* **Tier 1, discovery** (``_DEFAULT_PROJECT_ROOTS``, :func:`tasks_db_path`,
  :func:`resolve_project_roots`, :func:`discover_project_roots`,
  :func:`discover_db_paths`) — adopted by ALL THREE sweep scripts.
* **Tier 2, leak-scanner CLI plumbing** — adopted by the two leak scanners
  only. ``audit_wiped_metadata_files.py`` deliberately keeps its own
  ``format_report``/``format_json``/``_build_parser``/``main``: its fourth
  exit code (3 = roots resolved but every one failed to audit, kept distinct
  from 0 per ``docs/legibility/design-invariants.md``'s no-silent-fail-soft
  rule), its ``--min-fidelity`` filter and its object-shaped JSON are
  genuinely different behaviour, not duplication. Folding them in here would
  silently delete those semantics.

IMPORT-RESOLUTION CONTRACT — read before moving this file.
This module MUST stay a flat sibling at ``scripts/_task_db_scan.py``. The
sweep scripts' CLI tests drive ``main()`` by shelling out
(``subprocess.run([python, <script path>, ...])``), and
``scripts/tests/conftest.py``'s sys.path insertion does not reach those child
processes. They resolve ``import _task_db_scan`` solely because a
DIRECTLY-EXECUTED script places its own directory at ``sys.path[0]``.
Therefore: never relocate this module into a package, and never invoke any of
the three sweep scripts via ``python -m`` — either change breaks every CLI
test at once. (In-process pytest resolution is separately handled by
``scripts/tests/conftest.py``, which already puts ``scripts/`` on sys.path.)
"""
from __future__ import annotations

import os
from pathlib import Path

# Multi-project discovery fallback, mirroring
# scripts/migrate_metadata_modules_to_files.py's DEFAULT_ROOTS.
_DEFAULT_PROJECT_ROOTS = ("/home/leo/src/dark-factory",)


def tasks_db_path(project_root: str) -> Path:
    """``<root>/.taskmaster/tasks/tasks.db`` — the live task store."""
    return Path(project_root) / ".taskmaster" / "tasks" / "tasks.db"


def resolve_project_roots(
    project_roots: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Resolve the list of project roots, WITHOUT filtering for existence.

    Precedence (first supplied wins): *project_roots* >
    ``DASHBOARD_KNOWN_PROJECT_ROOTS`` (comma-separated, read from *env*,
    defaulting to the real ``os.environ`` when *env* is None) > the
    dark-factory default root. Env entries are stripped, empties are dropped,
    and order is preserved.

    Existence filtering is deliberately left to the callers, because they
    filter on different things: :func:`discover_db_paths` drops a missing
    ``tasks.db``, while :func:`discover_project_roots` drops the ROOT owning
    one.
    """
    if project_roots is not None:
        return list(project_roots)

    environ = env if env is not None else os.environ
    roots_env = (environ.get("DASHBOARD_KNOWN_PROJECT_ROOTS") or "").strip()
    if roots_env:
        return [r.strip() for r in roots_env.split(",") if r.strip()]
    return list(_DEFAULT_PROJECT_ROOTS)


def discover_project_roots(
    project_roots: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Resolve the list of project roots to audit.

    Precedence (first supplied wins): *project_roots* >
    ``DASHBOARD_KNOWN_PROJECT_ROOTS`` (comma-separated, read from *env*,
    defaulting to the real ``os.environ``) > the dark-factory default root.

    A root whose ``tasks.db`` does not exist is silently dropped — this never
    raises on a missing or not-yet-set-up project.
    """
    return [
        root
        for root in resolve_project_roots(project_roots=project_roots, env=env)
        if tasks_db_path(root).exists()
    ]


def discover_db_paths(
    explicit_dbs: list[str] | None = None,
    project_roots: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Resolve the list of tasks.db paths to scan.

    Precedence (first supplied wins): *explicit_dbs* > *project_roots* >
    ``DASHBOARD_KNOWN_PROJECT_ROOTS`` (read from *env*, defaulting to the
    real ``os.environ`` when *env* is None) > the dark-factory default root.

    A resolved db path that does not exist on disk is silently skipped —
    this never raises on a missing/not-yet-set-up project.
    """
    if explicit_dbs is not None:
        candidates = list(explicit_dbs)
    else:
        candidates = [
            str(tasks_db_path(root))
            for root in resolve_project_roots(project_roots=project_roots, env=env)
        ]

    return [path for path in candidates if os.path.exists(path)]
