"""Tests for scripts/_task_db_scan.py — the shared plumbing extracted out of
the three READ-ONLY tasks.db sweep scripts (task 3336, following task 3286's
"~134 identical lines" finding).

Tier 1 (discovery: _DEFAULT_PROJECT_ROOTS / tasks_db_path /
resolve_project_roots / discover_project_roots / discover_db_paths) is adopted
by ALL THREE sweep scripts. Tier 2 (leak-scanner CLI plumbing) is adopted by
the two leak scanners only — audit_wiped_metadata_files.py keeps its own
CLI layer, whose exit-code-3 and --min-fidelity semantics are genuinely
different behaviour rather than duplication.

The precedence cases below are the consolidated single home for assertions
that previously lived duplicated across test_scan_task_toolcall_leaks.py,
test_scan_provenance_note_log_leaks.py and test_audit_wiped_metadata_files.py.
Those files keep their own copies as the untouched regression gate for the
extraction; this file pins the shared implementation directly.
"""
from __future__ import annotations

from pathlib import Path

from _task_db_scan import (
    _DEFAULT_PROJECT_ROOTS,
    discover_db_paths,
    discover_project_roots,
    resolve_project_roots,
    tasks_db_path,
)


def _touch_tasks_db(project_root: Path) -> Path:
    """Create an (empty-content) tasks.db under project_root/.taskmaster/tasks/."""
    db_dir = project_root / ".taskmaster" / "tasks"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_file = db_dir / "tasks.db"
    db_file.write_text("")
    return db_file


# ---------------------------------------------------------------------------
# tasks_db_path(project_root) -> Path
# ---------------------------------------------------------------------------

def test_tasks_db_path_maps_root_to_taskmaster_tasks_db(tmp_path):
    result = tasks_db_path(str(tmp_path))

    assert result == tmp_path / ".taskmaster" / "tasks" / "tasks.db"


def test_tasks_db_path_returns_a_path_not_a_str(tmp_path):
    """audit_wiped_metadata_files.py's public spelling returns Path, and its
    internal call sites (e.g. audit_project) depend on Path methods."""
    assert isinstance(tasks_db_path(str(tmp_path)), Path)


# ---------------------------------------------------------------------------
# resolve_project_roots(project_roots, env) -> list[str]
# ---------------------------------------------------------------------------

def test_resolve_project_roots_explicit_arg_wins_over_env():
    env = {"DASHBOARD_KNOWN_PROJECT_ROOTS": "/from/env"}

    assert resolve_project_roots(project_roots=["/explicit"], env=env) == ["/explicit"]


def test_resolve_project_roots_splits_env_on_comma_strips_and_drops_empties():
    # Comma-separated, whitespace padded, with an empty entry (",,") that must
    # be dropped rather than mapped to a bogus root. Order is preserved.
    env = {"DASHBOARD_KNOWN_PROJECT_ROOTS": " /a , /b ,, "}

    assert resolve_project_roots(env=env) == ["/a", "/b"]


def test_resolve_project_roots_whitespace_only_env_falls_back_to_default():
    assert resolve_project_roots(env={"DASHBOARD_KNOWN_PROJECT_ROOTS": "   "}) == list(
        _DEFAULT_PROJECT_ROOTS
    )


def test_resolve_project_roots_absent_env_key_falls_back_to_default():
    assert resolve_project_roots(env={}) == list(_DEFAULT_PROJECT_ROOTS)


def test_resolve_project_roots_does_not_filter_for_on_disk_existence(tmp_path):
    """Existence filtering belongs to discover_*, not to the resolve ladder —
    discover_db_paths and discover_project_roots filter on DIFFERENT things
    (a db path vs a root), so the shared ladder must hand back everything."""
    missing = tmp_path / "not-created"

    assert resolve_project_roots(project_roots=[str(missing)]) == [str(missing)]


def test_resolve_project_roots_default_is_the_dark_factory_root(monkeypatch):
    """The default-fallback contract, pinned pre-existence-filter so it holds
    on a machine where the real dark-factory checkout is absent."""
    monkeypatch.delenv("DASHBOARD_KNOWN_PROJECT_ROOTS", raising=False)

    roots = resolve_project_roots()

    assert roots == ["/home/leo/src/dark-factory"]
    assert str(tasks_db_path(roots[0])) == (
        "/home/leo/src/dark-factory/.taskmaster/tasks/tasks.db"
    )


# ---------------------------------------------------------------------------
# discover_project_roots(project_roots, env) -> list[str]  (ROOTS, not dbs)
# ---------------------------------------------------------------------------

def test_discover_project_roots_returns_roots_not_db_paths(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _touch_tasks_db(root)

    assert discover_project_roots(project_roots=[str(root)]) == [str(root)]


def test_discover_project_roots_drops_root_whose_tasks_db_is_absent(tmp_path):
    with_db = tmp_path / "with_db"
    without_db = tmp_path / "without_db"
    with_db.mkdir()
    without_db.mkdir()
    _touch_tasks_db(with_db)

    result = discover_project_roots(project_roots=[str(with_db), str(without_db)])

    assert result == [str(with_db)]


def test_discover_project_roots_reads_env_when_no_kwargs(tmp_path):
    root = tmp_path / "envproj"
    root.mkdir()
    _touch_tasks_db(root)

    env = {"DASHBOARD_KNOWN_PROJECT_ROOTS": f" {root} ,, "}

    assert discover_project_roots(env=env) == [str(root)]


# ---------------------------------------------------------------------------
# discover_db_paths(explicit_dbs, project_roots, env) -> list[str]  (db STRINGS)
# ---------------------------------------------------------------------------

def test_discover_db_paths_returns_db_path_strings(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    db_file = _touch_tasks_db(root)

    result = discover_db_paths(project_roots=[str(root)])

    assert result == [str(db_file)]
    assert all(isinstance(p, str) for p in result)


def test_discover_db_paths_explicit_dbs_win_over_project_roots(tmp_path):
    explicit = tmp_path / "explicit.db"
    explicit.write_text("")
    root = tmp_path / "proj"
    root.mkdir()
    _touch_tasks_db(root)

    result = discover_db_paths(explicit_dbs=[str(explicit)], project_roots=[str(root)])

    assert result == [str(explicit)]


def test_discover_db_paths_project_roots_win_over_env(tmp_path):
    arg_root = tmp_path / "arg"
    env_root = tmp_path / "env"
    arg_root.mkdir()
    env_root.mkdir()
    arg_db = _touch_tasks_db(arg_root)
    _touch_tasks_db(env_root)

    env = {"DASHBOARD_KNOWN_PROJECT_ROOTS": str(env_root)}

    assert discover_db_paths(project_roots=[str(arg_root)], env=env) == [str(arg_db)]


def test_discover_db_paths_env_used_when_no_explicit_kwargs(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    db_a = _touch_tasks_db(root_a)
    db_b = _touch_tasks_db(root_b)

    env = {"DASHBOARD_KNOWN_PROJECT_ROOTS": f" {root_a} , {root_b} ,, "}

    assert discover_db_paths(env=env) == [str(db_a), str(db_b)]


def test_discover_db_paths_drops_candidates_missing_on_disk(tmp_path):
    existing = tmp_path / "a.db"
    existing.write_text("")
    missing = tmp_path / "missing.db"

    assert discover_db_paths(explicit_dbs=[str(existing), str(missing)]) == [str(existing)]


def test_discover_db_paths_skips_project_root_without_tasks_db(tmp_path):
    root = tmp_path / "empty_proj"
    root.mkdir()

    assert discover_db_paths(project_roots=[str(root)]) == []
