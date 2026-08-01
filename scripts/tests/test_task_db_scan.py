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

import argparse
import json
import sqlite3
from pathlib import Path
from typing import NamedTuple

from _task_db_scan import (
    _DEFAULT_PROJECT_ROOTS,
    NO_DB_RESOLVED_MESSAGE,
    add_db_discovery_args,
    discover_db_paths,
    discover_project_roots,
    format_json,
    group_matches_by_db,
    resolve_project_roots,
    run_scan_cli,
    sweep_databases,
    tasks_db_path,
    truncate,
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


# ---------------------------------------------------------------------------
# Tier 2 — leak-scanner CLI plumbing (the two leak scanners only).
#
# audit_wiped_metadata_files.py deliberately does NOT adopt this tier; see the
# module docstring of scripts/_task_db_scan.py for why.
# ---------------------------------------------------------------------------

class _Match(NamedTuple):
    """Stand-in for LeakMatch / NoteLeakMatch — any NamedTuple with db_path."""

    db_path: str
    task_id: int
    text: str


def _match(db_path="/db/a.db", task_id=1, text="frag"):
    return _Match(db_path=db_path, task_id=task_id, text=text)


# --- format_json(matches) -> str -------------------------------------------

def test_format_json_round_trips_namedtuples_via_asdict():
    matches = [_match(task_id=7, text="x"), _match(db_path="/db/b.db", task_id=8)]

    payload = json.loads(format_json(matches))

    assert payload == [
        {"db_path": "/db/a.db", "task_id": 7, "text": "x"},
        {"db_path": "/db/b.db", "task_id": 8, "text": "frag"},
    ]


def test_format_json_empty_list_is_an_empty_json_array():
    assert json.loads(format_json([])) == []


# --- truncate(text, max_len) -> str ----------------------------------------

def test_truncate_appends_ellipsis_only_when_over_the_limit():
    assert truncate("abcdef", 3) == "abc..."


def test_truncate_is_a_noop_at_exactly_the_limit():
    assert truncate("abc", 3) == "abc"


def test_truncate_is_a_noop_below_the_limit():
    assert truncate("ab", 3) == "ab"


# --- group_matches_by_db(matches) -> dict[str, list] ------------------------

def test_group_matches_by_db_keys_iterate_in_sorted_order():
    matches = [_match(db_path="/db/c.db"), _match(db_path="/db/a.db"), _match(db_path="/db/b.db")]

    assert list(group_matches_by_db(matches)) == ["/db/a.db", "/db/b.db", "/db/c.db"]


def test_group_matches_by_db_preserves_input_order_within_a_group():
    matches = [
        _match(db_path="/db/a.db", task_id=3),
        _match(db_path="/db/b.db", task_id=99),
        _match(db_path="/db/a.db", task_id=1),
    ]

    grouped = group_matches_by_db(matches)

    assert [m.task_id for m in grouped["/db/a.db"]] == [3, 1]
    assert [m.task_id for m in grouped["/db/b.db"]] == [99]


def test_group_matches_by_db_empty_input_yields_empty_dict():
    assert group_matches_by_db([]) == {}


# --- NO_DB_RESOLVED_MESSAGE ------------------------------------------------

def test_no_db_resolved_message_is_the_exact_string_both_scanners_emit():
    """Pinned verbatim: both leak scanners' CLI tests assert on this text, and
    it is the documented exit-2 signal."""
    assert NO_DB_RESOLVED_MESSAGE == (
        "no tasks.db resolvable (checked --db / --project-root / "
        "DASHBOARD_KNOWN_PROJECT_ROOTS / the dark-factory default)"
    )


# --- add_db_discovery_args(parser, json_help=...) --------------------------

def _json_action_help(parser):
    return next(a for a in parser._actions if a.dest == "json").help


def test_add_db_discovery_args_parses_repeatable_db_and_project_root():
    parser = argparse.ArgumentParser()
    add_db_discovery_args(parser, json_help="whatever")

    args = parser.parse_args(
        ["--db", "one.db", "--db", "two.db", "--project-root", "/r", "--json"]
    )

    assert args.dbs == ["one.db", "two.db"]
    assert args.project_roots == ["/r"]
    assert args.json is True


def test_add_db_discovery_args_defaults_are_none_and_false():
    parser = argparse.ArgumentParser()
    add_db_discovery_args(parser, json_help="whatever")

    args = parser.parse_args([])

    assert args.dbs is None
    assert args.project_roots is None
    assert args.json is False


def test_add_db_discovery_args_threads_the_callers_json_help_through():
    """The two scanners name different payloads ('fragments' vs 'leak lines'),
    so --json's help is a parameter rather than shared text."""
    parser = argparse.ArgumentParser()
    add_db_discovery_args(parser, json_help="Emit a JSON array of widgets.")

    assert _json_action_help(parser) == "Emit a JSON array of widgets."


# --- sweep_databases(db_paths, scan_fn) -> (matches, unreadable) -----------

def test_sweep_databases_concatenates_results_in_db_order():
    def scan(db_path):
        return [_match(db_path=db_path)]

    matches, unreadable = sweep_databases(["/db/b.db", "/db/a.db"], scan)

    assert [m.db_path for m in matches] == ["/db/b.db", "/db/a.db"]
    assert unreadable == []


def test_sweep_databases_skips_unreadable_db_and_warns_on_stderr(capsys):
    def scan(db_path):
        if db_path == "/db/bad.db":
            raise sqlite3.Error("file is not a database")
        return [_match(db_path=db_path)]

    matches, unreadable = sweep_databases(["/db/good.db", "/db/bad.db"], scan)

    assert [m.db_path for m in matches] == ["/db/good.db"]
    assert unreadable == ["/db/bad.db"]

    err = capsys.readouterr().err
    assert (
        "warning: skipping unreadable database /db/bad.db: file is not a database" in err
    )
    assert "results below are incomplete" in err


def test_sweep_databases_omits_the_aggregate_warning_when_all_readable(capsys):
    matches, unreadable = sweep_databases(["/db/a.db"], lambda db_path: [])

    assert (matches, unreadable) == ([], [])
    assert "incomplete" not in capsys.readouterr().err


def test_sweep_databases_does_not_swallow_non_sqlite_errors():
    def scan(db_path):
        raise ValueError("a real bug, not an unreadable db")

    try:
        sweep_databases(["/db/a.db"], scan)
    except ValueError:
        return
    raise AssertionError("a non-sqlite3 error must propagate, not be skipped")


# --- run_scan_cli(argv, parser=, scan_fn=, render=) -> int ------------------

def _cli(argv, scan_fn, render=lambda matches, args: f"rendered:{len(matches)}"):
    parser = argparse.ArgumentParser()
    add_db_discovery_args(parser, json_help="json")
    return run_scan_cli(argv, parser=parser, scan_fn=scan_fn, render=render)


def test_run_scan_cli_exits_2_when_nothing_resolves(tmp_path, capsys):
    missing = tmp_path / "nope.db"

    assert _cli(["--db", str(missing)], lambda db_path: []) == 2
    assert NO_DB_RESOLVED_MESSAGE in capsys.readouterr().err


def test_run_scan_cli_exits_0_when_the_scan_finds_nothing(tmp_path, capsys):
    db = tmp_path / "a.db"
    db.write_text("")

    assert _cli(["--db", str(db)], lambda db_path: []) == 0
    assert capsys.readouterr().out.strip() == "rendered:0"


def test_run_scan_cli_exits_1_when_the_scan_finds_a_match(tmp_path, capsys):
    db = tmp_path / "a.db"
    db.write_text("")

    assert _cli(["--db", str(db)], lambda db_path: [_match(db_path=db_path)]) == 1
    assert capsys.readouterr().out.strip() == "rendered:1"


def test_run_scan_cli_prints_exactly_the_render_output(tmp_path, capsys):
    db = tmp_path / "a.db"
    db.write_text("")

    def render(matches, args):
        assert args.json is True  # the parsed Namespace reaches the renderer
        return "custom body\nsecond line"

    assert _cli(["--db", str(db), "--json"], lambda db_path: [], render=render) == 0
    assert capsys.readouterr().out == "custom body\nsecond line\n"
