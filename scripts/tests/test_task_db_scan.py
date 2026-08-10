"""Tests for scripts/_task_db_scan.py — the shared plumbing extracted out of
the four READ-ONLY tasks.db sweep scripts (tasks 3336 and 3616, following task
3286's "~134 identical lines" finding).

Tier 1 (discovery: _DEFAULT_PROJECT_ROOTS / tasks_db_path /
resolve_project_roots / discover_project_roots / discover_db_paths) is adopted
by ALL FOUR sweep scripts. Tier 2 (leak-scanner CLI plumbing) is adopted by the
two LEAK SCANNERS only; Tier 3 (audit-script CLI plumbing: run_audit_cli /
sweep_project_roots / the AUDIT_EXIT_* codes / format_kv_line /
format_coverage_block) by the two AUDIT scripts only. The split is
load-bearing rather than cosmetic: Tier 2 sweeps db PATHS and accumulates
MATCHES, Tier 3 sweeps project ROOTS and collects exactly one audit per root,
which is why their exit-3 gates are spelled differently (see
test_run_audit_cli_exits_0_when_some_roots_unreadable_and_rest_clean and its
Tier-2 sibling).

Both audit scripts keep their own format_report/format_json/_build_parser,
whose --min-fidelity filter, --project-id handling, object-shaped JSON, epilog
wording and COVERAGE caveat/labels are genuinely different behaviour rather
than duplication. Exit 3 is NOT one of those differences: task 3474 gave
run_scan_cli the same "nothing was scanned, so this is not a clean run"
semantics (see test_run_scan_cli_exits_3_when_every_db_is_unreadable below), so
that exit code is shared by every tier rather than a differentiator.

The precedence cases below are the consolidated single home for assertions
that previously lived duplicated across test_scan_task_toolcall_leaks.py,
test_scan_provenance_note_log_leaks.py and test_audit_wiped_metadata_files.py.
Those files keep their own copies as the untouched regression gate for the
extraction; this file pins the shared implementation directly.

COVERAGE — this file IS executed by the verify chains that matter for it
(task 3384 closed the gap a former caveat here recorded; re-verified
2026-08-06). Both of these run the directory this file lives in:

    scripts/orchestrator.yaml (test_command)
        uv run --project shared pytest tests/scripts/ scripts/tests/ \\
            --tb=short -q --timeout=300
    dark-factory-orchestrator.yaml (test_command, tail segment)
        uv run --project shared pytest tests/scripts/ scripts/tests/ \\
            --timeout=300

STILL THE TRAP, so it is kept: ``scripts/tests/`` (this directory) and
``tests/scripts/`` (a suite at the repo root) are near-homographs but DIFFERENT
directories. A command naming only one of them covers only one of them — which
is exactly how the original gap arose. ``tests/scripts/orchestrator.yaml`` is
correctly scoped to ``tests/scripts/`` alone and does not collect this file;
the two chains above are what gate it.

To run just this suite:

    uv run --project shared pytest scripts/tests/test_task_db_scan.py -q
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import NamedTuple

from _task_db_scan import (
    _DEFAULT_PROJECT_ROOTS,
    AUDIT_EXIT_FINDINGS,
    AUDIT_EXIT_NO_ROOT,
    AUDIT_EXIT_NOTHING_AUDITED,
    AUDIT_EXIT_OK,
    NO_DB_RESOLVED_MESSAGE,
    NO_PROJECT_ROOT_RESOLVED_MESSAGE,
    add_db_discovery_args,
    discover_db_paths,
    discover_project_roots,
    format_coverage_block,
    format_json,
    format_kv_line,
    group_matches_by_db,
    resolve_project_roots,
    run_audit_cli,
    run_scan_cli,
    sweep_databases,
    sweep_project_roots,
    tasks_db_path,
    truncate,
)

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

def test_discover_project_roots_returns_roots_not_db_paths(tmp_path, project_root_with_tasks_db):
    root = tmp_path / "proj"
    root.mkdir()
    project_root_with_tasks_db(root)

    assert discover_project_roots(project_roots=[str(root)]) == [str(root)]


def test_discover_project_roots_drops_root_whose_tasks_db_is_absent(
    tmp_path, project_root_with_tasks_db
):
    with_db = tmp_path / "with_db"
    without_db = tmp_path / "without_db"
    with_db.mkdir()
    without_db.mkdir()
    project_root_with_tasks_db(with_db)

    result = discover_project_roots(project_roots=[str(with_db), str(without_db)])

    assert result == [str(with_db)]


def test_discover_project_roots_reads_env_when_no_kwargs(tmp_path, project_root_with_tasks_db):
    root = tmp_path / "envproj"
    root.mkdir()
    project_root_with_tasks_db(root)

    env = {"DASHBOARD_KNOWN_PROJECT_ROOTS": f" {root} ,, "}

    assert discover_project_roots(env=env) == [str(root)]


# ---------------------------------------------------------------------------
# discover_db_paths(explicit_dbs, project_roots, env) -> list[str]  (db STRINGS)
# ---------------------------------------------------------------------------

def test_discover_db_paths_returns_db_path_strings(tmp_path, project_root_with_tasks_db):
    root = tmp_path / "proj"
    root.mkdir()
    db_file = project_root_with_tasks_db(root)

    result = discover_db_paths(project_roots=[str(root)])

    assert result == [str(db_file)]
    assert all(isinstance(p, str) for p in result)


def test_discover_db_paths_explicit_dbs_win_over_project_roots(
    tmp_path, project_root_with_tasks_db
):
    explicit = tmp_path / "explicit.db"
    explicit.write_text("")
    root = tmp_path / "proj"
    root.mkdir()
    project_root_with_tasks_db(root)

    result = discover_db_paths(explicit_dbs=[str(explicit)], project_roots=[str(root)])

    assert result == [str(explicit)]


def test_discover_db_paths_project_roots_win_over_env(tmp_path, project_root_with_tasks_db):
    arg_root = tmp_path / "arg"
    env_root = tmp_path / "env"
    arg_root.mkdir()
    env_root.mkdir()
    arg_db = project_root_with_tasks_db(arg_root)
    project_root_with_tasks_db(env_root)

    env = {"DASHBOARD_KNOWN_PROJECT_ROOTS": str(env_root)}

    assert discover_db_paths(project_roots=[str(arg_root)], env=env) == [str(arg_db)]


def test_discover_db_paths_env_used_when_no_explicit_kwargs(tmp_path, project_root_with_tasks_db):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    db_a = project_root_with_tasks_db(root_a)
    db_b = project_root_with_tasks_db(root_b)

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


def test_run_scan_cli_exits_3_when_every_db_is_unreadable(tmp_path, capsys):
    """Every resolved database failing to open is NOT a clean sweep.

    Nothing was scanned at all, so exit 3 keeps that outcome distinct from 0
    per docs/legibility/design-invariants.md's no-silent-fail-soft rule —
    a cron/CI consumer reading only the exit code must never take a total
    failure for a clean run. Mirrors audit_wiped_metadata_files.py's exit-3
    semantics for the same condition.

    Contrast the partial-failure sibling below, which stays 0: there at least
    one database was genuinely scanned and found clean.
    """
    db = tmp_path / "a.db"
    db.write_text("")

    def scan(db_path):
        raise sqlite3.Error("file is not a database")

    assert _cli(["--db", str(db)], scan) == 3

    captured = capsys.readouterr()
    # The report is still printed, so stdout alone STILL reads clean — it is
    # the exit code plus stderr that now unambiguously disagree with it.
    assert captured.out.strip() == "rendered:0"
    assert "results below are incomplete" in captured.err
    assert "NOTHING was scanned" in captured.err


def test_run_scan_cli_exits_0_when_some_dbs_unreadable_and_rest_clean(tmp_path, capsys):
    """Partial failure keeps the sweep going and reports on what WAS readable.

    Unlike the all-unreadable case above (which exits 3), exit 0 here is
    correct-as-designed: at least one database was genuinely scanned and found
    clean. This is also the standing guard against gating exit 3 on "some db
    was unreadable AND no matches were found" — that condition would wrongly
    fire here.
    """
    good = tmp_path / "good.db"
    bad = tmp_path / "bad.db"
    good.write_text("")
    bad.write_text("")

    def scan(db_path):
        if db_path == str(bad):
            raise sqlite3.Error("file is not a database")
        return []

    assert _cli(["--db", str(good), "--db", str(bad)], scan) == 0
    assert "results below are incomplete" in capsys.readouterr().err


def test_run_scan_cli_still_exits_1_when_a_readable_db_matches_despite_failures(tmp_path):
    """A match found in a readable database is not masked by a sibling failure."""
    good = tmp_path / "good.db"
    bad = tmp_path / "bad.db"
    good.write_text("")
    bad.write_text("")

    def scan(db_path):
        if db_path == str(bad):
            raise sqlite3.Error("file is not a database")
        return [_match(db_path=db_path)]

    assert _cli(["--db", str(good), "--db", str(bad)], scan) == 1


def test_run_scan_cli_prints_exactly_the_render_output(tmp_path, capsys):
    db = tmp_path / "a.db"
    db.write_text("")

    def render(matches, args):
        assert args.json is True  # the parsed Namespace reaches the renderer
        return "custom body\nsecond line"

    assert _cli(["--db", str(db), "--json"], lambda db_path: [], render=render) == 0
    assert capsys.readouterr().out == "custom body\nsecond line\n"


# ---------------------------------------------------------------------------
# Tier 3 — audit-script CLI plumbing (the two audit scripts).
#
# Adopted by audit_wiped_metadata_files.py and audit_combine_gate_marker_loss.py
# (task 3616). Unlike Tier 2, this tier sweeps PROJECT ROOTS rather than db
# paths, and its per-root callback returns exactly ONE audit object per root —
# that difference is what lets the exit-3 gate be spelled `unreadable and not
# audits` here while Tier 2 must count (see run_scan_cli's comment).
# ---------------------------------------------------------------------------

# --- AUDIT_EXIT_* constants ------------------------------------------------

def test_audit_exit_constants_carry_the_documented_values():
    assert AUDIT_EXIT_OK == 0
    assert AUDIT_EXIT_FINDINGS == 1
    assert AUDIT_EXIT_NO_ROOT == 2
    assert AUDIT_EXIT_NOTHING_AUDITED == 3


def test_audit_exit_constants_are_pairwise_distinct():
    """3 must never collapse into 0.

    Each constant denotes EXACTLY ONE outcome. Folding "audited nothing at all"
    into "audited everything, found nothing" is the silent fail-soft that
    docs/legibility/design-invariants.md's no-silent-fail-soft rule forbids, and
    a careless renumbering is exactly how it would come back.
    """
    codes = [
        AUDIT_EXIT_OK,
        AUDIT_EXIT_FINDINGS,
        AUDIT_EXIT_NO_ROOT,
        AUDIT_EXIT_NOTHING_AUDITED,
    ]

    assert len(set(codes)) == len(codes)


# --- sweep_project_roots(roots, audit_fn) -> (audits, unreadable) ----------

def test_sweep_project_roots_returns_one_audit_per_root_in_root_order():
    audits, unreadable = sweep_project_roots(
        ["/proj/b", "/proj/a"], lambda root: f"audit:{root}"
    )

    assert audits == ["audit:/proj/b", "audit:/proj/a"]
    assert unreadable == []


def test_sweep_project_roots_skips_unreadable_root_and_warns_on_stderr(capsys):
    def audit(root):
        if root == "/proj/bad":
            raise sqlite3.Error("file is not a database")
        return f"audit:{root}"

    audits, unreadable = sweep_project_roots(
        ["/proj/bad", "/proj/good"], audit
    )

    # The BAD root is first on purpose: the sweep must CONTINUE past a failure
    # so every later root is still audited.
    assert audits == ["audit:/proj/good"]
    assert unreadable == ["/proj/bad"]

    err = capsys.readouterr().err
    assert (
        "warning: skipping unreadable project /proj/bad: file is not a database" in err
    )


def test_sweep_project_roots_warns_once_in_aggregate_that_results_are_incomplete(capsys):
    def audit(root):
        raise sqlite3.Error("file is not a database")

    sweep_project_roots(["/proj/a", "/proj/b"], audit)

    err = capsys.readouterr().err
    aggregate = (
        "warning: 2 project(s) skipped due to read errors (see warnings above); "
        "results below are incomplete"
    )
    assert aggregate in err
    assert err.count("results below are incomplete") == 1


def test_sweep_project_roots_omits_the_aggregate_warning_when_all_readable(capsys):
    audits, unreadable = sweep_project_roots(["/proj/a"], lambda root: "audit")

    assert (audits, unreadable) == (["audit"], [])
    assert "incomplete" not in capsys.readouterr().err


def test_sweep_project_roots_does_not_swallow_non_sqlite_errors():
    """Mirrors test_sweep_databases_does_not_swallow_non_sqlite_errors.

    A real bug in *audit_fn* must surface, never be silently downgraded to
    "unreadable project" — that would turn a crash into a quietly incomplete
    report.
    """
    def audit(root):
        raise ValueError("a real bug, not an unreadable project")

    try:
        sweep_project_roots(["/proj/a"], audit)
    except ValueError:
        return
    raise AssertionError("a non-sqlite3 error must propagate, not be skipped")


# --- run_audit_cli(argv, parser=, audit_fn=, render=, is_dirty=, on_roots=) -

def _audit_cli(
    argv,
    audit_fn,
    render=lambda audits, args: f"rendered:{len(audits)}",
    is_dirty=lambda audits: False,
    on_roots=None,
):
    """In-process harness modelled on ``_cli`` above.

    Subprocess execution is left to the two audit scripts' own suites, where it
    is load-bearing (it is what proves the flat-sibling ``import _task_db_scan``
    resolves for a DIRECTLY-EXECUTED script).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", dest="project_roots", action="append")
    parser.add_argument("--json", action="store_true")
    return run_audit_cli(
        argv,
        parser=parser,
        audit_fn=audit_fn,
        render=render,
        is_dirty=is_dirty,
        on_roots=on_roots,
    )


def _unreadable(root, args):
    raise sqlite3.Error("file is not a database")


def test_run_audit_cli_exits_2_when_no_root_resolves(tmp_path, capsys):
    """The report must NOT be rendered on the exit-2 path.

    Nothing was audited and no audit list exists, so emitting a well-formed
    empty payload here would hand a stdout-only consumer a clean-looking result
    for a run that never started.
    """
    missing = tmp_path / "nope"

    assert _audit_cli(["--project-root", str(missing)], lambda root, args: "a") == 2

    captured = capsys.readouterr()
    assert NO_PROJECT_ROOT_RESOLVED_MESSAGE in captured.err
    assert captured.out == ""


def test_run_audit_cli_exits_0_when_nothing_is_dirty(project_root_with_tasks_db, tmp_path, capsys):
    root = tmp_path / "proj"
    project_root_with_tasks_db(root)

    exit_code = _audit_cli(
        ["--project-root", str(root)],
        lambda root_, args: "audit",
        is_dirty=lambda audits: False,
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "rendered:1"


def test_run_audit_cli_exits_1_when_is_dirty_fires(project_root_with_tasks_db, tmp_path):
    root = tmp_path / "proj"
    project_root_with_tasks_db(root)

    exit_code = _audit_cli(
        ["--project-root", str(root)],
        lambda root_, args: "audit",
        is_dirty=lambda audits: True,
    )

    assert exit_code == 1


def test_run_audit_cli_exits_3_when_every_root_is_unreadable(
    project_root_with_tasks_db, tmp_path, capsys
):
    """Every resolved project failing to audit is NOT a clean sweep.

    The report is still rendered before the exit-3 branch, so stdout STILL
    reads clean — that is precisely why the exit code is the only honest
    signal, and why 3 is kept distinct from 0 per
    docs/legibility/design-invariants.md's no-silent-fail-soft rule.
    """
    root = tmp_path / "proj"
    project_root_with_tasks_db(root)

    assert _audit_cli(["--project-root", str(root)], _unreadable) == 3

    captured = capsys.readouterr()
    assert captured.out.strip() == "rendered:0"
    assert "results below are incomplete" in captured.err
    assert "NOTHING was audited (this is not a clean result)" in captured.err


def test_run_audit_cli_exits_0_when_some_roots_unreadable_and_rest_clean(
    project_root_with_tasks_db, tmp_path, capsys
):
    """THE PARTIAL-FAILURE GUARD — the standing regression check on the gate.

    One root raising beside one clean root is exit 0, not 3: a project was
    genuinely audited and found clean. Mirrors
    test_run_scan_cli_exits_0_when_some_dbs_unreadable_and_rest_clean; gating
    exit 3 on "some root failed and nothing was found" would wrongly fire here.
    """
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    project_root_with_tasks_db(good)
    project_root_with_tasks_db(bad)

    def audit(root, args):
        if root == str(bad):
            raise sqlite3.Error("file is not a database")
        return "audit"

    exit_code = _audit_cli(
        ["--project-root", str(good), "--project-root", str(bad)], audit
    )

    assert exit_code == 0
    assert "results below are incomplete" in capsys.readouterr().err


def test_run_audit_cli_still_exits_1_when_a_readable_root_is_dirty_despite_failures(
    project_root_with_tasks_db, tmp_path
):
    """A finding in a readable project is not masked by a sibling failure."""
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    project_root_with_tasks_db(good)
    project_root_with_tasks_db(bad)

    def audit(root, args):
        if root == str(bad):
            raise sqlite3.Error("file is not a database")
        return "dirty"

    exit_code = _audit_cli(
        ["--project-root", str(good), "--project-root", str(bad)],
        audit,
        is_dirty=lambda audits: "dirty" in audits,
    )

    assert exit_code == 1


def test_run_audit_cli_passes_the_parsed_namespace_to_audit_fn_and_render(
    project_root_with_tasks_db, tmp_path, capsys
):
    """Both callbacks see the parsed flags.

    argv is parsed INSIDE the helper, so a per-root flag (wiped's
    --min-fidelity, combine's --project-id) can only reach the callbacks this
    way — the calling script has nothing to close over.
    """
    root = tmp_path / "proj"
    project_root_with_tasks_db(root)
    seen_roots = []

    def audit(root_, args):
        assert args.json is True
        seen_roots.append(root_)
        return "audit"

    def render(audits, args):
        assert args.json is True
        return f"json={args.json} audits={audits}"

    assert _audit_cli(["--project-root", str(root), "--json"], audit, render=render) == 0
    assert seen_roots == [str(root)]
    assert capsys.readouterr().out.strip() == "json=True audits=['audit']"


def test_run_audit_cli_gives_is_dirty_only_the_successfully_audited_objects(
    project_root_with_tasks_db, tmp_path
):
    """An unreadable root contributes NO placeholder to the is_dirty input.

    This is the caller-visible half of sweep_project_roots' one-audit-per-root
    contract: a predicate iterating the list must never meet a None standing in
    for a project that failed.
    """
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    project_root_with_tasks_db(good)
    project_root_with_tasks_db(bad)
    seen = []

    def audit(root, args):
        if root == str(bad):
            raise sqlite3.Error("file is not a database")
        return "audit:good"

    def is_dirty(audits):
        seen.append(list(audits))
        return False

    _audit_cli(["--project-root", str(good), "--project-root", str(bad)], audit, is_dirty=is_dirty)

    assert seen == [["audit:good"]]


def test_run_audit_cli_calls_on_roots_before_the_empty_roots_return(tmp_path, capsys):
    """The hook fires even when NO root resolved, and its output comes first.

    This is the verbatim position audit_combine_gate_marker_loss.py's multi-root
    --project-id warning has always occupied: after root resolution, BEFORE the
    empty-roots early return. Preserving it keeps the extraction a pure diff and
    lets a future hook observe the empty case.
    """
    missing = tmp_path / "nope"
    seen = []

    def on_roots(roots, args):
        seen.append((list(roots), args.json))
        print("hook ran", file=sys.stderr)

    exit_code = _audit_cli(
        ["--project-root", str(missing)], lambda root, args: "a", on_roots=on_roots
    )

    assert exit_code == 2
    assert seen == [([], False)]

    err = capsys.readouterr().err
    assert err.index("hook ran") < err.index(NO_PROJECT_ROOT_RESOLVED_MESSAGE)


def test_run_audit_cli_calls_on_roots_once_with_the_resolved_roots(
    project_root_with_tasks_db, tmp_path
):
    root = tmp_path / "proj"
    project_root_with_tasks_db(root)
    calls = []

    def on_roots(roots, args):
        calls.append(list(roots))

    _audit_cli(["--project-root", str(root)], lambda root_, args: "a", on_roots=on_roots)

    assert calls == [[str(root)]]


def test_run_audit_cli_works_without_an_on_roots_hook(project_root_with_tasks_db, tmp_path):
    """on_roots defaults to None — only one of the two adopters supplies it."""
    root = tmp_path / "proj"
    project_root_with_tasks_db(root)
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", dest="project_roots", action="append")
    parser.add_argument("--json", action="store_true")

    exit_code = run_audit_cli(
        ["--project-root", str(root)],
        parser=parser,
        audit_fn=lambda root_, args: "a",
        render=lambda audits, args: "body",
        is_dirty=lambda audits: False,
    )

    assert exit_code == 0


# --- format_kv_line(pairs) -> str ------------------------------------------

def test_format_kv_line_uses_a_two_space_indent_and_single_space_separator():
    assert format_kv_line([("task_id", 7), ("tag", "master")]) == "  task_id=7 tag=master"


def test_format_kv_line_leaves_no_trailing_space():
    line = format_kv_line([("task_id", 7)])

    assert line == "  task_id=7"
    assert line == line.rstrip()


def test_format_kv_line_reproduces_the_wiped_audit_candidate_line_byte_for_byte():
    """audit_wiped_metadata_files.py's _format_candidate_line, exactly.

    Asserted against the literal string rather than a regex: the acceptance bar
    for the extraction is byte-for-byte identical output, and a regex would
    happily accept a changed separator or indent.
    """
    line = format_kv_line([
        ("task_id", 3264),
        ("tag", "master"),
        ("status", "done"),
        ("signature", "null-merge-sha"),
        ("source", "plan.files"),
        ("fidelity", "file-level"),
        ("plan_files", 3),
    ])

    assert line == (
        "  task_id=3264 tag=master status=done signature=null-merge-sha "
        "source=plan.files fidelity=file-level plan_files=3"
    )


def test_format_kv_line_reproduces_the_combine_audit_finding_line_byte_for_byte():
    """audit_combine_gate_marker_loss.py's _format_finding_line, exactly."""
    line = format_kv_line([
        ("task_id", 3088),
        ("tag", "master"),
        ("status", "pending"),
        ("key", "delivered_checks"),
        ("severity", "gate_removing"),
        ("source", "ticket"),
    ])

    assert line == (
        "  task_id=3088 tag=master status=pending key=delivered_checks "
        "severity=gate_removing source=ticket"
    )


def test_format_kv_line_renders_an_empty_pair_list_as_the_bare_indent():
    assert format_kv_line([]) == "  "


def test_format_kv_line_renders_every_value_with_str():
    """An int, a str and a len() result must format identically to today's
    f-strings, which interpolate all three through str()."""
    assert format_kv_line([
        ("count", 3),
        ("name", "master"),
        ("plan_files", len(["a.py", "b.py"])),
    ]) == "  count=3 name=master plan_files=2"


# --- format_coverage_block(caveat, rows, details) -> list[str] --------------

def test_format_coverage_block_places_every_value_at_column_40():
    """THE GUTTER CONTRACT, pinned from three different label widths.

    These are the real padded literals from both audit scripts: labels of 30,
    26 and 31 characters, all of which put their value at column 40 today. A
    drifting gutter is the exact failure mode this helper exists to prevent, so
    the rendered lines are asserted whole.
    """
    lines = format_coverage_block(
        "  CAVEAT:",
        [
            ("with a file-level plan signal:", 3),
            ("with NO comparison source:", 25),
            ("manifests that failed to parse:", 2),
        ],
    )

    assert lines[1:] == [
        "    with a file-level plan signal:      3",
        "    with NO comparison source:          25",
        "    manifests that failed to parse:     2",
    ]
    for line in lines[1:]:
        # Every value starts at column 40, i.e. after 4 spaces of indent plus a
        # 36-character label gutter.
        assert len(line) - len(line.split()[-1]) == 40


def test_format_coverage_block_emits_a_multi_line_caveat_verbatim_above_the_rows():
    """wiped's caveat is a 3-line list literal, including an em dash."""
    caveat = [
        "  COVERAGE (the candidate list above is an OBSERVABLE SUBSET, not the",
        "  full damaged population — tasks with no recoverable plan scope are",
        "  UNKNOWN, neither clean nor damaged):",
    ]

    lines = format_coverage_block(caveat, [("total tasks scanned:", 41)])

    assert lines == caveat + ["    total tasks scanned:                41"]


def test_format_coverage_block_accepts_a_single_pre_joined_caveat_string():
    """combine's _COVERAGE_CAVEAT is one module-level string constant."""
    caveat = (
        "  COVERAGE (the findings above are an OBSERVABLE SUBSET, not the full "
        "damaged population - a combine target with no comparison source is "
        "UNKNOWN, neither clean nor damaged):"
    )

    lines = format_coverage_block(caveat, [("combine targets scanned:", 67)])

    assert lines == [caveat, "    combine targets scanned:            67"]


def test_format_coverage_block_renders_trailing_detail_lines():
    """combine NAMES its unreadable sidecars rather than only counting them."""
    lines = format_coverage_block(
        "  CAVEAT:",
        [("manifests that failed to parse:", 2)],
        details=["plans/a.yaml: bad yaml", "plans/b.yaml: missing key"],
    )

    assert lines[-2:] == [
        "      - plans/a.yaml: bad yaml",
        "      - plans/b.yaml: missing key",
    ]


def test_format_coverage_block_omits_the_detail_section_entirely_when_none_given():
    lines = format_coverage_block("  CAVEAT:", [("total tasks scanned:", 0)])

    assert lines == ["  CAVEAT:", "    total tasks scanned:                0"]
    assert not any(line.lstrip().startswith("- ") for line in lines)


def test_format_coverage_block_keeps_a_separating_space_for_an_overlong_label():
    """A label past the 36-char gutter must not collide with its value.

    Every label in both scripts is <= 31 chars today, so this guards a FUTURE
    long label against silently producing an unreadable ``label:12`` line.
    """
    label = "a label that is quite a lot longer than the gutter:"
    assert len(label) > 36

    lines = format_coverage_block("  CAVEAT:", [(label, 12)])

    assert lines[1] == f"    {label} 12"


# ---------------------------------------------------------------------------
# Shared test fixtures (scripts/tests/conftest.py).
#
# Self-checks for the two fixtures that replace helpers previously copied
# across the three sweep-script test files: the fake-db builder (3 copies
# under two names and two return types) and the project-root toucher (2
# copies). A fixture used by ~30 tests deserves its own direct coverage —
# otherwise a bug in it surfaces as a confusing failure somewhere downstream
# rather than here.
#
# The DDL itself (`_TASKS_SCHEMA`, private to conftest.py) gets no separate
# test: `make_tasks_db` executes it on every use, so a dropped column fails
# loudly at INSERT time in ~30 places already.
# ---------------------------------------------------------------------------

def test_make_tasks_db_roundtrips_rows_through_a_plain_sqlite_read(make_tasks_db):
    db_path = make_tasks_db([
        {"id": 1, "tag": "master", "description": "first"},
        {"id": 2, "tag": "master", "description": "second"},
    ])

    assert isinstance(db_path, Path)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, description FROM tasks ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [(1, "first"), (2, "second")]


def test_make_tasks_db_leaves_an_omitted_nullable_column_null(make_tasks_db):
    db_path = make_tasks_db([{"id": 1}])

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT description, details, test_strategy, metadata FROM tasks"
        ).fetchone()
    finally:
        conn.close()
    assert row == (None, None, None, None)


def test_make_tasks_db_json_encodes_a_non_str_metadata_value(make_tasks_db):
    """audit's copy does this today and its tests depend on it.

    A str metadata value must still pass through VERBATIM, so a test can
    insert deliberately malformed JSON to exercise the decoder's error path.
    """
    db_path = make_tasks_db([
        {"id": 1, "metadata": {"files": ["a.py"]}},
        {"id": 2, "metadata": "{not valid json"},
        {"id": 3, "metadata": None},
    ])

    conn = sqlite3.connect(db_path)
    try:
        rows = dict(conn.execute("SELECT id, metadata FROM tasks").fetchall())
    finally:
        conn.close()
    assert json.loads(rows[1]) == {"files": ["a.py"]}
    assert rows[2] == "{not valid json"
    assert rows[3] is None


def test_make_tasks_db_can_build_several_databases_by_name(make_tasks_db):
    """Factory, not a plain fixture: existing multi-db tests (e.g. the report
    grouping ones) build two databases inside a single test."""
    first = make_tasks_db([{"id": 1}], name="a.db")
    second = make_tasks_db([{"id": 2}], name="b.db")

    assert first != second
    assert first.exists() and second.exists()


def test_make_tasks_db_accepts_an_explicit_directory(make_tasks_db, tmp_path):
    """Needed so a caller can seed a db at a project-root-relative location
    (``<root>/.taskmaster/tasks/``) rather than at the bare tmp_path."""
    target = tmp_path / "proj" / ".taskmaster" / "tasks"
    target.mkdir(parents=True)

    db_path = make_tasks_db([{"id": 1}], directory=target)

    assert db_path == target / "tasks.db"
    assert db_path.exists()


def test_project_root_with_tasks_db_is_discoverable(project_root_with_tasks_db, tmp_path):
    root = tmp_path / "proj"

    db_path = project_root_with_tasks_db(root)

    assert db_path == root / ".taskmaster" / "tasks" / "tasks.db"
    assert discover_db_paths(project_roots=[str(root)]) == [str(db_path)]


def test_project_root_with_tasks_db_is_idempotent(project_root_with_tasks_db, tmp_path):
    """Called twice on the same root it must not raise — one existing test
    builds a root and then re-touches it."""
    root = tmp_path / "proj"

    first = project_root_with_tasks_db(root)
    second = project_root_with_tasks_db(root)

    assert first == second


def test_project_root_with_tasks_db_does_not_truncate_a_seeded_db(
    project_root_with_tasks_db, make_tasks_db, tmp_path
):
    """Idempotent must also mean NON-DESTRUCTIVE.

    ``mkdir(exist_ok=True)`` only makes the directory half idempotent; an
    unconditional ``write_text('')`` would blank an already-seeded tasks.db.
    Seed-then-make-discoverable is the natural call ordering, and because this
    fixture auto-resolves for every file in ``scripts/tests/`` the failure mode
    would be silent: the rows vanish and a downstream ``scan_db(...) == []``
    passes vacuously.
    """
    tasks_dir = tmp_path / "proj" / ".taskmaster" / "tasks"
    tasks_dir.mkdir(parents=True)
    seeded = make_tasks_db([{"id": 1, "description": "keep me"}], directory=tasks_dir)

    db_path = project_root_with_tasks_db(tmp_path / "proj")

    assert db_path == seeded
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT description FROM tasks").fetchall() == [("keep me",)]
    finally:
        conn.close()
