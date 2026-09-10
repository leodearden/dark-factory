"""Tests for scripts/tasks_db_schema.py — the "ask the store what shape it is"
forensic tool (task 5330).

THE CONTRACT UNDER TEST IS TRUTH-TELLING ABOUT WHATEVER STORE IT IS HANDED,
never knowledge of the tasks schema. So every assertion here is a round-trip
property check against a temp db built by ``scripts/tests/conftest.py``'s
``make_tasks_db``: the tool's report must equal what ``PRAGMA table_info``
says about that same connection, and the python type it claims for a column
must be the type a real insert/select actually hands back. A test that pinned
the live tasks columns would make this suite the fifth hand-maintained copy of
the column list — the very population the tool exists to retire.

NOTHING HERE IMPORTS ``fused_memory.backends.sqlite_task_backend``, and that is
deliberate rather than an oversight. It hard-imports ``aiosqlite``, which the
``uv run --project shared`` environment this suite runs under does not carry —
a ``ModuleNotFoundError`` shape the confusion codebook records at least five
separate times. Asserting against ``_SCHEMA_SQL`` would also test the wrong
property: the tool must be right about a store it has never seen.

COVERAGE — the two chains that run this directory (see
``scripts/tests/test_task_db_scan.py``'s docstring for the near-homograph trap
between ``scripts/tests/`` and ``tests/scripts/``):

    scripts/orchestrator.yaml (test_command)
    dark-factory-orchestrator.yaml (test_command, tail segment)

To run just this suite:

    uv run --project shared pytest scripts/tests/test_tasks_db_schema.py -q
"""
from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest
from _task_db_scan import connect_ro
from tasks_db_schema import MainCheckoutUnresolved, introspect, resolve_live_db_path

# ---------------------------------------------------------------------------
# Git fixtures. Real repositories rather than a stubbed `git worktree list`:
# the property under test is that the tool resolves the MAIN checkout from
# wherever it is standing, and a stub of git's own output would be a stub of
# the only thing that can be wrong.
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise AssertionError(f"git {args} failed (rc={result.returncode}): {result.stderr}")
    return result


def _main_checkout(tmp_path: Path) -> Path:
    """`git init` a checkout carrying one commit; return it."""
    main = tmp_path / "checkout"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    _git(main, "config", "user.email", "test@example.com")
    _git(main, "config", "user.name", "Tasks Db Schema Test")
    (main / "README.md").write_text("hello\n")
    _git(main, "add", "-A")
    _git(main, "commit", "-q", "--no-verify", "-m", "initial")
    return main


def _linked_worktree(main: Path, name: str = "5330") -> Path:
    """`git worktree add` a task lane at ``<main>/.worktrees/<name>``."""
    dest = main / ".worktrees" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(main, "worktree", "add", "-q", "-b", f"task/{name}", str(dest))
    return dest


def _seed_store(root: Path, make_tasks_db, rows=({"id": 1, "status": "done"},)) -> Path:
    """Seed ``<root>/.taskmaster/tasks/tasks.db`` and return its path."""
    tasks_dir = root / ".taskmaster" / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    return make_tasks_db(list(rows), directory=tasks_dir)


# ---------------------------------------------------------------------------
# resolve_live_db_path(start) -> Path
# ---------------------------------------------------------------------------

def test_resolve_live_db_path_resolves_the_main_checkout_from_a_worktree(
    tmp_path, make_tasks_db
):
    """The whole point: a worktree has NO .taskmaster/ of its own.

    ``.taskmaster/`` is not tracked in git, so it exists only at the main
    checkout — but a forensic reader is almost always standing in a task lane
    when they reach for it, and a cwd-relative guess resolves to a path that
    does not exist there.
    """
    main = _main_checkout(tmp_path)
    worktree = _linked_worktree(main)
    seeded = _seed_store(main, make_tasks_db)
    deep = worktree / "scripts" / "legibility"
    deep.mkdir(parents=True)

    resolved = resolve_live_db_path(deep)

    assert resolved.is_absolute()
    assert resolved.resolve() == seeded.resolve()
    assert resolved.resolve() != (worktree / ".taskmaster" / "tasks" / "tasks.db").resolve()


def test_resolve_live_db_path_is_the_same_answer_from_the_main_checkout(
    tmp_path, make_tasks_db
):
    """Standing in the main checkout is not a special case to remember."""
    main = _main_checkout(tmp_path)
    worktree = _linked_worktree(main)
    seeded = _seed_store(main, make_tasks_db)

    assert resolve_live_db_path(main).resolve() == seeded.resolve()
    assert resolve_live_db_path(worktree).resolve() == resolve_live_db_path(main).resolve()


def test_resolve_live_db_path_refuses_a_start_outside_any_git_tree(tmp_path):
    """No git tree, no main checkout — so refuse rather than return a guess.

    Returning ``<start>/.taskmaster/tasks/tasks.db`` here would be a guess
    dressed as an answer, and the reader would meet it as an unexplained
    `no such table` or `unable to open database file` several steps later.
    """
    outside = tmp_path / "not-a-repo"
    outside.mkdir()

    with pytest.raises(MainCheckoutUnresolved) as excinfo:
        resolve_live_db_path(outside)

    assert excinfo.value.start == outside.resolve()
    assert str(excinfo.value.start) in str(excinfo.value)


# ---------------------------------------------------------------------------
# introspect(conn) -> tuple[Table, ...]
#
# The tool must be right about a store it has never seen, so every expectation
# below is derived from the SAME connection through PRAGMA table_info rather
# than written down here. A test that spelled out the tasks columns would be
# the fifth hand-maintained copy of them.
# ---------------------------------------------------------------------------

def _arbitrary_db(tmp_path: Path) -> Path:
    """A store with shapes the tasks fixture does not have.

    A composite primary key exercises the pk ORDINAL rather than a boolean,
    and ``loose`` declares a column with no type at all — sqlite's dynamic
    affinity, which the tool must report as unconstrained rather than guess.
    """
    path = tmp_path / "arbitrary.db"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE pair (
                left_id  INTEGER NOT NULL,
                right_id INTEGER NOT NULL,
                note     TEXT,
                PRIMARY KEY (left_id, right_id)
            );
            CREATE TABLE loose (anything);
            """
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _assert_report_matches_pragma(conn: sqlite3.Connection) -> None:
    """The report equals what the database itself says, table for table."""
    reported = introspect(conn)

    listed = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert listed, "the fixture seeded no tables, so this check would be vacuous"
    assert {table.name for table in reported} == listed

    for table in reported:
        pragma = conn.execute(f"PRAGMA table_info({table.name})").fetchall()
        assert [
            (column.name, column.declared_type, column.notnull, column.pk)
            for column in table.columns
        ] == [(row[1], row[2], bool(row[3]), row[5]) for row in pragma]


def test_introspect_matches_pragma_table_info_for_a_task_store(make_tasks_db):
    conn = connect_ro(make_tasks_db([{"id": 1, "status": "done"}]))
    try:
        _assert_report_matches_pragma(conn)
    finally:
        conn.close()


def test_introspect_matches_pragma_table_info_for_an_arbitrary_store(tmp_path):
    conn = connect_ro(_arbitrary_db(tmp_path))
    try:
        _assert_report_matches_pragma(conn)
    finally:
        conn.close()


def test_introspect_reports_pk_ordinals_and_nullability(tmp_path):
    """pk is a POSITION, not a flag — a composite key has a first and a second
    column, and a reader joining on one of them needs to know which."""
    conn = connect_ro(_arbitrary_db(tmp_path))
    try:
        tables = {table.name: table for table in introspect(conn)}
        pair = {column.name: column for column in tables["pair"].columns}
    finally:
        conn.close()

    assert (pair["left_id"].pk, pair["right_id"].pk, pair["note"].pk) == (1, 2, 0)
    assert pair["left_id"].notnull is True
    assert pair["note"].notnull is False


def test_introspect_returns_records_rather_than_rendered_text(make_tasks_db):
    """`introspect` produces DATA; formatting is `render`'s job.

    Returning text would force every caller — including this suite — into an
    ad-hoc parser of the tool's own layout (heuristic 12).
    """
    conn = connect_ro(make_tasks_db([{"id": 1}]))
    try:
        reported = introspect(conn)
    finally:
        conn.close()

    assert not isinstance(reported, str)
    first_column = reported[0].columns[0]
    with pytest.raises(AttributeError):
        first_column.name = "renamed"
