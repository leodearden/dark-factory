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
from tasks_db_schema import (
    EXIT_OK,
    EXIT_UNREADABLE,
    MainCheckoutUnresolved,
    introspect,
    main,
    resolve_live_db_path,
)

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

    ``assorted.payload`` is a BLOB column holding a **str**, deliberately. A
    bytes value there would satisfy a ``-> bytes`` claim by coincidence and
    leave the round-trip assertions passing for the wrong reason; BLOB affinity
    coerces nothing, so a str is what actually comes back out.
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
            CREATE TABLE assorted (
                label   VARCHAR(20),
                ratio   DOUBLE,
                payload BLOB
            );
            """
        )
        conn.execute(
            "INSERT INTO assorted (label, ratio, payload) VALUES (?, ?, ?)",
            ("a label", 1.5, "a str in a BLOB column"),
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
    pytest.raises(AttributeError, setattr, first_column, "name", "renamed")


# ---------------------------------------------------------------------------
# Column.python_type — the remedy for the AttributeError half of the sighting.
#
# Every claim here is checked against a REAL insert/select round trip, never
# against a lookup table the test also wrote.
# ---------------------------------------------------------------------------

def _columns_of(conn: sqlite3.Connection, table: str) -> dict:
    return {
        column.name: column
        for reported in introspect(conn)
        if reported.name == table
        for column in reported.columns
    }


def test_reported_python_type_is_what_a_read_actually_hands_back(make_tasks_db):
    conn = connect_ro(make_tasks_db([{"id": 7, "status": "done"}]))
    try:
        columns = _columns_of(conn, "tasks")
        row_id, row_status = conn.execute("SELECT id, status FROM tasks").fetchone()
    finally:
        conn.close()

    assert type(row_id) is columns["id"].python_type
    assert type(row_status) is columns["status"].python_type
    assert (columns["id"].python_type, columns["status"].python_type) == (int, str)


def test_a_task_id_read_from_the_store_has_no_isdigit(make_tasks_db):
    """The sighting itself: ``'int' object has no attribute 'isdigit'``.

    A task id is spelled as a string everywhere a human meets one — in a
    branch name, a worktree path, an escalation id — so reaching for a string
    method on one read out of the store is the natural mistake. The reported
    python type is what makes it avoidable BEFORE the query is written.
    """
    conn = connect_ro(make_tasks_db([{"id": 7}]))
    try:
        columns = _columns_of(conn, "tasks")
        (row_id,) = conn.execute("SELECT id FROM tasks").fetchone()
    finally:
        conn.close()

    assert columns["id"].python_type is int
    assert getattr(row_id, "isdigit", None) is None


def test_python_type_follows_affinity_rules_not_an_exact_type_table(tmp_path):
    """``VARCHAR(20)`` and ``DOUBLE`` are neither of them the canonical
    spellings, and both round-trip to the reported type."""
    conn = connect_ro(_arbitrary_db(tmp_path))
    try:
        columns = _columns_of(conn, "assorted")
        row = conn.execute("SELECT label, ratio FROM assorted").fetchone()
    finally:
        conn.close()

    assert [type(value) for value in row] == [
        columns[name].python_type for name in ("label", "ratio")
    ]
    assert [columns[name].python_type for name in ("label", "ratio")] == [str, float]


def test_a_blob_column_is_unconstrained_because_blob_affinity_coerces_nothing(
    tmp_path,
):
    """BLOB is the one affinity that is the ABSENCE of coercion.

    "A column with affinity BLOB does not prefer one storage class over
    another and no attempt is made to coerce data" — so a ``-> bytes`` claim
    is false for every value that was not written as bytes, and is exactly the
    authoritative-looking wrong answer this tool exists to remove. Measured
    here rather than argued: the fixture writes a **str** into the BLOB column
    and a str is what comes back.
    """
    conn = connect_ro(_arbitrary_db(tmp_path))
    try:
        payload = _columns_of(conn, "assorted")["payload"]
        (stored,) = conn.execute("SELECT payload FROM assorted").fetchone()
    finally:
        conn.close()

    assert payload.declared_type == "BLOB"
    assert type(stored) is str
    assert payload.python_type is None


def test_a_column_with_no_declared_type_is_reported_as_unconstrained(tmp_path):
    """sqlite constrains nothing here, so the tool must claim nothing.

    A guessed ``str`` would be exactly the failure mode this tool exists to
    remove, one layer further in — an authoritative-looking wrong answer.
    """
    conn = connect_ro(_arbitrary_db(tmp_path))
    try:
        loose = _columns_of(conn, "loose")["anything"]
    finally:
        conn.close()

    assert loose.declared_type == ""
    assert loose.python_type is None


# ---------------------------------------------------------------------------
# main(argv) — the one-command answer, and its two refusals.
#
# Asserted on exit codes and stream separation, never on layout: the contract
# is "the reader gets the shape, or a diagnosis, and never silence".
# ---------------------------------------------------------------------------

def _column_line(out: str, column_name: str) -> str:
    """The single output line whose whitespace-separated tokens name *column_name*."""
    lines = [line for line in out.splitlines() if column_name in line.split()]
    assert len(lines) == 1, f"expected one line naming {column_name!r}, got {lines!r}"
    return lines[0]


def test_main_prints_every_table_and_its_columns_for_an_explicit_db(
    make_tasks_db, capsys
):
    db = make_tasks_db([{"id": 1, "status": "done"}])
    conn = connect_ro(db)
    try:
        reported = introspect(conn)
    finally:
        conn.close()

    exit_code = main(["--db", str(db)])

    out = capsys.readouterr().out
    assert exit_code == 0
    for table in reported:
        assert table.name in out
        for column in table.columns:
            line = _column_line(out, column.name)
            assert column.declared_type in line
            if column.python_type is not None:
                assert column.python_type.__name__ in line


def test_main_resolves_the_store_under_an_explicit_project_root(
    tmp_path, make_tasks_db, capsys
):
    root = tmp_path / "some-project"
    root.mkdir()
    _seed_store(root, make_tasks_db)

    exit_code = main(["--project-root", str(root)])

    assert exit_code == 0
    assert "tasks" in capsys.readouterr().out


def test_main_with_no_path_arguments_resolves_from_the_cwd(
    tmp_path, make_tasks_db, monkeypatch, capsys
):
    """The zero-argument spelling is the one a reader will actually type, and
    it has to work from inside a task lane."""
    main_checkout = _main_checkout(tmp_path)
    worktree = _linked_worktree(main_checkout)
    _seed_store(main_checkout, make_tasks_db)
    monkeypatch.chdir(worktree)

    exit_code = main([])

    assert exit_code == 0
    assert "tasks" in capsys.readouterr().out


def test_main_diagnoses_a_git_that_wedges_past_the_timeout(tmp_path, monkeypatch, capsys):
    """`subprocess.TimeoutExpired` is a SubprocessError, NOT an OSError.

    A handler catching only `OSError` lets a wedged git escape `main`'s own
    ``except`` and surface as a raw traceback — breaking the contract its
    docstring states ("the reader gets the shape, or a diagnosis, and never
    silence") in precisely the incident this tool is for: a loaded box with a
    contended ``.git``.
    """
    def _wedged(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0] if args else "git", timeout=30)

    monkeypatch.setattr("tasks_db_schema.subprocess.run", _wedged)
    monkeypatch.chdir(tmp_path)

    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == EXIT_UNREADABLE
    assert captured.out == ""
    assert str(tmp_path.resolve()) in captured.err


def test_main_exits_non_zero_and_says_why_when_the_store_is_absent(tmp_path, capsys):
    """A forensic tool that exits 0 having reported nothing is worse than no
    tool: it tells the reader the store has no tables."""
    absent = tmp_path / "nowhere" / "tasks.db"

    exit_code = main(["--db", str(absent)])

    captured = capsys.readouterr()
    assert exit_code == EXIT_UNREADABLE
    assert captured.out == ""
    assert str(absent) in captured.err


def test_main_diagnoses_a_zero_byte_stub_differently_from_an_absent_store(
    tmp_path, capsys
):
    stub = tmp_path / "tasks.db"
    stub.write_bytes(b"")

    stub_exit = main(["--db", str(stub)])
    stub_captured = capsys.readouterr()
    absent_exit = main(["--db", str(tmp_path / "nowhere" / "tasks.db")])
    absent_captured = capsys.readouterr()

    assert (stub_exit, absent_exit) == (EXIT_UNREADABLE, EXIT_UNREADABLE)
    assert stub_captured.out == ""
    assert str(stub) in stub_captured.err
    assert stub_captured.err != absent_captured.err


def test_main_refuses_a_readable_store_that_has_no_tables(tmp_path, capsys):
    """Exiting 0 with an empty report IS the "this store has no tables" claim
    that `main`'s own docstring forbids.

    Creating then dropping a table leaves 8192 bytes of perfectly valid
    sqlite, so neither the existence check nor the size check can see it —
    and the reader who typed the wrong `--db` would be told the store is
    empty rather than that it is the wrong file.
    """
    table_less = tmp_path / "tasks.db"
    conn = sqlite3.connect(table_less)
    try:
        conn.execute("CREATE TABLE placeholder (x INTEGER)")
        conn.execute("DROP TABLE placeholder")
        conn.commit()
    finally:
        conn.close()

    exit_code = main(["--db", str(table_less)])

    captured = capsys.readouterr()
    assert exit_code == EXIT_UNREADABLE
    assert captured.out == ""
    assert str(table_less) in captured.err


def test_main_refuses_a_file_that_is_not_a_sqlite_database(tmp_path, capsys):
    """Pointing at the wrong file entirely earns a diagnosis, not a traceback."""
    not_a_database = tmp_path / "tasks.db"
    not_a_database.write_text('{"tasks": []}')

    exit_code = main(["--db", str(not_a_database)])

    captured = capsys.readouterr()
    assert exit_code == EXIT_UNREADABLE
    assert captured.out == ""
    assert str(not_a_database) in captured.err


def test_main_refuses_an_empty_db_argument_instead_of_reading_the_live_store(
    tmp_path, monkeypatch, capsys
):
    """``--db ''`` is a path the reader NAMED, not an argument they omitted.

    Branching on truthiness would fall through to the live store and hand back
    a confident report about a different database than the one asked for — the
    same authoritative-wrong-answer class as the column types, one layer up.
    """
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--db", ""])

    captured = capsys.readouterr()
    assert exit_code == EXIT_UNREADABLE
    assert captured.out == ""


def test_main_refuses_an_empty_project_root_argument_the_same_way(
    tmp_path, monkeypatch, capsys
):
    """The sibling flag has the sibling bug, so it gets the sibling guard."""
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--project-root", ""])

    captured = capsys.readouterr()
    assert exit_code == EXIT_UNREADABLE
    assert captured.out == ""


def test_the_unreadable_exit_does_not_collide_with_argparses_usage_error():
    """A scripted caller must tell "I mistyped a flag" from "the store was
    unreadable", and argparse already owns 2.

    Measured against argparse itself rather than asserted as a literal, so the
    guard still holds if that code ever moves.
    """
    with pytest.raises(SystemExit) as excinfo:
        main(["--no-such-flag"])

    assert excinfo.value.code == 2
    assert excinfo.value.code != EXIT_UNREADABLE
    assert EXIT_UNREADABLE != EXIT_OK
