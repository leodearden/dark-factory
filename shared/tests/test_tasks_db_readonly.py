"""Tests for shared/src/shared/tasks_db_readonly.py::connect_ro (task 5330).

Ported from scripts/tests/test_task_db_scan.py when task 6268 promoted the
read-only open out of that scripts/ sibling. scripts/_task_db_scan.py owns the
tasks.db PATH; connect_ro is the "...and open it" half. The contract under
test is REFUSAL: a forensic reader who points at the wrong file must be told
which mistake they made, not handed a connection whose first query answers
`no such table: tasks`.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from shared.task_store_schema import TASK_STORE_DDL
from shared.tasks_db_readonly import TaskDbProblem, TaskDbUnreadable, connect_ro


def _make_store(directory: Path, rows: list[dict]) -> Path:
    """A real task store at ``directory/tasks.db``, built from TASK_STORE_DDL."""
    path = directory / "tasks.db"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(TASK_STORE_DDL)
        for row in rows:
            values = {"title": "t", "status": "done", "updated_at": "now", **row}
            columns = ", ".join(values)
            placeholders = ", ".join("?" for _ in values)
            conn.execute(
                f"INSERT INTO tasks ({columns}) VALUES ({placeholders})",
                tuple(values.values()),
            )
        conn.commit()
    finally:
        conn.close()
    return path


def _empty_the_tables_of(path: Path) -> Path:
    """Make *path* a READABLE sqlite database carrying zero tables.

    Creating then dropping a table leaves 8192 bytes of valid sqlite behind —
    so this shape passes both the existence check and the size check while
    being exactly as useless as the 0-byte decoy.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE placeholder (x INTEGER)")
        conn.execute("DROP TABLE placeholder")
        conn.commit()
    finally:
        conn.close()
    return path


def test_connect_ro_refuses_a_path_that_does_not_exist(tmp_path):
    absent = tmp_path / "absent" / "tasks.db"

    with pytest.raises(TaskDbUnreadable) as excinfo:
        connect_ro(absent)

    assert excinfo.value.reason is TaskDbProblem.ABSENT
    assert excinfo.value.path == absent.resolve()


def test_connect_ro_refusal_names_the_path_in_its_message(tmp_path):
    """A reason a caller can branch on AND prose a human can act on.

    The discriminator is the FIELD asserted above, never a substring of this
    message — tests and callers that match on prose are an ad-hoc parser
    (heuristic 12) and pin wording that is free to improve.
    """
    absent = tmp_path / "absent" / "tasks.db"

    with pytest.raises(TaskDbUnreadable) as excinfo:
        connect_ro(absent)

    assert str(excinfo.value.path) in str(excinfo.value)


def test_connect_ro_does_not_create_the_database_it_refuses(tmp_path):
    """The load-bearing one: refusing must not leave a store behind.

    A read-WRITE ``sqlite3.connect`` on a path that does not exist silently
    CREATES an empty database — which is how a worktree-relative guess turns
    into a real 0-byte file whose every query then answers `no such table:
    tasks`. That error is recorded four times in the confusion codebook, and
    every one of them reads as "the store is empty" rather than "you are
    looking in the wrong place".
    """
    absent = tmp_path / "absent" / "tasks.db"

    with pytest.raises(TaskDbUnreadable):
        connect_ro(absent)

    assert not absent.exists()
    assert not absent.parent.exists()


def test_connect_ro_refuses_a_zero_byte_stub_with_its_own_reason(tmp_path):
    """A stub is a DIFFERENT mistake from an absent path, so it gets its own
    discriminator: the reader pointed at a real file that is not a store."""
    stub = tmp_path / "tasks.db"
    stub.write_bytes(b"")

    with pytest.raises(TaskDbUnreadable) as excinfo:
        connect_ro(stub)

    assert excinfo.value.reason is TaskDbProblem.EMPTY_STUB
    assert excinfo.value.path == stub.resolve()


def test_connect_ro_refuses_a_directory_rather_than_reporting_a_disk_io_error(
    tmp_path,
):
    """A directory is a fifth mistake, and sqlite's own answer to it is unusable.

    Measured: a read-only open of a directory raises ``disk I/O error``
    (``SQLITE_IOERR``) from ``sqlite3.connect`` itself — a message that names
    no path and reads as failing hardware rather than as a mistyped argument.
    Two spellings land here and both are live: naming ``.taskmaster/tasks``
    instead of the store inside it, and an empty path string, which
    ``Path("").resolve()`` turns into the current working directory.
    """
    with pytest.raises(TaskDbUnreadable) as excinfo:
        connect_ro(tmp_path)

    assert excinfo.value.reason is TaskDbProblem.IS_A_DIRECTORY
    assert excinfo.value.path == tmp_path.resolve()
    assert str(tmp_path.resolve()) in str(excinfo.value)


def test_connect_ro_refuses_an_empty_path_as_the_directory_it_resolves_to(tmp_path,
                                                                          monkeypatch):
    """``Path("")`` is not "no path" — it resolves to the cwd.

    A caller forwarding an empty ``--db``/``--project-root`` therefore hands
    this function a real, existing directory, which is why the arm above is
    reached by an argument the reader typed rather than only by a path they
    chose.
    """
    monkeypatch.chdir(tmp_path)

    with pytest.raises(TaskDbUnreadable) as excinfo:
        connect_ro("")

    assert excinfo.value.reason is TaskDbProblem.IS_A_DIRECTORY
    assert excinfo.value.path == tmp_path.resolve()


def test_connect_ro_refuses_a_readable_store_that_has_no_tables(tmp_path):
    """A third mistake with a third remedy: a real sqlite file, wrong file."""
    table_less = _empty_the_tables_of(tmp_path / "tasks.db")

    with pytest.raises(TaskDbUnreadable) as excinfo:
        connect_ro(table_less)

    assert excinfo.value.reason is TaskDbProblem.NO_TABLES
    assert excinfo.value.path == table_less.resolve()
    assert str(excinfo.value.path) in str(excinfo.value)


def test_every_refusal_of_one_path_reads_differently(tmp_path):
    """No two refusals may read the same, and EVERY arm is covered.

    Collapsing any of them into one "unusable store" message would restore
    exactly the unactionable signal this guard exists to remove — the whole
    point of the enum is that the reader learns WHICH mistake they made, and
    each has its own remedy (resolve the main checkout; you named the
    containing directory; you are one directory too high; you are pointing at
    some other .db; that is not a database at all).

    The expected arms are read off ``TaskDbProblem`` rather than counted here,
    so a new arm shipped without its own remedy prose fails THIS test instead
    of waiting for someone to notice the count. The SAME path takes every
    shape in turn, so a difference between the messages can only come from the
    remedy prose and never from the path each of them names.
    """
    path = tmp_path / "tasks.db"

    path.write_bytes(b"")
    with pytest.raises(TaskDbUnreadable) as stub_refusal:
        connect_ro(path)

    _empty_the_tables_of(path)
    with pytest.raises(TaskDbUnreadable) as table_less_refusal:
        connect_ro(path)

    path.write_text('{"tasks": []}')
    with pytest.raises(TaskDbUnreadable) as not_a_database_refusal:
        connect_ro(path)

    path.unlink()
    with pytest.raises(TaskDbUnreadable) as absent_refusal:
        connect_ro(path)

    path.mkdir()
    with pytest.raises(TaskDbUnreadable) as directory_refusal:
        connect_ro(path)

    refusals = (
        stub_refusal,
        table_less_refusal,
        not_a_database_refusal,
        absent_refusal,
        directory_refusal,
    )
    assert {r.value.reason for r in refusals} == set(TaskDbProblem)
    assert len({str(r.value) for r in refusals}) == len(refusals)


def test_connect_ro_refuses_a_stub_that_grew_past_zero_bytes_without_tables(tmp_path):
    """The guard has to be SEMANTIC, not size-based — measured here.

    Give the 0-byte decoy one read-write open and a ``PRAGMA
    journal_mode=WAL`` and it becomes 4096 bytes with still no tables: the
    ``st_size == 0`` arm stops firing on the very shape it was written to
    catch, while the file stays exactly as unusable. Size is evidence, never
    the question.
    """
    grown = tmp_path / "tasks.db"
    grown.write_bytes(b"")
    conn = sqlite3.connect(grown)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    finally:
        conn.close()
    assert grown.stat().st_size > 0

    with pytest.raises(TaskDbUnreadable) as excinfo:
        connect_ro(grown)

    assert excinfo.value.reason is TaskDbProblem.NO_TABLES


def test_an_unguarded_read_only_open_of_a_stub_answers_no_such_table(tmp_path):
    """WHY the guard earns its place — the error it replaces, reproduced.

    ``mode=ro`` alone does not refuse a 0-byte file: it hands back a working
    connection to a database with no tables in it. This is the state behind
    the four ``no such table: tasks`` sightings in the confusion codebook, and
    behind the verified decoy at the main checkout's ``.taskmaster/tasks.db``
    — one directory above the real store, 0 bytes.
    """
    stub = tmp_path / "tasks.db"
    stub.write_bytes(b"")

    conn = sqlite3.connect(f"file:{stub}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            conn.execute("SELECT * FROM tasks")
    finally:
        conn.close()

    assert "no such table: tasks" in str(excinfo.value)


def test_connect_ro_refuses_a_file_that_is_not_a_sqlite_database(tmp_path):
    """The probe is now the first code to touch the file's bytes, so the
    "not a database at all" error surfaces HERE.

    `connect_ro` promises to open the store or refuse with
    `TaskDbUnreadable`; letting `sqlite3.DatabaseError` escape would leave a
    fresh instance of the same "the contract says X, the code does Y" defect
    this round exists to close — and the reader would get a raw traceback
    instead of a diagnosis.
    """
    not_a_database = tmp_path / "tasks.db"
    not_a_database.write_text('{"tasks": []}')

    with pytest.raises(TaskDbUnreadable) as excinfo:
        connect_ro(not_a_database)

    assert excinfo.value.reason is TaskDbProblem.NOT_A_DATABASE
    assert excinfo.value.path == not_a_database.resolve()
    assert str(excinfo.value.path) in str(excinfo.value)


@pytest.mark.skipif(os.geteuid() == 0, reason="mode 000 does not deny root")
def test_a_store_that_cannot_be_OPENED_is_not_relabelled_not_a_database(tmp_path):
    """The discrimination that keeps the diagnosis honest.

    `sqlite3.OperationalError` is a SUBCLASS of `sqlite3.DatabaseError`, so a
    catch that branched on the exception CLASS would report every unreadable
    store as "not a database" — turning a permissions problem, a locked
    store, or one that vanished mid-open into a confident wrong answer.

    Measured here: non-sqlite bytes report `SQLITE_NOTADB` (26) while a real
    store at mode 000 reports `SQLITE_CANTOPEN` (14). Only the structured
    code tells them apart, so only the structured code may be branched on.

    Asserted as the POSITIVE fact, not as "the reason is not NOT_A_DATABASE".
    That negative was satisfied by several genuine regressions it read as
    covering — a `connect_ro` that relabelled this store `ABSENT` or
    `EMPTY_STUB`, or that stopped refusing in some other way, would have
    passed. Requiring the original `sqlite3.OperationalError` through, with
    `SQLITE_CANTOPEN` on it, pins the discrimination the production
    `if exc.sqlite_errorcode != sqlite3.SQLITE_NOTADB` branch implements.
    """
    unopenable = _make_store(tmp_path, [{"id": 1, "status": "done"}])
    unopenable.chmod(0o000)
    try:
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            connect_ro(unopenable)
    finally:
        unopenable.chmod(0o600)

    assert excinfo.value.sqlite_errorcode == sqlite3.SQLITE_CANTOPEN
    assert excinfo.value.sqlite_errorcode != sqlite3.SQLITE_NOTADB
    assert not isinstance(excinfo.value, TaskDbUnreadable)


def test_an_unguarded_read_only_open_of_a_table_less_store_answers_no_such_table(
    tmp_path,
):
    """WHY the third arm earns its place — the error it replaces, reproduced.

    A store that once had tables and no longer does is 8192 bytes of perfectly
    valid sqlite, so neither the existence check nor the size check can see
    it. ``mode=ro`` opens it happily and every query then answers `no such
    table: tasks` — the same unactionable signal, now from a file that looks
    entirely real.
    """
    table_less = _empty_the_tables_of(tmp_path / "tasks.db")

    conn = sqlite3.connect(f"file:{table_less}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            conn.execute("SELECT * FROM tasks")
    finally:
        conn.close()

    assert "no such table: tasks" in str(excinfo.value)


def test_connect_ro_reads_the_rows_of_a_real_store(tmp_path):
    db = _make_store(tmp_path, [{"id": 1, "status": "done"}, {"id": 2, "status": "pending"}])

    conn = connect_ro(db)
    try:
        rows = conn.execute("SELECT id, status FROM tasks ORDER BY id").fetchall()
    finally:
        conn.close()

    assert rows == [(1, "done"), (2, "pending")]


@pytest.mark.parametrize("uri_syntax", ["#", "?", "%41"])
def test_connect_ro_opens_a_store_whose_path_holds_uri_syntax(tmp_path, uri_syntax):
    """A path character that is URI syntax must reach sqlite as data.

    Spliced raw into ``file:{path}?mode=ro``, a ``#`` starts a fragment that
    swallows ``?mode=ro``, and a ``?`` starts the query early: sqlite then
    opens the truncated prefix READ-WRITE, creates a 0-byte sibling there, and
    refuses it as NO_TABLES — the decoy and the mislabel this module exists to
    prevent. A ``%41`` is decoded as an escape and names another file.
    """
    store_dir = tmp_path / f"a{uri_syntax}b"
    store_dir.mkdir()
    db = _make_store(store_dir, [{"id": 1, "status": "done"}])

    conn = connect_ro(db)
    try:
        rows = conn.execute("SELECT id, status FROM tasks").fetchall()
    finally:
        conn.close()

    assert rows == [(1, "done")]
    assert list(tmp_path.iterdir()) == [store_dir]


def test_connect_ro_connection_cannot_write_to_the_store(tmp_path):
    """``mode=ro`` is actually in force, so a forensic reader can never mutate
    the live store it is measuring — including by accident."""
    db = _make_store(tmp_path, [{"id": 1, "status": "done"}])

    conn = connect_ro(db)
    try:
        with pytest.raises(sqlite3.OperationalError) as update_refusal:
            conn.execute("UPDATE tasks SET status = 'cancelled'")
        with pytest.raises(sqlite3.OperationalError) as insert_refusal:
            conn.execute(
                "INSERT INTO tasks (tag, id, title, status, updated_at) "
                "VALUES ('master', 2, 't', 'done', 'now')"
            )
    finally:
        conn.close()

    assert "readonly" in str(update_refusal.value)
    assert "readonly" in str(insert_refusal.value)


def test_connect_ro_returns_a_plain_connection_not_a_wrapper(tmp_path):
    """Callers keep the whole stdlib API — this helper adds guards, not a
    facade with its own surface to learn and keep in sync."""
    db = _make_store(tmp_path, [{"id": 1}])

    conn = connect_ro(db)
    try:
        assert type(conn) is sqlite3.Connection
    finally:
        conn.close()
