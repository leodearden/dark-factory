"""Behavioral contract: every doc handing out the tasks.db shape recipe has one that works.

Task 5330. Confusion-codebook entry ``fused-memory-api-traps``, sighting
2026-09-09 (session cfae559b, ``manifested_phase: ops``): turn 204 died with
``AttributeError: 'int' object has no attribute 'isdigit'`` and turn 439 with
``sqlite3.OperationalError: no such column: created_at``, both against
``.taskmaster/tasks/tasks.db``. The same class recurs across the codebook at
least eight more times (``no such column: updatedAt / depends_on_id / type /
id / payload``, ``no such table: tasks`` four times, ``unable to open database
file``). Direct read-only sqlite access to the live store is the ENDORSED
house convention; what was missing was any way to learn the store's shape
before executing a query against it.

WHY A MACHINE CHECK FOR A DOCUMENTATION FIX — the reasoning
``test_package_source_lookup_convention.py`` (task 3959) records, which this
file follows structurally: prose only reaches agents who read it, and this
guard goes one step past a string mirror by EXECUTING the command each doc in
:data:`RECIPE_DOCS` hands them. A recipe that no longer runs fails here instead
of failing an agent mid-incident.

WHAT THIS FILE DELIBERATELY DOES NOT DO. It asserts nothing about prose,
wording, headings or ordering, and it does not pin the live tasks column list
— hand-maintained copies of that list were the defect, and task 5334 collapsed
them onto this recipe. This guard keeps every doc that hands the recipe out
runnable. It certifies the recipe RUNS and reports the shape of the store it is
pointed at; the store it is pointed at here is built by this test, so the
expectation is derived rather than remembered.
``skills/hotspot-survey/references/orchestration.md`` is deliberately NOT in
:data:`RECIPE_DOCS`: it is the generic multi-repo template and names no
dark-factory command to execute.

It also certifies CLAUDE.md's ticket-store recipe (task 7438), which hands out
a host-absolute ``--db`` path to the one shared reconciliation ``tickets.db``;
the guard swaps that path for a store it builds, so it never reads production
state.

PLACEMENT IS LOAD-BEARING. ``tests/scripts/`` carries its own module config, so
this guard runs under FULL_SUITE and merge-role ``merge_verify_breadth: full``.
"""
from __future__ import annotations

import pathlib
import shlex
import sqlite3
import subprocess
import sys

import pytest
import verify_command_invariants as vci
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]

CLAUDE_MD_PATH = REPO_ROOT / "CLAUDE.md"

RECIPE_DOCS = (
    CLAUDE_MD_PATH,
    REPO_ROOT / ".claude" / "skills" / "hotspot-survey" / "project.md",
)

SHAPE_LOOKUP_MARKER = "tasks-db-schema-lookup"
SHAPE_LOOKUP_LABEL = "Task store shape"

TICKETS_LOOKUP_MARKER = "tickets-db-schema-lookup"
TICKETS_LOOKUP_LABEL = "Ticket store shape"

_TOOL_RELATIVE_PATH = "scripts/tasks_db_schema.py"

_PYTHON_ARGV0 = ("python", "python3")

_RUN_TIMEOUT_SECS = 60

# The store this guard builds — an ARBITRARY shape, not a copy of the live
# one. The tool's contract is truth-telling about whatever store it is handed,
# so the expectation below is what this DDL declares and nothing else.
_FIXTURE_TABLE = "tasks"
_FIXTURE_COLUMNS = ("id", "status")
_FIXTURE_DDL = "CREATE TABLE tasks (id INTEGER NOT NULL PRIMARY KEY, status TEXT NOT NULL)"

# Likewise ARBITRARY for the ticket store: not a copy of ticket_store.py's DDL.
_TICKETS_FIXTURE_TABLE = "tickets"
_TICKETS_FIXTURE_COLUMNS = ("ticket_id", "status")
_TICKETS_FIXTURE_DDL = "CREATE TABLE tickets (ticket_id TEXT PRIMARY KEY, status TEXT NOT NULL)"


_HAPPY_DOC = """\
Ask the store what shape it is; never guess a column name or a value type.

<!-- tasks-db-schema-lookup:begin
     EXECUTED by tests/scripts/test_tasks_db_schema_convention.py. Edit this
     into anything that is not `python scripts/tasks_db_schema.py` and that
     guard goes red. -->
- **Task store shape**: `python scripts/tasks_db_schema.py`
<!-- tasks-db-schema-lookup:end -->

Writes still go through the fused-memory MCP tools.
"""

_HAPPY_COMMAND = "python scripts/tasks_db_schema.py"

_SELF_TEST_DOC_NAME = "self-test-doc.md"

_DECOY_DOC = """\
The store is at `.taskmaster/tasks/tasks.db`, never inside your worktree.

<!-- tasks-db-schema-lookup:begin
     Mirrors nothing; it is EXECUTED as written by
     tests/scripts/test_tasks_db_schema_convention.py. Contains prose inline
     code such as `.taskmaster/tasks.db` and `sqlite3` that a backtick-keyed
     extractor would grab first. -->
- **Task store shape**: `python scripts/tasks_db_schema.py`
<!-- tasks-db-schema-lookup:end -->

Afterwards, `SELECT` against the columns it printed.
"""


def test_tasks_db_schema_bullet_extracts_the_marked_span():
    """Only the marked bullet's command is returned, backticks stripped."""
    assert (
        vci.marked_bullet_command(
            _HAPPY_DOC,
            slug=SHAPE_LOOKUP_MARKER,
            bullet_label=SHAPE_LOOKUP_LABEL,
            source=_SELF_TEST_DOC_NAME,
            task="5330",
        )
        == _HAPPY_COMMAND
    )


def test_tasks_db_schema_bullet_is_immune_to_inline_code_in_the_begin_comment():
    """Prose inline code inside the marked slice is never extracted.

    An extractor keyed on "the first backtick span after the begin marker"
    would return ``.taskmaster/tasks.db`` here — a plausible-looking string,
    so the mistake would survive review and then be executed as a command.
    """
    assert (
        vci.marked_bullet_command(
            _DECOY_DOC,
            slug=SHAPE_LOOKUP_MARKER,
            bullet_label=SHAPE_LOOKUP_LABEL,
            source=_SELF_TEST_DOC_NAME,
            task="5330",
        )
        == _HAPPY_COMMAND
    )


def _documented_tool_argv(command, doc_name):
    """*command* re-pointed at this interpreter and this checkout's copy of the tool.

    The SHAPE assertion is load-bearing: a recipe degraded into a raw
    ``sqlite3 ... 'SELECT ...'`` one-liner, or into a python ``-c`` that
    hardcodes a column list, fails here rather than being executed.

    ``sys.executable`` rather than a bare ``python`` off PATH, because
    ``verify._target_subprocess_env`` strips ``VIRTUAL_ENV`` and the venv's bin
    from PATH, so a PATH-resolved interpreter could be a system one. The script
    argument is absolutised against REPO_ROOT because the guard runs it with
    cwd set to a temp project root — which is the whole point, since the recipe
    resolves the store from the checkout it is standing in.
    """
    argv = shlex.split(command)
    assert len(argv) == 2 and argv[0] in _PYTHON_ARGV0, (
        f"the command inside the {SHAPE_LOOKUP_MARKER!r} marker in {doc_name} is "
        f"not a bare run of the shape tool (task 5330): {command!r} tokenises to "
        f"{argv!r}, expected exactly [python|python3, {_TOOL_RELATIVE_PATH!r}]. "
        f"The convention is that an agent asks the store and reads the answer — "
        f"a hand-written sqlite query in its place is the confusion this closes."
    )
    assert argv[1] == _TOOL_RELATIVE_PATH, (
        f"the {SHAPE_LOOKUP_MARKER!r} command in {doc_name} runs {argv[1]!r}, "
        f"expected {_TOOL_RELATIVE_PATH!r} (task 5330)"
    )
    return [sys.executable, str(REPO_ROOT / argv[1])]


def _seeded_project_root(tmp_path):
    """A git checkout whose ``.taskmaster/tasks/tasks.db`` holds :data:`_FIXTURE_DDL`.

    Git-initialised because the recipe resolves the live store through
    ``git worktree list --porcelain`` — the mechanism that keeps it correct
    from inside a task lane, where ``.taskmaster/`` does not exist.
    """
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(
        ["git", "-C", str(root), "init", "-q", "-b", "main"],
        check=True,
        capture_output=True,
        timeout=_RUN_TIMEOUT_SECS,
    )

    db_path = root / ".taskmaster" / "tasks" / "tasks.db"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_FIXTURE_DDL)
        conn.commit()
    finally:
        conn.close()
    return root


@pytest.mark.parametrize(
    "doc_path",
    RECIPE_DOCS,
    ids=lambda doc_path: doc_path.relative_to(REPO_ROOT).as_posix(),
)
def test_documented_store_shape_lookup_reports_the_stores_real_shape(doc_path, tmp_path):
    """Each doc's recipe must actually print the shape of the store it finds.

    Executed, not string-compared: this certifies the recipe WORKS, which is
    the claim the doc is making to every agent that reads it.
    """
    doc_name = doc_path.relative_to(REPO_ROOT).as_posix()
    command = vci.marked_bullet_command(
        doc_path.read_text(encoding="utf-8"),
        slug=SHAPE_LOOKUP_MARKER,
        bullet_label=SHAPE_LOOKUP_LABEL,
        source=doc_name,
        task="5330",
    )
    argv = _documented_tool_argv(command, doc_name)
    root = _seeded_project_root(tmp_path)

    out = _recipe_stdout(
        argv, cwd=root, marker=SHAPE_LOOKUP_MARKER, doc_name=doc_name, task="5330"
    )
    _assert_reports_shape(
        out,
        marker=SHAPE_LOOKUP_MARKER,
        store=root,
        table=_FIXTURE_TABLE,
        columns=_FIXTURE_COLUMNS,
        task="5330",
    )


def _documented_ticket_store_argv(command, doc_name, store):
    """*command* re-pointed at this checkout's tool and at *store* in place of the live one.

    The documented operand must be ABSOLUTE: the ticket store is one per-host
    file under the fused-memory server's checkout, and a task worktree has no
    ``data/``, so a relative operand fails from every lane. It is then swapped
    for *store*, so this guard never reads production state.
    """
    argv = shlex.split(command)
    expected = f"[python|python3, {_TOOL_RELATIVE_PATH!r}, '--db', <absolute .../tickets.db>]"
    assert len(argv) == 4 and argv[0] in _PYTHON_ARGV0, (
        f"the command inside the {TICKETS_LOOKUP_MARKER!r} marker in {doc_name} is "
        f"not a bare run of the shape tool on the ticket store (task 7438): "
        f"{command!r} tokenises to {argv!r}, expected exactly {expected}. A "
        f"hand-written sqlite query in its place is the confusion this closes."
    )
    assert argv[1] == _TOOL_RELATIVE_PATH, (
        f"the {TICKETS_LOOKUP_MARKER!r} command in {doc_name} runs {argv[1]!r}, "
        f"expected {_TOOL_RELATIVE_PATH!r} (task 7438): {argv!r}"
    )
    assert argv[2] == "--db", (
        f"the {TICKETS_LOOKUP_MARKER!r} command in {doc_name} passes {argv[2]!r} "
        f"where '--db' belongs (task 7438): {argv!r}, expected {expected}"
    )
    documented_store = pathlib.PurePosixPath(argv[3])
    assert documented_store.is_absolute(), (
        f"the {TICKETS_LOOKUP_MARKER!r} command in {doc_name} names the store as "
        f"{argv[3]!r}, which is relative (task 7438): it resolves against the "
        f"caller's cwd, and a task worktree has no data/. Name the one shared "
        f"store by its absolute path."
    )
    assert documented_store.name == "tickets.db", (
        f"the {TICKETS_LOOKUP_MARKER!r} command in {doc_name} points --db at "
        f"{argv[3]!r}, which is not a tickets.db (task 7438): {argv!r}"
    )
    return [sys.executable, str(REPO_ROOT / argv[1]), "--db", str(store)]


def _seeded_ticket_store(tmp_path):
    """A ``tickets.db`` holding :data:`_TICKETS_FIXTURE_DDL`, laid out like the live one."""
    db_path = tmp_path / "data" / "reconciliation" / "tickets.db"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_TICKETS_FIXTURE_DDL)
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_documented_ticket_store_shape_lookup_reports_the_stores_real_shape(tmp_path):
    """CLAUDE.md's ticket-store recipe must print the shape of the store it is given.

    Executed with the documented store swapped for one this test builds, so
    the guard certifies the recipe works without reading production state.
    """
    doc_name = CLAUDE_MD_PATH.relative_to(REPO_ROOT).as_posix()
    command = vci.marked_bullet_command(
        CLAUDE_MD_PATH.read_text(encoding="utf-8"),
        slug=TICKETS_LOOKUP_MARKER,
        bullet_label=TICKETS_LOOKUP_LABEL,
        source=doc_name,
        task="7438",
    )
    store = _seeded_ticket_store(tmp_path)
    argv = _documented_ticket_store_argv(command, doc_name, store)

    out = _recipe_stdout(
        argv, cwd=tmp_path, marker=TICKETS_LOOKUP_MARKER, doc_name=doc_name, task="7438"
    )
    first_line = out.splitlines()[0] if out else ""
    assert first_line == str(store.resolve()), (
        f"the {TICKETS_LOOKUP_MARKER!r} command's first line is {first_line!r}, "
        f"expected the store it was handed, {str(store.resolve())!r} (task 7438). "
        f"Full output: {out!r}"
    )
    _assert_reports_shape(
        out,
        marker=TICKETS_LOOKUP_MARKER,
        store=store,
        table=_TICKETS_FIXTURE_TABLE,
        columns=_TICKETS_FIXTURE_COLUMNS,
        task="7438",
    )


def _recipe_stdout(argv, *, cwd, marker, doc_name, task):
    """Stdout of the *marker* recipe from *doc_name*, run as *argv* in *cwd*; it must exit 0."""
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_SECS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the {marker!r} command documented in {doc_name} did not "
            f"finish within {_RUN_TIMEOUT_SECS}s (task {task}); argv: {argv!r}"
        )

    assert completed.returncode == 0, (
        f"the {marker!r} command documented in {doc_name} exited "
        f"{completed.returncode} (task {task}) — agents are being handed a recipe "
        f"that does not work.\n"
        f" argv: {argv!r}\n cwd: {cwd}\n"
        f" stdout: {completed.stdout.strip()!r}\n"
        f" stderr: {completed.stderr.strip()!r}"
    )
    return completed.stdout


def _assert_reports_shape(out, *, marker, store, table, columns, task):
    """*out* names *table* and every one of *columns* of the store this test built."""
    assert table in out, (
        f"the {marker!r} command exited 0 without naming the "
        f"{table!r} table of the store at {store} (task {task}): {out!r}"
    )
    missing = [column for column in columns if column not in out]
    assert not missing, (
        f"the {marker!r} command did not report {missing!r} for the "
        f"store this test built (task {task}) — it is meant to answer 'what "
        f"columns does this store have', so a report missing them would leave an "
        f"agent guessing exactly as before. Full output: {out!r}"
    )
