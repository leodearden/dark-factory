"""Behavioral contract: CLAUDE.md's tasks.db shape recipe actually works.

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
guard goes one step past a string mirror by EXECUTING the command CLAUDE.md
hands them. A recipe that no longer runs fails here instead of failing an
agent mid-incident.

WHAT THIS FILE DELIBERATELY DOES NOT DO. It asserts nothing about prose,
wording, headings or ordering, and it does not pin the live tasks column list
— that population IS the defect (four hand-maintained copies already exist in
this repo and one of them is stale, naming a ``parent_id`` column dropped at
the flat-schema change). It certifies the recipe RUNS and reports the shape of
the store it is pointed at; the store it is pointed at here is built by this
test, so the expectation is derived rather than remembered.

THE EXTRACTOR IS RE-IMPLEMENTED RATHER THAN IMPORTED from its two siblings in
this directory, per the no-cross-import-between-guards convention recorded in
``tests/scripts/conftest.py`` — task 3959 copied it from task 3558 for the
same reason.

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

REPO_ROOT = pathlib.Path(__file__).parents[2]

CLAUDE_MD_PATH = REPO_ROOT / "CLAUDE.md"

SHAPE_LOOKUP_MARKER = "tasks-db-schema-lookup"
SHAPE_LOOKUP_LABEL = "Task store shape"

_TOOL_RELATIVE_PATH = "scripts/tasks_db_schema.py"

_PYTHON_ARGV0 = ("python", "python3")

_RUN_TIMEOUT_SECS = 60

# The store this guard builds — an ARBITRARY shape, not a copy of the live
# one. The tool's contract is truth-telling about whatever store it is handed,
# so the expectation below is what this DDL declares and nothing else.
_FIXTURE_TABLE = "tasks"
_FIXTURE_COLUMNS = ("id", "status")
_FIXTURE_DDL = "CREATE TABLE tasks (id INTEGER NOT NULL PRIMARY KEY, status TEXT NOT NULL)"


def _marked_command(markdown_text, marker, bullet_label):
    """The inline-code command on the *bullet_label* bullet delimited by *marker*.

    Every failure is a loud ``AssertionError`` naming the marker literal and
    CLAUDE.md, never a ``''``/``None`` return: an extractor that silently
    yields nothing turns the execution assertion vacuously green while
    certifying nothing, which is strictly worse than no guard because the
    check still reports success. The span regex is anchored on the LABEL
    rather than on backticks, because keying on backticks alone would extract
    the begin comment's own explanatory inline code — a plausible-looking
    string, so the mistake would not announce itself.
    """
    begin = f"{marker}:begin"
    end = f"{marker}:end"

    begin_count = markdown_text.count(begin)
    assert begin_count == 1, (
        f"expected exactly one {begin!r} marker, found {begin_count} (task 5330). "
        f"It delimits the copy-pasteable store-shape command in CLAUDE.md's "
        f"`## Task Routing` section. If it was deleted, restore it around that "
        f"bullet; if it was duplicated, one of the two copies is unpinned and "
        f"free to rot into a command that no longer runs."
    )
    end_count = markdown_text.count(end)
    assert end_count == 1, (
        f"expected exactly one {end!r} marker to close {begin!r} in CLAUDE.md, "
        f"found {end_count} (task 5330) — restore the closing marker below the "
        f"bullet it wraps"
    )

    # Inverted markers yield an empty slice, so the next assertion catches that
    # too, loudly and with the same remedy.
    marked = markdown_text[markdown_text.index(begin):markdown_text.index(end)]
    prefix = f"- **{bullet_label}**: `"
    spans = [
        segment.split("`", 1)[0]
        for segment in marked.split(prefix)[1:]
        if "`" in segment
    ]
    assert len(spans) == 1, (
        f"expected exactly one ``{prefix}<command>``` bullet between {begin!r} "
        f"and {end!r} in CLAUDE.md, found {len(spans)}: {spans!r} (task 5330). "
        f"The marker must wrap that bullet and nothing else; if the bullet was "
        f"relabelled or the markers were inverted, move the marker back around "
        f"the copy-pasteable command."
    )

    command = spans[0].strip()
    assert command, (
        f"the command between {begin!r} and {end!r} in CLAUDE.md is empty (task 5330)"
    )
    return command


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

_NO_MARKER_DOC = """\
- **Task store shape**: `python scripts/tasks_db_schema.py`

Somewhere else entirely, an unmarked mention of `.taskmaster/tasks/tasks.db`.
"""

_DUPLICATE_MARKER_DOC = """\
<!-- tasks-db-schema-lookup:begin -->
- **Task store shape**: `python scripts/tasks_db_schema.py`
<!-- tasks-db-schema-lookup:end -->

## Some later section

<!-- tasks-db-schema-lookup:begin -->
- **Task store shape**: `python scripts/tasks_db_schema.py --db /elsewhere.db`
<!-- tasks-db-schema-lookup:end -->
"""

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


def test_marked_command_extracts_the_marked_span():
    """Only the marked bullet's command is returned, backticks stripped."""
    assert (
        _marked_command(_HAPPY_DOC, SHAPE_LOOKUP_MARKER, SHAPE_LOOKUP_LABEL)
        == _HAPPY_COMMAND
    )


@pytest.mark.parametrize(
    ("markdown_text", "case"),
    [
        (_NO_MARKER_DOC, "missing"),
        (_DUPLICATE_MARKER_DOC, "duplicated"),
    ],
)
def test_marked_command_fails_loudly_on_a_broken_marker(markdown_text, case):
    """A missing or duplicated marker RAISES — never '' or None.

    Missing is the vacuity hazard: an extractor that silently returns nothing
    turns every downstream assertion green while executing nothing at all.
    Duplicated is the same failure one level down — silently taking the first
    leaves the second copy unexecuted and free to rot.
    """
    with pytest.raises(AssertionError) as excinfo:
        _marked_command(markdown_text, SHAPE_LOOKUP_MARKER, SHAPE_LOOKUP_LABEL)

    message = str(excinfo.value)
    assert SHAPE_LOOKUP_MARKER in message, case
    assert "CLAUDE.md" in message, case


def test_marked_command_is_immune_to_inline_code_in_the_begin_comment():
    """Prose inline code inside the marked slice is never extracted.

    An extractor keyed on "the first backtick span after the begin marker"
    would return ``.taskmaster/tasks.db`` here — a plausible-looking string,
    so the mistake would survive review and then be executed as a command.
    """
    assert (
        _marked_command(_DECOY_DOC, SHAPE_LOOKUP_MARKER, SHAPE_LOOKUP_LABEL)
        == _HAPPY_COMMAND
    )


def _documented_tool_argv(command):
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
        f"the command inside the {SHAPE_LOOKUP_MARKER!r} marker in CLAUDE.md is "
        f"not a bare run of the shape tool (task 5330): {command!r} tokenises to "
        f"{argv!r}, expected exactly [python|python3, {_TOOL_RELATIVE_PATH!r}]. "
        f"The convention is that an agent asks the store and reads the answer — "
        f"a hand-written sqlite query in its place is the confusion this closes."
    )
    assert argv[1] == _TOOL_RELATIVE_PATH, (
        f"the {SHAPE_LOOKUP_MARKER!r} command in CLAUDE.md runs {argv[1]!r}, "
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


def test_documented_store_shape_lookup_reports_the_stores_real_shape(tmp_path):
    """CLAUDE.md's recipe must actually print the shape of the store it finds.

    Executed, not string-compared: this certifies the recipe WORKS, which is
    the claim CLAUDE.md is making to every agent that reads it.
    """
    command = _marked_command(
        CLAUDE_MD_PATH.read_text(encoding="utf-8"),
        SHAPE_LOOKUP_MARKER,
        SHAPE_LOOKUP_LABEL,
    )
    argv = _documented_tool_argv(command)
    root = _seeded_project_root(tmp_path)

    try:
        completed = subprocess.run(
            argv,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_SECS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the {SHAPE_LOOKUP_MARKER!r} command documented in CLAUDE.md did not "
            f"finish within {_RUN_TIMEOUT_SECS}s (task 5330); argv: {argv!r}"
        )

    assert completed.returncode == 0, (
        f"the {SHAPE_LOOKUP_MARKER!r} command documented in CLAUDE.md exited "
        f"{completed.returncode} (task 5330) — agents are being handed a recipe "
        f"that does not work.\n"
        f" argv: {argv!r}\n cwd: {root}\n"
        f" stdout: {completed.stdout.strip()!r}\n"
        f" stderr: {completed.stderr.strip()!r}"
    )

    out = completed.stdout
    assert _FIXTURE_TABLE in out, (
        f"the {SHAPE_LOOKUP_MARKER!r} command exited 0 without naming the "
        f"{_FIXTURE_TABLE!r} table of the store at {root} (task 5330): {out!r}"
    )
    missing = [column for column in _FIXTURE_COLUMNS if column not in out]
    assert not missing, (
        f"the {SHAPE_LOOKUP_MARKER!r} command did not report {missing!r} for the "
        f"store this test built (task 5330) — it is meant to answer 'what "
        f"columns does this store have', so a report missing them would leave an "
        f"agent guessing exactly as before. Full output: {out!r}"
    )
