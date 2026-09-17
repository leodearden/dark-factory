#!/usr/bin/env python3
"""Report the true shape of a task store — ask the database, never guess.

Read-only forensics on ``.taskmaster/tasks/tasks.db``. Direct sqlite reads of
the live store are the endorsed house convention (see
``scripts/merge_lane_throughput.py::_connect_ro``, the hotspot-survey skill,
and the escalation-watcher's MCP-down fallback); what was missing is any way
to learn the store's SHAPE before executing a query against it. Guessing it
produced a recurring, unactionable error class — ``no such column:
created_at``, ``no such table: tasks``, and ``'int' object has no attribute
'isdigit'`` for a task id read straight out of an INTEGER column.

This tool therefore states no column names of its own. It introspects whatever
store it is pointed at and prints what is actually there, which is the only
answer that cannot rot: four hand-maintained copies of the tasks column list
already exist in this repo and one of them is stale.

SQLITE IS DYNAMICALLY TYPED, so the reported python type is a statement about
AFFINITY — what a value is coerced to on the way in — and not a guarantee
about every value already stored. Affinity is nonetheless the answer to the
question that keeps being asked wrong: ``tasks.id`` is declared INTEGER, so a
task id read out of the store arrives as ``int`` and ``.isdigit()`` on it
raises. A declaration that pins no single python type is reported as
unconstrained rather than guessed, and three shapes land there: no declaration
at all, a NUMERIC-ish one that can come back as either int or float, and BLOB —
whose affinity is precisely the absence of coercion, so a str written to a BLOB
column reads back as str (measured) and any claim of ``bytes`` would be the
authoritative wrong answer this tool exists to retire.

WHY IT RESOLVES THE MAIN CHECKOUT ITSELF, rather than importing
``fused_memory.models.scope.resolve_main_checkout`` — a knowing duplication,
recorded so it is not read as an oversight. That function is the right one and
its docstring even states the governing fact, but importing it drags pydantic
and ``fused_memory.utils.validation`` into a script whose whole value is
running from a cold shell during an incident. The duplicated part is a
subprocess call plus first-entry selection, not the caching and sanity
checking that function layers on top.
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from _task_db_scan import (
    TABLE_NAMES_SQL,
    TaskDbUnreadable,
    connect_ro,
    tasks_db_path,
)

_GIT_TIMEOUT_SECS = 30

EXIT_OK = 0
# "I could not read a store" — 3, matching _task_db_scan.AUDIT_EXIT_NOTHING_AUDITED,
# whose semantics are the same "nothing was read, so this is not a clean run".
# NOT 2: argparse spends that code on its own usage errors (measured), so a
# scripted caller reading 2 could not tell a mistyped flag from an unreadable
# store.
EXIT_UNREADABLE = 3


class MainCheckoutUnresolved(Exception):
    """No git working tree at *start*, so the live store cannot be located.

    Carries the resolved :attr:`start` and git's own :attr:`detail` as fields.
    """

    def __init__(self, start: Path, detail: str) -> None:
        self.start = start
        self.detail = detail
        super().__init__(
            f"{start}: cannot resolve a main git checkout from here — {detail}. "
            f"The live task store is the MAIN checkout's "
            f".taskmaster/tasks/tasks.db; run from inside the project, or name "
            f"the store explicitly."
        )


def resolve_live_db_path(start: str | Path) -> Path:
    """The live task store of the git checkout *start* stands in.

    ``.taskmaster/`` is not tracked in git, so it exists only at the MAIN
    checkout — never inside a worktree. ``git worktree list --porcelain``
    names that checkout on its first entry whichever tree it is run from,
    which is why this asks git rather than walking up looking for ``.git``.

    Returns the path whether or not a store is there; :func:`connect_ro` owns
    the refusal, so one reader never gets two different diagnoses for a
    missing store.
    """
    resolved_start = Path(start).resolve()
    try:
        listed = subprocess.run(
            ["git", "-C", str(resolved_start), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # SubprocessError, not just OSError: a git wedged past the timeout
        # raises TimeoutExpired, which is NOT an OSError — and a loaded box
        # with a contended .git is exactly the incident this tool is for, so
        # that shape must arrive as a diagnosis rather than as a traceback.
        raise MainCheckoutUnresolved(resolved_start, str(exc)) from exc

    if listed.returncode != 0:
        raise MainCheckoutUnresolved(resolved_start, listed.stderr.strip())

    for line in listed.stdout.splitlines():
        if line.startswith("worktree "):
            return tasks_db_path(line.removeprefix("worktree ").strip())

    raise MainCheckoutUnresolved(
        resolved_start, "`git worktree list --porcelain` named no worktree at all"
    )


class Column(NamedTuple):
    """One column, exactly as the database describes it.

    *pk* is a POSITION, not a flag: 0 for a column outside the primary key,
    otherwise its 1-based place within a possibly-composite one.

    *python_type* is the type values of this column arrive as, derived from
    *declared_type* by sqlite's affinity rules, or None when the declaration
    constrains nothing (see this module's docstring). It is the field that
    answers "can I call a string method on this?" before the query is written.
    """

    name: str
    declared_type: str
    notnull: bool
    pk: int
    python_type: type | None


class Table(NamedTuple):
    """One table and its columns, in declaration order."""

    name: str
    columns: tuple[Column, ...]


# sqlite's affinity rules, in the order it applies them, keyed on the
# SUBSTRINGS the rules are actually written in terms of. Matching on substrings
# rather than on a table of exact declarations is what lets an unfamiliar
# spelling — VARCHAR(20), DOUBLE, BIGINT — be classified instead of crashing or
# silently defaulting to str.
#
# BLOB resolves to None — unconstrained — rather than to bytes, because BLOB
# affinity is sqlite's one rule that coerces NOTHING. It stays in the table,
# and in this position, because the ORDER is load-bearing: sqlite applies the
# BLOB rule before the REAL one, so dropping the rung would let a declaration
# carrying both spellings fall through to float.
_AFFINITY_RULES: tuple[tuple[tuple[str, ...], type | None], ...] = (
    (("INT",), int),
    (("CHAR", "CLOB", "TEXT"), str),
    (("BLOB",), None),
    (("REAL", "FLOA", "DOUB"), float),
)


def _python_type(declared_type: str) -> type | None:
    """The type *declared_type* coerces values to, or None if it constrains none."""
    declared = declared_type.upper()
    for needles, python_type in _AFFINITY_RULES:
        if any(needle in declared for needle in needles):
            return python_type
    return None


def _table_columns(conn: sqlite3.Connection, table: str) -> tuple[Column, ...]:
    return tuple(
        Column(
            name=name,
            declared_type=declared,
            notnull=bool(notnull),
            pk=pk,
            python_type=_python_type(declared),
        )
        for _cid, name, declared, notnull, _default, pk in conn.execute(
            f'PRAGMA table_info("{table}")'
        )
    )


def introspect(conn: sqlite3.Connection) -> tuple[Table, ...]:
    """Every table in *conn*, with the columns the database reports for it.

    Asks the store and nothing else, so it is equally right about a live
    tasks.db, a fixture, and a store whose DDL has since moved on. Produces
    records; rendering them is :func:`render`'s job.

    sqlite's own bookkeeping tables (``sqlite_master``, ``sqlite_sequence``…)
    are left out: they are the same in every store and never the subject of
    the question being asked.
    """
    names = [row[0] for row in conn.execute(f"{TABLE_NAMES_SQL} ORDER BY name")]
    return tuple(Table(name=name, columns=_table_columns(conn, name)) for name in names)


def _rendered_python_type(column: Column) -> str:
    return column.python_type.__name__ if column.python_type is not None else "unconstrained"


def _rendered_flags(column: Column) -> str:
    marks = []
    if column.pk:
        marks.append(f"pk{column.pk}")
    if column.notnull:
        marks.append("not null")
    return " ".join(marks)


def render(tables: Sequence[Table]) -> str:
    """One aligned block per table: column, declared type, python type, flags."""
    blocks = []
    for table in tables:
        name_width = max((len(c.name) for c in table.columns), default=0)
        type_width = max((len(c.declared_type) for c in table.columns), default=0)
        lines = [f"{table.name} — {len(table.columns)} columns"]
        lines += [
            f"  {column.name:<{name_width}}  {column.declared_type:<{type_width}}  "
            f"-> {_rendered_python_type(column)} {_rendered_flags(column)}".rstrip()
            for column in table.columns
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print the tables, columns, declared types and python value types of "
            "a task store — so a forensic query is written against the shape the "
            "store actually has."
        ),
        epilog=(
            "Strictly READ-ONLY: the store is opened mode=ro and this tool never "
            "writes. Task writes go through the fused-memory MCP tools, which "
            "emit the reconciliation events a direct sqlite write would skip."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--db", help="a store to read; default: the live one for this checkout")
    where.add_argument(
        "--project-root", help="a project root, whose .taskmaster/tasks/tasks.db is read"
    )
    return parser


def _resolve_db_path(args: argparse.Namespace) -> Path:
    """Where to read, from the arguments — ``is not None``, never truthiness.

    An empty ``--db ''`` is a path the reader NAMED, so it must reach
    :func:`connect_ro` and earn a refusal naming it. Branching on truthiness
    would silently read the live store instead and report a confident answer
    about a different database than the one asked for.
    """
    if args.db is not None:
        return Path(args.db)
    if args.project_root is not None:
        return tasks_db_path(args.project_root)
    return resolve_live_db_path(Path.cwd())


def main(argv: Sequence[str] | None = None) -> int:
    """Print the store's shape, or diagnose why it could not be read.

    Only this module's own typed refusals are caught, and each exits non-zero
    with its message on STDERR and NOTHING on stdout: reporting an empty
    schema would tell the reader the store has no tables.
    """
    args = _build_parser().parse_args(argv)
    try:
        db_path = _resolve_db_path(args)
        conn = connect_ro(db_path)
    except (MainCheckoutUnresolved, TaskDbUnreadable) as refusal:
        print(refusal, file=sys.stderr)
        return EXIT_UNREADABLE

    try:
        tables = introspect(conn)
    finally:
        conn.close()

    print(db_path)
    print(render(tables))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
