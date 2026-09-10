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
raises. A declaration that carries no affinity rule of its own — none at all,
or a NUMERIC-ish one that can come back as either int or float — is reported
as unconstrained rather than guessed.

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

import sqlite3
import subprocess
from pathlib import Path
from typing import NamedTuple

from _task_db_scan import tasks_db_path

_GIT_TIMEOUT_SECS = 30


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
    except OSError as exc:
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
_AFFINITY_RULES: tuple[tuple[tuple[str, ...], type], ...] = (
    (("INT",), int),
    (("CHAR", "CLOB", "TEXT"), str),
    (("BLOB",), bytes),
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
    names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]
    return tuple(Table(name=name, columns=_table_columns(conn, name)) for name in names)
