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

import subprocess
from pathlib import Path

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
