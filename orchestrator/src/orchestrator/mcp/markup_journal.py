"""The DURABLE journal a boundary guard's per-call markup facts land in.

Task 4744. ``shared.mcp_markup_middleware.MarkupGuardMiddleware`` emits one
``markup_detected`` fact on EVERY outcome (unrepairable, rejected, repaired),
and that record is the only thing anywhere that names WHICH call leaked. On an
orchestrator MCP boundary it reached exactly one place: a ``logger.warning``
inside a per-agent STDIO SUBPROCESS whose stderr the CLI agent that spawned it
consumes. Measured 2026-08-25::

    journalctl --user --since 2026-08-22 | grep 'markup guard:'   ->  0 lines

against 35 REAL plan-tools rejections in ``data/orchestrator/agent-transcripts/``
over the same span. The contrast is not neglect: fused-memory's identical
middleware DOES reach journald because that server runs under
``systemd --user``, while plan-tools does not run under anything.

The consequence is what this module exists to end. A markup STORM files an
escalation telling an operator to identify the leaking caller from the guard's
own log lines — an instruction that cannot be followed, because the lines were
written to a stream nobody retains. Anyone following it correctly concludes "no
evidence" and is wrong.

So the fact channel gets a consumer: one append-only JSONL under
``<project_root>/data/orchestrator/markup-guard/<server_label>.jsonl``, one line
per event, carrying the identity the storm summary structurally cannot.

WHAT IS AND IS NOT IN A LINE. Names, identity and the matched pattern — never
the raw caller payload. The verbatim payload already has exactly one owner, the
residue escalation (``markup_sink.residue_detail``), which is the only surviving
copy of data the caller may never be able to resend. A second, weaker copy of it
here would be the INV-5 duplication this PRD rules against, and would put an
unbounded caller payload inside a file whose concurrency safety depends on a
bounded write. This journal answers "who leaked, from which tool, into which
parameter, when" — the question the storm record asks a human to answer and
gives them no way to.

Shape inherited from ``markup_sink.make_escalation_sink``: never raises, does
its blocking work under ``asyncio.to_thread``, memoizes successes only, returns
a locator string or ``None``. The two injected channels on one boundary behave
identically under failure, so a reader who has read one has read both.

PRD: ``plans/toolcall-markup-containment-prd.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orchestrator.mcp import markup_sink

logger = logging.getLogger(__name__)

#: Where the journal lives under project_root. Beside
#: ``data/orchestrator/agent-transcripts``, in the tree an operator already
#: greps, and under the repo-root-gitignored ``/data/`` so a journal line can
#: never contaminate a diff or a verify.
#:
#: THIS CONSTANT IS THE SINGLE OWNER of the path an operator is told to read:
#: the storm escalation's ``attribution_source`` is composed from it rather than
#: respelling it, so the instruction and the artifact cannot drift apart.
MARKUP_JOURNAL_DIRNAME = 'data/orchestrator/markup-guard'


def journal_path(project_root: Path, server_label: str) -> Path:
    """The one file this server's markup facts are appended to.

    ONE file per server, not one per record. The per-record shape
    ``data/orchestrator/retry_cap_reports`` uses would turn "who leaked in this
    window" back into a directory walk, which is the shape of failure this
    module exists to end: an operator gets a single exact path to grep or pipe
    to ``jq``.

    No rotation, deliberately. Measured volume is 35 lines over three days
    across the whole fleet, and a dated or rotated filename would put a moving
    part into the one path an operator is told to open. If volume ever changes,
    adding rotation is a later, separable decision.
    """
    return project_root / MARKUP_JOURNAL_DIRNAME / f'{server_label}.jsonl'


def make_fact_journal(
    *,
    worktree: Path,
    server_label: str,
    subject_task_id: Callable[[], str],
    resolve_root: Callable[[Path], Path | None] = markup_sink.resolve_project_root,
    now: Callable[[], float] = time.time,
) -> Callable[[dict[str, Any]], Awaitable[str | None]]:
    """Build the ``fact_sink`` a boundary guard journals its facts through.

    Returns the journal path as a locator string — the same contract
    ``make_escalation_sink`` keeps with the queued record's id, so the two
    injected channels report their result the same way.

    *subject_task_id* is a THUNK, not a value, for the reason ``markup_sink``
    records: the middleware cannot resolve the leaking task
    (``MarkupGuardMiddleware._identity`` reads identity off the call's own
    arguments, and no tool on these servers declares ``agent_id`` /
    ``project_root`` / ``project_id``), and the attribution a concrete site can
    offer may not exist yet at ``create_server`` time — a plan-tools
    ``create_plan`` refused before any plan exists is the case. So it is
    evaluated PER RECORD, on the worker thread, never at wiring time.

    *resolve_root* defaults to ``--git-common-dir`` resolution, which from a
    task worktree names the MAIN checkout. That is the point: a worktree-local
    artifact dies with the lane at ``git worktree remove --force`` and nothing
    ever reads it — the documented reason ``TaskArtifacts.write_markup_residue``
    is wired only as a last resort and never as a channel.

    ASYNC, with the blocking work on a worker thread: the middleware calls this
    from inside the server's event loop, and one record can run a
    ``git rev-parse`` subprocess and a filesystem write.
    """
    project_root: Path | None = None

    def _subject() -> str:
        """The attribution ladder, applied per record.

        Same ladder ``markup_sink.make_escalation_sink`` applies: the site's own
        answer, then the worktree directory name, then an explicit
        "unattributed" — never ``None`` and never absent, because a consumer
        must be able to tell "nobody knows" from "that emitter forgot the key".
        """
        return subject_task_id() or worktree.name or markup_sink.MARKUP_UNATTRIBUTED_SUBJECT

    def append(record: dict[str, Any]) -> str | None:
        """The blocking body, run on a worker thread."""
        nonlocal project_root
        if project_root is None:
            project_root = resolve_root(worktree)
            if project_root is None:
                return None

        path = journal_path(project_root, server_label)
        line = json.dumps(
            {
                'ts': datetime.fromtimestamp(now(), tz=UTC).isoformat(),
                'server': server_label,
                'subject_task_id': _subject(),
                'worktree': str(worktree),
                'pid': os.getpid(),
                **record,
            },
            # Deterministic key order is what makes the file greppable BY
            # FIELD, and matches the encoding the existing fact emitters
            # (verdict_tools._emit_markup_fact, escalation.server's twin) use —
            # so a journal line and a log line are the same bytes plus this
            # module's envelope.
            sort_keys=True,
        ) + '\n'

        path.parent.mkdir(parents=True, exist_ok=True)
        # ONE ``os.write`` on an O_APPEND fd, and NEVER a read-modify-write.
        #
        # Many of these servers run at once — one stdio subprocess per agent,
        # across concurrent tasks — and all of them append to this one file. A
        # "read the file, concatenate, rewrite" implementation would silently
        # destroy the other processes' lines, which is the same class of data
        # loss the whole markup guard exists to prevent. O_APPEND makes the
        # seek-and-write atomic with no cross-process lock to take, but that
        # atomicity is guaranteed only for a BOUNDED write — so the record is
        # capped rather than trusted to be small.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode('utf-8'))
        finally:
            os.close(fd)
        return str(path)

    async def journal(record: dict[str, Any]) -> str | None:
        return await asyncio.to_thread(append, record)

    return journal
