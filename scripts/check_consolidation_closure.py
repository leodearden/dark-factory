#!/usr/bin/env python3
"""Run the consolidation-gate closure check by hand (task 3112).

The gate task's own ``set_task_status(..., 'done')`` already re-runs this
check and refuses when the cluster is not in the PRD §3 Option-C end state
(``TaskInterceptor._consolidation_closure_error``). That refusal is the
enforcement; this script is the same question asked *before* you try, so a
curator can see the offending ids without burning a refused transition.

It is deliberately a THIN wrapper, not a second opinion. The verdict comes
from ``fused_memory.reconciliation.consolidation_gate.evaluate_closure`` —
the same pure predicate the seam calls — and the scroll cap is imported
from ``TaskInterceptor._CONSOLIDATION_SCROLL_LIMIT`` rather than restated
(INV-5). Those two imports are what make "the same mechanical check"
literally true: a CLI with its own predicate, or merely its own cap, could
report ``closed`` on a view the seam would reject as ``scroll_incomplete``,
which is exactly the false reassurance this whole gate exists to prevent.

This replaces the old prose "re-search before merging" guard with something
a curator can actually execute.

EXIT-CODE CONTRACT (following the convention in
``predicate_contradiction.py::render_predicate_contradiction_section`` —
reasons are printed on stdout, but a caller branches on the exit code
alone):

  - exit 0  — CLOSED. The live same-topic cluster is in the Option-C end
              state: exactly one canonical over its short single-claim
              peers, nothing claimed-absorbed still live, and the scroll
              that established it was complete.
  - exit 1  — NOT CLOSED. The store answered and the cluster is malformed.
              Every offending id is named on stdout. This is the verdict
              the seam would refuse the ``done`` transition with.
  - exit 2  — COULD NOT CHECK (usage or infra). Bad arguments, no such
              task, an unreadable ``tasks.db``, a task that is not a
              consolidation gate, a gate block with no usable ``topic``, or
              a store that did not answer.

Exit 2 is a SEPARATE code on purpose. An unreachable Qdrant and a
genuinely malformed cluster are different facts and must not share an exit
code: collapsing them would let a store outage read as "your consolidation
is broken" (or, worse under a `!= 0` reading, let a real refusal read as a
transient blip). Note the asymmetry with the seam, which is deliberate: the
seam is fail-CLOSED and converts an unreadable store into a refusal,
because a gate whose job is refuting a false closure claim must not pass
when it cannot see. This script is a diagnostic, not an authority, so it
reports "could not check" as its own outcome. Either way the cluster is not
closeable — exit 2 never means "go ahead".

Usage
-----
  # Check a filed gate task by id (reads .taskmaster/tasks/tasks.db read-only).
  python scripts/check_consolidation_closure.py --task-id 3092

  # Check an arbitrary metadata blob (e.g. one you are about to file).
  python scripts/check_consolidation_closure.py --metadata-json "$(cat gate.json)"
  python scripts/check_consolidation_closure.py --metadata-file gate.json

  # Machine-readable verdict for a wrapper script.
  python scripts/check_consolidation_closure.py --task-id 3092 --json

READ-ONLY. Every database handle this script opens is a read-only SQLite
URI and the only store call is a metadata scroll; it never writes a task,
a memory or a status.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

# IMPORT-RESOLUTION CONTRACT (same idiom, and the same precedence argument,
# as scripts/scan_task_toolcall_leaks.py:92-116): bind `fused_memory` and
# `shared` to THIS checkout via a __file__-relative path. The fused-memory
# editable install is an ordinary .pth entry, so sys.path ORDER decides the
# winner — without these inserts a worktree run would silently check its
# cluster against the MAIN checkout's copy of the predicate.
_FM_SRC = Path(__file__).resolve().parent.parent / "fused-memory" / "src"
if str(_FM_SRC) not in sys.path:
    sys.path.insert(0, str(_FM_SRC))

_SHARED_SRC = Path(__file__).resolve().parent.parent / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

# `_task_db_scan` is a flat sibling in scripts/ and resolves solely because a
# DIRECTLY-EXECUTED script puts its own directory at sys.path[0] — so never
# invoke this via `python -m`. It is the single home for the tasks.db path.
from _task_db_scan import tasks_db_path  # noqa: E402
from fused_memory.middleware.task_interceptor import TaskInterceptor  # noqa: E402
from fused_memory.reconciliation.consolidation_gate import (  # noqa: E402
    EXIT_CLOSED,
    EXIT_NOT_CLOSED,
    GATE_METADATA_KEY,
    evaluate_closure,
)

#: "I could not run the check." Distinct from EXIT_NOT_CLOSED by design —
#: see the EXIT-CODE CONTRACT above.
EXIT_USAGE = 2

#: The seam's cap, imported rather than restated: the CLI must look at
#: exactly as much of the cluster as the seam does, or the two can disagree
#: about `scroll_incomplete` on the very cluster the operator is checking.
SCROLL_LIMIT = TaskInterceptor._CONSOLIDATION_SCROLL_LIMIT

DEFAULT_PROJECT_ROOT = "/home/leo/src/dark-factory"
DEFAULT_PROJECT_ID = "dark_factory"


class UsageError(Exception):
    """Anything that makes the check unrunnable — reported as EXIT_USAGE."""


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O) — the testable band
# --------------------------------------------------------------------------- #


def extract_gate_block(metadata: Any) -> dict:
    """Return the gate block from a task's *metadata*, or raise UsageError.

    *metadata* may be a dict or a JSON string: normalised through the seam's
    own ``TaskInterceptor._extract_metadata_dict`` (a staticmethod), never a
    hand-rolled ``json.loads``, so the CLI and the seam apply one shape
    policy to the same blob.
    """
    meta = TaskInterceptor._extract_metadata_dict(metadata)
    if not isinstance(meta, dict):
        raise UsageError(
            "metadata could not be parsed as a dict, so there is no gate "
            "block to check."
        )
    block = meta.get(GATE_METADATA_KEY)
    if not isinstance(block, dict):
        raise UsageError(
            f"metadata carries no `{GATE_METADATA_KEY}` block, so this is not "
            "a consolidation gate. The seam is dormant for such a task and "
            "would let it close untouched; there is nothing here to check."
        )
    topic = block.get("topic")
    if not isinstance(topic, str) or not topic:
        raise UsageError(
            f"gate_topic_missing: the `{GATE_METADATA_KEY}` block carries no "
            "usable `topic`, so the live cluster cannot be located. The seam "
            "REFUSES this shape rather than passing it — fix the block."
        )
    return block


def render_human(verdict: Any, *, scroll: dict) -> str:
    """Human-readable verdict: the headline, then every offending id."""
    lines = [
        f"topic:  {verdict.topic}",
        f"closed: {verdict.closed}",
        (
            "scroll: returned={returned} total={total} truncated={truncated} "
            "available={available} limit={limit}".format(**scroll)
        ),
        "",
        verdict.message,
    ]
    if verdict.reasons:
        lines.append("")
        lines.append(f"reasons ({len(verdict.reasons)}):")
        for reason in verdict.reasons:
            ids = ", ".join(reason.get("ids") or []) or "-"
            lines.append(f"  [{reason['code']}] ids: {ids}")
            detail = (reason.get("detail") or "").strip()
            if detail:
                lines.append(f"      {detail}")
    if verdict.waived:
        lines.append("")
        lines.append(f"waived ({len(verdict.waived)}):")
        for waiver in verdict.waived:
            lines.append(
                f"  {waiver.get('id')} — {waiver.get('note')} "
                f"(recorded_by={waiver.get('recorded_by')})"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# I/O bindings
# --------------------------------------------------------------------------- #


def load_task_metadata(project_root: str, task_id: str, tag: str) -> Any:
    """Read one task's raw metadata column from tasks.db, READ-ONLY."""
    db_path = tasks_db_path(project_root)
    if not db_path.exists():
        raise UsageError(f"no tasks.db at {db_path}")
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise UsageError(f"could not open {db_path} read-only: {exc}") from exc
    try:
        row = con.execute(
            "SELECT metadata FROM tasks WHERE tag = ? AND id = ?",
            (tag, str(task_id)),
        ).fetchone()
    except sqlite3.Error as exc:
        raise UsageError(f"could not read task {task_id} from {db_path}: {exc}") from exc
    finally:
        con.close()
    if row is None:
        raise UsageError(f"no task id={task_id} under tag={tag} in {db_path}")
    return row[0]


async def scroll_cluster(memory: Any, project_id: str, topic: str) -> dict:
    """Bind the real fused-memory scroll for one topic.

    COST ORDERING copies the seam: count first, scroll only on a non-zero
    count, and DISCLOSE truncation — so the common path is cheap and a
    capped scroll never reads as complete.
    """
    filters = {"topic": topic}
    total = await memory.count_memories_by_metadata(project_id, filters)
    members = (
        []
        if total == 0
        else list(
            await memory.get_memories_by_metadata(project_id, filters, limit=SCROLL_LIMIT)
        )
    )
    return {
        "members": members,
        "total": total,
        "truncated": len(members) >= SCROLL_LIMIT or total > len(members),
        "available": True,
    }


def build_memory_service() -> Any:
    """``MemoryService(config)`` — the exact object the seam's scroll is bound to.

    ``initialize()`` is deliberately NOT awaited: ``MemoryService.__init__``
    already constructs ``self.mem0 = Mem0Backend(config)``, whose Qdrant
    client is lazy, and ``get_memories_by_metadata`` /
    ``count_memories_by_metadata`` go straight through it. So this read-only
    check needs neither FalkorDB/Graphiti nor an embedder / OPENAI_API_KEY —
    the same argument ``census_memory_metadata.py::_build_backend`` makes.
    """
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    return MemoryService(FusedMemoryConfig())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the consolidation-gate closure predicate against a live "
            "topic cluster. Exit 0 = closed, 1 = not closed, 2 = could not check."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task-id", help="gate task id to read from tasks.db")
    source.add_argument("--metadata-json", help="a task metadata blob as JSON")
    source.add_argument(
        "--metadata-file", help="path to a file holding a task metadata JSON blob"
    )
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help="project root owning tasks.db (default: %(default)s)",
    )
    parser.add_argument(
        "--tag", default="master", help="taskmaster tag (default: %(default)s)"
    )
    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help="memory project_id to scroll (default: %(default)s)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="emit the verdict as JSON"
    )
    return parser


def _emit(payload: dict, *, as_json: bool, human: str) -> None:
    print(json.dumps(payload, indent=2, default=str) if as_json else human)


async def run(args: argparse.Namespace, *, memory: Any = None) -> int:
    try:
        if args.task_id:
            raw_metadata: Any = load_task_metadata(
                args.project_root, args.task_id, args.tag
            )
        elif args.metadata_file:
            raw_metadata = Path(args.metadata_file).read_text()
        else:
            raw_metadata = args.metadata_json
        block = extract_gate_block(raw_metadata)
    except UsageError as exc:
        _emit(
            {"checked": False, "exit_code": EXIT_USAGE, "error": str(exc)},
            as_json=args.as_json,
            human=f"COULD NOT CHECK: {exc}",
        )
        return EXIT_USAGE
    except OSError as exc:
        _emit(
            {"checked": False, "exit_code": EXIT_USAGE, "error": str(exc)},
            as_json=args.as_json,
            human=f"COULD NOT CHECK: {exc}",
        )
        return EXIT_USAGE

    topic = block["topic"]
    if memory is None:
        memory = build_memory_service()

    store_error: str | None = None
    try:
        scrolled = await scroll_cluster(memory, args.project_id, topic)
    except Exception as exc:  # noqa: BLE001 — any store failure is "could not check"
        store_error = f"{type(exc).__name__}: {exc}"
        scrolled = {"members": [], "total": None, "truncated": False, "available": False}

    verdict = evaluate_closure(
        block,
        members=scrolled["members"],
        scroll_total=scrolled["total"],
        scroll_truncated=scrolled["truncated"],
        scroll_available=scrolled["available"],
    )
    scroll_facts = {
        "returned": len(scrolled["members"]),
        "total": scrolled["total"],
        "truncated": scrolled["truncated"],
        "available": scrolled["available"],
        "limit": SCROLL_LIMIT,
    }
    payload = {
        "checked": store_error is None,
        "closed": verdict.closed,
        "topic": verdict.topic,
        "message": verdict.message,
        "reasons": [dict(r) for r in verdict.reasons],
        "waived": [dict(w) for w in verdict.waived],
        "scroll": scroll_facts,
    }
    if store_error is not None:
        # The predicate still ran, so its `scroll_unavailable` reason is
        # printed verbatim — one text, not a CLI-local paraphrase. Only the
        # EXIT CODE differs from the verdict's, so a store outage can never
        # be misread as a real not-closed verdict.
        payload["error"] = store_error
        payload["exit_code"] = EXIT_USAGE
        _emit(
            payload,
            as_json=args.as_json,
            human=(
                f"COULD NOT CHECK: the store did not answer ({store_error}).\n\n"
                + render_human(verdict, scroll=scroll_facts)
            ),
        )
        return EXIT_USAGE

    payload["exit_code"] = verdict.exit_code
    _emit(payload, as_json=args.as_json, human=render_human(verdict, scroll=scroll_facts))
    assert verdict.exit_code in (EXIT_CLOSED, EXIT_NOT_CLOSED)
    return verdict.exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
