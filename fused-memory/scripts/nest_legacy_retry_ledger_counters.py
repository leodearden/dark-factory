#!/usr/bin/env python3
"""Nest the legacy top-level infra-resume counters under ``metadata.retry_ledger``.

One-shot for task 7135. Some stored task blobs still carry
``consecutive_infra_resume_failures`` and ``last_infra_resume_iteration_count``
at the TOP level, and ``shared.task_metadata`` lifts them into ``retry_ledger``
at parse time. This script moves them in storage with the lift's own semantics
(a nested value wins and the shadowed top-level copy is dropped; a non-dict
ledger is refused), so every migrated blob parses to the ``RetryLedger`` its
readers already see and the lift can be deleted.

It talks to the running fused-memory MCP server: targets come from
``find_tasks_by_metadata``, each raw blob from ``get_task``, and each write is
a metadata-only ``update_task(metadata_mode='replace')`` checked by a fresh
read-back. It is a dry run by default. ``--apply`` requires a ``--backup-path``
that does not exist yet; every pre-write blob is saved there before the first
write. A re-run finds no targets, so it doubles as the re-measure.

Usage:
    python fused-memory/scripts/nest_legacy_retry_ledger_counters.py \\
        --project-root /home/leo/src/dark-factory [--project-root ...] \\
        [--server-url http://127.0.0.1:8002] [--apply --backup-path PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from shared.mcp_envelope import write_failure_reason

from fused_memory.utils.script_loader import load_script

DEFAULT_SERVER = 'http://127.0.0.1:8002'
COUNTER_KEYS = ('consecutive_infra_resume_failures', 'last_infra_resume_iteration_count')
# update_task strips these from every incoming payload, so a replace cannot
# keep them: fused-memory/src/fused_memory/backends/sqlite_task_backend.py::
# _RESERVED_METADATA_CONTROL_KEYS and ::_MACHINE_AUTHORED_METADATA_KEYS.
REPLACE_STRIPPED_KEYS = frozenset(
    {'append', 'metadata_mode', 'pending_since', 'pending_since_backfilled'},
)


class ToolClient(Protocol):
    async def call_tool(self, name: str, arguments: dict) -> Any: ...


class MalformedRetryLedgerError(ValueError):
    """``retry_ledger`` is present but not a dict, so nothing can be nested into it."""


class UnwritableBlobError(ValueError):
    """The blob carries a key a replace write would drop, so it is left unmigrated."""


class ToolReplyError(RuntimeError):
    """A tool answered with an error payload or an unusable shape — never "no match"."""


class BackupRefusedError(RuntimeError):
    """``--apply`` cannot proceed without a fresh, writable backup path."""


@dataclass(frozen=True)
class NestingPlan:
    after: dict
    lifted: tuple[str, ...]
    shadowed: tuple[str, ...]


@dataclass(frozen=True)
class Target:
    project_root: str
    task_id: str
    before: dict
    plan: NestingPlan

    @property
    def label(self) -> str:
        return f'{self.project_root} #{self.task_id}'


def nest_counters(meta: dict) -> NestingPlan | None:
    """Plan moving the top-level counters of *meta* into its ``retry_ledger``.

    ``None`` when *meta* carries no top-level counter. *meta* is never mutated.
    """
    present = tuple(key for key in COUNTER_KEYS if key in meta)
    if not present:
        return None
    ledger = meta.get('retry_ledger')
    if ledger is None:
        ledger = {}
    elif not isinstance(ledger, dict):
        raise MalformedRetryLedgerError(f'retry_ledger is not a dict: {ledger!r}')
    lifted = tuple(key for key in present if key not in ledger)
    shadowed = tuple(key for key in present if key in ledger)
    after = {key: copy.deepcopy(value) for key, value in meta.items() if key not in COUNTER_KEYS}
    after['retry_ledger'] = {**copy.deepcopy(ledger), **{key: meta[key] for key in lifted}}
    return NestingPlan(after=after, lifted=lifted, shadowed=shadowed)


def build_update_payload(task_id: str, project_root: str, meta: dict) -> dict:
    return {
        'id': task_id,
        'project_root': project_root,
        'metadata': json.dumps(meta),
        'metadata_mode': 'replace',
    }


def _require_ok_reply(tool: str, project_root: str, reply: Any) -> dict:
    if not isinstance(reply, dict):
        raise ToolReplyError(f'{tool} on {project_root}: reply is {type(reply).__name__}')
    if 'error' in reply:
        raise ToolReplyError(
            f'{tool} on {project_root}: {reply.get("error_type")}: {reply["error"]}'
        )
    return reply


async def _ids_carrying(client: ToolClient, project_root: str, key: str, *, page_size: int) -> set[int]:
    found: set[int] = set()
    offset = 0
    while True:
        reply = _require_ok_reply('find_tasks_by_metadata', project_root, await client.call_tool(
            'find_tasks_by_metadata',
            {'project_root': project_root, 'key': key, 'limit': page_size, 'offset': offset},
        ))
        found.update(int(row['task_id']) for row in reply['results'])
        pagination = reply['pagination']
        if not pagination['has_more']:
            return found
        if not pagination['returned']:
            raise ToolReplyError(f'find_tasks_by_metadata on {project_root}: has_more on an empty page')
        offset += pagination['returned']


async def find_target_ids(client: ToolClient, project_root: str, *, page_size: int) -> list[str]:
    """Ids of every task in *project_root* carrying either counter at top level."""
    found: set[int] = set()
    for key in COUNTER_KEYS:
        found |= await _ids_carrying(client, project_root, key, page_size=page_size)
    return [str(task_id) for task_id in sorted(found)]


async def _read_metadata(client: ToolClient, project_root: str, task_id: str) -> dict:
    reply = _require_ok_reply('get_task', project_root, await client.call_tool(
        'get_task', {'id': task_id, 'project_root': project_root},
    ))
    meta = reply.get('metadata')
    if not isinstance(meta, dict):
        raise ToolReplyError(f'get_task {project_root} #{task_id}: metadata is {type(meta).__name__}')
    return meta


def _require_replace_writable(meta: dict) -> None:
    unwritable = sorted(REPLACE_STRIPPED_KEYS & meta.keys())
    if unwritable:
        raise UnwritableBlobError(
            f'carries {unwritable}, which update_task strips from a replace payload'
        )


async def _plan_target(client: ToolClient, project_root: str, task_id: str) -> Target | None:
    before = await _read_metadata(client, project_root, task_id)
    plan = nest_counters(before)
    if plan is None:
        return None
    _require_replace_writable(before)
    return Target(project_root, task_id, before, plan)


async def plan_root(
    client: ToolClient, project_root: str, *, page_size: int,
) -> tuple[list[Target], list[str]]:
    """Every migratable target in *project_root*, plus one problem per refused task."""
    try:
        task_ids = await find_target_ids(client, project_root, page_size=page_size)
    except ToolReplyError as exc:
        return [], [str(exc)]
    targets: list[Target] = []
    problems: list[str] = []
    for task_id in task_ids:
        try:
            target = await _plan_target(client, project_root, task_id)
        except (ToolReplyError, MalformedRetryLedgerError, UnwritableBlobError) as exc:
            problems.append(f'{project_root} #{task_id}: {exc}')
            continue
        if target is not None:
            targets.append(target)
    return targets, problems


def validated_backup_path(backup_path: Path | None) -> Path:
    if backup_path is None:
        raise BackupRefusedError('--apply requires --backup-path')
    if backup_path.exists():
        raise BackupRefusedError(f'backup path already exists: {backup_path}')
    return backup_path


def write_backup(backup_path: Path, targets: Sequence[Target]) -> None:
    """Save every pre-write blob as ``{project_root: {task_id: metadata}}``; never overwrites."""
    snapshot: dict[str, dict[str, dict]] = {}
    for target in targets:
        snapshot.setdefault(target.project_root, {})[target.task_id] = target.before
    try:
        with backup_path.open('x') as handle:
            json.dump(snapshot, handle, indent=2, sort_keys=True)
    except OSError as exc:
        raise BackupRefusedError(f'cannot write backup {backup_path}: {exc}') from exc


async def apply_target(client: ToolClient, target: Target, backup_path: Path) -> str | None:
    """Write *target*'s planned blob and read it back; the problem found, if any."""
    reply = await client.call_tool(
        'update_task', build_update_payload(target.task_id, target.project_root, target.plan.after),
    )
    reason = write_failure_reason(reply)
    if reason is not None:
        return f'{target.label}: update_task refused, nothing written: {reason}'
    recover = f'the pre-write blob is in {backup_path}'
    try:
        stored = await _read_metadata(client, target.project_root, target.task_id)
    except ToolReplyError as exc:
        return f'{target.label}: read-back failed ({exc}); {recover}'
    if stored != target.plan.after:
        return f'{target.label}: read-back differs from the planned blob; {recover}'
    return None


def _describe(target: Target) -> str:
    return (
        f'  {target.label}: lifted={list(target.plan.lifted)} '
        f'shadowed={list(target.plan.shadowed)}'
    )


async def run(
    client: ToolClient,
    project_roots: Sequence[str],
    *,
    apply: bool,
    backup_path: Path | None,
    page_size: int = 100,
) -> int:
    """Plan every root, then (with *apply*) back up and write; 1 if any problem occurred."""
    print(f'Mode: {"APPLY" if apply else "dry-run"}')
    try:
        backup = validated_backup_path(backup_path) if apply else None
    except BackupRefusedError as exc:
        print(f'[error] {exc}', file=sys.stderr)
        return 1

    targets: list[Target] = []
    problems: list[str] = []
    for project_root in project_roots:
        root_targets, root_problems = await plan_root(client, project_root, page_size=page_size)
        targets += root_targets
        problems += root_problems
    for target in targets:
        print(_describe(target))

    written = 0
    if backup is not None and targets:
        try:
            write_backup(backup, targets)
        except BackupRefusedError as exc:
            print(f'[error] {exc}', file=sys.stderr)
            return 1
        print(f'Backup: {backup}')
        for target in targets:
            problem = await apply_target(client, target, backup)
            if problem is None:
                written += 1
            else:
                problems.append(problem)

    for problem in problems:
        print(f'[problem] {problem}', file=sys.stderr)
    print(
        f'Summary: {len(targets)} target(s) across {len(project_roots)} root(s), '
        f'{written} written, {len(problems)} problem(s).'
    )
    return 1 if problems else 0


def _load_sibling_client() -> type:
    """``FusedMemoryClient`` from ``strip_leaked_control_keys.py``, loaded lazily by path."""
    return load_script(
        Path(__file__).parent / 'strip_leaked_control_keys.py', 'strip_leaked_control_keys',
    ).FusedMemoryClient


async def main_async(args: argparse.Namespace) -> int:
    async with _load_sibling_client()(args.server_url) as client:
        return await run(
            client, args.project_roots, apply=args.apply, backup_path=args.backup_path,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project-root', dest='project_roots', action='append', required=True,
        help='A project root whose task store to migrate. Repeat for each store; no default.',
    )
    parser.add_argument(
        '--server-url', default=DEFAULT_SERVER,
        help=f'Fused-memory MCP server URL (default: {DEFAULT_SERVER}).',
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Actually write. Without it the script only reports what it would do.',
    )
    parser.add_argument(
        '--backup-path', type=Path, default=None,
        help='Where to save every pre-write blob. Required with --apply; must not exist.',
    )
    return parser


def main() -> None:
    sys.exit(asyncio.run(main_async(build_parser().parse_args())))


if __name__ == '__main__':
    main()
