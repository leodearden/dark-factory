#!/usr/bin/env python3
"""Re-stamp ``metadata.branch_base_sha`` on pending tasks whose zero-commit branch moved (task 7332).

Until task 7332 the orchestrator recorded a branch's base once, with an
OLD-wins write, and never again: a later dispatch that rebased a zero-commit
branch onto a newer main, or re-created it, left the recorded base at the
first creation point. ``merge_lane/landing_evidence.py::branch_is_degenerate``
(tip == recorded base) then returns False for a branch carrying no work, and
the already-landed dispatch gate pays an uncapped citation ``git log`` for it
on every scheduler tick. The orchestrator now re-stamps on every move
(``orchestrator/src/orchestrator/workflow.py::TaskWorkflow._restamp_branch_base_if_no_own_commits``);
this script repairs the rows written before that.

A PENDING task is eligible when its ``task/<id>`` branch exists, its tip
differs from the recorded base, the tip is an ancestor of main
(``git rev-list --count main..<tip>`` is 0, so every commit on the branch is
also on main), and main holds neither a merge marker for the branch nor a
commit citing the task. The last two rule out a branch whose commits on main
ARE its own landed work. The marker and citation lookups are the
orchestrator's own, in their checked form
(``GitOps.find_merge_marker_checked``,
``GitOps.find_task_citation_commit_checked`` with the project's
``git.commit_citation_pattern``), so a git failure excludes the task as
``git_error`` instead of reading as "no landing". An eligible task gets
``branch_base_sha = <tip>``.

``--check`` (the default) reports and never dials the MCP server. ``--apply``
re-judges each eligible task against git, then re-reads it through the MCP
``get_task`` tool immediately before writing, and skips it unless it still
judges eligible at the same tip and is still pending with the census's
recorded base. It writes through ``update_task``
('merge' mode), classifies every reply with ``write_failure_reason`` and
re-reads the task to confirm the stored value. Exit status: 0 success, 1 any
write failure, 2 unreadable task store.

Usage:
    uv run --frozen python scripts/restamp_degenerate_branch_bases.py --project-root ROOT [--check]
    uv run --frozen python scripts/restamp_degenerate_branch_bases.py --project-root ROOT --apply

Run it directly, never via ``python -m``: it imports flat siblings in scripts/.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import _checkout as _checkout
from _task_db_scan import UNREADABLE_STORE_ERRORS, decode_metadata, tasks_db_path
from migrate_metadata_modules_to_files import (
    DEFAULT_SERVER,
    FusedMemoryClient,
    call_tool_with_retry,
    write_failure_reason,
)
from orchestrator.config import load_config
from orchestrator.git_ops import CitationPatternInvalid, GitOps, GitQueryFailed
from shared.tasks_db_readonly import connect_ro

CONFIG_FILENAME = 'dark-factory-orchestrator.yaml'


class Exclusion(StrEnum):
    BRANCH_ABSENT = 'branch_absent'
    ALREADY_DEGENERATE = 'already_degenerate'
    COMMITS_BEYOND_MAIN = 'commits_beyond_main'
    MERGE_MARKER = 'merge_marker'
    CITATION = 'citation'
    GIT_ERROR = 'git_error'


@dataclass(frozen=True)
class BranchVerdict:
    task_id: int
    tip: str | None
    recorded_base: str | None
    exclusion: Exclusion | None

    @property
    def eligible(self) -> bool:
        return self.exclusion is None


class RestampClient(FusedMemoryClient):
    _client_name = 'restamp-degenerate-branch-bases'


async def judge_branch(
    git_ops: GitOps, task_id: int, metadata: Mapping[str, Any],
) -> BranchVerdict:
    """Decide whether ``task/<task_id>``'s recorded base may be moved to its tip.

    Cheapest checks first; the two full-history walks run only for a branch
    already proven to sit on main with a stale base.
    """
    branch = f'{git_ops.config.branch_prefix}{task_id}'
    tip = await git_ops.resolve_branch_sha(branch)
    recorded = metadata.get('branch_base_sha')
    recorded_base = recorded if isinstance(recorded, str) else None

    def verdict(exclusion: Exclusion | None) -> BranchVerdict:
        return BranchVerdict(task_id, tip, recorded_base, exclusion)

    if tip is None:
        return verdict(Exclusion.BRANCH_ABSENT)
    if tip == recorded_base:
        return verdict(Exclusion.ALREADY_DEGENERATE)
    on_main = await git_ops.is_ancestor_checked(tip, git_ops.config.main_branch)
    if on_main is None:
        return verdict(Exclusion.GIT_ERROR)
    if not on_main:
        return verdict(Exclusion.COMMITS_BEYOND_MAIN)
    try:
        if await git_ops.find_merge_marker_checked(branch):
            return verdict(Exclusion.MERGE_MARKER)
        if await git_ops.find_task_citation_commit_checked(
            str(task_id), pattern_template=git_ops.config.commit_citation_pattern,
        ):
            return verdict(Exclusion.CITATION)
    except (GitQueryFailed, CitationPatternInvalid):
        return verdict(Exclusion.GIT_ERROR)
    return verdict(None)


def read_pending_tasks(project_root: Path) -> list[tuple[int, dict]]:
    with contextlib.closing(connect_ro(tasks_db_path(project_root))) as conn:
        rows = conn.execute(
            "SELECT id, metadata FROM tasks WHERE status = 'pending' ORDER BY id",
        ).fetchall()
    return [(task_id, decode_metadata(raw)) for task_id, raw in rows]


async def census(
    git_ops: GitOps, pending: Iterable[tuple[int, Mapping[str, Any]]],
) -> list[BranchVerdict]:
    return [await judge_branch(git_ops, task_id, metadata) for task_id, metadata in pending]


def render_report(project_root: Path, main_sha: str, verdicts: list[BranchVerdict]) -> str:
    eligible = [v for v in verdicts if v.eligible]
    excluded = Counter(v.exclusion for v in verdicts if v.exclusion is not None)
    lines = [
        f'project_root: {project_root}',
        f'main: {main_sha}',
        f'pending scanned: {len(verdicts)}',
        f'  with a task branch: {len(verdicts) - excluded[Exclusion.BRANCH_ABSENT]}',
        f'eligible: {len(eligible)}',
        'excluded:',
        *(f'  {reason.value}: {excluded[reason]}' for reason in Exclusion),
        'eligible ids: ' + ' '.join(str(v.task_id) for v in eligible),
        *(
            f'  {v.task_id} tip={v.tip} recorded_base={v.recorded_base or "absent"}'
            for v in eligible
        ),
    ]
    return '\n'.join(lines)


def _metadata_of(reply: Any) -> dict:
    raw = reply.get('metadata') if isinstance(reply, dict) else None
    return raw if isinstance(raw, dict) else decode_metadata(raw)


async def _read_task(client: FusedMemoryClient, project_root: Path, task_id: int) -> dict | str:
    """The task as the MCP server holds it now, or why it could not be read."""
    reply = await call_tool_with_retry(
        client, 'get_task', {'id': str(task_id), 'project_root': str(project_root)},
    )
    if (reason := write_failure_reason(reply)) is not None:
        return reason
    return reply


async def restamp_one(
    client: FusedMemoryClient, git_ops: GitOps, project_root: Path, prior: BranchVerdict,
) -> str | None:
    """Re-check *prior*, write its tip, confirm; return why it failed, or None.

    A task that no longer judges eligible at the census's tip, is no longer
    pending, or whose recorded base moved is skipped (reported, not a
    failure): something else has touched it since the census. The slow git
    re-judge runs first, so only one MCP round trip separates the status read
    from the write.
    """
    verdict = await judge_branch(
        git_ops, prior.task_id, {'branch_base_sha': prior.recorded_base},
    )
    if verdict.tip != prior.tip or not verdict.eligible:
        print(f'  skip {prior.task_id}: changed since the census ({verdict.exclusion})')
        return None
    task = await _read_task(client, project_root, prior.task_id)
    if isinstance(task, str):
        return f'pre-write read failed: {task}'
    if (
        task.get('status') != 'pending'
        or _metadata_of(task).get('branch_base_sha') != prior.recorded_base
    ):
        print(f'  skip {prior.task_id}: status or recorded base changed since the census')
        return None
    reply = await call_tool_with_retry(client, 'update_task', {
        'id': str(prior.task_id),
        'project_root': str(project_root),
        'metadata': json.dumps({'branch_base_sha': verdict.tip}),
        'metadata_mode': 'merge',
    })
    if (reason := write_failure_reason(reply)) is not None:
        return reason
    reread = await _read_task(client, project_root, prior.task_id)
    if isinstance(reread, str):
        return f're-read failed: {reread}'
    if (stored := _metadata_of(reread).get('branch_base_sha')) != verdict.tip:
        return f're-read shows branch_base_sha={stored!r}, expected {verdict.tip}'
    print(f'  restamped {prior.task_id}: branch_base_sha={verdict.tip}')
    return None


async def apply_restamps(
    client: FusedMemoryClient,
    git_ops: GitOps,
    project_root: Path,
    eligible: Iterable[BranchVerdict],
) -> int:
    """Re-stamp each eligible task; return the number of failures."""
    failures = 0
    for prior in eligible:
        reason = await restamp_one(client, git_ops, project_root, prior)
        if reason is not None:
            failures += 1
            print(f'  FAILED {prior.task_id}: {reason}', file=sys.stderr)
    return failures


async def main_async(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config) if args.config else project_root / CONFIG_FILENAME
    git_ops = GitOps(load_config(config_path).git, project_root)
    try:
        pending = read_pending_tasks(project_root)
    except UNREADABLE_STORE_ERRORS as exc:
        print(f'task store unreadable: {exc}', file=sys.stderr)
        return 2
    verdicts = await census(git_ops, pending)
    print(render_report(project_root, await git_ops.get_main_sha(), verdicts))
    if not args.apply:
        return 0
    async with RestampClient(args.server_url) as client:
        failures = await apply_restamps(
            client, git_ops, project_root, [v for v in verdicts if v.eligible],
        )
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Re-stamp branch_base_sha on pending tasks whose zero-commit branch moved.',
    )
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--config', help=f'orchestrator config (default: <root>/{CONFIG_FILENAME})')
    parser.add_argument('--server-url', default=DEFAULT_SERVER)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='report only (the default)')
    mode.add_argument('--apply', action='store_true', help='write the re-stamps')
    sys.exit(asyncio.run(main_async(parser.parse_args())))


if __name__ == '__main__':
    main()
