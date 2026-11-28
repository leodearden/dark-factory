#!/usr/bin/env python3
"""Did a systemd user unit start after a task's merge landed on main? A predicate.

Compares the unit's ExecMainStartTimestamp, read as systemd's ``@<epoch>``
(``systemctl --user show <unit> -p ExecMainStartTimestamp --timestamp=unix
--value``), with the committer time of the newest first-parent commit on main
whose subject is exactly ``Merge task/<id> into main``. Both are whole seconds.
A start in the landing's own second cannot be shown to have loaded the merged
tree, so it counts as not after. Read-only.

Exit codes (docs/task-authoring.md section 6):
  0   the unit started after the landing
  75  not yet: it started at or before the landing, or the task has not landed
  1   error: systemctl or git failed or answered in another shape (an empty
      start value means the unit is unknown or never started), or a bad argument
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import _checkout as _checkout
from _predicate_exit import EXIT_ERROR, EXIT_NOT_YET, PredicateParser
from git_env import scrubbed_git_env
from shared.merge_subject import MergeSubject, merge_subject_pattern, parse_merge_subject

EXIT_STARTED_AFTER = 0

MAIN_BRANCH = 'main'
TASK_BRANCH_PREFIX = 'task/'
_SYSTEMCTL_TIMEOUT_S = 10
_GIT_TIMEOUT_S = 30
_UNIX_TIMESTAMP = re.compile(r'@(\d+)')


class PredicateUnanswered(Exception):
    """systemctl or git could not answer; the message names what was asked and the reply."""


class UnitStartUnreadable(PredicateUnanswered):
    pass


class LandingUnreadable(PredicateUnanswered):
    pass


def _completed(
    argv: Sequence[str], timeout_s: int, env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env=env)


def unit_main_start(unit: str) -> datetime:
    """When *unit*'s main process last started, from systemd's ``@<epoch>``."""
    argv = [
        'systemctl', '--user', 'show', unit, '-p', 'ExecMainStartTimestamp',
        '--timestamp=unix', '--value',
    ]
    try:
        completed = _completed(argv, _SYSTEMCTL_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UnitStartUnreadable(f'{unit}: {shlex.join(argv)}: {exc}') from exc
    reply = completed.stdout.strip()
    if completed.returncode != 0:
        raise UnitStartUnreadable(
            f'{unit}: {shlex.join(argv)} exited {completed.returncode}: '
            f'{completed.stderr.strip() or reply!r}'
        )
    match = _UNIX_TIMESTAMP.fullmatch(reply)
    if match is None:
        raise UnitStartUnreadable(
            f'{unit}: ExecMainStartTimestamp is {reply!r}, not @<epoch>; an empty value means '
            'the unit is unknown or has never started'
        )
    return datetime.fromtimestamp(int(match.group(1)), UTC)


def landing_commit_time(repo: Path, task_id: int) -> datetime | None:
    """The committer time of the newest first-parent main commit merging task/<task_id>, else None."""
    expected = MergeSubject(f'{TASK_BRANCH_PREFIX}{task_id}', MAIN_BRANCH)
    pattern = merge_subject_pattern(
        branch_prefix=TASK_BRANCH_PREFIX, task_id=str(task_id), main_branch=MAIN_BRANCH,
    )
    argv = [
        'git', '-C', str(repo), 'log', MAIN_BRANCH, '--first-parent', '-E', f'--grep={pattern}',
        '--format=%H%x1f%ct%x1f%s',
    ]
    try:
        completed = _completed(argv, _GIT_TIMEOUT_S, scrubbed_git_env(os.environ))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LandingUnreadable(f'{repo}: {shlex.join(argv[3:])}: {exc}') from exc
    if completed.returncode != 0:
        raise LandingUnreadable(
            f'{repo}: git {shlex.join(argv[3:])} exited {completed.returncode}: '
            f'{completed.stderr.strip()}'
        )
    for line in completed.stdout.splitlines():
        _, committed_at, subject = line.split('\x1f', 2)
        if parse_merge_subject(subject) == expected:
            return datetime.fromtimestamp(int(committed_at), UTC)
    return None


def _parser() -> PredicateParser:
    parser = PredicateParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('unit', help='a systemd user unit, e.g. fused-memory.service')
    parser.add_argument('task_id', type=int, help='the task whose merge must land first')
    parser.add_argument(
        '--repo', type=Path, default=Path(__file__).resolve().parents[1],
        help="a checkout whose main is read (default: this one; every worktree shares main)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        started = unit_main_start(args.unit)
        landed = landing_commit_time(args.repo, args.task_id)
    except PredicateUnanswered as exc:
        print(f'unit_started_after_landing: {exc}', file=sys.stderr)
        return EXIT_ERROR
    branch = f'{TASK_BRANCH_PREFIX}{args.task_id}'
    if landed is None:
        print(f'{args.unit} started {started.isoformat()}; {branch} has not landed on '
              f'{MAIN_BRANCH}: not yet')
        return EXIT_NOT_YET
    after = started > landed
    print(f'{args.unit} started {started.isoformat()}; {branch} landed on {MAIN_BRANCH} '
          f'{landed.isoformat()}: {"started after" if after else "not yet"}')
    return EXIT_STARTED_AFTER if after else EXIT_NOT_YET


if __name__ == '__main__':
    sys.exit(main())
