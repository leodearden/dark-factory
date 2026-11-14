#!/usr/bin/env python3
"""Incremental repack + commit-graph of every orchestrator-managed shared repo,
one at a time (task 8076); prints one JSON record per repo. Why the shared repos
need it, and the safety envelope it must stay inside, live in
docs/shared-repo-git-maintenance.md.
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import git_env
import orchestrator_projects

_LOG_PREFIX = 'shared_repo_git_maintenance.py:'

_NICED = ('nice', '-n', '19', 'ionice', '-c3')
# Never `-a`: docs/shared-repo-git-maintenance.md, 'What the timer may and may not run'.
REPACK = ('repack', '-d', '-q', '-n')
COMMIT_GRAPH = ('commit-graph', 'write', '--reachable')
GIT_TIMEOUT_SECS = 1800


class Outcome(enum.Enum):
    MAINTAINED = 'maintained'
    FAILED = 'failed'
    NOT_A_REPO = 'not-a-repo'
    BUSY = 'busy'


@dataclass(frozen=True)
class RepoResult:
    root: Path
    outcome: Outcome
    loose_before: int | None
    loose_after: int | None
    repack_rc: int | None
    commit_graph_rc: int | None
    seconds: float

    def record(self) -> dict[str, object]:
        return {
            **asdict(self),
            'root': str(self.root),
            'outcome': self.outcome.value,
            'seconds': round(self.seconds, 1),
        }


def _git(root: Path, *args: str, niced: bool = False) -> tuple[int, str]:
    """``git -C <root> <args>`` -> ``(rc, stdout)``: the one subprocess chokepoint.

    Every call runs with ambient git redirection scrubbed, so ``-C`` really
    selects the repo. A failure's stderr is relayed to this process's stderr.
    """
    argv = [*(_NICED if niced else ()), 'git', '-C', str(root), *args]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            env=git_env.scrubbed_git_env(os.environ),
            timeout=GIT_TIMEOUT_SECS,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        _log(f'git {" ".join(args)} in {root}: {err}')
        return 1, ''
    if proc.returncode != 0:
        _log(f'git {" ".join(args)} in {root} exited {proc.returncode}: {proc.stderr.strip()}')
    return proc.returncode, proc.stdout


def _log(message: str) -> None:
    print(f'{_LOG_PREFIX} {message}', file=sys.stderr, flush=True)


def _loose_count(root: Path) -> int | None:
    rc, out = _git(root, 'count-objects', '-v')
    if rc != 0:
        return None
    for line in out.splitlines():
        key, _, value = line.partition(':')
        if key == 'count':
            return int(value)
    return None


def maintain(root: Path) -> RepoResult:
    started = time.monotonic()
    loose_before = _loose_count(root)
    repack_rc, _ = _git(root, *REPACK, niced=True)
    commit_graph_rc = _git(root, *COMMIT_GRAPH, niced=True)[0] if repack_rc == 0 else None
    outcome = Outcome.MAINTAINED if repack_rc == 0 and commit_graph_rc == 0 else Outcome.FAILED
    return RepoResult(
        root=root,
        outcome=outcome,
        loose_before=loose_before,
        loose_after=_loose_count(root),
        repack_rc=repack_rc,
        commit_graph_rc=commit_graph_rc,
        seconds=time.monotonic() - started,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Incrementally repack and write the commit-graph of every orchestrator-managed repo.',
        epilog='Exit 0: nothing failed. Exit 1: some repo failed (see its record).',
    )
    orchestrator_projects.add_arguments(parser)
    parser.add_argument('--lock-file', type=Path, default=None, help='the host-wide run lock')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    targets, unit_scan = orchestrator_projects.targets_from_args(args)
    for problem in unit_scan.problems:
        _log(problem)
    failed = False
    for target in targets:
        result = maintain(target.root)
        print(json.dumps(result.record(), sort_keys=True), flush=True)
        failed = failed or result.outcome is Outcome.FAILED
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
