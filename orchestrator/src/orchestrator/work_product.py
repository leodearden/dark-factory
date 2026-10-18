"""Which task branches carry resumable work product, from one TTL-cached off-thread git probe.

plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C3.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

# 8.4s measured cold over 1,323 branches at load ~90; 60s separates slow from hung.
PROBE_TIMEOUT_SECS: float = 60.0
_STDERR_SHOWN: int = 500


@dataclass(frozen=True)
class BranchWork:
    commits_ahead: int
    tip_committed_at: float


@dataclass(frozen=True)
class WorkProductSnapshot:
    branches: Mapping[str, BranchWork]
    zero_progress: frozenset[str]

    EMPTY: ClassVar[WorkProductSnapshot]

    def has_work_product(self, task_id: str, *, now: float, max_age_secs: float) -> bool:
        """PRD C3: the task's branch is ahead of main, its newest commit is no older than *max_age_secs*, and the zero-progress detector never fired for it."""
        work = self.branches.get(task_id)
        return (
            work is not None
            and work.commits_ahead >= 1
            and now - work.tip_committed_at <= max_age_secs
            and task_id not in self.zero_progress
        )


WorkProductSnapshot.EMPTY = WorkProductSnapshot(branches=MappingProxyType({}), zero_progress=frozenset())


class BranchProbeError(Exception):
    """The probe could not establish which task branches carry work; ``str()`` renders every known fact."""

    def __init__(
        self,
        *,
        reason: str,
        cmd: Sequence[str] = (),
        cwd: str | None = None,
        returncode: int | None = None,
        stderr: str = '',
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cmd = tuple(cmd)
        self.cwd = cwd
        self.returncode = returncode
        self.stderr = stderr

    def __str__(self) -> str:
        facts = [self.reason]
        if self.cmd:
            facts.append(f'cmd: {shlex.join(self.cmd)}')
        if self.cwd is not None:
            facts.append(f'cwd: {self.cwd}')
        if self.returncode is not None:
            facts.append(f'rc: {self.returncode}')
        if self.stderr.strip():
            facts.append(f'stderr: {self.stderr.strip()[:_STDERR_SHOWN]}')
        return '; '.join(facts)


def parse_branch_work(stdout: str, *, branch_prefix: str) -> dict[str, BranchWork]:
    """Each non-blank line must be ``refs/heads/<prefix><key> <ahead> <behind> <unix>``; anything else raises."""
    ref_prefix = f'refs/heads/{branch_prefix}'
    return dict(
        _parse_line(line, ref_prefix) for line in stdout.splitlines() if line.strip()
    )


def _parse_line(line: str, ref_prefix: str) -> tuple[str, BranchWork]:
    fields = line.split()
    unparseable = BranchProbeError(reason=f'unparseable for-each-ref line {line!r}')
    if len(fields) != 4 or not fields[0].startswith(ref_prefix):
        raise unparseable
    ref, ahead, behind, tip = fields
    try:
        int(behind)
        work = BranchWork(commits_ahead=int(ahead), tip_committed_at=float(int(tip)))
    except ValueError:
        raise unparseable from None
    return ref.removeprefix(ref_prefix), work


def read_branch_work(
    project_root: Path,
    *,
    main_branch: str,
    branch_prefix: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, BranchWork]:
    """One ``git for-each-ref`` child reading the repository AT *project_root*, never one enclosing it.

    Synchronous by design: the caller runs it on a worker thread.
    """
    cmd = (
        'git', '-C', str(project_root), 'for-each-ref',
        f'--format=%(refname) %(ahead-behind:{main_branch}) %(committerdate:unix)',
        f'refs/heads/{branch_prefix}',
    )
    env = {
        **os.environ,
        'LC_ALL': 'C',
        'LANG': 'C',
        'GIT_CEILING_DIRECTORIES': str(project_root.resolve().parent),
    }
    failure = partial(BranchProbeError, cmd=cmd, cwd=str(project_root))
    try:
        completed = run(
            list(cmd), capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECS, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise failure(
            reason=f'git for-each-ref timed out after {PROBE_TIMEOUT_SECS:g}s',
            stderr=_as_text(exc.stderr),
        ) from exc
    except OSError as exc:
        raise failure(reason=f'git could not be started: {exc}') from exc
    if completed.returncode != 0:
        raise failure(
            reason=f'git for-each-ref exited {completed.returncode}',
            returncode=completed.returncode,
            stderr=completed.stderr,
        )
    try:
        return parse_branch_work(completed.stdout, branch_prefix=branch_prefix)
    except BranchProbeError as exc:
        raise failure(reason=exc.reason, returncode=0, stderr=completed.stderr) from exc


def _as_text(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        return output.decode(errors='replace')
    return output or ''
