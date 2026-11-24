"""Diagnose who holds a work tree's index.lock and what is safe to tell an operator.

The holder of a contended ``.git/index.lock`` is found by its PROCESS, not by
its file descriptor.  git closes the lockfile's fd before it runs hooks, so
for the whole of a ``git commit``'s pre-commit hook (10-60+ minutes in some
projects) no process has the lock open: ``fuser``/``lsof`` and the fd-based
``orchestrator/src/orchestrator/rebase_recovery.py::scan_lock_holders`` all
answer "no holder" for exactly a healthy commit in progress (measured with git
2.43.0, task 5229).  The committing ``git`` process stays alive throughout,
and git chdirs to the work-tree toplevel even when launched from a
subdirectory, so "a process named ``git`` whose cwd is in this work tree" finds
it.

Unknown is never "none": a scan that could not see every git process says so
in ``confirmed``, and only a confirmed scan may be read as "nobody holds it".
A cwd is a ptrace-gated read, so inside an agent's landlock sandbox the cwd of
any process outside the sandbox's domain is unreadable, and a scan run there is
unconfirmed whenever such a git process exists.  The merge worker scans from
the unsandboxed orchestrator, where every same-uid process can be placed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from shared.process_identity import PROC_ROOT, ProcessIdentityUnreadable, command_name

_GIT_COMMAND_NAME = 'git'
_COMMAND_DISPLAY_CHARS = 200
_HOLDERS_DISPLAYED = 5


@dataclass(frozen=True)
class GitProcess:
    pid: int
    command: str


@dataclass(frozen=True)
class GitProcessScan:
    """The live git processes in one work tree, and whether the scan saw them all.

    ``confirmed`` is false when the process table could not be enumerated, or
    a git process could not be placed.  An empty ``processes`` then means
    "unknown", never "unheld".
    """

    processes: tuple[GitProcess, ...]
    confirmed: bool


def scan_work_tree_git_processes(
    work_tree: Path, *, proc_root: Path = PROC_ROOT,
) -> GitProcessScan:
    """Every live ``git`` process whose cwd is in *work_tree*'s own work tree, in one pass.

    A cwd inside a nested linked worktree or clone (any directory carrying its
    own ``.git`` between the cwd and *work_tree*) belongs to that other work
    tree and is excluded.  A pid that vanishes mid-scan held nothing and is
    skipped; a git process whose cwd cannot be read might be the holder, so it
    makes the scan unconfirmed.
    """
    try:
        pids = sorted(int(entry.name) for entry in proc_root.iterdir() if entry.name.isdigit())
    except OSError:
        return GitProcessScan((), confirmed=False)
    root = work_tree.resolve()
    confirmed = True
    found: list[GitProcess] = []
    for pid in pids:
        try:
            cwd = _git_process_cwd(pid, proc_root)
        except ProcessIdentityUnreadable:
            confirmed = False
            continue
        if cwd is not None and _in_own_work_tree(cwd, root):
            found.append(GitProcess(pid, _command_line(pid, proc_root)))
    return GitProcessScan(tuple(found), confirmed)


def _git_process_cwd(pid: int, proc_root: Path) -> Path | None:
    """*pid*'s cwd when it is a live git process; None when it is not one or is gone."""
    if command_name(pid, proc_root=proc_root) != _GIT_COMMAND_NAME:
        return None
    cwd_link = proc_root / str(pid) / 'cwd'
    try:
        return Path(os.readlink(cwd_link))
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise ProcessIdentityUnreadable(cwd_link, exc.strerror or repr(exc)) from exc


def _in_own_work_tree(cwd: Path, root: Path) -> bool:
    if not cwd.is_relative_to(root):
        return False
    directory = cwd
    while directory != root:
        if os.path.lexists(directory / '.git'):
            return False
        directory = directory.parent
    return True


def _command_line(pid: int, proc_root: Path) -> str:
    try:
        raw = (proc_root / str(pid) / 'cmdline').read_bytes()
    except OSError:
        return '?'
    text = raw.replace(b'\0', b' ').decode('utf-8', errors='replace').strip()
    if len(text) > _COMMAND_DISPLAY_CHARS:
        return text[:_COMMAND_DISPLAY_CHARS] + '…'
    return text or '?'


class IndexLockVerdict(StrEnum):
    LIVE_HOLDER = 'LIVE_HOLDER'
    CRASHED_LEFTOVER = 'CRASHED_LEFTOVER'
    UNPROVEN = 'UNPROVEN'


@dataclass(frozen=True)
class IndexLockDiagnosis:
    """What is known about one contended index.lock, and the only advice it supports.

    CRASHED_LEFTOVER is the one verdict whose advice deletes the lock, so it
    is refused at construction unless it carries no holder and a first-observed
    age past ``stale_after_s``.  Holders are named exactly when the verdict is
    LIVE_HOLDER, so a listed holder can never be rendered as anything else.
    """

    verdict: IndexLockVerdict
    holders: tuple[GitProcess, ...]
    initial_age_s: float | None
    stale_after_s: float | None

    def __post_init__(self) -> None:
        if bool(self.holders) != (self.verdict is IndexLockVerdict.LIVE_HOLDER):
            raise ValueError(
                'an index.lock diagnosis names holders exactly when it is LIVE_HOLDER; '
                f'got verdict={self.verdict}, holders={self.holders!r}'
            )
        if self.verdict is IndexLockVerdict.CRASHED_LEFTOVER and not _older_than(
            self.initial_age_s, self.stale_after_s,
        ):
            raise ValueError(
                'CRASHED_LEFTOVER requires initial_age_s > stale_after_s; '
                f'got verdict={self.verdict}, initial_age_s={self.initial_age_s!r}, '
                f'stale_after_s={self.stale_after_s!r}'
            )

    def advice(self, lock_path: str) -> str:
        """The recovery sentence for an operator, with a leading space; '' when none is safe."""
        if self.verdict is IndexLockVerdict.LIVE_HOLDER:
            return (
                f" It is held by a LIVE git process in project_root's work tree: "
                f'{_describe(self.holders)}. A long pre-commit hook is normal, so this is '
                f'NOT a crashed leftover and the lock must NOT be removed: wait for that '
                f'process to exit, then re-submit the merge.'
            )
        if self.verdict is IndexLockVerdict.CRASHED_LEFTOVER:
            return (
                f' The lock was ALREADY {self.initial_age_s:.0f}s old when the merge worker '
                f'FIRST observed it, past the {self.stale_after_s:.0f}s staleness floor, and '
                f"a complete /proc scan found no git process in project_root's work tree, "
                f'so it is likely a crashed-git leftover: confirm no git process is running '
                f'in project_root, then clear it with `rm -f {lock_path}` and re-submit the '
                f'merge.'
            )
        return ''


def diagnose_index_lock(
    *,
    initial_age_s: float | None,
    grace_s: float | None,
    stale_floor_s: float,
    scan: GitProcessScan | None,
) -> IndexLockDiagnosis:
    """Classify a contended index.lock from its first-observed age and a holder scan.

    A listed holder decides it, whatever the lock's age.  Age is only the
    second conjunct of CRASHED_LEFTOVER, never sufficient alone: a pre-commit
    hook can legitimately run for an hour.  The bar is
    ``max(grace_s, stale_floor_s)`` because the grace is tunable down to 0, at
    which every age would clear it.  A missing scan, an unconfirmed one, or a
    missing age fact is UNPROVEN, never CRASHED_LEFTOVER: unknown is never
    "none".
    """
    stale_after_s = None if grace_s is None else max(grace_s, stale_floor_s)
    holders = () if scan is None else scan.processes
    if holders:
        verdict = IndexLockVerdict.LIVE_HOLDER
    elif scan is not None and scan.confirmed and _older_than(initial_age_s, stale_after_s):
        verdict = IndexLockVerdict.CRASHED_LEFTOVER
    else:
        verdict = IndexLockVerdict.UNPROVEN
    return IndexLockDiagnosis(verdict, holders, initial_age_s, stale_after_s)


def _older_than(age_s: float | None, floor_s: float | None) -> bool:
    return age_s is not None and floor_s is not None and age_s > floor_s


def _describe(holders: tuple[GitProcess, ...]) -> str:
    named = ', '.join(
        f'pid {holder.pid} (`{holder.command}`)' for holder in holders[:_HOLDERS_DISPLAYED]
    )
    unnamed = len(holders) - _HOLDERS_DISPLAYED
    return f'{named} (+{unnamed} more)' if unnamed > 0 else named
