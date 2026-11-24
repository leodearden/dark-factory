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
it.  Only such a process that started before the lock was last written can
have created it, so a git started later (the merge worker's own read-only
``git cherry`` / ``git log`` children, most often) is never named a holder.

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

from shared.proc_group import MalformedStatLine, parse_start_ticks
from shared.process_identity import PROC_ROOT, ProcessIdentityUnreadable, command_name

_GIT_COMMAND_NAME = 'git'
# How far past the lock's mtime a creator's computed start may fall.  The
# computed start already reads early (btime is truncated to whole seconds), so
# this only absorbs a small forward wall-clock step after the lock was written.
_START_SLACK_S = 2.0
_COMMAND_DISPLAY_CHARS = 200
_HOLDERS_DISPLAYED = 5


@dataclass(frozen=True)
class GitProcess:
    pid: int
    command: str


@dataclass(frozen=True)
class GitProcessScan:
    """The live git processes in one work tree, and whether the scan saw them all.

    A listed process is a CANDIDATE holder, never a proven one: a live git in
    the work tree that, when the scan was bounded by a lock, had started
    before that lock was last written (a later starter cannot have created it).
    ``confirmed`` is false when the process table could not be enumerated, a
    git process could not be placed, or the lock could not be read.  An empty
    ``processes`` then means "unknown", never "unheld".
    """

    processes: tuple[GitProcess, ...]
    confirmed: bool


def scan_index_lock_holders(
    lock_path: Path, work_tree: Path, *, proc_root: Path = PROC_ROOT,
) -> GitProcessScan:
    """The live git processes in *work_tree* that could have written *lock_path*.

    A lock that is gone or cannot be stat'ed gives an unconfirmed empty scan:
    nothing may be inferred about a lock that cannot be observed.
    """
    try:
        lock_mtime = lock_path.stat().st_mtime
    except OSError:
        return GitProcessScan((), confirmed=False)
    return scan_work_tree_git_processes(work_tree, proc_root=proc_root, lock_mtime=lock_mtime)


def scan_work_tree_git_processes(
    work_tree: Path, *, proc_root: Path = PROC_ROOT, lock_mtime: float | None = None,
) -> GitProcessScan:
    """Every live ``git`` process whose cwd is in *work_tree*'s own work tree, in one pass.

    A cwd inside a nested linked worktree or clone (any directory carrying its
    own ``.git`` between the cwd and *work_tree*) belongs to that other work
    tree and is excluded.  A pid that vanishes mid-scan held nothing and is
    skipped; a git process whose cwd cannot be read might be the holder, so it
    makes the scan unconfirmed.  With *lock_mtime*, a process proven to have
    started after it is excluded too.
    """
    try:
        pids = sorted(int(entry.name) for entry in proc_root.iterdir() if entry.name.isdigit())
    except OSError:
        return GitProcessScan((), confirmed=False)
    root = work_tree.resolve()
    confirmed = True
    in_work_tree: list[int] = []
    for pid in pids:
        try:
            cwd = _git_process_cwd(pid, proc_root)
        except ProcessIdentityUnreadable:
            confirmed = False
            continue
        if cwd is not None and _in_own_work_tree(cwd, root):
            in_work_tree.append(pid)
    if lock_mtime is not None and in_work_tree:
        in_work_tree = _could_have_written(in_work_tree, proc_root, lock_mtime)
    return GitProcessScan(
        tuple(GitProcess(pid, _command_line(pid, proc_root)) for pid in in_work_tree),
        confirmed,
    )


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


def _could_have_written(pids: list[int], proc_root: Path, lock_mtime: float) -> list[int]:
    """The *pids* not proven to have exited or to have started after *lock_mtime*.

    git creates index.lock (O_EXCL) after it starts and before it runs any
    hook, and the mtime is at or after the creation, so the creator started no
    later than the mtime.  A start time that cannot be read keeps the pid:
    listing is the direction that never yields ``rm -f``.
    """
    boot_epoch = _boot_epoch(proc_root)
    latest_creator_start = lock_mtime + _START_SLACK_S
    return [
        pid for pid in pids
        if not _exited_or_started_after(pid, proc_root, boot_epoch, latest_creator_start)
    ]


def _exited_or_started_after(
    pid: int, proc_root: Path, boot_epoch: int | None, instant: float,
) -> bool:
    stat_path = proc_root / str(pid) / 'stat'
    try:
        stat_text = stat_path.read_text()
    except (FileNotFoundError, ProcessLookupError):
        return True
    except OSError:
        return False
    if boot_epoch is None:
        return False
    try:
        start_ticks = parse_start_ticks(stat_path, stat_text)
    except MalformedStatLine:
        return False
    return boot_epoch + start_ticks / os.sysconf('SC_CLK_TCK') > instant


def _boot_epoch(proc_root: Path) -> int | None:
    """The boot time (``btime``, whole epoch seconds) of *proc_root*; None when unreadable."""
    try:
        lines = (proc_root / 'stat').read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        key, _, value = line.partition(' ')
        if key == 'btime':
            return int(value) if value.strip().isdecimal() else None
    return None


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
                ' A live git process that started before the lock was written is running in '
                f"project_root's work tree and may hold it: {describe_holders(self.holders)}. "
                'A long pre-commit hook is normal, so do NOT remove the lock: wait for that '
                'process to exit, then re-submit the merge.'
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


def describe_holders(holders: tuple[GitProcess, ...]) -> str:
    """``pid N (`command`)`` for the first few *holders*, then how many more; '' for none."""
    named = ', '.join(
        f'pid {holder.pid} (`{holder.command}`)' for holder in holders[:_HOLDERS_DISPLAYED]
    )
    unnamed = len(holders) - _HOLDERS_DISPLAYED
    return f'{named} (+{unnamed} more)' if unnamed > 0 else named
