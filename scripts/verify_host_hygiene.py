#!/usr/bin/env python3
"""Find, and with --apply reclaim, DF verify-host leftovers that no live run holds.

Classes: `_merge-<8hex>` and task worktrees under each repo's `.worktrees/`,
and `df-verify-*.scope` user units. Report-only by default. Stdlib only, so
`--host H` can run this same file on H over ssh. Which mechanism bounds each
class, and what this tool never touches: OPERATIONS.md, "Verify-host disk
hygiene (leo-laptop, task 7269)".
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DEFAULT_REPO = Path("/home/leo/src/dark-factory")
DEFAULT_SSH = "ssh -o BatchMode=yes -o ConnectTimeout=15"
WORKTREE_BASE_NAME = ".worktrees"
PGID_DIR_NAME = ".merge_verify_pgids"
PGID_WORKTREE_SUFFIX = ".worktree"
MERGE_WORKTREE_NAME = re.compile(r"_merge-[0-9a-f]{8}")
INFRA_BAND_PREFIXES = ("_", ".")
GIT_LOCATION_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")
GIT_TIMEOUT_SECONDS = 120
SCOPE_UNIT_GLOB = "df-verify-*.scope"
SCOPE_UNIT_NAME = re.compile(r"df-verify-.+\.scope")
SYSTEMCTL_TIMEOUT_SECONDS = 20
UNIT_STILL_RUNNING = frozenset({"active", "activating", "deactivating"})
SERVICE_MANAGER_COMM = "systemd"
EXIT_OK, EXIT_SOME_FAILED, EXIT_USAGE = 0, 1, 2
PROG = "verify_host_hygiene"


class Kind(StrEnum):
    MERGE_WORKTREE = "merge-worktree"
    TASK_WORKTREE = "task-worktree"
    VERIFY_SCOPE = "verify-scope"


WORKTREE_KINDS = frozenset({Kind.MERGE_WORKTREE, Kind.TASK_WORKTREE})


class Action(StrEnum):
    RECLAIM = "reclaim"
    KEEP = "keep"
    RECLAIMED = "reclaimed"
    FAILED = "failed"


class Reason(StrEnum):
    STALE = "stale"
    TOO_YOUNG = "too-young"
    LOCK_HELD = "lock-held"
    PGID_LIVE = "pgid-live"
    PROCESS_CWD = "process-cwd"
    GIT_LOCKED = "git-locked"
    DIRTY = "dirty"
    UNREGISTERED = "unregistered"
    ORPHANED = "orphaned"
    UNIT_FAILED = "unit-failed"
    CONTROLLER_ALIVE = "controller-alive"
    CGROUP_UNREADABLE = "cgroup-unreadable"


@dataclass(frozen=True)
class Candidate:
    kind: Kind
    target: str


@dataclass(frozen=True)
class Verdict:
    candidate: Candidate
    action: Action
    reason: Reason
    bytes: int = 0

    def line(self) -> str:
        fields = (self.candidate.kind, self.action, self.reason, str(self.bytes),
                  self.candidate.target)
        return "\t".join(fields)


class GitFailed(Exception):
    """A git command exited non-zero, timed out or could not start."""


def git(cwd: Path, *args: str) -> str:
    command = ["git", "-C", str(cwd), *args]
    env = {k: v for k, v in os.environ.items() if k not in GIT_LOCATION_VARS}
    try:
        proc = subprocess.run(command, capture_output=True, text=True, env=env,
                              timeout=GIT_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitFailed(f"{' '.join(command)}: {exc}") from exc
    if proc.returncode != 0:
        raise GitFailed(f"{' '.join(command)}: exit {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def real(path: Path | str) -> Path:
    return Path(os.path.realpath(path))


def is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


# --- facts collected once per run -------------------------------------------

def process_cwds() -> frozenset[Path]:
    """Every readable /proc/<pid>/cwd target; unreadable entries are skipped."""
    cwds = set()
    for entry in os.scandir("/proc"):
        if entry.name.isdigit():
            try:
                cwds.add(Path(os.readlink(f"/proc/{entry.name}/cwd")))
            except OSError:
                continue
    return frozenset(cwds)


@dataclass(frozen=True)
class Registration:
    locked: bool
    prunable: bool


def registrations(repo: Path) -> dict[Path, Registration]:
    """`git worktree list --porcelain`, keyed by each worktree's real path."""
    table = {}
    for block in git(repo, "worktree", "list", "--porcelain").split("\n\n"):
        head, *attributes = block.splitlines() or [""]
        if head.startswith("worktree "):
            keys = {attribute.split(" ", 1)[0] for attribute in attributes}
            table[real(head.removeprefix("worktree "))] = Registration(
                locked="locked" in keys, prunable="prunable" in keys)
    return table


@dataclass(frozen=True)
class PgidRecord:
    live: bool
    worktree: Path | None


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def pgid_is_live(pgid: int | None) -> bool:
    """Only a provably absent group is dead: unreadable or EPERM counts as live."""
    if pgid is None or pgid <= 1:
        return True
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pgid_record(pgid_file: Path) -> PgidRecord:
    try:
        recorded = pgid_file.with_name(pgid_file.name + PGID_WORKTREE_SUFFIX).read_text().strip()
    except OSError:
        recorded = ""
    return PgidRecord(live=pgid_is_live(read_int(pgid_file)),
                      worktree=real(recorded) if recorded else None)


def pgid_records(base: Path) -> tuple[PgidRecord, ...]:
    """The `.merge_verify_pgids/<id>` records (verify_cancel.py::pgid_file layout)."""
    try:
        entries = list(os.scandir(base / PGID_DIR_NAME))
    except FileNotFoundError:
        return ()
    return tuple(
        read_pgid_record(Path(entry.path)) for entry in sorted(entries, key=lambda e: e.name)
        if entry.is_file(follow_symlinks=False)
        and not entry.name.startswith(".")
        and not entry.name.endswith((PGID_WORKTREE_SUFFIX, ".tmp"))
    )


@dataclass(frozen=True)
class RepoFacts:
    repo: Path
    registrations: Mapping[Path, Registration]
    pgid_records: tuple[PgidRecord, ...]
    cwds: frozenset[Path]

    @property
    def base(self) -> Path:
        return self.repo / WORKTREE_BASE_NAME

    def pgid_live_for(self, path: Path) -> bool:
        return any(record.live and record.worktree is not None
                   and is_within(record.worktree, path) for record in self.pgid_records)

    def cwd_inside(self, path: Path) -> bool:
        return any(is_within(cwd, path) for cwd in self.cwds)


# --- worktree discovery and classification ----------------------------------

def worktree_kind(name: str) -> Kind | None:
    """`_merge-<8hex>` is a merge worktree; every other `_`/`.` band has its own owner."""
    if MERGE_WORKTREE_NAME.fullmatch(name):
        return Kind.MERGE_WORKTREE
    if name.startswith(INFRA_BAND_PREFIXES) or name.endswith(".lock"):
        return None
    return Kind.TASK_WORKTREE


def worktree_candidates(base: Path, kinds: frozenset[Kind]) -> list[Candidate]:
    try:
        entries = list(os.scandir(base))
    except FileNotFoundError:
        return []
    found = []
    for entry in entries:
        kind = worktree_kind(entry.name)
        if kind is not None and kind in kinds and entry.is_dir(follow_symlinks=False):
            found.append(Candidate(kind, str(real(base) / entry.name)))
    return found


def lane_lock_path(worktree: Path) -> Path:
    """The `<dir>.lock` sibling, verify_cancel.py::lane_lock_path's convention."""
    return worktree.with_name(worktree.name + ".lock")


def try_lock_exclusive(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def lock_is_held(lock: Path) -> bool:
    """Probe an existing lane lock without ever creating one."""
    try:
        fd = os.open(lock, os.O_RDONLY)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        return not try_lock_exclusive(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def held_lane_lock(lock: Path) -> Iterator[bool]:
    """Hold *lock*, created if absent, across the block; False when anyone else holds it."""
    try:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        yield False
        return
    try:
        yield try_lock_exclusive(fd)
    finally:
        os.close(fd)


def newest_mtime(path: Path) -> float:
    """Newest lstat mtime of *path* and its direct children; unreadable is newest."""
    try:
        newest = os.lstat(path).st_mtime
        with os.scandir(path) as entries:
            for entry in entries:
                newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
    except OSError:
        return math.inf
    return newest


def is_dirty(worktree: Path) -> bool:
    """Any `git status` output, or any failure to get it, counts as dirty."""
    try:
        return bool(git(worktree, "--no-optional-locks", "status", "--porcelain").strip())
    except GitFailed:
        return True


@dataclass(frozen=True)
class WorktreeFacts:
    kind: Kind
    git_locked: bool
    lock_held: bool
    pgid_live: bool
    process_cwd: bool
    too_young: bool
    registered: bool
    dirty: bool


def worktree_facts(candidate: Candidate, facts: RepoFacts, *, lock_held: bool,
                   min_age_seconds: float) -> WorktreeFacts:
    path = Path(candidate.target)
    registration = facts.registrations.get(path)
    registered = registration is not None and not registration.prunable
    return WorktreeFacts(
        kind=candidate.kind,
        git_locked=registration is not None and registration.locked,
        lock_held=lock_held,
        pgid_live=facts.pgid_live_for(path),
        process_cwd=facts.cwd_inside(path),
        too_young=time.time() - newest_mtime(path) < min_age_seconds,
        registered=registered,
        dirty=candidate.kind is Kind.TASK_WORKTREE and registered and is_dirty(path),
    )


def classify_worktree(facts: WorktreeFacts) -> Reason:
    """The first liveness signal that holds keeps the dir; none holding means stale."""
    if facts.git_locked:
        return Reason.GIT_LOCKED
    if facts.lock_held:
        return Reason.LOCK_HELD
    if facts.pgid_live:
        return Reason.PGID_LIVE
    if facts.process_cwd:
        return Reason.PROCESS_CWD
    if facts.too_young:
        return Reason.TOO_YOUNG
    if facts.kind is Kind.TASK_WORKTREE and not facts.registered:
        return Reason.UNREGISTERED
    if facts.dirty:
        return Reason.DIRTY
    return Reason.STALE


def tree_bytes(root: Path) -> int:
    """Allocated bytes under *root*: lstat only, never following a symlink or a mount."""
    top = os.lstat(root)
    total, pending = top.st_blocks * 512, [root]
    while pending:
        try:
            entries = list(os.scandir(pending.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if info.st_dev != top.st_dev:
                continue
            total += info.st_blocks * 512
            if stat.S_ISDIR(info.st_mode):
                pending.append(Path(entry.path))
    return total


def report_worktree(candidate: Candidate, facts: RepoFacts, min_age_seconds: float) -> Verdict:
    path = Path(candidate.target)
    reason = classify_worktree(worktree_facts(
        candidate, facts, lock_held=lock_is_held(lane_lock_path(path)),
        min_age_seconds=min_age_seconds))
    if reason is Reason.STALE:
        return Verdict(candidate, Action.RECLAIM, reason, tree_bytes(path))
    return Verdict(candidate, Action.KEEP, reason)


def remove_worktree(candidate: Candidate, facts: RepoFacts) -> None:
    """git removes what it tracks (refusing a dirty task worktree); rmtree the rest."""
    path = Path(candidate.target)
    registration = facts.registrations.get(path)
    if registration is None or registration.prunable:
        shutil.rmtree(path)
    elif candidate.kind is Kind.MERGE_WORKTREE:
        git(facts.repo, "worktree", "remove", "--force", str(path))
    else:
        git(facts.repo, "worktree", "remove", str(path))
    if os.path.lexists(path):
        raise OSError(f"{path} still exists after removal")


def reclaim_worktree(candidate: Candidate, facts: RepoFacts, min_age_seconds: float) -> Verdict:
    """Acquire-then-remove under the lane lock, the remove_merge_worktree_guarded discipline."""
    path = Path(candidate.target)
    lock = lane_lock_path(path)
    with held_lane_lock(lock) as held:
        if not held:
            return Verdict(candidate, Action.KEEP, Reason.LOCK_HELD)
        reason = classify_worktree(worktree_facts(
            candidate, facts, lock_held=False, min_age_seconds=min_age_seconds))
        if reason is not Reason.STALE:
            return Verdict(candidate, Action.KEEP, reason)
        size = tree_bytes(path)
        try:
            remove_worktree(candidate, facts)
        except (GitFailed, OSError) as exc:
            print(f"{PROG}: could not remove {path}: {exc}", file=sys.stderr)
            return Verdict(candidate, Action.FAILED, reason, size)
        lock.unlink(missing_ok=True)
        return Verdict(candidate, Action.RECLAIMED, reason, size)


def prune_registry(repo: Path) -> None:
    try:
        git(repo, "worktree", "prune")
    except GitFailed as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)


# --- verify scopes -----------------------------------------------------------

@dataclass(frozen=True)
class ScopeMember:
    pid: int
    ppid: int
    parent_comm: str | None


def scope_is_orphaned(members: Iterable[ScopeMember]) -> bool:
    """True iff no member has an out-of-scope parent that could still be its controller.

    Only PID 0/1 or a systemd instance rules a parent out; an unreadable one is live.
    """
    members = tuple(members)
    pids = {member.pid for member in members}
    return not any(member.ppid not in pids and member.ppid > 1
                   and member.parent_comm != SERVICE_MANAGER_COMM for member in members)


def systemctl(*args: str) -> subprocess.CompletedProcess[str] | None:
    """`systemctl --user ARGS`, or None when it could not run to completion."""
    try:
        return subprocess.run(["systemctl", "--user", *args], capture_output=True,
                              text=True, timeout=SYSTEMCTL_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return None


def scope_units() -> list[tuple[str, str]] | None:
    """(unit, active state) per df-verify scope, or None when the listing failed."""
    proc = systemctl("list-units", "--all", "--plain", "--no-legend", SCOPE_UNIT_GLOB)
    if proc is None or proc.returncode != 0:
        return None
    units = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and SCOPE_UNIT_NAME.fullmatch(fields[0]):
            units.append((fields[0], fields[2]))
    return units


def read_comm(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return None


def scope_member(pid: int) -> ScopeMember | None:
    """*pid*'s parent facts, or None once it has exited. The comm may hold spaces or ')'."""
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    ppid = int(stat_text[stat_text.rindex(")") + 1:].split()[1])
    return ScopeMember(pid=pid, ppid=ppid, parent_comm=read_comm(ppid))


def scope_members(unit: str, cgroup_root: Path) -> tuple[ScopeMember, ...] | None:
    """The live members of *unit*'s cgroup, or None when it cannot be read."""
    proc = systemctl("show", "-p", "ControlGroup", "--value", unit)
    cgroup = proc.stdout.strip() if proc is not None and proc.returncode == 0 else ""
    if not cgroup:
        return None
    try:
        pids = [int(line) for line in
                (cgroup_root / cgroup.lstrip("/") / "cgroup.procs").read_text().split()]
        members = [scope_member(pid) for pid in pids]
    except (OSError, ValueError, IndexError):
        return None
    return tuple(member for member in members if member is not None)


def reap_scope(unit: str) -> bool:
    """SIGKILL, stop, then confirm, as verify.py::reap_leftover_verify_scopes does."""
    systemctl("kill", "--signal=SIGKILL", unit)
    systemctl("stop", unit)
    probe = systemctl("is-active", unit)
    state = probe.stdout.strip() if probe is not None else "unknown"
    if probe is None or state in UNIT_STILL_RUNNING:
        print(f"{PROG}: {unit} is still {state} after kill and stop", file=sys.stderr)
        return False
    return True


def reset_failed(unit: str) -> bool:
    proc = systemctl("reset-failed", unit)
    if proc is None or proc.returncode != 0:
        detail = proc.stderr.strip() if proc is not None else "did not complete"
        print(f"{PROG}: reset-failed {unit}: {detail}", file=sys.stderr)
        return False
    return True


def judge_scope(unit: str, state: str, cgroup_root: Path, apply: bool) -> Verdict:
    candidate = Candidate(Kind.VERIFY_SCOPE, unit)
    if state == "failed":
        reason = Reason.UNIT_FAILED
    elif state != "active":
        return Verdict(candidate, Action.KEEP, Reason.CONTROLLER_ALIVE)
    else:
        members = scope_members(unit, cgroup_root)
        if members is None:
            return Verdict(candidate, Action.KEEP, Reason.CGROUP_UNREADABLE)
        if not scope_is_orphaned(members):
            return Verdict(candidate, Action.KEEP, Reason.CONTROLLER_ALIVE)
        reason = Reason.ORPHANED
    if not apply:
        return Verdict(candidate, Action.RECLAIM, reason)
    done = reset_failed(unit) if reason is Reason.UNIT_FAILED else reap_scope(unit)
    return Verdict(candidate, Action.RECLAIMED if done else Action.FAILED, reason)


def scope_pass(cgroup_root: Path, apply: bool) -> list[Verdict]:
    if shutil.which("systemctl") is None:
        print("verify-scope: systemctl not found, skipped")
        return []
    units = scope_units()
    if units is None:
        print("verify-scope: systemctl list-units failed, skipped")
        return []
    return [judge_scope(unit, state, cgroup_root, apply) for unit, state in units]


# --- per-repo pass ----------------------------------------------------------

@dataclass(frozen=True)
class RepoReport:
    repo: Path
    verdicts: tuple[Verdict, ...]
    pgid_total: int
    pgid_dead: int
    free_before: int
    free_after: int | None

    def summary_lines(self) -> list[str]:
        records = (f"pgid-records\t{self.repo}\ttotal={self.pgid_total}\t"
                   f"dead={self.pgid_dead}\tleft for task 4196")
        if self.free_after is None:
            free = f"free\t{self.repo}\tnow={self.free_before}"
        else:
            free = f"free\t{self.repo}\tbefore={self.free_before}\tafter={self.free_after}"
        return [records, free]


def repo_pass(repo: Path, kinds: frozenset[Kind], cwds: frozenset[Path],
              min_age_seconds: float, apply: bool) -> RepoReport:
    free_before = shutil.disk_usage(repo).free
    facts = RepoFacts(repo=repo, registrations=registrations(repo),
                      pgid_records=pgid_records(repo / WORKTREE_BASE_NAME), cwds=cwds)
    judge = reclaim_worktree if apply else report_worktree
    verdicts = tuple(judge(candidate, facts, min_age_seconds)
                     for candidate in worktree_candidates(facts.base, kinds))
    if apply:
        prune_registry(repo)
    return RepoReport(
        repo=repo, verdicts=verdicts, pgid_total=len(facts.pgid_records),
        pgid_dead=sum(not record.live for record in facts.pgid_records),
        free_before=free_before, free_after=shutil.disk_usage(repo).free if apply else None)


def checkout_toplevel(repo: Path) -> Path | None:
    """*repo*'s real path when it is its own git top level, else None."""
    try:
        toplevel = git(repo, "rev-parse", "--show-toplevel").strip()
    except GitFailed:
        return None
    return real(repo) if real(toplevel) == real(repo) else None


# --- command line -------------------------------------------------------------

def non_negative_hours(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value >= 0):
        raise argparse.ArgumentTypeError(f"must be a finite number >= 0, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--repo", action="append", type=Path,
                        help=f"checkout whose .worktrees/ to scan (repeatable; default {DEFAULT_REPO})")
    parser.add_argument("--class", dest="classes", action="append", type=Kind,
                        choices=list(Kind), help="class to consider (repeatable; default all)")
    parser.add_argument("--min-age-hours", type=non_negative_hours, default=24.0,
                        help="keep a worktree touched within this many hours (default 24)")
    parser.add_argument("--apply", action="store_true", help="reclaim; default is report only")
    parser.add_argument("--host", help="run this same check on HOST over $SSH")
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"),
                        help="cgroup v2 mount, read by the verify-scope class")
    return parser


def forwarded_args(ns: argparse.Namespace) -> list[str]:
    """Every parsed option except --host, re-serialised for the remote run."""
    args = [word for repo in ns.repo or [] for word in ("--repo", str(repo))]
    args += [word for kind in ns.classes or [] for word in ("--class", str(kind))]
    args += ["--min-age-hours", repr(ns.min_age_hours), "--cgroup-root", str(ns.cgroup_root)]
    return [*args, "--apply"] if ns.apply else args


def run_remote(ns: argparse.Namespace) -> int:
    """Pipe this file to `python3 -I -` on the host and relay its output and exit code."""
    ssh = shlex.split(os.environ.get("SSH", DEFAULT_SSH))
    remote = ["python3", "-I", "-", *forwarded_args(ns)]
    proc = subprocess.run([*ssh, ns.host, *map(shlex.quote, remote)],
                          input=Path(__file__).read_bytes())
    if proc.returncode != 0:
        print(f"{PROG}: run on {ns.host} exited {proc.returncode}", file=sys.stderr)
    return proc.returncode


def checked_repos(requested: list[Path] | None) -> list[Path] | None:
    """Each repo's real top level, or None after naming the first that is not one."""
    repos = []
    for repo in requested or [DEFAULT_REPO]:
        toplevel = checkout_toplevel(repo)
        if toplevel is None:
            print(f"{PROG}: --repo {repo} is not a git checkout's top level", file=sys.stderr)
            return None
        repos.append(toplevel)
    return repos


def run_local(ns: argparse.Namespace) -> int:
    kinds = frozenset(ns.classes or Kind)
    repos = checked_repos(ns.repo) if kinds & WORKTREE_KINDS else []
    if repos is None:
        return EXIT_USAGE
    cwds = process_cwds() if repos else frozenset()
    reports = [repo_pass(repo, kinds, cwds, ns.min_age_hours * 3600, ns.apply)
               for repo in repos]
    verdicts = [verdict for report in reports for verdict in report.verdicts]
    if Kind.VERIFY_SCOPE in kinds:
        verdicts += scope_pass(ns.cgroup_root, ns.apply)
    for verdict in sorted(verdicts, key=lambda v: (v.candidate.kind, v.candidate.target)):
        print(verdict.line())
    for report in reports:
        print("\n".join(report.summary_lines()))
    failed = any(verdict.action is Action.FAILED for verdict in verdicts)
    return EXIT_SOME_FAILED if failed else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    return run_remote(ns) if ns.host else run_local(ns)


if __name__ == "__main__":
    sys.exit(main())
