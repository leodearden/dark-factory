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
import fcntl
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DEFAULT_REPO = Path("/home/leo/src/dark-factory")
WORKTREE_BASE_NAME = ".worktrees"
PGID_DIR_NAME = ".merge_verify_pgids"
PGID_WORKTREE_SUFFIX = ".worktree"
MERGE_WORKTREE_NAME = re.compile(r"_merge-[0-9a-f]{8}")
INFRA_BAND_PREFIXES = ("_", ".")
GIT_LOCATION_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")
GIT_TIMEOUT_SECONDS = 120
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
        if kind in kinds and entry.is_dir(follow_symlinks=False):
            found.append(Candidate(kind, str(real(base) / entry.name)))
    return found


def lane_lock_path(worktree: Path) -> Path:
    """The `<dir>.lock` sibling, verify_cancel.py::lane_lock_path's convention."""
    return worktree.with_name(worktree.name + ".lock")


def lock_is_held(lock: Path) -> bool:
    """Probe an existing lane lock without ever creating one."""
    try:
        fd = os.open(lock, os.O_RDONLY)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
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
              min_age_seconds: float) -> RepoReport:
    free_before = shutil.disk_usage(repo).free
    facts = RepoFacts(repo=repo, registrations=registrations(repo),
                      pgid_records=pgid_records(repo / WORKTREE_BASE_NAME), cwds=cwds)
    verdicts = tuple(report_worktree(candidate, facts, min_age_seconds)
                     for candidate in worktree_candidates(facts.base, kinds))
    return RepoReport(
        repo=repo, verdicts=verdicts, pgid_total=len(facts.pgid_records),
        pgid_dead=sum(not record.live for record in facts.pgid_records),
        free_before=free_before, free_after=None)


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
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.splitlines()[0])
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


def run_local(ns: argparse.Namespace) -> int:
    kinds = frozenset(ns.classes or Kind)
    repos: list[Path] = []
    if kinds & WORKTREE_KINDS:
        for repo in ns.repo or [DEFAULT_REPO]:
            toplevel = checkout_toplevel(repo)
            if toplevel is None:
                print(f"{PROG}: --repo {repo} is not a git checkout's top level", file=sys.stderr)
                return EXIT_USAGE
            repos.append(toplevel)
    cwds = process_cwds()
    reports = [repo_pass(repo, kinds, cwds, ns.min_age_hours * 3600) for repo in repos]
    verdicts = [verdict for report in reports for verdict in report.verdicts]
    for verdict in sorted(verdicts, key=lambda v: (v.candidate.kind, v.candidate.target)):
        print(verdict.line())
    for report in reports:
        print("\n".join(report.summary_lines()))
    failed = any(verdict.action is Action.FAILED for verdict in verdicts)
    return EXIT_SOME_FAILED if failed else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    return run_local(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
