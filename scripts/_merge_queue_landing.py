"""Landing new content for a set of files on a target project's main through
that target's own merge queue, rather than by a direct commit to its main.

A direct commit to a target's main is drift its queue never verified, which
forces every merge in flight there to re-verify in full
(orchestrator/src/orchestrator/merge_lane/gates.py::_disjoint_skip_blockers).
A config-deploy script lands its knob edit this way before it hot-reloads
(scripts/merge-deep-set-cap.sh, scripts/merge-pytest-n-ab-switch.sh), and a
docs session lands its new or edited documents the same way:

    landing = land_files(repo, {"plans/foo-prd.md": prd, "plans/foo.yaml": manifest},
                         base_sha, port=port, label="docs-prd-foo", message=message)

lands every path as ONE commit through ONE merge request on the queue of the
orchestrator whose escalation MCP listens on 127.0.0.1:*port*. `land_file` is
its single-path form.

Paths are relative to the checkout *repo*; an absolute path inside it is
accepted. The content is the caller's, authored outside the target checkout,
and is never read from it. Nothing here writes the checkout, and its state
outside the landed paths is left alone; at a landed path it must be clean,
because the lane's advance would refuse to move main over uncommitted state
there, halting the whole queue, or overwrite it.

Refused before any queue call: an empty set; a path outside the checkout,
through a symlink in it, with a .git component, or the checkout itself; two
spellings of one path; a path whose content equals its copy at *base_sha* (a
path the base lacks is a new file); all INVALID_CHANGE. Uncommitted state at
a landed path is CHECKOUT_DIRTY. A path the target ignores is refused by git
itself, as GIT_FAILED, and a write the host refuses in the scratch
directory is SCRATCH_WRITE_FAILED.

A confirmed landing returns `Landing`, whose `landed_as` is the commit the
queue landed it as. Any other outcome raises `LandingNotConfirmed`: `.failure`
is a wire tag a caller may print, `.state` the last merge state seen, and
`.reason` the queue's own stated reason for a terminal refusal.

Every landing takes the normal lane, unverified, so it pays the target's full
merge gate; nothing is special-cased for docs.

The change is committed on a short-lived `task/deploy-<label>-<UTC stamp>`
branch, in a worktree outside the target checkout, submitted through
merge_request, and polled by the request's own handle until it settles. The
landing is then confirmed by git: the deploy commit is on main, and the
target checkout's copy of every landed path is main's.

The deploy branch is torn down once its request can no longer land. A request
still in flight when the deadline passes, or when SIGTERM/SIGINT arrives, is
cancelled first; if the queue refuses that cancel, the branch and its worktree
are kept, because the merge worker reads the request's source from them. A
submit whose reply the transport lost may still have been accepted, so the
queue is then asked about the deploy branch by its unique name, and the
request it names is waited for like any other.

The calling heredoc's side of the contract (argv-only inputs, the script's
directory on sys.path, the checkout venv's interpreter) is
scripts/_config_reload_gate.py's.
"""
from __future__ import annotations

import contextlib
import enum
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _config_reload_gate import load_transport, transport_not_importable_remedy

try:
    from shared.merge_state import (
        LIVE_STATES,
        SUBMIT_NON_TERMINAL,
        SUBMIT_TERMINAL,
        MergeState,
        MergeSubmitStatus,
    )
except ImportError as exc:
    # Not stdlib. Under the venvless fallback interpreter this fails, and
    # _MergeQueue.connect turns it into the remedy before any name is used.
    _MERGE_STATE_UNAVAILABLE: ImportError | None = exc
else:
    _MERGE_STATE_UNAVAILABLE = None

DEFAULT_LANDING_TIMEOUT_SECS = 10800
SUBMIT_WAIT_SECS = 100
# merge_request may hold its reply for SUBMIT_WAIT_SECS before answering.
SUBMIT_REPLY_TIMEOUT_SECS = 130.0
POLL_REPLY_TIMEOUT_SECS = 30.0
FIRST_BACKOFF_SECS = 15.0
MAX_POLL_DELAY_SECS = 60.0

_LOG_PREFIX = "merge-queue-landing:"
_STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class LandingFailure(enum.StrEnum):
    """Why a landing was not confirmed."""

    TRANSPORT_NOT_IMPORTABLE = "transport_not_importable"
    TRANSPORT_FAILED = "transport_failed"
    SUBMIT_REJECTED = "submit_rejected"
    NOT_LANDED = "not_landed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    QUEUE_DONE_NOT_ON_MAIN = "queue_done_not_on_main"
    CHECKOUT_NOT_SYNCED = "checkout_not_synced"
    CHECKOUT_DIRTY = "checkout_dirty"
    INVALID_CHANGE = "invalid_change"
    SCRATCH_WRITE_FAILED = "scratch_write_failed"
    GIT_FAILED = "git_failed"


class LandingNotConfirmed(Exception):
    """The content did not provably land: *failure* names why, *detail* tells
    the operator, *state* is the last merge state seen (None before any), and
    *reason* is the queue's own stated reason for a terminal refusal (None
    when it gave none, or the refusal is not the queue's)."""

    def __init__(self, failure: LandingFailure, detail: str, state: str | None = None, *,
                 reason: str | None = None):
        super().__init__(detail)
        self.failure = failure
        self.detail = detail
        self.state = state
        self.reason = reason


class _GitFailed(LandingNotConfirmed):
    """A git step of the landing failed; no merge state follows from it."""

    def __init__(self, detail: str):
        super().__init__(LandingFailure.GIT_FAILED, detail)


class _ReplyLost(LandingNotConfirmed):
    """A queue call whose reply the transport lost after the request may have
    reached the tool: whether it ran is unknown, and so is the merge state."""

    def __init__(self, detail: str):
        super().__init__(LandingFailure.TRANSPORT_FAILED, detail, MergeState.unknown)


@dataclass(frozen=True)
class Landing:
    """A confirmed landing: the request that carried it (None when the queue's
    answer named none), the deploy branch and commit, and the commit the queue
    reports it landed as."""

    request_id: str | None
    branch: str
    tip_sha: str
    landed_as: str


def land_files(
    repo: str, files: Mapping[str, bytes], base_sha: str, *,
    port: str, label: str, message: str,
    timeout_secs: float | str | None = None, main_branch: str = "main",
    clock: Callable[[], float] = time.monotonic,
) -> Landing:
    """Land each of *files*' content as its path (relative to the checkout
    *repo*) on *main_branch*, as one commit branching from *base_sha*, through
    the merge queue of the orchestrator whose escalation MCP listens on
    127.0.0.1:*port*.

    *timeout_secs* bounds the wait for the queue, defaulting to
    DEFAULT_LANDING_TIMEOUT_SECS. A string is read as a number, so a caller
    can pass an environment variable through unparsed. The deadline is checked
    between polls: a queue call in flight when it passes runs to its own reply
    timeout (SUBMIT_REPLY_TIMEOUT_SECS for the submit, POLL_REPLY_TIMEOUT_SECS
    for a poll), and a request still unsettled is then cancelled
    (POLL_REPLY_TIMEOUT_SECS more), so land_files can return up to about 190 s,
    plus its git teardown, after the deadline. *clock* is the monotonic clock
    the deadline is measured on, time.monotonic by default.
    """
    queue = _MergeQueue.connect(port)
    target = _Target.of(Path(repo), files, main_branch)
    target.check_change(base_sha)
    target.check_checkout_clean()
    deadline = _Deadline.after(float(timeout_secs or DEFAULT_LANDING_TIMEOUT_SECS), clock)
    with (
        _stop_signals_held() as stop,
        _deploy_branch(target, base_sha, label=label, message=message) as deploy,
    ):
        return _land(queue, target, deploy, deadline, stop, description=message)


def land_file(
    repo: str, path: str, base_sha: str, content: bytes, *,
    port: str, label: str, message: str,
    timeout_secs: float | str | None = None, main_branch: str = "main",
    clock: Callable[[], float] = time.monotonic,
) -> Landing:
    """`land_files` of *content* as the one *path*."""
    return land_files(
        repo, {path: content}, base_sha, port=port, label=label, message=message,
        timeout_secs=timeout_secs, main_branch=main_branch, clock=clock,
    )


def _land(queue: _MergeQueue, target: _Target, deploy: _DeployBranch, deadline: _Deadline,
          stop: _StopSignals, *, description: str) -> Landing:
    stop.check()
    try:
        submitted = queue.submit(deploy, description=description)
    except _ReplyLost as lost:
        print(f"{_LOG_PREFIX} {lost.detail}; asking the queue about {deploy.branch} instead",
              file=sys.stderr)
        pending = _Pending.after_lost_submit(deploy)
    else:
        if not _submit_pending(submitted, target, deploy):
            return _confirmed(target, deploy, submitted)
        pending = _Pending.submitted(submitted, deploy)
    return _confirmed(target, deploy,
                      _poll_until_landed(queue, target, deploy, pending, deadline, stop))


# ---------------------------------------------------------------------------
# Stopping: the deadline and the stop signals
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Deadline:
    """The moment *at* which the landing's ceiling passes, on the monotonic
    *clock* it was taken from."""

    at: float
    clock: Callable[[], float]

    @classmethod
    def after(cls, timeout_secs: float, clock: Callable[[], float]) -> _Deadline:
        return cls(clock() + timeout_secs, clock)

    def remaining(self) -> float:
        return self.at - self.clock()


class _CeilingReached(Exception):
    """The landing's deadline passed with its request still unsettled."""

    failure = LandingFailure.TIMED_OUT


class _Interrupted(BaseException):
    """SIGTERM or SIGINT arrived. A BaseException, so that no transport-fault
    handler can swallow it."""

    failure = LandingFailure.INTERRUPTED

    def __init__(self, received: signal.Signals):
        super().__init__(f"interrupted by {received.name}")


class _StopSignals:
    """SIGTERM and SIGINT, held until the landing can act on them.

    A stop that arrives while the landing waits between polls interrupts the
    wait. One that arrives during a queue call or a git step is held until the
    step returns: cut mid-call, a submit would leave a request the queue may
    have accepted with no id to cancel, and an interrupted connection can
    outlive its call. `check` raises a held stop at a safe point.
    """

    def __init__(self) -> None:
        self.received: signal.Signals | None = None
        self._waiting = False

    def check(self) -> None:
        if self.received is not None:
            raise _Interrupted(self.received)

    def wait(self, seconds: float) -> None:
        self._waiting = True
        try:
            self.check()
            time.sleep(seconds)
        finally:
            self._waiting = False

    def handle(self, signum: int, frame: object) -> None:
        # Later stop signals are ignored: the cancel and teardown the first one
        # starts must run to the end.
        for sig in _STOP_SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        self.received = signal.Signals(signum)
        if self._waiting:
            raise _Interrupted(self.received)


@contextlib.contextmanager
def _stop_signals_held() -> Iterator[_StopSignals]:
    """SIGTERM and SIGINT are a _StopSignals' for the duration, so a killpg
    or a unit stop cancels the request instead of killing the process with it
    live. One escaping this block as _Interrupted arrived before any request
    was submitted. One still held when the landing returns is re-delivered to
    the previous handler; a refusal already fails the caller."""
    stop = _StopSignals()
    previous = {sig: signal.signal(sig, stop.handle) for sig in _STOP_SIGNALS}
    try:
        yield stop
    except _Interrupted as interrupted:
        raise LandingNotConfirmed(
            LandingFailure.INTERRUPTED, f"{interrupted} before any merge request was submitted",
        ) from None
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if stop.received is not None:
        signal.raise_signal(stop.received)


def _sleep_before_poll(delay: float, deadline: _Deadline, stop: _StopSignals) -> None:
    remaining = deadline.remaining()
    if remaining <= 0:
        raise _CeilingReached("the landing timed out")
    stop.wait(min(delay, remaining))


# ---------------------------------------------------------------------------
# The target checkout and the deploy branch
# ---------------------------------------------------------------------------

def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True)


def _git(cwd: Path, *args: str) -> str:
    """git's stdout, stripped; raising _GitFailed with git's own stderr on
    failure."""
    proc = _run_git(cwd, *args)
    if proc.returncode != 0:
        raise _GitFailed(
            f"git -C {cwd} {' '.join(args)} failed (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout.decode().strip()


@dataclass(frozen=True)
class _Target:
    """The target project's checkout, its main, and the change being landed:
    each path's new content, in path order."""

    repo: Path
    main_branch: str
    changes: tuple[tuple[str, bytes], ...]

    @classmethod
    def of(cls, repo: Path, files: Mapping[str, bytes], main_branch: str) -> _Target:
        """The change *files* make to *repo*, every path made repo-relative:
        INVALID_CHANGE for an empty set, a path _repo_relative refuses, and
        two spellings of one path."""
        if not files:
            raise LandingNotConfirmed(
                LandingFailure.INVALID_CHANGE, f"nothing to land: no paths given for {repo}",
            )
        spelled: dict[str, str] = {}
        changes: dict[str, bytes] = {}
        for given, content in files.items():
            relative = _repo_relative(repo, given)
            if relative in spelled:
                raise LandingNotConfirmed(
                    LandingFailure.INVALID_CHANGE,
                    f"{spelled[relative]} and {given} both name {relative} in {repo}",
                )
            spelled[relative], changes[relative] = given, content
        return cls(repo, main_branch, tuple(sorted(changes.items())))

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(path for path, _ in self.changes)

    def check_change(self, base_sha: str) -> None:
        """Raise INVALID_CHANGE unless *base_sha* is a commit and every path's
        content differs from the copy *base_sha* carries, if any: a path it
        lacks is a new file."""
        if _run_git(self.repo, "rev-parse", "--verify", "--quiet",
                    f"{base_sha}^{{commit}}").returncode != 0:
            raise LandingNotConfirmed(
                LandingFailure.INVALID_CHANGE, f"base {base_sha} is not a commit in {self.repo}",
            )
        unchanged = [path for path, content in self.changes
                     if self.blob(base_sha, path) == content]
        if unchanged:
            raise LandingNotConfirmed(
                LandingFailure.INVALID_CHANGE,
                f"nothing to land for {', '.join(unchanged)}: base {base_sha} already carries "
                f"this content",
            )

    def check_checkout_clean(self) -> None:
        """Raise CHECKOUT_DIRTY when any landed path carries uncommitted state
        in the checkout."""
        dirty = self.dirty_paths()
        if dirty:
            raise LandingNotConfirmed(
                LandingFailure.CHECKOUT_DIRTY,
                f"the checkout {self.repo} carries uncommitted state at {', '.join(dirty)}: "
                f"the merge lane would refuse to move {self.main_branch} over it (wip_overlap, "
                f"which halts the whole queue) or overwrite it; pass content authored outside "
                f"the checkout, and leave the landed paths clean there",
            )

    def dirty_paths(self) -> list[str]:
        """The landed paths carrying staged, unstaged or untracked state in the
        checkout, read without taking its index.lock: the merge worker treats
        a held lock as contention."""
        return [path for path in self.paths
                if _git(self.repo, "--no-optional-locks", "status", "--porcelain",
                        "--untracked-files=all", "--", path)]

    def write_into(self, worktree: Path) -> None:
        """Write every path's new content into *worktree*, the base checked
        out. A path git's tree cannot hold there as a file (under a file, over
        a directory, through a symlink the base carries but the target
        checkout no longer does) is INVALID_CHANGE; any other write the host
        refuses is SCRATCH_WRITE_FAILED."""
        for path, content in self.changes:
            destination = worktree / path
            if _crosses_symlink(worktree, path):
                raise LandingNotConfirmed(
                    LandingFailure.INVALID_CHANGE,
                    f"{path} crosses a symlink at the base, so it would be written somewhere "
                    f"other than the path it names",
                )
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            except (IsADirectoryError, NotADirectoryError, FileExistsError) as exc:
                raise LandingNotConfirmed(
                    LandingFailure.INVALID_CHANGE, f"{path} cannot be written as a file: {exc}",
                ) from exc
            except OSError as exc:
                raise LandingNotConfirmed(
                    LandingFailure.SCRATCH_WRITE_FAILED,
                    f"the host refused writing {path} into the deploy worktree {worktree}: {exc}",
                ) from exc

    def blob(self, rev: str, path: str) -> bytes | None:
        """*path*'s content at *rev*, or None when *rev* has no such file."""
        proc = _run_git(self.repo, "cat-file", "blob", f"{rev}:{path}")
        return proc.stdout if proc.returncode == 0 else None

    def carries(self, tip: str) -> bool:
        """Whether main holds *tip*: as an ancestor, or as an equal patch (the
        queue may land a rebased copy)."""
        if _run_git(self.repo, "merge-base", "--is-ancestor", tip, self.main_branch).returncode == 0:
            return True
        cherry = _git(self.repo, "cherry", self.main_branch, tip).splitlines()
        return bool(cherry) and all(line.startswith("-") for line in cherry)

    def unsynced_paths(self) -> list[str]:
        """The landed paths whose copy on disk in the checkout is not main's:
        what a reload, or any reader of the checkout, would see instead."""
        return [path for path in self.paths
                if not _holds(self.repo / path, self.blob(self.main_branch, path))]

    def main_tip(self) -> str:
        return _git(self.repo, "rev-parse", self.main_branch)


def _holds(on_disk: Path, content: bytes | None) -> bool:
    return content is not None and on_disk.is_file() and on_disk.read_bytes() == content


def _repo_relative(repo: Path, path: str) -> str:
    """*path*, relative to the checkout *repo* or absolute inside it, as a
    repo-relative POSIX path. INVALID_CHANGE when it names the checkout
    itself or anything outside it, has a .git component, or crosses a symlink
    in it."""
    root = repo.absolute()
    try:
        relative = Path(os.path.normpath(root / path)).relative_to(root)
    except ValueError:
        problem = f"is outside the checkout {repo}"
    else:
        if not relative.parts:
            problem = f"names the checkout {repo} itself, not a file in it"
        elif any(part.lower() == ".git" for part in relative.parts):
            problem = "has a .git component (in some letter case), which git refuses to commit"
        elif _crosses_symlink(repo, relative.as_posix()):
            problem = (f"crosses a symlink in the checkout {repo}, so it would be written "
                       f"somewhere other than the path it names")
        else:
            return relative.as_posix()
    raise LandingNotConfirmed(LandingFailure.INVALID_CHANGE, f"{path} {problem}")


def _crosses_symlink(root: Path, relative: str) -> bool:
    resolved_root = root.resolve()
    return (resolved_root / relative).resolve() != resolved_root / relative


@dataclass(frozen=True)
class _DeployBranch:
    """The deploy commit, on its own branch, checked out in a worktree under a
    scratch directory outside the target checkout."""

    repo: Path
    scratch: Path
    branch: str
    tip: str

    @classmethod
    def create(cls, target: _Target, base_sha: str, *, label: str, message: str) -> _DeployBranch:
        scratch = Path(tempfile.mkdtemp(prefix="deploy-landing-"))
        branch = f"task/deploy-{label}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        worktree = _worktree_in(scratch)
        try:
            # Its own step, not `worktree add -b`: that leaves the branch it
            # made when the worktree fails, and fails the same way when a
            # branch of that name already exists, which is not ours to delete.
            _git(target.repo, "branch", "--no-track", branch, base_sha)
        except BaseException:
            shutil.rmtree(scratch, ignore_errors=True)
            raise
        try:
            _git(target.repo, "worktree", "add", "-q", str(worktree), branch)
            target.write_into(worktree)
            # Staged first: `commit -- <path>` refuses a path git does not track yet.
            _git(worktree, "add", "--", *target.paths)
            _git(worktree, "commit", "-q", "--no-verify", "-m", message, "--", *target.paths)
            tip = _git(worktree, "rev-parse", "HEAD")
        except BaseException:
            _tear_down(target.repo, scratch, branch)
            raise
        return cls(target.repo, scratch, branch, tip)

    @property
    def worktree(self) -> Path:
        return _worktree_in(self.scratch)

    @property
    def bare_id(self) -> str:
        """The branch without its `task/` prefix: the id the queue knows it by."""
        return self.branch.removeprefix("task/")

    @property
    def status_handle(self) -> dict[str, str]:
        """The merge_status arguments naming this branch's most recent
        request: the branch is unique, so that request is this deploy's."""
        return {"branch": self.bare_id}

    def tear_down(self) -> None:
        _tear_down(self.repo, self.scratch, self.branch)

    def retained(self, refusal: LandingNotConfirmed) -> LandingNotConfirmed:
        """*refusal*, naming what was kept for a request that may still land."""
        return LandingNotConfirmed(
            refusal.failure,
            f"{refusal.detail}; kept the deploy worktree {self.worktree} and branch "
            f"{self.branch}, which the request may still land from: if it does, its change "
            f"reaches main unconfirmed (a config change then goes live at the next reload or "
            f"restart)",
            refusal.state,
            reason=refusal.reason,
        )


@contextlib.contextmanager
def _deploy_branch(target: _Target, base_sha: str, *,
                   label: str, message: str) -> Iterator[_DeployBranch]:
    """The deploy branch, torn down on the way out unless a refusal says its
    request may still land."""
    deploy = _DeployBranch.create(target, base_sha, label=label, message=message)
    retained = False
    try:
        yield deploy
    except LandingNotConfirmed as refusal:
        retained = _may_still_land(refusal.state)
        if retained:
            raise deploy.retained(refusal) from None
        raise
    finally:
        if not retained:
            deploy.tear_down()


def _worktree_in(scratch: Path) -> Path:
    return scratch / "worktree"


def _tear_down(repo: Path, scratch: Path, branch: str) -> None:
    """Remove the deploy worktree, its branch and the scratch directory,
    whichever of them still exist. A step git refuses is reported on stderr,
    never raised: tidying up must not replace the landing's own outcome."""
    worktree = _worktree_in(scratch)
    removals: list[tuple[str, ...]] = (
        [("worktree", "remove", "--force", str(worktree))] if worktree.exists() else []
    )
    removals.append(("worktree", "prune"))
    if _run_git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}").returncode == 0:
        removals.append(("branch", "-D", branch))
    for args in removals:
        proc = _run_git(repo, *args)
        if proc.returncode != 0:
            print(f"{_LOG_PREFIX} could not tear down {branch} in {repo}: git {' '.join(args)} "
                  f"failed (rc={proc.returncode}): {proc.stderr.decode(errors='replace').strip()}",
                  file=sys.stderr)
    shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# The merge queue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MergeQueue:
    """The target's merge queue, reached through its escalation MCP.

    The MCP is STATEFUL, so every call opens a `shared.mcp_session`
    `SyncMcpSession`, as reload_config's does. *reply_lost* are the transport
    faults after which the request may have reached the tool.
    """

    base_url: str
    session: Any
    reply_lost: tuple[type[Exception], ...]

    @classmethod
    def connect(cls, port: str) -> _MergeQueue:
        try:
            mcp_session, httpx = load_transport()
        except ImportError as exc:
            raise _not_importable(exc) from exc
        if _MERGE_STATE_UNAVAILABLE is not None:
            raise _not_importable(_MERGE_STATE_UNAVAILABLE) from _MERGE_STATE_UNAVAILABLE
        # A read that timed out or broke, or a server that hung up without
        # answering, can follow a request the server received in full. A
        # fault in the handshake reads the same, so it is treated the same:
        # asking the queue afterwards costs one poll.
        reply_lost = (httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError)
        return cls(f"http://127.0.0.1:{port}", mcp_session.SyncMcpSession, reply_lost)

    def submit(self, deploy: _DeployBranch, *, description: str) -> dict:
        return self._call("merge_request", {
            "task_id": deploy.bare_id,
            "branch": deploy.bare_id,
            "worktree": str(deploy.worktree),
            "description": description,
            "wait_secs": SUBMIT_WAIT_SECS,
            "verified_green": False,
            "lane": "normal",
        }, timeout=SUBMIT_REPLY_TIMEOUT_SECS)

    def poll(self, handle: dict[str, str]) -> dict | None:
        """merge_status for *handle*, or None when the transport dropped the
        poll: a target restart mid-verify must not abandon a merge that will
        still land."""
        try:
            return self._call("merge_status", handle, timeout=POLL_REPLY_TIMEOUT_SECS)
        except LandingNotConfirmed as fault:
            print(f"{_LOG_PREFIX} a poll was dropped; polling on: {fault.detail}", file=sys.stderr)
            return None

    def cancel(self, request_id: str) -> dict:
        return self._call("merge_cancel", {"request_id": request_id},
                          timeout=POLL_REPLY_TIMEOUT_SECS)

    def _call(self, tool: str, arguments: dict, *, timeout: float) -> dict:
        try:
            with self.session(
                self.base_url, client_name="merge-queue-landing", timeout=timeout,
            ) as session:
                return session.call_tool(tool, arguments)
        except Exception as exc:
            # Broad on purpose, as in _config_reload_gate: every transport
            # fault leaves the queue's answer unknown.
            if isinstance(exc.__cause__, self.reply_lost):
                raise _ReplyLost(
                    f"{tool} sent to {self.base_url} got no reply, so it may have run: {exc}"
                ) from exc
            raise LandingNotConfirmed(
                LandingFailure.TRANSPORT_FAILED,
                f"{tool} never reached the tool at {self.base_url}: {type(exc).__name__}: {exc}",
            ) from exc


def _not_importable(missing: ImportError) -> LandingNotConfirmed:
    return LandingNotConfirmed(
        LandingFailure.TRANSPORT_NOT_IMPORTABLE, transport_not_importable_remedy(missing),
    )


def _submit_pending(submitted: dict, target: _Target, deploy: _DeployBranch) -> bool:
    """Whether the submitted request must now be polled. False when the submit
    settled it and git finds the deploy commit on main, or the queue says it
    landed (which _confirmed then holds to git). Raises for any other
    settlement."""
    status = submitted.get("status")
    if "error" in submitted or (status not in SUBMIT_NON_TERMINAL and status not in SUBMIT_TERMINAL):
        code = f" (code {submitted['code']})" if submitted.get("code") else ""
        raise LandingNotConfirmed(
            LandingFailure.SUBMIT_REJECTED,
            f"merge_request refused the submission: {submitted.get('error') or submitted}{code}",
            reason=submitted.get("error") or None,
        )
    if status in SUBMIT_NON_TERMINAL:
        return True
    if _submit_says_landed(status) or target.carries(deploy.tip):
        return False
    raise _not_landed(submitted, state=status)


def _submit_says_landed(status: str | None) -> bool:
    """Whether a settled submit *status* reports the merge landed: the ones
    merge_status_answer.py::map_terminal_state serves as done."""
    return status in (MergeSubmitStatus.done, MergeSubmitStatus.already_merged,
                      MergeSubmitStatus.done_wip_recovery)


@dataclass(frozen=True)
class _Pending:
    """A request still to be polled: the merge_status handle naming it, the
    state it was last seen in, the queue's estimate of the wait (None when it
    gave none), and the wait before the first poll (None: that estimate, else
    the backoff)."""

    handle: dict[str, str]
    state: str
    eta: float | None
    first_wait: float | None

    @classmethod
    def submitted(cls, reply: dict, deploy: _DeployBranch) -> _Pending:
        eta = _eta(reply)
        return cls(_poll_handle(reply, deploy), reply["status"], eta, eta)

    @classmethod
    def after_lost_submit(cls, deploy: _DeployBranch) -> _Pending:
        """A submit whose reply was lost: asked at once, by the deploy
        branch, what the queue made of it."""
        return cls(deploy.status_handle, MergeState.unknown, None, 0.0)


def _poll_until_landed(queue: _MergeQueue, target: _Target, deploy: _DeployBranch,
                       pending: _Pending, deadline: _Deadline, stop: _StopSignals) -> dict:
    """The merge_status reply that shows the *pending* request landed,
    cancelling the request if the deadline, a stop signal or a git failure
    comes first."""
    handle, state, eta = pending.handle, pending.state, pending.eta
    request_id = handle.get("request_id")
    wait, backoff = pending.first_wait, _backoff()
    try:
        while True:
            _sleep_before_poll(wait if wait is not None else next(backoff), deadline, stop)
            reply = queue.poll(handle)
            if reply is not None:
                state, eta = reply.get("state"), _eta(reply)
                request_id = handle.get("request_id") or reply.get("request_id") or request_id
                next_handle = _next_handle(reply, handle, target, deploy)
                if next_handle is None:
                    return reply
                handle = next_handle
            wait = eta
    except (_CeilingReached, _Interrupted, _GitFailed) as stopped_by:
        raise _stopped(queue, request_id, handle, state, stopped_by) from None


def _poll_handle(submitted: dict, deploy: _DeployBranch) -> dict[str, str]:
    """The merge_status arguments naming the submitted request, by the handle
    an attached submit names in `poll_by`; one that names neither a request
    nor a task is polled by the deploy branch."""
    poll_by = submitted.get("poll_by", "request_id")
    if poll_by == "task_id":
        return {"task_id": submitted["inflight_task_id"]}
    if poll_by == "branch":
        return deploy.status_handle
    return {"request_id": submitted["request_id"]}


def _next_handle(reply: dict, handle: dict[str, str], target: _Target,
                 deploy: _DeployBranch) -> dict[str, str] | None:
    """The handle to poll next, or None once *reply* shows the request
    landed: the queue answers done, or, in any state but a live one, git finds
    the deploy commit on main. Raises when it settled any other way."""
    state = reply.get("state")
    if state == MergeState.done:
        return None
    if state in LIVE_STATES:
        return handle
    if target.carries(deploy.tip):
        return None
    if state == MergeState.superseded:
        return _successor(reply)
    if _may_still_land(state):
        return handle
    raise _not_landed(reply, state=state)


def _successor(reply: dict) -> dict[str, str]:
    """The handle of the next-generation request that superseded this one.

    An MCP submission is never absorbed into a coalesce train (no workflow
    awaits it), so a coalesce-* successor is a shape this module does not
    chase: it fails closed."""
    successor = reply.get("superseded_by") or ""
    if successor.startswith("mr-"):
        return {"request_id": successor}
    raise _not_landed(reply, state=MergeState.superseded)


def _may_still_land(state: str | None) -> bool:
    """Whether a request last seen in *state* may still land: it is in
    flight, or the server cannot rule that out."""
    return (state in LIVE_STATES or state in SUBMIT_NON_TERMINAL
            or state in (MergeState.journaled, MergeState.unknown))


def _not_landed(reply: dict, *, state: str | None) -> LandingNotConfirmed:
    request_id = reply.get("request_id")
    subject = f"merge request {request_id}" if request_id else "the deploy's merge request"
    reason = f" ({reply['reason']})" if reply.get("reason") else ""
    successor = f" superseded_by={reply['superseded_by']}" if reply.get("superseded_by") else ""
    return LandingNotConfirmed(
        LandingFailure.NOT_LANDED,
        f"{subject} settled {state}{reason}{successor}: nothing landed",
        state,
        reason=reply.get("reason") or None,
    )


def _stopped(queue: _MergeQueue, request_id: str | None, handle: dict[str, str],
             state: str | None,
             stopped_by: _CeilingReached | _Interrupted | _GitFailed) -> LandingNotConfirmed:
    """The refusal for a poll *stopped_by* the deadline, a stop signal or a
    git failure with the request still *state*, after asking the queue to
    cancel it."""
    stopped = f"{stopped_by} with merge request {request_id or handle} still {state}"
    if request_id is None:
        return LandingNotConfirmed(
            stopped_by.failure, f"{stopped}; it has no request id to cancel", state,
        )
    try:
        reply = queue.cancel(request_id)
    except LandingNotConfirmed as fault:
        return LandingNotConfirmed(stopped_by.failure, f"{stopped}; {fault.detail}", state)
    code = f" code={reply['code']}" if reply.get("code") else ""
    return LandingNotConfirmed(
        stopped_by.failure,
        f"{stopped}; merge_cancel answered cancelled={reply.get('cancelled')} "
        f"state={reply.get('state')}{code}",
        reply.get("state") or state,
    )


def _eta(reply: dict) -> float | None:
    """The queue's own positive estimate of the wait, capped; else None."""
    eta = reply.get("eta_seconds")
    if isinstance(eta, int | float) and eta > 0:
        return min(float(eta), MAX_POLL_DELAY_SECS)
    return None


def _backoff() -> Iterator[float]:
    """15 s, 30 s, then 60 s for every wait the queue gives no estimate for."""
    delay = FIRST_BACKOFF_SECS
    while True:
        yield delay
        delay = min(delay * 2, MAX_POLL_DELAY_SECS)


# ---------------------------------------------------------------------------
# Confirming the landing
# ---------------------------------------------------------------------------

def _confirmed(target: _Target, deploy: _DeployBranch, reply: dict) -> Landing:
    """The Landing *reply* settled, once git agrees: the deploy commit is on
    main, and the checkout's copy of every landed path is main's."""
    request_id = reply.get("request_id")
    if not target.carries(deploy.tip):
        reported = reply.get("state") or reply.get("status")
        raise LandingNotConfirmed(
            LandingFailure.QUEUE_DONE_NOT_ON_MAIN,
            f"the queue reported request {request_id} {reported}, but {target.main_branch} "
            f"carries {deploy.tip} neither as an ancestor nor as an equal patch",
            MergeState.done,
        )
    unsynced = target.unsynced_paths()
    if unsynced:
        on_disk = ", ".join(str(target.repo / path) for path in unsynced)
        raise LandingNotConfirmed(
            LandingFailure.CHECKOUT_NOT_SYNCED,
            f"{deploy.branch} landed on {target.main_branch}, but on disk the checkout's "
            f"{on_disk} is not {target.main_branch}'s copy, so a reader of the checkout, a "
            f"config reload among them, would see the old content",
            MergeState.done,
        )
    return Landing(
        request_id=request_id,
        branch=deploy.branch,
        tip_sha=deploy.tip,
        landed_as=_reported_sha(reply) or target.main_tip(),
    )


def _reported_sha(reply: dict) -> str | None:
    """The commit the queue says the request landed as, when it says."""
    last_outcome = reply.get("last_outcome") or {}
    return reply.get("commit") or reply.get("merge_sha") or last_outcome.get("merge_sha")
