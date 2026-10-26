"""Landing one file's new content on a target project's main through that
target's own merge queue: the step a config-deploy script takes before it
hot-reloads (scripts/merge-deep-set-cap.sh, scripts/merge-pytest-n-ab-switch.sh).

A direct commit to a target's main is drift its queue never verified, which
forces every merge in flight there to re-verify in full
(orchestrator/src/orchestrator/merge_lane/gates.py::_disjoint_skip_blockers).

`land_file` commits the content on a short-lived `task/deploy-<label>-<UTC
stamp>` branch, in a worktree outside the target checkout, submits it through
merge_request, polls the request's own handle until it settles, and confirms
the landing by git: the deploy commit is on main, and the target checkout's
file is main's. Any other outcome raises `LandingNotConfirmed`, whose
`.failure` is a wire tag a caller may print. `land_or_exit` is the same call,
exiting through `_config_reload_gate.die`.

The deploy branch is torn down once its request can no longer land. A request
still in flight when the deadline passes, or when SIGTERM/SIGINT arrives, is
cancelled first; if the queue refuses that cancel, the branch and its worktree
are kept, because the merge worker reads the request's source from them.

The calling heredoc's side of the contract (argv-only inputs, the script's
directory on sys.path, the checkout venv's interpreter) is
scripts/_config_reload_gate.py's.
"""
from __future__ import annotations

import contextlib
import enum
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _config_reload_gate import die, load_transport, transport_not_importable_remedy

try:
    from shared.merge_state import (
        EPISTEMIC_STATES,
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
    INVALID_CHANGE = "invalid_change"


class LandingNotConfirmed(Exception):
    """The content did not provably land: *failure* names why, *detail* tells
    the operator, *state* is the last merge state seen (None before any)."""

    def __init__(self, failure: LandingFailure, detail: str, state: str | None = None):
        super().__init__(detail)
        self.failure = failure
        self.detail = detail
        self.state = state


@dataclass(frozen=True)
class Landing:
    """A confirmed landing: the request that carried it, the deploy branch
    and commit, and the commit the queue reports it landed as."""

    request_id: str
    branch: str
    tip_sha: str
    landed_as: str


def land_file(
    repo: str, path: str, base_sha: str, content: bytes, *,
    port: str, label: str, message: str,
    timeout_secs: float | str | None = None, main_branch: str = "main",
) -> Landing:
    """Land *content* as *path* (relative to the checkout *repo*) on
    *main_branch*, branching from *base_sha*, through the merge queue of the
    orchestrator whose escalation MCP listens on 127.0.0.1:*port*.

    *timeout_secs* bounds the whole landing, defaulting to
    DEFAULT_LANDING_TIMEOUT_SECS. A string is read as a number, so a caller
    can pass an environment variable through unparsed.
    """
    queue = _MergeQueue.connect(port)
    target = _Target(Path(repo), path, main_branch)
    target.check_change(base_sha, content)
    deadline = time.monotonic() + float(timeout_secs or DEFAULT_LANDING_TIMEOUT_SECS)
    with (
        _stop_signals_held() as stop,
        _deploy_branch(target, base_sha, content, label=label, message=message) as deploy,
    ):
        return _land(queue, target, deploy, deadline, stop, description=message)


def land_or_exit(repo: str, path: str, base_sha: str, content: bytes, *, prog: str,
                 **kwargs: Any) -> Landing:
    """`land_file`, exiting through `die` with the refusal's detail."""
    try:
        return land_file(repo, path, base_sha, content, **kwargs)
    except LandingNotConfirmed as exc:
        die(prog, exc.detail)


def _land(queue: _MergeQueue, target: _Target, deploy: _DeployBranch, deadline: float,
          stop: _StopSignals, *, description: str) -> Landing:
    stop.check()
    submitted = queue.submit(deploy, description=description)
    settled = (
        _poll_until_landed(queue, target, deploy, submitted, deadline, stop)
        if _submit_pending(submitted) else submitted
    )
    return _confirmed(target, deploy, settled)


# ---------------------------------------------------------------------------
# Stopping: the deadline and the stop signals
# ---------------------------------------------------------------------------

class _CeilingReached(Exception):
    """The landing's deadline passed with its request still unsettled."""


class _Interrupted(BaseException):
    """SIGTERM or SIGINT arrived. A BaseException, so that no transport-fault
    handler can swallow it."""

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


def _sleep_before_poll(delay: float, deadline: float, stop: _StopSignals) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _CeilingReached("the landing timed out")
    stop.wait(min(delay, remaining))


# ---------------------------------------------------------------------------
# The target checkout and the deploy branch
# ---------------------------------------------------------------------------

def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True)


def _git(cwd: Path, *args: str) -> str:
    """git's stdout, stripped; raising with git's own stderr on failure."""
    proc = _run_git(cwd, *args)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git -C {cwd} {' '.join(args)} failed (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout.decode().strip()


@dataclass(frozen=True)
class _Target:
    """The target project's checkout, the file being landed, and its main."""

    repo: Path
    path: str
    main_branch: str

    def check_change(self, base_sha: str, content: bytes) -> None:
        """Raise INVALID_CHANGE unless *content* changes the file *base_sha*
        carries."""
        if _run_git(self.repo, "rev-parse", "--verify", "--quiet",
                    f"{base_sha}^{{commit}}").returncode != 0:
            raise LandingNotConfirmed(
                LandingFailure.INVALID_CHANGE, f"base {base_sha} is not a commit in {self.repo}",
            )
        base_content = self.blob(base_sha)
        if base_content is None:
            raise LandingNotConfirmed(
                LandingFailure.INVALID_CHANGE, f"{self.path} is not tracked at base {base_sha}",
            )
        if base_content == content:
            raise LandingNotConfirmed(
                LandingFailure.INVALID_CHANGE,
                f"nothing to land: {self.path} at base {base_sha} already carries this content",
            )

    def blob(self, rev: str) -> bytes | None:
        """The file's content at *rev*, or None when *rev* has no such file."""
        proc = _run_git(self.repo, "show", f"{rev}:{self.path}")
        return proc.stdout if proc.returncode == 0 else None

    def carries(self, tip: str) -> bool:
        """Whether main holds *tip*: as an ancestor, or as an equal patch (the
        queue may land a rebased copy)."""
        if _run_git(self.repo, "merge-base", "--is-ancestor", tip, self.main_branch).returncode == 0:
            return True
        cherry = _git(self.repo, "cherry", self.main_branch, tip).splitlines()
        return bool(cherry) and all(line.startswith("-") for line in cherry)

    def checkout_synced(self) -> bool:
        """Whether the checkout's file on disk is main's: what a reload re-reads."""
        on_disk = self.repo / self.path
        return on_disk.is_file() and on_disk.read_bytes() == self.blob(self.main_branch)

    def main_tip(self) -> str:
        return _git(self.repo, "rev-parse", self.main_branch)


@dataclass(frozen=True)
class _DeployBranch:
    """The deploy commit, on its own branch, checked out in a worktree under a
    scratch directory outside the target checkout."""

    repo: Path
    scratch: Path
    branch: str
    tip: str

    @classmethod
    def create(cls, target: _Target, base_sha: str, content: bytes, *,
               label: str, message: str) -> _DeployBranch:
        scratch = Path(tempfile.mkdtemp(prefix="deploy-landing-"))
        branch = f"task/deploy-{label}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        worktree = _worktree_in(scratch)
        try:
            _git(target.repo, "worktree", "add", "-q", "-b", branch, str(worktree), base_sha)
            (worktree / target.path).write_bytes(content)
            _git(worktree, "commit", "-q", "--no-verify", "-m", message, "--", target.path)
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

    def tear_down(self) -> None:
        _tear_down(self.repo, self.scratch, self.branch)

    def retained(self, refusal: LandingNotConfirmed) -> LandingNotConfirmed:
        """*refusal*, naming what was kept for a request that may still land."""
        return LandingNotConfirmed(
            refusal.failure,
            f"{refusal.detail}; kept the deploy worktree {self.worktree} and branch "
            f"{self.branch}, which the request may still land from: if it lands, its value "
            f"goes live at the next reload or restart",
            refusal.state,
        )


@contextlib.contextmanager
def _deploy_branch(target: _Target, base_sha: str, content: bytes, *,
                   label: str, message: str) -> Iterator[_DeployBranch]:
    """The deploy branch, torn down on the way out unless a refusal says its
    request may still land."""
    deploy = _DeployBranch.create(target, base_sha, content, label=label, message=message)
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
    whichever of them still exist."""
    worktree = _worktree_in(scratch)
    if worktree.exists():
        _git(repo, "worktree", "remove", "--force", str(worktree))
    _git(repo, "worktree", "prune")
    if _run_git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}").returncode == 0:
        _git(repo, "branch", "-D", branch)
    shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# The merge queue
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MergeQueue:
    """The target's merge queue, reached through its escalation MCP.

    The MCP is STATEFUL, so every call goes through the shared transport
    (`legibility.census_trigger.post_mcp_tool_call`), as reload_config's does.
    """

    url: str
    post: Any

    @classmethod
    def connect(cls, port: str) -> _MergeQueue:
        missing = _MERGE_STATE_UNAVAILABLE
        try:
            census_trigger, _ = load_transport()
        except ImportError as exc:
            missing = exc
        if missing is not None:
            raise LandingNotConfirmed(
                LandingFailure.TRANSPORT_NOT_IMPORTABLE, transport_not_importable_remedy(missing),
            ) from missing
        return cls(f"http://127.0.0.1:{port}/mcp", census_trigger.post_mcp_tool_call)

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
            return self.post(self.url, tool, arguments, timeout=timeout)
        except Exception as exc:
            # Broad on purpose, as in _config_reload_gate: every transport
            # fault leaves the queue's answer unknown.
            raise LandingNotConfirmed(
                LandingFailure.TRANSPORT_FAILED,
                f"{tool} never reached the tool at {self.url}: {type(exc).__name__}: {exc}",
            ) from exc


def _submit_pending(submitted: dict) -> bool:
    """Whether the submitted request must now be polled; False when the
    submit itself settled it as landed. Raises for any other settlement."""
    status = submitted.get("status")
    if "error" in submitted or (status not in SUBMIT_NON_TERMINAL and status not in SUBMIT_TERMINAL):
        code = f" (code {submitted['code']})" if submitted.get("code") else ""
        raise LandingNotConfirmed(
            LandingFailure.SUBMIT_REJECTED,
            f"merge_request refused the submission: {submitted.get('error') or submitted}{code}",
        )
    if status in SUBMIT_NON_TERMINAL:
        return True
    if status in (MergeSubmitStatus.done, MergeSubmitStatus.already_merged):
        return False
    raise _not_landed(submitted, state=status)


def _poll_until_landed(queue: _MergeQueue, target: _Target, deploy: _DeployBranch,
                       submitted: dict, deadline: float, stop: _StopSignals) -> dict:
    """The merge_status reply that shows the submitted request landed,
    cancelling the request if the deadline or a stop signal comes first."""
    handle = _poll_handle(submitted)
    state = submitted["status"]
    eta = _eta(submitted)
    backoff = _backoff()
    try:
        while True:
            _sleep_before_poll(eta if eta is not None else next(backoff), deadline, stop)
            reply = queue.poll(handle)
            if reply is None:
                continue
            state, eta = reply.get("state"), _eta(reply)
            next_handle = _next_handle(reply, handle, target, deploy)
            if next_handle is None:
                return reply
            handle = next_handle
    except (_CeilingReached, _Interrupted) as stopped_by:
        raise _stopped(queue, handle, state, stopped_by) from None


def _poll_handle(submitted: dict) -> dict[str, str]:
    """The merge_status arguments naming the submitted request: an attached
    submit names its own handle in `poll_by`."""
    poll_by = submitted.get("poll_by", "request_id")
    if poll_by == "task_id":
        return {"task_id": submitted["inflight_task_id"]}
    if poll_by == "branch":
        return {"branch": submitted["branch"]}
    return {"request_id": submitted["request_id"]}


def _next_handle(reply: dict, handle: dict[str, str], target: _Target,
                 deploy: _DeployBranch) -> dict[str, str] | None:
    """The handle to poll next, or None once *reply* shows the request
    landed. Raises when it settled any other way."""
    state = reply.get("state")
    if state == MergeState.done:
        return None
    if state == MergeState.superseded:
        return _successor(reply)
    if state in EPISTEMIC_STATES and target.carries(deploy.tip):
        return None
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
    reason = f" ({reply['reason']})" if reply.get("reason") else ""
    successor = f" superseded_by={reply['superseded_by']}" if reply.get("superseded_by") else ""
    return LandingNotConfirmed(
        LandingFailure.NOT_LANDED,
        f"merge request {request_id} settled {state}{reason}{successor}: nothing landed",
        state,
    )


def _stopped(queue: _MergeQueue, handle: dict[str, str], state: str | None,
             stopped_by: BaseException) -> LandingNotConfirmed:
    """The refusal for a poll *stopped_by* the deadline or a stop signal with
    the request still *state*, after asking the queue to cancel it."""
    failure = (LandingFailure.TIMED_OUT if isinstance(stopped_by, _CeilingReached)
               else LandingFailure.INTERRUPTED)
    request_id = handle.get("request_id")
    stopped = f"{stopped_by} with merge request {request_id or handle} still {state}"
    if request_id is None:
        return LandingNotConfirmed(failure, f"{stopped}; it has no request id to cancel", state)
    try:
        reply = queue.cancel(request_id)
    except LandingNotConfirmed as fault:
        return LandingNotConfirmed(failure, f"{stopped}; {fault.detail}", state)
    code = f" code={reply['code']}" if reply.get("code") else ""
    return LandingNotConfirmed(
        failure,
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
    """The Landing the queue's done *reply* claims, once git agrees."""
    request_id = reply.get("request_id")
    if not target.carries(deploy.tip):
        raise LandingNotConfirmed(
            LandingFailure.QUEUE_DONE_NOT_ON_MAIN,
            f"the queue reported request {request_id} done, but {target.main_branch} carries "
            f"{deploy.tip} neither as an ancestor nor as an equal patch",
            MergeState.done,
        )
    if not target.checkout_synced():
        raise LandingNotConfirmed(
            LandingFailure.CHECKOUT_NOT_SYNCED,
            f"{deploy.branch} landed on {target.main_branch}, but {target.repo / target.path} "
            f"on disk is not {target.main_branch}'s copy, so a reload would re-read the old file",
            MergeState.done,
        )
    return Landing(
        request_id=str(request_id),
        branch=deploy.branch,
        tip_sha=deploy.tip,
        landed_as=_reported_sha(reply) or target.main_tip(),
    )


def _reported_sha(reply: dict) -> str | None:
    """The commit the queue says the request landed as, when it says."""
    last_outcome = reply.get("last_outcome") or {}
    return reply.get("commit") or reply.get("merge_sha") or last_outcome.get("merge_sha")
