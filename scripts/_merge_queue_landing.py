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

The calling heredoc's side of the contract (argv-only inputs, the script's
directory on sys.path, the checkout venv's interpreter) is
scripts/_config_reload_gate.py's.
"""
from __future__ import annotations

import enum
import importlib
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _config_reload_gate import die, load_transport, transport_not_importable_remedy

DEFAULT_LANDING_TIMEOUT_SECS = 10800
SUBMIT_WAIT_SECS = 100
# merge_request may hold its reply for SUBMIT_WAIT_SECS before answering.
SUBMIT_REPLY_TIMEOUT_SECS = 130.0
POLL_REPLY_TIMEOUT_SECS = 30.0
FIRST_BACKOFF_SECS = 15.0
MAX_POLL_DELAY_SECS = 60.0


class LandingFailure(enum.StrEnum):
    """Why a landing was not confirmed."""

    TRANSPORT_NOT_IMPORTABLE = "transport_not_importable"
    TRANSPORT_FAILED = "transport_failed"
    NOT_LANDED = "not_landed"
    QUEUE_DONE_NOT_ON_MAIN = "queue_done_not_on_main"
    CHECKOUT_NOT_SYNCED = "checkout_not_synced"


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

    *timeout_secs* bounds the wait for the request to settle, defaulting to
    DEFAULT_LANDING_TIMEOUT_SECS. A string is read as a number, so a caller
    can pass an environment variable through unparsed.
    """
    queue = _MergeQueue.connect(port)
    target = _Target(Path(repo), path, main_branch)
    deadline = time.monotonic() + float(timeout_secs or DEFAULT_LANDING_TIMEOUT_SECS)
    deploy = _DeployBranch.create(target, base_sha, content, label=label, message=message)
    try:
        submitted = queue.submit(deploy, description=message)
        settled = _await_landed(queue, submitted, deadline)
        return _confirmed(target, deploy, settled)
    finally:
        deploy.tear_down()


def land_or_exit(repo: str, path: str, base_sha: str, content: bytes, *, prog: str,
                 **kwargs: Any) -> Landing:
    """`land_file`, exiting through `die` with the refusal's detail."""
    try:
        return land_file(repo, path, base_sha, content, **kwargs)
    except LandingNotConfirmed as exc:
        die(prog, exc.detail)


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
    scratch directory outside the target checkout: the merge worker reads the
    request's source from that worktree."""

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
        try:
            census_trigger, _ = load_transport()
            # The poll vocabulary is not stdlib either; proving it here keeps
            # its absence on the remedy path.
            importlib.import_module("shared.merge_state")
        except ImportError as exc:
            raise LandingNotConfirmed(
                LandingFailure.TRANSPORT_NOT_IMPORTABLE, transport_not_importable_remedy(exc),
            ) from exc
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

    def status(self, handle: dict[str, str]) -> dict:
        return self._call("merge_status", handle, timeout=POLL_REPLY_TIMEOUT_SECS)

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


def _await_landed(queue: _MergeQueue, submitted: dict, deadline: float) -> dict:
    """The reply that settled the request as landed: the submit's own, or the
    poll that reached done."""
    from shared.merge_state import LIVE_STATES, SUBMIT_NON_TERMINAL, MergeState

    status = submitted.get("status")
    if status not in SUBMIT_NON_TERMINAL:
        return submitted
    handle = {"request_id": submitted["request_id"]}
    reply = submitted
    while True:
        time.sleep(min(_poll_delay(reply), max(0.0, deadline - time.monotonic())))
        reply = queue.status(handle)
        state = reply.get("state")
        if state == MergeState.done:
            return reply
        if state not in LIVE_STATES:
            raise LandingNotConfirmed(
                LandingFailure.NOT_LANDED,
                f"merge request {handle['request_id']} settled {state}", state=state,
            )


def _poll_delay(reply: dict) -> float:
    """The queue's own positive estimate, capped; else the first backoff."""
    eta = reply.get("eta_seconds")
    if isinstance(eta, int | float) and eta > 0:
        return min(float(eta), MAX_POLL_DELAY_SECS)
    return FIRST_BACKOFF_SECS


def _confirmed(target: _Target, deploy: _DeployBranch, reply: dict) -> Landing:
    """The Landing the queue's done *reply* claims, once git agrees."""
    request_id = reply.get("request_id")
    if not target.carries(deploy.tip):
        raise LandingNotConfirmed(
            LandingFailure.QUEUE_DONE_NOT_ON_MAIN,
            f"the queue reported request {request_id} done, but {target.main_branch} carries "
            f"{deploy.tip} neither as an ancestor nor as an equal patch",
            state=reply.get("state") or reply.get("status"),
        )
    if not target.checkout_synced():
        raise LandingNotConfirmed(
            LandingFailure.CHECKOUT_NOT_SYNCED,
            f"{deploy.branch} landed on {target.main_branch}, but {target.repo / target.path} "
            f"on disk is not {target.main_branch}'s copy, so a reload would re-read the old file",
            state=reply.get("state") or reply.get("status"),
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
