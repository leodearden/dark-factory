"""A scripted merge queue that lands for real, for tests of a deploy script
that lands a config edit through its target's merge queue.

`FakeMergeQueue` answers merge_request, merge_status and merge_cancel in the
reply shapes escalation/src/escalation/server.py emits, plugged into
`config_reload_script_fakes.FakeEscalationMcp` through its `tool_handlers`
seam, so every call crosses the real stateful MCP transport. Reaching 'done'
performs a REAL git landing in the temp repo whose main is checked out, the
way the merge lane's advance_main moves main and syncs the working tree.

Stdlib only, and free of pytest, like the fakes module it builds on.
"""
from __future__ import annotations

import os
import subprocess

from config_reload_script_fakes import git

DEPLOY_BRANCH_PREFIX = "refs/heads/task/deploy-"

# A merge_status step that answers no tool result at all: the transport
# refuses the envelope, as it would a reply cut off by a server restart.
TRANSPORT_FAULT = object()


class FakeMergeQueue:
    """One deploy request's scripted journey through a target's merge queue.

    *submit* is merge_request's status, or a dict of reply fields (an
    `{error: ...}` reply among them). A submit of 'done' lands at once and
    carries `commit`, as a wait_secs merge that finished inside the wait does.

    *states* is the merge_status sequence for this request's own id, one entry
    per poll with the last repeating; or a dict mapping each pollable handle to
    its own sequence. An entry is a state, a dict of reply fields carrying
    `state`, or TRANSPORT_FAULT. A handle the script does not know answers
    *foreign_state*, as the queue's other requests would, and never lands
    anything.

    *landing* is what reaching 'done' does to the repo: 'merge' (a `--no-ff`
    merge in the main checkout, which syncs the working tree), 'cherry_pick'
    (the rebased landing: patch-equal, tip not an ancestor), 'ref_only' (main
    moves, the working tree does not) or 'none' (a phantom done).
    """

    REQUEST_ID = "mr-deploy0001"

    def __init__(self, repo, *, submit: str | dict = "queued", states=("done",), landing="merge",
                 eta_seconds=0.01, cancel_reply=None, foreign_state="done",
                 request_id=REQUEST_ID):
        self.repo = repo
        self.submit = submit if isinstance(submit, dict) else {"status": submit}
        self.sequences = {
            handle: list(steps)
            for handle, steps in (
                states.items() if isinstance(states, dict) else [(request_id, states)]
            )
        }
        self.landing = landing
        self.eta_seconds = eta_seconds
        self.cancel_reply = cancel_reply or {"cancelled": True, "state": "abandoned", "reason": None}
        self.foreign_state = foreign_state
        self.request_id = request_id
        self.calls = []
        self.branch = None
        self.merge_sha = None
        self._landed = False
        self._polls = {}

    def handlers(self):
        """The `tool_handlers` this queue answers through."""
        return {
            "merge_request": self._merge_request,
            "merge_status": self._merge_status,
            "merge_cancel": self._merge_cancel,
        }

    def has_landed(self):
        """Whether a scripted 'done' has run its landing yet."""
        return self._landed

    def calls_to(self, tool):
        """The arguments of every call to *tool*, in order."""
        return [call["arguments"] for call in self.calls if call["tool"] == tool]

    # ------------------------------------------------------------------
    # The three tools
    # ------------------------------------------------------------------

    def _merge_request(self, arguments):
        self.branch = f"task/{arguments.get('branch')}"
        self._record("merge_request", arguments, self._observe_submission(arguments))
        if "error" in self.submit:
            return dict(self.submit)
        status = self.submit["status"]
        if status in ("queued", "attached"):
            return {**self._queued_reply(status), **self.submit}
        if status == "done":
            self._land()
        return {
            "status": status,
            "request_id": self.request_id,
            "reason": "",
            "conflict_details": "",
            "push_status": None,
            "commit": self.merge_sha if status == "done" else None,
            **self.submit,
        }

    def _merge_status(self, arguments):
        self._record("merge_status", arguments, {})
        handle = arguments.get("request_id") or arguments.get("task_id") or arguments.get("branch")
        steps = self.sequences.get(handle)
        if steps is None:
            return self._status_reply(handle, {"state": self.foreign_state})
        seen = self._polls.get(handle, 0)
        self._polls[handle] = seen + 1
        step = steps[min(seen, len(steps) - 1)]
        if step is TRANSPORT_FAULT:
            return None
        fields = step if isinstance(step, dict) else {"state": step}
        if fields["state"] == "done":
            self._land()
        return self._status_reply(handle, fields)

    def _merge_cancel(self, arguments):
        self._record("merge_cancel", arguments, {})
        return dict(self.cancel_reply)

    # ------------------------------------------------------------------
    # Replies
    # ------------------------------------------------------------------

    def _queued_reply(self, status):
        reply = {
            "status": status,
            "request_id": self.request_id,
            "snapshot_tip": self._branch_tip(),
            "generation": 0,
            "position": 0,
            "queue_depth": 1,
            "eta_seconds": self.eta_seconds,
            "lane": "normal",
            "lane_source": "argument",
            "lane_applied": status == "queued",
        }
        if status == "attached":
            reply.update({
                "inflight_task_id": None,
                "source": "registry",
                "inflight_request_id": self.request_id,
                "poll_by": "request_id",
                "pollable": True,
            })
        return reply

    def _status_reply(self, handle, fields):
        """A merge_status answer. Every one carries *eta_seconds*, which sets
        the poller's cadence: the real server sends it on live answers only."""
        reply = {
            "request_id": handle,
            "generation": 0,
            "source": "live",
            "eta_seconds": self.eta_seconds,
            "last_outcome": None,
        }
        if fields["state"] == "done" and handle in self.sequences:
            reply["merge_sha"] = self.merge_sha
        reply.update(fields)
        return reply

    # ------------------------------------------------------------------
    # The repo
    # ------------------------------------------------------------------

    def _record(self, tool, arguments, observed):
        self.calls.append({"tool": tool, "arguments": dict(arguments), "observed": observed})

    def _observe_submission(self, arguments):
        """What the merge worker would find on disk when it picks this up."""
        worktree = arguments.get("worktree") or ""
        exists = os.path.isdir(worktree)
        return {
            "worktree_exists": exists,
            "worktree_head": _rev_parse(worktree, "HEAD") if exists else None,
            "branch_tip": self._branch_tip(),
        }

    def _branch_tip(self):
        return _rev_parse(self.repo, self.branch) if self.branch else None

    def _land(self):
        if self._landed:
            return
        self._landed = True
        message = f"Merge {self.branch} into main"
        if self.landing == "merge":
            git(self.repo, "merge", "--no-ff", "-q", "-m", message, self.branch)
        elif self.landing == "cherry_pick":
            # -x makes the copy a distinct commit even within the same second.
            git(self.repo, "cherry-pick", "-x", self.branch)
        elif self.landing == "ref_only":
            tip = git(self.repo, "rev-parse", self.branch).strip()
            merged = git(
                self.repo, "commit-tree", f"{tip}^{{tree}}", "-p", "main", "-p", tip, "-m", message,
            ).strip()
            git(self.repo, "update-ref", "refs/heads/main", merged)
        elif self.landing != "none":
            raise ValueError(f"unknown landing {self.landing!r}")
        if self.landing != "none":
            self.merge_sha = git(self.repo, "rev-parse", "main").strip()


def _rev_parse(repo, rev):
    """*rev*'s sha in *repo*, or None when it does not resolve."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", rev],
        capture_output=True, text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


# ---------------------------------------------------------------------------
# What a deploy must leave behind
# ---------------------------------------------------------------------------

def assert_target_checkout_clean(repo):
    """*repo* carries no uncommitted change, no worktree but its own, and no
    deploy branch: everything a landing created was torn down."""
    status = git(repo, "status", "--porcelain")
    assert status == "", f"the target checkout is dirty:\n{status}"
    worktrees = [
        line for line in git(repo, "worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    ]
    assert len(worktrees) == 1, f"a deploy worktree was left behind: {worktrees}"
    refs = git(repo, "for-each-ref", "--format=%(refname)", "refs/heads/").split()
    left = [ref for ref in refs if ref.startswith(DEPLOY_BRANCH_PREFIX)]
    assert left == [], f"a deploy branch was left behind: {left}"


def deploy_tmpdir_env(tmp_path, base_env=None):
    """*base_env* (default: this process's) with TMPDIR a fresh, empty
    directory, so a test can assert a deploy removed its scratch space."""
    scratch = tmp_path / "deploy-tmp"
    scratch.mkdir()
    env = dict(os.environ if base_env is None else base_env)
    env["TMPDIR"] = str(scratch)
    return env
