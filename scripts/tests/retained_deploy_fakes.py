"""A deploy that land_files RETAINED, for tests of reclaim_retained_deploys and
of the scripts that call it before they land.

A request still live when its landing's ceiling passes, whose cancel the queue
refuses, keeps its deploy branch and worktree: the merge worker may still land
from them. `retained_deploy` leaves one behind in a temp repo through the real
land_files, against a FakeMergeQueue, so its branch name and worktree are the
ones production makes.

Stdlib only, and free of pytest, like the fakes it builds on.
"""
from __future__ import annotations

import contextlib
import itertools
import subprocess
from collections.abc import Mapping
from pathlib import Path

from _merge_queue_landing import LandingNotConfirmed, land_files
from config_reload_script_fakes import FakeEscalationMcp, git
from merge_queue_fakes import FakeMergeQueue

# merge_cancel's refusal for a request whose landing CAS is in progress.
CANCEL_REFUSED_WHILE_LANDING = {
    "cancelled": False,
    "state": "finalizing",
    "reason": "Request is landing (its CAS is in progress)",
    "code": "finalize_window",
}


def retained_deploy(repo, files: Mapping[str, bytes], *, label: str,
                    message: str = "chore(test): a deploy the queue would not cancel"):
    """The (branch, worktree) land_files kept in *repo* for *files*, labelled
    *label*, branching from main.

    Its clock passes the ceiling at the first reading after the deadline is
    set, so the request is cancelled on its submit's state, before any poll,
    however long git and the queue calls really take. The deploy's scratch
    directory is made wherever tempfile makes one.
    """
    queue = FakeMergeQueue(repo, states=["verifying"], cancel_reply=CANCEL_REFUSED_WHILE_LANDING)
    readings = itertools.count()
    with FakeEscalationMcp(tool_handlers=queue.handlers()) as server, \
            contextlib.suppress(LandingNotConfirmed):
        land_files(str(repo), files, git(repo, "rev-parse", "main").strip(),
                   port=str(server.port), label=label, message=message,
                   timeout_secs=0.5, clock=lambda: float(next(readings)))
    (submitted,) = queue.calls_to("merge_request")
    branch, worktree = f"task/{submitted['branch']}", Path(submitted["worktree"])
    assert worktree.is_dir() and branch_exists(repo, branch), (
        f"land_files did not retain {branch} at {worktree}")
    return branch, worktree


def branch_exists(repo, branch: str) -> bool:
    """Whether *repo* has the local branch *branch*."""
    return subprocess.run(
        ["git", "-C", str(repo), "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
    ).returncode == 0
