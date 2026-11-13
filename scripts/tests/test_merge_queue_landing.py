"""Tests for scripts/_merge_queue_landing.py's public seam, driven in-process
over a real loopback socket against the stateful escalation MCP fake, with a
FakeMergeQueue behind it that lands for real in a temp git repo."""
from __future__ import annotations

import contextlib
import errno
import os
import pathlib
import signal
import subprocess
import tempfile
import threading
import time

import pytest
from _merge_queue_landing import (
    Landing,
    LandingFailure,
    LandingNotConfirmed,
    land_file,
    land_files,
)
from config_reload_script_fakes import (
    TOOL_RAISED,
    TRANSPORT_FAULTS,
    ClosedPort,
    FakeEscalationMcp,
    commit_config,
    git,
    init_repo,
)
from merge_queue_fakes import (
    TRANSPORT_FAULT,
    FakeMergeQueue,
    assert_no_landing_left_behind,
    assert_target_checkout_clean,
    deploy_tmpdir_env,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

CONFIG = "dark-factory-orchestrator.yaml"
SEED = "project_root: /nowhere\nknob: 0\n"
EDITED = b"project_root: /nowhere\nknob: 1\n"
LABEL = "test-knob-1"
MESSAGE = "chore(test): set knob=1"

PRD = "plans/demo-prd.md"
PRD_SEED = b"# Demo PRD\n\nDraft.\n"
DOCS = {
    PRD: b"# Demo PRD\n\nDecomposed.\n",
    "plans/demo-prd.capability-manifest.md": b"# Demo PRD capability manifest\n",
    "plans/demo-prd.capability-manifest.yaml": b"capabilities: []\n",
}


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """The fresh, empty directory every tempfile call lands in."""
    path = deploy_tmpdir_env(tmp_path)["TMPDIR"]
    monkeypatch.setattr(tempfile, "tempdir", path)
    return path


@pytest.fixture
def repo(tmp_path):
    """A target checkout with main checked out, carrying SEED."""
    checkout = init_repo(tmp_path)
    commit_config(checkout, SEED)
    return checkout


@pytest.fixture
def docs_repo(repo):
    """The target checkout, also tracking PRD as PRD_SEED; DOCS' other paths
    are new to it."""
    (repo / PRD).parent.mkdir()
    (repo / PRD).write_bytes(PRD_SEED)
    git(repo, "add", PRD)
    git(repo, "commit", "-q", "-m", "seed the PRD")
    return repo


def _main(repo):
    return git(repo, "rev-parse", "main").strip()


def _land_docs(repo, queue, files=DOCS, *, base=None, **kwargs):
    """land_files of *files* onto *repo*'s main, from *base* (default: main),
    through *queue*."""
    with FakeEscalationMcp(tool_handlers=queue.handlers()) as server:
        return land_files(
            str(repo), files, base or _main(repo),
            port=str(server.port), label=LABEL, message=MESSAGE, **kwargs,
        )


def _files_in(repo, commit):
    return sorted(git(repo, "show", "--name-only", "--format=", commit).split())


def _land(repo, queue, **kwargs):
    """land_file of EDITED onto *repo*'s main, through *queue*. Returns
    (landing, the server's tool calls)."""
    with FakeEscalationMcp(tool_handlers=queue.handlers()) as server:
        landing = land_file(
            str(repo), CONFIG, _main(repo), EDITED,
            port=str(server.port), label=LABEL, message=MESSAGE, **kwargs,
        )
        calls = server.tool_calls()
    return landing, calls


def _assert_left_nothing_behind(repo, scratch):
    assert_target_checkout_clean(repo)
    _assert_scratch_removed(scratch)


def _assert_scratch_removed(scratch):
    assert os.listdir(scratch) == [], f"the deploy scratch dir was left: {os.listdir(scratch)}"


def _checkout_state(repo, *paths):
    """What a landing must leave alone in *repo*: its staged and unstaged
    diffs, and the bytes on disk at each of *paths*."""
    return (git(repo, "diff", "--cached"), git(repo, "diff"),
            {path: (repo / path).read_bytes() for path in paths})


# ---------------------------------------------------------------------------
# A request that lands
# ---------------------------------------------------------------------------

def test_a_request_that_verifies_and_lands_returns_its_landing(repo, scratch):
    base = _main(repo)
    queue = FakeMergeQueue(repo, states=["verifying", "done"])

    landing, _ = _land(repo, queue)

    assert isinstance(landing, Landing)
    assert landing.request_id == queue.request_id
    assert landing.landed_as == _main(repo) == queue.merge_sha
    landed = git(repo, "rev-list", f"{base}..main").split()
    assert len(landed) == 2, f"expected the deploy commit and its merge, got {landed}"
    subject = git(repo, "log", "-1", "--format=%s", "main").strip()
    assert subject.startswith(f"Merge task/deploy-{LABEL}-"), subject
    assert subject == f"Merge {landing.branch} into main", subject
    touched = git(repo, "diff", "--name-only", base, "main").split()
    assert touched == [CONFIG]
    deploy_commit = git(repo, "rev-parse", "main^2").strip()
    assert deploy_commit == landing.tip_sha
    assert git(repo, "log", "-1", "--format=%s", deploy_commit).strip() == MESSAGE
    assert git(repo, "show", "--name-only", "--format=", deploy_commit).split() == [CONFIG]
    assert (repo / CONFIG).read_bytes() == EDITED
    _assert_left_nothing_behind(repo, scratch)


def test_every_poll_names_the_requests_own_id(repo, scratch):
    queue = FakeMergeQueue(repo, states=["queued", "verifying", "gate", "finalizing", "done"])

    _land(repo, queue)

    polls = queue.calls_to("merge_status")
    assert len(polls) == 5, polls
    assert all(poll == {"request_id": queue.request_id} for poll in polls), polls


def test_the_merge_request_names_a_deploy_branch_the_worker_can_read(repo, scratch):
    """The merge worker reads the source from the submitted worktree, so it
    must exist, at the deploy commit, when the request is submitted."""
    queue = FakeMergeQueue(repo)

    landing, calls = _land(repo, queue)

    assert [name for name, _ in calls][0] == "merge_request", calls
    (submitted,) = [call for call in queue.calls if call["tool"] == "merge_request"]
    arguments = submitted["arguments"]
    assert arguments["task_id"] == arguments["branch"], arguments
    assert arguments["branch"].startswith(f"deploy-{LABEL}-"), arguments
    assert f"task/{arguments['branch']}" == landing.branch
    assert os.path.isabs(arguments["worktree"]), arguments
    assert arguments["wait_secs"] == 100
    assert arguments["lane"] == "normal"
    assert arguments["verified_green"] is False
    observed = submitted["observed"]
    assert observed["worktree_exists"], observed
    assert observed["worktree_head"] == observed["branch_tip"] == landing.tip_sha, observed
    assert not os.path.exists(arguments["worktree"]), "the deploy worktree outlived its landing"


@pytest.mark.parametrize("status", ["done", "already_merged", "done_wip_recovery"])
def test_a_submit_that_settled_as_landed_lands_without_polling(repo, scratch, status):
    """done_wip_recovery landed too: only the WIP stash-pop after it conflicted."""
    queue = FakeMergeQueue(repo, submit=status)

    landing, calls = _land(repo, queue)

    assert [name for name, _ in calls] == ["merge_request"], calls
    assert landing.landed_as == queue.merge_sha == _main(repo)
    assert (repo / CONFIG).read_bytes() == EDITED
    _assert_left_nothing_behind(repo, scratch)


def test_a_submit_settlement_is_judged_by_git_not_by_its_status(repo, scratch):
    """A submit status that claims no landing, while main carries the deploy
    commit, is a landing."""
    queue = FakeMergeQueue(repo, submit="blocked", lands_on={"blocked"})

    landing, _ = _land(repo, queue)

    assert landing.landed_as == _main(repo)
    assert (repo / CONFIG).read_bytes() == EDITED
    _assert_left_nothing_behind(repo, scratch)


def test_a_wip_recovery_that_left_the_checkout_unsynced_is_refused(repo, scratch):
    """The merge landed but the checkout's file is not main's: a reload
    would re-read the old file, and the refusal says so rather than that
    nothing landed."""
    queue = FakeMergeQueue(repo, submit="done_wip_recovery", landing="ref_only")

    refusal = _refused_by(repo, queue)

    assert refusal.failure is LandingFailure.CHECKOUT_NOT_SYNCED
    assert "nothing landed" not in refusal.detail, refusal.detail


def test_a_rebased_landing_is_confirmed_by_patch_identity(repo, scratch):
    """The queue may land the deploy commit rebased: its tip is then not an
    ancestor of main, but an equal patch is."""
    queue = FakeMergeQueue(repo, landing="cherry_pick")

    landing, _ = _land(repo, queue)

    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", landing.tip_sha, "main"],
    )
    assert ancestry.returncode == 1, "the cherry-pick landing left the tip on main's ancestry"
    assert landing.landed_as == _main(repo)
    assert (repo / CONFIG).read_bytes() == EDITED
    _assert_left_nothing_behind(repo, scratch)


def test_three_docs_paths_land_as_one_commit_through_one_request(docs_repo, scratch):
    """Two of the three paths are new at base: a docs landing adds files."""
    repo = docs_repo
    base = _main(repo)
    queue = FakeMergeQueue(repo, states=["verifying", "done"])

    landing = _land_docs(repo, queue)

    assert len(queue.calls_to("merge_request")) == 1, queue.calls
    assert isinstance(landing, Landing)
    assert landing.request_id == queue.request_id
    assert landing.landed_as == _main(repo) == queue.merge_sha
    assert _files_in(repo, landing.tip_sha) == sorted(DOCS)
    assert git(repo, "log", "-1", "--format=%s", landing.tip_sha).strip() == MESSAGE
    assert sorted(git(repo, "diff", "--name-only", base, "main").split()) == sorted(DOCS)
    for path, content in DOCS.items():
        assert (repo / path).read_bytes() == content, path
    _assert_left_nothing_behind(repo, scratch)


# ---------------------------------------------------------------------------
# Every way a request can fail to land
#
# Each case pins its own failure member and names the merge state in its
# detail, and, unless it says otherwise, leaves main and the checkout as
# they were.
# ---------------------------------------------------------------------------

def _refusal(repo, port, *, path=CONFIG, base=None, content=EDITED, **kwargs):
    """The LandingNotConfirmed land_file raises against *port*."""
    with pytest.raises(LandingNotConfirmed) as caught:
        land_file(
            str(repo), path, base or _main(repo), content,
            port=str(port), label=LABEL, message=MESSAGE, **kwargs,
        )
    return caught.value


def _refused_by(repo, queue, **kwargs):
    """The LandingNotConfirmed land_file raises through *queue*."""
    with FakeEscalationMcp(tool_handlers=queue.handlers()) as server:
        return _refusal(repo, server.port, **kwargs)


def _docs_refused_by(repo, queue, files, **kwargs):
    """The LandingNotConfirmed land_files of *files* raises through *queue*."""
    with pytest.raises(LandingNotConfirmed) as caught:
        _land_docs(repo, queue, files, **kwargs)
    return caught.value


def _assert_untouched(repo, before, scratch):
    assert _main(repo) == before, "main moved on a landing that was refused"
    _assert_left_nothing_behind(repo, scratch)


CEILING_SECS = 1


class _ManualClock:
    """A monotonic clock that stands still until a test moves it."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _refused_at_the_ceiling(repo, queue, *, passed_by):
    """The LandingNotConfirmed land_file raises through *queue* on a
    CEILING_SECS budget whose clock passes the ceiling as the queue answers
    *passed_by*, however long git setup and the queue calls really take."""
    clock = _ManualClock()
    handlers = queue.handlers()
    handlers[passed_by] = _acting_after(
        handlers[passed_by], lambda: clock.advance(CEILING_SECS + 1),
    )
    with FakeEscalationMcp(tool_handlers=handlers) as server:
        return _refusal(repo, server.port, timeout_secs=CEILING_SECS, clock=clock)


@pytest.mark.parametrize("terminal", [
    {"state": "blocked", "reason": "verify failed: 3 tests"},
    {"state": "conflict"},
    {"state": "abandoned"},
], ids=lambda terminal: terminal["state"])
def test_a_poll_that_settles_without_landing_is_not_landed(repo, scratch, terminal):
    before = _main(repo)
    queue = FakeMergeQueue(repo, states=["verifying", terminal])

    refusal = _refused_by(repo, queue)

    assert refusal.failure is LandingFailure.NOT_LANDED
    assert refusal.state == terminal["state"]
    assert refusal.reason == terminal.get("reason")
    assert terminal["state"] in refusal.detail, refusal.detail
    assert terminal.get("reason", "") in refusal.detail, refusal.detail
    _assert_untouched(repo, before, scratch)


@pytest.mark.parametrize("status", ["blocked", "conflict", "unknown_branch"])
def test_a_submit_that_settles_without_landing_is_not_landed(repo, scratch, status):
    before = _main(repo)
    queue = FakeMergeQueue(repo, submit={"status": status, "reason": f"why-{status}"})

    refusal = _refused_by(repo, queue)

    assert refusal.failure is LandingFailure.NOT_LANDED
    assert refusal.reason == f"why-{status}"
    assert status in refusal.detail, refusal.detail
    assert f"why-{status}" in refusal.detail, refusal.detail
    assert queue.calls_to("merge_status") == []
    _assert_untouched(repo, before, scratch)


def test_a_three_path_landing_the_queue_refuses_names_its_outcome_and_reason(docs_repo,
                                                                             scratch):
    repo = docs_repo
    before = _main(repo)
    queue = FakeMergeQueue(
        repo, states=["verifying", {"state": "blocked", "reason": "verify failed: 2 tests"}],
    )

    refusal = _docs_refused_by(repo, queue, DOCS)

    assert len(queue.calls_to("merge_request")) == 1, queue.calls
    assert refusal.failure is LandingFailure.NOT_LANDED
    assert refusal.state == "blocked"
    assert refusal.reason == "verify failed: 2 tests"
    _assert_untouched(repo, before, scratch)


def test_a_submit_the_server_refuses_is_submit_rejected(repo, scratch):
    before = _main(repo)
    error = "Merge queue not available — orchestrator not running"
    queue = FakeMergeQueue(repo, submit={"error": error})

    refusal = _refused_by(repo, queue)

    assert refusal.failure is LandingFailure.SUBMIT_REJECTED
    assert refusal.reason == error
    assert error in refusal.detail, refusal.detail
    _assert_untouched(repo, before, scratch)


def test_a_dead_socket_at_submit_is_transport_failed(repo, scratch):
    before = _main(repo)

    with ClosedPort() as closed:
        refusal = _refusal(repo, closed.port)

    assert refusal.failure is LandingFailure.TRANSPORT_FAILED
    assert "merge_request" in refusal.detail, refusal.detail
    _assert_untouched(repo, before, scratch)


@pytest.mark.parametrize(
    "fault", list(TRANSPORT_FAULTS.values()), ids=list(TRANSPORT_FAULTS)
)
def test_every_transport_fault_at_submit_is_transport_failed(repo, scratch, fault):
    before = _main(repo)

    with FakeEscalationMcp(**fault) as server:
        refusal = _refusal(repo, server.port)

    assert refusal.failure is LandingFailure.TRANSPORT_FAILED
    assert "merge_request" in refusal.detail, refusal.detail
    _assert_untouched(repo, before, scratch)


def test_a_merge_request_tool_error_at_submit_fails_closed(repo, scratch):
    """Keeps the tool-error shape covered now that TRANSPORT_FAULTS no longer
    carries it. TRANSPORT_FAILED is today's tag, not the contract: task 8025
    owns this case's tag and prose."""
    before = _main(repo)

    with FakeEscalationMcp(**TOOL_RAISED) as server:
        refusal = _refusal(repo, server.port)

    assert refusal.failure is LandingFailure.TRANSPORT_FAILED
    assert "merge_request" in refusal.detail, refusal.detail
    _assert_untouched(repo, before, scratch)


def test_a_request_still_live_at_the_ceiling_is_cancelled(repo, scratch):
    before = _main(repo)
    queue = FakeMergeQueue(repo, states=["verifying"])

    refusal = _refused_at_the_ceiling(repo, queue, passed_by="merge_status")

    assert refusal.failure is LandingFailure.TIMED_OUT
    assert len(queue.calls_to("merge_status")) == 1
    assert "verifying" in refusal.detail, refusal.detail
    assert queue.calls_to("merge_cancel") == [{"request_id": queue.request_id}]
    _assert_untouched(repo, before, scratch)


def test_a_ceiling_passed_during_the_submit_cancels_without_a_poll(repo, scratch):
    """The ceiling counts the submit's own time: a request whose submit
    answers past it is cancelled on the submit's state, and the abandoned
    request leaves nothing behind."""
    before = _main(repo)
    queue = FakeMergeQueue(repo, states=["verifying"])

    refusal = _refused_at_the_ceiling(repo, queue, passed_by="merge_request")

    assert refusal.failure is LandingFailure.TIMED_OUT
    assert "still queued" in refusal.detail, refusal.detail
    assert queue.calls_to("merge_status") == []
    assert queue.calls_to("merge_cancel") == [{"request_id": queue.request_id}]
    _assert_untouched(repo, before, scratch)


def test_a_cancel_refused_while_live_keeps_the_deploy_branch(repo, scratch):
    """The merge worker reads the source from the deploy worktree, so a
    request that may still land must keep it."""
    queue = FakeMergeQueue(repo, states=["verifying"], cancel_reply={
        "cancelled": False,
        "state": "finalizing",
        "reason": "Request is landing (its CAS is in progress)",
        "code": "finalize_window",
    })

    refusal = _refused_at_the_ceiling(repo, queue, passed_by="merge_status")

    assert refusal.failure is LandingFailure.TIMED_OUT
    assert len(queue.calls_to("merge_status")) == 1
    (submitted,) = queue.calls_to("merge_request")
    worktree, branch = submitted["worktree"], f"task/{submitted['branch']}"
    assert os.path.isdir(worktree), "a live request's worktree was removed"
    assert git(repo, "rev-parse", "--verify", branch).strip()
    for named in (worktree, branch, "finalizing", "finalize_window"):
        assert named in refusal.detail, (named, refusal.detail)


def test_a_done_that_git_cannot_find_on_main_is_refused(repo, scratch):
    before = _main(repo)
    queue = FakeMergeQueue(repo, landing="none")

    refusal = _refused_by(repo, queue)

    assert refusal.failure is LandingFailure.QUEUE_DONE_NOT_ON_MAIN
    assert queue.request_id in refusal.detail, refusal.detail
    _assert_untouched(repo, before, scratch)


def test_a_git_step_that_fails_while_creating_the_branch_is_git_failed(repo, scratch):
    """No worktree can be added; the branch made for it is removed again."""
    before = _main(repo)
    blocker = repo / ".git" / "worktrees"
    blocker.write_text("not a directory\n")
    queue = FakeMergeQueue(repo)

    refusal = _refused_by(repo, queue)
    blocker.unlink()

    assert refusal.failure is LandingFailure.GIT_FAILED
    assert "worktree add" in refusal.detail, refusal.detail
    assert queue.calls == []
    _assert_untouched(repo, before, scratch)


def test_a_git_failure_while_polling_cancels_the_request(repo, scratch):
    """git cannot say whether a journaled request landed: the request is
    cancelled rather than left live without its worktree."""
    before = _main(repo)
    queue = FakeMergeQueue(repo, states=["verifying", "journaled"])
    handlers = queue.handlers()
    handlers["merge_status"] = _acting_after(
        handlers["merge_status"], lambda: git(repo, "update-ref", "-d", "refs/heads/main"),
        when=lambda reply: reply["state"] == "journaled",
    )

    with FakeEscalationMcp(tool_handlers=handlers) as server:
        refusal = _refusal(repo, server.port, timeout_secs=20)
    git(repo, "update-ref", "refs/heads/main", before)

    assert refusal.failure is LandingFailure.GIT_FAILED, refusal.detail
    assert "journaled" in refusal.detail, refusal.detail
    assert queue.calls_to("merge_cancel") == [{"request_id": queue.request_id}]
    _assert_untouched(repo, before, scratch)


def test_a_landing_the_checkout_never_synced_is_refused(repo, scratch):
    """main moved but the checkout's file did not: a reload would re-read
    the old file. The fake leaves the checkout dirty itself, so no clean
    assertion here."""
    queue = FakeMergeQueue(repo, landing="ref_only")

    refusal = _refused_by(repo, queue)

    assert refusal.failure is LandingFailure.CHECKOUT_NOT_SYNCED
    assert str(repo / CONFIG) in refusal.detail, refusal.detail


def test_content_equal_to_the_base_is_refused_before_any_call(repo, scratch):
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _refused_by(repo, queue, content=SEED.encode())

    assert refusal.failure is LandingFailure.INVALID_CHANGE
    assert queue.calls == []
    _assert_untouched(repo, before, scratch)


def test_an_unchanged_path_among_changed_ones_is_refused_before_any_call(docs_repo, scratch):
    """The landed commit carries exactly the named paths, so each must change."""
    repo = docs_repo
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _docs_refused_by(repo, queue, {**DOCS, PRD: PRD_SEED})

    assert refusal.failure is LandingFailure.INVALID_CHANGE
    assert PRD in refusal.detail, refusal.detail
    changed = [path for path in DOCS if path != PRD]
    assert not any(path in refusal.detail for path in changed), refusal.detail
    assert queue.calls == []
    _assert_untouched(repo, before, scratch)


def test_a_base_that_is_not_a_commit_is_refused_before_any_call(repo, scratch):
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _refused_by(repo, queue, base="0" * 40)

    assert refusal.failure is LandingFailure.INVALID_CHANGE
    assert queue.calls == []
    _assert_untouched(repo, before, scratch)


# ---------------------------------------------------------------------------
# The path set is validated before any call
# ---------------------------------------------------------------------------

def test_an_empty_path_set_is_refused_before_any_call(repo, scratch):
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _docs_refused_by(repo, queue, {})

    assert refusal.failure is LandingFailure.INVALID_CHANGE
    assert queue.calls == []
    _assert_untouched(repo, before, scratch)


@pytest.mark.parametrize("case", [
    "parent_relative", "absolute_elsewhere", "through_a_symlink", "under_dot_git",
    "nested_dot_git", "dot_git_in_another_case", "the_root",
])
def test_a_path_outside_the_checkout_is_refused_before_any_call(docs_repo, scratch, tmp_path,
                                                                case):
    """Refused per path: each bad path rides alongside a good one. The
    refusal names the rule, because a later backstop would refuse some of
    these shapes too, less clearly."""
    repo = docs_repo
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "linked").symlink_to(outside, target_is_directory=True)
    git(repo, "add", "linked")
    git(repo, "commit", "-q", "-m", "link a directory outside the checkout")
    bad, escape, rule = {
        "parent_relative": ("../outside.md", tmp_path / "outside.md", "is outside the checkout"),
        "absolute_elsewhere": (str(tmp_path / "elsewhere.md"), tmp_path / "elsewhere.md",
                               "is outside the checkout"),
        "through_a_symlink": ("linked/escape.md", outside / "escape.md",
                              "crosses a symlink in the checkout"),
        "under_dot_git": (".git/config", None, "has a .git component"),
        "nested_dot_git": ("plans/.git/x", None, "has a .git component"),
        "dot_git_in_another_case": ("plans/.Git/x", None, "has a .git component"),
        "the_root": (".", None, "itself"),
    }[case]
    git_config = (repo / ".git" / "config").read_bytes()
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _docs_refused_by(repo, queue, {PRD: DOCS[PRD], bad: b"escaped\n"})

    assert refusal.failure is LandingFailure.INVALID_CHANGE, refusal.detail
    assert bad in refusal.detail, refusal.detail
    assert rule in refusal.detail, refusal.detail
    assert queue.calls == []
    assert escape is None or not escape.exists(), f"{escape} was written"
    assert (repo / ".git" / "config").read_bytes() == git_config
    _assert_untouched(repo, before, scratch)


def test_a_symlink_only_the_base_carries_is_refused_before_any_call(docs_repo, scratch,
                                                                    tmp_path):
    """The deploy commit is written at the base, which may carry a symlink
    the checkout has since dropped."""
    repo = docs_repo
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "linked").symlink_to(outside, target_is_directory=True)
    git(repo, "add", "linked")
    git(repo, "commit", "-q", "-m", "link a directory outside the checkout")
    base = _main(repo)
    git(repo, "rm", "-q", "linked")
    git(repo, "commit", "-q", "-m", "drop the link")
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _docs_refused_by(
        repo, queue, {PRD: DOCS[PRD], "linked/escape.md": b"escaped\n"}, base=base,
    )

    assert refusal.failure is LandingFailure.INVALID_CHANGE, refusal.detail
    assert "linked/escape.md" in refusal.detail, refusal.detail
    assert queue.calls == []
    assert not (outside / "escape.md").exists(), "the write followed the base's symlink"
    _assert_untouched(repo, before, scratch)


def test_a_symlink_the_checkout_gained_after_the_base_is_refused_before_any_call(docs_repo,
                                                                                 scratch,
                                                                                 tmp_path):
    """The base has no symlink to follow, so only the checkout shows that the
    path lands where main now carries one."""
    repo = docs_repo
    base = _main(repo)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "linked").symlink_to(outside, target_is_directory=True)
    git(repo, "add", "linked")
    git(repo, "commit", "-q", "-m", "link a directory outside the checkout")
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _docs_refused_by(
        repo, queue, {PRD: DOCS[PRD], "linked/escape.md": b"escaped\n"}, base=base,
    )

    assert refusal.failure is LandingFailure.INVALID_CHANGE, refusal.detail
    assert "linked/escape.md" in refusal.detail, refusal.detail
    assert queue.calls == []
    assert not (outside / "escape.md").exists(), "the write followed the checkout's symlink"
    _assert_untouched(repo, before, scratch)


@pytest.mark.parametrize("bad", [f"{CONFIG}/under-a-file.md", f"{CONFIG}/deeper/under.md",
                                 "plans"], ids=["under_a_file", "deeper_under_a_file",
                                                "over_a_directory"])
def test_a_path_git_cannot_hold_as_a_file_at_the_base_is_invalid(docs_repo, scratch, bad):
    repo = docs_repo
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _docs_refused_by(repo, queue, {PRD: DOCS[PRD], bad: b"misplaced\n"})

    assert refusal.failure is LandingFailure.INVALID_CHANGE, refusal.detail
    assert bad in refusal.detail, refusal.detail
    assert queue.calls == []
    _assert_untouched(repo, before, scratch)


def test_a_scratch_write_the_host_refuses_is_not_blamed_on_the_change(docs_repo, scratch,
                                                                      monkeypatch):
    """A full or read-only scratch filesystem is the host's fault: the tag
    must not send the operator looking at the paths."""
    repo = docs_repo
    write_bytes = pathlib.Path.write_bytes

    def full_scratch(self, data):
        if str(self).startswith(scratch):
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(self))
        return write_bytes(self, data)

    monkeypatch.setattr(pathlib.Path, "write_bytes", full_scratch)
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _docs_refused_by(repo, queue, DOCS)

    assert refusal.failure is LandingFailure.SCRATCH_WRITE_FAILED, refusal.detail
    assert os.strerror(errno.ENOSPC) in refusal.detail, refusal.detail
    assert queue.calls == []
    _assert_untouched(repo, before, scratch)


@pytest.mark.parametrize("spelling", ["dot_segment", "absolute"])
def test_two_spellings_of_one_path_are_refused(docs_repo, scratch, spelling):
    repo = docs_repo
    other = {"dot_segment": "plans/./demo-prd.md", "absolute": str(repo / PRD)}[spelling]
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _docs_refused_by(repo, queue, {PRD: DOCS[PRD], other: b"# Another draft\n"})

    assert refusal.failure is LandingFailure.INVALID_CHANGE
    assert queue.calls == []
    _assert_untouched(repo, before, scratch)


def test_an_absolute_path_inside_the_checkout_lands_as_its_repo_relative_path(docs_repo,
                                                                              scratch):
    repo = docs_repo
    queue = FakeMergeQueue(repo)

    landing = _land_docs(repo, queue, {str(repo / PRD): DOCS[PRD]})

    assert _files_in(repo, landing.tip_sha) == [PRD]
    assert (repo / PRD).read_bytes() == DOCS[PRD]
    _assert_left_nothing_behind(repo, scratch)


# ---------------------------------------------------------------------------
# The target checkout's own state
#
# The lane's advance refuses to move main over uncommitted state at a landed
# path, and halts the whole queue doing so; untracked state there it
# overwrites. Any other state in the checkout is not the landing's to touch.
# ---------------------------------------------------------------------------

def _commit(repo, path, content):
    (repo / path).write_bytes(content)
    git(repo, "add", path)
    git(repo, "commit", "-q", "-m", f"add {path}")


@pytest.mark.parametrize("dirt", ["unstaged_edit", "staged_edit", "untracked_new_file"])
def test_uncommitted_state_at_a_landed_path_is_refused_before_any_call(docs_repo, scratch, dirt):
    repo = docs_repo
    dirty = "plans/demo-prd.capability-manifest.md" if dirt == "untracked_new_file" else PRD
    (repo / dirty).write_bytes(b"# a session's uncommitted draft\n")
    if dirt == "staged_edit":
        git(repo, "add", dirty)
    state = _checkout_state(repo, dirty)
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _docs_refused_by(repo, queue, DOCS)

    assert refusal.failure is LandingFailure.CHECKOUT_DIRTY, refusal.detail
    assert dirty in refusal.detail, refusal.detail
    clean = [path for path in DOCS if path != dirty]
    assert not any(path in refusal.detail for path in clean), refusal.detail
    assert queue.calls == []
    assert _main(repo) == before
    assert _checkout_state(repo, dirty) == state
    assert_no_landing_left_behind(repo)
    _assert_scratch_removed(scratch)


def test_other_staged_and_dirty_state_is_left_alone_when_the_queue_refuses(docs_repo, scratch):
    repo = docs_repo
    _commit(repo, "notes.md", b"notes\n")
    (repo / CONFIG).write_bytes(EDITED)
    git(repo, "add", CONFIG)
    (repo / "notes.md").write_bytes(b"notes, edited\n")
    (repo / "draft.txt").write_bytes(b"an untracked draft\n")
    state = _checkout_state(repo, CONFIG, "notes.md", "draft.txt")
    before = _main(repo)
    queue = FakeMergeQueue(
        repo, states=["verifying", {"state": "blocked", "reason": "verify failed: 2 tests"}],
    )

    refusal = _docs_refused_by(repo, queue, DOCS)

    assert refusal.failure is LandingFailure.NOT_LANDED, refusal.detail
    assert _main(repo) == before
    assert _checkout_state(repo, CONFIG, "notes.md", "draft.txt") == state
    assert_no_landing_left_behind(repo)
    _assert_scratch_removed(scratch)


def test_unrelated_dirty_state_survives_a_landing(docs_repo, scratch):
    """No staged change here: the fake lands with a real `git merge`, which
    aborts on a dirty index that the lane's own advance would park."""
    repo = docs_repo
    (repo / CONFIG).write_bytes(EDITED)
    (repo / "draft.txt").write_bytes(b"an untracked draft\n")
    state = _checkout_state(repo, CONFIG, "draft.txt")
    queue = FakeMergeQueue(repo, states=["verifying", "done"])

    landing = _land_docs(repo, queue)

    assert landing.landed_as == _main(repo) == queue.merge_sha
    assert _files_in(repo, landing.tip_sha) == sorted(DOCS)
    assert _checkout_state(repo, CONFIG, "draft.txt") == state
    assert_no_landing_left_behind(repo)
    _assert_scratch_removed(scratch)


# ---------------------------------------------------------------------------
# Waiting for this request's own terminal state
# ---------------------------------------------------------------------------

def test_another_requests_done_is_not_this_landing(repo, scratch):
    """The queue answers done for every other request: a head that lands
    ahead of this one says nothing about it."""
    queue = FakeMergeQueue(repo, states=["queued", "queued", "done"], foreign_state="done")

    landing, _ = _land(repo, queue)

    polls = queue.calls_to("merge_status")
    assert polls == [{"request_id": queue.request_id}] * 3, polls
    assert landing.request_id == queue.request_id


def test_a_request_superseded_by_a_next_generation_is_followed(repo, scratch):
    queue = FakeMergeQueue(repo, states={
        FakeMergeQueue.REQUEST_ID: [{"state": "superseded", "superseded_by": "mr-next"}],
        "mr-next": ["verifying", "done"],
    })

    landing, _ = _land(repo, queue)

    polls = queue.calls_to("merge_status")
    assert polls == [
        {"request_id": queue.request_id}, {"request_id": "mr-next"}, {"request_id": "mr-next"},
    ], polls
    assert landing.request_id == "mr-next"
    assert (repo / CONFIG).read_bytes() == EDITED


def test_a_request_absorbed_into_a_train_is_not_landed(repo, scratch):
    """MCP submissions are never coalesced into a train, so a coalesce-*
    successor is a shape this module does not chase."""
    before = _main(repo)
    train = "coalesce-mr-other-0a1b2c3d"
    queue = FakeMergeQueue(repo, states=[{"state": "superseded", "superseded_by": train}])

    refusal = _refused_by(repo, queue)

    assert refusal.failure is LandingFailure.NOT_LANDED
    assert "superseded" in refusal.detail and train in refusal.detail, refusal.detail
    _assert_untouched(repo, before, scratch)


def test_an_attached_submit_polls_the_inflight_request(repo, scratch):
    inflight = "mr-inflight"
    queue = FakeMergeQueue(
        repo,
        submit={"status": "attached", "request_id": inflight,
                "inflight_request_id": inflight, "poll_by": "request_id"},
        states={inflight: ["verifying", "done"]},
    )

    _land(repo, queue)

    assert queue.calls_to("merge_status") == [{"request_id": inflight}] * 2


def test_an_attached_submit_without_a_request_id_polls_the_task(repo, scratch):
    queue = FakeMergeQueue(
        repo,
        submit={"status": "attached", "request_id": "mr-own-submission",
                "inflight_request_id": None, "inflight_task_id": "inflight-task",
                "poll_by": "task_id"},
        states={"inflight-task": ["done"]},
    )

    _land(repo, queue)

    assert queue.calls_to("merge_status") == [{"task_id": "inflight-task"}]


def test_journaled_and_unknown_keep_polling_to_done(repo, scratch):
    queue = FakeMergeQueue(repo, states=["journaled", "unknown", "done"])

    landing, _ = _land(repo, queue)

    assert len(queue.calls_to("merge_status")) == 3
    assert landing.landed_as == _main(repo)


@pytest.mark.parametrize("state", ["no_record", "stale_record"])
def test_an_unattributable_answer_with_the_commit_on_main_is_a_landing(repo, scratch, state):
    queue = FakeMergeQueue(
        repo, states=["verifying", {"state": state, "reason": "landing_unattributable"}],
        lands_on={state},
    )

    landing, _ = _land(repo, queue)

    assert landing.landed_as == _main(repo)
    assert (repo / CONFIG).read_bytes() == EDITED
    _assert_left_nothing_behind(repo, scratch)


def test_an_attached_submit_naming_no_handle_polls_the_deploy_branch(repo, scratch):
    """The server's attach can name neither an in-flight request nor a task;
    the deploy branch is unique, so it names this deploy's request."""
    queue = FakeMergeQueue(
        repo,
        submit={"status": "attached", "inflight_request_id": None, "inflight_task_id": None,
                "poll_by": "branch", "pollable": False},
        states=["verifying", "done"],
    )

    landing, _ = _land(repo, queue)

    (submitted,) = queue.calls_to("merge_request")
    assert queue.calls_to("merge_status") == [{"branch": submitted["branch"]}] * 2
    assert landing.landed_as == _main(repo)


def test_no_record_with_nothing_on_main_is_not_landed(repo, scratch):
    before = _main(repo)
    queue = FakeMergeQueue(repo, states=["queued", {"state": "no_record", "reason": "never_submitted"}])

    refusal = _refused_by(repo, queue)

    assert refusal.failure is LandingFailure.NOT_LANDED
    assert "no_record" in refusal.detail, refusal.detail
    _assert_untouched(repo, before, scratch)


def test_a_poll_the_transport_drops_is_retried(repo, scratch, capsys):
    """A target restart mid-verify drops a poll; the merge may still land."""
    queue = FakeMergeQueue(repo, states=["verifying", TRANSPORT_FAULT, "done"])

    landing, _ = _land(repo, queue)

    assert landing.landed_as == _main(repo)
    assert "merge_status" in capsys.readouterr().err
    _assert_left_nothing_behind(repo, scratch)


def test_a_submit_whose_reply_was_lost_is_followed_by_its_branch(repo, scratch, capsys):
    """The target restarted mid-submit: the queue may hold the request, so it
    is asked about the deploy branch and the request it names is landed."""
    queue = FakeMergeQueue(repo, lose_submit_reply=True, states=["verifying", "done"])

    landing, _ = _land(repo, queue)

    (submitted,) = queue.calls_to("merge_request")
    assert queue.calls_to("merge_status") == [{"branch": submitted["branch"]}] * 2
    assert landing.request_id == queue.request_id
    assert landing.landed_as == _main(repo) == queue.merge_sha
    assert "merge_request" in capsys.readouterr().err
    _assert_left_nothing_behind(repo, scratch)


def test_a_lost_submit_the_queue_never_recorded_is_torn_down(repo, scratch):
    before = _main(repo)
    queue = FakeMergeQueue(
        repo, lose_submit_reply=True, states=[{"state": "no_record", "reason": "never_submitted"}],
    )

    refusal = _refused_by(repo, queue)

    assert refusal.failure is LandingFailure.NOT_LANDED
    assert "no_record" in refusal.detail, refusal.detail
    _assert_untouched(repo, before, scratch)


def test_a_lost_submit_the_queue_never_answers_for_keeps_the_deploy_branch(repo, scratch):
    """Nothing rules out that the queue holds the request, so the worktree it
    would land from is kept."""
    queue = FakeMergeQueue(repo, lose_submit_reply=True, states=[TRANSPORT_FAULT])

    refusal = _refused_at_the_ceiling(repo, queue, passed_by="merge_status")

    assert refusal.failure is LandingFailure.TIMED_OUT
    assert len(queue.calls_to("merge_status")) == 1
    assert refusal.state == "unknown"
    assert queue.calls_to("merge_cancel") == []
    (submitted,) = queue.calls_to("merge_request")
    worktree, branch = submitted["worktree"], f"task/{submitted['branch']}"
    assert os.path.isdir(worktree), "a request that may be live lost its worktree"
    for named in (worktree, branch):
        assert named in refusal.detail, (named, refusal.detail)


def test_teardown_tolerates_a_lane_that_already_removed_the_deploy_branch(repo, scratch):
    def lane_cleans_up(repo, branch, worktree):
        git(repo, "worktree", "remove", "--force", worktree)
        git(repo, "branch", "-D", branch)

    queue = FakeMergeQueue(repo, after_landing=lane_cleans_up)

    landing, _ = _land(repo, queue)

    assert landing.landed_as == _main(repo)
    _assert_left_nothing_behind(repo, scratch)


def test_a_teardown_git_refuses_keeps_the_landing(repo, scratch, capsys):
    """The knob is on main: a deploy branch git will not remove is reported,
    not raised over the landing."""
    def lane_locks_the_worktree(repo, branch, worktree):
        git(repo, "worktree", "lock", worktree)

    queue = FakeMergeQueue(repo, after_landing=lane_locks_the_worktree)

    landing, _ = _land(repo, queue)

    assert landing.landed_as == _main(repo)
    assert (repo / CONFIG).read_bytes() == EDITED
    err = capsys.readouterr().err
    assert landing.branch in err and "locked" in err, err


class _SignalStandIn:
    """A stop signal's disposition for a test: records each delivery rather
    than ending the test run, so a signal land_file failed to take over, or
    re-delivered, is observable."""

    def __init__(self):
        self.caught = []

    def __call__(self, signum, frame):
        self.caught.append(signum)


@contextlib.contextmanager
def _stand_in_for(received):
    stand_in = _SignalStandIn()
    previous = signal.signal(received, stand_in)
    try:
        yield stand_in
    finally:
        signal.signal(received, previous)


def _acting_after(handler, action, *, when=lambda reply: True):
    """*handler*, running *action* once it has a reply *when* accepts and
    before that reply goes out."""
    def act_mid_call(arguments):
        reply = handler(arguments)
        if when(reply):
            action()
        return reply
    return act_mid_call


def _terminating_after(handler, *, when=lambda reply: True):
    """*handler*, sending this process SIGTERM once it has a reply *when*
    accepts and holding that reply back a moment, so the signal reaches the
    main thread while it is still waiting on the call."""
    def terminate():
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.2)
    return _acting_after(handler, terminate, when=when)


@pytest.mark.parametrize("received", [signal.SIGTERM, signal.SIGINT], ids=lambda sig: sig.name)
def test_a_stop_signal_while_polling_cancels_and_tears_down(repo, scratch, received):
    """The deterministic runner stops a deploy with killpg, an operator with
    ^C: either must cancel the request rather than kill the process with it
    still live."""
    before = _main(repo)
    queue = FakeMergeQueue(repo, states=["verifying"])

    def stop_once_polled():
        deadline = time.monotonic() + 10
        while not queue.calls_to("merge_status") and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), received)

    with _stand_in_for(received) as stand_in:
        threading.Thread(target=stop_once_polled, daemon=True).start()
        refusal = _refused_by(repo, queue, timeout_secs=20)
        restored = signal.getsignal(received)

    assert stand_in.caught == [], f"land_file let {received.name} through to the previous handler"
    assert restored is stand_in, f"land_file did not restore the previous {received.name} handler"
    assert refusal.failure is LandingFailure.INTERRUPTED
    assert queue.calls_to("merge_cancel") == [{"request_id": queue.request_id}]
    _assert_untouched(repo, before, scratch)


def test_sigterm_during_the_submit_is_held_until_the_request_can_be_cancelled(repo, scratch):
    """Cut into the submit, a stop would leave a request the queue accepted
    with no id to cancel, and tear its worktree down under it."""
    before = _main(repo)
    queue = FakeMergeQueue(repo, states=["verifying"])
    handlers = queue.handlers()
    handlers["merge_request"] = _terminating_after(handlers["merge_request"])

    with _stand_in_for(signal.SIGTERM) as stand_in, \
            FakeEscalationMcp(tool_handlers=handlers) as server:
        refusal = _refusal(repo, server.port, timeout_secs=20)

    assert stand_in.caught == [], "land_file let SIGTERM through to the previous handler"
    assert refusal.failure is LandingFailure.INTERRUPTED, refusal.detail
    assert queue.calls_to("merge_cancel") == [{"request_id": queue.request_id}], refusal.detail
    assert queue.calls_to("merge_status") == []
    _assert_untouched(repo, before, scratch)


def test_sigterm_held_while_the_request_lands_is_redelivered(repo, scratch):
    """A stop held through the poll that answers done leaves a real landing
    to report; the stop then reaches the caller's own handler."""
    queue = FakeMergeQueue(repo, states=["verifying", "done"])
    handlers = queue.handlers()
    handlers["merge_status"] = _terminating_after(
        handlers["merge_status"], when=lambda reply: reply["state"] == "done",
    )

    with _stand_in_for(signal.SIGTERM) as stand_in, \
            FakeEscalationMcp(tool_handlers=handlers) as server:
        landing = land_file(
            str(repo), CONFIG, _main(repo), EDITED,
            port=str(server.port), label=LABEL, message=MESSAGE, timeout_secs=20,
        )

    assert landing.landed_as == _main(repo) == queue.merge_sha
    assert stand_in.caught == [signal.SIGTERM], "a held SIGTERM was swallowed"
    assert queue.calls_to("merge_cancel") == []
    _assert_left_nothing_behind(repo, scratch)


def test_failure_tags_are_the_wire_vocabulary():
    """A calling script prints these strings in its verdict, so renaming one
    breaks every consumer that branches on it."""
    assert {failure.value for failure in LandingFailure} == {
        "transport_not_importable",
        "transport_failed",
        "submit_rejected",
        "not_landed",
        "timed_out",
        "interrupted",
        "queue_done_not_on_main",
        "checkout_not_synced",
        "checkout_dirty",
        "invalid_change",
        "scratch_write_failed",
        "git_failed",
    }
