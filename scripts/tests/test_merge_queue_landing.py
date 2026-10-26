"""Tests for scripts/_merge_queue_landing.py's public seam, driven in-process
over a real loopback socket against the stateful escalation MCP fake, with a
FakeMergeQueue behind it that lands for real in a temp git repo."""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
import time

import pytest
from _merge_queue_landing import Landing, LandingFailure, LandingNotConfirmed, land_file
from config_reload_script_fakes import (
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
    assert_target_checkout_clean,
    deploy_tmpdir_env,
)

CONFIG = "dark-factory-orchestrator.yaml"
SEED = "project_root: /nowhere\nknob: 0\n"
EDITED = b"project_root: /nowhere\nknob: 1\n"
LABEL = "test-knob-1"
MESSAGE = "chore(test): set knob=1"


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


def _main(repo):
    return git(repo, "rev-parse", "main").strip()


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
    assert os.listdir(scratch) == [], f"the deploy scratch dir was left: {os.listdir(scratch)}"


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


def test_a_submit_that_is_already_done_lands_without_polling(repo, scratch):
    queue = FakeMergeQueue(repo, submit="done")

    landing, calls = _land(repo, queue)

    assert [name for name, _ in calls] == ["merge_request"], calls
    assert landing.landed_as == queue.merge_sha == _main(repo)
    assert (repo / CONFIG).read_bytes() == EDITED
    _assert_left_nothing_behind(repo, scratch)


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


def _assert_untouched(repo, before, scratch):
    assert _main(repo) == before, "main moved on a landing that was refused"
    _assert_left_nothing_behind(repo, scratch)


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
    assert terminal["state"] in refusal.detail, refusal.detail
    assert terminal.get("reason", "") in refusal.detail, refusal.detail
    _assert_untouched(repo, before, scratch)


@pytest.mark.parametrize("status", ["blocked", "conflict", "unknown_branch"])
def test_a_submit_that_settles_without_landing_is_not_landed(repo, scratch, status):
    before = _main(repo)
    queue = FakeMergeQueue(repo, submit={"status": status, "reason": f"why-{status}"})

    refusal = _refused_by(repo, queue)

    assert refusal.failure is LandingFailure.NOT_LANDED
    assert status in refusal.detail, refusal.detail
    assert f"why-{status}" in refusal.detail, refusal.detail
    assert queue.calls_to("merge_status") == []
    _assert_untouched(repo, before, scratch)


def test_a_submit_the_server_refuses_is_submit_rejected(repo, scratch):
    before = _main(repo)
    error = "Merge queue not available — orchestrator not running"
    queue = FakeMergeQueue(repo, submit={"error": error})

    refusal = _refused_by(repo, queue)

    assert refusal.failure is LandingFailure.SUBMIT_REJECTED
    assert error in refusal.detail, refusal.detail
    _assert_untouched(repo, before, scratch)


def test_a_dead_socket_at_submit_is_transport_failed(repo, scratch):
    before = _main(repo)

    refusal = _refusal(repo, ClosedPort().port)

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


def test_a_request_still_live_at_the_ceiling_is_cancelled(repo, scratch):
    before = _main(repo)
    queue = FakeMergeQueue(repo, states=["verifying"])

    refusal = _refused_by(repo, queue, timeout_secs=1)

    assert refusal.failure is LandingFailure.TIMED_OUT
    assert "verifying" in refusal.detail, refusal.detail
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

    refusal = _refused_by(repo, queue, timeout_secs=1)

    assert refusal.failure is LandingFailure.TIMED_OUT
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


@pytest.mark.parametrize("bad", [
    {"base": "0" * 40},
    {"path": "absent.yaml"},
], ids=["base_not_a_commit", "path_not_at_base"])
def test_a_base_without_the_file_is_refused_before_any_call(repo, scratch, bad):
    before = _main(repo)
    queue = FakeMergeQueue(repo)

    refusal = _refused_by(repo, queue, **bad)

    assert refusal.failure is LandingFailure.INVALID_CHANGE
    assert queue.calls == []
    _assert_untouched(repo, before, scratch)


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


def test_sigterm_while_polling_cancels_and_tears_down(repo, scratch):
    """The deterministic runner stops a deploy with killpg: SIGTERM must
    cancel the request rather than kill the process with it still live."""
    before = _main(repo)
    queue = FakeMergeQueue(repo, states=["verifying"])
    caught = []

    def terminate_once_polled():
        deadline = time.monotonic() + 10
        while not queue.calls_to("merge_status") and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGTERM)

    def stand_in(signum, frame):
        caught.append(signum)

    # Stands in for pytest's own disposition, so a SIGTERM land_file failed to
    # take over is recorded rather than ending the test run.
    previous = signal.signal(signal.SIGTERM, stand_in)
    try:
        threading.Thread(target=terminate_once_polled, daemon=True).start()
        refusal = _refused_by(repo, queue, timeout_secs=20)
        restored = signal.getsignal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert caught == [], "land_file let SIGTERM through to the previous handler"
    assert restored is stand_in, "land_file did not restore the previous SIGTERM handler"
    assert refusal.failure is LandingFailure.INTERRUPTED
    assert queue.calls_to("merge_cancel") == [{"request_id": queue.request_id}]
    _assert_untouched(repo, before, scratch)


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
        "invalid_change",
    }
