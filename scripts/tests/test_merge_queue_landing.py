"""Tests for scripts/_merge_queue_landing.py's public seam, driven in-process
over a real loopback socket against the stateful escalation MCP fake, with a
FakeMergeQueue behind it that lands for real in a temp git repo."""
from __future__ import annotations

import os
import subprocess
import tempfile

import pytest
from _merge_queue_landing import Landing, land_file
from config_reload_script_fakes import FakeEscalationMcp, commit_config, git, init_repo
from merge_queue_fakes import FakeMergeQueue, assert_target_checkout_clean, deploy_tmpdir_env

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
