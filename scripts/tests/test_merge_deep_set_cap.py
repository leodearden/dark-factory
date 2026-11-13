"""Tests for merge-deep-set-cap.sh: the cap lands through the TARGET's merge
queue, then hot-reloads.

Drives the real script via subprocess against a REAL loopback stateful
escalation MCP on an ephemeral port (config_reload_script_fakes), whose merge
tools answer through a FakeMergeQueue that lands for real in a REAL temp git
repo with main checked out (merge_queue_fakes). A direct commit to a target's
main is drift its queue never verified, which forces every merge in flight
there to re-verify in full (task 7296); the reload half carries the task-5398
regressions, a single-shot `tools/call` POST and a heredoc fed by a pipe, so
only a real socket is a fair test of it.

Every run gets a fresh TMPDIR, so a test can assert the deploy left no
scratch behind. Nothing here touches a live orchestrator: the only server
reachable is the one the test itself owns, bound to an ephemeral port.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
import types
from pathlib import Path

import pytest
from config_reload_script_fakes import (
    TOOL_RAISED,
    TRANSPORT_FAULTS,
    ClosedPort,
    FakeEscalationMcp,
    commit_config,
    failure_tag,
    git,
    head,
    init_repo,
    last_verdict,
    outcome_verdict_lines,
    path_python3_shimmed_to,
    path_python3_that_cannot_run,
    reload_report,
    system_python_without_the_transport,
    venvless_checkout_copy,
)
from merge_queue_fakes import FakeMergeQueue, assert_target_checkout_clean, deploy_tmpdir_env
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / "merge-deep-set-cap.sh"
KNOB = "merge_deep.chain_cap"
CONFIG = "dark-factory-orchestrator.yaml"
CHECKOUT_VENV_PYTHON = SCRIPT.parent.parent / ".venv" / "bin" / "python3"
DEPLOY_MESSAGE = "chore(merge-deep): set merge_deep.chain_cap=6 (deep merge-ahead canary)"
DEPLOY_MERGE_SUBJECT = re.compile(r"^Merge task/deploy-merge-deep-chain-cap-6-\d{8}T\d{6}Z into main$")
APPLIED_6 = {KNOB: {"old": 0, "new": 6}}

SEEDS = {
    "block_present": (
        "project_root: /nowhere\n"
        "merge_deep:\n"
        "  chain_cap: 0\n"
        "\n"
        "review:\n"
        "  enabled: true\n"
    ),
    "block_absent": (
        "project_root: /nowhere\n"
        "review:\n"
        "  enabled: true\n"
    ),
}
SEEDS_AT_6 = {
    "block_present": SEEDS["block_present"].replace("chain_cap: 0", "chain_cap: 6"),
    "block_absent": SEEDS["block_absent"] + "\nmerge_deep:\n  chain_cap: 6\n",
}


@pytest.fixture
def deploy_env(tmp_path):
    """This process's env with TMPDIR a fresh, empty directory."""
    return deploy_tmpdir_env(tmp_path)


def _invoke(*args, env, script=SCRIPT, merge_stderr=False):
    """Run the script with the arguments *args*. *merge_stderr* folds stderr
    into stdout, the one stream the deterministic runner reads a deploy as
    (orchestrator/src/orchestrator/deterministic_runner.py::DeterministicRunner._default_run_script)."""
    return subprocess.run(
        ["bash", str(script), *args], env=env, text=True, timeout=120,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
    )


def _run(server, config, cap, *, env, **options):
    """Run the script for *cap* against *config*, pointing its landing and
    reload at *server*'s real ephemeral port; *options* are `_invoke`'s."""
    return _invoke(cap, str(config), str(server.port), env=env, **options)


def _verdict_after_prose(merged):
    """The verdict ending the merged stream of *merged*, asserted to come
    after the script's prose."""
    *before, _ = [line for line in merged.stdout.splitlines() if line.strip()]
    assert any(line.startswith("merge-deep-set-cap: ") for line in before), merged.stdout
    return last_verdict(merged)


def _seeded(tmp_path, seed="block_present"):
    """A committed temp repo carrying SEEDS[*seed*]. Returns the yaml path."""
    return commit_config(init_repo(tmp_path), SEEDS[seed])


def _target(queue, report=None, **kwargs):
    """The target's escalation MCP: *queue* answers the merge tools, *report*
    answers reload_config."""
    return FakeEscalationMcp(report, tool_handlers=queue.handlers(), **kwargs)


def _tool_names(server):
    return [name for name, _ in server.tool_calls()]


def _assert_left_nothing_behind(repo, env):
    assert_target_checkout_clean(repo)
    scratch = env["TMPDIR"]
    assert os.listdir(scratch) == [], f"the deploy left scratch behind: {os.listdir(scratch)}"


# ---------------------------------------------------------------------------
# The forward transition: landed through the queue, then hot-applied
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", list(SEEDS))
def test_a_cap_change_lands_through_the_queue_then_is_hot_applied(tmp_path, deploy_env, seed):
    """0 -> 6 lands ONE deploy commit, touching only the yaml, as a merge of
    a task/deploy-* branch the queue verified; only then does the script
    reload, and it reads the knob out of the reload's `applied` disposition.
    Every tool call handshakes and releases its session."""
    config = _seeded(tmp_path, seed)
    repo = config.parent
    before = head(repo)
    queue = FakeMergeQueue(repo, states=["verifying", "done"])
    reloads = []

    def reload_config(arguments):
        reloads.append({"landed": queue.has_landed(), "config_on_disk": config.read_text()})
        return reload_report(config_path=str(config), applied=APPLIED_6)

    with FakeEscalationMcp(tool_handlers={**queue.handlers(), "reload_config": reload_config}) as server:
        proc = _run(server, config, "6", env=deploy_env)
        tools = _tool_names(server)
        methods = server.rpc_methods()
        deletes = server.delete_count()

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert tools == ["merge_request", "merge_status", "merge_status", "reload_config"], tools
    (reload,) = reloads
    assert reload["landed"], "reload_config ran before the queue landed the cap"
    assert "merge_deep:\n  chain_cap: 6\n" in reload["config_on_disk"], reload

    assert DEPLOY_MERGE_SUBJECT.match(git(repo, "log", "-1", "--format=%s", "main").strip())
    assert git(repo, "rev-list", "--count", f"{before}..main").strip() == "2"
    deploy_commit = git(repo, "rev-parse", "main^2").strip()
    assert git(repo, "log", "-1", "--format=%s", deploy_commit).strip() == DEPLOY_MESSAGE
    assert git(repo, "show", "--name-only", "--format=", deploy_commit).split() == [CONFIG]
    landed = git(repo, "show", f"{deploy_commit}:{CONFIG}")
    assert landed == SEEDS_AT_6[seed], landed

    (submitted,) = queue.calls_to("merge_request")
    assert (f"merge-deep-set-cap: landed task/{submitted['branch']} "
            f"request_id={queue.request_id} as {queue.merge_sha}") in proc.stderr, proc.stderr
    assert f"{KNOB} applied old=0 new=6" in proc.stderr, proc.stderr
    assert last_verdict(proc) == {"chain_cap": 6, "commit": queue.merge_sha, "outcome": "applied"}

    assert methods.count("initialize") == len(tools), (
        f"a tool call skipped the session handshake; the server saw {methods}"
    )
    assert deletes == len(tools), (
        f"a session was opened and not released; the server saw {methods} and {deletes} DELETEs"
    )
    _assert_left_nothing_behind(repo, deploy_env)


# ---------------------------------------------------------------------------
# A cap the queue did not land is never reloaded
# ---------------------------------------------------------------------------

def test_a_request_the_queue_blocks_fails_naming_the_state_and_reason(tmp_path, deploy_env):
    config = _seeded(tmp_path)
    repo = config.parent
    before = head(repo)
    queue = FakeMergeQueue(repo, states=["verifying", {"state": "blocked", "reason": "verify failed: 3 tests"}])

    with _target(queue, reload_report(config_path=str(config), applied=APPLIED_6)) as server:
        proc = _run(server, config, "6", env=deploy_env)
        tools = _tool_names(server)

    assert failure_tag(proc) == "not_landed"
    assert last_verdict(proc)["commit"] is None
    assert outcome_verdict_lines(proc) == []
    assert "blocked" in proc.stderr and "verify failed: 3 tests" in proc.stderr, proc.stderr
    assert "reload_config" not in tools, tools
    assert head(repo) == before
    _assert_left_nothing_behind(repo, deploy_env)


def test_a_request_still_live_at_the_ceiling_is_cancelled_and_not_reloaded(tmp_path, deploy_env):
    config = _seeded(tmp_path)
    repo = config.parent
    before = head(repo)
    queue = FakeMergeQueue(repo, states=["verifying"])
    env = {**deploy_env, "MERGE_QUEUE_LANDING_TIMEOUT_SECS": "1"}

    with _target(queue, reload_report(config_path=str(config), applied=APPLIED_6)) as server:
        proc = _run(server, config, "6", env=env)
        tools = _tool_names(server)

    assert failure_tag(proc) == "timed_out"
    assert outcome_verdict_lines(proc) == []
    # A loaded host can spend the whole second before the first poll answers,
    # leaving the submit's own 'queued' as the last state the script saw.
    assert any(f"still {live}" in proc.stderr for live in ("queued", "verifying")), proc.stderr
    assert queue.calls_to("merge_cancel") == [{"request_id": queue.request_id}]
    assert "reload_config" not in tools, tools
    assert head(repo) == before
    _assert_left_nothing_behind(repo, env)


def test_sigterm_while_the_landing_polls_cancels_it_and_tears_down(tmp_path, deploy_env):
    """The deterministic runner stops a deploy with killpg: the script must
    cancel its request and clean up rather than die with the request live."""
    config = _seeded(tmp_path)
    repo = config.parent
    before = head(repo)
    queue = FakeMergeQueue(repo, states=["verifying"])

    with _target(queue, reload_report(config_path=str(config), applied=APPLIED_6)) as server:
        proc = subprocess.Popen(
            ["bash", str(SCRIPT), "6", str(config), str(server.port)],
            env=deploy_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        try:
            # Polls are sequential, so a second one means the script has read
            # the first reply: the request is 'verifying' to it.
            deadline = time.monotonic() + 60
            while len(queue.calls_to("merge_status")) < 2:
                assert proc.poll() is None, f"the script exited before polling: {proc.communicate()}"
                assert time.monotonic() < deadline, "the script never polled its request"
                time.sleep(0.02)
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        tools = _tool_names(server)

    # The TERM trap's own `exit 143` follows the landing step's verdict.
    stopped = types.SimpleNamespace(returncode=proc.returncode, stdout=stdout, stderr=stderr)
    assert failure_tag(stopped) == "interrupted"
    assert outcome_verdict_lines(stopped) == []
    assert queue.calls_to("merge_cancel") == [{"request_id": queue.request_id}], stderr
    assert "reload_config" not in tools, tools
    assert head(repo) == before
    _assert_left_nothing_behind(repo, deploy_env)


RERUN_SEEDS = {
    "block_last": "project_root: /nowhere\nmerge_deep:\n  chain_cap: 6\n",
    "blank_line_after_block": SEEDS_AT_6["block_present"],
    "blank_line_at_eof": "project_root: /nowhere\nmerge_deep:\n  chain_cap: 6\n\n",
}


@pytest.mark.parametrize("seed", list(RERUN_SEEDS.values()), ids=list(RERUN_SEEDS))
def test_rerunning_at_the_cap_main_carries_dies_before_any_request(tmp_path, deploy_env, seed):
    """Pins the documented non-idempotence: at the cap main already carries
    there is nothing to land, and the script dies before the server sees
    anything."""
    repo = init_repo(tmp_path)
    config = commit_config(repo, seed)
    before = head(repo)
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config))) as server:
        proc = _run(server, config, "6", env=deploy_env)
        received = list(server.received)

    assert failure_tag(proc) == "nothing_to_land"
    assert received == [], f"a no-op deploy reached the server: {received}"
    assert head(repo) == before
    _assert_left_nothing_behind(repo, deploy_env)


def test_a_cap_inserted_into_the_block_lands_above_its_blank_separator(tmp_path, deploy_env):
    config = commit_config(
        init_repo(tmp_path),
        "project_root: /nowhere\n"
        "merge_deep:\n"
        "  # canary\n"
        "\n"
        "review:\n"
        "  enabled: true\n",
    )
    repo = config.parent
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config), applied=APPLIED_6)) as server:
        proc = _run(server, config, "6", env=deploy_env)

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert queue.has_landed()
    assert git(repo, "show", f"main:{CONFIG}") == (
        "project_root: /nowhere\n"
        "merge_deep:\n"
        "  # canary\n"
        "  chain_cap: 6\n"
        "\n"
        "review:\n"
        "  enabled: true\n"
    )
    _assert_left_nothing_behind(repo, deploy_env)


@pytest.mark.parametrize("cap", ["06", "010"])
def test_a_cap_with_a_leading_zero_is_refused_before_anything_lands(tmp_path, deploy_env, cap):
    """YAML reads `chain_cap: 06` as 6 and `chain_cap: 010` as octal 8, so a
    leading zero would land one spelling and gate on another value -- a
    failure that only surfaces after the landing. The cap is refused at
    validation instead, so the yaml, the history and the server are all
    untouched."""
    repo = init_repo(tmp_path)
    config = commit_config(repo, SEEDS["block_present"])
    before = head(repo)
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config))) as server:
        proc = _run(server, config, cap, env=deploy_env)
        received = list(server.received)

    assert failure_tag(proc) == "invalid_cap"
    assert head(repo) == before
    assert config.read_text() == SEEDS["block_present"]
    assert received == [], f"a refused cap reached the server: {received}"
    _assert_left_nothing_behind(repo, deploy_env)


def test_a_refusal_ends_the_merged_stream_with_its_verdict(tmp_path, deploy_env):
    """The runner reads one merged stream, so bash `die` writes its prose
    before its verdict: the other order ends that stream with prose."""
    config = _seeded(tmp_path)

    merged = _run(types.SimpleNamespace(port=1), config, "06", env=deploy_env, merge_stderr=True)

    assert merged.returncode != 0
    assert _verdict_after_prose(merged) == {"failure": "invalid_cap"}


def test_a_landing_failure_ends_the_merged_stream_with_its_verdict(tmp_path, deploy_env):
    """Step 3's `fail()` keeps the same order, and the teardown after it adds
    nothing to the stream."""
    config = _seeded(tmp_path)
    queue = FakeMergeQueue(config.parent, states=[{"state": "blocked", "reason": "verify failed"}])

    with _target(queue, reload_report(config_path=str(config), applied=APPLIED_6)) as server:
        merged = _run(server, config, "6", env=deploy_env, merge_stderr=True)

    assert merged.returncode != 0
    assert _verdict_after_prose(merged) == {"chain_cap": 6, "commit": None, "failure": "not_landed"}


@pytest.mark.parametrize("bad_port", ["0", "65536", "99999"])
def test_an_unconnectable_port_is_refused_before_the_landing(tmp_path, deploy_env, bad_port):
    """A port outside 1-65535 can never deliver the landing or the reload, so
    it is refused before anything is landed."""
    config = _seeded(tmp_path)
    repo = config.parent
    before = head(repo)

    proc = _run(types.SimpleNamespace(port=bad_port), config, "6", env=deploy_env)

    assert failure_tag(proc) == "invalid_port"
    assert head(repo) == before
    assert config.read_text() == SEEDS["block_present"]
    _assert_left_nothing_behind(repo, deploy_env)


# id -> (arguments built from this repo's config, the tag die() refuses them with).
MALFORMED_INVOCATIONS = {
    "two_arguments": (lambda config: ["6", str(config)], "usage"),
    "config_absent": (lambda config: ["6", str(config.parent / "absent.yaml"), "1"], "config_not_found"),
}


@pytest.mark.parametrize(
    "arguments,tag", list(MALFORMED_INVOCATIONS.values()), ids=list(MALFORMED_INVOCATIONS)
)
def test_a_malformed_invocation_is_refused_with_its_own_tag(tmp_path, deploy_env, arguments, tag):
    config = _seeded(tmp_path)
    repo = config.parent
    before = head(repo)

    proc = _invoke(*arguments(config), env=deploy_env)

    assert failure_tag(proc) == tag
    assert head(repo) == before
    assert config.read_text() == SEEDS["block_present"]
    _assert_left_nothing_behind(repo, deploy_env)


def test_a_config_main_does_not_track_is_refused_before_anything_lands(tmp_path, deploy_env):
    """A config file main does not carry has no committed copy to edit, so
    the run is refused with scripts/lib/deploy_main_copy.sh's own tag before
    the server sees anything. The untracked file leaves the checkout dirty by
    design, so only the scratch space is checked."""
    config = _seeded(tmp_path)
    repo = config.parent
    before = head(repo)
    untracked = repo / "untracked.yaml"
    untracked.write_text(SEEDS["block_present"])
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(untracked))) as server:
        proc = _run(server, untracked, "6", env=deploy_env)
        received = list(server.received)

    assert failure_tag(proc) == "config_not_on_main"
    assert head(repo) == before
    assert received == [], f"an untracked config reached the server: {received}"
    assert os.listdir(deploy_env["TMPDIR"]) == [], os.listdir(deploy_env["TMPDIR"])


def test_a_config_step_1_cannot_edit_ends_in_a_verdict(tmp_path, deploy_env):
    """main's copy is not UTF-8, so step 1's editor cannot read it: the run
    ends in a tagged verdict rather than a bare traceback, before anything
    lands."""
    repo = init_repo(tmp_path)
    config = repo / CONFIG
    config.write_bytes(b"project_root: /nowhere\nmerge_deep:\n  chain_cap: 0\n# \xff\n")
    git(repo, "add", CONFIG)
    git(repo, "commit", "-q", "-m", "seed")
    before = head(repo)
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config))) as server:
        proc = _run(server, config, "6", env=deploy_env)
        received = list(server.received)

    assert failure_tag(proc) == "config_edit_failed"
    assert head(repo) == before
    assert received == [], f"an unedited config reached the server: {received}"
    _assert_left_nothing_behind(repo, deploy_env)


def test_a_dead_socket_fails_at_submit_without_claiming_a_landing(tmp_path, deploy_env):
    """Nothing listening at all: the landing's submit fails, so nothing landed
    and no committed value may be claimed."""
    config = _seeded(tmp_path)
    repo = config.parent
    before = head(repo)

    with ClosedPort() as closed:
        proc = _run(closed, config, "6", env=deploy_env)

    assert failure_tag(proc) == "transport_failed"
    assert last_verdict(proc)["commit"] is None
    assert outcome_verdict_lines(proc) == []
    assert "merge_request never reached the tool" in proc.stderr, proc.stderr
    assert "committed as" not in proc.stderr, proc.stderr
    assert "lands at the next restart" not in proc.stderr, proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    assert head(repo) == before
    _assert_left_nothing_behind(repo, deploy_env)


# ---------------------------------------------------------------------------
# A landed cap whose reload does NOT confirm it live
#
# Each case pins its SPECIFIC failure tag, not merely a non-zero exit, so one
# failure cannot stand in for another and pass for the wrong reason. The cap
# has landed by then, so every verdict names the commit it landed as.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fault", list(TRANSPORT_FAULTS.values()), ids=list(TRANSPORT_FAULTS)
)
def test_a_reload_that_never_reached_the_tool_fails_closed(tmp_path, deploy_env, fault):
    """Every way the transport can refuse to deliver reload_config, armed only
    once the cap has landed, even with a report that WOULD have passed
    waiting behind it: a loud failure naming the TRANSPORT, the sha the cap
    landed as and the restart that will pick it up -- never the
    orchestrator's disposition, which a request that reached no tool says
    nothing about."""
    config = _seeded(tmp_path)
    repo = config.parent
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config), applied=APPLIED_6),
                 faults_armed=queue.has_landed, **fault) as server:
        proc = _run(server, config, "6", env=deploy_env)

    assert failure_tag(proc) == "no_reload_report"
    assert outcome_verdict_lines(proc) == []
    landed_as = head(repo)
    assert queue.has_landed() and landed_as == queue.merge_sha
    assert last_verdict(proc)["commit"] == queue.merge_sha
    assert landed_as in proc.stderr, f"stderr={proc.stderr}"
    assert "lands at the next restart" in proc.stderr, f"stderr={proc.stderr}"
    assert "Traceback" not in proc.stderr, f"stderr={proc.stderr}"
    _assert_left_nothing_behind(repo, deploy_env)


def test_a_reload_that_raised_fails_closed_naming_the_tool(tmp_path, deploy_env):
    """reload_config was reached and raised once the cap had landed: the
    failure names the TOOL, which may have applied the cap before failing,
    never a transport that did not deliver it."""
    config = _seeded(tmp_path)
    repo = config.parent
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config), applied=APPLIED_6),
                 faults_armed=queue.has_landed, **TOOL_RAISED) as server:
        proc = _run(server, config, "6", env=deploy_env)

    assert failure_tag(proc) == "reload_raised"
    assert outcome_verdict_lines(proc) == []
    landed_as = head(repo)
    assert queue.has_landed() and landed_as == queue.merge_sha
    assert last_verdict(proc)["commit"] == queue.merge_sha
    assert landed_as in proc.stderr, f"stderr={proc.stderr}"
    assert "lands at the next restart" in proc.stderr, f"stderr={proc.stderr}"
    assert "Traceback" not in proc.stderr, f"stderr={proc.stderr}"
    _assert_left_nothing_behind(repo, deploy_env)


# id -> (reload-report overrides, the failure tag they must end in).
DISPOSITION_FAILURES = {
    # The knob present under `applied` at some OTHER value -- a concurrent
    # edit, say -- is not the cap this deploy set.
    "applied_a_different_value": (
        {"applied": {KNOB: {"old": 0, "new": 32}}}, "value_mismatch",
    ),
    # A cap the reload deferred to a restart is landed but NOT live.
    "restart_required": (
        {"applied": {}, "restart_required": APPLIED_6}, "restart_required",
    ),
    # A reload of THIS file that changed nothing about the knob: the cap just
    # landed, so the landing did not take effect as expected.
    "knob_absent": ({"applied": {}}, "not_applied"),
    # reload_config's OWN error field: a config that failed to parse, reported
    # by a perfectly successful tools/call.
    "reload_config_error": (
        {"reloaded": False, "error": "load_config: while parsing a block mapping"},
        "reload_error",
    ),
}


@pytest.mark.parametrize(
    "overrides,tag", list(DISPOSITION_FAILURES.values()), ids=list(DISPOSITION_FAILURES)
)
def test_a_landed_cap_the_reload_does_not_apply_fails(tmp_path, deploy_env, overrides, tag):
    config = _seeded(tmp_path)
    repo = config.parent
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config), **overrides)) as server:
        proc = _run(server, config, "6", env=deploy_env)

    assert failure_tag(proc) == tag
    assert outcome_verdict_lines(proc) == []
    assert last_verdict(proc)["commit"] == queue.merge_sha
    assert head(repo) == queue.merge_sha
    _assert_left_nothing_behind(repo, deploy_env)


# id -> (builder taking (this repo's config, tmp_path) and returning
# reload-report fields that fail to show THIS config file was re-read and
# committed, the failure tag naming which corroborator is missing). The gate's
# own suite pins every breach shape; these two prove the script hands the gate
# its config path and keeps the gate's tag.
UNCORROBORATED_RELOADS = {
    # The wrong-port story: another orchestrator's own pending change to the
    # same cap reports exactly the applied entry this deploy expects.
    "another_projects_orchestrator": (
        lambda config, tmp_path: {
            "reloaded": True, "config_path": str(tmp_path / "other-project" / CONFIG),
        },
        "different_config_file",
    ),
    # apply_reload rolled every leaf back, so the live config is untouched.
    "reload_not_committed": (
        lambda config, tmp_path: {"reloaded": False, "error": None, "config_path": str(config)},
        "reload_not_committed",
    ),
}


@pytest.mark.parametrize(
    "report_fields,tag", list(UNCORROBORATED_RELOADS.values()), ids=list(UNCORROBORATED_RELOADS)
)
def test_a_cap_reported_by_an_uncorroborated_reload_is_not_applied(
    tmp_path, deploy_env, report_fields, tag,
):
    """An `applied` entry carrying our cap is no evidence about OUR file unless
    the reload committed a re-read of it, so the deploy fails naming the
    missing corroborator and the commit the cap landed as."""
    config = _seeded(tmp_path)
    repo = config.parent
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(**report_fields(config, tmp_path), applied=APPLIED_6)) as server:
        proc = _run(server, config, "6", env=deploy_env)

    assert failure_tag(proc) == tag
    assert outcome_verdict_lines(proc) == []
    assert last_verdict(proc)["commit"] == queue.merge_sha == head(repo)
    _assert_left_nothing_behind(repo, deploy_env)


# ---------------------------------------------------------------------------
# Which interpreter runs the script's python steps
# ---------------------------------------------------------------------------

def test_no_step_runs_under_path_python3_when_the_checkout_venv_exists(tmp_path):
    """PATH's `python3` cannot run at all; the cap still lands.

    That can only pass if the YAML edit, the landing and the reload all
    resolved the SCRIPT's own checkout venv rather than inheriting PATH.
    """
    if not CHECKOUT_VENV_PYTHON.exists():
        pytest.skip(f"{CHECKOUT_VENV_PYTHON} is absent (an un-synced worktree)")

    config = _seeded(tmp_path)
    repo = config.parent
    env = deploy_tmpdir_env(tmp_path, path_python3_that_cannot_run(tmp_path))
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config), applied=APPLIED_6)) as server:
        proc = _run(server, config, "6", env=env)

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert last_verdict(proc)["outcome"] == "applied"
    assert head(repo) == queue.merge_sha
    _assert_left_nothing_behind(repo, env)


def test_an_interpreter_without_the_transport_fails_loud_before_any_landing(tmp_path):
    """No venv to fall back on: fail naming the interpreter and the remedy,
    before anything is submitted.

    The script is copied into a checkout carrying its real landing and reload
    steps but NO `.venv`, so the bare-`python3` fallback leg is taken and the
    interpreter is the only thing that differs from the passing case above. A
    raw ImportError traceback would leave an operator with no next step.
    """
    system_python, why = system_python_without_the_transport()
    if system_python is None:
        pytest.skip(why)

    script = venvless_checkout_copy(tmp_path, SCRIPT)
    config = _seeded(tmp_path)
    repo = config.parent
    before = head(repo)
    env = deploy_tmpdir_env(tmp_path, path_python3_shimmed_to(tmp_path, system_python))
    tried = subprocess.run(
        [system_python, "-c", "import sys; print(sys.executable)"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config), applied=APPLIED_6)) as server:
        proc = _run(server, config, "6", script=script, env=env)
        received = list(server.received)

    assert failure_tag(proc) == "transport_not_importable"
    assert outcome_verdict_lines(proc) == []
    assert tried in proc.stderr, f"stderr={proc.stderr}"
    assert "uv run --project shared" in proc.stderr, f"stderr={proc.stderr}"
    assert "Traceback" not in proc.stderr, f"stderr={proc.stderr}"
    assert received == [], f"an interpreter without the transport reached the server: {received}"
    assert head(repo) == before
    _assert_left_nothing_behind(repo, env)
