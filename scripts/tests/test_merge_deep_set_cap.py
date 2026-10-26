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
from pathlib import Path

import pytest
from config_reload_script_fakes import (
    TRANSPORT_FAULTS,
    ClosedPort,
    FakeEscalationMcp,
    commit_config,
    git,
    head,
    init_repo,
    path_python3_shimmed_to,
    reload_report,
    system_python_without_the_transport,
    venvless_checkout_copy,
)
from merge_queue_fakes import FakeMergeQueue, assert_target_checkout_clean, deploy_tmpdir_env

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


@pytest.fixture
def deploy_env(tmp_path):
    """This process's env with TMPDIR a fresh, empty directory."""
    return deploy_tmpdir_env(tmp_path)


def _run(server, config, cap, *, env, script=SCRIPT):
    """Run the script for *cap* against *config*, pointing its landing and
    reload at *server*'s real ephemeral port."""
    return subprocess.run(
        ["bash", str(script), cap, str(config), str(server.port)],
        env=env, capture_output=True, text=True, timeout=120,
    )


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

@pytest.mark.parametrize("seed", list(SEEDS.values()), ids=list(SEEDS))
def test_a_cap_change_lands_through_the_queue_then_is_hot_applied(tmp_path, deploy_env, seed):
    """0 -> 6 lands ONE deploy commit, touching only the yaml, as a merge of
    a task/deploy-* branch the queue verified; only then does the script
    reload, and it reads the knob out of the reload's `applied` disposition.
    Every tool call handshakes and releases its session."""
    config = commit_config(init_repo(tmp_path), seed)
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
    assert "merge_deep:\n  chain_cap: 6\n" in landed, landed
    assert "chain_cap: 0" not in landed, landed

    (submitted,) = queue.calls_to("merge_request")
    assert (f"merge-deep-set-cap: landed task/{submitted['branch']} "
            f"request_id={queue.request_id} as {queue.merge_sha}") in proc.stdout, proc.stdout
    assert f"{KNOB} applied old=0 new=6" in proc.stdout, proc.stdout
    assert proc.stdout.splitlines()[-1].startswith("merge-deep-set-cap: done cap=6")

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

    assert proc.returncode != 0, f"stdout={proc.stdout}"
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

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert "timed out" in proc.stderr and "verifying" in proc.stderr, proc.stderr
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
            deadline = time.monotonic() + 60
            while not queue.calls_to("merge_status"):
                assert proc.poll() is None, f"the script exited before polling: {proc.communicate()}"
                assert time.monotonic() < deadline, "the script never polled its request"
                time.sleep(0.02)
            os.killpg(proc.pid, signal.SIGTERM)
            _, stderr = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        tools = _tool_names(server)

    assert proc.returncode != 0, f"stderr={stderr}"
    assert "SIGTERM" in stderr and "verifying" in stderr, stderr
    assert queue.calls_to("merge_cancel") == [{"request_id": queue.request_id}]
    assert "reload_config" not in tools, tools
    assert head(repo) == before
    _assert_left_nothing_behind(repo, deploy_env)


def test_rerunning_at_the_cap_main_carries_dies_before_any_request(tmp_path, deploy_env):
    """Pins the documented non-idempotence: at the cap main already carries
    there is nothing to land, and the script dies before the server sees
    anything.

    The seed puts the merge_deep block LAST with no blank line after it on
    purpose: step 1's editor drops a blank line FOLLOWING the block, so with
    one there the re-run is not byte-identical and the no-op path this pins is
    never exercised.
    """
    repo = init_repo(tmp_path)
    config = commit_config(repo, "project_root: /nowhere\nmerge_deep:\n  chain_cap: 6\n")
    before = head(repo)
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config))) as server:
        proc = _run(server, config, "6", env=deploy_env)
        received = list(server.received)

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert f"nothing to land: main already carries {KNOB}=6" in proc.stderr, proc.stderr
    assert received == [], f"a no-op deploy reached the server: {received}"
    assert head(repo) == before
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

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert "leading zero" in proc.stderr, f"stderr={proc.stderr}"
    assert head(repo) == before
    assert config.read_text() == SEEDS["block_present"]
    assert received == [], f"a refused cap reached the server: {received}"
    _assert_left_nothing_behind(repo, deploy_env)


def test_a_dead_socket_fails_at_submit_without_claiming_a_landing(tmp_path, deploy_env):
    """Nothing listening at all: the landing's submit fails, so nothing landed
    and no committed value may be claimed."""
    config = _seeded(tmp_path)
    repo = config.parent
    before = head(repo)

    proc = _run(ClosedPort(), config, "6", env=deploy_env)

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert "merge_request never reached the tool" in proc.stderr, proc.stderr
    assert "committed as" not in proc.stderr, proc.stderr
    assert "lands at the next restart" not in proc.stderr, proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    assert head(repo) == before
    _assert_left_nothing_behind(repo, deploy_env)


# ---------------------------------------------------------------------------
# A landed cap whose reload does NOT confirm it live
#
# Each case pins its SPECIFIC diagnostic, not merely a non-zero exit, so one
# failure cannot stand in for another and pass for the wrong reason.
# ---------------------------------------------------------------------------

TRANSPORT_MARKER = "reload_config never reached the tool"


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

    assert proc.returncode != 0, f"a transport fault was read as success: {proc.stdout}"
    assert queue.has_landed() and head(repo) == queue.merge_sha
    assert TRANSPORT_MARKER in proc.stderr, f"stderr={proc.stderr}"
    assert queue.merge_sha in proc.stderr, f"stderr={proc.stderr}"
    assert "lands at the next restart" in proc.stderr, f"stderr={proc.stderr}"
    assert "applied disposition missing" not in proc.stderr, (
        f"a request that never reached the tool was blamed on the disposition: {proc.stderr}"
    )
    assert "Traceback" not in proc.stderr, f"stderr={proc.stderr}"
    _assert_left_nothing_behind(repo, deploy_env)


DISPOSITION_FAILURES = {
    # The knob present under `applied` at some OTHER value -- a concurrent
    # edit, say -- is not the cap this deploy set.
    "applied_a_different_value": (
        {"applied": {KNOB: {"old": 0, "new": 32}}}, f"{KNOB} does not carry 6",
    ),
    # A cap the reload deferred to a restart is landed but NOT live.
    "restart_required": (
        {"applied": {}, "restart_required": APPLIED_6}, f"applied disposition missing {KNOB}",
    ),
    # reload_config's OWN error field: a config that failed to parse, reported
    # by a perfectly successful tools/call.
    "reload_config_error": (
        {"reloaded": False, "error": "load_config: while parsing a block mapping"},
        "reload_config error",
    ),
}


@pytest.mark.parametrize(
    "overrides,marker", list(DISPOSITION_FAILURES.values()), ids=list(DISPOSITION_FAILURES)
)
def test_a_landed_cap_the_reload_does_not_apply_fails(tmp_path, deploy_env, overrides, marker):
    config = _seeded(tmp_path)
    repo = config.parent
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config), **overrides)) as server:
        proc = _run(server, config, "6", env=deploy_env)

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert marker in proc.stderr, f"stderr={proc.stderr}"
    assert "applied old=" not in proc.stdout, proc.stdout
    assert head(repo) == queue.merge_sha
    _assert_left_nothing_behind(repo, deploy_env)


# ---------------------------------------------------------------------------
# Which interpreter runs the landing and the reload
# ---------------------------------------------------------------------------

def test_the_landing_ignores_a_path_python3_that_cannot_import_the_transport(tmp_path):
    """PATH's `python3` cannot import the transport; the cap still lands.

    That can only happen if the landing and reload step resolved the SCRIPT's
    own checkout venv rather than inheriting PATH. Step 1's YAML editor is
    stdlib-only and keeps working under the shim, so this isolates the
    landing interpreter specifically.
    """
    system_python, why = system_python_without_the_transport()
    if system_python is None:
        pytest.skip(why)
    if not CHECKOUT_VENV_PYTHON.exists():
        pytest.skip(f"{CHECKOUT_VENV_PYTHON} is absent (an un-synced worktree)")

    config = _seeded(tmp_path)
    repo = config.parent
    env = deploy_tmpdir_env(tmp_path, path_python3_shimmed_to(tmp_path, system_python))
    queue = FakeMergeQueue(repo)

    with _target(queue, reload_report(config_path=str(config), applied=APPLIED_6)) as server:
        proc = _run(server, config, "6", env=env)

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert f"{KNOB} applied old=0 new=6" in proc.stdout, proc.stdout
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

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert tried in proc.stderr, f"stderr={proc.stderr}"
    assert "uv run --project shared" in proc.stderr, f"stderr={proc.stderr}"
    assert "Traceback" not in proc.stderr, f"stderr={proc.stderr}"
    assert received == [], f"an interpreter without the transport reached the server: {received}"
    assert head(repo) == before
    _assert_left_nothing_behind(repo, env)
