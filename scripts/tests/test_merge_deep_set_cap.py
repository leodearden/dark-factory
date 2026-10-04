"""Tests for merge-deep-set-cap.sh — the regression suite for task 5398.

Drives the real script via subprocess against a REAL loopback stateful
escalation MCP on an ephemeral port and a REAL temp git repo holding a temp
dark-factory-orchestrator.yaml (both from config_reload_script_fakes). Two
defects lived in the script's reload: a single-shot `tools/call` POST, which
the stateful server rejects with a 400 before any tool runs, and a step 4
that piped the response into `python3 -` — whose stdin is the heredoc
program, so the pipe was silently discarded. Only a real socket catches the
first; a faked `curl` returns what its author already believes the wire
looks like.

Nothing here touches a live orchestrator: the only server reachable is the
one the test itself owns, bound to an ephemeral port.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from config_reload_script_fakes import (
    TRANSPORT_FAULTS,
    ClosedPort,
    FakeEscalationMcp,
    commit_config,
    head,
    init_repo,
    path_python3_shimmed_to,
    reload_report,
    system_python_without_the_transport,
)

SCRIPT = Path(__file__).parent.parent / "merge-deep-set-cap.sh"
KNOB = "merge_deep.chain_cap"
CHECKOUT_VENV_PYTHON = SCRIPT.parent.parent / ".venv" / "bin" / "python3"

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


def _run(server, config, cap, *, script=SCRIPT, env=None):
    """Run the script for *cap* against *config*, pointing its reload at
    *server*'s real ephemeral port. *script* and *env* are the seams the
    interpreter-resolution tests need."""
    return subprocess.run(
        ["bash", str(script), cap, str(config), str(server.port)],
        env=env, capture_output=True, text=True, timeout=60,
    )


def _git_out(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout


# ---------------------------------------------------------------------------
# The forward transition, hot-applied
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", list(SEEDS.values()), ids=list(SEEDS))
def test_a_cap_change_is_hot_applied_through_the_session_handshake(tmp_path, seed):
    """0 -> 6 edits the yaml, lands one commit touching only that file, and
    reads the knob out of the reload's `applied` disposition — fetched
    THROUGH the session handshake, with the session released afterwards."""
    config = commit_config(init_repo(tmp_path), seed)
    repo = config.parent
    before = head(repo)

    with FakeEscalationMcp(reload_report(
        config_path=str(config),
        applied={KNOB: {"old": 0, "new": 6}},
    )) as server:
        proc = _run(server, config, "6")
        methods = server.rpc_methods()
        tools = server.called_tools()
        deletes = server.delete_count()

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert f"{KNOB} applied old=0 new=6" in proc.stdout, proc.stdout
    assert proc.stdout.splitlines()[-1].startswith("merge-deep-set-cap: done cap=6")

    landed = _git_out(repo, "rev-list", f"{before}..HEAD").split()
    assert len(landed) == 1, f"expected exactly one new commit, got {landed}"
    touched = _git_out(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert touched == ["dark-factory-orchestrator.yaml"]
    text = config.read_text()
    assert "merge_deep:\n  chain_cap: 6\n" in text, text
    assert "chain_cap: 0" not in text, text

    assert "initialize" in methods, f"the script never handshook; the server saw {methods}"
    assert "tools/call" in methods[methods.index("initialize"):], methods
    assert set(tools) == {"reload_config"}, f"methods={methods}"
    assert deletes == 1, (
        f"the session was opened and not released; the server saw {methods} "
        f"and {deletes} DELETEs"
    )


# ---------------------------------------------------------------------------
# Responses that must NOT be read as success
#
# Each case pins its SPECIFIC diagnostic, not merely a non-zero exit, so one
# failure cannot stand in for another and pass for the wrong reason.
# ---------------------------------------------------------------------------

TRANSPORT_MARKER = "reload_config never reached the tool"


def _seeded(tmp_path, seed="block_present"):
    """A committed temp repo carrying SEEDS[*seed*]. Returns the yaml path."""
    return commit_config(init_repo(tmp_path), SEEDS[seed])


def _short_head(repo):
    return _git_out(repo, "rev-parse", "--short", "HEAD").strip()


def _assert_failed_closed_on_the_transport(proc, repo):
    """The one verdict every transport fault must reach: a loud failure naming
    the TRANSPORT, the sha the cap is already committed as, and the restart
    that will pick it up -- never the orchestrator's disposition, which a
    request that reached no tool says nothing about."""
    assert proc.returncode != 0, f"a transport fault was read as success: {proc.stdout}"
    assert TRANSPORT_MARKER in proc.stderr, f"stderr={proc.stderr}"
    assert _short_head(repo) in proc.stderr, f"stderr={proc.stderr}"
    assert "lands at the next restart" in proc.stderr, f"stderr={proc.stderr}"
    assert "applied disposition missing" not in proc.stderr, (
        f"a request that never reached the tool was blamed on the disposition: {proc.stderr}"
    )
    assert "Traceback" not in proc.stderr, f"stderr={proc.stderr}"


@pytest.mark.parametrize(
    "fault", list(TRANSPORT_FAULTS.values()), ids=list(TRANSPORT_FAULTS)
)
def test_a_reload_that_never_reached_the_tool_fails_closed(tmp_path, fault):
    """Every way the transport can refuse to deliver reload_config, even with
    a report that WOULD have passed waiting behind it."""
    config = _seeded(tmp_path)

    with FakeEscalationMcp(reload_report(
        config_path=str(config),
        applied={KNOB: {"old": 0, "new": 6}},
    ), **fault) as server:
        proc = _run(server, config, "6")

    _assert_failed_closed_on_the_transport(proc, config.parent)


def test_a_dead_socket_fails_closed(tmp_path):
    """Nothing listening at all -- httpx's own transport exception, which is
    neither of the two census_trigger raises and must fail the same way."""
    config = _seeded(tmp_path)

    proc = _run(ClosedPort(), config, "6")

    _assert_failed_closed_on_the_transport(proc, config.parent)


def test_applied_knob_carrying_a_different_value_fails(tmp_path):
    """The knob present under `applied` at some OTHER value -- a concurrent
    edit, say -- is not the cap this deploy set."""
    config = _seeded(tmp_path)

    with FakeEscalationMcp(reload_report(
        config_path=str(config),
        applied={KNOB: {"old": 0, "new": 32}},
    )) as server:
        proc = _run(server, config, "6")

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert f"{KNOB} does not carry 6" in proc.stderr, f"stderr={proc.stderr}"
    assert "applied old=" not in proc.stdout, proc.stdout


def test_a_knob_bucketed_restart_required_is_not_hot_applied(tmp_path):
    """Pins existing behaviour: a cap the reload deferred to a restart is
    committed but NOT live, so the deploy gate fails."""
    config = _seeded(tmp_path)

    with FakeEscalationMcp(reload_report(
        config_path=str(config),
        applied={},
        restart_required={KNOB: {"old": 0, "new": 6}},
    )) as server:
        proc = _run(server, config, "6")

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert f"applied disposition missing {KNOB}" in proc.stderr, f"stderr={proc.stderr}"


def test_a_reload_config_error_fails_loudly(tmp_path):
    """Pins existing behaviour: reload_config's OWN error field -- a config
    that failed to parse, reported by a perfectly successful tools/call."""
    config = _seeded(tmp_path)

    with FakeEscalationMcp(reload_report(
        reloaded=False, error="load_config: while parsing a block mapping",
    )) as server:
        proc = _run(server, config, "6")

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert "reload_config error" in proc.stderr, f"stderr={proc.stderr}"


def test_rerunning_at_the_current_cap_dies_before_the_reload(tmp_path):
    """Pins the documented non-idempotence: at the current cap there is
    nothing to commit, and the commit gate dies before the reload.

    The seed puts the merge_deep block LAST with no blank line after it on
    purpose: step 1's editor drops a blank line FOLLOWING the block, so with
    one there the re-run is not byte-identical, a commit lands, and the
    no-op path this pins is never exercised.
    """
    repo = init_repo(tmp_path)
    config = commit_config(repo, "project_root: /nowhere\nmerge_deep:\n  chain_cap: 6\n")
    before = head(repo)

    with FakeEscalationMcp(reload_report(config_path=str(config))) as server:
        proc = _run(server, config, "6")
        received = list(server.received)

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert "git commit --only" in proc.stderr, f"stderr={proc.stderr}"
    assert head(repo) == before
    assert received == [], f"the commit gate must precede the reload: {received}"


# ---------------------------------------------------------------------------
# Which interpreter runs the reload
# ---------------------------------------------------------------------------

def test_the_reload_ignores_a_path_python3_that_cannot_import_the_transport(tmp_path):
    """PATH's `python3` cannot import the transport; the cap still lands.

    That can only happen if the reload step resolved the SCRIPT's own
    checkout venv rather than inheriting PATH. Step 1's YAML editor is
    stdlib-only and keeps working under the shim, so this isolates the
    reload interpreter specifically.
    """
    system_python, why = system_python_without_the_transport()
    if system_python is None:
        pytest.skip(why)
    if not CHECKOUT_VENV_PYTHON.exists():
        pytest.skip(f"{CHECKOUT_VENV_PYTHON} is absent (an un-synced worktree)")

    config = commit_config(init_repo(tmp_path), SEEDS["block_present"])
    env = path_python3_shimmed_to(tmp_path, system_python)

    with FakeEscalationMcp(reload_report(
        config_path=str(config),
        applied={KNOB: {"old": 0, "new": 6}},
    )) as server:
        proc = _run(server, config, "6", env=env)

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert f"{KNOB} applied old=0 new=6" in proc.stdout, proc.stdout


def test_an_interpreter_without_the_transport_fails_loud_with_a_remedy(tmp_path):
    """No venv to fall back on: fail naming the interpreter and the remedy.

    The script is copied into a checkout carrying the real `scripts/legibility`
    but NO `.venv`, so the bare-`python3` fallback leg is taken and the
    interpreter is the only thing that differs from the passing case above. A
    raw ImportError traceback would leave an operator with no next step.
    """
    system_python, why = system_python_without_the_transport()
    if system_python is None:
        pytest.skip(why)

    scripts_dir = tmp_path / "checkout" / "scripts"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(SCRIPT, scripts_dir / SCRIPT.name)
    (scripts_dir / "legibility").symlink_to(SCRIPT.parent / "legibility")

    config = _seeded(tmp_path)
    env = path_python3_shimmed_to(tmp_path, system_python)
    tried = subprocess.run(
        [system_python, "-c", "import sys; print(sys.executable)"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    with FakeEscalationMcp(reload_report(
        config_path=str(config),
        applied={KNOB: {"old": 0, "new": 6}},
    )) as server:
        proc = _run(server, config, "6", script=scripts_dir / SCRIPT.name, env=env)

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert tried in proc.stderr, f"stderr={proc.stderr}"
    assert "uv run --project shared" in proc.stderr, f"stderr={proc.stderr}"
    assert "Traceback" not in proc.stderr, f"stderr={proc.stderr}"
