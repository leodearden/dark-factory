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

import subprocess
from pathlib import Path

import pytest
from config_reload_script_fakes import (
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
