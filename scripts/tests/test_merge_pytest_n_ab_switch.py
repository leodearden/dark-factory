"""Tests for merge-pytest-n-ab-switch.sh — drives the real script via
subprocess against a fake `curl` shimmed onto PATH (the
test_deploy_w5_recon_reliability.py idiom, reduced to one single-shot
canned response) and a REAL temp git repo holding a temp
dark-factory-orchestrator.yaml (the test_deploy_w11_lane_lifecycle.py
idiom), so step 1's YAML editor, step 2's commit idempotency and step 4's
assertion are exercised in COMPOSITION — which is where the crash-resume
defect lives: file already at the value -> nothing to commit -> the reload
reports no verify_env change.

Nothing here touches a live orchestrator: `curl` is faked, so the port
passed is inert.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "merge-pytest-n-ab-switch.sh"
KEY = "PYTEST_XDIST_AUTO_NUM_WORKERS"

# Inert: the fake curl never opens a socket. Deliberately not 8102 (the real
# dark-factory escalation MCP) so a fake that regressed into the real curl
# would fail loudly rather than hot-reload a live orchestrator.
INERT_PORT = "19999"

_RESPONSE_ENV = "FAKE_RELOAD_RESPONSE"


# ---------------------------------------------------------------------------
# Fake curl (canned single-shot response)
# ---------------------------------------------------------------------------

_FAKE_CURL_SRC = '''#!/usr/bin/env python3
"""Fake `curl` for testing merge-pytest-n-ab-switch.sh's reload step.

Ignores argv entirely: the script issues exactly ONE single-shot
tools/call POST, so unlike test_deploy_w5_recon_reliability.py's curl
there is no retry budget to pace and no ordering witness to record --
each test simply writes the one response it wants. Prints the canned
JSON-RPC envelope named by $FAKE_RELOAD_RESPONSE verbatim on stdout and
exits 0, which is what the script's `RESP="$(curl ...)"` captures.
"""
import os
import sys

with open(os.environ["FAKE_RELOAD_RESPONSE"], encoding="utf-8") as fh:
    sys.stdout.write(fh.read())
'''


def _write_fake_curl(tmp_path):
    """Write the executable fake `curl` into <tmp_path>/bin/, return that dir."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "curl"
    fake.write_text(_FAKE_CURL_SRC)
    fake.chmod(0o755)
    return bin_dir


# ---------------------------------------------------------------------------
# Real temp git repo holding a temp dark-factory-orchestrator.yaml
# ---------------------------------------------------------------------------

def _git(repo, *args):
    """Run a git command against *repo*, raising loudly on failure -- test
    setup must never silently produce a repo that doesn't match the spec."""
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path, verify_env_value, *, marker=False):
    """Build a real temp git repo at <tmp_path>/repo carrying a
    dark-factory-orchestrator.yaml whose top-level `verify_env:` block pins
    KEY to *verify_env_value*, and commit it so the tree is clean.

    A real repo (rather than a faked `git`) makes both halves of step 2
    faithful -- the `git rev-parse --show-toplevel` resolution and the
    "nothing to commit" idempotent path this bug lives on -- with no fake to
    drift. `marker=True` precedes the key with an `  # A/B arm: ...` line,
    the state a previously-switched config file is left in.

    Returns the yaml path.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")

    marker_line = (
        f'  # A/B arm: {KEY} set to "{verify_env_value}" at '
        "2026-09-10T12:42:14Z by scripts/merge-pytest-n-ab-switch.sh\n"
    )
    config = repo / "dark-factory-orchestrator.yaml"
    config.write_text(
        "project_root: /nowhere\n"
        "max_concurrent_tasks: 3\n"
        "\n"
        "verify_env:\n"
        + (marker_line if marker else "")
        + f'  {KEY}: "{verify_env_value}"\n'
        '  PYTHONWARNINGS: "ignore"\n'
        "\n"
        "review:\n"
        "  enabled: true\n"
    )
    _git(repo, "add", "dark-factory-orchestrator.yaml")
    _git(repo, "commit", "-q", "-m", "seed")
    return config


# ---------------------------------------------------------------------------
# Wire shapes + script driver
# ---------------------------------------------------------------------------

def _envelope(**report):
    """Wrap a reload report as the JSON-RPC envelope the script unwraps.

    Defaults every field of orchestrator/src/orchestrator/harness.py::
    reload_config's return shape so each test states only what it varies.
    `unchanged` defaults to an INT count because that is what is really on
    the wire (config.py::ConfigDiff declares `unchanged: int`) -- it carries
    no key names, which is why absence from `applied` is the only converged
    signal there is.
    """
    body = {
        "reloaded": True,
        "config_path": None,
        "applied": {},
        "restart_required": {},
        "unchanged": 37,
        "error": None,
    }
    body.update(report)
    return {"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": body}}


def _run(tmp_path, config_path, value, response, *, port=INERT_PORT):
    """Run the real script for *value* against *config_path*, with the fake
    curl on PATH primed to answer the reload with *response*."""
    bin_dir = _write_fake_curl(tmp_path)
    response_path = tmp_path / "reload_response.json"
    response_path.write_text(json.dumps(response))

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env[_RESPONSE_ENV] = str(response_path)
    return subprocess.run(
        ["bash", str(SCRIPT), value, str(config_path), port],
        env=env, capture_output=True, text=True, timeout=60,
    )


def _verdict(proc):
    """Parse the script's last non-empty stdout line as the JSON verdict."""
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, f"no stdout at all; rc={proc.returncode} stderr={proc.stderr}"
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------
# The flip path: a real value change, hot-applied
# ---------------------------------------------------------------------------

def test_flip_reports_outcome_applied_and_commits(tmp_path):
    """A genuine 16 -> 8 switch edits the yaml, lands exactly one commit
    touching only that file, and reports `outcome: applied`."""
    config = _make_repo(tmp_path, "16")
    repo = config.parent
    before = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    proc = _run(tmp_path, config, "8", _envelope(
        reloaded=True,
        config_path=str(config),
        applied={"verify_env": {"old": {KEY: "16"}, "new": {KEY: "8"}}},
    ))

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    verdict = _verdict(proc)
    assert verdict["outcome"] == "applied"
    assert verdict["switched_to"] == "8"
    assert verdict["commit"] != "already-at-8"

    landed = subprocess.run(
        ["git", "-C", str(repo), "rev-list", f"{before}..HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    assert len(landed) == 1, f"expected exactly one new commit, got {landed}"
    touched = subprocess.run(
        ["git", "-C", str(repo), "show", "--name-only", "--format=", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    assert touched == ["dark-factory-orchestrator.yaml"]
    assert f'{KEY}: "8"' in config.read_text()
