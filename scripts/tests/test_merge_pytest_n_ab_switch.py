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

import pytest

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


# ---------------------------------------------------------------------------
# The converged path: a crash-resume onto a config that already carries the
# value, against an orchestrator whose live config already matches it
# ---------------------------------------------------------------------------

def test_converged_resume_exits_zero_when_reload_reports_no_verify_env_change(tmp_path):
    """Re-running the switch for a value the file and the running config
    BOTH already carry is success, not failure.

    The reload genuinely happens and genuinely applies other hot leaves, but
    verify_env is equal on both sides so it never appears under `applied` --
    `unchanged` is a bare int count, so absence is the only converged signal
    on the wire. Nothing to commit is likewise already success (step 2's
    `already-at-<value>` sentinel); step 4 must reach the same verdict.
    """
    config = _make_repo(tmp_path, "8", marker=True)
    repo = config.parent
    before_head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    before_bytes = config.read_bytes()

    proc = _run(tmp_path, config, "8", _envelope(
        reloaded=True,
        error=None,
        config_path=str(config),
        applied={"max_turns.architect": {"old": 40, "new": 60}},
    ))

    assert proc.returncode == 0, f"stdout={proc.stdout} stderr={proc.stderr}"
    assert "applied.verify_env does not carry" not in proc.stderr
    verdict = _verdict(proc)
    assert verdict["outcome"] == "already_converged"
    assert verdict["switched_to"] == "8"
    assert verdict["commit"] == "already-at-8"

    after_head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert after_head == before_head, "a converged re-run must commit nothing"
    assert config.read_bytes() == before_bytes, (
        "the idempotent rewrite must restore the prior A/B marker byte for "
        "byte, not stack a fresh one"
    )


# ---------------------------------------------------------------------------
# Responses that must NOT be read as success
#
# Absence of verify_env from `applied` is strictly WEAKER than "converged":
# the very same absence is produced by a reload that rolled every leaf back,
# and by a reload of a different orchestrator entirely (the port is a
# caller-supplied argument, so a wrong one reaches another project's MCP).
# This is a DEPLOY gate -- blessing an undeployed arm is worse than the
# over-strict assertion the converged branch removes.
# ---------------------------------------------------------------------------

def _converged_verdict_lines(proc):
    """Every stdout line that parses as a verdict claiming convergence."""
    claims = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("outcome") == "already_converged":
            claims.append(line)
    return claims


# Each builder takes (this repo's config, another project's config) and
# returns a reload response with verify_env ABSENT from `applied`.
_UNCORROBORATED_ABSENCE = {
    # Control: the existing error branch already rejects this one, so a green
    # here proves the group's harness really drives the script.
    "reload_failed_loudly": lambda config, other: _envelope(
        reloaded=False, error="load_config: while parsing a block mapping",
        config_path=str(config),
    ),
    # apply_reload rolled every leaf back, so the live config is untouched.
    "reload_failed_silently": lambda config, other: _envelope(
        reloaded=False, error=None, config_path=str(config),
    ),
    # We reloaded something that is not the file we just edited, so its
    # verify_env says nothing about ours.
    "different_config_file": lambda config, other: _envelope(
        reloaded=True, config_path=str(other),
    ),
    "no_config_path": lambda config, other: _envelope(
        reloaded=True, config_path=None,
    ),
}


@pytest.mark.parametrize(
    "build_response",
    list(_UNCORROBORATED_ABSENCE.values()),
    ids=list(_UNCORROBORATED_ABSENCE),
)
def test_uncorroborated_absence_is_not_convergence(tmp_path, build_response):
    """An absent verify_env that is NOT corroborated by a committed reload of
    THIS config file must fail, not pass."""
    config = _make_repo(tmp_path, "8", marker=True)
    other = tmp_path / "other-project" / "dark-factory-orchestrator.yaml"
    other.parent.mkdir()
    other.write_text(f'verify_env:\n  {KEY}: "2"\n')

    proc = _run(tmp_path, config, "8", build_response(config, other))

    assert proc.returncode != 0, (
        f"an uncorroborated absence was read as success: stdout={proc.stdout}"
    )
    assert not _converged_verdict_lines(proc)


def test_applied_verify_env_carrying_a_different_value_still_fails(tmp_path):
    """The strict branch stays strict: a PRESENT verify_env carrying some
    other value is a contradiction, never convergence."""
    config = _make_repo(tmp_path, "8", marker=True)

    proc = _run(tmp_path, config, "8", _envelope(
        reloaded=True,
        config_path=str(config),
        applied={"verify_env": {"old": {KEY: "8"}, "new": {KEY: "16"}}},
    ))

    assert proc.returncode != 0, f"stdout={proc.stdout}"
    assert "applied.verify_env does not carry" in proc.stderr
    assert not _converged_verdict_lines(proc)
