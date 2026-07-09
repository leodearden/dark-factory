"""Tests for enable_laptop_persistent_worktree.sh — drives the script via
subprocess against a throwaway temp file standing in for the laptop's
reify-laptop.yaml (this suite never touches a real laptop or the real
/home/leo/.config/orchestrator/reify-laptop.yaml).

Mirrors scripts/tests/test_flip_reify_gate_exclude_heavy.py's fake-target
pattern, adapted for an ssh-driven remote edit: `_write_fake_ssh_shim`
writes an executable shim that drops the host argument and execs the
remaining command locally (`shift; exec "$@"`), so the script's real
single-quoted-heredoc REMOTE PAYLOAD runs against a temp config file the
test controls, with SSH=<shim> injected via the environment.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).parent.parent / "deploy" / "enable_laptop_persistent_worktree.sh"
KEY = "persistent_merge_worktree"
SAFETY_VALVE_KEY = "persistent_merge_worktree_safety_valve_every_n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixture_laptop_config(state="unflipped"):
    """Build the fixture reify-laptop.yaml text for the given `state`.

    Modeled on defaults.yaml's git: block (main_branch, branch_prefix,
    remote, worktree_dir, persistent_merge_worktree, and the
    prefix-colliding persistent_merge_worktree_safety_valve_every_n
    sibling), which the script must preserve untouched apart from the
    targeted key.

    state:
      "unflipped" - git.persistent_merge_worktree: false present (default;
                    mirrors defaults.yaml's shipped default).
    """
    key_line = {
        "unflipped": f"  {KEY}: false\n",
    }[state]

    return (
        "concurrent_verify: false\n"
        "\n"
        "# Git\n"
        "git:\n"
        '  main_branch: "main"\n'
        '  branch_prefix: "task/"\n'
        '  remote: "origin"\n'
        '  worktree_dir: ".worktrees"\n'
        f"{key_line}"
        f"  {SAFETY_VALVE_KEY}: 5\n"
        "\n"
        "other_top_level_key: true\n"
    )


def _write_fake_ssh_shim(tmp_path):
    """Write an executable fake `ssh` shim into <tmp_path>/bin/ssh.

    Drops the host argument (argv[1]) and execs the remainder locally, so
    `$SSH "$HOST" bash -s -- ...` runs `bash -s -- ...` on THIS machine
    instead of over a real network connection -- the real payload's stdin
    (the heredoc) flows straight through since exec replaces the shim
    process rather than spawning a subshell.

    Returns the path to the shim.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    shim = bin_dir / "ssh"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "shift\n"
        'exec "$@"\n'
    )
    shim.chmod(0o755)
    return shim


def _run_script(config_path, *args, env=None):
    """Run enable_laptop_persistent_worktree.sh against `config_path`.

    Injects SSH=<fake shim>, LAPTOP_HOST=dummy, LAPTOP_CONFIG_PATH=<config_path>,
    REMOTE_PYTHON=<sys.executable>, BACKUP_LABEL=fixed so the real remote
    payload runs locally and deterministically.
    """
    tmp_path = config_path.parent
    shim = _write_fake_ssh_shim(tmp_path)

    full_env = dict(os.environ)
    full_env["SSH"] = str(shim)
    full_env["LAPTOP_HOST"] = "dummy"
    full_env["LAPTOP_CONFIG_PATH"] = str(config_path)
    full_env["REMOTE_PYTHON"] = sys.executable
    full_env["BACKUP_LABEL"] = "fixed"
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _read_config(config_path):
    return config_path.read_text()


def _parsed(config_path):
    return yaml.safe_load(_read_config(config_path))


# ---------------------------------------------------------------------------
# step-1: RED -- default (apply) mode flips the flag and validates readback
# ---------------------------------------------------------------------------

def test_apply_flips_flag_to_true(tmp_path):
    """Default (no-arg) apply mode flips an unflipped git.persistent_merge_worktree
    from false to true, the result parses, and the flag reads back True."""
    config_path = tmp_path / "reify-laptop.yaml"
    config_path.write_text(_fixture_laptop_config("unflipped"))

    result = _run_script(config_path)

    assert result.returncode == 0, (
        f"Expected exit 0 on a fresh apply; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    config_text = _read_config(config_path)
    assert f"{KEY}: true" in config_text, (
        f"Expected {KEY}: true in the edited config; got:\n{config_text}"
    )

    parsed = _parsed(config_path)
    assert parsed.get("git", {}).get(KEY) is True, (
        f"Expected git.{KEY} to read back True via yaml.safe_load; parsed={parsed!r}"
    )
