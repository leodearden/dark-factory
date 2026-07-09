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
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).parent.parent / "deploy" / "enable_laptop_persistent_worktree.sh"
KEY = "persistent_merge_worktree"
SAFETY_VALVE_KEY = "persistent_merge_worktree_safety_valve_every_n"
NOOP_MARKER = "already enabled (no-op)"


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
      "unflipped"     - git.persistent_merge_worktree: false present (default;
                        mirrors defaults.yaml's shipped default).
      "flipped"       - git.persistent_merge_worktree: true already present.
      "malformed"     - deliberately-malformed YAML (unterminated flow
                        sequence under git:); must not parse.
      "no_key"        - git: block present with its usual siblings, but NO
                        persistent_merge_worktree key at all (insert-as-
                        first-child case).
      "no_git_anchor" - neither the key nor any top-level git: block is
                        present at all (refuse-blind-edit case).
    """
    if state == "malformed":
        return (
            "concurrent_verify: false\n"
            "\n"
            "# Git\n"
            "git:\n"
            '  main_branch: "main"\n'
            f"  {KEY}: false\n"
            "  bad_value: [1, 2\n"
            "\n"
            "other_top_level_key: true\n"
        )

    if state == "no_git_anchor":
        return (
            "concurrent_verify: false\n"
            "\n"
            "other_top_level_key: true\n"
        )

    key_line = {
        "unflipped": f"  {KEY}: false\n",
        "flipped": f"  {KEY}: true\n",
        "no_key": "",
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


def _count_anchored_key(config_text, key):
    """Count lines where `key` appears as the anchored YAML key -- mirrors
    the script's `^[[:space:]]*key[[:space:]]*:` anchoring so a naive
    substring count doesn't also match the prefix-colliding
    persistent_merge_worktree_safety_valve_every_n sibling."""
    pattern = re.compile(rf"^[ \t]*{re.escape(key)}[ \t]*:", re.MULTILINE)
    return len(pattern.findall(config_text))


def _backup_files(config_path):
    return sorted(p.name for p in config_path.parent.glob(f"{config_path.name}.bak-*"))


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


# ---------------------------------------------------------------------------
# step-3: RED -- idempotency: a second apply is a clean no-op
# ---------------------------------------------------------------------------

def test_apply_twice_is_idempotent_noop_second_run(tmp_path):
    """Running apply twice against the same config leaves exactly one
    occurrence of persistent_merge_worktree (set to true), both runs exit
    0, and the second run is a clean no-op: it emits an already-enabled
    marker and creates no NEW backup file."""
    config_path = tmp_path / "reify-laptop.yaml"
    config_path.write_text(_fixture_laptop_config("unflipped"))

    first = _run_script(config_path)
    assert first.returncode == 0, (
        f"Expected exit 0 on first apply; got {first.returncode}\n"
        f"stdout={first.stdout!r} stderr={first.stderr!r}"
    )

    backups_before_second = _backup_files(config_path)

    second = _run_script(config_path)
    assert second.returncode == 0, (
        f"Expected exit 0 on second (already-enabled) apply; got {second.returncode}\n"
        f"stdout={second.stdout!r} stderr={second.stderr!r}"
    )
    assert NOOP_MARKER in second.stdout, (
        f"Expected an already-enabled/no-op marker on the second run; "
        f"got: {second.stdout!r}"
    )

    backups_after_second = _backup_files(config_path)
    assert backups_after_second == backups_before_second, (
        f"Expected no NEW backup file from the no-op second run; "
        f"before={backups_before_second} after={backups_after_second}"
    )

    config_text = _read_config(config_path)
    assert _count_anchored_key(config_text, KEY) == 1, (
        f"Expected exactly one occurrence of {KEY}; got "
        f"{_count_anchored_key(config_text, KEY)}\nconfig:\n{config_text}"
    )
    parsed = yaml.safe_load(config_text)
    assert parsed.get("git", {}).get(KEY) is True, (
        f"Expected git.{KEY} to still read back True; parsed={parsed!r}"
    )


# ---------------------------------------------------------------------------
# step-5: RED -- --check / --dry-run mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--check", "--dry-run"])
def test_check_mode_prints_intended_diff_without_mutating(tmp_path, flag):
    """--check (and its --dry-run alias) on an unflipped config prints the
    intended change (naming persistent_merge_worktree + true, referencing
    git:), exits 0, and leaves the config byte-identical with no backup."""
    config_path = tmp_path / "reify-laptop.yaml"
    before_text = _fixture_laptop_config("unflipped")
    config_path.write_text(before_text)

    result = _run_script(config_path, flag)

    assert result.returncode == 0, (
        f"Expected exit 0 for {flag}; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert KEY in result.stdout and "true" in result.stdout, (
        f"Expected a line naming {KEY} + true in stdout; got: {result.stdout!r}"
    )
    assert "git" in result.stdout, (
        f"Expected the intended diff to reference git:; got: {result.stdout!r}"
    )

    assert _read_config(config_path) == before_text, (
        f"{flag} must not mutate the config"
    )
    assert _backup_files(config_path) == [], f"{flag} must not create a backup"


def test_check_mode_on_already_enabled_reports_noop(tmp_path):
    """--check on an ALREADY-enabled config reports the same
    already-enabled/no-op state as apply, not a pending-diff line."""
    config_path = tmp_path / "reify-laptop.yaml"
    before_text = _fixture_laptop_config("flipped")
    config_path.write_text(before_text)

    result = _run_script(config_path, "--check")

    assert result.returncode == 0, (
        f"Expected exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert NOOP_MARKER in result.stdout, (
        f"Expected --check on an already-enabled config to report the "
        f"no-op state; got: {result.stdout!r}"
    )
    assert _read_config(config_path) == before_text, "--check must never mutate the config"
    assert _backup_files(config_path) == [], "--check must never create a backup"


# ---------------------------------------------------------------------------
# step-7: RED -- a malformed target YAML must fail loudly in both modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args", [(), ("--check",)], ids=["apply", "--check"])
def test_malformed_yaml_exits_nonzero_without_mutating(tmp_path, args):
    """A deliberately-malformed target YAML makes the script exit non-zero
    in both apply and --check modes, mutating nothing and creating no
    backup."""
    config_path = tmp_path / "reify-laptop.yaml"
    before_text = _fixture_laptop_config("malformed")
    config_path.write_text(before_text)

    result = _run_script(config_path, *args)

    assert result.returncode != 0, (
        f"Expected non-zero exit on malformed YAML (args={args!r}); got 0\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stderr.strip(), (
        f"Expected an error message on stderr; got stderr={result.stderr!r}"
    )
    assert _read_config(config_path) == before_text, (
        f"malformed YAML must not be mutated (args={args!r})"
    )
    assert _backup_files(config_path) == [], (
        f"malformed YAML must not produce a backup (args={args!r})"
    )


# ---------------------------------------------------------------------------
# step-9: RED -- a real apply creates a pre-edit backup; a no-op does not
# ---------------------------------------------------------------------------

def test_apply_creates_backup_with_preedit_contents(tmp_path):
    """A real apply (false -> true) creates a <config>.bak-<label> file
    whose contents equal the pre-edit config; a subsequent no-op apply
    creates no additional backup."""
    config_path = tmp_path / "reify-laptop.yaml"
    before_text = _fixture_laptop_config("unflipped")
    config_path.write_text(before_text)

    first = _run_script(config_path)
    assert first.returncode == 0, (
        f"Expected exit 0 on first apply; got {first.returncode}\n"
        f"stdout={first.stdout!r} stderr={first.stderr!r}"
    )

    backup_path = config_path.parent / f"{config_path.name}.bak-fixed"
    assert backup_path.exists(), (
        f"Expected backup {backup_path} to exist after the first apply; "
        f"found backups={_backup_files(config_path)!r}"
    )
    assert backup_path.read_text() == before_text, (
        "Expected the backup to hold the PRE-edit config bytes"
    )

    backups_after_first = _backup_files(config_path)

    second = _run_script(config_path)
    assert second.returncode == 0, (
        f"Expected exit 0 on second (already-enabled) apply; got {second.returncode}\n"
        f"stdout={second.stdout!r} stderr={second.stderr!r}"
    )

    backups_after_second = _backup_files(config_path)
    assert backups_after_second == backups_after_first, (
        f"Expected no additional backup from the no-op second run; "
        f"before={backups_after_first} after={backups_after_second}"
    )


# ---------------------------------------------------------------------------
# step-11: RED -- structural cases: insert-when-key-absent, and refuse a
# blind edit when there is no git: anchor to insert under.
# ---------------------------------------------------------------------------

def test_apply_inserts_key_as_first_child_when_git_block_has_no_key(tmp_path):
    """apply on a config whose git: block exists but has no
    persistent_merge_worktree key inserts `persistent_merge_worktree: true`
    as the first child of git: (immediately after the git: line), exits 0,
    reads back True, and leaves the other git: children untouched."""
    config_path = tmp_path / "reify-laptop.yaml"
    config_path.write_text(_fixture_laptop_config("no_key"))

    result = _run_script(config_path)

    assert result.returncode == 0, (
        f"Expected exit 0 inserting into a keyless git: block; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    config_text = _read_config(config_path)
    lines = config_text.splitlines()
    git_index = lines.index("git:")
    assert lines[git_index + 1].strip() == f"{KEY}: true", (
        f"Expected {KEY}: true as the first child immediately after git:; got:\n{config_text}"
    )

    for other in (
        'main_branch: "main"',
        'branch_prefix: "task/"',
        'remote: "origin"',
        'worktree_dir: ".worktrees"',
        f"{SAFETY_VALVE_KEY}: 5",
    ):
        assert other in config_text, (
            f"Expected untouched sibling {other!r} to survive the insert; got:\n{config_text}"
        )

    parsed = _parsed(config_path)
    assert parsed.get("git", {}).get(KEY) is True, (
        f"Expected git.{KEY} to read back True; parsed={parsed!r}"
    )


def test_apply_refuses_blind_edit_when_no_git_anchor(tmp_path):
    """apply on a config with NEITHER the key NOR any top-level git: anchor
    fails loudly (non-zero exit, stderr error) instead of silently
    succeeding, and makes no mutation and creates no backup."""
    config_path = tmp_path / "reify-laptop.yaml"
    before_text = _fixture_laptop_config("no_git_anchor")
    config_path.write_text(before_text)

    result = _run_script(config_path)

    assert result.returncode != 0, (
        f"Expected non-zero exit when neither the key nor a git: anchor exists; got 0\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert result.stderr.strip(), (
        f"Expected an error message on stderr; got stderr={result.stderr!r}"
    )
    assert _read_config(config_path) == before_text, (
        "Refused blind edit must not mutate the config"
    )
    assert _backup_files(config_path) == [], (
        "Refused blind edit must not create a backup"
    )


# ---------------------------------------------------------------------------
# step-13: RED -- safety-valve sibling isolation (prefix-collision gotcha)
# ---------------------------------------------------------------------------

def test_apply_does_not_touch_safety_valve_sibling(tmp_path):
    """apply on a config containing both persistent_merge_worktree: false
    and the prefix-colliding persistent_merge_worktree_safety_valve_every_n:
    5 flips only the boolean to true, leaves the safety-valve key's value
    (5) byte-for-byte untouched, and results in exactly one occurrence of
    each key. Pins the prefix-collision gotcha: an unanchored regex would
    also match/strip/duplicate the safety-valve sibling."""
    config_path = tmp_path / "reify-laptop.yaml"
    config_path.write_text(_fixture_laptop_config("unflipped"))

    result = _run_script(config_path)

    assert result.returncode == 0, (
        f"Expected exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    config_text = _read_config(config_path)

    assert _count_anchored_key(config_text, KEY) == 1, (
        f"Expected exactly one occurrence of {KEY}; got "
        f"{_count_anchored_key(config_text, KEY)}\nconfig:\n{config_text}"
    )
    assert _count_anchored_key(config_text, SAFETY_VALVE_KEY) == 1, (
        f"Expected exactly one occurrence of {SAFETY_VALVE_KEY}; got "
        f"{_count_anchored_key(config_text, SAFETY_VALVE_KEY)}\nconfig:\n{config_text}"
    )
    assert f"{SAFETY_VALVE_KEY}: 5" in config_text, (
        f"Expected the safety-valve sibling's value to remain untouched (5); got:\n{config_text}"
    )

    parsed = _parsed(config_path)
    assert parsed.get("git", {}).get(KEY) is True, (
        f"Expected git.{KEY} to read back True; parsed={parsed!r}"
    )
    assert parsed.get("git", {}).get(SAFETY_VALVE_KEY) == 5, (
        f"Expected git.{SAFETY_VALVE_KEY} to remain 5, untouched; parsed={parsed!r}"
    )
