"""Tests for flip-reify-gate-exclude-heavy.sh — drives the script via
subprocess against a throwaway temp git repo standing in for the real
reify checkout (this suite never touches /home/leo/src/reify).

Mirrors scripts/tests/test_restart_orchestrator.py's fake-shim pattern:
`_make_fake_reify_repo` git-inits a temp repo with a fixture
orchestrator.yaml containing a `verify_env:` block (modeled on the real
reify config's comments + existing children), and `_run_script` drives the
script against it via `REIFY_REPO=<repo>`.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "deploy" / "flip-reify-gate-exclude-heavy.sh"
CONFIG_FILE = "orchestrator.yaml"
KNOB = "REIFY_GATE_EXCLUDE_HEAVY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixture_config(state):
    """Build the fixture orchestrator.yaml text for the given `state`.

    Modeled on the real reify orchestrator.yaml's verify_env: block: a
    couple of pre-existing children plus a hand-written operational
    comment, which the script must preserve untouched.
    """
    knob_line = {
        "unflipped": "",
        "flipped": f'  {KNOB}: "1"\n',
        "flipped_zero": f'  {KNOB}: "0"\n',
    }[state]

    return (
        "concurrent_verify: false\n"
        "\n"
        "# Env injected into verify subprocesses (merged on top of os.environ).\n"
        "verify_env:\n"
        f"{knob_line}"
        "  RUSTC_WRAPPER: sccache\n"
        "  # CARGO_INCREMENTAL is required alongside sccache (mutually\n"
        "  # exclusive with incremental compilation).\n"
        '  CARGO_INCREMENTAL: "0"\n'
        "\n"
        "other_top_level_key: true\n"
    )


def _make_fake_reify_repo(tmp_path, *, state="unflipped", failing_precommit=False):
    """git-init a temp repo standing in for the reify checkout, with a
    fixture orchestrator.yaml committed as the initial commit.

    state:
      "unflipped"    - no REIFY_GATE_EXCLUDE_HEAVY key present (default).
      "flipped"      - REIFY_GATE_EXCLUDE_HEAVY: "1" already present.
      "flipped_zero" - REIFY_GATE_EXCLUDE_HEAVY: "0" already present (a
                        stray/stale value apply must normalize to "1").

    failing_precommit=True installs an executable .git/hooks/pre-commit
    that always exits 1 -- AFTER the initial commit, so only the script's
    own flip commit is subject to it (mirrors the real reify repo's
    main-branch commit gate).

    Returns the repo path.
    """
    repo = tmp_path / "reify"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

    (repo / CONFIG_FILE).write_text(_fixture_config(state))

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)

    if failing_precommit:
        hook = repo / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/usr/bin/env bash\nexit 1\n")
        hook.chmod(0o755)

    return repo


def _run_script(repo, *args, env=None):
    """Run flip-reify-gate-exclude-heavy.sh with REIFY_REPO=<repo>."""
    full_env = dict(os.environ)
    full_env["REIFY_REPO"] = str(repo)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _commit_count(repo):
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--count", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def _read_config(repo):
    return (repo / CONFIG_FILE).read_text()


def _head_files(repo):
    result = subprocess.run(
        ["git", "-C", str(repo), "show", "--name-only", "--pretty=format:", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.split()


# ---------------------------------------------------------------------------
# step-1: RED -- default (apply) mode inserts the knob and commits
# ---------------------------------------------------------------------------

def test_apply_inserts_knob_and_commits(tmp_path):
    """Default (no-arg) apply mode inserts the knob as the first child of
    verify_env:, preserves the surrounding comments/children, commits the
    change in the reify repo, and exits 0."""
    repo = _make_fake_reify_repo(tmp_path)
    before_commits = _commit_count(repo)

    result = _run_script(repo)

    assert result.returncode == 0, (
        f"Expected exit 0 on a fresh apply; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    lines = _read_config(repo).splitlines()
    verify_env_idx = lines.index("verify_env:")
    assert lines[verify_env_idx + 1] == f'  {KNOB}: "1"', (
        f"Expected the knob as the first child right after verify_env:; "
        f"got: {lines[verify_env_idx:verify_env_idx + 3]!r}"
    )

    config_text = _read_config(repo)
    assert "RUSTC_WRAPPER: sccache" in config_text, "pre-existing child dropped"
    assert "CARGO_INCREMENTAL is required alongside sccache" in config_text, (
        "pre-existing comment dropped"
    )
    assert 'CARGO_INCREMENTAL: "0"' in config_text, "pre-existing child dropped"
    assert "other_top_level_key: true" in config_text, "unrelated top-level key dropped"

    after_commits = _commit_count(repo)
    assert after_commits == before_commits + 1, (
        f"Expected exactly one new commit; before={before_commits} after={after_commits}"
    )
    assert CONFIG_FILE in _head_files(repo), (
        f"Expected HEAD to name {CONFIG_FILE}; got {_head_files(repo)!r}"
    )


# ---------------------------------------------------------------------------
# step-3: RED -- idempotency + single-key normalization
# ---------------------------------------------------------------------------

def test_apply_twice_is_idempotent_no_second_commit(tmp_path):
    """Running apply twice on the same repo leaves exactly one knob
    occurrence and creates no second commit."""
    repo = _make_fake_reify_repo(tmp_path)

    first = _run_script(repo)
    assert first.returncode == 0, (
        f"Expected exit 0 on first apply; got {first.returncode}\n"
        f"stdout={first.stdout!r} stderr={first.stderr!r}"
    )
    commits_after_first = _commit_count(repo)

    second = _run_script(repo)
    assert second.returncode == 0, (
        f"Expected exit 0 on second (already-flipped, no-op) apply; got "
        f"{second.returncode}\nstdout={second.stdout!r} stderr={second.stderr!r}"
    )

    config_text = _read_config(repo)
    assert config_text.count(KNOB) == 1, (
        f"Expected exactly one occurrence of {KNOB}; got {config_text.count(KNOB)}\n"
        f"config:\n{config_text}"
    )
    commits_after_second = _commit_count(repo)
    assert commits_after_second == commits_after_first, (
        f"Expected no new commit on the second (already-flipped) run; "
        f"before={commits_after_first} after={commits_after_second}"
    )


def test_apply_normalizes_stray_zero_value_to_single_key(tmp_path):
    """A pre-existing REIFY_GATE_EXCLUDE_HEAVY: "0" is normalized to "1"
    with no duplicate key."""
    repo = _make_fake_reify_repo(tmp_path, state="flipped_zero")

    result = _run_script(repo)

    assert result.returncode == 0, (
        f"Expected exit 0; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    config_text = _read_config(repo)
    assert config_text.count(KNOB) == 1, (
        f"Expected exactly one occurrence of {KNOB} after normalization; "
        f"got {config_text.count(KNOB)}\nconfig:\n{config_text}"
    )
    assert f'{KNOB}: "1"' in config_text, (
        f"Expected the stray \"0\" value normalized to \"1\"; config:\n{config_text}"
    )
    assert f'{KNOB}: "0"' not in config_text


# ---------------------------------------------------------------------------
# step-5: RED -- --check / --dry-run mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--check", "--dry-run"])
def test_check_mode_prints_diff_without_mutating(tmp_path, flag):
    """--check (and its --dry-run alias) on an unflipped config prints the
    intended one-line diff, exits 0, and leaves the repo untouched."""
    repo = _make_fake_reify_repo(tmp_path)
    before_config = _read_config(repo)
    before_commits = _commit_count(repo)

    result = _run_script(repo, flag)

    assert result.returncode == 0, (
        f"Expected exit 0 for {flag}; got {result.returncode}\n"
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "+" in result.stdout and KNOB in result.stdout and '"1"' in result.stdout, (
        f"Expected a '+'-prefixed line naming {KNOB} with value \"1\" in stdout; "
        f"got: {result.stdout!r}"
    )
    assert "verify_env" in result.stdout, (
        f"Expected the intended diff to reference verify_env; got: {result.stdout!r}"
    )

    assert _read_config(repo) == before_config, f"{flag} must not mutate orchestrator.yaml"
    assert _commit_count(repo) == before_commits, f"{flag} must not create a commit"


# ---------------------------------------------------------------------------
# step-7: RED -- config-reload seam fires only on a real flip
# ---------------------------------------------------------------------------

RELOAD_MARKER = "flip-reify-gate: config-reload signal emitted"


def test_reload_marker_emitted_only_on_real_apply(tmp_path):
    """The reload signal fires on a genuine flip, not on --check, and not
    on an already-flipped no-op apply."""
    repo = _make_fake_reify_repo(tmp_path)

    applied = _run_script(repo)
    assert applied.returncode == 0, (
        f"Expected exit 0 on real apply; got {applied.returncode}\n"
        f"stdout={applied.stdout!r} stderr={applied.stderr!r}"
    )
    assert RELOAD_MARKER in applied.stdout, (
        f"Expected the reload marker on a real apply; got: {applied.stdout!r}"
    )

    check_base = tmp_path / "check"
    check_base.mkdir()
    check_repo = _make_fake_reify_repo(check_base)
    checked = _run_script(check_repo, "--check")
    assert checked.returncode == 0
    assert RELOAD_MARKER not in checked.stdout, (
        f"--check must not emit the reload marker; got: {checked.stdout!r}"
    )

    noop = _run_script(repo)
    assert noop.returncode == 0, (
        f"Expected exit 0 on already-flipped no-op apply; got {noop.returncode}\n"
        f"stdout={noop.stdout!r} stderr={noop.stderr!r}"
    )
    assert RELOAD_MARKER not in noop.stdout, (
        f"An already-flipped no-op apply must not emit the reload marker; "
        f"got: {noop.stdout!r}"
    )
