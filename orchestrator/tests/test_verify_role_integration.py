"""Integration test: DF_VERIFY_ROLE propagates end-to-end to reify verify-plan.

PRD δ leaf-signal: test_role_env_propagates_to_reify_verify_plan.

Tests the full cross-repo seam:
  - orchestrator producer: _resolve_verify_env(role=...) stamps DF_VERIFY_ROLE (β/γ)
  - reify consumer: verify.sh --print-plan emits CARGO_PRIO prefix on every cargo
    command that matches the role (α)

The integration test is skipped when the reify checkout is absent OR nice/ionice
are unavailable (function-level skipif).  The unit test
test_command_lines_excludes_env_comment_substrings runs unconditionally — it
exercises _command_lines() with a synthetic fixture and requires no subprocess.

Conventional environment-gated integration-test idiom (cf. test_landlock.py:106,
test_reviewer_trial_corpus.py:129).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.verify import _resolve_verify_env

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REIFY_VERIFY_SH: Path = (
    Path(os.environ.get("REIFY_ROOT", "/home/leo/src/reify")) / "scripts" / "verify.sh"
)

_INTEGRATION_SKIP = pytest.mark.skipif(
    not REIFY_VERIFY_SH.exists()
    or shutil.which("nice") is None
    or shutil.which("ionice") is None,
    reason=(
        "reify checkout or nice/ionice unavailable — cross-repo integration gate"
        " cannot run; reify verify.sh degrades its CARGO_PRIO without nice/ionice"
    ),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _command_lines(stdout: str) -> list[str]:
    """Return only the real command lines from verify.sh --print-plan stdout.

    verify.sh emits a '# --- commands' marker line that separates the process-level
    env-comment block from the actual command list.  Lines in the env block are
    '# ...' comments; some contain lowercase 'cargo ' substrings (e.g.
    '# CARGO_MAKEFLAGS left unset ... cargo uses its own job pool') that must not
    be scanned as cargo invocations.

    Algorithm:
    - Split stdout into lines via .splitlines().
    - Find the first line that starts with '# --- commands'.
    - Take lines AFTER that marker.  If the marker is absent fall back to ALL lines
      (defensive — keeps the function useful even against future verify.sh layout
      changes).
    - Return those that are non-blank AND do not .lstrip().startswith('#').
    """
    lines = stdout.splitlines()
    marker_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("# --- commands")),
        None,
    )
    candidates = lines[marker_idx + 1 :] if marker_idx is not None else lines
    return [ln for ln in candidates if ln.strip() and not ln.lstrip().startswith("#")]


def _run_reify_print_plan(verify_env: dict[str, str]) -> str:
    """Run reify verify.sh --print-plan test and return stdout.

    Merges verify_env onto os.environ before spawning — mirrors orchestrator's
    _run_cmd merge (verify.py:1131) so the child inherits a working PATH for
    nice/ionice/cargo/git/bash plus the injected DF_VERIFY_ROLE.

    Invokes the script directly; falls back to `bash <script>` if the +x bit
    is absent in this checkout.
    """
    reify_root = REIFY_VERIFY_SH.parents[1]
    child_env = {**os.environ, **verify_env}

    try:
        result = subprocess.run(
            [str(REIFY_VERIFY_SH), "--print-plan", "test"],
            env=child_env,
            cwd=str(reify_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except PermissionError:
        # Script lacks +x bit in this checkout — fall back to bash invocation.
        result = subprocess.run(
            ["bash", str(REIFY_VERIFY_SH), "--print-plan", "test"],
            env=child_env,
            cwd=str(reify_root),
            capture_output=True,
            text=True,
            timeout=60,
        )

    assert result.returncode == 0, (
        f"verify.sh --print-plan exited {result.returncode}; stderr:\n{result.stderr}"
    )
    return result.stdout


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_INTEGRATION_SKIP
@pytest.mark.parametrize(
    "role, expected_prefix, forbid_ionice",
    [
        ("merge", "nice -n 5 ", True),
        ("task", "nice -n 15 ionice -c 2 -n 7 ", False),
    ],
    ids=["merge", "task"],
)
def test_role_env_propagates_to_reify_verify_plan(
    role: Literal["merge", "task"],
    expected_prefix: str,
    forbid_ionice: bool,
) -> None:
    """DF_VERIFY_ROLE flows from orchestrator producer into reify consumer.

    Step 1 (producer side — β/γ): _resolve_verify_env stamps DF_VERIFY_ROLE=role.
    Step 2 (consumer side — α): verify.sh --print-plan emits the role's nice prefix
    immediately before every real cargo command in its output plan.
    """
    # --- producer side (β/γ): orchestrator stamps the role into verify_env ---
    verify_env = _resolve_verify_env(OrchestratorConfig(verify_env={}), None, role=role)
    assert verify_env["DF_VERIFY_ROLE"] == role, (
        f"Expected DF_VERIFY_ROLE={role!r} in verify_env, got {verify_env!r}"
    )

    # --- consumer side (α): reify emits the correct CARGO_PRIO prefix ---
    stdout = _run_reify_print_plan(verify_env)
    cmd_lines = _command_lines(stdout)

    # Scan only real command lines (post-marker, non-comment) to avoid false
    # positives from env-comment substrings like 'cargo uses its own job pool'.
    total_cargo_count = 0
    for line in cmd_lines:
        for m in re.finditer(r"cargo ", line):
            pos = m.start()
            prefix_start = pos - len(expected_prefix)
            actual_prefix = line[prefix_start:pos] if prefix_start >= 0 else ""
            assert actual_prefix == expected_prefix, (
                f"cargo command in line {line!r} has prefix {actual_prefix!r},"
                f" expected {expected_prefix!r} for role={role!r}.\nstdout:\n{stdout}"
            )
            total_cargo_count += 1

    # Anti-vacuous guard: the plan must contain at least one cargo invocation.
    assert total_cargo_count >= 1, (
        f"Expected at least one 'cargo ' command in verify.sh --print-plan output"
        f" for role={role!r}, but found none.\nstdout:\n{stdout}"
    )

    # For merge role: ionice must not appear in any real command line.
    if forbid_ionice:
        assert "ionice" not in "\n".join(cmd_lines), (
            f"ionice must not appear in merge-role command lines, but found it."
            f"\nstdout:\n{stdout}"
        )


# ---------------------------------------------------------------------------
# Pure-unit test — host-independent regression guard (no subprocess, no skip)
# ---------------------------------------------------------------------------

# Synthetic --print-plan stdout reproducing the FIFO-ABSENT layout:
#   - env-comment block BEFORE the '# --- commands' marker, containing the trap
#     line whose lowercase 'cargo ' (in 'cargo uses') must NOT be scanned as a
#     real cargo command.
#   - two real command lines AFTER the marker.
_SYNTHETIC_PRINT_PLAN_STDOUT = """\
# DF_VERIFY_ROLE=task
# CARGO_PRIO=nice -n 15 ionice -c 2 -n 7
# CARGO_MAKEFLAGS left unset (no /tmp/reify-jobserver FIFO) — cargo uses its own job pool
# --- commands
./scripts/tree-sitter-generate.sh
timeout --kill-after=60 30m nice -n 15 ionice -c 2 -n 7 cargo nextest run --workspace
"""


def test_command_lines_excludes_env_comment_substrings() -> None:
    """_command_lines() must filter comment lines BEFORE the '# --- commands' marker.

    The FIFO-absent env-comment block contains a lowercase 'cargo ' substring
    ('cargo uses its own job pool') that would trigger a spurious nice-prefix
    assertion failure if the entire stdout were scanned naively.  This test
    verifies that _command_lines() returns ONLY the real post-marker command lines,
    dropping the comment trap.

    Runs unconditionally — no subprocess, no reify checkout, no nice/ionice required.
    """
    synthetic = _SYNTHETIC_PRINT_PLAN_STDOUT

    # (1) Naive whole-stdout scan finds MORE 'cargo ' matches than the helper-scoped
    #     scan: the comment-trap line contributes one extra match.
    naive_count = len(re.findall(r"cargo ", synthetic))
    helper_lines = _command_lines(synthetic)
    helper_count = len(re.findall(r"cargo ", "\n".join(helper_lines)))
    assert naive_count > helper_count, (
        f"Expected naive_count ({naive_count}) > helper_count ({helper_count});"
        f" comment-trap 'cargo ' should be filtered out but was not."
    )

    # (2) Every returned line is a real command: non-blank and not a comment.
    for line in helper_lines:
        assert line.strip(), f"_command_lines returned a blank line: {line!r}"
        assert not line.lstrip().startswith("#"), (
            f"_command_lines returned a comment line: {line!r}"
        )

    # (3) The trap substring must not appear in any returned line.
    for line in helper_lines:
        assert "cargo uses its own job pool" not in line, (
            f"Trap substring found in _command_lines result: {line!r}"
        )

    # (4) Both real command lines ARE returned.
    assert any("tree-sitter-generate.sh" in ln for ln in helper_lines), (
        "_command_lines did not return the tree-sitter-generate.sh line"
    )
    assert any("cargo nextest run" in ln for ln in helper_lines), (
        "_command_lines did not return the cargo-nextest line"
    )
