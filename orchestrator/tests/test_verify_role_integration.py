"""Integration test: DF_VERIFY_ROLE propagates end-to-end to reify verify-plan.

PRD δ leaf-signal: test_role_env_propagates_to_reify_verify_plan.

Tests the full cross-repo seam:
  - orchestrator producer: _resolve_verify_env(role=...) stamps DF_VERIFY_ROLE (β/γ)
  - reify consumer: verify.sh --print-plan emits CARGO_PRIO prefix on every cargo
    command that matches the role (α)

Skipped when the reify checkout is absent OR nice/ionice are unavailable: reify
verify.sh degrades its CARGO_PRIO to a reduced/empty prefix in those environments,
so the strict full-prefix assertions cannot hold.  The conventional
environment-gated integration-test idiom (cf. test_landlock.py:106,
test_reviewer_trial_corpus.py:129).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.verify import _resolve_verify_env

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REIFY_VERIFY_SH: Path = (
    Path(os.environ.get("REIFY_ROOT", "/home/leo/src/reify")) / "scripts" / "verify.sh"
)

# ---------------------------------------------------------------------------
# Module-level skip guard
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
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


@pytest.mark.parametrize(
    "role, expected_prefix, forbid_ionice",
    [
        ("merge", "nice -n 5 ", True),
        ("task", "nice -n 15 ionice -c 2 -n 7 ", False),
    ],
    ids=["merge", "task"],
)
def test_role_env_propagates_to_reify_verify_plan(
    role: str,
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

    # Find all real cargo command positions: `cargo ` (trailing space) to exclude
    # cargo-test-occt-gated.sh (cargo+hyphen), .cargo/env (cargo+slash), and
    # uppercase CARGO_* env-comment lines.
    cargo_positions = [m.start() for m in re.finditer(r"cargo ", stdout)]

    # Anti-vacuous guard: the plan must contain at least one cargo invocation.
    assert len(cargo_positions) >= 1, (
        f"Expected at least one 'cargo ' command in verify.sh --print-plan output"
        f" for role={role!r}, but found none.\nstdout:\n{stdout}"
    )

    # Every real cargo command must be immediately preceded by the role's prefix.
    for pos in cargo_positions:
        prefix_start = pos - len(expected_prefix)
        actual_prefix = stdout[prefix_start:pos] if prefix_start >= 0 else ""
        assert actual_prefix == expected_prefix, (
            f"cargo command at stdout[{pos}] has prefix {actual_prefix!r},"
            f" expected {expected_prefix!r} for role={role!r}.\nstdout:\n{stdout}"
        )

    # For merge role: ionice must not appear anywhere in the plan output.
    if forbid_ionice:
        assert "ionice" not in stdout, (
            f"ionice must not appear in merge-role plan, but found it.\nstdout:\n{stdout}"
        )
