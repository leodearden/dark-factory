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

Environment
-----------
REIFY_ROOT
    Path to the reify repository checkout.  There is NO hardcoded default: the
    checkout is DISCOVERED by walking up from this file to the nearest ancestor
    carrying ``reify/scripts/verify.sh``, which works for any side-by-side
    checkout on any machine and in either the bare-checkout or worktree layout.
    Set REIFY_ROOT only to point somewhere else:

        export REIFY_ROOT=/path/to/reify

    Precedence, and the full contract, live in ``shared.reify_checkout`` — the
    single source shared with fused-memory's lock-charter drift guard, so one
    ``export REIFY_ROOT=`` steers every reify-dependent test in dark-factory.
    Two consequences worth knowing here:

    - REIFY_ROOT is honored VERBATIM even when it does not exist on disk, so a
      typo skips LOUDLY naming the bad path rather than silently falling back to
      a discovered checkout that would answer for a different repo.
    - The constants below are resolved at IMPORT time, so REIFY_ROOT must be
      exported BEFORE pytest starts to steer them.

    When the integration test is skipped — reify undiscoverable, REIFY_ROOT
    naming a path without the script, or nice/ionice unavailable — a
    ``UserWarning`` carrying the specific reason is emitted in CI runs (``CI``
    env-var set) so the silent no-op is visible in the CI log rather than
    silently providing no cross-repo coverage.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
from shared.reify_checkout import REIFY_ROOT_ENV, reify_skip_reason, resolve_reify_checkout

from orchestrator.config import OrchestratorConfig
from orchestrator.verify import _resolve_verify_env

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#
# Resolved at IMPORT time via shared.reify_checkout — the single source of reify
# discovery, shared with fused-memory's lock-charter drift guard.  Discovery is
# a nearest-ancestor walk from THIS file, so it is correct in both the bare
# checkout and the .worktrees/<id> layout; see that module's docstring for the
# measured evidence table and why no fixed parents[N] index can serve both.
# ---------------------------------------------------------------------------

_REIFY_VERIFY_RELPATH = Path("scripts") / "verify.sh"
# One resolution, carrying both the root and WHERE it came from — the skip
# reason below is formatted from this same result, so it can never blame
# REIFY_ROOT for a path that was actually discovered.
_REIFY_CHECKOUT = resolve_reify_checkout(_REIFY_VERIFY_RELPATH, start=Path(__file__))
REIFY_ROOT: Path | None = _REIFY_CHECKOUT.root
REIFY_VERIFY_SH: Path | None = (
    REIFY_ROOT / _REIFY_VERIFY_RELPATH if REIFY_ROOT is not None else None
)

# The skip decision is kept in TWO parts on purpose.  _REIFY_SKIP_REASON is the
# reify half alone: it is host-independent, so the planted-layout tests at the
# bottom of this file can assert on the gate's ADMISSION DECISION from a module
# copy loaded out of a synthetic tree without also requiring the host to have
# nice/ionice.  Splitting also names WHICH of the three causes fired — the old
# single reason string conflated reify-absent, nice-absent and ionice-absent, so
# an operator reading `pytest -rs` could not tell them apart.
_REIFY_SKIP_REASON: str | None = reify_skip_reason(
    _REIFY_VERIFY_RELPATH, REIFY_ROOT, named_by_env=_REIFY_CHECKOUT.named_by_env
)
_MISSING_TOOLS = [t for t in ("nice", "ionice") if shutil.which(t) is None]
_TOOL_SKIP_REASON: str | None = (
    f"{'/'.join(_MISSING_TOOLS)} unavailable — reify verify.sh degrades its "
    f"CARGO_PRIO without nice/ionice"
    if _MISSING_TOOLS
    else None
)
_SKIP_REASON: str | None = _REIFY_SKIP_REASON or _TOOL_SKIP_REASON

_INTEGRATION_SKIP = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

# Emit a visible warning when the integration gate is silently skipped in CI so
# the coverage gap is not hidden.  Fires at collection time (module import), and
# carries the resolved reason VERBATIM so the CI log names the actual cause (and
# the resolved path, when one was named) rather than a generic three-way string.
if os.environ.get("CI") and _SKIP_REASON is not None:
    import warnings

    warnings.warn(
        f"Integration gate test_role_env_propagates_to_reify_verify_plan will be "
        f"SKIPPED in this CI run: {_SKIP_REASON}. "
        f"(REIFY_ROOT={REIFY_ROOT}, REIFY_VERIFY_SH={REIFY_VERIFY_SH}, "
        f"nice={shutil.which('nice')!r}, ionice={shutil.which('ionice')!r}.) "
        f"Set {REIFY_ROOT_ENV} to a reify checkout path to enable cross-repo "
        f"integration coverage.",
        UserWarning,
        stacklevel=1,
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
    - Take lines AFTER that marker.  If the marker is absent, raise ValueError
      (loud failure — surfaces a cross-repo contract break rather than silently
      scanning the entire stdout and creating false positives from comment substrings).
    - Return those that are non-blank AND do not .lstrip().startswith('#').
    """
    lines = stdout.splitlines()
    marker_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("# --- commands")),
        None,
    )
    if marker_idx is None:
        raise ValueError(
            "verify.sh --print-plan stdout is missing the '# --- commands' marker; "
            "this likely indicates a cross-repo contract break — verify.sh may have "
            "renamed or dropped the separator between the env-comment block and the "
            "command list. stdout excerpt:\n" + stdout[:400]
        )
    return [
        ln for ln in lines[marker_idx + 1 :]
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def _run_reify_print_plan(verify_env: dict[str, str]) -> str:
    """Run reify verify.sh --print-plan test and return stdout.

    Merges verify_env onto os.environ before spawning — mirrors orchestrator's
    _run_cmd merge (verify.py:1131) so the child inherits a working PATH for
    nice/ionice/cargo/git/bash plus the injected DF_VERIFY_ROLE.

    Invokes the script directly; falls back to `bash <script>` if the +x bit
    is absent in this checkout.
    """
    # Every caller is guarded by _INTEGRATION_SKIP, so reaching here with an
    # unresolved checkout means a caller lost its guard.  Fail loudly rather
    # than stringifying None into a bogus subprocess argv.  (Mirrors the
    # identical assert in fused-memory's _reify_guard_vector.)
    assert REIFY_VERIFY_SH is not None and REIFY_ROOT is not None, (
        "no reify checkout resolved, but a test reached _run_reify_print_plan — "
        "its @_INTEGRATION_SKIP guard is missing or was evaluated against a "
        f"different module state (_SKIP_REASON={_SKIP_REASON!r})"
    )
    reify_root = REIFY_ROOT
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
    # Use \bcargo\s (word-boundary + whitespace) so embedded 'cargo' substrings in
    # positional args (e.g. 'cargo-test-occt-gated.sh', '--feature cargo-something')
    # are not mistaken for real cargo invocations.
    total_cargo_count = 0
    for line in cmd_lines:
        for m in re.finditer(r"\bcargo\s", line):
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

    # (1) Pin the exact post-marker output: both the comment-trap exclusion AND the
    #     real-command inclusion are captured in a single assertion.  This is strictly
    #     stronger than a naive_count > helper_count inequality — it catches bugs that
    #     drop one real cargo line while still filtering the comment trap (which the
    #     inequality would silently pass).
    helper_lines = _command_lines(synthetic)
    assert helper_lines == [
        "./scripts/tree-sitter-generate.sh",
        "timeout --kill-after=60 30m nice -n 15 ionice -c 2 -n 7 cargo nextest run --workspace",
    ], f"_command_lines returned unexpected lines: {helper_lines!r}"

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


# ---------------------------------------------------------------------------
# Off-machine gate resolution — the layout-independence demonstration
#
# A test that merely passes on THIS machine does not discriminate: the gate's
# old hardcoded ``/home/leo/src/reify`` default resolves to a path that really
# exists here, so "the integration test ran" was equally consistent with a
# working resolver and with a broken one that happened to name the developer's
# own checkout.  These cases load a COPY of this module out of a SYNTHETIC
# checkout tree and assert on the constants it resolves there, so the answer is
# about the planted layout rather than about this host.
#
# The RED is host-independent in both directions: a fixed literal can never
# equal a tmp_path, and a tmp_path with no reify in its ancestry must resolve
# to None on a machine where /home/leo/src/reify exists.
# ---------------------------------------------------------------------------


@pytest.fixture
def planted_env(monkeypatch):
    """Neutralize the ambient env that would otherwise steer a module copy.

    REIFY_ROOT would override the discovery the planted-layout cases exist to
    exercise; CI would make the copy's import-time warning block fire on the
    deliberately-unresolvable trees.  Returns monkeypatch so a case can set
    REIFY_ROOT back to a value it chose itself.
    """
    monkeypatch.delenv("REIFY_ROOT", raising=False)
    monkeypatch.delenv("CI", raising=False)
    return monkeypatch


def _load_module_copy(tmp_path: Path, tests_relpath: str, *, plant_verify_sh: bool = True):
    """Import a COPY of THIS module from a synthetic checkout layout.

    Plants ``<tmp>/src/reify/scripts/verify.sh`` (a real file; content
    irrelevant) when requested, creates ``<tmp>/src/<tests_relpath>/``, copies
    this module in, and loads it via ``spec_from_file_location``.

    The copy's ``orchestrator.config`` / ``orchestrator.verify`` /
    ``shared.reify_checkout`` imports resolve from the parent process's
    sys.path, while its ``__file__`` is the PLANTED path — so its import-time
    constant resolution walks the SYNTHETIC ancestry, which is the whole point.
    The temporary sys.modules entry is popped in a finally block so no copy
    outlives the call.
    """
    src = tmp_path / "src"
    if plant_verify_sh:
        planted = src / "reify" / "scripts" / "verify.sh"
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted.write_text("#!/bin/sh\necho stub\n")

    tests_dir = src / tests_relpath
    tests_dir.mkdir(parents=True, exist_ok=True)
    copied = tests_dir / "test_copy_probe.py"
    shutil.copy2(__file__, copied)

    module_name = "_reify_gate_probe_" + re.sub(r"\W+", "_", tests_relpath)
    spec = importlib.util.spec_from_file_location(module_name, copied)
    assert spec is not None and spec.loader is not None, f"could not load {copied}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(module_name, None)
    return mod


_BARE_LAYOUT = "dark-factory/orchestrator/tests"
_WORKTREE_LAYOUT = "dark-factory/.worktrees/3978/orchestrator/tests"


def test_gate_resolves_against_the_planted_checkout_not_this_machine(tmp_path, planted_env):
    """The gate must find the reify sibling of the tree it is RUNNING in."""
    mod = _load_module_copy(tmp_path, _BARE_LAYOUT)

    expected = tmp_path / "src" / "reify" / "scripts" / "verify.sh"
    assert expected == mod.REIFY_VERIFY_SH, (
        f"the gate resolved to {mod.REIFY_VERIFY_SH!r} instead of the reify "
        f"checkout planted beside it at {expected!r} — a hardcoded default "
        f"answers for this developer's machine rather than for the tree under test"
    )


def test_gate_admits_the_run_from_a_planted_checkout(tmp_path, planted_env):
    """Resolution is not enough — the skip decision must ADMIT the run.

    Asserts on the reify half alone (``_REIFY_SKIP_REASON``), not the composed
    ``_SKIP_REASON``, so this case stays green on a host without nice/ionice
    instead of smuggling a host dependency into the very test written to
    remove one.
    """
    mod = _load_module_copy(tmp_path, _BARE_LAYOUT)

    assert mod._REIFY_SKIP_REASON is None, (
        f"the gate would SKIP against a planted checkout that really carries "
        f"scripts/verify.sh: {mod._REIFY_SKIP_REASON!r}"
    )


def test_worktree_and_bare_layouts_resolve_to_the_same_planted_checkout(tmp_path, planted_env):
    """No fixed parents[N] index can satisfy both layouts."""
    bare = _load_module_copy(tmp_path, _BARE_LAYOUT)
    worktree = _load_module_copy(tmp_path, _WORKTREE_LAYOUT)

    expected = tmp_path / "src" / "reify" / "scripts" / "verify.sh"
    assert expected == worktree.REIFY_VERIFY_SH
    assert worktree.REIFY_VERIFY_SH == bare.REIFY_VERIFY_SH, (
        f"'.worktrees/<id>' adds exactly two path segments, so a fixed "
        f"parents[N] cannot be correct in both layouts (got "
        f"worktree={worktree.REIFY_VERIFY_SH!r} vs bare={bare.REIFY_VERIFY_SH!r})"
    )
    assert worktree._REIFY_SKIP_REASON is None


def test_discovery_miss_skips_instead_of_answering_from_this_machine(tmp_path, planted_env):
    """THE structural pin against a hardcoded default.

    Nothing named reify exists anywhere in the planted ancestry, so the honest
    answer is None + a skip.  A hardcoded ``/home/leo/src/reify`` produces a
    path that EXISTS on this machine, so the gate would silently run the
    cross-repo integration against a checkout unrelated to the tree under test.
    """
    mod = _load_module_copy(tmp_path, _BARE_LAYOUT, plant_verify_sh=False)

    assert mod.REIFY_VERIFY_SH is None, (
        f"no reify checkout exists in the planted ancestry, but the gate "
        f"resolved {mod.REIFY_VERIFY_SH!r} — an off-tree answer"
    )
    reason = mod._REIFY_SKIP_REASON
    assert isinstance(reason, str) and reason
    assert "REIFY_ROOT" in reason, f"the skip must name the override: {reason!r}"
    assert "scripts/verify.sh" in reason, f"the skip must name the marker: {reason!r}"


def test_env_override_is_honored_verbatim_and_names_the_bad_path(tmp_path, planted_env):
    """A REIFY_ROOT typo must skip loudly, not fall back to a lucky discovery."""
    bad_root = tmp_path / "does-not-exist" / "reify-typo"
    planted_env.setenv("REIFY_ROOT", str(bad_root))

    mod = _load_module_copy(tmp_path, _BARE_LAYOUT)

    assert bad_root.resolve() == mod.REIFY_ROOT, (
        f"REIFY_ROOT must win over the discoverable planted sibling, not be "
        f"shadowed by it (got {mod.REIFY_ROOT!r})"
    )
    assert (tmp_path / "src" / "reify") != mod.REIFY_ROOT, (
        "a typo'd REIFY_ROOT silently falling back to a discovered checkout "
        "would answer for a DIFFERENT repo than the operator named"
    )
    reason = mod._REIFY_SKIP_REASON
    assert isinstance(reason, str) and reason
    assert str(bad_root) in reason, (
        f"the skip reason must name the bad path verbatim so it is self-evident "
        f"in `pytest -rs` output which path was wrong (got {reason!r})"
    )


def test_verify_sh_constant_tracks_the_resolved_root(tmp_path, planted_env):
    """Wiring pin: the two module constants must not re-diverge."""
    mod = _load_module_copy(tmp_path, _BARE_LAYOUT)

    assert mod.REIFY_ROOT is not None
    assert mod.REIFY_VERIFY_SH == mod.REIFY_ROOT / mod._REIFY_VERIFY_RELPATH
