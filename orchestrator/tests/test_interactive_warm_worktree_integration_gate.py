"""ζ B+H integration gate: interactive warm-worktree boundary test (task 2015).

Ropes task α (``GitOps.create_interactive_worktree`` primitive, 2010), task β
(``claim_warm_worktree`` / ``release_warm_worktree`` escalation MCP verbs,
2011), and task δ (the harness-side interactive-worktree reaper cadence,
2012 — merged into main) into ONE end-to-end boundary test that proves the
interactive warm-worktree contract holds across the full B+H (escalation
verb + harness) surface — a composition none of the three tasks' own tests
exercises:

  - α's test_interactive_worktree.py::TestCreateInteractiveWorktreeIsolation
    already proves isolation invariant I1 at the raw git-primitive level
    (direct ``create_interactive_worktree`` + ``git worktree remove``).
  - β's escalation/tests/test_warm_worktree_verbs.py already covers
    claim/release verbs — but against a ``SimpleNamespace(git_ops=...)``
    fake harness with NO real ``WarmLanePool``.
  - δ's test_harness_interactive_reaper.py covers the reaper pass on a bare
    ``Harness`` with no pool involved.

This module drives claim/release THROUGH the β escalation verbs
(``create_server(...)`` + the FastMCP ``_call_tool`` unit-invocation
pattern) against a harness that ALSO carries a real ``WarmLanePool`` (K FREE
lanes) AND a seeded ``scheduler._dispatched`` set, proving I1 holds across
the entire verb surface (the "B+H integration gate"), then composes the δ
reaper (I2) into the same flow and re-asserts the pool is still untouched.

Like the sibling gates (test_warm_lane_integration_gate.py,
test_config_reload_integration_gate.py) this gate is EXPECTED GREEN on the
existing α/β/δ production code: the new artifact here is the test itself.
Impl steps touch production (git_ops.py / harness.py / escalation
server.py) ONLY if end-to-end composition surfaces a genuine defect.

G6 / PRD Open-Q#3 scope note: the "warm target/ → near-zero recompile"
observable is a reify-side (cargo ``target/``, filefrag/CoW extent-sharing)
guarantee that is NOT producible in dark-factory's Python CI (no cargo
build cache). That observable is EXPLICITLY OUT of scope in this module and
is deferred to the reify deploy capstone's out-of-band verification. This
module asserts only the CI-producible orchestration-observable boundary
invariants: I1 isolation across the verb surface, the claim/release
roundtrip, seed-invocation via ``target/seeded.bin`` existence, fail-soft
cold claims, reap I2 composed with I1, and the cold (no-harness) fallback.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from escalation.queue import EscalationQueue
from escalation.server import create_server

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import _run
from orchestrator.harness import Harness
from orchestrator.warm_lane_pool import LaneState, WarmLanePool

# ---------------------------------------------------------------------------
# Repo fixture helpers — copied/adapted per codebase convention (NOT
# cross-imported from sibling test files; see test_interactive_worktree.py,
# escalation/tests/test_warm_worktree_verbs.py,
# test_harness_interactive_reaper.py, test_warm_lane_integration_gate.py).
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    """git init -b main + one initial commit."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _add_seed_script(repo: Path, *, exit_code: int = 0) -> None:
    """Commit a seed-warm-lane.sh stub into repo/scripts/.

    On exit_code == 0: creates ``<lane>/target/seeded.bin`` (orchestration-
    observable seededness) AND records the full received argv, one
    positional arg per line, to ``<lane>/target/seed-argv.txt``. Recording
    argv pins the actual invocation contract (``<base_target> <lane_dir>
    <mode>`` — see ``GitOps._seed_warm_lane``) so a test can assert on the
    mode flag the script received, not merely on the fact that *some* seed
    ran and populated target/. On non-zero: exits with that code immediately
    (no target/ created) — models a faulting seed for the fail-soft case.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed = scripts_dir / 'seed-warm-lane.sh'
    if exit_code == 0:
        seed.write_text(
            '#!/usr/bin/env bash\n'
            'mkdir -p "$2/target"\n'
            'echo "seeded" > "$2/target/seeded.bin"\n'
            'printf "%s\\n" "$@" > "$2/target/seed-argv.txt"\n'
        )
    else:
        seed.write_text(
            f'#!/usr/bin/env bash\necho "seed failure" >&2\nexit {exit_code}\n'
        )
    seed.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add seed-warm-lane.sh stub'], cwd=repo)


def _backdate_stamp(path: Path, created_at: datetime) -> None:
    """Rewrite the ``.task/interactive.json`` stamp's ``created_at`` field."""
    stamp_path = path / '.task' / 'interactive.json'
    stamp = json.loads(stamp_path.read_text())
    stamp['created_at'] = created_at.isoformat()
    stamp_path.write_text(json.dumps(stamp))


async def _registered_worktree_paths(repo: Path) -> set[str]:
    """Return the set of registered worktree paths (resolved) via `git worktree list`."""
    rc, out, _ = await _run(['git', 'worktree', 'list', '--porcelain'], cwd=repo)
    assert rc == 0, 'git worktree list --porcelain failed'
    paths = set()
    for line in out.splitlines():
        if line.startswith('worktree '):
            paths.add(str(Path(line[len('worktree '):].strip()).resolve()))
    return paths


# ---------------------------------------------------------------------------
# Harness + server builder — mirrors _build_harness/_make_orch_config from
# test_warm_lane_integration_gate.py / test_harness_warm_lane_wiring.py, plus
# the create_server(...)/_call_tool pattern from
# escalation/tests/test_warm_worktree_verbs.py.
# ---------------------------------------------------------------------------


def _build_harness(config: OrchestratorConfig) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    Mirrors _build_harness from test_harness_warm_lane_wiring.py /
    test_warm_lane_integration_gate.py. ``harness.git_ops`` is a REAL GitOps
    (not patched), so its ``warm_lane_pool`` is a real WarmLanePool and
    ``harness._run_interactive_worktree_reaper_pass`` is a real bound method.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        return Harness(config)


def _make_config(
    repo: Path, *, max_concurrent_tasks: int = 3,
) -> OrchestratorConfig:
    """Build a minimal OrchestratorConfig with a real warm-lane pool enabled.

    Mirrors _make_orch_config from test_warm_lane_integration_gate.py. Pool
    size == max_concurrent_tasks exactly (GitConfig.spare_warm_lanes
    defaults to 0), giving K FREE lanes for the I1 dispatch-capacity
    snapshot.
    """
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=max_concurrent_tasks,
        git=GitConfig(warm_lane_pool=True),
    )


def _build_harness_and_server(
    config: OrchestratorConfig, tmp_path: Path,
) -> tuple[Harness, Any]:
    """Build a (harness, server) pair sharing one real GitOps/WarmLanePool.

    ``harness.scheduler`` is a MagicMock (Scheduler is patched out in
    ``_build_harness``); a real sentinel set is assigned to
    ``harness.scheduler._dispatched`` so the I1 dispatch-capacity assertion
    is a byte-for-byte set-equality snapshot rather than resting on a
    MagicMock auto-attribute. ``server`` is wired with ``harness=harness``
    so the β claim/release verbs operate on this exact GitOps/pool instance.
    """
    harness = _build_harness(config)
    harness.scheduler._dispatched = {'sentinel-a', 'sentinel-b'}
    server = create_server(EscalationQueue(tmp_path / 'esc'), harness=harness)
    return harness, server


async def _call_tool(server: Any, name: str, **kwargs: Any) -> Any:
    """Invoke an async MCP tool by name (mirrors test_server.py's _call_tool)."""
    tool = await server.get_tool(name)
    return await tool.fn(**kwargs)


# ---------------------------------------------------------------------------
# Shared fixture + I1 snapshot/assert helpers — most scenarios below need the
# SAME repo+harness+server+pool shape (committed seed stub,
# max_concurrent_tasks=3 real FREE lanes). Extracted once here rather than
# repeated per test class. Two scenarios deliberately build their own
# repo/harness inline instead of using this fixture, because they are NOT
# the shared shape: the fail-soft cold-claim case (no seed script committed)
# and the no-harness cold-fallback case (no harness wired at all).
# ---------------------------------------------------------------------------


@dataclass
class WarmGate:
    """A repo + harness + server sharing one real GitOps/WarmLanePool."""

    repo: Path
    harness: Harness
    server: Any
    pool: WarmLanePool


@pytest.fixture
def warm_gate(tmp_path: Path) -> WarmGate:
    """Build the shared WarmGate shape: committed seed stub, K=3 FREE lanes.

    Uses ``asyncio.run`` for the async repo setup rather than an async
    fixture, mirroring ``ig_git_repo`` in test_warm_lane_integration_gate.py
    — this fixture itself runs synchronously, before pytest-asyncio's own
    event loop for the (async) test function is created.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    asyncio.run(_add_seed_script(repo))
    config = _make_config(repo, max_concurrent_tasks=3)
    harness, server = _build_harness_and_server(config, tmp_path)
    pool = harness.git_ops.warm_lane_pool
    assert pool is not None, 'setup: warm_lane_pool must be enabled'
    return WarmGate(repo=repo, harness=harness, server=server, pool=pool)


def _free_count(pool: WarmLanePool) -> int:
    return sum(1 for s in pool._lanes.values() if s == LaneState.FREE)


@dataclass(frozen=True)
class _I1Snapshot:
    free: int
    assignments: Any
    dispatched: frozenset[str]


def _snapshot_i1(gate: WarmGate) -> _I1Snapshot:
    return _I1Snapshot(
        free=_free_count(gate.pool),
        assignments=gate.pool.assignments_snapshot(),
        dispatched=frozenset(gate.harness.scheduler._dispatched),
    )


def _assert_i1_unchanged(gate: WarmGate, before: _I1Snapshot, *, where: str) -> None:
    """Assert the WarmLanePool + scheduler dispatch capacity are unperturbed.

    The FREE-count and assignments_snapshot() checks are the load-bearing
    regression signal here: they run against the ONE real GitOps/WarmLanePool
    instance the whole flow shares, so a genuine isolation leak (e.g. the
    interactive path accidentally touching a _lane-* slot) would flip them.

    The scheduler._dispatched check is comparatively weak as a regression
    catcher — ``harness.scheduler`` is a MagicMock (see ``_build_harness``)
    and the interactive claim/release/reap path structurally never
    references the scheduler, so this equality cannot itself catch a real
    production regression. It is asserted anyway, uniformly at every call
    site (claim, release, AND reap), to document the PRD Contract's I1
    corollary — "dispatch capacity is unperturbed" — consistently across
    the whole gate rather than in some scenarios and not others.
    """
    free_now = _free_count(gate.pool)
    assert free_now == before.free, (
        f'I1 VIOLATION at {where}: FREE lane count changed '
        f'(before={before.free}, after={free_now})'
    )
    assert gate.pool.assignments_snapshot() == before.assignments, (
        f'I1 VIOLATION at {where}: assignments_snapshot() changed'
    )
    dispatched_now = frozenset(gate.harness.scheduler._dispatched)
    assert dispatched_now == before.dispatched, (
        f'I1 VIOLATION at {where}: scheduler._dispatched changed '
        f'(before={set(before.dispatched)}, after={set(dispatched_now)})'
    )


# ---------------------------------------------------------------------------
# Step-01/02: I1 isolation across claim→release THROUGH the β verbs, with a
# real WarmLanePool + a seeded scheduler._dispatched set live on the harness.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestIsolationInvariantI1:
    """I1: claiming/releasing an interactive worktree THROUGH the β verbs
    must never perturb WarmLanePool state or scheduler dispatch capacity —
    proven across the full verb surface (distinct from α's primitive-level
    isolation test, which never goes through claim_warm_worktree/
    release_warm_worktree).
    """

    async def test_pool_and_dispatch_capacity_unchanged_across_claim_release(
        self, warm_gate: WarmGate,
    ) -> None:
        gate = warm_gate
        free_before = _free_count(gate.pool)
        assert free_before == 3, f'setup: expected 3 FREE lanes, got {free_before}'
        before = _snapshot_i1(gate)

        claim = await _call_tool(
            gate.server, 'claim_warm_worktree', slug='iso', project_root=str(gate.repo),
        )
        assert 'error' not in claim, f'setup: claim failed: {claim}'

        # I1 must hold immediately AFTER claim.
        _assert_i1_unchanged(gate, before, where='after claim_warm_worktree')
        claim_path = Path(claim['path'])
        assert gate.pool.is_lane(claim_path) is False, (
            'I1 VIOLATION: the interactive worktree path is registered as a pool lane'
        )
        assert gate.pool.state(claim_path) is None, (
            'I1 VIOLATION: the pool reports a LaneState for the interactive '
            'worktree path'
        )
        lane_dirs = sorted(
            p.name for p in gate.harness.git_ops.worktree_base.iterdir()
            if p.name.startswith('_lane-')
        )
        assert lane_dirs == [], (
            f'I1 VIOLATION: a _lane-* dir was created on disk: {lane_dirs}'
        )
        assert claim_path.name.startswith('_iact-'), (
            f'expected the claimed path in the _iact-* band, got {claim_path.name!r}'
        )

        rel = await _call_tool(
            gate.server, 'release_warm_worktree',
            path_or_branch=claim['path'], project_root=str(gate.repo),
        )
        assert rel['removed'] is True, f'expected removed=True, got: {rel}'

        # I1 must hold again AFTER release.
        _assert_i1_unchanged(gate, before, where='after release_warm_worktree')


# ---------------------------------------------------------------------------
# Step-03/04: claim/release roundtrip — worktree registered after claim,
# gone from `git worktree list` after release.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestClaimReleaseRoundtrip:
    """The user-observable claim→worktree / release→removed roundtrip,
    driven through the β verbs (distinct from TestIsolationInvariantI1,
    which asserts pool isolation rather than the roundtrip observable).
    """

    async def test_claim_registers_release_removes(self, warm_gate: WarmGate) -> None:
        gate = warm_gate

        claim = await _call_tool(
            gate.server, 'claim_warm_worktree', slug='rt', project_root=str(gate.repo),
        )
        assert 'error' not in claim, f'setup: claim failed: {claim}'
        assert set(claim.keys()) >= {'path', 'branch', 'warm', 'base_ref'}, (
            f'expected claim to carry path/branch/warm/base_ref, got: {claim.keys()}'
        )
        assert claim['branch'] == 'task/rt', (
            f"expected branch == 'task/rt', got {claim['branch']!r}"
        )
        claimed_path = str(Path(claim['path']).resolve())
        registered = await _registered_worktree_paths(gate.repo)
        assert claimed_path in registered, (
            f'expected {claimed_path} to be a registered git worktree; got: {registered}'
        )

        rel = await _call_tool(
            gate.server, 'release_warm_worktree',
            path_or_branch=claim['path'], project_root=str(gate.repo),
        )
        assert rel['removed'] is True, f'expected removed=True, got: {rel}'

        registered_after = await _registered_worktree_paths(gate.repo)
        assert claimed_path not in registered_after, (
            f'expected {claimed_path} gone from `git worktree list` after release; '
            f'got: {registered_after}'
        )


# ---------------------------------------------------------------------------
# Step-05/06: seed invocation `--fresh-checkout` — warm success path THROUGH
# the β verb, and the fail-soft cold path when the seed script is absent.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSeedInvocationFreshCheckout:
    """Proves claim_warm_worktree's underlying create_interactive_worktree
    invokes _seed_warm_lane in '--fresh-checkout' mode: a committed seed
    stub CoW-seeds target/ (warm=True) AND records the mode flag it actually
    received (asserted below, so a regression to the wrong mode or a
    reordered/dropped argument would be caught — not just the fact that
    *some* seed ran), while an absent seed script degrades fail-soft
    (warm=False, no error, worktree retained) rather than raising or losing
    the worktree. Does NOT assert any recompile/timing/warmth magnitude
    (reify-side, out of scope — see module docstring's G6 note).
    """

    async def test_committed_seed_script_populates_target_seeded_bin(
        self, warm_gate: WarmGate,
    ) -> None:
        gate = warm_gate

        claim = await _call_tool(
            gate.server, 'claim_warm_worktree', slug='warm', project_root=str(gate.repo),
        )
        assert 'error' not in claim, f'setup: claim failed: {claim}'
        assert claim['warm'] is True, (
            f"expected warm is True (committed seed script ran), got {claim['warm']!r}"
        )
        seeded = Path(claim['path']) / 'target' / 'seeded.bin'
        assert seeded.exists(), (
            f'expected target/seeded.bin to exist after a --fresh-checkout CoW '
            f'seed run invoked via claim_warm_worktree, got no file at {seeded}'
        )

        # Pin the invocation contract itself (<base_target> <lane_dir>
        # <mode> — see GitOps._seed_warm_lane), not just target/'s
        # existence: the recorded argv's last positional arg must be the
        # '--fresh-checkout' mode flag.
        argv_file = Path(claim['path']) / 'target' / 'seed-argv.txt'
        argv = argv_file.read_text().splitlines()
        assert len(argv) == 3, (
            f'expected 3 recorded positional args (base_target, lane_dir, '
            f'mode), got: {argv!r}'
        )
        assert argv[-1] == '--fresh-checkout', (
            f"expected the seed script's mode arg (last positional) to be "
            f"'--fresh-checkout', got argv={argv!r}"
        )

    async def test_absent_seed_script_is_fail_soft_cold(self, tmp_path: Path) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)
        # Deliberately do NOT commit scripts/seed-warm-lane.sh — models the
        # absent-script fail-soft path (_seed_warm_lane returns rc=127).
        # Separate repo + its own harness/server, per the plan's isolation
        # of the fail-soft case from the warm-path fixture above.
        config = _make_config(repo, max_concurrent_tasks=3)
        _harness, server = _build_harness_and_server(config, tmp_path)

        claim = await _call_tool(
            server, 'claim_warm_worktree', slug='cold', project_root=str(repo),
        )
        assert 'error' not in claim, (
            f'fail-soft is a SUCCESS, not an error — got: {claim}'
        )
        assert claim['warm'] is False, (
            f"expected warm is False (no seed script present), got {claim['warm']!r}"
        )
        registered = await _registered_worktree_paths(repo)
        claimed_path = str(Path(claim['path']).resolve())
        assert claimed_path in registered, (
            f'expected the cold worktree to still be registered (retained, never '
            f'removed on a seed fault); got: {registered}'
        )


# ---------------------------------------------------------------------------
# Step-07/08: reap I2 through the harness reaper pass, composed with I1 pool
# isolation — δ's harness-cadence wiring folded into the same B+H gate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapI2ViaHarnessPass:
    """I2: harness._run_interactive_worktree_reaper_pass() reclaims a stale
    _iact-* worktree claimed via the β verb, preserves a within-TTL live
    one, and — composing with I1 — never perturbs WarmLanePool state OR
    scheduler dispatch capacity while doing so (the reaper structurally only
    ever touches the _iact-* band). Asserted via the same
    ``_assert_i1_unchanged`` helper used by the claim/release scenarios, so
    the I1 invariant is checked identically (pool AND scheduler._dispatched)
    regardless of whether it is claim, release, or reap that ran.
    """

    async def test_stale_reaped_live_preserved_pool_untouched(
        self, warm_gate: WarmGate,
    ) -> None:
        gate = warm_gate
        before = _snapshot_i1(gate)

        stale_claim = await _call_tool(
            gate.server, 'claim_warm_worktree', slug='stale', project_root=str(gate.repo),
        )
        assert 'error' not in stale_claim, f'setup: stale claim failed: {stale_claim}'
        live_claim = await _call_tool(
            gate.server, 'claim_warm_worktree', slug='live', project_root=str(gate.repo),
        )
        assert 'error' not in live_claim, f'setup: live claim failed: {live_claim}'

        stale_path = Path(stale_claim['path'])
        live_path = Path(live_claim['path'])

        # Make 'live' active: a fresh commit beyond main means its age is
        # measured from the newest-commit time (not the stamp), comfortably
        # within TTL — mirrors test_harness_interactive_reaper.py's
        # end-to-end convention.
        (live_path / 'work.txt').write_text('work\n')
        await _run(['git', 'add', '-A'], cwd=live_path)
        await _run(['git', 'commit', '-m', 'wip on live'], cwd=live_path)

        # 'stale' has no commits beyond main, so its age is measured from
        # the .task/interactive.json stamp — backdate it past the TTL.
        _backdate_stamp(
            stale_path,
            datetime.now(UTC)
            - timedelta(seconds=gate.harness.config.git.interactive_worktree_ttl + 3600),
        )

        await gate.harness._run_interactive_worktree_reaper_pass()

        registered = await _registered_worktree_paths(gate.repo)
        assert str(stale_path.resolve()) not in registered, (
            f'expected the stale interactive worktree to be reaped; '
            f'registered: {registered}'
        )
        assert str(live_path.resolve()) in registered, (
            f'expected the live interactive worktree to be preserved; '
            f'registered: {registered}'
        )

        # I1 ∧ I2 composition: the reap must never perturb the pool OR the
        # scheduler's dispatch capacity.
        _assert_i1_unchanged(gate, before, where='after the reaper pass')


# ---------------------------------------------------------------------------
# Step-09/10: cold fallback — claim_warm_worktree on a standalone (no
# harness) server reports a structured error, never raises.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestColdFallbackNoHarness:
    """The CI-producible boundary of /do's "escalation MCP unreachable"
    path (the actual /do cold-fall-back-to-EnterWorktree wiring is task ε,
    out of ζ scope): claim_warm_worktree on a standalone server
    (``harness=None``, modeling an unreachable/absent orchestrator) must
    report a structured ``{'error': ...}`` rather than raise.
    """

    async def test_claim_reports_structured_error_never_raises(
        self, tmp_path: Path,
    ) -> None:
        server_no_harness = create_server(EscalationQueue(tmp_path / 'esc'))

        res = await _call_tool(
            server_no_harness, 'claim_warm_worktree',
            slug='cf', project_root=str(tmp_path),
        )

        assert 'error' in res, (
            f'expected a reported, structured fallback error when no harness '
            f'is wired (never a raise), got: {res}'
        )


# ---------------------------------------------------------------------------
# Step-11/12: the combined B+H integration gate — the single ζ deliverable
# roping α (create_interactive_worktree) / β (claim/release verbs) / δ (the
# reaper) into one end-to-end flow, proving I1 ∧ I2 hold throughout.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInteractiveWarmWorktreeIntegrationGate:
    """The ζ deliverable: one harness+pool+server drives claim → release →
    (claim stale + live) → reap, and asserts the WarmLanePool FREE count,
    assignments_snapshot(), and scheduler._dispatched are byte-for-byte
    unchanged across the ENTIRE flow — the one "boundary test passes in CI"
    signal for the PRD's B+H (escalation verb + harness) gate.
    """

    async def test_full_claim_release_reap_preserves_isolation(
        self, warm_gate: WarmGate,
    ) -> None:
        gate = warm_gate
        free_before = _free_count(gate.pool)
        assert free_before == 3, f'setup: expected 3 FREE lanes, got {free_before}'
        before = _snapshot_i1(gate)

        # ── Phase 1 (β claim) ──────────────────────────────────────────
        claim = await _call_tool(
            gate.server, 'claim_warm_worktree', slug='gate', project_root=str(gate.repo),
        )
        assert 'error' not in claim, f'setup: claim failed: {claim}'
        assert claim['warm'] is True, (
            f"expected warm is True (committed seed script ran), got {claim['warm']!r}"
        )
        claim_path = Path(claim['path'])
        assert gate.pool.is_lane(claim_path) is False, (
            'I1 VIOLATION: the interactive worktree path is registered as a pool lane'
        )
        lane_dirs = sorted(
            p.name for p in gate.harness.git_ops.worktree_base.iterdir()
            if p.name.startswith('_lane-')
        )
        assert lane_dirs == [], (
            f'I1 VIOLATION: a _lane-* dir was created on disk: {lane_dirs}'
        )
        _assert_i1_unchanged(gate, before, where='after phase 1 claim')

        # ── Phase 2 (β release) ────────────────────────────────────────
        rel = await _call_tool(
            gate.server, 'release_warm_worktree',
            path_or_branch=claim['path'], project_root=str(gate.repo),
        )
        assert rel['removed'] is True, f'expected removed=True, got: {rel}'
        registered_after_release = await _registered_worktree_paths(gate.repo)
        assert str(claim_path.resolve()) not in registered_after_release, (
            f'expected {claim_path} gone from `git worktree list` after release; '
            f'got: {registered_after_release}'
        )
        _assert_i1_unchanged(gate, before, where='after phase 2 release')

        # ── Phase 3 (δ reap, I2) ───────────────────────────────────────
        stale_claim = await _call_tool(
            gate.server, 'claim_warm_worktree', slug='gate-stale', project_root=str(gate.repo),
        )
        assert 'error' not in stale_claim, f'setup: gate-stale claim failed: {stale_claim}'
        live_claim = await _call_tool(
            gate.server, 'claim_warm_worktree', slug='gate-live', project_root=str(gate.repo),
        )
        assert 'error' not in live_claim, f'setup: gate-live claim failed: {live_claim}'

        stale_path = Path(stale_claim['path'])
        live_path = Path(live_claim['path'])

        # Keep 'gate-live' active with a fresh commit beyond main (mirrors
        # TestReapI2ViaHarnessPass's convention).
        (live_path / 'work.txt').write_text('work\n')
        await _run(['git', 'add', '-A'], cwd=live_path)
        await _run(['git', 'commit', '-m', 'wip on gate-live'], cwd=live_path)

        # 'gate-stale' has no commits beyond main — backdate its stamp past
        # the TTL so the reaper reclaims it.
        _backdate_stamp(
            stale_path,
            datetime.now(UTC)
            - timedelta(seconds=gate.harness.config.git.interactive_worktree_ttl + 3600),
        )

        await gate.harness._run_interactive_worktree_reaper_pass()

        registered_after_reap = await _registered_worktree_paths(gate.repo)
        assert str(stale_path.resolve()) not in registered_after_reap, (
            f'expected gate-stale to be reaped; registered: {registered_after_reap}'
        )
        assert str(live_path.resolve()) in registered_after_reap, (
            f'expected gate-live to be preserved; registered: {registered_after_reap}'
        )

        # ── Final re-assertion: I1 (pool AND scheduler dispatch capacity)
        # held immediately after phase 3's reap, and across the ENTIRE flow.
        _assert_i1_unchanged(
            gate, before, where='after phase 3 reap (end of the full claim/release/reap flow)',
        )
