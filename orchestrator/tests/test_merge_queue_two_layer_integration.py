"""Integration-gate suite λ=1895: full δ/ε/ζ/η/θ/ι pipeline composition.

This file is the G2 boundary-test leaf for the two-layer merge-queue PRD
(plans/two-layer-merge-queue-prd.md §8 Phase 4).  It drives the FULL
pipeline (δ conflict-graph + ε frontier + ζ aging + η bounce + θ breaker
+ ι metrics) through the REAL public methods against a REAL git repo with
the DEFAULT footprint detector.

Fake/instrumented runners only:
  - rebase_onto_main stubbed (AsyncMock)
  - run_scoped_verification + shutil.disk_usage mocked
  - no real ssh/build/merge-to-main

Integration delta over upstream unit suites
-------------------------------------------
The upstream unit suites hand-seed ``_suffix_conflict_graph`` directly.
This suite drives ``recompute_suffix_conflict_graph()`` against REAL git
branches with the DEFAULT footprint detector so the real graph flows into
aging / bounce / frontier.  Scenario 9 replaces the MagicMock worker
(used by the upstream θ test) with a real SpeculativeMergeWorker so the
breaker reads ι's REAL ``landings_total`` from the real ``snapshot()``
and a real landing drives the resume path.

§5.3 invariants asserted in a single run
-----------------------------------------
  I1. Frozen prefix immutable — inflight entries are never reordered or
      re-based out from under an in-flight verify.
  I2. Verify base equals frozen tip — a real-verify dispatch starts only
      against the tip of the frozen prefix (base-chain integrity via
      check_frozen_prefix_invariant).
  I3. Reorder touches only the unfrozen suffix — recompute_suffix_conflict_graph
      updates only _lane_buffers; _inflight order and base_sha are unchanged.
  I4. Liveness / no-starvation — an item in a clique eventually becomes head
      of clique and is picked; disjoint items bypass a blocked clique.

§9 boundary scenarios (DEFAULT detector; scenario 7=κ out of scope)
--------------------------------------------------------------------
  1. Textual conflict bounces disk-free — younger→needs_rebase at recompute
     time; no _merge-* worktree created, no verify slot consumed.
  2. Clean speculative rebase re-queues — work preserved, no agent dispatched,
     merge_first_enqueued_at unchanged.
  3. Real conflict escalates capped — Nth bounce → cap → blocked without rebase
     (1688 thrash-signature backstop).
  4. Aging beats FIFO within a footprint clique (ζ=1891 comparator via real
     recomputed graph, not a hand-seeded one).
  5. Disjoint throughput bypass — disjoint items bypass a blocked clique while
     the frozen prefix is held.
  6. Verify frontier immutable under suffix reorder — inflight order and
     base_sha survive a suffix reorder + recompute; two_layer_invariants == []
     before AND after.
  8. Contract textual⇒overlap — textual_edges ⊆ footprint_edges (DEFAULT
     detector; real branches, real recompute).
  9. Circuit-breaker fires + escalates + resumes — REAL Harness + REAL
     SpeculativeMergeWorker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.harness import Harness
from orchestrator.merge_queue import (
    MERGE_BOUNCE_CAP,
    NEEDS_REBASE_REASON_PREFIX,
    InflightEntry,
    MergeOutcome,
    MergeRequest,
    NoLandingsCircuitBreaker,
    SpeculativeItem,
    SpeculativeMergeWorker,
    SuffixConflictGraph,
)
from orchestrator.run_store import RunStore

# ── Module-level sentinel ─────────────────────────────────────────────────────

_SENTINEL_VERIFY_TASK = object()  # noqa: PD901
"""Sentinel for verify_task in pure unit tests.

frozen_prefix() checks ``e.verify_task is not None`` only, so any non-None
object doubles as a verifying-entry marker in tests that do not need a real
asyncio.Task.
"""

# Harness breaker escalation constants (must match harness.py implementation)
_BREAKER_ROLE = 'orchestrator-no-landings-breaker'
_BREAKER_SENTINEL = '__no_landings_breaker__'


# ── Repo seeding ──────────────────────────────────────────────────────────────


async def _setup_repo(repo: Path) -> None:
    """Initialise a git repo with README.md + shared.txt + disjoint.txt on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    (repo / 'shared.txt').write_text('line1\nline2\nline3\n')
    (repo / 'disjoint.txt').write_text('disjoint-line1\ndisjoint-line2\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# ── Request / item / entry builders ──────────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    lane: Literal['normal', 'high'] = 'normal',
    *,
    merge_first_enqueued_at: float | None = 1000.0,
    request_id: str | None = None,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
    The optional *request_id* kwarg lets tests pin a stable identity for
    graph-key assertions; when omitted a fresh UUID is auto-generated.
    """
    kwargs: dict = {}
    if request_id is not None:
        kwargs['request_id'] = request_id
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane=lane,
        merge_first_enqueued_at=merge_first_enqueued_at,
        **kwargs,
    )


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring)."""
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


def _make_fake_item(
    task_id: str,
    *,
    base_sha: str = 'aaaa0000',
    merge_commit: str | None = 'bbbb1111',
    config: OrchestratorConfig,
    git_repo: Path,
) -> tuple[MergeRequest, SpeculativeItem]:
    """Build a (MergeRequest, SpeculativeItem) pair from fake SHAs.

    Does NOT create real git branches — suitable for pure-unit tests that only
    exercise frozen-prefix / frozen_prefix_tip accessor methods without real
    git operations.  Must be called from an async context (for the future).
    """
    req = _make_req(task_id, f'task/{task_id}', config, git_repo)
    fake_merge_result = (
        MergeResult(success=True, merge_commit=merge_commit)
        if merge_commit is not None
        else None
    )
    item = SpeculativeItem(
        request=req,
        merge_result=fake_merge_result,
        merge_wt=None,
        base_sha=base_sha,
        speculative=False,
        skip_verify=False,
    )
    return req, item


def _make_inflight_entry(
    item: SpeculativeItem,
    *,
    verifying: bool = True,
) -> InflightEntry:
    """Build an InflightEntry for unit tests.

    verifying=True  → verify_task set to _SENTINEL_VERIFY_TASK (any non-None
                       object; frozen_prefix() only checks ``is not None``).
    verifying=False → verify_task=None (passthrough entry; excluded from the
                       frozen prefix by §5.3 definition).
    """
    return InflightEntry(
        item=item,
        lease=None,
        verify_task=_SENTINEL_VERIFY_TASK if verifying else None,  # type: ignore[arg-type]
        merge_wt=None,
        was_speculative=False,
        phase='verifying',
    )


# ── Branch / commit helpers ───────────────────────────────────────────────────


async def _create_branch_editing(
    repo: Path,
    branch_name: str,
    filename: str,
    content: str,
    base_branch: str = 'main',
) -> str:
    """Create a branch editing *filename* with *content*; return the branch SHA."""
    await _run(['git', 'checkout', '-b', branch_name], cwd=repo)
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Edit {filename} in {branch_name}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    await _run(['git', 'checkout', base_branch], cwd=repo)
    return sha.strip()


async def _create_frozen_tip_commit(
    repo: Path,
    filename: str,
    content: str,
) -> str:
    """Commit a change directly on main (simulates a frozen-prefix merge commit).

    Assumes the current branch is already 'main' (left by _setup_repo).
    Returns the resulting commit SHA.
    """
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Frozen-tip commit: {filename}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    return sha.strip()


# ── Harness factory + real-worker attachment (for scenario 9) ─────────────────


def _make_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Harness with a real Scheduler + EscalationQueue and a spy RunStore.

    Mirrors test_harness_no_landings_breaker._make_harness.
    Returns (harness, mock_run_store).
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-integration-0001'
    harness.event_store = EventStore(tmp_path / 'events.db', 'run-integration-0001')
    harness._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    return harness, mock_run_store


def _attach_real_worker(harness: Harness, git_ops: GitOps) -> SpeculativeMergeWorker:
    """Attach a REAL SpeculativeMergeWorker to the harness as harness._merge_worker.

    Integration delta over the upstream θ test (which used a MagicMock worker):
    the breaker reads ι's REAL ``landings_total`` from the real ``snapshot()``
    and a real landing drives the resume path.

    Returns the attached worker for direct manipulation in tests.
    """
    worker = SpeculativeMergeWorker(
        git_ops,
        asyncio.Queue(),
        escalation_queue=harness._escalation_queue,
    )
    harness._merge_worker = worker
    return worker


def _small_breaker(window: int = 3, margin: int = 1000) -> NoLandingsCircuitBreaker:
    """Return a NoLandingsCircuitBreaker with a small window for test speed."""
    return NoLandingsCircuitBreaker(window_samples=window, disk_margin_bytes=margin)


def _make_falling_disk_iter(
    start: int = 200_000,
    drop: int = 10_000,
):
    """Generate strictly-falling free-bytes values (for breaker-trip scenarios)."""
    current = start
    while True:
        yield current
        current -= drop
