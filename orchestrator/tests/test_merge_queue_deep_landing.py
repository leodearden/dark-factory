"""Deep merge-ahead δ — prefix landing on tip pass, the in-order CAS walk (task 3186).

PRD: ``plans/deep-merge-ahead-prd.md`` task δ (Phase 2 vertical slice).
Capability sidecar: ``plans/deep-merge-ahead-prd.capability-manifest.yaml``.

δ is the ONLY place adoption may ever be introduced.  γ (task 3185) built the
chain, redirected the verify onto its TIP, and recorded the verdict into the
halving state machine — but deliberately landed nothing, exiting BOTH arms
through ``_requeue_request`` (merge_queue.py:18603-18663, "THE NON-ADOPTING
EXIT").  δ replaces ONLY the tip-PASS arm with the in-order CAS walk over
``chain.links``, reusing ``_finalize_inflight``'s existing terminal trio
(``_journal_landed_then_advance`` + ``_finalize_advanced_merge`` +
``_resolve_or_drop_abandoned``) once per link.

The two soundness fences γ installed that δ must NOT relax:
  * The chain-arm ``except Exception`` branch stays non-adopting
    (merge_queue.py:18623) — a tip verdict proves the cumulative TREE, so a
    green tip licenses landing every prefix member (each is a verified subset),
    while an infra ERROR proves nothing about anyone.
  * A chain-landed link never acquired a ``SpecPermit`` or a ``CapPermit``, so
    the walk must retire it WITHOUT calling ``PermitLedger.release`` — which
    raises ``AssertionError`` on a non-live token
    (merge_speculation_controller.py:213-239) and would break the structural
    identity ``slot_available + len(live) == depth``.

Step → coverage map:
  step-01 RED — ``landed_via_chain`` reaches the ``merge_finalized`` payload
  step-03 RED — the tip-pass ADOPTION signal out of ``_run_inflight_verify``
  step-05 RED — the in-order CAS walk over ``entry.chain.links``
  step-07 RED — stale-CAS abort (PRD decision #9) + 3003 DEFER inheritance
  step-09 RED — head-verify cancellation with a clean verify-lease release
  step-11 RED — conservation: the walk consumes no per-item speculation permits
  step-13 RED — δ end to end (landing walk, tip fail, kill switch, hot reload)

Harness notes (see plan pre-1; conventions cloned from
test_merge_queue_deep_dispatch.py:1-36):
  * ``orchestrator/pyproject.toml`` does NOT set ``asyncio_mode`` → pytest-asyncio
    runs STRICT, so ``@pytest.mark.asyncio`` is required on async test classes.
  * That same config turns "marked with @pytest.mark.asyncio but not an async
    function" into an ERROR — never put a sync ``test_*`` inside a marked class.
    Sync tests live in their OWN unmarked class.
  * Default per-test ``timeout = 60``; any class doing real-git worktree/merge
    work carries ``@pytest.mark.timeout(180)``.
  * ``orchestrator/tests/`` has no ``__init__.py``, so flat helpers are imported
    by bare module name — which is why this module CLONES γ's fixture block
    rather than importing it from a sibling test file (that would couple the
    two suites' collection order).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import pytest

from orchestrator.config import GitConfig, MergeDeepConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeRequest, SpeculativeMergeWorker
from orchestrator.merge_types import MergeResult, QueuedBranch, RealMergeItem

# ── repo fixtures (cloned from test_merge_queue_deep_dispatch.py:59-112) ──────


async def _setup_repo(repo: Path) -> None:
    """Init a repo with a 20-line shared.txt plus disjoint.txt.

    ``shared.txt`` gets **20** numbered lines rather than 3: git's 3-line diff
    context window makes near-line edits in a tiny file conflict even when they
    touch different lines (gotcha documented at
    test_merge_queue_conflict_graph.py:454-460).  20 lines makes a line-1 vs
    line-15 edit pair genuinely non-conflicting, so this file can build both
    conflicting and non-conflicting chain fixtures from one seed.
    """
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    (repo / 'shared.txt').write_text(''.join(f'line{i}\n' for i in range(1, 21)))
    (repo / 'disjoint.txt').write_text('aaa\nbbb\nccc\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _add_recording_seed_to_repo(repo: Path) -> None:
    """Commit a recording ``scripts/seed-warm-lane.sh`` into the repo at HEAD.

    Without a COMMITTED seed script, ``acquire_spec_lane`` soft-degrades to a
    cold ephemeral worktree (``warm=False``), silently making every "warm
    ``_spec-`` lane" assertion vacuous.  conftest's autouse
    ``_isolate_warm_lane_script_dir`` pins ``ORCH_WARM_LANE_SCRIPT_DIR`` at an
    absent dir, but a repo-local ``scripts/seed-warm-lane.sh`` still resolves
    first — which is why committing it here works.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'seed-warm-lane.sh'
    script.write_text(
        '#!/usr/bin/env bash\n'
        '# argv: <base_target> <lane_dir> <mode>\n'
        'ARGV_FILE="$2/scripts/seed-warm-lane.sh.argv"\n'
        'echo "$@" >> "$ARGV_FILE"\n',
    )
    script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add recording seed-warm-lane.sh'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    asyncio.run(_add_recording_seed_to_repo(repo))
    return repo


# ── config / GitOps helpers ──────────────────────────────────────────────────


def _make_spec_git_config(*, on: bool = True, **extra) -> GitConfig:
    """Build a GitConfig with ``merge_spec_warm_lane_pool=on``."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        merge_spec_warm_lane_pool=on,
        **extra,
    )


def _make_git_ops(repo: Path, *, pool: bool = True, size: int = 1) -> GitOps:
    """Build a GitOps over *repo* with (or without) a ``_spec-`` warm lane pool.

    ``merge_spec_warm_lane_pool_size`` is a **GitOps constructor kwarg**, not a
    GitConfig field (git_ops.py:2041).  The pool is only constructed when
    ``size > 0 AND config.merge_spec_warm_lane_pool`` (git_ops.py:2093).
    """
    return GitOps(
        _make_spec_git_config(on=pool), repo, merge_spec_warm_lane_pool_size=size,
    )


def _make_config(
    repo: Path,
    git_config: GitConfig | None = None,
    *,
    chain_cap: int = 0,
) -> OrchestratorConfig:
    """Build an OrchestratorConfig whose ``merge_deep.chain_cap`` is *chain_cap*.

    ``chain_cap`` defaults to 0 — α's shipped kill switch — so a test that wants
    the deep path must opt in explicitly, exactly as an operator would.
    """
    return OrchestratorConfig(
        project_root=repo,
        git=git_config or _make_spec_git_config(),
        merge_deep=MergeDeepConfig(chain_cap=chain_cap),
    )


# ── MergeRequest / item helpers ──────────────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    *,
    lane: Literal['normal', 'high'] = 'normal',
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    CAUTION: builds its future via ``asyncio.get_running_loop()``, so this can
    ONLY be called from inside an async test — never at module/fixture scope.
    *branch* is the BARE suffix (``'101'``), not the prefixed name.
    """
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane=lane,
    )


def _make_item(
    req: MergeRequest,
    merge_commit: str,
    merge_wt: Path,
    *,
    speculative: bool = True,
    base_sha: str = 'dead' * 10,
) -> RealMergeItem:
    """Build a RealMergeItem around *req* sitting at *merge_commit*.

    ``speculative=True`` is SLOT 2 (the merge-ahead item, stacked on the
    predecessor's merge commit) — the ONLY kind ``_deep_chain_placement`` gates
    on (merge_queue.py:12268), and therefore the only kind δ's walk ever sees;
    ``speculative=False`` is SLOT 1, the head trust-anchor verify against real
    main, which δ CANCELS and lands on the tip's authority (decision #3).
    """
    return RealMergeItem(
        request=req,
        merge_result=MergeResult(
            success=True, merge_commit=merge_commit, merge_worktree=merge_wt,
        ),
        merge_wt=merge_wt,
        base_sha=base_sha,
        speculative=speculative,
    )


def _ephemeral_merge_wt(git_ops: GitOps, tag: str) -> Path:
    """Create (and return) a stand-in ephemeral ``_merge-<tag>`` worktree dir.

    Real dispatch hands `RealMergeItem.merge_wt` an ephemeral `_merge-<uuid>`
    minted by `merge_to_main`.  Tests that reach code which DISPOSES of that
    worktree must not pass the repo root in its place: the chain arm calls
    `_cleanup_owned_merge_worktree(item.merge_wt)`, whose rmtree fallback would
    then delete the fixture repo out from under the test (and every later git
    call in it) — a destructive pass, not a real one.

    Deliberately NOT created on disk: materialising ``worktree_base`` by hand
    makes ``acquire_spec_lane``'s create-once pool-storage check see a
    directory it did not provision and cold-fall-back, silently turning every
    warm ``_spec-`` lane assertion in this module vacuous.  A bare path is
    enough — disposal of a missing worktree is a no-op either way.
    """
    return git_ops.worktree_base / f'_merge-{tag}'


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring)."""
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


# ── event capture (from test_merge_queue_deep_dispatch.py:240-262) ────────────


class _CapturingEventStore(EventStore):
    """Capturing EventStore — records emit() calls without touching sqlite.

    Mirrors ``_LateArrivalFakeEventStore`` (test_merge_speculation.py) /
    ``_FakeEventStore`` (test_merge_queue_concurrent_verify.py).
    """

    def __init__(self) -> None:
        object.__init__(self)
        self.emitted: list[dict] = []

    def emit(  # type: ignore[override]
        self, event_type, *, task_id=None, phase=None, role=None,
        data=None, cost_usd=None, duration_ms=None,
    ) -> None:
        self.emitted.append({'event_type': event_type, 'data': data or {}})

    def events_of(self, event_type: EventType) -> list[dict]:
        return [e for e in self.emitted if e['event_type'] == event_type]


# ── git helpers ──────────────────────────────────────────────────────────────


async def _create_branch_editing(
    repo: Path,
    branch_name: str,
    filename: str,
    content: str,
    base_branch: str = 'main',
) -> str:
    """Create a branch that edits *filename* with *content*; return its SHA.

    Callers pass the FULL prefixed branch name (``'task/101'``), while
    ``_make_req`` takes the bare suffix (``'101'``).
    """
    await _run(['git', 'checkout', '-b', branch_name], cwd=repo)
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Edit {filename} in {branch_name}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    await _run(['git', 'checkout', base_branch], cwd=repo)
    return sha.strip()


async def _rev_parse(cwd: Path, rev: str = 'HEAD') -> str:
    """Return the stripped SHA of *rev* resolved inside *cwd*."""
    _, sha, _ = await _run(['git', 'rev-parse', rev], cwd=cwd)
    return sha.strip()


def _shared_txt_with(line_no: int, text: str) -> str:
    """Return a 20-line shared.txt body with line *line_no* replaced by *text*."""
    lines = [f'line{i}\n' for i in range(1, 21)]
    lines[line_no - 1] = f'{text}\n'
    return ''.join(lines)


# ── verify-lease / runner helpers ────────────────────────────────────────────


def _local_lease():
    """A LOCAL :class:`HostLease` whose runner is never actually driven.

    LOCAL deliberately, not remote: `lease.is_local` is precisely what selects
    the warm-swap block the chain arm SKIPS, and it is also the axis δ's
    head-cancel cleanliness argument splits on — a LOCAL head verify frees BOTH
    lease axes by construction when its task is cancelled (GitOps
    ``merge_verify_lease``'s finally at git_ops.py:3552-3554), while a REMOTE
    one SIGKILLs and leaks the fixed-key holder-pgid rendezvous file
    (verify_cancel.py:303-336) unless ``cli.py cancel_verify`` clears it.
    """
    from unittest.mock import MagicMock

    from orchestrator.verify_runner import HostLease

    runner = MagicMock()
    runner.name = 'local'
    runner.is_local = True
    return HostLease(name='local', runner=runner, is_local=True)


def _fake_pass_runner(name: str = 'fake-runner'):
    """A RemoteRunner-shaped fake whose ``run_merge_verify`` always passes.

    Copied from test_merge_queue_deep_dispatch.py:1398 rather than imported:
    ``orchestrator/tests/`` has no ``__init__.py``, so a cross-module helper
    import would be a bare-module-name import of a sibling TEST file, which
    couples the two suites' collection order.
    """
    from unittest.mock import AsyncMock, MagicMock

    from orchestrator.verify import VerifyResult

    fake = MagicMock()
    fake.name = name
    fake.is_local = False
    fake.run_merge_verify = AsyncMock(return_value=VerifyResult(
        passed=True, test_output='ok', lint_output='', type_output='',
        summary='ok', category='',
    ))
    fake.cancel_verify = AsyncMock(return_value=0)
    fake.probe_clean = AsyncMock(return_value=True)
    return fake


def _fail_verify_result():
    """A failing :class:`VerifyResult` — a RED tip verdict."""
    from orchestrator.verify import VerifyResult

    return VerifyResult(
        passed=False, test_output='tip is red', lint_output='', type_output='',
        summary='fail', category='',
    )


# ── quiescence oracle (from test_merge_queue_invariant_integration_gate.py:508) ─


def _assert_quiescent(
    worker: SpeculativeMergeWorker,
    main_sha: str,
    requests: list[MergeRequest],
) -> None:
    """Assert the 6-surface QUIESCENCE contract holds for *worker*.

    Called after each δ scenario — full landing, partial stale-CAS landing, and
    tip fail alike — to confirm the pipeline returned to a clean resting state
    with no leaked permits, worktrees, ledger entries, unresolved Futures, or
    registry residue:

      (a) every request in *requests* has resolved (done or cancelled) — no
          dangling in-flight work left over from the scenario.
      (b) worker.speculation_accounting_violations() == [] — I4 permit/cap
          conservation holds.  Requires *worker* to still be ``_running``
          (both accounting methods short-circuit to [] when not running —
          see their docstrings — so a stopped worker would make this
          assertion vacuous, not meaningful).
      (c) worker.worktree_ledger_violations() == [] — I6 on-disk
          ``_merge-*`` worktree ledger is fully accounted for.  Also
          requires a running worker (same short-circuit).
      (d) the request-liveness ledger is empty AFTER sweeping resolved
          entries.  Resolution is detected PASSIVELY — RequestLedger has no
          on-resolve hook, so a resolved request stays armed until
          ``sweep_resolved()`` runs; calling it here before ``is_empty()``
          is required, not optional.
      (e) worker.two_layer_invariants(main_sha) == [] — *main_sha* MUST be a
          REAL sha, never 'unknown': the base-chain and verify-base
          sub-checks are silently skipped for the 'unknown' sentinel, which
          would make this assertion pass vacuously rather than meaningfully.
      (f) set(worker._lifecycle.non_terminal_items()) == set() — the
          ItemLifecycle registry has retired every request_id; no registry
          leak survives quiescence.  Placed after the request-ledger sweep
          (d) so it samples a truly-drained pipeline.

    Cloned rather than imported for the no-``__init__.py`` reason above.
    """
    for req in requests:
        assert req.result.done() or req.result.cancelled(), (
            f'Expected request {req.request_id!r} (task {req.task_id!r}) to '
            f'have resolved (done or cancelled) at quiescence, but it is '
            f'still pending'
        )

    spec_violations = worker.speculation_accounting_violations()
    assert spec_violations == [], (
        f'speculation_accounting_violations() non-empty at quiescence: {spec_violations!r}'
    )

    wt_violations = worker.worktree_ledger_violations()
    assert wt_violations == [], (
        f'worktree_ledger_violations() non-empty at quiescence: {wt_violations!r}'
    )

    # Resolution is passive — sweep before asserting emptiness (see
    # RequestLedger.sweep_resolved's docstring).
    worker._request_ledger.sweep_resolved()
    assert worker._request_ledger.is_empty(), (
        f'request-liveness ledger non-empty at quiescence: '
        f'{worker._request_ledger.open_request_ids()!r}'
    )

    assert main_sha and main_sha != 'unknown', (
        f'_assert_quiescent requires a REAL main_sha, got {main_sha!r}'
    )
    tli_violations = worker.two_layer_invariants(main_sha)
    assert tli_violations == [], (
        f'two_layer_invariants({main_sha!r}) non-empty at quiescence: {tli_violations!r}'
    )

    registry_ids = set(worker._lifecycle.non_terminal_items())
    assert registry_ids == set(), (
        f'ItemLifecycle registry non-terminal at quiescence: {registry_ids!r}'
    )


# ── the SHIPPED canary arithmetic, transcribed ───────────────────────────────


def _canary_predicate_items_per(
    merge_verify_data: list[dict], merge_finalized_data: list[dict],
) -> float | None:
    """η1's ``items_per`` statistic, transcribed from the SHIPPED predicate.

    Transcribed from ``scripts/merge-deep-canary-predicate.sh:84-91`` — already
    COMMITTED CODE on main — rather than restated in the assertion's own words,
    following γ's ``_canary_says_deep`` precedent
    (test_merge_queue_deep_dispatch.py:2373).  This is what PINS
    ``landed_via_chain``'s numeric encoding: the shipped comment calls the
    result "items landed per deep verify run", and that arithmetic is only
    correct if the per-walk contributions SUM to the number of items the walk
    landed.  Emitting the chain size k on every one of k items would yield
    k²/n_deep; emitting 1-indexed positions would yield k(k+1)/2.  So a δ
    assertion that this expression computes the TRUE items-landed-per-deep-
    verify settles the encoding empirically instead of arguing the PRD's prose
    (which contradicts itself three ways — see plan decision #1).

    The shipped predicate carries a ``dur`` alongside each event's parsed
    ``data`` (for its unrelated p90 statistic); this helper takes the ``data``
    dicts alone, because durations play no part in the ``items_per`` arithmetic
    reproduced here.
    """
    # A "deep" verify is one whose verified tree carried >= 2 chained items.
    deep = [
        d for d in merge_verify_data
        if isinstance(d.get('chain_items'), int) and d['chain_items'] >= 2
    ]
    n_deep = len(deep)

    landed = [
        d['landed_via_chain'] for d in merge_finalized_data
        if isinstance(d.get('landed_via_chain'), int) and d['landed_via_chain'] >= 1
    ]
    return (sum(landed) / n_deep) if n_deep else None  # items landed per deep verify run
