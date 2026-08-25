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
import os
from pathlib import Path
from typing import Literal, TypedDict

import pytest

from orchestrator import merge_queue
from orchestrator.config import GitConfig, MergeDeepConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeRequest, SpeculativeMergeWorker
from orchestrator.merge_types import (
    CapPermit,
    MergeResult,
    QueuedBranch,
    RealMergeItem,
    SpecPermit,
)

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


# ── dispatch-scene spies (cloned from test_merge_queue_deep_dispatch.py) ─────


async def _merge_commit_off_main(repo: Path, branch: str, label: str) -> str:
    """Return a REAL merge commit of *branch* into main, WITHOUT advancing main.

    Stands in for the dispatching item's ``merge_result.merge_commit``.  It must
    not be reachable from ``main``, or every "landed on main" assertion below
    would hold vacuously before the walk ever ran.
    """
    await _run(['git', 'checkout', '-b', f'_tmp-{label}', 'main'], cwd=repo)
    await _run(['git', 'merge', '--no-ff', '-m', f'merge {branch}', branch], cwd=repo)
    sha = await _rev_parse(repo)
    await _run(['git', 'checkout', 'main'], cwd=repo)
    return sha


def _spy_post_merge_verify(monkeypatch, outcome=None, *, raises=None) -> list[dict]:
    """Replace ``_run_post_merge_verify`` with a recorder returning *outcome*.

    ``outcome=None`` is a PASS in this function's vocabulary; a
    :class:`VerifyResult` is a FAIL.  *raises* makes the verify blow up
    instead, which is the third exit — the one that stays NON-adopting under δ
    (merge_queue.py:18623).
    """
    calls: list[dict] = []

    async def _recording(git_ops, req, merge_wt, **kwargs):
        calls.append({'merge_wt': merge_wt, **kwargs})
        if raises is not None:
            raise raises
        return outcome

    monkeypatch.setattr(
        'orchestrator.merge_queue._run_post_merge_verify', _recording,
    )
    return calls


def _spy_chain_lane_release(monkeypatch) -> list[tuple]:
    """Record ``release_chain_build_lane`` calls WITHOUT suppressing them.

    Passthrough, not a stub: the lane must genuinely go back to FREE, so the
    pool-state assertions stay real while the call count stays observable.
    The ChainResult "Lane ownership" contract is EXACTLY-once, and δ adds a
    fourth exit to the three γ shipped — so this is what proves the adopting
    exit did not skip, nor double, the release.
    """
    calls: list[tuple] = []
    real = merge_queue.release_chain_build_lane

    async def _recording(git_ops, lane, *, warm):
        calls.append((lane, warm))
        return await real(git_ops, lane, warm=warm)

    monkeypatch.setattr(
        'orchestrator.merge_queue.release_chain_build_lane', _recording,
    )
    return calls


def _spy_advance_main(git_ops: GitOps, monkeypatch, *, hook=None) -> list[tuple]:
    """Record every ``advance_main`` call as ``(merge_sha, expected_main)``.

    Spied on the GitOps INSTANCE (not a merge_queue module reach-back, which
    test_merge_queue_reachback_patch_guard.py freezes) and PASSTHROUGH, so main
    really moves and the recorded ``expected_main`` chain can be checked against
    real history.  Template: test_merge_speculation.py:2280-2295.

    *hook*, when given, is awaited with the 1-based call ordinal BEFORE the
    passthrough.  Returning ``None`` falls through to the real
    ``advance_main``; returning an :class:`~orchestrator.git_ops.AdvanceOutcome`
    (or raising) SHORT-CIRCUITS that one call, which is how a δ scenario
    injects the mid-walk failure it is about.
    """
    calls: list[tuple] = []
    real = git_ops.advance_main

    async def _recording(merge_sha, merge_worktree=None, **kwargs):
        calls.append((merge_sha, kwargs.get('expected_main'), merge_worktree))
        if hook is not None:
            injected = await hook(len(calls))
            if injected is not None:
                return injected
        return await real(merge_sha, merge_worktree, **kwargs)

    monkeypatch.setattr(git_ops, 'advance_main', _recording)
    return calls


class _PermitCensus(TypedDict):
    """The shape :func:`_permit_census` returns — declared, not inferred.

    A bare ``dict[str, object]`` return erases every value type, so the
    conservation identity ``slot_available + len(live) == depth`` stops
    type-checking at the assertion sites that read it (``object`` supports
    neither ``+``/``>=`` nor ``len()``).  Spelling the six keys out keeps the
    counts ``int`` and the token views ``frozenset`` for the checker as well
    as for the reader.
    """

    spec_live: frozenset[SpecPermit]
    spec_available: int
    spec_depth: int
    cap_live: frozenset[CapPermit]
    cap_available: int
    cap_depth: int


def _permit_census(worker: SpeculativeMergeWorker) -> _PermitCensus:
    """Snapshot BOTH permit ledgers' conservation state in one comparable dict.

    ``live`` is captured as the frozenset of actual TOKENS, not merely a count:
    δ's hazard is not "how many permits" but "whose".  A walk that released a
    link's token would raise ``AssertionError`` (the token was never issued —
    merge_speculation_controller.py:213-239), while a walk that released the
    DISPATCHING item's token early would keep every count plausible and break
    only ownership — invisible to a size comparison, obvious to a set one.

    ``slot_available``/``depth`` come along so a reader can check the structural
    identity ``slot_available + len(live) == depth`` directly at any point,
    rather than only through ``speculation_accounting_violations``'s
    ``_running``-gated wrapper.
    """
    spec, cap = worker._speculation_ledger, worker._merge_ahead_ledger
    return {
        'spec_live': spec.live,
        'spec_available': spec.slot_available,
        'spec_depth': spec.depth,
        'cap_live': cap.live,
        'cap_available': cap.slot_available,
        'cap_depth': cap.depth,
    }


def _drain_residue(worker: SpeculativeMergeWorker) -> set[str]:
    """Retire whatever δ deliberately LEFT queued; return its task ids.

    δ's contract is "the walk touches the prefix and NOTHING else": the
    truncator, and every link past an abort, stay buffered with unresolved
    futures for their ordinary sequential path on a later round.  A scene that
    stops after one finalize therefore rests with real, INTENDED residue — so
    the whole-registry surfaces of :func:`_assert_quiescent` ((a) every future
    resolved, (f) nothing non-terminal) cannot hold until that residue is taken
    off the pipeline the way a later round would take it.

    This stands in for that round: each still-buffered request is detached by
    its waiter (``cancel()``) and retired through ``_retire_item``, the same
    registry chokepoint every terminal path funnels through.

    The RETURNED SET is what makes this safe rather than a whitewash — every
    caller asserts it equals the residue δ promised BEFORE trusting the
    quiescence that follows, so a landed link that wrongly stayed buffered (the
    double-land hazard) shows up as a residue-set mismatch instead of being
    quietly drained away.
    """
    drained: set[str] = set()
    for lane in ('high', 'normal'):
        buf = worker._lane_buffers[lane]
        while buf:
            req = buf.popleft()  # a deque, not a list — `pop(0)` is a TypeError
            drained.add(req.task_id)
            if not req.result.done():
                req.result.cancel()
            worker._retire_item(req.request_id)
    return drained


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


# ═══════════════════════════════════════════════════════════════════════════
# step-01: RED — `landed_via_chain` reaches the merge_finalized payload
#
# The carrier, before anything populates it.  `merge_finalized` has exactly ONE
# emit site (merge_queue.py:4763-4777, the `_on_finalized` done-callback that
# `enqueue_merge_request` registers at :4785) and `EventStore.emit` applies no
# schema validation — it just json.dumps()es the dict — so the ONLY thing that
# can carry a new field to η1's predicate is a new field on `MergeOutcome`
# threaded into that payload, following the `superseded_by`/`dedupe_fingerprint`
# /`disposition` optional-metadata precedent (merge_types.py:920).
# ═══════════════════════════════════════════════════════════════════════════


def _finalized_rows(db_path: Path) -> list[dict]:
    """Return every ``merge_finalized`` row's parsed ``data`` dict, in order.

    Reads the durable tier through real sqlite (the idiom at
    test_merge_queue.py:7588-7605) rather than a capturing fake, because the
    field only reaches η1 if it survives ``json.dumps`` into the ``data``
    column — a fake that records the dict by reference would pass even for a
    value the real emit path drops or cannot serialise.
    """
    import json
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT data FROM events WHERE event_type = 'merge_finalized' "
            'ORDER BY rowid'
        ).fetchall()
    finally:
        conn.close()
    return [json.loads(r[0]) for r in rows]


@pytest.mark.asyncio
class TestLandedViaChainCarrier:
    """``MergeOutcome.landed_via_chain`` → the ``merge_finalized`` payload."""

    def _config(self, tmp_path: Path) -> OrchestratorConfig:
        return _make_config(tmp_path, chain_cap=6)

    async def test_merge_outcome_defaults_landed_via_chain_to_none(
        self, tmp_path: Path,
    ) -> None:
        """(a) The field exists, defaults to None, and accepts an int.

        Defaulted-to-None is what keeps every EXISTING ``MergeOutcome(...)``
        construction in the tree untouched — the same shape ``superseded_by``
        (merge_types.py:949) and ``disposition`` (:955) use.
        """
        from orchestrator.merge_types import MergeOutcome

        assert MergeOutcome(status='done').landed_via_chain is None
        assert MergeOutcome(status='done', landed_via_chain=1).landed_via_chain == 1

    async def test_chain_landing_puts_the_key_in_the_finalized_payload(
        self, tmp_path: Path,
    ) -> None:
        """(b) A landed_via_chain outcome reaches the durable payload."""
        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import enqueue_merge_request
        from orchestrator.merge_types import MergeOutcome

        config = self._config(tmp_path)
        db_path = tmp_path / 'runs.db'
        event_store = EventStore(db_path, 'run-delta-step1')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req = _make_req('101', '101', config, tmp_path)

        await enqueue_merge_request(queue, req, event_store)
        req.result.set_result(
            MergeOutcome(status='done', merge_sha='abc123', landed_via_chain=1),
        )
        await asyncio.sleep(0)  # yield so the done-callback runs

        rows = _finalized_rows(db_path)
        assert len(rows) == 1
        assert rows[0]['landed_via_chain'] == 1
        assert rows[0]['state'] == 'done'
        assert rows[0]['merge_sha'] == 'abc123'

    async def test_ordinary_landing_is_dropped_by_the_canary_filter(
        self, tmp_path: Path,
    ) -> None:
        """(c) A non-chain landing leaves the key absent-or-None.

        This is the "iff landed by a chain walk" half of the PRD contract, and
        it is asserted THROUGH the shipped predicate's own filter rather than
        against the raw payload: η1 keeps a row only when
        ``isinstance(d.get('landed_via_chain'), int) and >= 1``, so an ordinary
        sequential landing must fall out of that filter — otherwise every
        non-deep merge would inflate ``items_per`` and the deploy signal would
        read a chain that never happened.
        """
        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import enqueue_merge_request
        from orchestrator.merge_types import MergeOutcome

        config = self._config(tmp_path)
        db_path = tmp_path / 'runs.db'
        event_store = EventStore(db_path, 'run-delta-step1-plain')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req = _make_req('102', '102', config, tmp_path)

        await enqueue_merge_request(queue, req, event_store)
        req.result.set_result(MergeOutcome(status='done', merge_sha='def456'))
        await asyncio.sleep(0)

        rows = _finalized_rows(db_path)
        assert len(rows) == 1
        assert rows[0].get('landed_via_chain') is None

        # Through the shipped filter: one deep verify, zero chain-landed items.
        items_per = _canary_predicate_items_per(
            [{'chain_items': 3, 'passed': True}], rows,
        )
        assert items_per == 0.0

    async def test_preexisting_payload_keys_are_unchanged(
        self, tmp_path: Path,
    ) -> None:
        """(d) The eight pre-existing keys keep their names and values.

        Adding a key to the single emit site must not rename, drop or reorder
        what is already there: every existing `merge_finalized` reader (the
        dashboard, the reconciler, η1's own `state`/`merge_sha` reads) keys off
        these names.
        """
        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import enqueue_merge_request
        from orchestrator.merge_types import MergeOutcome

        config = self._config(tmp_path)
        db_path = tmp_path / 'runs.db'
        event_store = EventStore(db_path, 'run-delta-step1-keys')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req = _make_req('103', '103', config, tmp_path)

        await enqueue_merge_request(queue, req, event_store)
        req.result.set_result(MergeOutcome(
            status='blocked', reason='nope', merge_sha=None,
            superseded_by='req-xyz',
        ))
        await asyncio.sleep(0)

        rows = _finalized_rows(db_path)
        assert len(rows) == 1
        assert set(rows[0]) == {
            'request_id', 'branch', 'state', 'snapshot_tip', 'merge_sha',
            'superseded_by', 'generation', 'reason', 'landed_via_chain',
        }
        assert rows[0]['request_id'] == req.request_id
        assert rows[0]['branch'] == '103'
        assert rows[0]['state'] == 'blocked'
        assert rows[0]['snapshot_tip'] == req.snapshot_tip
        assert rows[0]['merge_sha'] is None
        assert rows[0]['superseded_by'] == 'req-xyz'
        assert rows[0]['generation'] == req.generation
        assert rows[0]['reason'] == 'nope'

    async def test_terminal_outcome_record_mirrors_the_payload(
        self, tmp_path: Path,
    ) -> None:
        """The hot tier stays in lockstep with the durable tier.

        ``TerminalOutcomeRecord`` (merge_types.py:146) is the in-memory mirror
        of exactly the payload the emit site writes, and the ring is documented
        as LOSSLESS on eviction precisely because an evicted id falls through to
        the `merge_finalized` row.  A hot-tier record missing a field the
        durable row carries would silently break that equivalence, so the new
        field rides both or neither.
        """
        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import enqueue_merge_request
        from orchestrator.merge_types import MergeOutcome, TerminalOutcomeRetention

        config = self._config(tmp_path)
        db_path = tmp_path / 'runs.db'
        event_store = EventStore(db_path, 'run-delta-step1-ring')
        retention = TerminalOutcomeRetention()
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req = _make_req('104', '104', config, tmp_path)

        await enqueue_merge_request(
            queue, req, event_store, retention=retention,
        )
        req.result.set_result(
            MergeOutcome(status='done', merge_sha='cafe01', landed_via_chain=1),
        )
        await asyncio.sleep(0)

        record = retention.get(req.request_id)
        assert record is not None
        assert record.landed_via_chain == 1


# ═══════════════════════════════════════════════════════════════════════════
# step-03: RED — the tip-pass ADOPTION signal out of `_run_inflight_verify`
#
# γ's "THE NON-ADOPTING EXIT" (merge_queue.py:18603-18663) sends BOTH arms
# through `_requeue_request` + the REQUEUED sentinel.  δ replaces ONLY the PASS
# arm: the tip's verdict is about the CUMULATIVE tree, which is a verified
# SUPERSET of every prefix member, so a green tip licenses landing the whole
# prefix — while a tip FAIL (which proves nothing about any individual member)
# and a chain-arm EXCEPTION (which proves nothing about anyone at all) both
# stay exactly as γ shipped them.  That asymmetry is the whole of δ's licence,
# and merge_queue.py:18623 states it in as many words.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestTipPassAdoptionSignal:
    """The adopting exit: a PASS-shaped result the finalize half can walk."""

    async def _scene(
        self, git_repo: Path, monkeypatch, *, passed: bool = True, raises=None,
    ):
        """Drive one deep tip verify and return the scene.

        Mirrors γ's `TestDeepTipVerifyNeverAdopts._fixture` so the two modules'
        claims are about the SAME scene and the narrowing in that class can be
        read against this one.  ``advance_main`` is left un-spied here: this
        class is about the SIGNAL `_run_inflight_verify` returns, not about
        what the finalize half then does with it (step-05 owns that).
        """
        git_ops = _make_git_ops(git_repo, size=2)
        config = _make_config(git_repo, chain_cap=6)
        await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n')
        for tid, fn in (('102', 'b.txt'), ('103', 'c.txt')):
            await _create_branch_editing(git_repo, f'task/{tid}', fn, f'edit-{tid}\n')
        head = await _merge_commit_off_main(git_repo, 'task/101', '101')
        worker = _make_worker(git_ops)
        worker._lane_buffers['normal'].extend(
            _make_req(tid, tid, config, git_repo) for tid in ('102', '103')
        )
        store = _CapturingEventStore()
        worker._event_store = store
        item = _make_item(
            _make_req('101', '101', config, git_repo), head,
            _ephemeral_merge_wt(git_ops, 'adopt'),
        )
        chain = await worker._deep_chain_placement(item)
        assert chain is not None and len(chain.links) == 2

        _spy_post_merge_verify(
            monkeypatch,
            outcome=None if passed else _fail_verify_result(),
            raises=raises,
        )
        releases = _spy_chain_lane_release(monkeypatch)
        queued = list(worker._lane_buffers['normal'])

        res = await worker._run_inflight_verify(item, _local_lease(), chain=chain)
        return git_ops, worker, item, chain, res, store, queued, releases

    async def test_tip_pass_returns_an_adopting_result(
        self, git_repo: Path, monkeypatch,
    ):
        """(a) PASS-shaped, not REQUEUED — and the request is NOT re-queued.

        `_finalize_inflight` routes on exactly two things: the sentinel (checked
        ABOVE the VERIFYING -> FINALIZING hop) and `vr.outcome`.  An adopting
        exit must therefore present as an ordinary pass — `outcome is None` AND
        `status is None` — or the walk's own arm is unreachable.

        The negative half matters just as much: a request that is BOTH queued
        and landed is the double-land hazard `_requeue_request`'s three effects
        exist to make impossible.  So the queue must stay empty and the
        request-liveness ledger must show no requeue.
        """
        _g, worker, item, _chain, res, _store, _q, _rel = await self._scene(
            git_repo, monkeypatch, passed=True,
        )

        assert res.outcome is None, 'a green tip renders no failure outcome'
        assert res.status is None, (
            'the REQUEUED sentinel is checked above the FINALIZING hop, so an '
            'adopting exit must not carry it'
        )
        assert res.merge_wt is not None, (
            "_finalize_inflight's PASS arm asserts merge_wt is not None and "
            'threads it into advance_main'
        )
        assert worker._queue.qsize() == 0, 'an adopted item is not re-queued'
        assert not item.request.result.done(), (
            'the verify half never resolves the future — the finalize half does'
        )

    async def test_the_adopting_result_never_carries_the_chain_lane(
        self, git_repo: Path, monkeypatch,
    ):
        """The lane is already RELEASED by the time the finalize half runs.

        merge_types.py:1458-1464 states it for `InflightEntry.chain`, and it is
        equally true of the result: `_release_chain_lane()` fires in
        `_run_inflight_verify`'s own `finally`, so handing the lane onward would
        give `_finalize_inflight` a POOL-OWNED path it would then dispose a
        second time — permanently losing a `_spec-` slot, the exact hazard
        `_release_or_cleanup`'s "WHICH ONE DO I CALL?" rule warns about.
        """
        _g, _w, _item, chain, res, _store, _q, _rel = await self._scene(
            git_repo, monkeypatch, passed=True,
        )

        assert res.merge_wt != chain.lane
        assert res.spec_warm is False, (
            'a non-lane worktree must not be routed as a warm _spec- lane'
        )

    async def test_the_chain_lane_is_released_exactly_once_on_the_adopting_exit(
        self, git_repo: Path, monkeypatch,
    ):
        """(e) δ adds a FOURTH exit to the three γ shipped; the latch still holds.

        ChainResult's "Lane ownership" contract is exactly-once, and putting
        the release in the `finally` is what makes "every exit" structural.  An
        adopting exit that skipped it would strand the lane checked out
        forever; one that doubled it would hand the same slot to two builders.
        """
        git_ops, _w, _item, chain, _res, _store, _q, releases = await self._scene(
            git_repo, monkeypatch, passed=True,
        )

        assert len(releases) == 1
        assert releases[0][0] == chain.lane
        from orchestrator.warm_lane_pool import LaneState

        pool = git_ops.spec_warm_lane_pool
        assert pool is not None
        assert all(st is LaneState.FREE for st in pool._lanes.values()), (
            'the lane must genuinely be back in the pool, not merely counted'
        )

    async def test_tip_pass_still_feeds_the_halving_reset_exactly_once(
        self, git_repo: Path, monkeypatch,
    ):
        """(b) `_note_chain_outcome(True, 1 + len(chain.links))` is unchanged.

        The halving state machine is fed the BUILT depth in chain-item units;
        adoption must not disturb that feed, or a green deep round would stop
        resetting the bisector and the walk would ratchet downward forever.
        """
        notes: list[tuple] = []

        def _install(worker: SpeculativeMergeWorker) -> None:
            """Wrap `_note_chain_outcome` PASSTHROUGH on the worker instance.

            Instance-level, not a module patch: the halving state must really
            move, so the reset assertion below stays a fact about production
            code rather than about the spy.
            """
            real_note = worker._note_chain_outcome

            def _recording(passed_: bool, depth: int) -> None:
                notes.append((passed_, depth))
                real_note(passed_, depth)

            worker._note_chain_outcome = _recording  # type: ignore[method-assign]

        git_ops = _make_git_ops(git_repo, size=2)
        config = _make_config(git_repo, chain_cap=6)
        await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n')
        for tid, fn in (('102', 'b.txt'), ('103', 'c.txt')):
            await _create_branch_editing(git_repo, f'task/{tid}', fn, f'edit-{tid}\n')
        head = await _merge_commit_off_main(git_repo, 'task/101', '101')
        worker = _make_worker(git_ops)
        worker._lane_buffers['normal'].extend(
            _make_req(tid, tid, config, git_repo) for tid in ('102', '103')
        )
        worker._event_store = _CapturingEventStore()
        item = _make_item(
            _make_req('101', '101', config, git_repo), head,
            _ephemeral_merge_wt(git_ops, 'halving'),
        )
        chain = await worker._deep_chain_placement(item)
        assert chain is not None and len(chain.links) == 2
        _install(worker)
        _spy_post_merge_verify(monkeypatch, outcome=None)

        await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert notes == [(True, 3)], '1 dispatching item + 2 links'
        assert worker._chain_halving_state is None, 'a pass resets the bisector'

    async def test_tip_fail_arm_is_byte_identical_to_gamma(
        self, git_repo: Path, monkeypatch,
    ):
        """(c) A red tip still defers: REQUEUED, queued, no outcome, no event.

        δ's licence is one-directional.  A tip FAIL says the cumulative tree is
        red; it does NOT identify WHICH member broke it, so terminally failing
        any of them would be the same unsound attribution pointing the other
        way — and would feed `consecutive_merge_thrash` a deterministic
        signature on every retry.
        """
        from orchestrator.merge_types import InflightStatus

        _g, worker, item, _chain, res, store, queued, releases = await self._scene(
            git_repo, monkeypatch, passed=False,
        )

        assert res.status == InflightStatus.REQUEUED
        assert res.outcome is None, 'a defer renders no MergeOutcome at all'
        assert res.merge_wt is None, 'the fail arm disposes its own worktrees'
        assert not item.request.result.done()
        assert worker._queue.qsize() == 1
        assert worker._queue.get_nowait() is item.request
        assert list(worker._lane_buffers['normal']) == queued, 'same items, same order'
        assert all(not r.result.done() for r in queued)
        assert store.events_of(EventType.merge_attempt) == []
        assert worker._chain_halving_state == 1, '3 built items -> max(1, 3 // 2)'
        assert len(releases) == 1

    async def test_chain_arm_exception_stays_non_adopting(
        self, git_repo: Path, monkeypatch,
    ):
        """(d) An infra ERROR is not a tip verdict — merge_queue.py:18623.

        A green tip is a verified SUPERSET of every prefix member, which is
        what licenses δ's adoption.  An exception proves nothing about anyone,
        so the exception arm keeps γ's recipe verbatim: suppress the next
        chain for this task, requeue, land nothing.
        """
        from orchestrator.merge_types import InflightStatus

        _g, worker, item, _chain, res, store, queued, releases = await self._scene(
            git_repo, monkeypatch, passed=True,
            raises=RuntimeError('verify infra exploded'),
        )

        assert res.status == InflightStatus.REQUEUED
        assert res.outcome is None
        assert not item.request.result.done()
        assert worker._queue.qsize() == 1
        assert worker._queue.get_nowait() is item.request
        assert '101' in worker._chain_error_suppressed, (
            'the one-shot suppression is what makes an infra blip cost exactly '
            'one non-chained round'
        )
        assert list(worker._lane_buffers['normal']) == queued
        assert store.events_of(EventType.merge_attempt) == []
        assert len(releases) == 1


# ═══════════════════════════════════════════════════════════════════════════
# step-05: RED — the in-order CAS walk over `entry.chain.links`
#
# γ's `_DeepScene` deliberately stops short of finalize (it lands nothing, so
# there is nothing to finalize); this is the scene extended PAST
# `_finalize_inflight`, which is where δ's walk lives.  The head (chain item #1)
# is landed by the PASS arm that already exists; I2..Ik are landed by the walk
# δ appends to it, each through the SAME terminal trio
# (`_journal_landed_then_advance` → `_finalize_advanced_merge` →
# `_resolve_or_drop_abandoned`).
#
# Five properties, one per plan row:
#   (a) `advance_main` runs once per chain item, in LAND order
#   (b) each call's `expected_main` is its PREDECESSOR's merge commit, so main's
#       history is linear and no rebase fires
#   (c) every landed item resolves 'done' carrying `landed_via_chain`, and the
#       SHIPPED canary arithmetic over the resulting events computes exactly
#       items-landed-per-deep-verify
#   (d) each landed link leaves the queue AND reaches TERMINAL
#   (e) `chain.truncated_at` gets no outcome, no event, and stays queued
# ═══════════════════════════════════════════════════════════════════════════


_DELTA_LINKS = ('102', '103', '104')
"""The three followers that chain CLEANLY onto the head — links I2..I4."""

_DELTA_TRUNCATOR = '105'
"""The follower that CONFLICTS with 102, so ``build_chain`` truncates on it.

Decision #4 (merge_types.py:1218-1223): a chain conflict at position j may be a
conflict with an *unlanded* predecessor, so item j is NOT genuinely conflicted
and must take its ordinary sequential path — which is why the walk owes it no
outcome and no event.  Making the truncator conflict with a LINK (102's
shared.txt line 5), not with the head, is what makes that hazard real here.
"""


def _events_for_task(db_path: Path, task_id: str) -> list[str]:
    """Return every ``event_type`` recorded against *task_id*, in order."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            'SELECT event_type FROM events WHERE task_id = ? ORDER BY rowid',
            (task_id,),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


async def _prefix_scene_upto_finalize(
    git_repo: Path, tmp_path: Path, monkeypatch, *,
    db_name: str = 'delta-walk.db', advance_hook=None, verify_outcome=None,
) -> dict:
    """Build a 4-item clean chain with a green tip, stopping just BEFORE
    ``_finalize_inflight``, and return everything the assertions read.

    Shared by every δ landing scenario — full landing, stale-CAS abort and
    typed-lease defer alike — so each differs ONLY in what it interposes and
    in what it does with the returned ``entry``.  *advance_hook* is forwarded
    to :func:`_spy_advance_main`, which is how a scenario makes main move (or
    a synthetic :class:`AdvanceOutcome` appear) partway down the walk.

    *verify_outcome* is ``_run_post_merge_verify``'s return value in
    :func:`_spy_post_merge_verify`'s vocabulary — ``None`` is a PASS (a GREEN
    tip, the adopting arm), a :class:`VerifyResult` is a FAIL.

    ``advance_main`` is NOT stubbed — it is spied PASSTHROUGH, so main really
    moves and the recorded ``expected_main`` chain can be checked against real
    history.
    """
    from orchestrator.event_store import EventStore
    from orchestrator.merge_queue import enqueue_merge_request
    from orchestrator.merge_types import InflightEntry

    git_ops = _make_git_ops(git_repo, size=2)
    config = _make_config(git_repo, chain_cap=6)
    await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n')
    await _create_branch_editing(
        git_repo, 'task/102', 'shared.txt', _shared_txt_with(5, 'from-102'),
    )
    await _create_branch_editing(git_repo, 'task/103', 'c.txt', 'edit-103\n')
    await _create_branch_editing(git_repo, 'task/104', 'd.txt', 'edit-104\n')
    await _create_branch_editing(
        git_repo, 'task/105', 'shared.txt', _shared_txt_with(5, 'from-105'),
    )
    head_mc = await _merge_commit_off_main(git_repo, 'task/101', '101')
    main_sha = await _rev_parse(git_repo, 'main')

    db_path = tmp_path / db_name
    store = EventStore(db_path, 'run-delta-walk')
    worker = _make_worker(git_ops)
    worker._event_store = store

    # Every request goes on through the REAL enqueue chokepoint: that is
    # what registers `_on_finalized`, and `merge_finalized` has no other
    # emit site (merge_queue.py:4763-4777).  Draining them into the lane
    # buffers is likewise the real path — it is what registers each id at
    # LANE_BUFFERED, so the "reaches TERMINAL" assertion below is a claim
    # about a genuinely-tracked request rather than an unregistered one.
    reqs: dict[str, MergeRequest] = {}
    for tid in ('101', *_DELTA_LINKS, _DELTA_TRUNCATOR):
        reqs[tid] = _make_req(tid, tid, config, git_repo)
        await enqueue_merge_request(worker._queue, reqs[tid], store)
    worker._drain_queue_into_lanes()
    popped = worker._pop_next_pickable()
    assert popped is reqs['101'], 'the head must be the first pickable'

    item = RealMergeItem(
        request=reqs['101'],
        merge_result=MergeResult(
            success=True, merge_commit=head_mc,
            merge_worktree=_ephemeral_merge_wt(git_ops, 'walk'),
        ),
        merge_wt=_ephemeral_merge_wt(git_ops, 'walk'),
        base_sha=main_sha,          # the head CASes against REAL main
        speculative=True,
    )
    permits_before_build = _permit_census(worker)
    chain = await worker._deep_chain_placement(item)
    permits_after_build = _permit_census(worker)
    assert chain is not None
    assert [tid for tid, _ in chain.links] == list(_DELTA_LINKS)
    assert chain.truncated_at == _DELTA_TRUNCATOR
    assert chain.truncated_reason == 'conflict'

    _spy_post_merge_verify(monkeypatch, outcome=verify_outcome)
    # Installed AFTER `_deep_chain_placement`, deliberately: the lane the chain
    # build acquired is held across the verify and returned on the way OUT of
    # `_run_inflight_verify` (its `finally`), so a spy armed here sees exactly
    # the releases attributable to the EXIT — which is the thing δ adds a
    # fourth branch to.
    lane_releases = _spy_chain_lane_release(monkeypatch)
    # BOTH post-advance gates run FOR REAL here, deliberately: they are
    # part of what the walk must reuse per link, and stubbing them would
    # have to reach back through `orchestrator.merge_queue.<private>` —
    # the exact patch surface test_merge_queue_reachback_patch_guard.py
    # freezes.  The pyright gate quick-exits clean because no module here
    # defines a `type_check_command` (merge_gates.py:2105-2107), and the
    # equivalence gate passes because every commit in this scene is a
    # genuine `--no-ff` merge of the branch it names.
    adv = _spy_advance_main(git_ops, monkeypatch, hook=advance_hook)

    res = await worker._run_inflight_verify(item, _local_lease(), chain=chain)
    done: asyncio.Future = asyncio.get_running_loop().create_future()
    done.set_result(res)
    entry = InflightEntry(
        item=item, lease=_local_lease(), verify_task=done,  # type: ignore[arg-type]
        merge_wt=res.merge_wt, was_speculative=True, chain=chain,
    )
    return {
        'git_ops': git_ops, 'worker': worker, 'chain': chain, 'reqs': reqs,
        'head_mc': head_mc, 'main_sha': main_sha, 'adv': adv,
        'db_path': db_path, 'repo': git_repo, 'entry': entry, 'item': item,
        'lane_releases': lane_releases,
        'permits_before_build': permits_before_build,
        'permits_after_build': permits_after_build,
        'permits_after_verify': _permit_census(worker),
    }



@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestInOrderCasWalk:
    """δ's walk: the whole verified PREFIX lands, in order, by CAS."""

    async def _scene(self, git_repo: Path, tmp_path: Path, monkeypatch):
        """The clean 4-item chain, driven all the way THROUGH the finalize half.

        :func:`_prefix_scene_upto_finalize` owns the build; this adds only the
        ``_finalize_inflight`` call the walk hangs off, plus the event-loop
        turn that lets every ``_on_finalized`` done-callback run.
        """
        s = await _prefix_scene_upto_finalize(
            git_repo, tmp_path, monkeypatch, db_name='delta-walk.db',
        )
        s['advanced'] = await s['worker']._finalize_inflight(s['entry'])
        await asyncio.sleep(0)  # let every `_on_finalized` done-callback run
        return s

    async def test_advance_main_runs_once_per_chain_item_in_land_order(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a) One CAS per chain item, in LAND order — head first, then links.

        `chain.links` is a CONTIGUOUS PREFIX in land order precisely so this
        walk has no hole (build_chain's decision-4 purity; `chain_snapshot`
        refuses clique-minimality for the same reason).  Landing them out of
        order — or skipping one — would break the in-order/frozen-prefix
        invariant every downstream base_sha is computed against.
        """
        s = await self._scene(git_repo, tmp_path, monkeypatch)

        assert s['advanced'] is True
        assert [c[0] for c in s['adv']] == [
            s['head_mc'], *(mc for _tid, mc in s['chain'].links)
        ]

    async def test_each_expected_main_is_its_predecessors_merge_commit(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(b) The CAS chain is linear: main ← I1 ← I2 ← … ← Ik.

        `_frozen_base_chain` (merge_queue.py:11675) already encodes the
        chained-base property; this asserts the WALK honours it, so no rebase
        can fire (a rebase would rewrite a link the tip verdict was rendered
        against, voiding the very evidence the walk is landing on).
        """
        s = await self._scene(git_repo, tmp_path, monkeypatch)
        link_shas = [mc for _tid, mc in s['chain'].links]

        assert [c[1] for c in s['adv']] == [
            s['main_sha'], s['head_mc'], *link_shas[:-1]
        ]
        assert await _rev_parse(s['repo'], 'main') == s['chain'].tip
        # Linear by construction, not merely by the recorded arguments: every
        # landed sha is an ancestor of main, in order.
        for sha in [s['head_mc'], *link_shas]:
            rc, _, _ = await _run(
                ['git', 'merge-base', '--is-ancestor', sha, 'main'],
                cwd=s['repo'],
            )
            assert rc == 0, f'{sha[:8]} is not an ancestor of the landed main'

    async def test_every_landed_item_resolves_done_with_landed_via_chain(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(c) The whole prefix resolves 'done', and the SHIPPED canary agrees.

        The canary arithmetic is what PINS the encoding: `items_per` must come
        out as items-landed-divided-by-deep-verifies.  One deep verify of four
        chain items that lands all four must read 4.0 — a per-item constant k
        would read 16.0 and 1-indexed positions 10.0.
        """
        s = await self._scene(git_repo, tmp_path, monkeypatch)
        landed_ids = ['101', *_DELTA_LINKS]

        for tid in landed_ids:
            req = s['reqs'][tid]
            assert req.result.done(), f'task {tid} never resolved'
            outcome = req.result.result()
            assert outcome.status == 'done', f'task {tid}: {outcome.status}'
            assert outcome.landed_via_chain == 1, (
                f'task {tid} must carry its own 1 — the walk contributes k'
            )

        rows = _finalized_rows(s['db_path'])
        items_per = _canary_predicate_items_per(
            # ONE deep verify, of exactly this chain.  Synthesised rather than
            # read off a real `merge_verify` row because the emission of
            # `chain_items` is γ's claim, already pinned at
            # test_merge_queue_deep_dispatch.py; δ's claim is the NUMERATOR.
            [{'chain_items': 1 + len(s['chain'].links), 'passed': True}], rows,
        )
        assert items_per == float(len(landed_ids)), (
            'the shipped predicate must read "items landed per deep verify run"'
        )

    async def test_each_landed_link_leaves_the_queue_and_reaches_terminal(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(d) No request is BOTH landed and still queued.

        A link that lands but stays in its lane buffer would be re-picked by
        the merger and merged onto main a second time — the double-land hazard
        `_requeue_request`'s three effects exist to make impossible on the
        non-adopting arms.  Retirement is the registry half of the same claim.
        """
        from orchestrator.merge_types import ItemLifecycleState

        s = await self._scene(git_repo, tmp_path, monkeypatch)
        worker = s['worker']
        still_queued = {
            r.task_id for lane in ('high', 'normal')
            for r in worker._lane_buffers[lane]
        }

        assert still_queued == {_DELTA_TRUNCATOR}
        assert worker._queue.qsize() == 0
        for tid in ('101', *_DELTA_LINKS):
            rid = s['reqs'][tid].request_id
            current = worker._lifecycle.current(rid)
            assert current in (None, ItemLifecycleState.TERMINAL), (
                f'task {tid} left at {current!r} after landing'
            )

    async def test_the_truncated_item_gets_no_outcome_and_no_event(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(e) ChainResult decision #4: the truncator is UNTOUCHED.

        Its conflict was with an UNLANDED predecessor (102's shared.txt edit),
        so it is not genuinely conflicted — resolving it here would render a
        verdict no verify ever produced, and would feed `merge_attempt`
        (and thence `consecutive_merge_thrash`) a deterministic false signature
        on every deep round.
        """
        from orchestrator.merge_types import ItemLifecycleState

        s = await self._scene(git_repo, tmp_path, monkeypatch)
        trunc = s['reqs'][_DELTA_TRUNCATOR]

        assert not trunc.result.done(), 'the truncator renders no outcome'
        assert _events_for_task(s['db_path'], _DELTA_TRUNCATOR) == ['merge_queued'], (
            'only its own enqueue event — the walk emits nothing for it'
        )
        assert trunc in list(s['worker']._lane_buffers['normal'])
        assert s['worker']._lifecycle.current(trunc.request_id) == (
            ItemLifecycleState.LANE_BUFFERED
        ), 'it stays queued for its ordinary sequential path'


# ═══════════════════════════════════════════════════════════════════════════
# step-07 RED — stale-CAS abort (PRD decision #9) + 3003 DEFER inheritance
#
#   (a) an advance that does not return 'advanced' ABORTS the walk; the
#       already-landed prefix stays landed and every unlanded link is left
#       exactly as it was — still buffered, future unresolved, non-terminal
#   (b) no unlanded link renders a `MergeOutcome('blocked')` and none emits a
#       `merge_attempt`, so `consecutive_merge_thrash` cannot be fed
#   (c) a typed `MergeVerifyLeaseContended` / `MergeVerifyLeaseHeld` reaching
#       the walk inherits 3003's DEFER classification — streak bookkeeping
#       moves, and it NEVER becomes the bare-RuntimeError -> 'blocked' path
#   (d) the walk aborts on the FIRST failure — it never skips a link and
#       carries on with later ones (that would hole the contiguous prefix)
# ═══════════════════════════════════════════════════════════════════════════


async def _external_main_bump(repo: Path) -> str:
    """Land an unrelated commit directly on main; return its SHA.

    Written with ``commit-tree`` + ``update-ref`` rather than a checkout and a
    ``git commit`` so the bump does NOT disturb the working tree — by the time
    a scenario calls this, ``advance_main`` has already moved ``refs/heads/main``
    out from under the checkout, and a plumbing bump is the only way to model
    "another writer got there first" without also rewriting that state.

    The new commit carries main's CURRENT tree, so the bump is a pure ref move:
    the resulting abort is attributable to the moved ref alone, never to a
    content conflict the next link would have hit anyway.
    """
    cur = await _rev_parse(repo, 'main')
    tree = await _rev_parse(repo, 'main^{tree}')
    _, new, _ = await _run(
        ['git', 'commit-tree', tree, '-p', cur, '-m', 'external writer'], cwd=repo,
    )
    new = new.strip()
    await _run(['git', 'update-ref', 'refs/heads/main', new, cur], cwd=repo)
    return new


def _abort_hook_at(call_no: int, *, outcome=None, raises=None, bump_repo=None):
    """Build an :func:`_spy_advance_main` hook that disrupts call *call_no*.

    Exactly one of *outcome* (return a synthetic
    :class:`~orchestrator.git_ops.AdvanceOutcome` instead of advancing),
    *raises*, or *bump_repo* (move main for real, then fall through to the
    real ``advance_main``, which then refuses).
    """
    async def _hook(n: int):
        if n != call_no:
            return None
        if bump_repo is not None:
            await _external_main_bump(bump_repo)
            return None
        if raises is not None:
            raise raises
        return outcome

    return _hook


def _lane_of(worker: SpeculativeMergeWorker, task_id: str) -> str | None:
    """Return the lane whose buffer currently holds *task_id*, or None."""
    for lane in ('high', 'normal'):
        if any(r.task_id == task_id for r in worker._lane_buffers[lane]):
            return lane
    return None


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestStaleCasAbortLeavesTheRestAlone:
    """PRD decision #9: the walk ABORTS, it never FAILS anyone.

    ``advance_main`` returns an :class:`AdvanceOutcome` rather than raising
    (git_ops.py:13151), so a lost race is a VALUE the walk must read — and the
    only correct response is to stop.  Every link past the abort point was
    verified only as part of the cumulative tip; re-anchoring one onto the
    moved main would land a tree nothing verified, the same false-green the
    SOUNDNESS RULE at merge_queue.py:18606-18614 fences off.
    """

    async def _aborting_scene(
        self, git_repo: Path, tmp_path: Path, monkeypatch, hook, *, db_name: str,
    ):
        s = await _prefix_scene_upto_finalize(
            git_repo, tmp_path, monkeypatch, db_name=db_name, advance_hook=hook,
        )
        s['advanced'] = await s['worker']._finalize_inflight(s['entry'])
        await asyncio.sleep(0)
        return s

    async def test_external_main_move_aborts_and_keeps_the_landed_prefix(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a) Main moves under the walk → it stops, and what landed stays landed.

        The REAL-git shape of the race.  Note WHICH code the abort arrives as:
        the walk passes ``merge_wt=None`` deliberately (a link's tree is the one
        the tip verdict covered, so opting out of ``advance_main``'s rebase
        retry is the point), and with no worktree to rebase in, ``advance_main``
        refuses at its descendant check with ``'not_descendant'`` rather than
        reaching the CAS at all.  Decision #9 is about the FAMILY — "the first
        result that is not ``'advanced'``" — not about one code.
        """
        s = await self._aborting_scene(
            git_repo, tmp_path, monkeypatch,
            _abort_hook_at(3, bump_repo=git_repo), db_name='delta-stale.db',
        )
        worker, reqs = s['worker'], s['reqs']

        # head + link '102' landed; the walk stopped trying at '103'.
        assert [c[0] for c in s['adv']] == [
            s['head_mc'], s['chain'].links[0][1], s['chain'].links[1][1],
        ]
        for tid in ('101', '102'):
            assert reqs[tid].result.done(), f'task {tid} should have landed'
            assert reqs[tid].result.result().status == 'done'
        for sha in (s['head_mc'], s['chain'].links[0][1]):
            rc, _, _ = await _run(
                ['git', 'merge-base', '--is-ancestor', sha, 'main'], cwd=git_repo,
            )
            assert rc == 0, f'{sha[:8]} un-landed by the abort'
        assert not set(worker._lifecycle.non_terminal_items()) & {
            reqs[t].request_id for t in ('101', '102')
        }

    async def test_unlanded_links_are_left_exactly_as_they_were(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a) "Untouched" means buffered, unresolved, and non-terminal.

        Not merely "not failed": the link must go back to the SAME lane
        position it was taken from, because that position is its place in
        submission order for the next round's ``chain_snapshot``.
        """
        from orchestrator.git_ops import AdvanceOutcome
        from orchestrator.merge_types import ItemLifecycleState

        s = await self._aborting_scene(
            git_repo, tmp_path, monkeypatch,
            _abort_hook_at(3, outcome=AdvanceOutcome('cas_failed')),
            db_name='delta-cas.db',
        )
        worker, reqs = s['worker'], s['reqs']

        for tid in ('103', '104', _DELTA_TRUNCATOR):
            req = reqs[tid]
            assert not req.result.done(), f'task {tid} must render NO outcome'
            assert _lane_of(worker, tid) is not None, f'task {tid} left its buffer'
            assert worker._lifecycle.current(req.request_id) == (
                ItemLifecycleState.LANE_BUFFERED
            ), f'task {tid} left LANE_BUFFERED'
        # Submission order preserved inside the lane, so the next round's
        # chain_snapshot sees the same prefix it would have seen without δ.
        assert [
            r.task_id for r in worker._lane_buffers['normal']
        ] == ['103', '104', _DELTA_TRUNCATOR]

    async def test_the_walk_stops_at_the_first_failure(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(d) No skip-and-continue — that would hole the contiguous prefix.

        ``links`` is contiguous in LAND order precisely so each member CASes
        against its predecessor (merge_queue.py:6376-6379).  Skipping a failed
        link and landing the next one would CAS it against a commit that never
        reached main.
        """
        from orchestrator.git_ops import AdvanceOutcome

        s = await self._aborting_scene(
            git_repo, tmp_path, monkeypatch,
            _abort_hook_at(3, outcome=AdvanceOutcome('cas_failed')),
            db_name='delta-firstfail.db',
        )

        assert len(s['adv']) == 3, (
            f'the walk kept going past the failure: {[c[0][:8] for c in s["adv"]]}'
        )
        assert s['chain'].links[2][1] not in [c[0] for c in s['adv']]

    async def test_the_abort_feeds_the_thrash_ladder_nothing(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(b) Zero 'blocked' outcomes and zero ``merge_attempt`` rows.

        A rendered failure would give ``workflow.py``'s
        ``consecutive_merge_thrash`` ladder a DETERMINISTIC
        ``merge_outcome_signature`` on every deep round — a false-positive
        human escalation for a race that resolves itself on the next round.
        """
        from orchestrator.git_ops import AdvanceOutcome

        s = await self._aborting_scene(
            git_repo, tmp_path, monkeypatch,
            _abort_hook_at(3, outcome=AdvanceOutcome('cas_failed')),
            db_name='delta-thrash.db',
        )

        for tid in ('103', '104', _DELTA_TRUNCATOR):
            assert _events_for_task(s['db_path'], tid) == ['merge_queued'], (
                f'task {tid} emitted more than its own enqueue event'
            )

    async def test_two_consecutive_tip_fails_render_nothing_for_any_link(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(b) The same silence across two RED tips in a row.

        Two rounds is the interesting number: ``max_consecutive_merge_thrash``
        defaults to 2, so a pair of identical rendered failures is exactly what
        would trip the ladder.  The chain arm must produce no such pair.
        """
        import dataclasses

        from orchestrator.merge_types import InflightEntry

        s = await _prefix_scene_upto_finalize(
            git_repo, tmp_path, monkeypatch, db_name='delta-tipfail.db',
            verify_outcome=_fail_verify_result(),
        )
        worker, git_ops, reqs = s['worker'], s['git_ops'], s['reqs']
        await worker._finalize_inflight(s['entry'])
        await asyncio.sleep(0)

        # Round 2.  The head was REQUEUED by the fail arm, so it now sits on
        # `_queue` and is deliberately left there — draining it back into a lane
        # would make the head a member of its OWN next chain (chain_snapshot
        # walks lane buffers), which is a different scenario.  Everything the
        # claim is about — the three links — is still buffered where round 1
        # left it, so the rebuild sees exactly the prefix it saw before.
        assert worker._queue.qsize() == 1, 'the fail arm must have requeued the head'
        item2 = dataclasses.replace(
            s['item'], merge_wt=_ephemeral_merge_wt(git_ops, 'tipfail2'),
        )
        chain2 = await worker._deep_chain_placement(item2)
        assert chain2 is not None and chain2.links
        res2 = await worker._run_inflight_verify(item2, _local_lease(), chain=chain2)
        done2: asyncio.Future = asyncio.get_running_loop().create_future()
        done2.set_result(res2)
        await worker._finalize_inflight(InflightEntry(
            item=item2, lease=_local_lease(), verify_task=done2,  # type: ignore[arg-type]
            merge_wt=res2.merge_wt, was_speculative=True, chain=chain2,
        ))
        await asyncio.sleep(0)

        assert not s['adv'], 'a red tip lands NOTHING via the chain'
        for tid in (*_DELTA_LINKS, _DELTA_TRUNCATOR):
            assert not reqs[tid].result.done(), f'task {tid} rendered an outcome'
            assert _events_for_task(s['db_path'], tid) == ['merge_queued'], (
                f'task {tid} emitted a merge_attempt across two red tips'
            )

    async def test_a_halt_worthy_advance_result_still_halts_the_queue(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """A shared main-checkout fault is not a per-link race — halt anyway.

        ``_HALT_ADVANCE_RESULTS`` (merge_queue.py:676) names the codes that
        report a fault affecting EVERY subsequent task, not just this link
        (``unmerged_state``: project_root already has unresolved conflicts).
        Decision #9's silence is about not blaming the LINK; it was never a
        licence to swallow a queue-wide fault the head path halts on.
        """
        from orchestrator.git_ops import AdvanceOutcome

        s = await self._aborting_scene(
            git_repo, tmp_path, monkeypatch,
            _abort_hook_at(3, outcome=AdvanceOutcome('unmerged_state')),
            db_name='delta-halt.db',
        )

        assert s['worker'].is_wip_halted, (
            'a halt-worthy advance result inside the walk left the queue running'
        )
        assert not s['reqs']['103'].result.done(), (
            'the halt must not also render a per-link blocked outcome'
        )


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestContendedLeaseDeferInheritance:
    """3003's DEFER classification reaches the walk too.

    ``MergeVerifyLeaseContended`` / ``MergeVerifyLeaseHeld`` mean "the shared
    merge-verify lane was unavailable, so the raiser refused to act
    UNPROTECTED" — a transient come-back-later, never a verdict.  Mapping one
    to ``MergeOutcome('blocked')`` renders a DETERMINISTIC reason string, hence
    an identical ``merge_outcome_signature`` every time, which is exactly what
    tripped ``consecutive_merge_thrash`` into false-positive human escalations
    before task 3003 (git_ops.py:1673-1683).  The walk must inherit that fix,
    which means catching the typed pair BEFORE any generic ``except Exception``
    — the ordering that IS the opt-in at merge_queue.py:18039-18054.
    """

    async def _deferring_scene(
        self, git_repo: Path, tmp_path: Path, monkeypatch, exc, *, db_name: str,
    ):
        """Raise *exc* out of the THIRD ``push_main`` — link '103''s finalize.

        ``push_main`` is the last seam inside ``_finalize_advanced_merge``
        (merge_gates.py, after the POST_ADVANCE_GATES chain), so raising there
        models the realistic shape: the link's advance already succeeded and
        the lane went unavailable during the post-advance half.  Patched on the
        GitOps INSTANCE, not through ``orchestrator.merge_queue.<private>`` —
        the reach-back surface test_merge_queue_reachback_patch_guard.py
        freezes.  Call 1 is the head's, call 2 is link '102''s.
        """
        s = await _prefix_scene_upto_finalize(
            git_repo, tmp_path, monkeypatch, db_name=db_name,
        )
        git_ops = s['git_ops']
        real_push = git_ops.push_main
        pushes: list[int] = []

        async def _pushing(*a, **kw):
            pushes.append(len(pushes) + 1)
            if len(pushes) == 3:
                raise exc
            return await real_push(*a, **kw)

        monkeypatch.setattr(git_ops, 'push_main', _pushing)
        s['advanced'] = await s['worker']._finalize_inflight(s['entry'])
        await asyncio.sleep(0)
        s['pushes'] = pushes
        return s

    def _contended(self, tmp_path: Path):
        from orchestrator.git_ops import MergeVerifyLeaseContended

        return MergeVerifyLeaseContended(tmp_path / '_merge-verify.lock', 30.0)

    def _held(self, tmp_path: Path):
        from orchestrator.git_ops import MergeVerifyLeaseHeld

        return MergeVerifyLeaseHeld(tmp_path / '_merge-verify', 4242)

    @pytest.mark.parametrize('which', ['contended', 'held'])
    async def test_typed_lease_error_defers_and_moves_the_streak(
        self, which: str, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """Both typed raisers DEFER, and the 3003 streak bookkeeping moves.

        The streak is what BOUNDS the defer: without it a permanently wedged
        lane holder would defer forever with no terminal resolution at all.
        Feeding it from the walk keeps the walk's defers on the same budget the
        head path's are on, rather than inventing a second, unbounded one.
        """
        exc = (
            self._contended(tmp_path) if which == 'contended'
            else self._held(tmp_path)
        )
        s = await self._deferring_scene(
            git_repo, tmp_path, monkeypatch, exc, db_name=f'delta-lease-{which}.db',
        )
        worker = s['worker']

        assert s['pushes'] == [1, 2, 3], 'the walk must stop after the raiser'
        assert worker._contended_lease_requeues.get('103') == 1, (
            'the walk did not feed the contended-lane streak'
        )
        assert '103' in worker._contended_lease_first_defer_at
        assert '103' in worker._contended_lease_last_defer_at
        worker._clear_contended_lease_streak('103')
        assert '103' not in worker._contended_lease_requeues
        assert '103' not in worker._contended_lease_first_defer_at

    @pytest.mark.parametrize('which', ['contended', 'held'])
    async def test_typed_lease_error_never_renders_blocked(
        self, which: str, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """Never the bare-RuntimeError -> ``MergeOutcome('blocked')`` path."""
        from orchestrator.merge_types import ItemLifecycleState

        exc = (
            self._contended(tmp_path) if which == 'contended'
            else self._held(tmp_path)
        )
        s = await self._deferring_scene(
            git_repo, tmp_path, monkeypatch, exc,
            db_name=f'delta-lease-nb-{which}.db',
        )
        worker, reqs = s['worker'], s['reqs']

        for tid in ('103', '104'):
            assert not reqs[tid].result.done(), (
                f'task {tid} rendered an outcome on a transient lane refusal'
            )
            assert _lane_of(worker, tid) is not None
            assert worker._lifecycle.current(reqs[tid].request_id) == (
                ItemLifecycleState.LANE_BUFFERED
            )
        # …and the prefix that DID land is untouched by the defer.
        for tid in ('101', '102'):
            assert reqs[tid].result.result().status == 'done'

    async def test_a_bare_error_is_contained_without_moving_the_streak(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """The generic arm is NOT the typed arm — that distinction is the fix.

        A bare fault is still contained (one bad link may not strand the rest
        of the accounting) and still aborts the walk, but it must not be
        counted as lane contention: doing so would let a genuine git fault
        consume the contended-lane budget and mask the real pathology.
        """
        s = await self._deferring_scene(
            git_repo, tmp_path, monkeypatch, RuntimeError('git exploded'),
            db_name='delta-bare.db',
        )
        worker = s['worker']

        assert '103' not in worker._contended_lease_requeues
        assert '103' not in worker._contended_lease_first_defer_at
        assert not s['reqs']['103'].result.done()
        for tid in ('101', '102'):
            assert s['reqs'][tid].result.result().status == 'done'


# ═══════════════════════════════════════════════════════════════════════════
# step-9 RED — head-verify cancellation with a clean verify-lease release
#
# PRD decision #3 ("tip pass is authoritative for the whole prefix") in its
# sharpest form: slot 1's head I0 is the trust anchor verified against REAL
# main, and its merge commit is the base the speculative slot-2 item was
# stacked on — so the green tip's cumulative tree CONTAINS I0's tree.  δ
# therefore lands I0 on the TIP's authority and cancels its in-flight verify
# rather than waiting for (or obeying) a verdict about a subset it already
# has better evidence for.  Two boundary rows fall out:
#
#   * "Head-fail + tip-pass" — head verify red (flake), tip green ⇒ the FULL
#     prefix lands, head verify cancelled.  A red head resolves no
#     MergeOutcome('blocked'): it is not a verdict δ is entitled to act on.
#   * "Lease released on head-cancel" — 3071's oracle
#     (`warm-lane-lock-guard.sh check`) must read the `_merge-verify` lane
#     IDLE within one round.  BOTH BUSY axes are asserted here, because they
#     are independent and only one of them is what 3071 measures:
#       (a) the kernel flock on `<worktree_base>/_merge-verify.lock`, read via
#           the STRICT reader so an unreadable /proc/locks fails loudly rather
#           than letting a negative assertion pass vacuously (task 3604, the
#           `lane_is_free` precedent at test_lane_lock_leak_guard.py:496);
#       (b) the FIXED-key holder-pgid rendezvous, which
#           `reset_persistent_merge_worktree` reads FAIL-CLOSED — a leaked
#           entry there raises MergeVerifyLeaseHeld and wedges the warm lane.
#
# A LOCAL head is clean by construction: cancelling its task unwinds
# `GitOps.merge_verify_lease`'s finally (git_ops.py:3549-3554), which removes
# the holder pgid AND releases the flock.  A REMOTE head is NOT: the abort
# goes ssh -> `orchestrator cancel-verify` -> `cancel_request` -> SIGKILL,
# which skips cli.py's own finally (:620-623) and leaks the rendezvous file —
# the hazard verify_cancel.py:303-336 documents.  δ closes that by having
# `cli.py cancel_verify` clear the fixed key on its rc==0 path.
# ═══════════════════════════════════════════════════════════════════════════


async def _merge_commit_onto(
    repo: Path, branch: str, base_sha: str, label: str,
) -> str:
    """A REAL ``--no-ff`` merge commit of *branch* onto *base_sha*.

    The speculative-stacking twin of :func:`_merge_commit_off_main`: slot 2's
    item is merged onto the HEAD's merge commit, not onto main, which is
    exactly what makes the head's tree a subset of the chain tip's.
    """
    await _run(['git', 'checkout', '-b', f'_tmp-{label}', base_sha], cwd=repo)
    await _run(['git', 'merge', '--no-ff', '-m', f'merge {branch}', branch], cwd=repo)
    sha = await _rev_parse(repo)
    await _run(['git', 'checkout', 'main'], cwd=repo)
    return sha


def _remote_lease(order: list[str] | None = None):
    """A REMOTE :class:`HostLease` whose ``cancel_verify`` records its ordinal.

    The remote axis is the one with an ORDER contract: `_abort_remote_verify`
    must fire while `_inflight_request_id` is still live, i.e. BEFORE
    `verify_task.cancel()` — otherwise the verify coroutine's own finally has
    already cleared the id and the cancel RPC is a silent no-op that orphans a
    remote verify-merge process (merge_queue.py:17178-17186).
    """
    from unittest.mock import AsyncMock, MagicMock

    from orchestrator.verify_runner import HostLease

    runner = MagicMock()
    runner.name = 'remote-head'
    runner.is_local = False

    async def _cancel_verify():
        if order is not None:
            order.append('abort')
        return 0

    runner.cancel_verify = AsyncMock(side_effect=_cancel_verify)
    runner.probe_clean = AsyncMock(return_value=True)
    return HostLease(name='remote-head', runner=runner, is_local=False)


def _head_verify_task(
    order: list[str] | None = None,
    *,
    lease_ctx=None,
    started: asyncio.Event | None = None,
):
    """A live head verify task that never finishes on its own.

    *lease_ctx*, when given, is an async context manager entered for the whole
    (never-ending) span — used to hold the REAL ``merge_verify_lease`` so the
    lane-idle assertions measure a genuinely-held lane rather than an empty one.
    """
    async def _body():
        try:
            if lease_ctx is None:
                if started is not None:
                    started.set()
                await asyncio.Event().wait()
            else:
                async with lease_ctx:
                    if started is not None:
                        started.set()
                    await asyncio.Event().wait()
        except asyncio.CancelledError:
            if order is not None:
                order.append('cancelled')
            raise
        return None

    return asyncio.get_running_loop().create_task(_body())


async def _head_and_prefix_scene(
    git_repo: Path, tmp_path: Path, monkeypatch, *,
    db_name: str = 'delta-head.db',
    head_lease=None,
    head_task_factory=None,
    head_popped_for_finalize: bool = False,
) -> dict:
    """A REAL two-slot scene: head I0 at slot 1, deep chain at slot 2.

    Distinct from :func:`_prefix_scene_upto_finalize`, which has no head at
    all — there the chain's dispatching item CASes against real main.  Here the
    topology is production's:

        main ← I0 (head, non-speculative, base = real main)
                 ← I1 (speculative, base = I0's merge commit, carries the chain)
                    ← I2 … Ik (links, built on I1's merge commit)

    which is what makes the head's tree a SUBSET of the tip's and its landing
    on the tip's authority sound.  The head is appended through the real
    ``_inflight_append`` chokepoint so it is genuinely registered at VERIFYING
    and genuinely visible to ``_finalizing_head_entry`` / ``_inflight[0]``.
    """
    from orchestrator.event_store import EventStore
    from orchestrator.merge_queue import enqueue_merge_request
    from orchestrator.merge_types import InflightEntry, ItemLifecycleState

    git_ops = _make_git_ops(git_repo, size=2)
    config = _make_config(git_repo, chain_cap=6)
    await _create_branch_editing(git_repo, 'task/100', 'h.txt', 'edit-100\n')
    await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n')
    await _create_branch_editing(
        git_repo, 'task/102', 'shared.txt', _shared_txt_with(5, 'from-102'),
    )
    await _create_branch_editing(git_repo, 'task/103', 'c.txt', 'edit-103\n')
    await _create_branch_editing(git_repo, 'task/104', 'd.txt', 'edit-104\n')
    await _create_branch_editing(
        git_repo, 'task/105', 'shared.txt', _shared_txt_with(5, 'from-105'),
    )
    main_sha = await _rev_parse(git_repo, 'main')
    head_mc = await _merge_commit_off_main(git_repo, 'task/100', '100')
    spec_mc = await _merge_commit_onto(git_repo, 'task/101', head_mc, '101')

    db_path = tmp_path / db_name
    store = EventStore(db_path, 'run-delta-head')
    worker = _make_worker(git_ops)
    worker._event_store = store

    reqs: dict[str, MergeRequest] = {}
    for tid in ('100', '101', *_DELTA_LINKS, _DELTA_TRUNCATOR):
        reqs[tid] = _make_req(tid, tid, config, git_repo)
        await enqueue_merge_request(worker._queue, reqs[tid], store)
    worker._drain_queue_into_lanes()
    assert worker._pop_next_pickable() is reqs['100']
    assert worker._pop_next_pickable() is reqs['101']

    head_item = RealMergeItem(
        request=reqs['100'],
        merge_result=MergeResult(
            success=True, merge_commit=head_mc,
            merge_worktree=_ephemeral_merge_wt(git_ops, 'head'),
        ),
        merge_wt=_ephemeral_merge_wt(git_ops, 'head'),
        base_sha=main_sha,          # slot 1 CASes against REAL main
        speculative=False,          # the trust anchor — never chained
    )
    spec_item = RealMergeItem(
        request=reqs['101'],
        merge_result=MergeResult(
            success=True, merge_commit=spec_mc,
            merge_worktree=_ephemeral_merge_wt(git_ops, 'spec'),
        ),
        merge_wt=_ephemeral_merge_wt(git_ops, 'spec'),
        base_sha=head_mc,           # stacked on the head's merge commit
        speculative=True,
    )

    head_lease = head_lease if head_lease is not None else _local_lease()
    head_task = (
        head_task_factory(git_ops) if head_task_factory is not None
        else _head_verify_task()
    )
    head_entry = InflightEntry(
        item=head_item, lease=head_lease, verify_task=head_task,  # type: ignore[arg-type]
        merge_wt=head_item.merge_wt, was_speculative=False,
    )
    worker._note_transition(
        reqs['100'].request_id, ItemLifecycleState.MERGING,
        ItemLifecycleState.DISPATCHING, live_obj=head_item,
    )
    worker._inflight_append(head_entry)
    if head_popped_for_finalize:
        # The FINALIZE-HEAD window: `_finalize_inflight` popped the entry off
        # the deque before its long `await entry.verify_task`, so the head is
        # visible only through `_finalizing_head_entry()` and `_inflight[0]`
        # would be the SECOND entry (merge_queue.py:13504-13508).
        worker._inflight_popleft()

    chain = await worker._deep_chain_placement(spec_item)
    assert chain is not None
    assert [tid for tid, _ in chain.links] == list(_DELTA_LINKS)
    assert chain.truncated_at == _DELTA_TRUNCATOR

    _spy_post_merge_verify(monkeypatch, outcome=None)   # GREEN tip
    _spy_chain_lane_release(monkeypatch)
    adv = _spy_advance_main(git_ops, monkeypatch)

    return {
        'git_ops': git_ops, 'worker': worker, 'chain': chain, 'reqs': reqs,
        'head_mc': head_mc, 'spec_mc': spec_mc, 'main_sha': main_sha,
        'adv': adv, 'db_path': db_path, 'repo': git_repo,
        'head_entry': head_entry, 'head_item': head_item,
        'head_task': head_task, 'spec_item': spec_item,
    }


async def _adopt_and_land(s: dict) -> dict:
    """Run the adopting exit, then finalize head-then-spec in deque order.

    That order is the deque's, not a choice this helper makes: the head was
    appended first, so FINALIZE-HEAD reaches it first — which is exactly what
    makes the spec item's ``expected_main`` (= the head's merge commit) correct
    without any coordination.
    """
    from orchestrator.merge_types import InflightEntry

    worker = s['worker']
    res = await worker._run_inflight_verify(
        s['spec_item'], _local_lease(), chain=s['chain'],
    )
    s['verify_result'] = res
    # BOUNDED, and the bound is part of the contract: the head's verify never
    # completes on its own in these scenes, so a δ that failed to tear it down
    # would leave `_finalize_inflight` parked on `await entry.verify_task`
    # FOREVER — the whole queue behind it stalled on a verdict it no longer
    # needs.  Expressing that as a timeout makes the failure a fast, legible
    # RED instead of a hung suite.
    s['head_advanced'] = await asyncio.wait_for(
        worker._finalize_inflight(s['head_entry']), timeout=60,
    )
    spec_done: asyncio.Future = asyncio.get_running_loop().create_future()
    spec_done.set_result(res)
    spec_entry = InflightEntry(
        item=s['spec_item'], lease=_local_lease(), verify_task=spec_done,  # type: ignore[arg-type]
        merge_wt=res.merge_wt, was_speculative=True, chain=s['chain'],
    )
    s['spec_entry'] = spec_entry
    s['spec_advanced'] = await worker._finalize_inflight(spec_entry)
    await asyncio.sleep(0)  # let every `_on_finalized` done-callback run
    return s


def _delivered_terminal_task(reason: str):
    """A head verify that ALREADY DELIVERED a terminal ``'blocked'`` outcome.

    Reproduces the shape of `_run_inflight_verify`'s three terminal
    resolve-then-return paths — the dead-verify busy-loop cap-out
    (merge_queue.py:18106-18108), the contended-lease terminal cap-out
    (18330-18332) and the generic ``Verification error:`` handler
    (18528-18530).  Each does ``req.result.set_result(MergeOutcome('blocked'))``
    and THEN returns ``InflightVerifyResult(outcome=err_outcome, merge_wt=None)``
    — a NON-sentinel result (``status`` is None, so neither DROPPED nor
    REQUEUED) whose worktree was already handed to ``_dispose_verify_worktree``.

    The factory only builds the task; the caller resolves the future (the
    scene builder does not hand the factory its request), which is what
    :func:`_deliver_terminal_blocked` does.
    """
    from orchestrator.merge_types import InflightVerifyResult, MergeOutcome

    def _factory(_git_ops):
        async def _body():
            return InflightVerifyResult(
                outcome=MergeOutcome('blocked', reason=reason), merge_wt=None,
            )
        return asyncio.get_running_loop().create_task(_body())

    return _factory


async def _deliver_terminal_blocked(s: dict, reason: str) -> object:
    """Complete the head's verify AND hand its ``'blocked'`` to the workflow.

    The second half is the whole point: those three paths resolve the REQUEST
    FUTURE before returning, so by the time a green tip arrives the workflow
    has already been told BLOCKED and may have been escalated, re-dispatched or
    marked failed.  Also removes the head's ephemeral merge worktree, because
    each of the three paths reached ``_dispose_verify_worktree`` first — so a δ
    that adopts the head would thread a path that no longer exists into
    ``advance_main``.
    """
    import shutil

    await asyncio.sleep(0)  # let the head verify task actually finish
    assert s['head_task'].done(), 'the terminal head verify must have returned'
    delivered = s['head_task'].result().outcome
    assert delivered is not None and delivered.status == 'blocked'
    s['reqs']['100'].result.set_result(delivered)
    shutil.rmtree(s['head_item'].merge_wt, ignore_errors=True)
    s['delivered'] = delivered
    return delivered


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestHeadCancelOnAdoption:
    """(a)-(c) The head is torn down through the chokepoint and lands FIRST."""

    async def test_the_head_verify_is_torn_down_through_the_chokepoint(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a) `_teardown_verify_task` then `_cancel_and_release_tracked`.

        Hand-rolling `verify_task.cancel()` / `_abort_remote_verify` here would
        be caught by the AST ratchet
        (test_merge_queue_concurrent_verify.py::TestVerifyTeardownChokepoint),
        so the assertion is that δ went through the SANCTIONED pair, in the
        order the head-failure-cascade template uses, and cleared `entry.lease`
        afterwards so nothing double-releases.
        """
        s = await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-teardown.db',
        )
        worker = s['worker']
        calls: list[str] = []
        _real_td = worker._teardown_verify_task
        _real_cr = worker._cancel_and_release_tracked

        async def _td(lease, verify_task, task_id, **kw):
            calls.append(f'teardown:{task_id}')
            return await _real_td(lease, verify_task, task_id, **kw)

        async def _cr(lease):
            calls.append(f'cancel_release:{getattr(lease, "name", None)}')
            return await _real_cr(lease)

        monkeypatch.setattr(worker, '_teardown_verify_task', _td)
        monkeypatch.setattr(worker, '_cancel_and_release_tracked', _cr)

        await worker._run_inflight_verify(
            s['spec_item'], _local_lease(), chain=s['chain'],
        )

        assert calls == ['teardown:100', 'cancel_release:local'], calls
        assert s['head_task'].done(), 'the head verify task must be reaped'
        assert s['head_entry'].lease is None, (
            'entry.lease must be cleared after the release so no later path '
            'double-releases it'
        )

    async def test_the_remote_abort_precedes_the_cancel(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a) ORDER IS LOAD-BEARING: abort, THEN cancel.

        The verify coroutine's finally clears `_inflight_request_id` on
        cancellation, which turns a later `cancel_verify()` into a silent no-op
        and orphans the remote verify-merge process.  Observed directly by
        recording both sides into one list.
        """
        order: list[str] = []
        s = await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-order.db',
            head_lease=_remote_lease(order),
            head_task_factory=lambda _go: _head_verify_task(order),
        )

        await s['worker']._run_inflight_verify(
            s['spec_item'], _local_lease(), chain=s['chain'],
        )

        assert order == ['abort', 'cancelled'], order

    async def test_the_finalizing_head_is_found_when_it_is_off_the_deque(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a) The COMMON topology: the head is already mid-finalize.

        `_finalize_inflight` pops its entry BEFORE the long
        `await entry.verify_task`, so during that window the head is invisible
        to `_inflight[0]` and reachable only via `_finalizing_head_entry()`
        (merge_queue.py:10265).  A δ that only looked at the deque would silently
        skip the cancel in exactly the case that matters most.
        """
        s = await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-offdeque.db',
            head_popped_for_finalize=True,
        )
        worker = s['worker']
        assert worker._finalizing_head_entry() is s['head_entry']

        await worker._run_inflight_verify(
            s['spec_item'], _local_lease(), chain=s['chain'],
        )

        assert s['head_task'].cancelled(), (
            'the finalizing head\'s verify must be cancelled too'
        )
        assert s['head_entry'].lease is None

    async def test_the_head_lands_first_and_main_history_stays_linear(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(b) I0 → I1 → I2… in ONE linear first-parent chain on main.

        The head lands on the TIP's authority (its own verdict never arrived),
        and it lands FIRST — which is the only reason the spec item's
        `expected_main` (= the head's merge commit) matches.
        """
        s = await _adopt_and_land(await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-lands.db',
        ))
        link_shas = [mc for _tid, mc in s['chain'].links]

        assert s['head_advanced'] is True
        assert s['spec_advanced'] is True
        assert [c[0] for c in s['adv']] == [
            s['head_mc'], s['spec_mc'], *link_shas,
        ]
        assert [c[1] for c in s['adv']] == [
            s['main_sha'], s['head_mc'], s['spec_mc'], *link_shas[:-1],
        ]
        assert await _rev_parse(s['repo'], 'main') == s['chain'].tip
        assert s['reqs']['100'].result.result().status == 'done'
        # `merge_finalized`'s payload keys the item by BRANCH (`bare_id`);
        # `task_id` is a top-level event column, not a `data` field.
        by_branch = {r['branch']: r for r in _finalized_rows(s['db_path'])}
        assert by_branch['100']['state'] == 'done'

    async def test_the_head_landing_is_not_attributed_to_the_chain(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(b) The head is I0, NOT a chain link — the canary must not count it.

        `landed_via_chain` sums to items-landed-per-deep-verify
        (scripts/merge-deep-canary-predicate.sh:89-91).  The head carries no
        chain of its own, so stamping it would inflate that ratio by one on
        every deep round for an item the walk never touched.
        """
        s = await _adopt_and_land(await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-canary.db',
        ))
        by_branch = {r['branch']: r for r in _finalized_rows(s['db_path'])}

        assert by_branch['100']['landed_via_chain'] is None
        for tid in ('101', *_DELTA_LINKS):
            assert by_branch[tid]['landed_via_chain'] == 1

    async def test_a_red_head_verdict_does_not_block_the_prefix(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(c) HEAD-FAIL + TIP-PASS — the PRD boundary row, verbatim.

        A head verify that already came back RED is a verdict about a SUBSET
        tree that the green tip has strictly better evidence for; acting on it
        would fail an item the tip just proved good and would strand the whole
        prefix behind it.  So the red verdict resolves NO
        MergeOutcome('blocked') and the full prefix still lands.
        """
        from orchestrator.merge_types import InflightVerifyResult, MergeOutcome

        def _red_task(_git_ops):
            async def _body():
                return InflightVerifyResult(
                    outcome=MergeOutcome('blocked', reason='head verify red (flake)'),
                    merge_wt=None,
                    spec_warm=False,
                )
            return asyncio.get_running_loop().create_task(_body())

        s = await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-red.db',
            head_task_factory=_red_task,
        )
        await asyncio.sleep(0)  # let the red verdict actually land
        assert s['head_task'].done()
        s = await _adopt_and_land(s)

        head_outcome = s['reqs']['100'].result.result()
        assert head_outcome.status == 'done', (
            f'a red head verdict must not survive a green tip; got {head_outcome!r}'
        )
        assert await _rev_parse(s['repo'], 'main') == s['chain'].tip
        for tid in ('101', *_DELTA_LINKS):
            assert s['reqs'][tid].result.result().status == 'done'

    # ── step-15 (review fix #4): a DELIVERED head is not δ's to retract ──────

    async def test_a_head_whose_outcome_was_already_delivered_is_not_adopted(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a) Future already resolved ⇒ `_adopt_head_on_tip_authority` declines.

        Decision #3 makes the tip authoritative for an UNDECIDED head; it is
        not a licence to RETRACT an outcome the workflow has already been
        handed.  Today the only guard is `head.verify_task is None`
        (merge_queue.py:19099) — nothing looks at the request future — so the
        three terminal resolve-then-return paths produce a head that is adopted
        after its `MergeOutcome('blocked')` was already delivered.
        """
        s = await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-delivered.db',
            head_task_factory=_delivered_terminal_task('dead/hung verify cap-out x3'),
        )
        await _deliver_terminal_blocked(s, 'dead/hung verify cap-out x3')
        worker, head = s['worker'], s['head_entry']

        adopted = await worker._adopt_head_on_tip_authority(
            s['reqs']['101'].request_id, '101',
        )

        assert adopted is None, (
            'a head whose outcome is already delivered must not be adopted; '
            f'got {adopted!r}'
        )
        assert head.chain_adopted is False, (
            '`chain_adopted` is what makes _finalize_inflight skip the fail '
            'arm (merge_queue.py:19905) — it must stay unset for a delivered head'
        )
        assert head.verify_task is not None, (
            'the decline must leave the entry untouched: nulling verify_task '
            'would push it onto the PASS arm by the other route'
        )
        assert head.lease is not None, 'nothing was torn down, so nothing was released'

    async def test_a_delivered_head_never_advances_main_and_keeps_its_verdict(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(b)+(c) The ordinary fail arm runs; main never sees its merge commit.

        Today `chain_adopted` skips the fail arm, the PASS arm CASes the head's
        merge commit onto main from `entry.merge_wt` — the ephemeral path those
        three terminal routes already `_dispose_verify_worktree()`d — and
        `_resolve_or_drop_abandoned` then SILENTLY DROPS the resulting `'done'`
        onto the already-resolved future.  Net effect: the branch lands on main
        while the workflow is told BLOCKED and can be escalated, re-dispatched
        or marked failed for work that is already on main.
        """
        s = await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-delivered-fin.db',
            head_task_factory=_delivered_terminal_task('Verification error: boom'),
        )
        delivered = await _deliver_terminal_blocked(s, 'Verification error: boom')
        s = await _adopt_and_land(s)

        assert s['head_advanced'] is False, (
            'the head must take the ordinary FAIL/skip arm, not the PASS arm'
        )
        assert all(call[0] != s['head_mc'] for call in s['adv']), (
            'main must never be advanced to a declined head\'s merge commit; '
            f'advance_main calls: {[(c[0][:8], (c[1] or "")[:8]) for c in s["adv"]]}'
        )
        # (c) the disposed worktree is never threaded into a CAS advance.
        assert all(call[2] != s['head_item'].merge_wt for call in s['adv']), (
            'the fail arm exists so the already-disposed ephemeral worktree '
            'never reaches advance_main'
        )
        assert s['reqs']['100'].result.result() is delivered, (
            'the delivered outcome is the workflow\'s; nothing may overwrite it'
        )
        by_branch = {r['branch']: r for r in _finalized_rows(s['db_path'])}
        assert by_branch['100']['state'] == 'blocked'
        assert by_branch['100']['landed_via_chain'] is None

    async def test_no_link_is_cased_against_the_declined_heads_merge_commit(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(d) Decision #9: nothing lands on a base that never reached main.

        The tip's `expected_main` IS the declined head's merge commit, so the
        tip's FIRST CAS is refused and the walk over `chain.links` is not
        reached from it.

        PLAN PREMISE CORRECTED (step-15 (d) predicted every link would stay
        queued): it does not, and δ neither introduces nor may bypass the
        reason.  `cas_failed` is TRANSIENT — merge_queue.py:20296-20305
        re-anchors `item.base_sha` to `get_main_sha()` and retries, so the tip
        lands one attempt later against REAL main and the prefix follows it.
        What decision #9 actually forbids, and what is pinned here, is a link
        CASed against a base no `advance_main` ever put on main: every link's
        `expected_main` is its PREDECESSOR'S landed sha, and `head_mc` is not
        among them.
        """
        s = await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-delivered-cas.db',
            head_task_factory=_delivered_terminal_task('lane cap-out #1'),
        )
        await _deliver_terminal_blocked(s, 'lane cap-out #1')
        s = await _adopt_and_land(s)
        link_shas = [mc for _tid, mc in s['chain'].links]

        assert s['adv'][0] == (s['spec_mc'], s['head_mc'], s['adv'][0][2]), (
            'the tip must first try its real base — the head\'s merge commit'
        )
        assert s['adv'][1][:2] == (s['spec_mc'], s['main_sha']), (
            'that CAS must be REFUSED, so the retry re-anchors onto real main'
        )
        assert [c[0] for c in s['adv'][1:]] == [s['spec_mc'], *link_shas]
        assert [c[1] for c in s['adv'][2:]] == [s['spec_mc'], *link_shas[:-1]], (
            'each link CASes against its predecessor\'s landed sha, never '
            'against the head\'s un-landed merge commit'
        )
        assert s['head_mc'] not in [c[1] for c in s['adv'][1:]]

    async def test_a_red_head_with_an_unresolved_future_is_still_adopted(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """CONTRAST — the guard keys on DELIVERY, not on the verdict being red.

        Same head shape as the delivered case in every respect but one: the
        request future is still pending.  That is the genuine "Head-fail +
        tip-pass" boundary row, and it must still be adopted-and-declined — a
        guard that keyed on `vr.outcome is not None` instead would silently
        re-strand the whole prefix behind a verdict the green tip outranks.
        """
        s = await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-red-unresolved.db',
            head_task_factory=_delivered_terminal_task('head verify red (flake)'),
        )
        await asyncio.sleep(0)
        assert s['head_task'].done()
        assert not s['reqs']['100'].result.done(), 'future deliberately UNresolved'

        adopted = await s['worker']._adopt_head_on_tip_authority(
            s['reqs']['101'].request_id, '101',
        )

        assert adopted is s['head_entry']
        assert s['head_entry'].chain_adopted is True

    async def test_a_head_still_verifying_is_still_adopted(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """CONTRAST — an UNDECIDED head is exactly what decision #3 is about."""
        s = await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-still-verifying.db',
        )
        assert not s['head_task'].done()
        assert not s['reqs']['100'].result.done()

        adopted = await s['worker']._adopt_head_on_tip_authority(
            s['reqs']['101'].request_id, '101',
        )

        assert adopted is s['head_entry']
        assert s['head_entry'].chain_adopted is True
        assert s['head_entry'].verify_task is None, 'torn down, per topology 1'


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestHeadCancelLeavesTheLaneIdle:
    """(d) BOTH BUSY axes read IDLE after the cancel — 3071's precondition."""

    async def test_both_lease_axes_are_free_after_a_local_head_cancel(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """The flock axis AND the fixed-key rendezvous axis, independently.

        A LOCAL head is clean BY CONSTRUCTION — cancelling its task unwinds
        `GitOps.merge_verify_lease`'s finally, which does BOTH releases — but
        "by construction" is exactly the kind of claim that rots silently, and
        3071's admission guard reads the lane BUSY and defers the FLEET if it
        is wrong.  Held for real, then measured for real.
        """
        from orchestrator.verify_cancel import (
            lane_lock_holder_pids_strict,
            lane_lock_path,
            read_lock_holder_pgid,
        )

        started = asyncio.Event()

        def _leased_task(git_ops):
            # The scene hands its own GitOps to the factory, so the head's
            # verify holds the SAME `_merge-verify` lane lock a real local
            # verify would — no module-attribute patching needed.
            return _head_verify_task(
                lease_ctx=git_ops.merge_verify_lease(), started=started,
            )

        s = await _head_and_prefix_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-head-lease.db',
            head_task_factory=_leased_task,
        )
        git_ops = s['git_ops']
        await asyncio.wait_for(started.wait(), timeout=30)
        lock_path = lane_lock_path(git_ops.persistent_merge_worktree_path)
        # Staging check: the lane really IS busy on both axes before δ acts,
        # so the post-cancel assertions cannot pass vacuously.
        assert git_ops._merge_verify_lease_active() is True
        assert os.getpid() in lane_lock_holder_pids_strict(lock_path)

        await s['worker']._run_inflight_verify(
            s['spec_item'], _local_lease(), chain=s['chain'],
        )

        assert read_lock_holder_pgid(git_ops.worktree_base) is None, (
            'the fixed-key holder rendezvous must be cleared — '
            'reset_persistent_merge_worktree reads it FAIL-CLOSED'
        )
        assert git_ops._merge_verify_lease_active() is False
        assert lane_lock_holder_pids_strict(lock_path) == [], (
            'the kernel flock axis must be free — this is the ONLY axis '
            'warm-lane-lock-guard.sh measures'
        )
        # And the predicate `reset_persistent_merge_worktree` consumes
        # FAIL-CLOSED agrees, which is the whole point of asserting the
        # rendezvous axis separately from the flock one.
        assert git_ops._merge_verify_lease_active() is False


class TestRemoteCancelClearsTheHolderRendezvous:
    """(e) `cancel-verify` closes the merge-worker-initiated leak.

    Sync class, deliberately: pytest-asyncio is STRICT here and a sync
    ``test_*`` inside an ``@pytest.mark.asyncio`` class is an ERROR.
    """

    def test_cancel_verify_clears_the_fixed_key_holder_file(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """`cancel_request`'s SIGKILL skips cli.py's finally — so the CLI must.

        verify_cancel.py:303-336 records the consequence: the FIXED-key holder
        file is what `_merge_verify_lease_active` probes with `killpg(pgid, 0)`,
        and a leaked (or pid-recycled) entry there reads as a LIVE holder and
        fails CLOSED — a wedged warm lane, bounded only by the next run that
        happens to overwrite it.  δ cancels REMOTE head verifies through this
        exact command, so it must leave the rendezvous clean.
        """
        from unittest.mock import MagicMock

        from click.testing import CliRunner

        from orchestrator import cli as cli_module
        from orchestrator.cli import main
        from orchestrator.config import OrchestratorConfig
        from orchestrator.verify_cancel import (
            pgid_file,
            read_lock_holder_pgid,
            write_lock_holder_pgid,
        )

        worktree_base = tmp_path / '.worktrees'
        worktree_base.mkdir()
        write_lock_holder_pgid(worktree_base, os.getpgrp())
        assert read_lock_holder_pgid(worktree_base) == os.getpgrp()

        monkeypatch.setattr(
            cli_module, 'load_config',
            lambda _: OrchestratorConfig(project_root=tmp_path),
        )
        mock_git_ops = MagicMock()
        mock_git_ops.worktree_base = worktree_base
        monkeypatch.setattr(
            'orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops),
        )
        cfg_file = tmp_path / 'config.yaml'
        cfg_file.write_text('')

        r = CliRunner().invoke(main, [
            'cancel-verify', '--request-id', 'delta-head-req',
            '--config', str(cfg_file),
        ])

        assert r.exit_code == 0, r.output
        assert read_lock_holder_pgid(worktree_base) is None, (
            'cancel-verify must clear the fixed-key holder rendezvous the '
            'SIGKILLed verify-merge could not clear itself'
        )
        assert not pgid_file(worktree_base, '_merge_verify_lock_holder').exists()

    def test_cancel_verify_leaves_the_rendezvous_alone_when_the_kill_failed(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A non-zero rc means a LIVE process refused SIGKILL — it still holds.

        Clearing the rendezvous then would tell `_merge_verify_lease_active`
        the lane is free while a live verify still owns it, converting a
        visible retry-or-escalate into a silent unprotected overlap.  Same
        fail-closed reasoning as the retained per-request pgid file.
        """
        from unittest.mock import MagicMock

        from click.testing import CliRunner

        from orchestrator import cli as cli_module
        from orchestrator.cli import main
        from orchestrator.config import OrchestratorConfig
        from orchestrator.verify_cancel import (
            read_lock_holder_pgid,
            write_lock_holder_pgid,
        )

        worktree_base = tmp_path / '.worktrees'
        worktree_base.mkdir()
        write_lock_holder_pgid(worktree_base, os.getpgrp())

        monkeypatch.setattr(
            cli_module, 'load_config',
            lambda _: OrchestratorConfig(project_root=tmp_path),
        )
        mock_git_ops = MagicMock()
        mock_git_ops.worktree_base = worktree_base
        monkeypatch.setattr(
            'orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops),
        )
        monkeypatch.setattr(cli_module, 'cancel_request', lambda *a, **k: 3)
        cfg_file = tmp_path / 'config.yaml'
        cfg_file.write_text('')

        r = CliRunner().invoke(main, [
            'cancel-verify', '--request-id', 'delta-head-req',
            '--config', str(cfg_file),
        ])

        assert r.exit_code == 3
        assert read_lock_holder_pgid(worktree_base) == os.getpgrp()


# ═══════════════════════════════════════════════════════════════════════════
# step-11 RED — conservation: the walk consumes no per-item speculation permits
#
# THE δ-SPECIFIC HAZARD, stated once.  Every other landing path in this file
# lands an item that DISPATCHED: it acquired a `SpecPermit` from the
# speculation ledger (and possibly a `CapPermit` from the merge-ahead ledger),
# carried the token on its `InflightEntry`/`RealMergeItem`, and gives it back
# in `_finalize_inflight`'s single `finally`.  A chain LINK did none of that —
# it sat in a lane buffer for the whole round and never dispatched at all — so
# the walk lands k items while only ONE of them ever held a permit.
#
# That asymmetry is exactly what a "for symmetry" edit would break, in either
# direction: releasing a link's non-existent token raises AssertionError
# (merge_speculation_controller.py:213-239), and releasing the head's token
# once per landed item would over-release the semaphore, silently raising
# `slot_available` above `depth` and licensing more concurrent speculation than
# the operator configured.  Both are invisible to the landing assertions in
# TestInOrderCasWalk — they land the same four commits either way — which is
# why conservation gets its own oracle here.
#
# Note the SWALLOWING: `_land_chain_prefix`'s per-link `except Exception`
# contains a stray AssertionError rather than propagating it, so "no bad
# release happened" is NOT observable as a raised exception.  It is observable
# as (i) the landing running to completion instead of stopping short and
# (ii) the containment arm's WARNING never being logged — both asserted below.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestChainWalkConsumesNoPermits:
    """A chain consumes no per-item speculation permits — only the head's."""

    async def _landing_scene(
        self, git_repo: Path, tmp_path: Path, monkeypatch, *,
        db_name: str, hook=None,
    ) -> dict:
        """Build the 4-item chain and drive it THROUGH ``_finalize_inflight``."""
        s = await _prefix_scene_upto_finalize(
            git_repo, tmp_path, monkeypatch, db_name=db_name, advance_hook=hook,
        )
        s['advanced'] = await s['worker']._finalize_inflight(s['entry'])
        await asyncio.sleep(0)  # let every `_on_finalized` done-callback run
        return s

    async def test_a_full_landing_leaves_the_pipeline_quiescent(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a) k+1 landings, then all six quiescence surfaces are green.

        The audits are gated on ``_running`` and on a REAL ``main_sha`` (both
        traps are documented on :func:`_assert_quiescent`), so both are
        supplied here rather than left to whatever the scene happened to leave
        behind — a stopped worker or an 'unknown' sha would make this pass
        vacuously instead of meaningfully.
        """
        s = await self._landing_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-conserve-full.db',
        )
        worker = s['worker']
        landed = ['101', *_DELTA_LINKS]

        assert [c[0] for c in s['adv']] == [
            s['head_mc'], *[mc for _tid, mc in s['chain'].links],
        ], 'the whole prefix must have landed before conservation means anything'

        # Asserted BEFORE the drain: a landed link that wrongly stayed buffered
        # would show up here, not be quietly drained away.
        assert _drain_residue(worker) == {_DELTA_TRUNCATOR}

        worker._running = True
        main_now = await _rev_parse(git_repo, 'main')
        _assert_quiescent(
            worker, main_now,
            [s['reqs'][t] for t in (*landed, _DELTA_TRUNCATOR)],
        )

    async def test_a_stale_cas_partial_landing_leaves_the_pipeline_quiescent(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(b) The identities survive a MID-WALK abort too.

        The abort path is where a half-updated accounting would hide: the
        prefix that landed is terminal, the links past the abort are still
        buffered and unresolved, and the two populations must not have been
        conflated.  Decision #9's "left exactly as they were" is asserted as
        the residue SET; conservation is asserted over the rest.
        """
        from orchestrator.git_ops import AdvanceOutcome

        s = await self._landing_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-conserve-abort.db',
            hook=_abort_hook_at(3, outcome=AdvanceOutcome('cas_failed')),
        )
        worker = s['worker']

        assert len(s['adv']) == 3, 'the walk must have stopped at the 3rd CAS'
        assert _drain_residue(worker) == {'103', '104', _DELTA_TRUNCATOR}

        worker._running = True
        main_now = await _rev_parse(git_repo, 'main')
        _assert_quiescent(
            worker, main_now,
            [s['reqs'][t] for t in ('101', *_DELTA_LINKS, _DELTA_TRUNCATOR)],
        )

    async def test_only_the_dispatching_items_permit_is_ever_released(
        self, git_repo: Path, tmp_path: Path, monkeypatch, caplog,
    ) -> None:
        """(c) THE hazard: one release, and it is the head's own token.

        The head is given a REAL ``SpecPermit`` first, so "released exactly
        once" is a claim about a token that genuinely exists rather than the
        vacuous zero an entry with ``permit=None`` would produce.  Four items
        land; exactly one permit comes back.

        A stray ``release`` for a link cannot surface as a propagated
        exception — the walk's per-link ``except Exception`` contains it — so
        it is caught here by its two visible shadows: the landing would stop
        short of four, and the containment arm would log.
        """
        import logging

        s = await _prefix_scene_upto_finalize(
            git_repo, tmp_path, monkeypatch, db_name='delta-conserve-permit.db',
        )
        worker, entry = s['worker'], s['entry']

        before = _permit_census(worker)
        assert before['spec_available'] >= 1, (
            'the scene needs a free speculation slot to hand the head'
        )
        head_permit = await worker._speculation_ledger.acquire()
        entry.permit = head_permit

        spec_released: list = []
        cap_released: list = []
        _real_spec_release = worker._speculation_ledger.release
        _real_cap_release = worker._merge_ahead_ledger.release
        monkeypatch.setattr(
            worker._speculation_ledger, 'release',
            lambda p: (spec_released.append(p), _real_spec_release(p))[1],
        )
        monkeypatch.setattr(
            worker._merge_ahead_ledger, 'release',
            lambda p: (cap_released.append(p), _real_cap_release(p))[1],
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            await worker._finalize_inflight(entry)
            await asyncio.sleep(0)

        assert spec_released == [head_permit], (
            'the walk must release the DISPATCHING item\'s permit exactly once '
            'and no link\'s — a link never acquired one'
        )
        assert cap_released == [], 'no link ever held a merge-ahead cap permit'
        assert len(s['adv']) == 1 + len(s['chain'].links), (
            'the landing stopped short — a swallowed AssertionError from a '
            'release() on a non-live token looks exactly like this'
        )
        assert not [
            r for r in caplog.records if 'raised during its landing' in r.getMessage()
        ], 'the per-link containment arm fired — something raised inside the walk'

        after = _permit_census(worker)
        assert after['spec_live'] == before['spec_live'], (
            'the ledger must end the walk holding exactly the tokens it held '
            'before the head acquired one'
        )
        assert after['spec_available'] == before['spec_available']
        assert after['spec_available'] + len(after['spec_live']) == after['spec_depth']
        assert after['cap_available'] + len(after['cap_live']) == after['cap_depth']

    async def test_the_chain_build_and_verify_take_no_permits(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(d) The permit census is unmoved by the BUILD half as well.

        β acquires a worktree lane for the build (``acquire_chain_build_lane``)
        and nothing else — no speculation slot, no merge-ahead cap — and δ must
        not have quietly changed that while teaching the tip to adopt.  Read
        across both halves: the build itself, and the whole verify that follows.
        """
        s = await _prefix_scene_upto_finalize(
            git_repo, tmp_path, monkeypatch, db_name='delta-conserve-build.db',
        )

        assert s['permits_after_build'] == s['permits_before_build'], (
            'building a chain must not consume a permit — it takes a worktree '
            'lane and nothing else'
        )
        assert s['permits_after_verify'] == s['permits_before_build'], (
            'the tip verify must not consume a permit either'
        )

    async def test_the_chain_lane_is_released_exactly_once_on_the_adopting_exit(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(e) ChainResult's EXACTLY-once lane contract, on δ's new exit.

        γ shipped three exits through the release in ``_run_inflight_verify``'s
        ``finally``; δ adds the adopting fourth.  A double release would hand
        the same lane to two concurrent chain builds; a skipped one would
        starve the pool a lane per deep round.
        """
        s = await self._landing_scene(
            git_repo, tmp_path, monkeypatch, db_name='delta-conserve-lane.db',
        )

        assert [lane for lane, _warm in s['lane_releases']] == [s['chain'].lane], (
            'the adopting exit must return the chain lane exactly once'
        )


# ═══════════════════════════════════════════════════════════════════════════
# step-13 RED — δ END TO END: the task's USER-OBSERVABLE signal
#
# Everything above pins a seam.  This class is the only place the seams are
# driven the way production drives them — `_dispatch_item` →
# `_deep_chain_placement` → `_run_inflight_verify` → `_finalize_inflight` →
# the walk — against a real repo whose main really moves.
#
# γ's `_DeepScene` (test_merge_queue_deep_dispatch.py:2208) stopped one call
# short of exactly this, and said so (:2266-2268: "this scene stops short of
# finalize (γ lands nothing)").  `_DeltaScene` below is that scene with the
# missing call restored, which is the whole of δ's addition to the round.
#
# Four properties, one per PRD row:
#   (a) ONE PASSING TIP LANDS k ITEMS IN ORDER, linear on main
#   (b) TIP FAIL LANDS NOTHING VIA THE CHAIN — and the item still lands later
#   (c) KILL SWITCH — at the shipped cap=0 the round is byte-identical
#   (d) HOT RELOAD — cap 0 -> 4 through the REAL `apply_reload`, no restart
# ═══════════════════════════════════════════════════════════════════════════


_DELTA_E2E_FOLLOWERS = ('102', '103', '104', '105', '106')
"""Five chainable followers, all editing DISJOINT files.

``queue_len`` is therefore 6 (five followers plus the head), so a ``cap`` of 4
is what binds and the built depth is a statement about the POLICY rather than
about the fixture's length.  Cleanliness is the point too: nothing truncates,
so a short landing can only ever be δ's doing.
"""


class _DeltaScene:
    """One repo + worker + queue, driven round after round THROUGH finalize.

    Cloned from γ's ``_DeepScene`` rather than imported — ``orchestrator/tests``
    has no ``__init__.py``, so a cross-module import would couple the two
    suites' collection order (the reason recorded in this module's docstring).

    The differences from γ's are the ones δ is:
      * every head item CASes against the REAL main sha of the moment
        (``base_sha=main_before``), because these rounds actually land;
      * ``round_`` continues into ``_finalize_inflight``;
      * each round takes its own head BRANCH, since a branch that has landed
        cannot produce a second merge commit.
    """

    def __init__(self, git_ops, config, worker, repo, store, db_path) -> None:
        self.git_ops = git_ops
        self.config = config
        self.worker = worker
        self.repo = repo
        self.store = store
        self.db_path = db_path
        self.calls: list[dict] = []
        self.posted: list[dict] = []
        self.built: list[dict] = []
        self.lane_releases: list[tuple] = []
        self.reqs: dict[str, MergeRequest] = {}
        self._round_no = 0

    async def enqueue(self, task_ids) -> None:
        """Put *task_ids* on the queue through the REAL enqueue chokepoint.

        ``enqueue_merge_request`` is what registers ``_on_finalized``, and
        ``merge_finalized`` has no other emit site (merge_queue.py:4763-4777) —
        so a scene that stuffed ``_lane_buffers`` directly (γ's shortcut, valid
        for a scene that lands nothing) would make every landing assertion here
        blind to the payload δ exists to produce.
        """
        from orchestrator.merge_queue import enqueue_merge_request

        for tid in task_ids:
            self.reqs[tid] = _make_req(tid, tid, self.config, self.repo)
            await enqueue_merge_request(
                self.worker._queue, self.reqs[tid], self.store,
            )
        self.worker._drain_queue_into_lanes()

    async def round_(
        self, *, tag: str, head_tid: str, req: MergeRequest | None = None,
    ) -> dict:
        """Drive ONE round: dispatch → verify → finalize.

        *req* re-dispatches an EXISTING request (the one a previous round put
        back) instead of picking the next one off the lane buffers — which is
        how a scenario asserts that the very same request lands on a later
        round's own verdict.

        Returns the recorded ``_run_inflight_verify`` kwargs plus the round's
        own facts (``'advanced'``, ``'main_before'``, ``'head_mc'``).
        """
        from orchestrator.merge_types import MergeResult, RealMergeItem

        self._round_no += 1
        worker = self.worker
        main_before = await _rev_parse(self.repo, 'main')
        head_mc = await _merge_commit_off_main(
            self.repo, f'task/{head_tid}', f'{tag}-r{self._round_no}',
        )
        # The head leaves the buffers the way the merger takes it — it must not
        # still be queued when `chain_snapshot` runs, or it would chain itself.
        popped = req if req is not None else worker._pop_next_pickable()
        assert popped is not None and popped.task_id == head_tid, (
            f'expected {head_tid} to be the pickable head, got '
            f'{None if popped is None else popped.task_id}'
        )
        wt = _ephemeral_merge_wt(self.git_ops, f'{tag}-r{self._round_no}')
        item = RealMergeItem(
            request=popped,
            merge_result=MergeResult(
                success=True, merge_commit=head_mc, merge_worktree=wt,
            ),
            merge_wt=wt,
            base_sha=main_before,   # this round really CASes against main
            speculative=True,       # slot 2 — the only kind that chains
        )
        entry = await worker._dispatch_item(item)
        assert entry is not None, 'dispatch must not decline: a host is free'
        assert entry.verify_task is not None
        await entry.verify_task
        advanced = await worker._finalize_inflight(entry)
        await asyncio.sleep(0)  # let every `_on_finalized` done-callback run

        rec = self.calls[-1]
        rec.update({
            'round': self._round_no, 'item': item, 'advanced': advanced,
            'main_before': main_before, 'head_mc': head_mc, 'req': popped,
        })
        if entry.lease is not None:
            await worker._host_allocator.release(entry.lease)
        return rec


async def _make_delta_scene(
    repo: Path, tmp_path: Path, monkeypatch, *,
    chain_cap: int, script: list[bool], db_name: str,
    heads: tuple[str, ...] = ('101',),
) -> _DeltaScene:
    """Build a finalize-capable scene whose verify returns *script* in order."""
    from orchestrator.event_store import EventStore

    git_ops = _make_git_ops(repo, size=2)
    config = _make_config(repo, chain_cap=chain_cap)
    for tid in (*heads, *_DELTA_E2E_FOLLOWERS):
        await _create_branch_editing(repo, f'task/{tid}', f'f{tid}.txt', f'edit-{tid}\n')
    db_path = tmp_path / db_name
    store = EventStore(db_path, f'run-{db_name}')
    worker = _make_worker(git_ops)
    worker._event_store = store
    scene = _DeltaScene(git_ops, config, worker, repo, store, db_path)
    await scene.enqueue((*heads, *_DELTA_E2E_FOLLOWERS))

    # The round recorder, installed ONCE — re-wrapping per round would capture
    # the previous round's recorder as `real` and nest a wrapper deeper each
    # round (γ's note at test_merge_queue_deep_dispatch.py:2398-2401).
    real_verify = worker._run_inflight_verify

    async def _recording(_item, _lease, **kwargs):
        rec = dict(kwargs)
        scene.calls.append(rec)
        rec['result'] = await real_verify(_item, _lease, **kwargs)
        return rec['result']

    monkeypatch.setattr(worker, '_run_inflight_verify', _recording)
    scene.lane_releases = _spy_chain_lane_release(monkeypatch)

    real_build = merge_queue.build_chain

    async def _recording_build(git_ops_, queue_snapshot, head_merge_commit, **kw):
        scene.built.append({
            'queue_snapshot': tuple(queue_snapshot),
            'head_merge_commit': head_merge_commit, **kw,
        })
        return await real_build(git_ops_, queue_snapshot, head_merge_commit, **kw)

    monkeypatch.setattr(merge_queue, 'build_chain', _recording_build)

    verdicts = list(script)

    async def _oracle(_git_ops, _req, merge_wt, **kwargs):
        scene.posted.append({'merge_wt': merge_wt, **kwargs})
        passed = verdicts.pop(0) if verdicts else True
        return None if passed else _fail_verify_result()

    monkeypatch.setattr('orchestrator.merge_queue._run_post_merge_verify', _oracle)
    return scene


def _delta_round_transcript(scene: _DeltaScene, idx: int) -> dict:
    """Normalise ONE round into a repo-independent, comparable transcript.

    γ's normaliser (test_merge_queue_deep_dispatch.py:2348) with the FINALIZE
    facts appended — absolute paths and SHAs differ between fixture repos, so
    the comparison is on the facts: was a chain handed down, what was labelled,
    did the round advance main, and did anything carry δ's landing stamp.
    """
    rec = scene.calls[idx]
    posted = scene.posted[idx]
    item = rec['item']
    outcome = rec['req'].result.result() if rec['req'].result.done() else None
    return {
        'chain': rec['chain'],
        'chain_items': rec['chain_items'],
        'depth': rec['depth'],
        'probe_base': rec['probe_base'],
        'verified_the_items_own_merge_commit':
            posted['merge_sha'] == item.merge_result.merge_commit,
        'result_status': rec['result'].status,
        'result_has_worktree': rec['result'].merge_wt is not None,
        'advanced': rec['advanced'],
        'outcome_status': None if outcome is None else outcome.status,
        'landed_via_chain': None if outcome is None else outcome.landed_via_chain,
        'build_chain_calls': len(scene.built),
        'halving_state': scene.worker._chain_halving_state,
    }


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestDeepLandingEndToEnd:
    """δ driven the way production drives it, one round at a time."""

    async def test_one_passing_tip_lands_the_whole_prefix_in_order(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a) ONE verify → k in-order landings, and main stays LINEAR.

        The user-observable claim of the whole PRD: a single verify run pays
        for the whole clean prefix.  Linearity is asserted through
        ``rev-list --first-parent``, which is the shape the startup reconciler
        and every ``merge-base --is-ancestor`` consumer downstream assume.
        """
        scene = await _make_delta_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=4, script=[True], db_name='delta-e2e-land.db',
        )
        rec = await scene.round_(tag='land', head_tid='101')

        chain = rec['chain']
        assert chain is not None and len(chain.links) == 3, (
            f'cap=4 must bind: expected a 4-item chain, got {chain!r}'
        )
        # The dispatch-time kwarg is the FLOOR (merge_queue.py:20878 passes a
        # literal 1); `_run_inflight_verify` recomputes it from the chain it was
        # handed (:17588), which is what reaches η1's `merge_verify` row.
        assert rec['chain_items'] == 1
        assert len(scene.posted) == 1, 'ONE verify paid for the whole prefix'

        landed = ['101', *[tid for tid, _ in chain.links]]
        for tid in landed:
            req = scene.reqs[tid]
            assert req.result.done(), f'task {tid} never resolved'
            assert req.result.result().status == 'done', f'task {tid} did not land'
        # Every LINK carries the stamp; the head landed by the ordinary
        # advance, and its own stamp is asserted in TestLandedViaChainCarrier.
        for tid in landed[1:]:
            assert scene.reqs[tid].result.result().landed_via_chain == 1

        # ── main is LINEAR, in land order ────────────────────────────────────
        _rc, out, _err = await _run(
            ['git', 'rev-list', '--first-parent', 'main'], cwd=git_repo,
        )
        first_parents = out.split()
        expected = [rec['head_mc'], *[mc for _tid, mc in chain.links]]
        assert first_parents[:len(expected)] == list(reversed(expected)), (
            'main\'s first-parent history must be exactly the land order'
        )
        for earlier, later in zip(expected, expected[1:], strict=False):
            rc, _o, _e = await _run(
                ['git', 'merge-base', '--is-ancestor', earlier, later], cwd=git_repo,
            )
            assert rc == 0, f'{earlier[:8]} is not an ancestor of {later[:8]}'

        # ── the `_merge-verify` lane reads IDLE on BOTH axes ─────────────────
        from orchestrator.verify_cancel import (
            lane_lock_holder_pids_strict,
            lane_lock_path,
            read_lock_holder_pgid,
        )

        lock_path = lane_lock_path(scene.git_ops.persistent_merge_worktree_path)
        # An ABSENT lock file is the strongest form of idle — nothing in this
        # round ever opened the lane, let alone held it — but the strict reader
        # raises FileNotFoundError on it rather than returning [], so the two
        # shapes are spelled out separately instead of collapsed.
        assert (not lock_path.exists()) or lane_lock_holder_pids_strict(
            lock_path
        ) == [], 'the kernel flock axis must read idle after the round'
        assert read_lock_holder_pgid(scene.git_ops.worktree_base) is None, (
            'the rendezvous axis must read idle, or the next '
            'reset_persistent_merge_worktree raises MergeVerifyLeaseHeld'
        )

        assert _drain_residue(scene.worker) == {'105', '106'}
        scene.worker._running = True
        _assert_quiescent(
            scene.worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()),
        )

    async def test_a_failing_tip_lands_nothing_and_the_item_still_lands_later(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(b) A red tip is a NON-EVENT for the queue — and costs nobody a verdict.

        γ pinned "nothing lands on the fail arm" at the seam; this pins the
        ROUND consequence: the queue is exactly as it was, the halving state
        halved, and the very same request lands on the next round's own
        verdict.  A red tip that terminally failed anyone would be the
        false-green's mirror image — a false RED — and would feed the thrash
        ladder a signature every deep round.
        """
        scene = await _make_delta_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=4, script=[False, True], db_name='delta-e2e-fail.db',
        )
        from orchestrator.merge_types import InflightStatus

        main_before = await _rev_parse(git_repo, 'main')

        red = await scene.round_(tag='fail', head_tid='101')

        assert red['chain'] is not None and len(red['chain'].links) == 3
        assert red['result'].status is InflightStatus.REQUEUED, (
            f'the fail arm must stay non-adopting, got {red["result"].status!r}'
        )
        assert await _rev_parse(git_repo, 'main') == main_before, (
            'a red tip must not move main'
        )
        for tid in _DELTA_E2E_FOLLOWERS:
            assert not scene.reqs[tid].result.done(), (
                f'task {tid} was handed a verdict no verify produced'
            )
        assert not scene.reqs['101'].result.done()
        # Every FOLLOWER kept its exact lane slot — the chain took nothing.
        assert {
            lane: [r.task_id for r in scene.worker._lane_buffers[lane]]
            for lane in ('high', 'normal')
        } == {'high': [], 'normal': list(_DELTA_E2E_FOLLOWERS)}, (
            'the chain mutated the queue on a red tip'
        )
        # The head itself went back on `_queue` through the requeue chokepoint,
        # unresolved — "deferred", not "failed".
        assert scene.worker._queue.qsize() == 1
        requeued = scene.worker._queue.get_nowait()
        assert requeued is scene.reqs['101']
        assert not requeued.result.done()
        assert scene.worker._chain_halving_state == max(1, 4 // 2)

        # ── and now the ordinary path, on its own verdict ────────────────────
        green = await scene.round_(tag='fail', head_tid='101', req=requeued)
        assert scene.reqs['101'].result.done()
        assert scene.reqs['101'].result.result().status == 'done'
        assert await _rev_parse(git_repo, 'main') != main_before

        landed_by_chain = {
            tid for tid, _mc in (green['chain'].links if green['chain'] else [])
        }
        assert _drain_residue(scene.worker) == (
            set(_DELTA_E2E_FOLLOWERS) - landed_by_chain
        )
        scene.worker._running = True
        _assert_quiescent(
            scene.worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()),
        )

    async def test_the_shipped_kill_switch_reaches_no_delta_code(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(c) At ``chain_cap=0`` the round is byte-identical to pre-δ merging.

        The transcript is compared against a GOLDEN literal rather than
        against a second run, because "no δ code ran" is a claim about
        absences — no chain built, no chain handed down, no landing stamp, no
        halving state — and a literal states each absence by name.
        """
        scene = await _make_delta_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=0, script=[True], db_name='delta-e2e-killed.db',
        )
        rec = await scene.round_(tag='killed', head_tid='101')

        assert _delta_round_transcript(scene, 0) == {
            'chain': None,
            'chain_items': 1,
            'depth': 0,
            'probe_base': None,
            'verified_the_items_own_merge_commit': True,
            'result_status': rec['result'].status,
            'result_has_worktree': True,
            'advanced': True,
            'outcome_status': 'done',
            'landed_via_chain': None,
            'build_chain_calls': 0,
            'halving_state': None,
        }
        assert scene.lane_releases == [], 'no chain lane is taken at cap=0'
        # The ordinary path landed exactly ONE item; every follower is untouched.
        for tid in _DELTA_E2E_FOLLOWERS:
            assert not scene.reqs[tid].result.done()

        assert _drain_residue(scene.worker) == set(_DELTA_E2E_FOLLOWERS)
        scene.worker._running = True
        _assert_quiescent(
            scene.worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()),
        )

    async def test_flipping_the_cap_in_place_starts_landing_chains(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(d) HOT RELOAD: 0 -> 4 through the REAL `apply_reload`, no restart.

        ``merge_deep.chain_cap`` is a green-tier ``RELOADABLE_FIELDS`` leaf, and
        the worker reads it LIVE off the dispatching request's config
        (merge_queue.py:12348) — so the round after the flip must build and
        land a chain against the very same worker, queue and repo.  Driven
        through ``config.apply_reload`` rather than by assigning the attribute,
        because the operator-facing claim is about the reload path, not about
        Python attribute assignment.
        """
        from orchestrator.config import apply_reload

        scene = await _make_delta_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=0, script=[True, True], db_name='delta-e2e-reload.db',
            heads=('101', '107'),
        )
        cold = await scene.round_(tag='reload', head_tid='101')
        assert cold['chain'] is None and scene.built == []

        result = apply_reload(scene.config, _make_config(git_repo, chain_cap=4))
        assert result['reloaded'] is True
        assert 'merge_deep.chain_cap' in result['applied'], (
            f'chain_cap must be a green-tier leaf; got {result!r}'
        )
        assert result['restart_required'] == {}
        assert scene.config.merge_deep.chain_cap == 4

        hot = await scene.round_(tag='reload', head_tid='107')
        assert hot['chain'] is not None and len(hot['chain'].links) == 3, (
            'the very next round must build a chain — no restart'
        )
        landed = ['107', *[tid for tid, _mc in hot['chain'].links]]
        for tid in landed:
            assert scene.reqs[tid].result.result().status == 'done'
        for tid in landed[1:]:
            assert scene.reqs[tid].result.result().landed_via_chain == 1

        assert _drain_residue(scene.worker) == (
            set(_DELTA_E2E_FOLLOWERS) - set(landed[1:])
        )
        scene.worker._running = True
        _assert_quiescent(
            scene.worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()),
        )
