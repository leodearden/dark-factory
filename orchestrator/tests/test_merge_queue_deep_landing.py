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

from orchestrator import merge_queue
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
    chain = await worker._deep_chain_placement(item)
    assert chain is not None
    assert [tid for tid, _ in chain.links] == list(_DELTA_LINKS)
    assert chain.truncated_at == _DELTA_TRUNCATOR
    assert chain.truncated_reason == 'conflict'

    _spy_post_merge_verify(monkeypatch, outcome=verify_outcome)
    _spy_chain_lane_release(monkeypatch)
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
