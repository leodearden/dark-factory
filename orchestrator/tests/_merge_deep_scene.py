"""Shared scene layer for the deep merge-ahead test modules.

ONE home for the repo fixtures, builders, git helpers, spies, permit census and
durable-tier readers that test_merge_queue_deep_integration_gate.py and
test_merge_queue_deep_landing.py used to carry as code-identical clones.
Membership was decided by comparing the two modules' ASTs, docstrings stripped
and the ``merge_queue`` alias normalised: every helper here was identical in
both.  test_merge_deep_scene_dedup.py guards the single-sourcing.  Greek
letters (δ, η1) and "PRD decision #N" refer to plans/deep-merge-ahead-prd.md.

Deduplicate a BODY, never a NAME.  Some helpers share a name across consumers
but not a body, so they stay local to each consumer: ``_make_worker`` (each
builds a differently-defaulted lane), ``_assert_quiescent`` and
``_finalized_rows`` (different surfaces and readers), and the single-consumer
``_lane_of``, ``_spy_post_merge_verify`` and ``_fail_verify_result``.  The
invariant gate's ``_setup_repo``, ``git_repo`` and ``_make_req`` collide by
name only (a README-only repo with no recording seed, a positional ``lane``),
so that module imports nothing from here.

test_merge_queue_deep_dispatch.py and test_merge_queue_build_chain.py still
clone some of these helpers; task 5172 owns folding them in.

This module follows the flat-helper convention ``_merge_queue_harness.py``
documents: conftest.py puts the tests dir on ``sys.path``, so consumers import
it by bare module name, and the leading underscore keeps pytest from collecting
it.

CRITICAL: every ``merge_queue`` MODULE-LEVEL function is reached as
``_mq.<name>``, resolved on the live module at call time, never through a
from-import.  Consumers patch ``'orchestrator.merge_queue.<name>'``, and a
from-import would bind a reference here that such a patch cannot reach.
Classes constructed by value (``MergeRequest``, ``SpeculativeMergeWorker`` and
the rest) are never patched, so they stay ordinary from-imports.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal, TypedDict

import pytest

from orchestrator import merge_queue as _mq
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

# ── repo fixtures ────────────────────────────────────────────────────────────


async def _setup_repo(repo: Path) -> None:
    """Init a repo with a 20-line shared.txt plus disjoint.txt.

    ``shared.txt`` gets **20** numbered lines rather than 3: git's 3-line diff
    context window makes near-line edits in a tiny file conflict even when they
    touch different lines (gotcha documented in
    test_merge_queue_conflict_graph.py).  20 lines makes a line-1 vs line-15
    edit pair genuinely non-conflicting, so one seed can back both
    conflicting and non-conflicting chain fixtures.
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
    GitConfig field.  The pool is only constructed when ``size > 0 AND
    config.merge_spec_warm_lane_pool``.
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

    The main-health probe is switched off through its own operator flag: a red
    verdict here would otherwise spawn a DETACHED project-wide verify of the
    tmp repo that outlives the test (task 5811; conftest's leaked-task drain
    is the net under every scene that does not).
    """
    return OrchestratorConfig(
        project_root=repo,
        git=git_config or _make_spec_git_config(),
        merge_deep=MergeDeepConfig(chain_cap=chain_cap),
        escalate_preexisting_main_break=False,
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
    on, and therefore the only kind the chain walk ever sees;
    ``speculative=False`` is SLOT 1, the head trust-anchor verify against real
    main, which δ CANCELS and lands on the tip's authority (PRD decision #3).
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

    Real dispatch hands ``RealMergeItem.merge_wt`` an ephemeral ``_merge-<uuid>``
    minted by ``merge_to_main``.  Tests that reach code which DISPOSES of that
    worktree must not pass the repo root in its place: the chain arm calls
    ``_cleanup_owned_merge_worktree(item.merge_wt)``, whose rmtree fallback
    would then delete the fixture repo out from under the test — a destructive
    pass, not a real one.

    Deliberately NOT created on disk: materialising ``worktree_base`` by hand
    makes ``acquire_spec_lane``'s create-once pool-storage check see a
    directory it did not provision and cold-fall-back, silently turning every
    warm ``_spec-`` lane assertion in the scene vacuous.  A bare path is
    enough — disposal of a missing worktree is a no-op either way.
    """
    return git_ops.worktree_base / f'_merge-{tag}'


# ── event capture ────────────────────────────────────────────────────────────


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


async def _merge_commit_off_main(repo: Path, branch: str, label: str) -> str:
    """Return a REAL merge commit of *branch* into main, WITHOUT advancing main.

    Stands in for the dispatching item's ``merge_result.merge_commit``.  It must
    not be reachable from ``main``, or every "landed on main" assertion a
    caller makes would hold vacuously before the walk ever ran.
    """
    await _run(['git', 'checkout', '-b', f'_tmp-{label}', 'main'], cwd=repo)
    await _run(['git', 'merge', '--no-ff', '-m', f'merge {branch}', branch], cwd=repo)
    sha = await _rev_parse(repo)
    await _run(['git', 'checkout', 'main'], cwd=repo)
    return sha


async def _merge_commit_onto(
    repo: Path, branch: str, base_sha: str, label: str,
) -> str:
    """A REAL ``--no-ff`` merge commit of *branch* onto *base_sha*.

    The speculative-stacking twin of :func:`_merge_commit_off_main`: slot 2's
    item is merged onto the HEAD's merge commit, not onto main, which is
    exactly what makes the head's tree a SUBSET of the chain tip's — and
    therefore what makes landing the head on the tip's authority sound rather
    than merely convenient.
    """
    await _run(['git', 'checkout', '-b', f'_tmp-{label}', base_sha], cwd=repo)
    await _run(['git', 'merge', '--no-ff', '-m', f'merge {branch}', branch], cwd=repo)
    sha = await _rev_parse(repo)
    await _run(['git', 'checkout', 'main'], cwd=repo)
    return sha


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
    *raises*, or *bump_repo* (move main for REAL via
    :func:`_external_main_bump`, then fall through to the real
    ``advance_main``, which then refuses at its descendant check).

    Prefer *bump_repo* when a scene runs past the abort: a synthetic
    ``AdvanceOutcome`` states only the abort's CLASS, while a real ref move also
    leaves the repository in the state the NEXT round has to rebuild against.

    The ordinal is CUMULATIVE across the whole scene, because
    :func:`_spy_advance_main`'s ledger is: a hook armed at call 3 fires inside
    the first round's walk and never again, so a later round runs clean without
    the caller having to disarm anything.
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


# ── verify-lease / runner helpers ────────────────────────────────────────────


def _local_lease():
    """A LOCAL :class:`HostLease` whose runner is never actually driven.

    LOCAL deliberately, not remote: ``lease.is_local`` is precisely what selects
    the warm-swap block the chain arm SKIPS, and it is also the axis δ's
    head-cancel cleanliness argument splits on — a LOCAL head verify frees BOTH
    lease axes by construction when its task is cancelled (GitOps
    ``merge_verify_lease``'s finally), while a REMOTE one SIGKILLs and leaks the
    fixed-key holder-pgid rendezvous file unless ``cli.py cancel_verify``
    clears it.
    """
    from unittest.mock import MagicMock

    from orchestrator.verify_runner import HostLease

    runner = MagicMock()
    runner.name = 'local'
    runner.is_local = True
    return HostLease(name='local', runner=runner, is_local=True)


def _fake_pass_runner(name: str = 'fake-runner'):
    """A RemoteRunner-shaped fake whose ``run_merge_verify`` always passes."""
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


# ── dispatch-scene spies ─────────────────────────────────────────────────────


def _spy_chain_lane_release(monkeypatch) -> list[tuple]:
    """Record ``release_chain_build_lane`` calls WITHOUT suppressing them.

    Passthrough, not a stub: the lane must genuinely go back to FREE, so the
    pool-state assertions stay real while the call count stays observable.
    The ChainResult "Lane ownership" contract is EXACTLY-once.
    """
    calls: list[tuple] = []
    real = _mq.release_chain_build_lane

    async def _recording(git_ops, lane, *, warm):
        calls.append((lane, warm))
        return await real(git_ops, lane, warm=warm)

    monkeypatch.setattr(
        'orchestrator.merge_queue.release_chain_build_lane', _recording,
    )
    return calls


def _spy_advance_main(git_ops: GitOps, monkeypatch, *, hook=None) -> list[tuple]:
    """Record every ``advance_main`` call as ``(merge_sha, expected_main, wt)``.

    Spied on the GitOps INSTANCE (not a merge_queue module reach-back, which
    test_merge_queue_reachback_patch_guard.py freezes) and PASSTHROUGH, so main
    really moves and the recorded ``expected_main`` chain can be checked against
    real history.

    *hook*, when given, is awaited with the 1-based call ordinal BEFORE the
    passthrough.  Returning ``None`` falls through to the real ``advance_main``;
    returning an :class:`~orchestrator.git_ops.AdvanceOutcome` (or raising)
    SHORT-CIRCUITS that one call, which is how a scenario injects the mid-walk
    failure it is about.
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


# ── permit census / residue drain ────────────────────────────────────────────


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
    the hazard is not "how many permits" but "whose".  A walk that released a
    link's token would raise ``AssertionError`` (the token was never issued),
    while a walk that released the DISPATCHING item's token early would keep
    every count plausible and break only ownership — invisible to a size
    comparison, obvious to a set one.

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
    """Retire whatever a round deliberately LEFT queued; return its task ids.

    The δ contract is "the walk touches the prefix and NOTHING else": the
    truncator, and every link past an abort, stay buffered with unresolved
    futures for their ordinary sequential path on a later round.  A scene that
    stops after a finalize therefore rests with real, INTENDED residue — so the
    whole-registry surfaces of a consumer's quiescence oracle (every future
    resolved, nothing non-terminal) cannot hold until that residue is taken off
    the pipeline the way a later round would take it.

    The RETURNED SET is what makes this safe rather than a whitewash — every
    caller asserts it equals the residue the run promised BEFORE trusting the
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


# ── durable-tier readers ─────────────────────────────────────────────────────


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


# ── the SHIPPED canary arithmetic, transcribed ───────────────────────────────


def _canary_predicate_items_per(
    merge_verify_data: list[dict], merge_finalized_data: list[dict],
) -> float | None:
    """η1's ``items_per`` statistic, transcribed from the SHIPPED predicate.

    Transcribed from ``scripts/merge-deep-canary-predicate.sh`` — already
    COMMITTED CODE on main — rather than restated in the assertion's own words,
    following test_merge_queue_deep_dispatch.py::_canary_says_deep's precedent.
    This is what PINS
    ``landed_via_chain``'s numeric encoding: the shipped comment calls the
    result "items landed per deep verify run", and that arithmetic is only
    correct if the per-walk contributions SUM to the number of items the walk
    landed.  Emitting the chain size k on every one of k items would yield
    k²/n_deep; emitting 1-indexed positions would yield k(k+1)/2.

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
