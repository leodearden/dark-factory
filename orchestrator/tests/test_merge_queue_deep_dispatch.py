"""Deep merge-ahead γ — deep-tip verify dispatch + halving state (task 3185).

PRD: ``plans/deep-merge-ahead-prd.md`` task γ (Phase 2 vertical slice).
Capability sidecar: ``plans/deep-merge-ahead-prd.capability-manifest.yaml``.

γ is the FIRST production caller of β's ``build_chain`` (task 3184).  It adds
the dispatch-side half of deep merge-ahead: a slot-2 gate that builds a chain,
redirects the verify onto the chain TIP, and records a pass/fail into a halving
state machine.  It deliberately does NOT land anything — δ (task 3186) owns the
in-order prefix CAS walk.  See this module's step-17 class for the soundness
fence that keeps a tip verdict out of ``_finalize_inflight``'s CAS advance.

Step → coverage map:
  step-01 RED — ``select_chain_depth`` / ``next_halving_state`` pure policy
  step-03 RED — worker-resident halving state (``_chain_halving_state``,
                ``_note_chain_outcome``, ``_deep_target_depth``)
  step-11 RED — ``_deep_chain_placement`` gate + kill-switch zero-cost proof
  step-13 RED — the bounded ``build_chain`` call (timeout, empty, truncated)
  step-15 RED — the dispatch REDIRECT (verify ``chain.tip`` in ``chain.lane``)
  step-17 RED — the NON-ADOPTION invariant (γ's soundness fence)
  step-19 RED — integration: halving walk, d=1 floor byte-identity,
                ``chain_items >= 1`` everywhere, conservation

Harness notes (see plan pre-1):
  * ``orchestrator/pyproject.toml`` does NOT set ``asyncio_mode`` → pytest-asyncio
    runs STRICT, so ``@pytest.mark.asyncio`` is required on async test classes.
  * That same config turns "marked with @pytest.mark.asyncio but not an async
    function" into an ERROR — never put a sync ``test_*`` inside a marked class.
    Sync tests live in their OWN unmarked class.
  * Default per-test ``timeout = 60``; any class doing real-git worktree/merge
    work carries ``@pytest.mark.timeout(180)``.
  * ``orchestrator/tests/`` has no ``__init__.py``, so flat helpers are imported
    by bare module name.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import pytest

from orchestrator import merge_liveness, merge_queue
from orchestrator.config import GitConfig, MergeDeepConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeRequest, SpeculativeMergeWorker
from orchestrator.merge_types import (
    ChainResult,
    MergeResult,
    QueuedBranch,
    RealMergeItem,
)

# ── repo fixtures (adapted from test_merge_queue_build_chain.py:47-102) ───────


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


# ── config / GitOps helpers (adapted from build_chain's) ──────────────────────


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
    predecessor's merge commit); ``speculative=False`` is SLOT 1, the head
    trust-anchor verify against real main, which γ never chains.
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
    worktree must not pass the repo root in its place: γ's chain arm calls
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


# ── event capture (from test_merge_queue_depth_telemetry.py:163-181) ──────────


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


# ── git / lane spies ─────────────────────────────────────────────────────────


def _count_spec_lane_acquires(git_ops: GitOps, monkeypatch) -> list[str]:
    """Wrap ``acquire_spec_lane`` with a call recorder; return the record list.

    Template: test_merge_queue_build_chain.py:951.  The kill-switch tests below
    assert this list stays EMPTY — the cap=0 path must cost nothing, not merely
    do nothing.
    """
    calls: list[str] = []
    original = git_ops.acquire_spec_lane

    async def _recording(merge_commit: str):
        calls.append(merge_commit)
        return await original(merge_commit)

    monkeypatch.setattr(git_ops, 'acquire_spec_lane', _recording)
    return calls


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


async def _worktree_names(git_ops: GitOps) -> set[str]:
    """Return the set of registered worktree directory names."""
    _, out, _ = await _run(
        ['git', 'worktree', 'list', '--porcelain'],
        cwd=git_ops.project_root,
    )
    names: set[str] = set()
    for line in out.splitlines():
        if line.startswith('worktree '):
            wt_path = Path(line[len('worktree '):].strip())
            names.add(wt_path.name)
    return names


async def _rev_parse(cwd: Path, rev: str = 'HEAD') -> str:
    """Return the stripped SHA of *rev* resolved inside *cwd*."""
    _, sha, _ = await _run(['git', 'rev-parse', rev], cwd=cwd)
    return sha.strip()


def _shared_txt_with(line_no: int, text: str) -> str:
    """Return a 20-line shared.txt body with line *line_no* replaced by *text*."""
    lines = [f'line{i}\n' for i in range(1, 21)]
    lines[line_no - 1] = f'{text}\n'
    return ''.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# step-01: RED — pure dispatch policy (select_chain_depth / next_halving_state)
#
# Sync class, deliberately UNMARKED (see the module docstring's harness note):
# both functions are pure, so they must be callable with no running event loop.
# ═══════════════════════════════════════════════════════════════════════════


class TestSelectChainDepth:
    """``select_chain_depth(chain_cap, queue_len, halving_state) -> int | None``.

    The PRD's dispatch invariant ``target_depth = min(len(queue), cap,
    halving_state)`` plus its two gates (``cap > 0``, ``queue >= 2``) and its
    ``< 2 → None`` floor.  Units are 1-INDEXED item counts: item #1 is the
    dispatching slot-2 item itself.
    """

    def test_cap_zero_is_the_kill_switch_for_every_input(self) -> None:
        """cap=0 (α's shipped default) → None for EVERY queue_len/halving_state."""
        from orchestrator.merge_queue import select_chain_depth

        for queue_len in (0, 1, 2, 5, 50, 1000):
            for halving_state in (None, 1, 2, 3, 6, 999):
                assert select_chain_depth(0, queue_len, halving_state) is None, (
                    f'cap=0 must gate off at queue_len={queue_len}, '
                    f'halving_state={halving_state}'
                )

    def test_queue_shorter_than_two_never_chains(self) -> None:
        """The PRD's ``queue >= 2`` gate: a lone item is not a chain."""
        from orchestrator.merge_queue import select_chain_depth

        for cap in (1, 2, 6, 100):
            for halving_state in (None, 1, 6):
                assert select_chain_depth(cap, 0, halving_state) is None
                assert select_chain_depth(cap, 1, halving_state) is None

    def test_reset_state_evaluates_min_queue_cap(self) -> None:
        """``halving_state=None`` is the reset sentinel → ``min(queue_len, cap)``.

        Evaluated at DISPATCH time, not frozen at pass time, so a queue that
        grew since the last pass is honoured (plan design decision 4).
        """
        from orchestrator.merge_queue import select_chain_depth

        assert select_chain_depth(6, 4, None) == 4     # queue binds
        assert select_chain_depth(6, 10, None) == 6    # cap binds
        assert select_chain_depth(6, 6, None) == 6     # tie

    def test_halving_state_binds_when_smallest(self) -> None:
        from orchestrator.merge_queue import select_chain_depth

        assert select_chain_depth(6, 10, 3) == 3       # halving binds
        assert select_chain_depth(6, 10, 6) == 6       # equal to cap
        assert select_chain_depth(6, 2, 3) == 2        # queue still binds
        assert select_chain_depth(3, 10, 6) == 3       # cap still binds

    def test_target_below_two_is_the_d1_floor(self) -> None:
        """Any combination whose min is < 2 → None: no chain code runs at all.

        This is what makes "the floor is byte-identical to today's adjacent
        verify" true BY CONSTRUCTION rather than by careful mimicry.
        """
        from orchestrator.merge_queue import select_chain_depth

        assert select_chain_depth(1, 10, None) is None   # cap=1
        assert select_chain_depth(6, 10, 1) is None      # halving_state=1
        assert select_chain_depth(6, 2, 1) is None       # both floor-ish
        assert select_chain_depth(1, 2, 1) is None


class TestNextHalvingState:
    """``next_halving_state(passed, dispatched_depth) -> int | None`` (PRD dec. 5).

    Fail at d halves to ``max(1, d // 2)``; a pass resets to the ``None``
    sentinel (re-evaluated as ``min(queue, cap)`` at the next dispatch).
    """

    def test_fail_halves_with_a_floor_of_one(self) -> None:
        from orchestrator.merge_queue import next_halving_state

        assert next_halving_state(False, 6) == 3
        assert next_halving_state(False, 3) == 1
        assert next_halving_state(False, 2) == 1
        # The max(1, ...) floor never yields 0 or a negative state.
        assert next_halving_state(False, 1) == 1

    def test_pass_resets_to_the_none_sentinel(self) -> None:
        from orchestrator.merge_queue import next_halving_state

        for depth in (1, 2, 3, 6, 50):
            assert next_halving_state(True, depth) is None


class TestHalvingWalkComposition:
    """The composed walk — the PRD's "Halving walk isolates bad item" row.

    Starting from the reset sentinel at cap=6 / queue_len=10, successive FAILs
    step 6 → 3 → floor(None, no chain), and a PASS anywhere returns the next
    target to ``min(queue_len, cap)``.
    """

    def test_successive_fails_walk_six_three_then_floor(self) -> None:
        from orchestrator.merge_queue import next_halving_state, select_chain_depth

        cap, queue_len = 6, 10
        state: int | None = None
        walk: list[int | None] = []

        for _ in range(3):
            target = select_chain_depth(cap, queue_len, state)
            walk.append(target)
            if target is None:
                break  # floor reached: no chain dispatched, so nothing to halve
            state = next_halving_state(False, target)

        assert walk == [6, 3, None]
        assert state == 1, 'the halving state rests at the floor, never 0'

    def test_a_pass_resets_the_walk_to_min_queue_cap(self) -> None:
        from orchestrator.merge_queue import next_halving_state, select_chain_depth

        cap, queue_len = 6, 10
        state = next_halving_state(False, 6)       # one fail: 6 → 3
        assert select_chain_depth(cap, queue_len, state) == 3

        state = next_halving_state(True, 3)        # a pass at 3 resets
        assert state is None
        assert select_chain_depth(cap, queue_len, state) == 6

    def test_reset_re_evaluates_a_grown_queue(self) -> None:
        """A pass stores None, so a queue that grew is seen on the next round."""
        from orchestrator.merge_queue import next_halving_state, select_chain_depth

        cap = 6
        state = next_halving_state(True, 2)
        assert select_chain_depth(cap, 3, state) == 3   # short queue binds
        assert select_chain_depth(cap, 9, state) == 6   # grown queue: cap binds


# ═══════════════════════════════════════════════════════════════════════════
# step-03: RED — worker-resident halving state
#
# Sync class (unmarked): _note_chain_outcome / _deep_target_depth are pure and
# synchronous, matching the _record_verify_outcome / _recent_verify_fail_rate
# pair they sit beside.  Constructing a worker needs no running loop.
# ═══════════════════════════════════════════════════════════════════════════


class TestChainHalvingWorkerState:
    """``_chain_halving_state`` + its single mutation chokepoint.

    The worker owns the LIVE bisector state that the pure
    :func:`select_chain_depth` policy consumes, exactly as it owns
    ``_probe_round_counter`` for :func:`select_probe_depth`.
    """

    def _worker(self, tmp_path: Path) -> SpeculativeMergeWorker:
        return _make_worker(_make_git_ops(tmp_path, pool=False, size=0))

    def test_state_is_none_at_construction(self, tmp_path: Path) -> None:
        """A fresh (or restarted) worker has never failed → the reset sentinel."""
        worker = self._worker(tmp_path)

        assert worker._chain_halving_state is None

    def test_note_chain_outcome_halves_on_fail(self, tmp_path: Path) -> None:
        worker = self._worker(tmp_path)

        worker._note_chain_outcome(False, 6)
        assert worker._chain_halving_state == 3

        worker._note_chain_outcome(False, 3)
        assert worker._chain_halving_state == 1

        # The floor holds: a fail at 1 stays at 1, never 0.
        worker._note_chain_outcome(False, 1)
        assert worker._chain_halving_state == 1

    def test_note_chain_outcome_resets_on_pass(self, tmp_path: Path) -> None:
        worker = self._worker(tmp_path)

        worker._note_chain_outcome(False, 6)
        assert worker._chain_halving_state == 3

        worker._note_chain_outcome(True, 3)
        assert worker._chain_halving_state is None

    def test_note_chain_outcome_is_synchronous(self, tmp_path: Path) -> None:
        """Returns None, not a coroutine — it is a pure state fold, not I/O."""
        worker = self._worker(tmp_path)

        assert worker._note_chain_outcome(False, 4) is None

    def test_deep_target_depth_composes_live_state_with_the_formula(
        self, tmp_path: Path,
    ) -> None:
        """``_deep_target_depth(cap, queue_len)`` == min(queue, cap, state).

        Asserted against LITERAL expected values, not against
        ``select_chain_depth(...)`` re-evaluated with the same arguments: that
        earlier shape was ``x == x`` and passed even when both sides were
        wrong.  What actually needs pinning is that the method reads the LIVE
        halving state (rather than, say, a stale copy or ``None``), which the
        before/after pair below makes observable.
        """
        worker = self._worker(tmp_path)

        assert worker._deep_target_depth(0, 10) is None    # kill switch
        assert worker._deep_target_depth(6, 1) is None     # queue >= 2 gate
        assert worker._deep_target_depth(1, 10) is None    # d=1 floor
        assert worker._deep_target_depth(6, 4) == 4        # queue binds
        assert worker._deep_target_depth(6, 10) == 6       # cap binds

        worker._note_chain_outcome(False, 6)               # state → 3
        assert worker._chain_halving_state == 3
        assert worker._deep_target_depth(6, 10) == 3, 'live state now binds'
        assert worker._deep_target_depth(6, 2) == 2, 'queue still binds under it'
        assert worker._deep_target_depth(3, 10) == 3, 'cap ties with the state'

    def test_scripted_round_sequence_end_to_end(self, tmp_path: Path) -> None:
        """fail(6) → 3 → fail(3) → floor(None) → pass(1) → back to min(queue, cap).

        The worker-level restatement of the PRD's halving-walk row.
        """
        worker = self._worker(tmp_path)
        cap, queue_len = 6, 10

        assert worker._deep_target_depth(cap, queue_len) == 6
        worker._note_chain_outcome(False, 6)

        assert worker._deep_target_depth(cap, queue_len) == 3
        worker._note_chain_outcome(False, 3)

        # Floor: no chain is dispatched this round at all.
        assert worker._deep_target_depth(cap, queue_len) is None

        # A passing ordinary (d=1) verify clears the bisector.
        worker._note_chain_outcome(True, 1)
        assert worker._chain_halving_state is None
        assert worker._deep_target_depth(cap, queue_len) == 6

    def test_does_not_touch_neighbouring_probe_state(self, tmp_path: Path) -> None:
        """Negative control: the deep policy and the probe policy stay decoupled.

        ``_recent_verify_outcomes`` (the probe's flake-rate window) and
        ``_probe_round_counter`` (its round index) are the OTHER second-slot
        state living in the same __init__ block; a deep outcome must not
        silently perturb the probe's cadence or suppression signal.
        """
        worker = self._worker(tmp_path)
        outcomes_before = list(worker._recent_verify_outcomes)
        counter_before = worker._probe_round_counter

        worker._note_chain_outcome(False, 6)
        worker._note_chain_outcome(True, 2)

        assert list(worker._recent_verify_outcomes) == outcomes_before
        assert worker._probe_round_counter == counter_before


# ═══════════════════════════════════════════════════════════════════════════
# step-11: RED — the deep GATE, and the kill-switch byte-identity guarding it
#
# `_deep_chain_placement(item)` is the slot-2 policy hook: the deep twin of
# `_probe_verify_placement` (merge_queue.py:11988), structured identically —
# a slot-1 guard, then a HOISTED zero-cost kill-switch guard, then the pure
# policy, then (step-14) the bounded git work.
#
# Everything below asserts a NEGATIVE — the gate declines — so each test also
# proves the decline COSTS NOTHING: no `build_chain`, no `_spec-` lane, and,
# on the two guards that precede it, not even the O(n) `chain_snapshot()`
# scan. "No cost, not merely no effect" is the byte-identity claim the PRD
# makes for the shipped `chain_cap=0` default, and it is only checkable by
# spying the work that must NOT happen.
# ═══════════════════════════════════════════════════════════════════════════


def _spy_build_chain(
    monkeypatch,
    result: ChainResult | None = None,
    *,
    passthrough: bool = False,
) -> list[dict]:
    """Replace ``orchestrator.merge_queue.build_chain`` with a recorder.

    Returns the list of recorded call kwargs (``{'git_ops', 'queue_snapshot',
    'head_merge_commit', 'cap', 'target_depth'}``) — step-11 asserts it stays
    EMPTY on every declining path; step-13 reads its contents.

    *result* defaults to a ZERO-LINK :class:`ChainResult`, which per its
    docstring never holds a lane — so a stubbed call can never leak one, and
    these gate tests stay valid once step-14 lands the real build.

    ``passthrough=True`` records the arguments and then delegates to the REAL
    builder, so one test can pin both the call contract and the built chain
    without the two drifting apart.
    """
    calls: list[dict] = []
    real = merge_queue.build_chain

    async def _recording(git_ops, queue_snapshot, head_merge_commit, *, cap, target_depth):
        calls.append({
            'git_ops': git_ops,
            'queue_snapshot': tuple(queue_snapshot),
            'head_merge_commit': head_merge_commit,
            'cap': cap,
            'target_depth': target_depth,
        })
        if passthrough:
            return await real(
                git_ops, queue_snapshot, head_merge_commit,
                cap=cap, target_depth=target_depth,
            )
        return result if result is not None else ChainResult(links=[], tip=head_merge_commit)

    monkeypatch.setattr('orchestrator.merge_queue.build_chain', _recording)
    return calls


def _spy_chain_snapshot(worker: SpeculativeMergeWorker, monkeypatch) -> list[int]:
    """Wrap ``worker.chain_snapshot`` with a call recorder; return the record list.

    ``chain_snapshot()`` is an O(n) walk of every lane buffer, so it is the
    cheapest observable proof that a guard short-circuited BEFORE the scan
    rather than after it — the difference between the kill switch costing
    nothing and merely doing nothing.
    """
    seen: list[int] = []
    original = worker.chain_snapshot

    def _recording():
        out = original()
        seen.append(len(out))
        return out

    monkeypatch.setattr(worker, 'chain_snapshot', _recording)
    return seen


@pytest.mark.asyncio
class TestDeepChainPlacementGate:
    """`_deep_chain_placement` declines — and costs nothing while declining."""

    async def _fixture(
        self, git_repo: Path, monkeypatch, *, chain_cap: int, queued: int,
    ) -> tuple[SpeculativeMergeWorker, RealMergeItem, OrchestratorConfig, list[dict], list[str], list[int]]:
        """Build worker + a slot-2 item + *queued* buffered requests, all spied.

        Returns ``(worker, item, config, build_chain_calls, spec_lane_acquires,
        chain_snapshot_calls)``.  No real merge commit is needed: every
        step-11 path declines before touching git, and a test that reached git
        with this fake SHA would fail loudly rather than silently pass.
        """
        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo, chain_cap=chain_cap)
        worker = _make_worker(git_ops)
        worker._lane_buffers['normal'].extend(
            _make_req(str(200 + i), str(200 + i), config, git_repo)
            for i in range(queued)
        )
        item = _make_item(
            _make_req('101', '101', config, git_repo), 'beef' * 10, git_repo,
        )
        return (
            worker, item, config,
            _spy_build_chain(monkeypatch),
            _count_spec_lane_acquires(git_ops, monkeypatch),
            _spy_chain_snapshot(worker, monkeypatch),
        )

    async def test_cap_zero_kill_switch_costs_nothing(self, git_repo, monkeypatch):
        """cap=0 (the shipped default) declines without ANY work, deep queue or not.

        The hoisted guard must sit BEFORE the ``chain_snapshot()`` scan, so
        under stock config a speculative dispatch pays literally nothing for
        the feature being compiled in.
        """
        worker, item, _cfg, built, lanes, scans = await self._fixture(
            git_repo, monkeypatch, chain_cap=0, queued=5,
        )

        assert await worker._deep_chain_placement(item) is None
        assert built == []
        assert lanes == []
        assert scans == [], 'cap=0 must short-circuit BEFORE the O(n) queue scan'

    async def test_non_speculative_item_is_never_chained(self, git_repo, monkeypatch):
        """Slot 1 — the head trust anchor — is unchanged by contract, at ANY cap.

        Its tree is merged onto REAL main; chaining onto it would make the one
        verify the pipeline trusts for landing a cumulative tree instead.
        """
        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo, chain_cap=6)
        worker = _make_worker(git_ops)
        worker._lane_buffers['normal'].extend(
            _make_req(str(300 + i), str(300 + i), config, git_repo) for i in range(5)
        )
        built = _spy_build_chain(monkeypatch)
        lanes = _count_spec_lane_acquires(git_ops, monkeypatch)
        scans = _spy_chain_snapshot(worker, monkeypatch)
        head_item = _make_item(
            _make_req('101', '101', config, git_repo), 'beef' * 10, git_repo,
            speculative=False,
        )

        assert await worker._deep_chain_placement(head_item) is None
        assert built == []
        assert lanes == []
        assert scans == [], 'the slot-1 guard must precede the queue scan too'

    async def test_queue_shorter_than_two_declines(self, git_repo, monkeypatch):
        """An empty unfrozen suffix means ``queue_len == 1`` — nothing to chain ONTO.

        Distinct from the kill switch: here the gate is ENABLED, so the scan
        legitimately runs and it is the pure ``queue >= 2`` policy gate that
        declines.  Asserting the scan DID happen is what stops this test from
        passing for the wrong reason (an over-eager cap guard).
        """
        worker, item, _cfg, built, lanes, scans = await self._fixture(
            git_repo, monkeypatch, chain_cap=6, queued=0,
        )

        assert await worker._deep_chain_placement(item) is None
        assert built == []
        assert lanes == []
        assert scans, 'an enabled cap must reach the queue scan before declining'

    async def test_d1_floor_runs_no_chain_code(self, git_repo, monkeypatch):
        """At the halving floor the gate declines, so dispatch IS today's path.

        ``_chain_halving_state == 1`` makes ``select_chain_depth`` return
        ``None`` (its ``< 2`` floor), and because the gate then declines
        before building anything, the floor round is byte-identical to the
        ordinary adjacent speculative verify BY CONSTRUCTION.
        """
        worker, item, _cfg, built, lanes, _scans = await self._fixture(
            git_repo, monkeypatch, chain_cap=6, queued=5,
        )
        worker._chain_halving_state = 1

        assert await worker._deep_chain_placement(item) is None
        assert built == []
        assert lanes == [], 'the d=1 floor must not acquire a chain-build lane'

    async def test_delegates_to_the_pure_policy_counting_the_dispatching_item(
        self, git_repo, monkeypatch,
    ):
        """The gate consults ``_deep_target_depth(cap, 1 + len(snapshot))``.

        Two contracts in one call record:

          * DELEGATION — the ``min(queue_len, cap, halving_state)`` invariant
            is written down once, in :func:`select_chain_depth`; the gate must
            route through :meth:`_deep_target_depth` rather than re-derive it.
          * UNIT — ``queue_len`` is 1-INDEXED and counts the DISPATCHING item
            as chain item #1, so 5 buffered requests give 6, not 5.  Getting
            this off by one would silently shorten (or over-run) every chain.

        This also keeps ``test_d1_floor_runs_no_chain_code`` honest: it proves
        the gate reaches the policy at all, so the floor test cannot pass
        merely because the gate is wedged shut for every halving state.
        """
        worker, item, _cfg, _built, _lanes, _scans = await self._fixture(
            git_repo, monkeypatch, chain_cap=6, queued=5,
        )
        seen: list[tuple[int, int]] = []
        original = worker._deep_target_depth
        monkeypatch.setattr(
            worker, '_deep_target_depth',
            lambda cap, queue_len: (seen.append((cap, queue_len)),
                                    original(cap, queue_len))[1],
        )
        worker._chain_halving_state = 2

        await worker._deep_chain_placement(item)

        assert seen == [(6, 6)], 'cap read live; queue_len counts the dispatcher'

    async def test_cap_is_read_live_off_the_item_config(self, git_repo, monkeypatch):
        """PRD decision 7: flipping the knob flips the gate with no worker rebuild.

        ``chain_cap`` is green-tier hot-reloadable, and ``_set_leaf`` mutates
        the held :class:`MergeDeepConfig` IN PLACE (config.py:1011-1017), so
        the gate must re-read ``item.request.config.merge_deep.chain_cap`` on
        every call rather than cache it at construction.  Same worker, same
        item, same config OBJECT across both calls — only the leaf changes.
        """
        worker, item, config, built, _lanes, scans = await self._fixture(
            git_repo, monkeypatch, chain_cap=0, queued=5,
        )

        assert await worker._deep_chain_placement(item) is None
        assert scans == [] and built == []


        config.merge_deep.chain_cap = 6  # operator hot-reloads the knob ON
        await worker._deep_chain_placement(item)

        assert scans, 'the flipped-on cap must be observed on the very next call'


# ═══════════════════════════════════════════════════════════════════════════
# step-13: RED — the BOUNDED build
#
# β made bounding γ's obligation in build_chain's own docstring (:6212-6224):
# one lane acquisition plus up to `depth` sequential `git merge` subprocesses,
# with NO internal deadline BY DESIGN — "the honest bound belongs to the policy
# layer". These tests pin that bound, and pin that every degenerate outcome of
# the build (deadline, empty, truncated) degrades to a path that leaks nothing.
# ═══════════════════════════════════════════════════════════════════════════


async def _merge_commit_off_main(repo: Path, branch: str, label: str) -> str:
    """Return a REAL merge commit of *branch* into main, WITHOUT advancing main.

    Stands in for the dispatching item's ``merge_result.merge_commit``.  It must
    not be reachable from ``main`` and must not equal ``frozen_prefix_tip``, or
    the "the chain continues from the DISPATCHING item, never from the frozen
    prefix tip" assertion below would hold vacuously.
    """
    await _run(['git', 'checkout', '-b', f'_tmp-{label}', 'main'], cwd=repo)
    await _run(['git', 'merge', '--no-ff', '-m', f'merge {branch}', branch], cwd=repo)
    sha = await _rev_parse(repo)
    await _run(['git', 'checkout', 'main'], cwd=repo)
    return sha


def _lane_states(git_ops: GitOps) -> list:
    """Return the ``_spec-`` pool's lane states (``[]`` when there is no pool)."""
    pool = git_ops.spec_warm_lane_pool
    return [] if pool is None else list(pool._lanes.values())


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestDeepChainPlacementBuild:
    """The gate's build: exact call contract, the deadline, and the leak-free exits."""

    async def _fixture(
        self, git_repo: Path, *, chain_cap: int, queued: list[tuple[str, str, str]],
    ) -> tuple[GitOps, SpeculativeMergeWorker, RealMergeItem, str, OrchestratorConfig]:
        """Build a repo + worker + slot-2 item sitting on a REAL merge commit.

        *queued* is ``[(task_id, filename, content), ...]`` — each becomes a
        real branch AND a buffered request, in that order.
        """
        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo, chain_cap=chain_cap)
        await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n')
        for task_id, filename, content in queued:
            await _create_branch_editing(git_repo, f'task/{task_id}', filename, content)
        head = await _merge_commit_off_main(git_repo, 'task/101', '101')
        worker = _make_worker(git_ops)
        worker._lane_buffers['normal'].extend(
            _make_req(task_id, task_id, config, git_repo) for task_id, _, _ in queued
        )
        item = _make_item(_make_req('101', '101', config, git_repo), head, git_repo)
        return git_ops, worker, item, head, config

    async def test_build_chain_is_called_with_the_exact_contract(
        self, git_repo: Path, monkeypatch,
    ):
        """The four arguments, and the UNIT CONVERSION between the two depth spaces.

        ``select_chain_depth`` counts ITEMS IN THE CHAIN (the dispatching item
        is #1); ``build_chain`` counts ADDITIONAL LINKS BEYOND the base it is
        handed.  The gate is the seam between those units, so it must pass
        ``cap - 1`` / ``d - 1`` — an off-by-one here silently builds one item
        too many (over-running the operator's cap) or one too few.

        ``head_merge_commit`` is the DISPATCHING ITEM's merge commit, not
        ``frozen_prefix_tip``: the chain must be a contiguous continuation of
        the queue through this item, and seeding it at the frozen tip would
        skip the very item being dispatched — a hole that breaks the in-order
        prefix property δ's CAS walk depends on.
        """
        git_ops, worker, item, head, _cfg = await self._fixture(
            git_repo, chain_cap=6,
            queued=[('102', 'b.txt', 'edit-102\n'), ('103', 'c.txt', 'edit-103\n')],
        )
        built = _spy_build_chain(monkeypatch, passthrough=True)
        expected_snapshot = worker.chain_snapshot()
        main_sha = await _rev_parse(git_repo, 'main')

        res = await worker._deep_chain_placement(item)

        assert len(built) == 1
        call = built[0]
        assert call['git_ops'] is git_ops
        assert call['queue_snapshot'] == expected_snapshot
        assert call['head_merge_commit'] == head == item.merge_result.merge_commit
        assert call['head_merge_commit'] != worker.frozen_prefix_tip(main_sha)
        # queue_len = 1 + 2 = 3, cap 6, no halving -> d = 3 (items).
        assert call['cap'] == 5, 'cap - 1: extra links beyond the dispatching item'
        assert call['target_depth'] == 2, 'd - 1, same unit conversion'
        assert res is not None
        await merge_liveness.release_chain_build_lane(
            git_ops, res.lane, warm=res.lane_warm,
        )

    async def test_clean_chain_returns_two_links_in_one_lane(
        self, git_repo: Path, monkeypatch,
    ):
        """A clean 3-item fixture builds a real 2-link chain holding exactly ONE lane."""
        from orchestrator.warm_lane_pool import LaneState

        git_ops, worker, item, head, _cfg = await self._fixture(
            git_repo, chain_cap=6,
            queued=[('102', 'b.txt', 'edit-102\n'), ('103', 'c.txt', 'edit-103\n')],
        )
        before = await _worktree_names(git_ops)
        lanes = _count_spec_lane_acquires(git_ops, monkeypatch)

        res = await worker._deep_chain_placement(item)

        assert res is not None
        assert [tid for tid, _ in res.links] == ['102', '103']
        assert res.tip == res.links[-1][1]
        assert res.truncated_at is None
        # The tip is a REAL descendant of the dispatching item's merge commit,
        # which is what makes one verify on it authorise the whole prefix.
        code, _, _ = await _run(
            ['git', 'merge-base', '--is-ancestor', head, res.tip], cwd=git_repo,
        )
        assert code == 0
        assert res.tip != head

        assert len(lanes) == 1, 'exactly ONE lane acquisition for the whole chain'
        assert await _worktree_names(git_ops) - before == {'_spec-0'}
        assert res.lane == git_ops.worktree_base / '_spec-0'
        assert _lane_states(git_ops) == [LaneState.ASSIGNED], 'result HOLDS its lane'

        await merge_liveness.release_chain_build_lane(
            git_ops, res.lane, warm=res.lane_warm,
        )

    async def test_build_is_wrapped_in_a_deadline_and_leaks_no_lane(
        self, git_repo: Path, monkeypatch,
    ):
        """A build that overruns ``CHAIN_BUILD_TIMEOUT_SECS`` degrades to today's path.

        The REAL ``build_chain`` runs here (only the per-item merge is stalled),
        so the deadline genuinely cancels it mid-merge and its own
        ``except BaseException`` arm (merge_queue.py:6350-6355) is what returns
        the lane.  Stubbing the builder wholesale would prove only that a stub
        does not leak.
        """
        from orchestrator.warm_lane_pool import LaneState

        git_ops, worker, item, _head, _cfg = await self._fixture(
            git_repo, chain_cap=6,
            queued=[('102', 'b.txt', 'edit-102\n'), ('103', 'c.txt', 'edit-103\n')],
        )
        assert isinstance(merge_queue.CHAIN_BUILD_TIMEOUT_SECS, (int, float))
        assert merge_queue.CHAIN_BUILD_TIMEOUT_SECS > 0
        # 2.0s, not a tighter value: the deadline starts at the `async with`,
        # so it also covers acquire_chain_build_lane's reset+clean+CoW seed.
        # Too tight and the cancel lands INSIDE the acquisition under parallel
        # load, testing a different path than the one this test names.
        monkeypatch.setattr('orchestrator.merge_queue.CHAIN_BUILD_TIMEOUT_SECS', 2.0)
        reached: list[str] = []

        async def _stalled_merge(_lane, branch):
            reached.append(branch)
            await asyncio.sleep(60)

        monkeypatch.setattr(git_ops, 'merge_branch_into_worktree', _stalled_merge)

        assert await worker._deep_chain_placement(item) is None
        assert reached, 'the stall must be reached, so the cancel lands in the merge loop'
        assert _lane_states(git_ops) == [LaneState.FREE], 'deadline must not strand a lane'

    async def test_a_none_merge_commit_declines_before_any_build(
        self, git_repo: Path, monkeypatch,
    ):
        """The ``base is None`` guard: decline, never assert, never seed a chain.

        ``RealMergeItem.merge_result.merge_commit`` is Optional, and a chain
        seeded at a ``None`` base would be a chain with no anchor.  Unreachable
        in practice (a RealMergeItem only exists for a SUCCESSFUL merge), which
        is exactly why it needs a test: nothing else would notice if the guard
        were deleted and the OPTIONAL deep path started crashing the dispatch
        hot path on a shape the type system permits.
        """
        import dataclasses

        from orchestrator.warm_lane_pool import LaneState

        git_ops, worker, item, _head, _cfg = await self._fixture(
            git_repo, chain_cap=6,
            queued=[('102', 'b.txt', 'edit-102\n'), ('103', 'c.txt', 'edit-103\n')],
        )
        item.merge_result = dataclasses.replace(item.merge_result, merge_commit=None)
        built = _spy_build_chain(monkeypatch)

        assert await worker._deep_chain_placement(item) is None
        assert built == [], 'declined BEFORE the build, not after'
        assert _lane_states(git_ops) == [LaneState.FREE], 'and without a lane'

    async def test_a_raising_builder_fails_open_and_leaks_no_lane(
        self, git_repo: Path, monkeypatch, caplog,
    ):
        """A bug in the OPTIONAL deep path must never crash the dispatch hot path.

        The generic ``except Exception`` arm's contract, and the sibling of the
        deadline test above: a non-``TimeoutError`` blowing out of
        ``build_chain`` degrades to today's adjacent verify (``None``) rather
        than propagating into ``_dispatch_item``.  Loud, not silent — a
        ``logger.exception`` is required, since a silent fail-soft is what the
        no-silent-fail-soft design invariant forbids.

        Deliberately paired with the CANCEL test directly below: this arm must
        swallow an ordinary ``Exception`` and that one must NOT swallow a
        ``BaseException``.  The two together pin the split.
        """
        import logging

        from orchestrator.warm_lane_pool import LaneState

        git_ops, worker, item, _head, _cfg = await self._fixture(
            git_repo, chain_cap=6,
            queued=[('102', 'b.txt', 'edit-102\n'), ('103', 'c.txt', 'edit-103\n')],
        )

        async def _exploding(*_a, **_kw):
            raise RuntimeError('builder bug')

        monkeypatch.setattr('orchestrator.merge_queue.build_chain', _exploding)

        with caplog.at_level(logging.ERROR, logger='orchestrator.merge_queue'):
            assert await worker._deep_chain_placement(item) is None

        assert _lane_states(git_ops) == [LaneState.FREE], 'no lane stranded'
        blown = [r for r in caplog.records if 'deep chain build failed' in r.message]
        assert len(blown) == 1, 'the fail-open must be LOUD, not silent'
        assert blown[0].exc_info is not None, 'logger.exception, not logger.error'

    async def test_the_queue_snapshot_is_walked_once_on_the_firing_path(
        self, git_repo: Path, monkeypatch,
    ):
        """``chain_snapshot()`` is O(n) — the firing path must pay for it ONCE.

        Guard 2b is deliberately placed BEFORE the snapshot so a suppressed
        task pays no walk at all; taking two walks when the gate DOES fire
        would contradict that same cost discipline.  The bound-once snapshot is
        also what makes "``queue_len`` and ``build_chain``'s argument describe
        the same queue" true structurally rather than by luck.
        """
        git_ops, worker, item, _head, _cfg = await self._fixture(
            git_repo, chain_cap=6,
            queued=[('102', 'b.txt', 'edit-102\n'), ('103', 'c.txt', 'edit-103\n')],
        )
        walks = _spy_chain_snapshot(worker, monkeypatch)
        built = _spy_build_chain(monkeypatch, passthrough=True)

        res = await worker._deep_chain_placement(item)

        assert len(walks) == 1, f'expected exactly one O(n) walk, got {len(walks)}'
        assert built[0]['queue_snapshot'] == tuple(worker.chain_snapshot()), (
            'the reused snapshot is the one build_chain was handed'
        )
        assert res is not None
        await merge_liveness.release_chain_build_lane(
            git_ops, res.lane, warm=res.lane_warm,
        )

    async def test_a_firing_build_stamps_its_wall_clock_cost(
        self, git_repo: Path, monkeypatch,
    ):
        """``build_ms`` is stamped so the per-round DISPATCH STALL is measurable.

        The build is awaited INLINE on ``_verifier_loop``'s dispatch path, so
        for its whole duration no head can be finalized or landed and nothing
        else can dispatch.  ``CHAIN_BUILD_TIMEOUT_SECS`` bounds that stall but
        does not remove it, and the PRD's whole justification is throughput —
        so the cost has to reach telemetry (as ``chain_build_ms``) before ζ
        raises the cap.  Stamped by the CALLER, not by ``build_chain``, so it
        covers lane acquisition and the timeout wrapper too.
        """
        git_ops, worker, item, _head, _cfg = await self._fixture(
            git_repo, chain_cap=6,
            queued=[('102', 'b.txt', 'edit-102\n'), ('103', 'c.txt', 'edit-103\n')],
        )

        res = await worker._deep_chain_placement(item)

        assert res is not None
        assert isinstance(res.build_ms, int), 'stamped, not left None, on a real build'
        assert res.build_ms >= 0
        await merge_liveness.release_chain_build_lane(
            git_ops, res.lane, warm=res.lane_warm,
        )

    async def test_external_cancellation_propagates_and_leaks_no_lane(
        self, git_repo: Path, monkeypatch, caplog,
    ):
        """REVIEW FIX 2: a cancel from OUTSIDE must not be absorbed into ``None``.

        ``asyncio.timeout`` converts its OWN expiry into ``TimeoutError``
        (3.11+), so a ``CancelledError`` that escapes the context manager can
        ONLY be an external cancellation of the verifier-loop task.  Catching
        it alongside ``TimeoutError`` CONSUMES the cancel request: the coroutine
        returns ``None`` and ``_dispatch_item`` runs on as if nothing happened
        — breaking the "CancelledError is never absorbed" invariant
        ``_verifier_loop`` states.

        The REAL ``build_chain`` runs (only the per-item merge is stalled, on
        an Event so it can never return on its own), which is what makes the
        lane assertions real: its own ``except BaseException`` arm is what
        returns the lane on the cancellation path.
        """
        import logging

        from orchestrator.warm_lane_pool import LaneState

        git_ops, worker, item, _head, _cfg = await self._fixture(
            git_repo, chain_cap=6,
            queued=[('102', 'b.txt', 'edit-102\n'), ('103', 'c.txt', 'edit-103\n')],
        )
        entered = asyncio.Event()
        never = asyncio.Event()

        async def _stalled_merge(_lane, _branch):
            entered.set()
            await never.wait()

        monkeypatch.setattr(git_ops, 'merge_branch_into_worktree', _stalled_merge)
        caplog.set_level(logging.WARNING, logger='orchestrator.merge_queue')

        task = asyncio.ensure_future(worker._deep_chain_placement(item))
        # Wait on the stub's own signal, never a sleep: the cancel has to land
        # INSIDE the build, and a timed guess races the lane acquisition.
        await asyncio.wait_for(entered.wait(), timeout=60)
        task.cancel()
        await asyncio.wait([task])

        assert task.cancelled() is True, (
            'the cancel must be HONOURED, not absorbed — absorbing it shows up '
            'as an ordinary None return, which is why "no exception" is not '
            'the assertion here'
        )
        assert not [
            r for r in caplog.records
            if 'exceeded' in r.message or 'falling back to the adjacent' in r.message
        ], (
            'a cancel is not a deadline overrun — logging the timeout WARNING '
            'sends an operator hunting a wedged lane that does not exist'
        )
        assert _lane_states(git_ops) == [LaneState.FREE], 'no stranded lane'
        assert worker.worktree_ledger_violations(grace_secs=0.0) == []
        assert worker.speculation_accounting_violations() == []

    async def test_cancelling_the_enclosing_dispatch_stops_the_dispatch(
        self, tmp_path: Path, monkeypatch,
    ):
        """REVIEW FIX 2, where the real hazard lives: the ENCLOSING task.

        ``_deep_chain_placement`` is awaited inline by ``_dispatch_item``, which
        the verifier loop runs as a task.  If the cancel is swallowed down in
        the placement, ``_dispatch_item`` sails on past it and
        ``ensure_future(self._run_inflight_verify(...))`` launches a whole
        verify for a dispatch that was cancelled — so the observable is not
        just "the exception vanished" but "work happened after the cancel".
        """
        repo = await _fresh_repo(tmp_path, 'cancel-dispatch')
        scene = await _make_deep_scene(
            repo, chain_cap=6, script=[True], monkeypatch=monkeypatch,
        )
        entered = asyncio.Event()
        never = asyncio.Event()

        async def _blocking_build(*_a, **_kw):
            entered.set()
            await never.wait()
            raise AssertionError('unreachable: the build must not return on its own')

        monkeypatch.setattr('orchestrator.merge_queue.build_chain', _blocking_build)
        item = _make_item(
            _make_req('101', '101', scene.config, repo), scene.head,
            _ephemeral_merge_wt(scene.git_ops, 'cancel-dispatch'),
        )

        task = asyncio.ensure_future(scene.worker._dispatch_item(item))
        await asyncio.wait_for(entered.wait(), timeout=60)
        task.cancel()
        await asyncio.wait([task])

        assert task.cancelled() is True, 'the dispatch must not survive its cancel'
        assert scene.calls == [], (
            'a cancelled dispatch must never go on to launch a verify'
        )

    async def test_zero_link_result_declines_without_double_releasing(
        self, git_repo: Path, monkeypatch,
    ):
        """An empty ChainResult -> None, and the caller must NOT release.

        A zero-link result never holds a lane (``lane is None``,
        ``lane_warm is False`` — ChainResult "Lane ownership"), because
        ``build_chain`` already released it on that path.  A caller that
        released again would return a FREE lane to the pool a second time.

        It also must not consume a halving round: nothing was verified deeply,
        so there is no verdict to halve or reset on.
        """
        released: list = []
        monkeypatch.setattr(
            'orchestrator.merge_queue.release_chain_build_lane',
            lambda git_ops, lane, *, warm: released.append((lane, warm)),
        )
        git_ops, worker, item, _head, _cfg = await self._fixture(
            git_repo, chain_cap=6,
            queued=[('102', 'b.txt', 'edit-102\n'), ('103', 'c.txt', 'edit-103\n')],
        )
        # A first-item conflict is the realistic way to get a zero-link result.
        assert item.merge_result.merge_commit is not None
        built = _spy_build_chain(monkeypatch, ChainResult(
            links=[], tip=item.merge_result.merge_commit,
            truncated_at='102', truncated_reason='conflict',
        ))
        worker._chain_halving_state = 4

        assert await worker._deep_chain_placement(item) is None
        assert len(built) == 1, 'the gate must have BUILT — otherwise this passes vacuously'
        assert released == [], 'an empty result holds no lane — releasing would double-free'
        assert worker._chain_halving_state == 4, 'no chain verified, so no halving round'

    async def test_truncated_short_of_target_still_returns_the_chain(
        self, git_repo: Path, monkeypatch,
    ):
        """A short chain is still useful, and β's decision-4 purity survives its caller.

        ``104`` conflicts with ``102`` on the same line, so a 3-target build
        truncates after 2 links.  The gate must return that ChainResult rather
        than discard the work — and the truncated item must have NO outcome
        rendered for it: it may conflict only with an *unlanded* predecessor,
        and takes its normal sequential path later.
        """
        git_ops, worker, item, _head, _cfg = await self._fixture(
            git_repo, chain_cap=6,
            queued=[
                ('102', 'shared.txt', _shared_txt_with(1, 'from-102')),
                ('103', 'c.txt', 'edit-103\n'),
                ('104', 'shared.txt', _shared_txt_with(1, 'from-104')),
            ],
        )
        built = _spy_build_chain(monkeypatch, passthrough=True)
        conflicts: list[str] = []
        monkeypatch.setattr(worker, '_note_conflict_detected', conflicts.append)
        store = _CapturingEventStore()
        worker._event_store = store
        queued_reqs = list(worker._lane_buffers['normal'])

        res = await worker._deep_chain_placement(item)

        assert built[0]['target_depth'] == 3, 'queue_len 4 -> d 4 -> 3 extra links'
        assert res is not None, 'a short chain is useful — do not discard it'
        assert [tid for tid, _ in res.links] == ['102', '103']
        assert res.truncated_at == '104'
        assert res.truncated_reason == 'conflict'

        # PRD decision 4: NO outcome for the truncated item, end to end.
        assert conflicts == []
        assert store.events_of(EventType.merge_attempt) == []
        assert all(not r.result.done() for r in queued_reqs)
        assert list(worker._lane_buffers['normal']) == queued_reqs, 'queue untouched'

        await merge_liveness.release_chain_build_lane(
            git_ops, res.lane, warm=res.lane_warm,
        )


# ═══════════════════════════════════════════════════════════════════════════
# step-15: RED — the dispatch REDIRECT itself.
#
# THE genuinely novel capability of γ. Task 2359's variable-depth probe only
# ever RELABELLED a dispatch's depth/base attribution — its own docstring
# calls that the "KNOWN PHASE-1 LIMITATION", because the verify still
# exercised the dispatching item's own (shallow) tree. These tests pin the
# thing that closes it: `_run_inflight_verify(item, lease, chain=<ChainResult>)`
# verifies the CHAIN TIP, inside the CHAIN LANE, and neither the item's own
# merge commit nor its own merge worktree is exercised at all.
# ═══════════════════════════════════════════════════════════════════════════


def _local_lease():
    """A LOCAL :class:`HostLease` whose runner is never actually driven.

    LOCAL deliberately, not remote: `lease.is_local` is precisely what selects
    the warm-swap block these tests assert the chain arm SKIPS. A remote lease
    would skip it for an unrelated reason and make the assertion vacuous.
    """
    from unittest.mock import MagicMock

    from orchestrator.verify_runner import HostLease

    runner = MagicMock()
    runner.name = 'local'
    runner.is_local = True
    return HostLease(name='local', runner=runner, is_local=True)


# NOTE ON HOW THE WARM-SWAP BYPASS IS OBSERVED (no reach-back patch).
#
# The obvious spy — ``monkeypatch.setattr('orchestrator.merge_queue.
# _acquire_warm_verify_worktree', ...)`` — is a merge_queue reach-back patch,
# frozen by test_merge_queue_reachback_patch_guard.py.  Repointing it at the
# DEFINING module (``orchestrator.merge_liveness``) would not work either:
# ``_run_inflight_verify`` resolves the name through merge_queue's own module
# global, so a merge_liveness patch sits off the resolution path and any
# "was not called" assertion would pass VACUOUSLY — exactly the failure mode
# that guard exists to prevent.
#
# So the bypass is observed through the warm-swap block's REAL, externally
# visible effects instead, which is a stronger test anyway:
#   * ``git_ops.acquire_spec_lane`` — what _acquire_warm_verify_worktree calls
#     for a speculative item with the lane pool on (merge_liveness.py:715-731).
#     Spied on the GitOps INSTANCE via ``_count_spec_lane_acquires``, which is
#     not a module reach-back at all.
#   * ``worker._verify_attempt_count`` — staged and committed INSIDE that
#     block, and the driver of the periodic cold-verify safety valve.


def _spy_post_merge_verify(monkeypatch, outcome=None, *, raises=None) -> list[dict]:
    """Replace ``_run_post_merge_verify`` with a recorder returning *outcome*.

    Returns the recorded call list — each entry carries the positional
    ``merge_wt`` plus the kwargs (``merge_sha``, ``chain_items``, ``depth``,
    ``speculative``).  ``outcome=None`` is a PASS in this function's
    vocabulary; a :class:`VerifyResult` is a FAIL.  *raises* makes the verify
    blow up instead, which is the third exit the lane release must survive.
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



def _fake_pass_runner(name: str = 'fake-runner'):
    """A RemoteRunner-shaped fake whose ``run_merge_verify`` always passes.

    Copied from test_merge_queue_depth_telemetry.py:153 rather than imported:
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


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestRunInflightVerifyChainRedirect:
    """`_run_inflight_verify(..., chain=...)` verifies the TIP, in the LANE."""

    async def _fixture(
        self, git_repo: Path, *, chain_cap: int = 6, frontier: int = 0,
    ):
        """Build worker + slot-2 item + a REAL 2-link chain, ready to dispatch.

        Returns ``(git_ops, worker, item, chain, head)``.  *frontier* seeds
        ``_verify_frontier_depth()`` so the ``chain_items`` arithmetic is
        exercised with a non-zero frontier rather than only at the origin.
        """
        git_ops = _make_git_ops(git_repo, size=2)
        config = _make_config(git_repo, chain_cap=chain_cap)
        await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n')
        for tid, fn in (('102', 'b.txt'), ('103', 'c.txt')):
            await _create_branch_editing(git_repo, f'task/{tid}', fn, f'edit-{tid}\n')
        head = await _merge_commit_off_main(git_repo, 'task/101', '101')
        worker = _make_worker(git_ops)
        worker._lane_buffers['normal'].extend(
            _make_req(tid, tid, config, git_repo) for tid in ('102', '103')
        )
        worker._frozen_inflight_entries = lambda: [object()] * frontier  # type: ignore[assignment,method-assign]
        item = _make_item(
            _make_req('101', '101', config, git_repo), head,
            _ephemeral_merge_wt(git_ops, 'redirect'),
        )
        chain = await worker._deep_chain_placement(item)
        assert chain is not None and len(chain.links) == 2, 'fixture must build a chain'
        return git_ops, worker, item, chain, head

    async def test_verifies_the_chain_tip_inside_the_chain_lane(
        self, git_repo: Path, monkeypatch,
    ):
        """The redirect: ``merge_sha`` is ``chain.tip``, worktree is ``chain.lane``.

        And — the half that makes it a REDIRECT rather than a relabel —
        NEITHER the item's own merge commit NOR its own merge worktree is
        handed to the verify.
        """
        git_ops, worker, item, chain, head = await self._fixture(git_repo)
        posted = _spy_post_merge_verify(monkeypatch)

        await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert len(posted) == 1
        assert posted[0]['merge_sha'] == chain.tip
        assert posted[0]['merge_wt'] == chain.lane
        assert posted[0]['merge_sha'] != head == item.merge_result.merge_commit
        assert posted[0]['merge_wt'] != item.merge_wt

    async def test_warm_swap_is_bypassed_on_the_chain_arm(
        self, git_repo: Path, monkeypatch,
    ):
        """The lane already holds the tip — a second acquisition would be wrong.

        It would take a DIFFERENT lane, re-check-out a DIFFERENT commit, and
        could resolve to the serial ``_merge-verify`` lane that
        ``acquire_chain_build_lane`` explicitly refuses (DF-3071).

        Observed through the block's real effects rather than a spy on the
        function itself — see the note above ``_local_lease``.  This is a
        LOCAL lease deliberately: ``lease.is_local`` is what selects the block,
        so a remote lease would skip it for an unrelated reason and make the
        assertion vacuous.
        """
        git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        # Installed AFTER the build, so the chain's own single acquisition is
        # excluded and this list counts only what the VERIFY would take.
        lanes = _count_spec_lane_acquires(git_ops, monkeypatch)
        _spy_post_merge_verify(monkeypatch)
        before = worker._verify_attempt_count

        await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert lanes == [], 'chain arm must not acquire a second verify lane'
        assert worker._verify_attempt_count == before, (
            'the attempt counter is staged INSIDE the warm-swap block, so a '
            'bypassed block must leave the cold-verify safety valve untouched'
        )

    async def test_chain_items_counts_the_actually_built_depth(
        self, git_repo: Path, monkeypatch,
    ):
        """``chain_items == 1 + len(chain.links)`` — what actually ran.

        A 6-target that only built 2 links must emit 3, never 7: the field is a
        fact about the tree that ran, not about the target that was asked for.

        The arm COMPUTES the value rather than ACCUMULATING onto the caller's
        — which is why a deliberately wrong ``chain_items=99`` is passed here
        and 3 is still required.  That is what makes the emitted unit
        frontier-independent by construction, so it cannot silently regress if
        ``_dispatch_item``'s own derivation ever changes again.
        """
        _git_ops, worker, item, chain, _head = await self._fixture(git_repo, frontier=1)
        posted = _spy_post_merge_verify(monkeypatch)

        await worker._run_inflight_verify(
            item, _local_lease(), chain=chain, chain_items=99,  # deliberately wrong
        )

        assert posted[0]['chain_items'] == 3 == 1 + len(chain.links)

    @pytest.mark.parametrize(
        'verdict', ['pass', 'fail', 'raise'], ids=['tip-pass', 'tip-fail', 'verify-raises'],
    )
    async def test_lane_is_released_exactly_once_on_every_exit(
        self, git_repo: Path, monkeypatch, verdict: str,
    ):
        """Tip pass, tip fail and an exception all return the lane — once.

        PRD ι conservation: the pool lane is FREE afterwards and BOTH resource
        audits are clean.  A double release would hand a FREE lane back to the
        pool twice; a missed one strands a `_spec-` slot for the process's life.
        """
        from orchestrator.warm_lane_pool import LaneState

        git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        _spy_post_merge_verify(
            monkeypatch,
            outcome=_fail_verify_result() if verdict == 'fail' else None,
            raises=RuntimeError('verify exploded') if verdict == 'raise' else None,
        )
        released = _spy_chain_lane_release(monkeypatch)
        assert _lane_states(git_ops) == [LaneState.ASSIGNED, LaneState.FREE]

        await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert released == [(chain.lane, chain.lane_warm)], 'exactly one release'
        assert _lane_states(git_ops) == [LaneState.FREE, LaneState.FREE]
        assert worker.worktree_ledger_violations(grace_secs=0.0) == []
        assert worker.speculation_accounting_violations() == []

    @pytest.mark.parametrize(
        'passed', [True, False], ids=['tip-pass', 'tip-fail'],
    )
    async def test_note_chain_outcome_fires_once_with_the_built_depth(
        self, git_repo: Path, monkeypatch, passed: bool,
    ):
        """``_note_chain_outcome(passed, 1 + len(chain.links))``, exactly once.

        The depth fed to the halving policy is the BUILT depth in chain-item
        units (the dispatching item is #1), so a fail halves off what actually
        ran rather than off the target that was requested.
        """
        _git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        _spy_post_merge_verify(
            monkeypatch, outcome=None if passed else _fail_verify_result(),
        )
        noted: list[tuple] = []
        monkeypatch.setattr(
            worker, '_note_chain_outcome',
            lambda p, d: noted.append((p, d)),
        )

        await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert noted == [(passed, 3)], '1 + 2 built links, in chain-item units'

    async def test_note_chain_outcome_is_silent_when_the_verify_raises(
        self, git_repo: Path, monkeypatch,
    ):
        """An infra error is NOT a tip verdict, and NOT the item's fault either.

        Halving off a RuntimeError would walk the policy down to the d=1 floor
        on repeated infra noise and silently disable deep merge-ahead — the
        two original assertions here.

        REVIEW FIX 3 adds the other half.  The verify ran against ``chain.tip``
        — a CUMULATIVE tree containing this item PLUS every chained successor's
        content — so an error raised out of it is no more attributable to this
        item than a tip PASS is.  γ's soundness fence at the non-adopting exit
        only guarded the pass/fail return path, so the argument held in the
        green direction and failed in the red one: a chain-caused
        ``Verification error:`` terminally blocked a request whose own subset
        tree was never even run.
        """
        from orchestrator.merge_types import InflightStatus
        from orchestrator.warm_lane_pool import LaneState

        _git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        store = _CapturingEventStore()
        worker._event_store = store
        _spy_post_merge_verify(monkeypatch, raises=RuntimeError('verify exploded'))
        released = _spy_chain_lane_release(monkeypatch)
        noted: list[tuple] = []
        monkeypatch.setattr(
            worker, '_note_chain_outcome', lambda p, d: noted.append((p, d)),
        )

        res = await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        # PRESERVED: an infra error is not a tip verdict.
        assert noted == []
        assert worker._chain_halving_state is None
        # NEW: it is not a resolution either — it is a defer.
        assert not item.request.result.done(), (
            'a chain-arm error must not resolve the dispatching item: the tree '
            'that blew up was not its own'
        )
        assert res.status is InflightStatus.REQUEUED
        assert res.outcome is None, 'a defer renders no MergeOutcome at all'
        assert worker._queue.qsize() == 1
        assert worker._queue.get_nowait() is item.request
        # ...and therefore never feeds workflow.py's consecutive_merge_thrash
        # ladder, the same boundary row the pass/fail arms already honour.
        assert store.events_of(EventType.merge_attempt) == []
        # The lane release lives in the `finally`, so this already holds —
        # asserted so the new early return cannot regress it.
        assert released == [(chain.lane, chain.lane_warm)], 'exactly one release'
        assert _lane_states(_git_ops) == [LaneState.FREE, LaneState.FREE]
        assert worker.worktree_ledger_violations(grace_secs=0.0) == []
        assert worker.speculation_accounting_violations() == []

    async def test_a_chain_arm_error_suppresses_the_next_chain_for_that_task(
        self, git_repo: Path, monkeypatch,
    ):
        """THE LOOP GUARD: a requeue that rebuilt the same chain would spin.

        The requeue above sends the item straight back for re-dispatch.  If the
        next dispatch built the SAME chain and hit the SAME error, nothing would
        ever converge — so after a chain-arm error the next placement for that
        task declines, and the re-dispatch verifies the item's OWN subset tree,
        which is the tree whose verdict is actually attributable to it.

        Declines at ZERO cost: no ``build_chain`` call at all.
        """
        _git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        _spy_post_merge_verify(monkeypatch, raises=RuntimeError('verify exploded'))

        await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        built = _spy_build_chain(monkeypatch, passthrough=True)
        assert await worker._deep_chain_placement(item) is None
        assert built == [], 'a suppressed task must not pay for a build at all'

    async def test_chain_suppression_is_one_shot_and_self_clearing(
        self, git_repo: Path, monkeypatch,
    ):
        """One bad round must not disable deep merge-ahead for the task forever.

        The suppression is discarded at the existing unconditional "the verify
        returned a result at all" reset site, so a single infra blip costs
        exactly one non-chained round.
        """
        _git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        _spy_post_merge_verify(monkeypatch, raises=RuntimeError('verify exploded'))
        await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        # The re-dispatch: no chain (suppressed), and the verify returns
        # normally this time — which is what clears the suppression.
        posted = _spy_post_merge_verify(monkeypatch, outcome=None)
        res = await worker._run_inflight_verify(item, _local_lease())
        assert len(posted) == 1, 'the ordinary arm ran'
        if res.merge_wt is not None:
            await worker._release_or_cleanup(res.merge_wt, spec_warm=res.spec_warm)

        built = _spy_build_chain(monkeypatch, passthrough=True)
        again = await worker._deep_chain_placement(item)
        assert len(built) == 1, 'chaining is re-enabled after a clean round'
        assert again is not None
        await merge_liveness.release_chain_build_lane(
            _git_ops, again.lane, warm=again.lane_warm,
        )

    async def test_the_ping_pong_terminates_after_exactly_one_deferred_round(
        self, git_repo: Path, monkeypatch,
    ):
        """An unbounded defer is its own wedge — drive the whole walk for real.

        The earlier shape of this test SEEDED the accumulated state
        (``_chain_verify_errors = {task: 1}`` plus a monkeypatched
        ``MAX_CHAIN_VERIFY_ERRORS``) and asserted only that the cap branch ran
        given a pre-set counter.  That proved nothing about whether the system
        could ever REACH the cap — and it could not: suppression forces the
        very next dispatch to ``chain is None``, so the count could never
        exceed 1 and the cap branch was unreachable dead code.  The counter and
        its cap are gone; the honest contract is one-shot suppression whose
        backstop is the pre-existing blocked resolution.

        So drive it organically instead — two consecutive raising verifies,
        no seeding, no monkeypatched constants — and assert the whole walk:
        round 1 defers and suppresses, round 2 is un-chained and TERMINATES.
        """
        from orchestrator.merge_types import InflightStatus

        _git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        task_id = item.request.task_id
        assert worker._chain_error_suppressed == set(), 'no stale suppression'

        # ── ROUND 1: chained, raises → defer + suppress (no resolution). ──
        _spy_post_merge_verify(monkeypatch, raises=RuntimeError('boom 1'))
        first = await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert first.status is InflightStatus.REQUEUED
        assert not item.request.result.done()
        assert worker._chain_error_suppressed == {task_id}
        assert worker._queue.get_nowait() is item.request

        # ── The re-dispatch DECLINES to chain, at zero build cost. ──
        built = _spy_build_chain(monkeypatch, passthrough=True)
        assert await worker._deep_chain_placement(item) is None
        assert built == [], 'a suppressed task pays for no build'

        # ── ROUND 2: un-chained, raises too → today's terminal resolution. ──
        _spy_post_merge_verify(monkeypatch, raises=RuntimeError('boom 2'))
        second = await worker._run_inflight_verify(item, _local_lease())

        assert item.request.result.done(), (
            'the un-chained round is the backstop: its verify ran against the '
            "item's OWN tree, so its error IS attributable and must resolve"
        )
        outcome = item.request.result.result()
        assert outcome.status == 'blocked'
        assert outcome.reason.startswith('Verification error:')
        assert second.status is not InflightStatus.REQUEUED
        assert second.outcome is outcome
        assert worker._queue.qsize() == 0, 'the terminal round does not requeue'
        # THE LEAK FENCE: the terminal exit clears the suppression regardless
        # of which arm produced the error.  Without this, the entry would
        # linger for the life of the worker (~8h between fleet redeploys) and
        # silently disable deep merge-ahead for this task_id forever.
        assert worker._chain_error_suppressed == set(), (
            'every terminal exit must clear the suppression set'
        )

    @pytest.mark.parametrize(
        'slot1,cap,expected',
        [
            (False, 6, None),   # positive control: slot 2, cap on -> resets
            (True, 6, 3),       # slot 1 is a DIFFERENT tree -> no reset
            (False, 0, 3),      # kill switch: the arm is not even reached
        ],
        ids=['slot2-cap-on-resets', 'slot1-does-not-reset', 'cap-zero-does-not-reset'],
    )
    async def test_the_floor_reset_arms_two_guards_are_each_load_bearing(
        self, git_repo: Path, monkeypatch, slot1: bool, cap: int, expected,
    ):
        """The d=1 floor's reset arm: ``item.speculative`` and ``chain_cap > 0``.

        Both guards previously had no test in which they were the DECIDING
        branch — every scenario that could have exercised them started from
        ``_chain_halving_state is None``, so either guard could have been
        deleted with the suite still green.  Here the state starts at 3 and
        each row differs in exactly one guard, so a deletion flips a row.

        Why each guard matters.  A slot-1 head verify is merged onto REAL main
        and is never chained by γ, so its green says nothing about whether
        chaining is safe — letting it reset would climb the bisector back out
        on evidence about a different tree.  And under the shipped
        ``chain_cap=0`` kill switch this arm must not merely have no effect: it
        must not be REACHED, so a cap-0 dispatch stays byte-identical down to
        "no method call".

        Runs with ``chain=None`` deliberately: the reset arm is on the ORDINARY
        path (at the floor the gate declines, so there is no chain by
        construction) — which is exactly why it is the only way the walk ever
        climbs back out.
        """
        git_ops, worker, item, chain, head = await self._fixture(
            git_repo, chain_cap=6,
        )
        await merge_liveness.release_chain_build_lane(
            git_ops, chain.lane, warm=chain.lane_warm,
        )
        if slot1 or cap == 0:
            item = _make_item(
                _make_req('101', '101', _make_config(git_repo, chain_cap=cap), git_repo),
                head, _ephemeral_merge_wt(git_ops, 'floorreset'),
                speculative=not slot1,
            )
        worker._chain_halving_state = 3
        posted = _spy_post_merge_verify(monkeypatch, outcome=None)   # a PASS

        res = await worker._run_inflight_verify(item, _local_lease())

        assert len(posted) == 1 and posted[0]['chain_items'] == 1
        assert res.outcome is None, 'the fixture must actually be a PASS'
        assert worker._chain_halving_state == expected
        if res.merge_wt is not None:
            await worker._release_or_cleanup(res.merge_wt, spec_warm=res.spec_warm)

    async def test_lane_is_released_exactly_once_on_a_dispose_exit(
        self, git_repo: Path, monkeypatch,
    ):
        """The DROPPED abort disposes FIRST — the ``finally``'s latch absorbs the rest.

        The parametrized release test above covers only the three exits that
        run the verify to completion (pass / fail / raise).  This is the OTHER
        shape, and the one the latch actually exists for: the abort branches
        call ``_dispose_verify_worktree()``, which on the chain arm ALREADY
        routes the lane back through ``release_chain_build_lane`` — so the
        ``finally``'s unconditional release is a SECOND call on an
        already-FREE lane.  Exactly one must reach the pool; a second real
        release would hand a free lane back twice.
        """
        from orchestrator.merge_types import InflightStatus
        from orchestrator.warm_lane_pool import LaneState

        git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        released = _spy_chain_lane_release(monkeypatch)
        monkeypatch.setattr(type(worker), 'VERIFY_ABANDON_POLL_SECS', 0.01)

        async def _slow_verify(_git_ops, _req, _merge_wt, **_kw):
            await asyncio.sleep(30)

        monkeypatch.setattr(
            'orchestrator.merge_queue._run_post_merge_verify', _slow_verify,
        )
        # The sole waiter gives up: this is what _request_abandoned reads.
        item.request.result.cancel()

        res = await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert res.status is InflightStatus.DROPPED
        assert len(released) == 1, (
            'the dispose released it; the finally must be a latched no-op'
        )
        assert released[0] == (chain.lane, chain.lane_warm)
        assert _lane_states(git_ops) == [LaneState.FREE, LaneState.FREE]
        assert worker.worktree_ledger_violations(grace_secs=0.0) == []
        assert worker.speculation_accounting_violations() == []
        assert worker._chain_error_suppressed == set(), (
            'a DROPPED request must not leave a suppression entry behind'
        )

    async def test_the_suppression_set_starts_empty_on_a_fresh_worker(
        self, git_repo: Path,
    ):
        """The state exists and is empty at construction (no stale suppression)."""
        git_ops = _make_git_ops(git_repo, size=2)
        worker = _make_worker(git_ops)

        assert worker._chain_error_suppressed == set()

    async def test_chain_none_with_a_raising_verify_is_byte_identical_to_today(
        self, git_repo: Path, monkeypatch,
    ):
        """NEGATIVE CONTROL — the always-on fence.

        The head path and the shipped ``chain_cap=0`` default must not observe
        REVIEW FIX 3 at all: a raising verify with no chain still resolves the
        request terminally as ``blocked``, exactly as before γ.  Pairs with
        ``test_chain_none_is_byte_identical_to_todays_path`` on the pass path.
        """
        from orchestrator.merge_types import InflightStatus

        git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        await merge_liveness.release_chain_build_lane(
            git_ops, chain.lane, warm=chain.lane_warm,
        )
        _spy_post_merge_verify(monkeypatch, raises=RuntimeError('verify exploded'))

        res = await worker._run_inflight_verify(item, _local_lease())

        assert item.request.result.done(), 'no chain: the ordinary terminal path'
        outcome = item.request.result.result()
        assert outcome.status == 'blocked'
        assert outcome.reason.startswith('Verification error:')
        assert res.outcome is outcome
        assert res.status is not InflightStatus.REQUEUED
        assert worker._queue.qsize() == 0, 'no requeue on the ordinary arm'
        assert worker._chain_error_suppressed == set(), (
            'the ordinary arm must not leave a suppression entry behind — and '
            'this terminal exit clears one even if some earlier round set it'
        )

    async def test_item_ephemeral_worktree_is_disposed_not_stranded(
        self, git_repo: Path, monkeypatch,
    ):
        """The item's own merge worktree is unused on this arm — clean it.

        The verify runs in the chain lane, so the ephemeral ``_merge-<uuid>``
        the item was merged in is dead weight.  Leaving it registered pins its
        mtime forever via the heartbeat; leaving it on disk UNregistered trips
        the I6 worktree-ledger audit once it ages past the grace window.
        """
        git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        ephemeral = _ephemeral_merge_wt(git_ops, 'deadbeef')
        item.merge_wt = ephemeral
        worker._register_owned_merge_worktree(ephemeral)
        _spy_post_merge_verify(monkeypatch)
        cleaned: list[Path] = []
        monkeypatch.setattr(
            git_ops, 'cleanup_merge_worktree',
            lambda wt: cleaned.append(wt) or asyncio.sleep(0),
        )

        await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert ephemeral not in worker._owned_merge_worktrees
        assert cleaned == [ephemeral]

    async def test_chain_none_is_byte_identical_to_todays_path(
        self, git_repo: Path, monkeypatch,
    ):
        """Every existing caller (``chain=None``) is untouched.

        The REAL warm-swap runs (no stub), the item's OWN merge commit is what
        gets verified, ``chain_items`` is the default 1, nothing is released as
        a chain lane, and ``merge_wt`` is still handed back to
        ``_finalize_inflight`` for disposal at its chokepoint.
        """
        git_ops, worker, item, chain, head = await self._fixture(git_repo)
        await merge_liveness.release_chain_build_lane(
            git_ops, chain.lane, warm=chain.lane_warm,
        )
        lanes = _count_spec_lane_acquires(git_ops, monkeypatch)
        posted = _spy_post_merge_verify(monkeypatch)
        released = _spy_chain_lane_release(monkeypatch)
        before = worker._verify_attempt_count

        res = await worker._run_inflight_verify(item, _local_lease())

        assert lanes == [item.merge_result.merge_commit], (
            'the ordinary path still routes a speculative LOCAL verify through '
            'the warm-swap block into a _spec- lane'
        )
        assert worker._verify_attempt_count == before + 1
        assert posted[0]['merge_sha'] == head == item.merge_result.merge_commit
        assert posted[0]['chain_items'] == 1, 'default: a one-item tree'
        assert released == [], 'no chain, nothing to release'
        assert res.merge_wt is not None and res.spec_warm is True, (
            'the ordinary pass path still hands the verified worktree (and its '
            'warmth) to _finalize_inflight rather than disposing of it here'
        )
        await worker._release_or_cleanup(res.merge_wt, spec_warm=res.spec_warm)


# ═══════════════════════════════════════════════════════════════════════════
# step-17: RED — the NON-ADOPTION invariant, NARROWED by δ (task 3186).
#
# THE rule this whole task is fenced by: a tip verdict proves the CUMULATIVE
# tree, never the dispatching item's own SUBSET tree.
#
# The rule is SYMMETRIC, and γ enforced BOTH halves by requeuing on both arms.
# δ licenses exactly ONE of them: the tip's tree is a verified SUPERSET of
# every prefix member, so a GREEN tip does license landing the whole prefix —
# and δ's in-order CAS walk over `chain.links` is where that happens
# (test_merge_queue_deep_landing.py).  Nothing else moved.  A tip FAIL still
# proves only that the cumulative tree is red, never WHICH member broke it, and
# a chain-arm EXCEPTION proves nothing about anyone at all — so those two arms
# keep γ's recipe verbatim and are what this class now pins
# (merge_queue.py:18623 states the asymmetry in as many words).
# ═══════════════════════════════════════════════════════════════════════════


_NON_ADOPTING_ARMS = [
    pytest.param(False, None, id='tip-fail'),
    pytest.param(True, RuntimeError('verify infra exploded'), id='chain-arm-error'),
]
"""The two arms that keep γ's REQUEUED recipe verbatim under δ (task 3186).

``(passed, raises)``.  The error arm passes ``passed=True`` deliberately — the
verdict it would have produced is irrelevant, because the exception preempts it;
what the arm proves is that an infra error is not a tip verdict at all.
"""


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestDeepTipVerifyNeverAdopts:
    """The two arms that STAY non-adopting under δ: tip fail, and tip error.

    Narrowed by δ (task 3186), which took over the PASS arm — see the banner
    above.  ``ARM`` below is the parametrize vocabulary these tests share.
    """

    async def _fixture(
        self, git_repo: Path, monkeypatch, *, passed: bool, raises=None,
    ):
        """Drive one deep tip verify to *passed* (or *raises*) and return the scene.

        Returns ``(git_ops, worker, item, chain, res, store, queued_reqs)``.
        ``advance_main`` is spied (not stubbed to fail) so a violation shows up
        as a recorded call rather than as an unrelated exception.  Note this is
        still meaningful on the PASS arm under δ: δ's walk lives in the
        FINALIZE half, so ``_run_inflight_verify`` alone must still reach no
        CAS advance on any arm.
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
            _ephemeral_merge_wt(git_ops, 'nonadopt'),
        )
        chain = await worker._deep_chain_placement(item)
        assert chain is not None and len(chain.links) == 2

        advances: list = []
        monkeypatch.setattr(
            git_ops, 'advance_main',
            lambda *a, **k: advances.append((a, k)) or asyncio.sleep(0),
        )
        _spy_post_merge_verify(
            monkeypatch, outcome=None if passed else _fail_verify_result(),
            raises=raises,
        )
        queued_reqs = list(worker._lane_buffers['normal'])
        main_before = await _rev_parse(git_repo, 'main')

        res = await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert advances == [], 'a tip verdict must never reach advance_main'
        assert await _rev_parse(git_repo, 'main') == main_before
        return git_ops, worker, item, chain, res, store, queued_reqs

    @pytest.mark.parametrize(('passed', 'raises'), _NON_ADOPTING_ARMS)
    async def test_requeues_instead_of_resolving_on_the_non_adopting_arms(
        self, git_repo: Path, monkeypatch, passed: bool, raises,
    ):
        """REQUEUED sentinel + the request genuinely back on ``_queue``.

        A requeue is a DEFER, not a resolution: the Future stays PENDING and
        the waiter receives the outcome of the RE-dispatch.  The three effects
        must all fire, which is why this goes through ``_requeue_request`` (the
        chokepoint enforced by
        test_merge_queue_lifecycle_registry.py::TestRequeueRecipeHasASingleChokepoint)
        rather than a bare ``_queue.put_nowait``.

        δ (task 3186) took the PASS arm — see
        test_merge_queue_deep_landing.py::TestTipPassAdoptionSignal for the
        adopting exit's own contract.
        """
        from orchestrator.merge_types import InflightStatus

        _git_ops, worker, item, _chain, res, _store, _q = await self._fixture(
            git_repo, monkeypatch, passed=passed, raises=raises,
        )

        assert res.status == InflightStatus.REQUEUED
        assert res.outcome is None, 'a defer renders no MergeOutcome at all'
        assert res.merge_wt is None, 'the chain arm disposes its own worktrees'
        assert not item.request.result.done(), 'a requeue is a defer, not a resolution'
        assert worker._queue.qsize() == 1
        assert worker._queue.get_nowait() is item.request

    @pytest.mark.parametrize(('passed', 'raises'), _NON_ADOPTING_ARMS)
    async def test_emits_no_merge_attempt_and_no_blocked_outcome(
        self, git_repo: Path, monkeypatch, passed: bool, raises,
    ):
        """Deep fails never feed workflow.py's ``consecutive_merge_thrash`` ladder.

        That ladder means "the SAME mechanical merge failure repeated", and it
        is fed by `merge_attempt` events and blocked `MergeOutcome`s.  A deep
        tip fail is neither — it is a defer — so two consecutive ones must
        leave the ladder's inputs completely untouched.  This is the same
        false-positive-escalation class ``build_chain``'s own decision-4
        purity comment guards against upstream.
        """
        _git_ops, _worker, item, _chain, res, store, _q = await self._fixture(
            git_repo, monkeypatch, passed=passed, raises=raises,
        )

        assert store.events_of(EventType.merge_attempt) == []
        assert res.outcome is None
        assert not item.request.result.done()

    async def test_tip_fail_halves_the_state_and_leaves_the_queue_intact(
        self, git_repo: Path, monkeypatch,
    ):
        """PRD boundary row "Tip fail leaves queue intact": zero landings.

        The chained items were merged only inside the scratch lane, so after a
        red tip they are still buffered, in the same order, with no outcome and
        no conflict rendered for any of them — build_chain's decision-4 purity
        surviving all the way through its first real caller.
        """
        _git_ops, worker, _item, _chain, _res, store, queued = await self._fixture(
            git_repo, monkeypatch, passed=False,
        )

        assert worker._chain_halving_state == 1, '3 built items -> max(1, 3 // 2)'
        assert list(worker._lane_buffers['normal']) == queued, 'same items, same order'
        assert all(not r.result.done() for r in queued)
        assert store.events_of(EventType.merge_attempt) == []

    async def test_tip_pass_resets_the_halving_state(
        self, git_repo: Path, monkeypatch,
    ):
        """A green tip resets to the ``None`` sentinel.

        Kept on the PASS arm after δ narrowed this class, because it is a claim
        about the HALVING FEED, not about adoption: δ moved the landing into the
        finalize half and left ``_note_chain_outcome(out is None, 1 + links)``
        exactly where γ put it.  The fixture's own ``advances == []`` assertion
        still holds for the same reason — ``_run_inflight_verify`` reaches no
        CAS advance on any arm, δ included.
        """
        _git_ops, worker, _item, _chain, _res, _store, queued = await self._fixture(
            git_repo, monkeypatch, passed=True,
        )

        assert worker._chain_halving_state is None
        assert list(worker._lane_buffers['normal']) == queued

    @pytest.mark.parametrize(('passed', 'raises'), _NON_ADOPTING_ARMS)
    async def test_finalize_disposes_the_entry_without_a_phantom_head(
        self, git_repo: Path, monkeypatch, passed: bool, raises,
    ):
        """``_finalize_inflight`` returns False and strands no finalize head.

        The REQUEUED sentinel is checked ABOVE the VERIFYING -> FINALIZING hop
        precisely so a requeued item is never recorded as finalizing; a
        phantom head there wedges the whole queue behind an entry that will
        never complete.
        """
        from orchestrator.merge_types import InflightEntry

        _git_ops, worker, item, chain, res, _store, _q = await self._fixture(
            git_repo, monkeypatch, passed=passed, raises=raises,
        )
        done: asyncio.Future = asyncio.get_running_loop().create_future()
        done.set_result(res)
        entry = InflightEntry(
            item=item, lease=_local_lease(), verify_task=done,  # type: ignore[arg-type]
            merge_wt=item.merge_wt, was_speculative=True, chain=chain,
        )

        assert await worker._finalize_inflight(entry) is False
        assert worker._finalizing_head_entry() is None
        assert not item.request.result.done()


# ═══════════════════════════════════════════════════════════════════════════
# step-19: RED — the task's USER-OBSERVABLE SIGNAL, as one integration class.
#
# Everything above pins a seam in isolation. This class is the only place the
# seams are driven the way production drives them — `_dispatch_item` →
# `_deep_chain_placement` → `_run_inflight_verify` → the halving state → the
# NEXT round's `_deep_chain_placement` — because γ's whole thesis is a
# multi-ROUND property: a tip verdict changes what the next dispatch builds.
# A per-seam unit test can never see that, and the capability sidecar's
# "manual/integration signal" for `deep-tip-dispatch-and-halving-state-machine`
# is exactly this.
#
# Four properties, one per PRD row:
#   (a) dispatched depths follow the halving policy across rounds  (6→3→floor→6)
#   (b) the d=1 floor is byte-identical to a `chain_cap=0` dispatch
#   (c) EVERY merge_verify carries an int `chain_items >= 1` (η1's reader)
#   (d) conservation: no permit, worktree or `_spec-` lane left behind
# ═══════════════════════════════════════════════════════════════════════════


_DEEP_QUEUE = ('102', '103', '104', '105', '106')
"""Five chainable followers → ``queue_len`` 6, so ``cap=6`` binds, not the queue.

Deliberately >= cap: with a shorter queue ``min(queue_len, cap)`` would be the
QUEUE's length and the halving-walk assertions below would be pinning the
fixture rather than the policy.
"""


async def _fresh_repo(tmp_path: Path, name: str) -> Path:
    """Build a second independent fixture repo beside ``git_repo``'s.

    The floor-byte-identity test needs two runs of the SAME fixture at
    different ``chain_cap`` values, and a single repo cannot host both (the
    branch names collide and the first run's worktree pool is already warm).
    """
    repo = tmp_path / name
    repo.mkdir()
    await _setup_repo(repo)
    await _add_recording_seed_to_repo(repo)
    return repo


class _StubRemoteAllocator:
    """Minimal HostAllocator stand-in handing out ONE remote lease at a time.

    Used only by the ``chain_items`` scene, and REMOTE deliberately: a remote
    lease makes ``_run_post_merge_verify`` build its pool from the injected
    runner (merge_queue.py:2747-2758) instead of a real ``LocalRunner``, so
    ``VerifyRunnerPool.dispatch`` — the single ``merge_verify`` emission site —
    genuinely runs and genuinely emits.  Stubbing ``_run_post_merge_verify``
    instead (what the other scenes do) would emit no event at all and make
    every "every merge verify carries chain_items" assertion vacuous.
    """

    def __init__(self, lease) -> None:
        self._lease = lease
        self._held = False

    def free_host_count(self) -> int:
        return 0 if self._held else 1

    async def acquire(self, _local_factory):
        if self._held:
            return None
        self._held = True
        return self._lease

    async def release(self, _lease) -> None:
        self._held = False

    async def cancel_and_release(self, _lease) -> bool:
        self._held = False
        return True


class _DeepScene:
    """One repo + worker + queue, driven round after round by :meth:`round_`."""

    def __init__(self, git_ops, config, worker, item_head, repo, store) -> None:
        self.git_ops = git_ops
        self.config = config
        self.worker = worker
        self.head = item_head
        self.repo = repo
        self.store = store
        self.calls: list[dict] = []
        self.posted: list[dict] = []
        self.built: list[dict] = []
        self.released: list[tuple] = []
        self.lanes: list[str] = []
        self._round_no = 0

    @property
    def depths(self) -> list[int | None]:
        """The depth ACTUALLY dispatched each round, ``None`` for "no chain".

        Read off the ``ChainResult`` handed to ``_run_inflight_verify`` — the
        one place the built depth is a fact rather than an intention, since
        ``build_chain`` truncates.
        """
        return [
            None if c['chain'] is None else 1 + len(c['chain'].links)
            for c in self.calls
        ]

    async def round_(self, *, tag: str) -> dict:
        """Drive ONE full dispatch round through ``_dispatch_item``.

        Returns the recorded ``_run_inflight_verify`` kwargs for the round
        (plus ``'result'``).  Releases the host lease and any worktree the
        ordinary (non-chain) arm hands back, so the next round starts from the
        same resource state — which is what makes the ROUND SEQUENCE, and not
        a slowly-leaking fixture, the thing under test.
        """
        self._round_no += 1
        worker = self.worker
        item = _make_item(
            _make_req('101', '101', self.config, self.repo), self.head,
            _ephemeral_merge_wt(self.git_ops, f'{tag}-r{self._round_no}'),
        )
        entry = await worker._dispatch_item(item)
        assert entry is not None, 'dispatch must not decline: a host is free'
        assert entry.verify_task is not None
        await entry.verify_task

        rec = self.calls[-1]
        rec['item'] = item
        rec['round'] = self._round_no
        if entry.lease is not None:
            await worker._host_allocator.release(entry.lease)
        vr = rec['result']
        if vr.merge_wt is not None:
            # The ordinary arm hands its verified worktree to _finalize_inflight;
            # this scene stops short of finalize (γ lands nothing), so return it
            # here or the pool starves the next round.
            await worker._release_or_cleanup(vr.merge_wt, spec_warm=vr.spec_warm)
        return rec


async def _make_deep_scene(
    repo: Path, *, chain_cap: int, script: list[bool] | None, monkeypatch,
    allocator=None, frontier: int = 0,
) -> _DeepScene:
    """Build a scene whose verify oracle returns *script* verdicts in order.

    *script* is one bool per round (``True`` == the verify passed), or ``None``
    to install no oracle at all and let the REAL ``_run_post_merge_verify``
    run — which is what drives the true ``merge_verify`` emission site.  When
    *allocator* is given it is installed on the worker BEFORE first use, so
    ``_ensure_host_allocator``'s cache check (merge_queue.py:10288) short-
    circuits and the real one is never built.

    *frontier* seeds ``_verify_frontier_depth()`` — a pure ``len()`` over
    ``_frozen_inflight_entries()`` — with that many opaque sentinels, using the
    idiom already at test_merge_queue_depth_telemetry.py:477-482
    (``_worker_with_frontier`` + ``_sentinel_entry``) and at this module's
    ``TestRunInflightVerifyChainRedirect._fixture``.  It has to be STUBBED
    rather than driven: ``_dispatch_item`` never appends to ``_inflight`` (the
    only ``self._inflight.append`` is ``_inflight_append``, merge_queue.py:15693,
    which only the MERGER loop calls), so a scene driven purely through
    ``_dispatch_item`` has a structurally empty frontier and every
    ``frontier + 1`` arithmetic is observationally identical to a flat ``1``.
    """
    git_ops = _make_git_ops(repo, size=2)
    config = _make_config(repo, chain_cap=chain_cap)
    await _create_branch_editing(repo, 'task/101', 'a.txt', 'edit-101\n')
    for tid in _DEEP_QUEUE:
        await _create_branch_editing(repo, f'task/{tid}', f'f{tid}.txt', f'edit-{tid}\n')
    head = await _merge_commit_off_main(repo, 'task/101', '101')
    store = _CapturingEventStore()
    worker = _make_worker(git_ops)
    worker._event_store = store
    worker._lane_buffers['normal'].extend(
        _make_req(tid, tid, config, repo) for tid in _DEEP_QUEUE
    )
    if allocator is not None:
        worker._host_allocator = allocator
    if frontier:
        worker._frozen_inflight_entries = (  # type: ignore[assignment,method-assign]
            lambda: [object()] * frontier
        )
    scene = _DeepScene(git_ops, config, worker, head, repo, store)

    # ── the round recorder, installed ONCE ───────────────────────────────────
    # Once, not per round: re-wrapping `worker._run_inflight_verify` each round
    # would capture the PREVIOUS round's recorder as `real` and nest the
    # wrappers one deeper every round.
    real_verify = worker._run_inflight_verify

    async def _recording(_item, _lease, **kwargs):
        rec = dict(kwargs)
        rec['lease_is_local'] = _lease.is_local
        scene.calls.append(rec)
        rec['result'] = await real_verify(_item, _lease, **kwargs)
        return rec['result']

    monkeypatch.setattr(worker, '_run_inflight_verify', _recording)
    scene.built = _spy_build_chain(monkeypatch, passthrough=True)
    scene.released = _spy_chain_lane_release(monkeypatch)
    scene.lanes = _count_spec_lane_acquires(git_ops, monkeypatch)

    if script is not None:
        verdicts = list(script)

        async def _oracle(_git_ops, _req, merge_wt, **kwargs):
            scene.posted.append({'merge_wt': merge_wt, **kwargs})
            passed = verdicts.pop(0) if verdicts else True
            return None if passed else _fail_verify_result()

        monkeypatch.setattr(
            'orchestrator.merge_queue._run_post_merge_verify', _oracle,
        )
    return scene


def _round_transcript(scene: _DeepScene, idx: int) -> dict:
    """Normalise ONE round into a repo-independent, comparable transcript.

    Absolute paths and SHAs differ between two fixture repos, so the
    comparison is made on the FACTS instead: was a chain handed down, what
    depth/base/chain_items were labelled, was the item's OWN merge commit the
    thing verified, and which pool lane did the verify run in.
    """
    rec = scene.calls[idx]
    posted = scene.posted[idx]
    item = rec['item']
    return {
        'chain': rec['chain'],
        'chain_items': rec['chain_items'],
        'depth': rec['depth'],
        'probe_base': rec['probe_base'],
        'verified_the_items_own_merge_commit':
            posted['merge_sha'] == item.merge_result.merge_commit,
        'verify_worktree_name': Path(posted['merge_wt']).name,
        'speculative': posted.get('speculative'),
        'result_status': rec['result'].status,
        'result_has_worktree': rec['result'].merge_wt is not None,
    }


def _canary_says_deep(data: dict) -> bool:
    """The SHIPPED deep-verify classifier, transcribed verbatim.

    Copied character for character from ``scripts/merge-deep-canary-predicate.sh:84``
    (η1's ``deep = [... if isinstance(d.get("chain_items"), int) and
    d["chain_items"] >= 2]``) rather than restated in the assertion's own
    words.  A restatement would drift the moment the emitter's unit changed —
    which is exactly the emitter/reader units mismatch this transcription
    exists to fence.  ζ's "first ``chain_items >= 2`` verify observed" deploy
    signal and η1's deep-fail-rate DENOMINATOR are both this expression, so if
    it disagrees with the emitter the canary is silently measuring nothing.
    """
    return isinstance(data.get('chain_items'), int) and data['chain_items'] >= 2


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestDeepDispatchRoundsIntegration:
    """γ end to end: successive rounds, a scripted oracle, real git."""

    async def test_dispatch_depths_follow_the_halving_walk(
        self, git_repo: Path, monkeypatch,
    ):
        """PRD row "Halving walk isolates bad item": 6 → 3 → floor → 6.

        Rounds 1-2 fail, so the bisector halves twice and lands on the d=1
        floor, where the gate declines and the pipeline is back on today's
        adjacent verify.  Round 3 (the floor round) PASSES — and PRD decision 5
        is explicit that **any** pass resets — so round 4 must chain at
        ``min(queue_len, cap)`` again.  Without that reset the walk is a
        one-way ratchet: two red rounds would disable deep merge-ahead for the
        entire life of the worker process, silently, which is precisely the
        failure mode ``next_halving_state``'s ``max(1, ...)`` floor exists to
        prevent one step earlier.
        """
        scene = await _make_deep_scene(
            git_repo, chain_cap=6, script=[False, False, True, True],
            monkeypatch=monkeypatch,
        )

        for _ in range(4):
            await scene.round_(tag='walk')

        assert scene.depths == [6, 3, None, 6]
        assert scene.worker._chain_halving_state is None, 'a pass resets'

    async def test_halving_state_walks_down_in_lockstep_with_the_depths(
        self, git_repo: Path, monkeypatch,
    ):
        """The state AFTER each round, which is what the next round reads."""
        scene = await _make_deep_scene(
            git_repo, chain_cap=6, script=[False, False, True, True],
            monkeypatch=monkeypatch,
        )
        seen: list[int | None] = []

        for _ in range(4):
            await scene.round_(tag='state')
            seen.append(scene.worker._chain_halving_state)

        assert seen == [3, 1, None, None]

    async def test_floor_round_is_byte_identical_to_a_cap_zero_dispatch(
        self, tmp_path: Path, monkeypatch,
    ):
        """(b) The d=1 floor IS today's adjacent verify — not a mimic of it.

        Byte-identity here is BY CONSTRUCTION: ``select_chain_depth`` returns
        ``None`` below 2, so at the floor no chain code executes at all.  This
        test is the fence around that construction — it compares the floor
        round of a ``chain_cap=6`` walk against the first round of the same
        fixture at ``chain_cap=0`` (the shipped kill switch) and requires the
        two dispatches to agree on every observable.
        """
        deep_repo = await _fresh_repo(tmp_path, 'deep')
        flat_repo = await _fresh_repo(tmp_path, 'flat')
        deep = await _make_deep_scene(
            deep_repo, chain_cap=6, script=[False, False, True],
            monkeypatch=monkeypatch,
        )
        for _ in range(2):
            await deep.round_(tag='floor')
        # Snapshot BEFORE the floor round, not after it: rounds 1-2 chained and
        # each took a lane, so a post-hoc count cannot tell the floor round's
        # own work from theirs.
        n_built_before_floor = len(deep.built)
        n_lanes_before_floor = len(deep.lanes)
        n_released_before_floor = len(deep.released)
        await deep.round_(tag='floor')

        flat = await _make_deep_scene(
            flat_repo, chain_cap=0, script=[True], monkeypatch=monkeypatch,
        )
        await flat.round_(tag='flat')

        assert deep.depths[2] is None, 'round 3 must be the FLOOR round'
        assert _round_transcript(deep, 2) == _round_transcript(flat, 0)
        # ...and the floor round did no chain work of any kind.
        assert len(deep.built) == n_built_before_floor, 'zero build_chain calls'
        assert len(deep.released) == n_released_before_floor, 'no lane to release'
        assert flat.built == [], 'cap=0 never reaches the builder either'
        # The warm-swap block ran on BOTH — one _spec- acquisition for the
        # item's own merge commit, which is what "today's adjacent verify"
        # means observably (see this module's note on how the bypass is seen).
        assert len(deep.lanes) == n_lanes_before_floor + 1
        assert deep.lanes[-1] == deep.calls[2]['item'].merge_result.merge_commit
        assert flat.lanes == [flat.calls[0]['item'].merge_result.merge_commit]

    @pytest.mark.parametrize('frontier', [0, 2], ids=['frontier-0', 'frontier-2'])
    async def test_every_merge_verify_carries_chain_items(
        self, tmp_path: Path, monkeypatch, frontier: int,
    ):
        """(c) η1's reader can see a deep verify at all.

        ``scripts/merge-deep-canary-predicate.sh:84`` counts merge verifies
        with ``chain_items >= 2``; today it always reports 0 because no such
        event exists.  This drives the REAL emission site
        (``VerifyRunnerPool.dispatch``) over a mixed run — a slot-1 head
        verify, a deep tip verify, a halved tip verify and a floor verify —
        and requires the field present, integral and truthful on all of them.

        Parameterised over the verify frontier, and the expected list is the
        SAME at both: ``chain_items`` is in CHAIN-ITEM units and is
        deliberately frontier-INDEPENDENT.  At ``frontier=0`` the list was
        already this, so the parameterisation is what converts a vacuous pass
        into a real pin — a frontier-derived emitter would produce
        ``[1, 8, 5, 3]`` at ``frontier=2`` and fail here.
        """
        from orchestrator.verify_runner import HostLease

        repo = await _fresh_repo(tmp_path, f'events{frontier}')
        lease = HostLease(
            name='laptop', runner=_fake_pass_runner(), is_local=False,
        )
        scene = await _make_deep_scene(
            repo, chain_cap=6, script=None, monkeypatch=monkeypatch,
            allocator=_StubRemoteAllocator(lease), frontier=frontier,
        )

        # slot 1 (the trust anchor) — never chained, so a flat 1.
        #
        # base_sha is REAL MAIN, deliberately: a slot-1 item is merged onto
        # main, and a stale base would fire _dispatch_item's Mechanism-2
        # staleness re-merge, which sets `_remerge_occurred` and makes every
        # LATER speculative round re-merge as `chain_invalidated` — arriving at
        # _deep_chain_placement non-speculative, so no round after this one
        # would chain at all and the whole scene would go quietly vacuous.
        head_item = _make_item(
            _make_req('101', '101', scene.config, repo), scene.head,
            _ephemeral_merge_wt(scene.git_ops, 'ev-head'), speculative=False,
            base_sha=await _rev_parse(repo, 'main'),
        )
        entry = await scene.worker._dispatch_item(head_item)
        assert entry is not None and entry.verify_task is not None
        await entry.verify_task
        await scene.worker._host_allocator.release(entry.lease)
        _vr = scene.calls[-1]['result']
        if _vr.merge_wt is not None:
            await scene.worker._release_or_cleanup(
                _vr.merge_wt, spec_warm=_vr.spec_warm,
            )

        # slot 2: deep (6) → halved (3) → floor (no chain).  The fake runner
        # always passes, so drive the halving by hand rather than by verdict —
        # this test is about the EVENT, not the policy (that is test (a)).
        for state, tag in ((None, 'ev-deep'), (3, 'ev-halved'), (1, 'ev-floor')):
            scene.worker._chain_halving_state = state
            await scene.round_(tag=tag)

        events = scene.store.events_of(EventType.merge_verify)
        assert len(events) == 4, 'one merge_verify per dispatched round'
        for ev in events:
            assert isinstance(ev['data'].get('chain_items'), int)
            assert not isinstance(ev['data']['chain_items'], bool)
            assert ev['data']['chain_items'] >= 1

        assert [e['data']['chain_items'] for e in events] == [1, 6, 3, 1]
        deep_events = [e for e in events if e['data']['chain_items'] >= 2]
        assert len(deep_events) == 2, 'η1 must be able to SEE a deep verify'

    async def test_non_chain_speculative_dispatch_emits_one_at_any_frontier(
        self, tmp_path: Path, monkeypatch,
    ):
        """REVIEW FIX 1: the SHIPPED default (``chain_cap=0``) emits a flat 1.

        The coverage hole this closes: no test anywhere combined "non-chain
        speculative dispatch" with "a non-empty verify frontier" AND the real
        ``VerifyRunnerPool.dispatch`` emission.  ``_make_deep_scene`` never
        populated ``_inflight`` (so its frontier was structurally 0 and
        ``frontier + 1`` was indistinguishable from ``1``), while the only
        non-empty-frontier tests stub ``_run_inflight_verify`` out and never
        reach the emitter at all.

        ``chain_items`` is in CHAIN-ITEM units: the dispatching item is chain
        item #1 and a dispatch that built no chain contributes exactly that
        one item.  The frozen-prefix height it used to (wrongly) fold in is
        still carried, unchanged, by the separate always-present ``depth``.
        """
        from orchestrator.verify_runner import HostLease

        repo = await _fresh_repo(tmp_path, 'flat-frontier')
        lease = HostLease(
            name='laptop', runner=_fake_pass_runner(), is_local=False,
        )
        scene = await _make_deep_scene(
            repo, chain_cap=0, script=None, monkeypatch=monkeypatch,
            allocator=_StubRemoteAllocator(lease), frontier=2,
        )

        await scene.round_(tag='flat-frontier')

        assert scene.built == [], 'cap=0 is the kill switch: no chain was built'
        assert scene.calls[0]['chain'] is None
        events = scene.store.events_of(EventType.merge_verify)
        assert len(events) == 1
        data = events[0]['data']
        assert data['chain_items'] == 1, (
            'a speculative dispatch with no chain verifies exactly one item, '
            'no matter how many other verifies are in flight ahead of it'
        )
        # The frontier height is not LOST — it moved to the field that owns it.
        assert data['depth'] == 2, 'the frozen-prefix height still rides `depth`'

    async def test_the_shipped_canary_classifier_fires_only_on_deep_rounds(
        self, tmp_path: Path, monkeypatch,
    ):
        """REVIEW FIX 1: agreement with η1's own expression, not a restatement.

        Applies ``scripts/merge-deep-canary-predicate.sh:84``'s classifier
        verbatim (see ``_canary_says_deep``) to a mixed run and requires it
        True for the genuine deep rounds and False for the head, adjacent and
        floor rounds.  A frontier-derived ``chain_items`` would make the
        adjacent round at ``frontier >= 1`` classify as "deep", inflating η1's
        deep-fail-rate denominator with rounds that chained nothing and firing
        ζ's "first deep verify observed" signal on day one under the shipped
        ``chain_cap=0``.
        """
        from orchestrator.verify_runner import HostLease

        repo = await _fresh_repo(tmp_path, 'canary')
        lease = HostLease(
            name='laptop', runner=_fake_pass_runner(), is_local=False,
        )
        scene = await _make_deep_scene(
            repo, chain_cap=6, script=None, monkeypatch=monkeypatch,
            allocator=_StubRemoteAllocator(lease), frontier=2,
        )

        # slot 1 (head, non-speculative) — see the sibling test for why
        # base_sha must be REAL MAIN here.
        head_item = _make_item(
            _make_req('101', '101', scene.config, repo), scene.head,
            _ephemeral_merge_wt(scene.git_ops, 'can-head'), speculative=False,
            base_sha=await _rev_parse(repo, 'main'),
        )
        entry = await scene.worker._dispatch_item(head_item)
        assert entry is not None and entry.verify_task is not None
        await entry.verify_task
        await scene.worker._host_allocator.release(entry.lease)
        _vr = scene.calls[-1]['result']
        if _vr.merge_wt is not None:
            await scene.worker._release_or_cleanup(
                _vr.merge_wt, spec_warm=_vr.spec_warm,
            )

        for state, tag in ((None, 'can-deep'), (3, 'can-halved'), (1, 'can-floor')):
            scene.worker._chain_halving_state = state
            await scene.round_(tag=tag)

        events = scene.store.events_of(EventType.merge_verify)
        assert len(events) == 4
        verdicts = [_canary_says_deep(e['data']) for e in events]
        assert verdicts == [False, True, True, False], (
            'head=False, deep(6)=True, halved(3)=True, floor(adjacent)=False'
        )
        # The nullable companion must agree with the SAME classifier, event for
        # event: `chain_build_ms` is the per-round DISPATCH STALL, so a
        # non-None on a round that chained nothing would attribute a build cost
        # to a build that never happened, and a None on a deep round would hide
        # the stall η1 reads it for.  Asserted against `verdicts` rather than
        # re-derived, so the two fields cannot drift apart.
        stalls = [e['data']['chain_build_ms'] for e in events]
        assert [v is not None for v in stalls] == verdicts, (
            'chain_build_ms is non-None exactly on the chain_items >= 2 rounds'
        )
        assert all(isinstance(v, int) and v >= 0 for v in stalls if v is not None)

    async def test_a_deep_round_is_frontier_independent_on_the_chain_arm(
        self, tmp_path: Path, monkeypatch,
    ):
        """REVIEW FIX 1: a 2-link chain at frontier 2 emits 3, never 5.

        The chain arm COMPUTES ``1 + len(chain.links)`` rather than
        accumulating onto whatever the caller passed, so the emitted unit is a
        fact about the verified tree — the dispatching item plus the links that
        were actually built — and carries no frontier contribution at all.
        """
        from orchestrator.verify_runner import HostLease

        repo = await _fresh_repo(tmp_path, 'deep-frontier')
        lease = HostLease(
            name='laptop', runner=_fake_pass_runner(), is_local=False,
        )
        scene = await _make_deep_scene(
            repo, chain_cap=3, script=None, monkeypatch=monkeypatch,
            allocator=_StubRemoteAllocator(lease), frontier=2,
        )

        await scene.round_(tag='deep-frontier')

        chain = scene.calls[0]['chain']
        assert chain is not None and len(chain.links) == 2, 'fixture must chain'
        events = scene.store.events_of(EventType.merge_verify)
        assert len(events) == 1
        assert events[0]['data']['chain_items'] == 3 == 1 + len(chain.links)
        assert _canary_says_deep(events[0]['data']) is True

    async def test_conservation_holds_across_the_whole_run(
        self, git_repo: Path, monkeypatch,
    ):
        """(d) PRD ι: nothing leaks over a full mixed run.

        Speculation permits, the worktree ledger and the ``_spec-`` pool are
        the three resources γ touches, and a chain that acquires a lane per
        round is exactly the shape that strands one.
        """
        from orchestrator.warm_lane_pool import LaneState

        scene = await _make_deep_scene(
            git_repo, chain_cap=6, script=[False, False, True, True],
            monkeypatch=monkeypatch,
        )

        for _ in range(4):
            await scene.round_(tag='conserve')

        assert scene.worker.speculation_accounting_violations() == []
        assert scene.worker.worktree_ledger_violations(grace_secs=0.0) == []
        assert all(s is LaneState.FREE for s in _lane_states(scene.git_ops)), \
            'every chain lane went back to the pool'
        assert len(scene.released) == 3, 'released once per round that chained'
