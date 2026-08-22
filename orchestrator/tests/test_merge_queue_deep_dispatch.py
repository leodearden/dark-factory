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

    def test_is_pure_with_no_running_event_loop(self) -> None:
        """No I/O, no worker, no clock — callable outside an event loop."""
        from orchestrator.merge_queue import select_chain_depth

        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()  # proves there is none
        assert select_chain_depth(6, 10, None) == 6


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

    def test_is_pure_with_no_running_event_loop(self) -> None:
        from orchestrator.merge_queue import next_halving_state

        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        assert next_halving_state(False, 6) == 3


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

    def test_deep_target_depth_delegates_to_the_pure_policy(
        self, tmp_path: Path,
    ) -> None:
        """``_deep_target_depth`` composes live state with select_chain_depth.

        It must DELEGATE rather than re-derive the formula, so there is exactly
        one place the dispatch invariant is written down.
        """
        from orchestrator.merge_queue import select_chain_depth

        worker = self._worker(tmp_path)

        for cap, queue_len in ((0, 10), (6, 1), (6, 4), (6, 10), (1, 10)):
            assert worker._deep_target_depth(cap, queue_len) == select_chain_depth(
                cap, queue_len, worker._chain_halving_state,
            )

        worker._note_chain_outcome(False, 6)   # state → 3
        for cap, queue_len in ((6, 10), (6, 2), (3, 10)):
            assert worker._deep_target_depth(cap, queue_len) == select_chain_depth(
                cap, queue_len, worker._chain_halving_state,
            )

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


def _spy_warm_acquire(monkeypatch) -> list[tuple]:
    """Record every ``_acquire_warm_verify_worktree`` call; return the record list.

    The chain arm must not call it AT ALL: the chain lane already holds the
    tip, and a second acquisition would take a DIFFERENT lane, re-check-out a
    different commit, and can land on the serial ``_merge-verify`` lane that
    ``acquire_chain_build_lane`` explicitly refuses (merge_liveness.py:806-814,
    DF-3071).
    """
    calls: list[tuple] = []

    async def _recording(git_ops, req, merge_wt, merge_commit, **kwargs):
        calls.append((merge_wt, merge_commit, kwargs))
        return merge_wt, False

    monkeypatch.setattr(
        'orchestrator.merge_queue._acquire_warm_verify_worktree', _recording,
    )
    return calls


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
        item = _make_item(_make_req('101', '101', config, git_repo), head, git_repo)
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
        _spy_warm_acquire(monkeypatch)
        posted = _spy_post_merge_verify(monkeypatch)

        await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert len(posted) == 1
        assert posted[0]['merge_sha'] == chain.tip
        assert posted[0]['merge_wt'] == chain.lane
        assert posted[0]['merge_sha'] != head == item.merge_result.merge_commit
        assert posted[0]['merge_wt'] != item.merge_wt

    async def test_warm_acquire_is_not_called_on_the_chain_arm(
        self, git_repo: Path, monkeypatch,
    ):
        """The lane already holds the tip — a second acquisition would be wrong.

        It would take a DIFFERENT lane, re-check-out a different commit, and
        could land on the serial ``_merge-verify`` lane that
        ``acquire_chain_build_lane`` explicitly refuses (DF-3071).
        """
        _git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        warm = _spy_warm_acquire(monkeypatch)
        _spy_post_merge_verify(monkeypatch)
        before = worker._verify_attempt_count

        await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert warm == [], 'chain arm must bypass the warm-swap block entirely'
        assert worker._verify_attempt_count == before, (
            'the attempt counter is staged INSIDE the warm-swap block, so a '
            'bypassed block must leave the cold-verify safety valve untouched'
        )

    async def test_chain_items_counts_the_actually_built_depth(
        self, git_repo: Path, monkeypatch,
    ):
        """``chain_items == frontier + 1 + len(chain.links)`` — what actually ran.

        A 6-target that only built 2 links must emit 3 at frontier 0, never 7:
        the field is a fact about the tree that ran, not about the target that
        was asked for.
        """
        _git_ops, worker, item, chain, _head = await self._fixture(git_repo, frontier=1)
        _spy_warm_acquire(monkeypatch)
        posted = _spy_post_merge_verify(monkeypatch)

        await worker._run_inflight_verify(
            item, _local_lease(), chain=chain, chain_items=2,  # frontier 1 + 1
        )

        assert posted[0]['chain_items'] == 4 == 1 + 1 + len(chain.links)

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
        _spy_warm_acquire(monkeypatch)
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
        _spy_warm_acquire(monkeypatch)
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
        """An infra error is NOT a tip verdict, so it must not halve the state.

        Halving off a RuntimeError would walk the policy down to the d=1 floor
        on repeated infra noise and silently disable deep merge-ahead.
        """
        _git_ops, worker, item, chain, _head = await self._fixture(git_repo)
        _spy_warm_acquire(monkeypatch)
        _spy_post_merge_verify(monkeypatch, raises=RuntimeError('verify exploded'))
        noted: list[tuple] = []
        monkeypatch.setattr(
            worker, '_note_chain_outcome', lambda p, d: noted.append((p, d)),
        )

        await worker._run_inflight_verify(item, _local_lease(), chain=chain)

        assert noted == []
        assert worker._chain_halving_state is None

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
        ephemeral = git_ops.worktree_base / '_merge-deadbeef'
        ephemeral.mkdir(parents=True, exist_ok=True)
        item.merge_wt = ephemeral
        worker._register_owned_merge_worktree(ephemeral)
        _spy_warm_acquire(monkeypatch)
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

        Warm-swap still runs, the item's OWN merge commit is what gets
        verified, and nothing is released as a chain lane.
        """
        _git_ops, worker, item, chain, head = await self._fixture(git_repo)
        warm = _spy_warm_acquire(monkeypatch)
        posted = _spy_post_merge_verify(monkeypatch)
        released = _spy_chain_lane_release(monkeypatch)

        res = await worker._run_inflight_verify(item, _local_lease())

        assert len(warm) == 1, 'warm-swap must still run on the ordinary path'
        assert posted[0]['merge_sha'] == head == item.merge_result.merge_commit
        assert posted[0]['merge_wt'] == item.merge_wt
        assert posted[0]['chain_items'] == 1, 'default: a one-item tree'
        assert released == [], 'no chain, nothing to release'
        assert res.merge_wt == item.merge_wt, (
            'the ordinary pass path still hands merge_wt to _finalize_inflight'
        )
        await merge_liveness.release_chain_build_lane(
            worker._git_ops, chain.lane, warm=chain.lane_warm,
        )
