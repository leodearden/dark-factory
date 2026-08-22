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

from orchestrator.config import GitConfig, MergeDeepConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeRequest, SpeculativeMergeWorker
from orchestrator.merge_types import MergeResult, QueuedBranch, RealMergeItem

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
