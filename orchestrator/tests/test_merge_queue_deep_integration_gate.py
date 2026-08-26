"""Integration gate ι: two-way boundary matrix for deep merge-ahead (B+H leaf).

PRD: ``plans/deep-merge-ahead-prd.md`` task ι (Phase 2 vertical slice, the
C-as-integration-gate leaf).  Task 3187.

All four prerequisite legs are LANDED and BEHAVIOUR-FROZEN for this batch:

  α (3183) — ``merge_deep.chain_cap`` + the ``chain_cap=0`` kill switch
      (config.py ``MergeDeepConfig``; RELOADABLE_FIELDS green tier).  Unit
      coverage: test_config_merge_deep.py.
  β (3184) — ``build_chain`` (build-on-dispatch): sequential submission-order
      merges in ONE scratch worktree, truncate at the first textual conflict,
      never emit per-item outcomes.  Unit coverage: test_merge_queue_build_chain.py.
  γ (3185) — deep-tip verify dispatch + the halving state machine
      (``_deep_chain_placement``, ``select_chain_depth``, ``next_halving_state``,
      ``chain_items`` telemetry).  Unit coverage: test_merge_queue_deep_dispatch.py.
  δ (3186) — prefix landing on tip pass: the in-order CAS walk, head-verify
      cancellation with a clean lease release, stale-CAS abort, and
      ``landed_via_chain``.  Unit coverage: test_merge_queue_deep_landing.py.

WHAT THIS FILE IS
-----------------
A TEST-ONLY COMPOSITION gate implementing the PRD's §Boundary-test-sketch
matrix against BOTH sides of the slice — the WORKER side (dispatch, halving,
chain build) and the CAS/LEDGER side (in-order advance, permit ledgers, the
worktree ledger, request liveness, ItemLifecycle, and workflow.py's merge
thrash ladder).  "Two-way" is the whole point: several rows are already pinned
one-way upstream, and the gate's added value is (a) driving the OTHER side of
the same claim and (b) asserting the rows hold in one CONTINUOUS multi-round
run rather than in isolated single-round scenes.

THE 11-ROW BOUNDARY MATRIX (verbatim from the PRD's §Boundary-test sketch)
--------------------------------------------------------------------------
  1.  Tip pass lands full prefix       | 4-item clean chain, tip verify green
      → 4 in-order landings, one verify, ``landed_via_chain=4``, audits green
  2.  Tip fail leaves queue intact     | chain built, tip verify red
      → zero landings via chain, items still queued, halving state = 2
  3.  Halving walk isolates bad item   | item 3 of 6 genuinely red
      → depths 6→3→1 over rounds; items 1–2 land sequentially at floor; deep
        resumes after the bad item blocks
  4.  Chain conflict truncates silently| item 2 conflicts with item 1 textually
      → chain = [item 1], no conflict outcome for item 2, item 2 handled
        sequentially later
  5.  Head-fail + tip-pass             | head verify red (flake), tip green
      → full prefix lands (tip authoritative), head verify cancelled
  6.  Stale CAS aborts walk            | main advanced externally mid-verify
      → walk aborts at first CAS failure; unlanded items requeue; next round
        rebuilds
  7.  Kill switch byte-identity        | cap=0
      → dispatch/behaviour identical to pre-PRD golden transcript
  8.  Deep fails never feed thrash guard| 2 consecutive tip fails
      → zero blocked MergeOutcomes, ``consecutive_merge_thrash`` untouched
        (3003's signature class cannot recur via chains)
  9.  Lease released on head-cancel    | tip pass cancels in-flight head verify
      → 3071's oracle reads IDLE within one round
  10. Hot-reload                       | cap 0→6 via reload_config
      → next dispatch round builds a chain; no restart
  11. Timeout margin                   | 16-item chain (cap 32)
      → verify completes ≪ 7200 s or times out cleanly via the existing path

Row 9's PRD postcondition names ``warm-lane-lock-guard.sh check`` (3071's
oracle).  That script is REIFY-side and does NOT exist in this repo, so the
equivalent in-process two-axis oracle is used instead:
``verify_cancel.lane_lock_holder_pids_strict(verify_cancel.lane_lock_path(...))``
(the kernel flock axis) plus ``verify_cancel.read_lock_holder_pgid(...)`` (the
rendezvous axis), which together are what ``GitOps._merge_verify_lease_active``
reads.

SUBSUMPTION TABLE — what each row OWES that upstream does not already pay
-------------------------------------------------------------------------
Measured against the two upstream modules read in full.  A row marked FULLY
SUBSUMED is deliberately NOT re-cloned as a unit assertion here; it appears
only at COMPOSITION level (inside a multi-round run, with the two-way oracle
asserted after every round).  Filled in per row as each is written; the
verdict column is re-confirmed in step-16.

  row | verdict vs upstream        | upstream owner (node id)
  ----+----------------------------+------------------------------------------
   1  | FULLY SUBSUMED (unit)      | test_merge_queue_deep_landing.py::
      |   → composition only       |   TestDeepLandingEndToEnd::
      |                            |   test_one_passing_tip_lands_the_whole_
      |                            |   prefix_in_order; ::TestInOrderCasWalk
   2  | FULLY SUBSUMED (unit)      | test_merge_queue_deep_dispatch.py (fail
      |   → composition only       |   arm) + ::TestDeepLandingEndToEnd::
      |                            |   test_a_failing_tip_lands_nothing_and_
      |                            |   the_item_still_lands_later
   3  | PARTIAL — policy only      | test_merge_queue_deep_dispatch.py
      |   → ORIGINAL: isolation    |   (the depths [6,3,None,6] walk, driven by
      |                            |   a POSITIONAL pass/fail script)
   4  | FULLY SUBSUMED (unit)      | test_merge_queue_deep_dispatch.py +
      |   → composition only       |   test_merge_queue_deep_landing.py
      |                            |   (truncator 105 vs link 102)
   5  | FULLY SUBSUMED (unit)      | test_merge_queue_deep_landing.py::
      |   → composition only       |   TestHeadCancelOnAdoption
   6  | FULLY SUBSUMED (unit)      | test_merge_queue_deep_landing.py::
      |   → composition only       |   TestStaleCasAbortLeavesTheRestAlone
   7  | PARTIAL — single round     | test_merge_queue_deep_dispatch.py
      |   → ORIGINAL: round-seq    |   (transcript compare) +
      |                            |   ::TestDeepLandingEndToEnd::
      |                            |   test_the_shipped_kill_switch_reaches_
      |                            |   no_delta_code (golden dict, ONE round)
   8  | PARTIAL — one-way          | test_merge_queue_deep_landing.py
      |   → ORIGINAL: the LADDER   |   (merge-queue half: event SILENCE)
   9  | FULLY SUBSUMED (unit)      | test_merge_queue_deep_landing.py::
      |   → composition only       |   TestHeadCancelLeavesTheLaneIdle
   10 | FULLY SUBSUMED (unit)      | test_merge_queue_deep_landing.py::
      |   → composition only       |   test_flipping_the_cap_in_place_starts_
      |                            |   landing_chains
   11 | NOT COVERED ANYWHERE       | (none — no test in the tree builds a
      |   → ORIGINAL, in full      |   chain deeper than 3 links, and there
      |                            |   are zero hits for the 7200 s budget on
      |                            |   any deep path)

Plus one CROSS-CUTTING gap this gate owns outright: CONSERVATION across a
MIXED multi-round run that actually lands.  deep_dispatch's conservation test
never finalizes (it lands nothing) and every deep_landing ``_assert_quiescent``
call is single- or two-round, so nothing upstream asserts the six worker
surfaces plus the CAS/ledger surfaces stay green round after round.

PROVENANCE — every helper below is a PORT, not an original
-----------------------------------------------------------
``orchestrator/tests/`` has no ``__init__.py``, so a cross-module helper import
would be a bare-module-name import of a sibling TEST file, coupling the two
suites' collection order.  CLONING is the sanctioned convention here, and this
table is the cost of it: a production change to any of these seams turns this
file AND its origin red, and the origin is where the unit-level contract lives.

  helper in this file            | ported from
  -------------------------------+-------------------------------------------
  _setup_repo,                   | test_merge_queue_deep_landing.py
  _add_recording_seed_to_repo,   |   (itself cloned from
  git_repo,                      |   test_merge_queue_deep_dispatch.py)
  _make_spec_git_config,         |
  _make_git_ops, _make_config,   |
  _make_req, _make_item,         |
  _ephemeral_merge_wt,           |
  _make_worker,                  |
  _CapturingEventStore,          |
  _create_branch_editing,        |
  _rev_parse, _shared_txt_with,  |
  _merge_commit_off_main,        |
  _local_lease, _fake_pass_runner|
  _fail_verify_result,           |
  _spy_post_merge_verify,        |
  _spy_chain_lane_release,       |
  _spy_advance_main,             |
  _PermitCensus, _permit_census, |
  _drain_residue,                |
  _finalized_rows,               |
  _events_for_task               |
  -------------------------------+-------------------------------------------
  _assert_two_way_quiescent      | the six-surface half is
     (six-surface half)          |   test_merge_queue_deep_landing.py::
                                 |   _assert_quiescent, ITSELF a clone of
                                 |   test_merge_queue_invariant_integration_
                                 |   gate.py::_assert_quiescent (the original).
                                 |   This is the THIRD clone; the CAS/ledger
                                 |   half (permit census by TOKEN, thrash
                                 |   ladder equality) is ORIGINAL here.
  -------------------------------+-------------------------------------------
  _canary_predicate_items_per    | a verbatim transcription of the SHIPPED
                                 |   scripts/merge-deep-canary-predicate.sh,
                                 |   re-cloned from
                                 |   test_merge_queue_deep_landing.py so the
                                 |   gate reads the shipped consumer's
                                 |   arithmetic rather than restating it.
  -------------------------------+-------------------------------------------
  _GateScene, _make_gate_scene,  | an n-follower GENERALISATION of
  _gate_round_transcript,        |   test_merge_queue_deep_landing.py::
  _gate_sequence_transcript      |   _DeltaScene / _make_delta_scene /
                                 |   _delta_round_transcript (hard-wired to 3
                                 |   links + 1 truncator, one round at a time).

SCOPE — TEST-ONLY, production BEHAVIOUR-FROZEN
-----------------------------------------------
Every production surface exercised below (α–δ, tasks 3183–3186) already
SHIPPED and is frozen for this batch.  If a scenario surfaces a GENUINE
production defect, ESCALATE (``category='design_concern'`` or
``'scope_violation'``) naming the row, the observed value and the
``merge_queue.py`` symbol — do NOT edit production here.  An edit from this
task would widen the concurrency lock onto merge_queue.py (21k lines), which
three sibling PRD leaves (ε, ζ, κ) queue behind.

NO LINE-OFFSET CITATIONS
-------------------------
Nothing in this file cites a ``module.py:NNNN`` offset, deliberately: an offset
into merge_queue.py / git_ops.py / config.py is stale within days of being
written, and a citation a reader has to re-verify costs more than it saves.
Every reference names the SYMBOL (function, method, class, constant, or pytest
node id) instead — locate it with grep/search.  Keep it that way when editing.

HARNESS NOTES
-------------
  * ``orchestrator/pyproject.toml`` does NOT set ``asyncio_mode`` → pytest-asyncio
    runs STRICT, so ``@pytest.mark.asyncio`` is required on async test classes.
    (``asyncio_mode`` is set only in the REPO-ROOT pyproject, which a
    ``cd orchestrator && uv run pytest`` invocation does not read.)
  * That same config turns "marked with @pytest.mark.asyncio but not an async
    function" into an ERROR — never put a sync ``test_*`` inside a marked class.
    Sync tests live in their OWN unmarked class.
  * The ini default per-test ``timeout`` is 60 s; every class doing real git plus
    a real worker carries ``@pytest.mark.timeout(300)``.  The CLI ``--timeout=300``
    some runners pass does NOT remove the need for the mark, and pytest-timeout's
    thread method ``os._exit()``s the xdist worker on overrun under
    ``--max-worker-restart=0``.
  * Monkeypatching is INSTANCE-level wherever possible:
    test_merge_queue_reachback_patch_guard.py freezes the
    ``orchestrator.merge_queue.<private>`` reach-back surface.  The sanctioned
    module-level exceptions are ``build_chain``, ``_run_post_merge_verify``,
    ``release_chain_build_lane`` and ``CHAIN_BUILD_TIMEOUT_SECS``.
  * ``fused-memory/scripts/check_bare_magicmock_config.py`` forbids a bare
    ``MagicMock()`` bound to a config-named variable — use ``_make_config(...)``.
  * No new pytest marker is introduced, so ``orchestrator/pyproject.toml`` stays
    untouched (test_marker_registration_drift.py fails an unregistered one).
"""

from __future__ import annotations

import asyncio
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

# ── repo fixtures (cloned from test_merge_queue_deep_landing.py) ──────────────


async def _setup_repo(repo: Path) -> None:
    """Init a repo with a 20-line shared.txt plus disjoint.txt.

    ``shared.txt`` gets **20** numbered lines rather than 3: git's 3-line diff
    context window makes near-line edits in a tiny file conflict even when they
    touch different lines (gotcha documented in
    test_merge_queue_conflict_graph.py).  20 lines makes a line-1 vs line-15
    edit pair genuinely non-conflicting, so this file can build both
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
    warm ``_spec-`` lane assertion in this module vacuous.  A bare path is
    enough — disposal of a missing worktree is a no-op either way.
    """
    return git_ops.worktree_base / f'_merge-{tag}'


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring)."""
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


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
    not be reachable from ``main``, or every "landed on main" assertion below
    would hold vacuously before the walk ever ran.
    """
    await _run(['git', 'checkout', '-b', f'_tmp-{label}', 'main'], cwd=repo)
    await _run(['git', 'merge', '--no-ff', '-m', f'merge {branch}', branch], cwd=repo)
    sha = await _rev_parse(repo)
    await _run(['git', 'checkout', 'main'], cwd=repo)
    return sha


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


def _fail_verify_result():
    """A failing :class:`VerifyResult` — a RED tip verdict."""
    from orchestrator.verify import VerifyResult

    return VerifyResult(
        passed=False, test_output='tip is red', lint_output='', type_output='',
        summary='fail', category='',
    )


# ── dispatch-scene spies ─────────────────────────────────────────────────────


def _spy_post_merge_verify(
    monkeypatch, outcome=None, *, raises=None,
    park: set[str] | None = None,
    parked: asyncio.Event | None = None,
) -> list[dict]:
    """Replace ``_run_post_merge_verify`` with a recorder returning *outcome*.

    ``outcome=None`` is a PASS in this function's vocabulary; a
    :class:`VerifyResult` is a FAIL.  *raises* makes the verify blow up
    instead, which is the third exit — the one that stays NON-adopting.

    *park* names task ids whose verify NEVER RETURNS — the shape a live head
    has while the speculative slot verifies the chain tip.  *parked*, when
    given, is set the moment such a call is entered, so a test can wait for the
    swap to have happened rather than sleeping for it.  The park is ONE-SHOT
    per task id, deliberately.
    """
    calls: list[dict] = []
    parked_once: set[str] = set()

    async def _recording(git_ops, req, merge_wt, **kwargs):
        calls.append({'task_id': req.task_id, 'merge_wt': merge_wt, **kwargs})
        if park is not None and req.task_id in park and req.task_id not in parked_once:
            parked_once.add(req.task_id)
            if parked is not None:
                parked.set()
            # Never completes on its own: δ's teardown is what ends it.
            await asyncio.Event().wait()
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
    The ChainResult "Lane ownership" contract is EXACTLY-once.
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
    whole-registry surfaces of :func:`_assert_two_way_quiescent` (every future
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


def _finalized_rows(db_path: Path) -> list[dict]:
    """Return every ``merge_finalized`` row's parsed ``data`` dict, in order.

    Reads the durable tier through real sqlite rather than a capturing fake,
    because a field only reaches η1 if it survives ``json.dumps`` into the
    ``data`` column — a fake that records the dict by reference would pass even
    for a value the real emit path drops or cannot serialise.
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
    following γ's ``_canary_says_deep`` precedent.  This is what PINS
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
