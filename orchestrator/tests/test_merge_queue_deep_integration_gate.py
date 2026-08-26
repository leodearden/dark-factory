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
from shared.task_metadata import RetryLedger

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


# ── the TWO-WAY quiescence oracle (step-2) ───────────────────────────────────


class _LadderClaim(TypedDict):
    """The ``ladder=`` argument's shape: a ledger PAIR, before and after.

    ``before`` is what was handed to workflow.py's ladder; ``after`` is what
    the ladder returned once the round's REAL observable outputs were folded
    through it (see ``_ladder_after``).  Row 8's claim is that a deep round
    moves neither field, so the oracle compares the two rather than comparing
    ``after`` against a hard-coded zero — a ladder that started mid-run (a
    genuine sequential block earlier in the same scene) still has an unmoved
    claim to make, and a literal would silently stop being one.
    """

    before: RetryLedger
    after: RetryLedger


def _assert_two_way_quiescent(
    worker: SpeculativeMergeWorker,
    main_sha: str,
    requests: list[MergeRequest],
    *,
    permits_before: _PermitCensus | None = None,
    ladder: _LadderClaim | None = None,
) -> None:
    """Assert the TWO-WAY quiescence contract holds for *worker*.

    The gate's single oracle, called after EVERY round of every multi-round
    scenario below.  It composes the six WORKER-side surfaces that
    test_merge_queue_deep_landing.py::_assert_quiescent already checks (itself
    a clone of test_merge_queue_invariant_integration_gate.py::_assert_quiescent
    — this is the third clone; see the module PROVENANCE table) with the
    CAS/LEDGER side, which is original here.

    Each surface is listed with the way it can go SILENT, because a surface
    that fails open is worth nothing in an oracle called eleven times.

    GUARD CLAUSES (both self-tested in :class:`TestTwoWayOracleContract`)
      * ``worker._running`` must be True.  ``speculation_accounting_violations``
        and ``worktree_ledger_violations`` BOTH ``return []`` immediately on a
        stopped worker — ``stop()`` over-releases both semaphores by depth+1 as
        a shutdown safety valve, which would otherwise read as a spurious
        violation — so an oracle that accepted a stopped worker would report
        green over a genuinely leaking one.
      * *main_sha* must be a REAL sha, never falsy and never the literal
        ``'unknown'``.  ``two_layer_invariants`` gates its base-chain
        (``check_frozen_prefix_invariant``) and verify-base sub-checks on
        exactly that condition, leaving only the two graph-consistency checks
        running for the sentinel.

    WORKER SIDE
      (a) every request in *requests* has resolved (done or cancelled) — no
          dangling in-flight work left over from the round.  Goes silent only
          if *requests* is empty, which is why every caller passes the scene's
          whole request registry rather than the round's own items.
      (b) ``speculation_accounting_violations() == []`` — I4 permit/cap
          conservation.  Silent when not ``_running`` (guarded above).
      (c) ``worktree_ledger_violations() == []`` — I6, the on-disk
          ``_merge-*`` worktree ledger.  Silent when not ``_running`` (guarded
          above), and additionally exempts trees younger than
          ``RESOURCE_AUDIT_WORKTREE_GRACE_SECS`` by design.
      (d) the request-liveness ledger is empty AFTER sweeping resolved
          entries.  Resolution is detected PASSIVELY — RequestLedger has no
          on-resolve hook, so a resolved request stays armed until
          ``sweep_resolved()`` runs; calling it here before ``is_empty()`` is
          required, not optional.
      (e) ``two_layer_invariants(main_sha) == []`` — §5.3.  Partially silent
          for the ``'unknown'`` sentinel (guarded above).
      (f) ``set(worker._lifecycle.non_terminal_items()) == set()`` — the
          ItemLifecycle registry has retired every request_id.  Placed after
          the (d) sweep so it samples a truly-drained pipeline.

    CAS/LEDGER SIDE
      (g) *permits_before* (optional): both ledgers' ``live`` views are
          compared as frozensets of TOKENS, not as counts.  ``SpecPermit`` and
          ``CapPermit`` are ``eq=False`` dataclasses, so identity IS the
          comparison: a round that released the dispatching head's token early
          and acquired a replacement keeps every count plausible and breaks
          only ownership.  Both endpoints are sampled AT REST, so the head's
          own permit — taken and returned inside the round — is absent from
          each; a head permit that SURVIVED the round shows up here as an
          extra token.  The structural identity
          ``slot_available + len(live) == depth`` is checked directly for both
          ledgers as well, so this surface stays meaningful even if the
          ``_running``-gated wrapper in (b) is ever loosened.
      (h) *ladder* (optional): the workflow.py merge-thrash ladder must be
          UNMOVED — ``consecutive_merge_thrash`` and
          ``last_merge_outcome_signature`` byte-equal between the ledger handed
          in and the ledger that came back out.  This is row 8's other half:
          the merge-queue side proves event silence, and this proves the
          silence reaches the consumer that 3003's signature class lives in.
    """
    # ── guard clauses: refuse the two inputs that would pass vacuously ───────
    assert worker._running is True, (
        f'_assert_two_way_quiescent requires a worker with _running=True, got '
        f'{worker._running!r}: speculation_accounting_violations() and '
        f'worktree_ledger_violations() BOTH short-circuit to [] on a stopped '
        f'worker, so surfaces (b) and (c) would pass vacuously'
    )
    assert main_sha and main_sha != 'unknown', (
        f'_assert_two_way_quiescent requires a REAL main_sha, got '
        f'{main_sha!r}: two_layer_invariants silently skips its base-chain and '
        f'verify-base sub-checks for a falsy or \'unknown\' sha, so surface '
        f'(e) would pass vacuously'
    )

    # ── (a) every tracked request resolved ──────────────────────────────────
    for req in requests:
        assert req.result.done() or req.result.cancelled(), (
            f'Expected request {req.request_id!r} (task {req.task_id!r}) to '
            f'have resolved (done or cancelled) at quiescence, but it is '
            f'still pending'
        )

    # ── (b) I4 permit/cap conservation ──────────────────────────────────────
    spec_violations = worker.speculation_accounting_violations()
    assert spec_violations == [], (
        f'speculation_accounting_violations() non-empty at quiescence: '
        f'{spec_violations!r}'
    )

    # ── (c) I6 on-disk worktree ledger ──────────────────────────────────────
    wt_violations = worker.worktree_ledger_violations()
    assert wt_violations == [], (
        f'worktree_ledger_violations() non-empty at quiescence: {wt_violations!r}'
    )

    # ── (d) request-liveness ledger, swept first ────────────────────────────
    worker._request_ledger.sweep_resolved()
    assert worker._request_ledger.is_empty(), (
        f'request-liveness ledger non-empty at quiescence: '
        f'{worker._request_ledger.open_request_ids()!r}'
    )

    # ── (e) §5.3 two-layer invariants ───────────────────────────────────────
    tli_violations = worker.two_layer_invariants(main_sha)
    assert tli_violations == [], (
        f'two_layer_invariants({main_sha!r}) non-empty at quiescence: '
        f'{tli_violations!r}'
    )

    # ── (f) ItemLifecycle registry fully retired ────────────────────────────
    registry_ids = set(worker._lifecycle.non_terminal_items())
    assert registry_ids == set(), (
        f'ItemLifecycle registry non-terminal at quiescence: {registry_ids!r}'
    )

    # ── (g) token-level permit census, both ledgers ─────────────────────────
    if permits_before is not None:
        after = _permit_census(worker)
        assert after['spec_live'] == permits_before['spec_live'], (
            f'spec_live moved across the run: '
            f'gained {set(after["spec_live"]) - set(permits_before["spec_live"])!r}, '
            f'lost {set(permits_before["spec_live"]) - set(after["spec_live"])!r}'
        )
        assert after['cap_live'] == permits_before['cap_live'], (
            f'cap_live moved across the run: '
            f'gained {set(after["cap_live"]) - set(permits_before["cap_live"])!r}, '
            f'lost {set(permits_before["cap_live"]) - set(after["cap_live"])!r}'
        )
        assert after['spec_available'] + len(after['spec_live']) == after['spec_depth'], (
            f'speculation identity broken at quiescence: '
            f'{after["spec_available"]!r} + {len(after["spec_live"])!r} != '
            f'{after["spec_depth"]!r}'
        )
        assert after['cap_available'] + len(after['cap_live']) == after['cap_depth'], (
            f'merge-ahead-cap identity broken at quiescence: '
            f'{after["cap_available"]!r} + {len(after["cap_live"])!r} != '
            f'{after["cap_depth"]!r}'
        )

    # ── (h) the workflow.py merge-thrash ladder, unmoved ────────────────────
    if ladder is not None:
        before_l, after_l = ladder['before'], ladder['after']
        assert (
            after_l.consecutive_merge_thrash == before_l.consecutive_merge_thrash
        ), (
            f'consecutive_merge_thrash moved across the run: '
            f'{before_l.consecutive_merge_thrash!r} -> '
            f'{after_l.consecutive_merge_thrash!r}'
        )
        assert (
            after_l.last_merge_outcome_signature
            == before_l.last_merge_outcome_signature
        ), (
            f'last_merge_outcome_signature moved across the run: '
            f'{before_l.last_merge_outcome_signature!r} -> '
            f'{after_l.last_merge_outcome_signature!r}'
        )


# ═══════════════════════════════════════════════════════════════════════════
# -- step-01 RED: the TWO-WAY oracle's own contract, self-tested for vacuity --
#
# Every row below leans on ONE oracle, so the oracle is the gate's single point
# of failure: an oracle that passes vacuously turns eleven rows green without
# testing anything.  Three of its surfaces are DOCUMENTED to go silent rather
# than to fail —
#
#   * `speculation_accounting_violations()` and `worktree_ledger_violations()`
#     both `return []` immediately when `not self._running` (stop() over-
#     releases both semaphores by depth+1 as a shutdown safety valve, which
#     would otherwise read as a spurious violation);
#   * `two_layer_invariants(main_sha)` SKIPS its base-chain and verify-base
#     sub-checks for a falsy or 'unknown' main_sha (the snapshot() sentinel).
#
# — so the oracle must REJECT those inputs rather than consume them.  This
# class pins all three as POSITIVE CONTROLS, and pins a forced permit leak as
# the proof that the green it reports is meaningful.
# ═══════════════════════════════════════════════════════════════════════════


class TestTwoWayOracleContract:
    """``_assert_two_way_quiescent`` — the gate's own oracle, self-tested.

    Sync deliberately: every surface here is pure/synchronous (no await, no
    git subprocess), and pytest-asyncio STRICT makes a sync ``test_*`` inside
    an ``@pytest.mark.asyncio`` class a hard ERROR.  The one surface that
    genuinely needs a running loop — an unresolved request future — lives in
    :class:`TestTwoWayOracleContractAsync` below.
    """

    def test_a_clean_at_rest_worker_passes_on_both_halves(
        self, git_repo: Path,
    ) -> None:
        """The GREEN baseline: an oracle that always raised would prove nothing.

        Every rejection test below is only evidence if the oracle can also say
        yes — so this pins the accepting branch of all eight surfaces at once
        (six worker-side, plus the permit census and the thrash ladder).
        """
        from shared.task_metadata import RetryLedger

        worker = _make_worker(_make_git_ops(git_repo))
        main_sha = asyncio.run(_rev_parse(git_repo, 'main'))
        census = _permit_census(worker)
        ledger = RetryLedger()

        _assert_two_way_quiescent(
            worker, main_sha, [],
            permits_before=census, ladder={'before': ledger, 'after': ledger},
        )

    def test_a_stopped_worker_is_rejected_rather_than_passing_vacuously(
        self, git_repo: Path,
    ) -> None:
        """VACUITY TRAP (i): both audits short-circuit to [] when not ``_running``.

        Demonstrated, not quoted: a REAL forced leak is installed first, and
        the two audits are observed answering ``[]`` about it while the worker
        is stopped.  That is exactly the state in which an oracle that merely
        called them would report a clean green over a broken worker — so the
        oracle owes a guard clause, and this asserts it has one.
        """
        worker = _make_worker(_make_git_ops(git_repo))
        main_sha = asyncio.run(_rev_parse(git_repo, 'main'))

        # A genuine identity break: a permit vanished from the shared semaphore
        # without being recorded held-by-merger, transferred, or available.
        worker._speculation_slot._value -= 1
        worker._running = False

        spec_violations = worker.speculation_accounting_violations()
        assert spec_violations == [], (
            f'the vacuity trap this guard exists for did not reproduce: a '
            f'stopped worker with a forced speculation-slot leak should have '
            f'short-circuited to [], got {spec_violations!r}'
        )
        wt_violations = worker.worktree_ledger_violations()
        assert wt_violations == [], (
            f'worktree_ledger_violations() should short-circuit to [] on a '
            f'stopped worker, got {wt_violations!r}'
        )

        with pytest.raises(AssertionError, match='_running'):
            _assert_two_way_quiescent(worker, main_sha, [])

    def test_the_unknown_main_sha_sentinel_is_rejected(
        self, git_repo: Path,
    ) -> None:
        """VACUITY TRAP (ii): ``two_layer_invariants`` skips sub-checks for 'unknown'.

        ``snapshot()`` passes the literal ``'unknown'`` when ``get_main_sha()``
        is unavailable, and the base-chain + verify-base sub-checks are gated
        off for it (and for a falsy sha) so startup does not report spurious
        violations.  A gate that handed the oracle that sentinel would be
        asserting only the two graph-consistency checks and calling it §5.3.

        The same worker with a REAL sha is accepted in the same test, so the
        rejection is attributable to the sentinel and not to the worker.
        """
        worker = _make_worker(_make_git_ops(git_repo))
        real_sha = asyncio.run(_rev_parse(git_repo, 'main'))

        for sentinel in ('unknown', '', None):
            with pytest.raises(AssertionError, match='main_sha'):
                _assert_two_way_quiescent(worker, sentinel, [])  # type: ignore[arg-type]

        _assert_two_way_quiescent(worker, real_sha, [])

    def test_a_forced_speculation_slot_leak_is_detected(
        self, git_repo: Path,
    ) -> None:
        """POSITIVE CONTROL: the green the oracle reports is meaningful.

        The control is transcribed from
        test_merge_queue_resource_audit.py::TestSpeculationAccountingViolations::
        test_forced_speculation_slot_leak_yields_one_violation — the identity
        (a) break that ``speculation_accounting_violations`` exists to catch.
        Without this, every "audits green" assertion in the gate would be
        consistent with an audit that never fires at all.
        """
        worker = _make_worker(_make_git_ops(git_repo))
        main_sha = asyncio.run(_rev_parse(git_repo, 'main'))

        worker._speculation_slot._value -= 1
        assert worker._running is True, (
            'the leak control is only meaningful on a RUNNING worker; got '
            f'_running={worker._running!r}'
        )

        with pytest.raises(AssertionError, match='speculation_accounting_violations'):
            _assert_two_way_quiescent(worker, main_sha, [])

    def test_a_permit_token_that_did_not_survive_the_run_is_rejected(
        self, git_repo: Path,
    ) -> None:
        """CAS/LEDGER half (a): the census is compared by TOKEN, not by count.

        ``SpecPermit``/``CapPermit`` are ``eq=False`` dataclasses — IDENTITY
        comparison — so a ledger's ``live`` set treats every acquired token as
        a distinct member.  A run that released the dispatching head's token
        early and acquired a replacement would keep every COUNT plausible and
        break only ownership: invisible to a size comparison, and exactly what
        a frozenset comparison catches.

        The forged census stands in for "a token that was live before the run
        and is not live after it", which is the same set difference either way
        round.
        """
        worker = _make_worker(_make_git_ops(git_repo))
        main_sha = asyncio.run(_rev_parse(git_repo, 'main'))

        census = _permit_census(worker)
        forged: _PermitCensus = {**census, 'spec_live': frozenset({SpecPermit()})}
        with pytest.raises(AssertionError, match='spec_live'):
            _assert_two_way_quiescent(worker, main_sha, [], permits_before=forged)

        forged_cap: _PermitCensus = {**census, 'cap_live': frozenset({CapPermit()})}
        with pytest.raises(AssertionError, match='cap_live'):
            _assert_two_way_quiescent(worker, main_sha, [], permits_before=forged_cap)

    def test_a_moved_thrash_counter_is_rejected(self, git_repo: Path) -> None:
        """CAS/LEDGER half (b): row 8's claim is about workflow.py's LADDER.

        The `after` ledger is produced by the REAL
        ``workflow._evaluate_merge_thrash`` over a real ``RetryLedger`` rather
        than by hand, so a change to the ladder's counter arithmetic moves this
        control rather than leaving it stale.  ``before is after`` is the
        UNMOVED shape row 8 asserts; a single genuine blocked signature folded
        through the ladder is the MOVED shape it must be able to tell apart.
        """
        from shared.task_metadata import RetryLedger

        from orchestrator.workflow import (
            _compute_merge_outcome_signature,
            _evaluate_merge_thrash,
        )

        worker = _make_worker(_make_git_ops(git_repo))
        main_sha = asyncio.run(_rev_parse(git_repo, 'main'))

        before = RetryLedger()
        signature = _compute_merge_outcome_signature('merge_conflict', 'shared.txt', '')
        moved = _evaluate_merge_thrash(before, None, signature, 2).ledger
        assert moved.consecutive_merge_thrash == 1, (
            f'the ladder control is inert: folding one genuine blocked '
            f'signature must move the counter off 0, got '
            f'{moved.consecutive_merge_thrash!r}'
        )

        with pytest.raises(AssertionError, match='consecutive_merge_thrash'):
            _assert_two_way_quiescent(
                worker, main_sha, [], ladder={'before': before, 'after': moved},
            )

        _assert_two_way_quiescent(
            worker, main_sha, [], ladder={'before': before, 'after': before},
        )

    def test_a_moved_outcome_signature_is_rejected(self, git_repo: Path) -> None:
        """CAS/LEDGER half (b), second field: the SIGNATURE must be unmoved too.

        A deep round that requeued WITHOUT rendering a block reason leaves both
        ladder fields untouched.  Pinning the counter alone would miss a round
        that re-keyed ``last_merge_outcome_signature`` while happening to leave
        the counter where it was — which is precisely the state that makes the
        NEXT genuine block increment instead of reset.
        """
        from shared.task_metadata import RetryLedger

        from orchestrator.workflow import _compute_merge_outcome_signature

        worker = _make_worker(_make_git_ops(git_repo))
        main_sha = asyncio.run(_rev_parse(git_repo, 'main'))

        before = RetryLedger(consecutive_merge_thrash=1, last_merge_outcome_signature=None)
        rekeyed = before.model_copy(update={
            'last_merge_outcome_signature': _compute_merge_outcome_signature(
                'verify_failure', 'tip is red', '',
            ),
        })
        assert rekeyed.consecutive_merge_thrash == before.consecutive_merge_thrash, (
            'this control must move ONLY the signature; the counter moved too'
        )

        with pytest.raises(AssertionError, match='last_merge_outcome_signature'):
            _assert_two_way_quiescent(
                worker, main_sha, [], ladder={'before': before, 'after': rekeyed},
            )


@pytest.mark.asyncio
class TestTwoWayOracleContractAsync:
    """The one oracle surface that needs a running loop: request futures."""

    async def test_an_unresolved_request_future_is_rejected(
        self, git_repo: Path,
    ) -> None:
        """Surface (a): every tracked request must have resolved at quiescence.

        ``_make_req`` builds its future off ``asyncio.get_running_loop()``, so
        this leg cannot be sync.  The SAME request is then cancelled and the
        oracle accepts it — pinning that "resolved" means done-OR-cancelled,
        the shape a requeued-then-drained item actually rests in.
        """
        git_ops = _make_git_ops(git_repo)
        worker = _make_worker(git_ops)
        config = _make_config(git_repo, chain_cap=0)
        main_sha = await _rev_parse(git_repo, 'main')

        req = _make_req('101', '101', config, git_repo)
        with pytest.raises(AssertionError, match='still pending'):
            _assert_two_way_quiescent(worker, main_sha, [req])

        req.result.cancel()
        _assert_two_way_quiescent(worker, main_sha, [req])


# ═══════════════════════════════════════════════════════════════════════════
# -- step-03 RED: Row 11 (part 1) — the gate must reach PRD DEPTH 16 --
#
# Nothing in the tree builds a chain deeper than THREE links: γ's scenes cap at
# 6 but seed 5 followers, and δ's `_DeltaScene` is hard-wired to 3 links plus a
# truncator.  The PRD's row 11 is stated at "16-item chain (cap 32)", the
# observed maximum reify queue depth — so the cap=32 promotion η2 ships is
# ENTIRELY unexercised today.  Before any timeout claim can be made about a
# depth-16 chain, a scene has to be able to BUILD one.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(300)
class TestDeepScaleBuild:
    """A real 16-item chain at ``chain_cap=32`` — the PRD's stated maximum."""

    async def test_sixteen_queued_items_build_one_fifteen_link_chain(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """16 queued items at cap=32 → ONE chain of 15 links in ONE spec lane.

        Every number here is a unit statement, and the units are the whole
        hazard.  ``chain_items`` counts ITEMS IN THE TREE with the dispatching
        item as #1, while ``build_chain`` counts ADDITIONAL LINKS BEYOND the
        base it is handed — and the base IS the dispatching item's merge
        commit.  ``_deep_chain_placement`` converts between them with
        ``cap - 1`` / ``d - 1``, so a 16-item tree is a 15-link build driven by
        ``cap=31, target_depth=15``.  Pinning the converted kwargs (rather than
        only the resulting length) is what would catch a conversion that
        silently landed on 16 links / 17 items — an off-by-one the PRD's
        ``depth`` field already suffered once.

        ``truncated_at is None`` is the other unit claim: a chain that stopped
        because it reached its TARGET DEPTH is complete, not truncated;
        ``truncated_at`` is reserved for the item that did not chain.
        """
        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=32, n_followers=15, db_name='gate-scale-16.db',
            remote=True,
        )
        rec = await scene.round_(tag='scale', head_tid='101')

        chain = rec['chain']
        assert chain is not None, 'cap=32 with 16 queued items must chain'
        assert len(chain.links) == 15, (
            f'expected a 15-LINK chain (16 items incl. the head), got '
            f'{len(chain.links)}: {[t for t, _ in chain.links]!r}'
        )
        assert 1 + len(chain.links) == 16
        assert chain.truncated_at is None, (
            f'a DEPTH stop is not a truncation, but truncated_at is '
            f'{chain.truncated_at!r} (reason {chain.truncated_reason!r})'
        )

        # ── the unit conversion, pinned at the build_chain boundary ──────────
        assert len(scene.built) == 1, (
            f'exactly one build for the round, got {len(scene.built)}'
        )
        assert scene.built[0]['cap'] == 31, (
            f'cap must be converted to LINK units (32 - 1), got '
            f'{scene.built[0]["cap"]!r}'
        )
        assert scene.built[0]['target_depth'] == 15, (
            f'target_depth must be converted to LINK units (16 - 1), got '
            f'{scene.built[0]["target_depth"]!r}'
        )

        # ── the tip is a real descendant of the head's merge commit ──────────
        assert chain.tip != rec['head_mc'], (
            'a 15-link chain whose tip equals its base built nothing'
        )
        rc, _o, _e = await _run(
            ['git', 'merge-base', '--is-ancestor', rec['head_mc'], chain.tip],
            cwd=git_repo,
        )
        assert rc == 0, (
            f'the chain tip {chain.tip[:8]} must descend from the head merge '
            f'commit {rec["head_mc"][:8]}'
        )

        # ── ONE scratch lane for the whole build (PRD decision #1) ───────────
        assert chain.lane is not None and chain.lane.name.startswith('_spec-'), (
            f'the build must claim a pooled _spec- lane, got {chain.lane!r}'
        )
        assert len(scene.lane_releases) == 1, (
            f'the chain lane is released EXACTLY once per build; got '
            f'{scene.lane_releases!r}'
        )

        # ── η1's reader sees a 16-item deep verify ───────────────────────────
        verify_rows = _verify_rows(scene.db_path)
        assert len(verify_rows) == 1, (
            f'ONE merge_verify paid for the whole 16-item tree, got '
            f'{len(verify_rows)}'
        )
        assert verify_rows[0]['chain_items'] == 16, (
            f'chain_items is in CHAIN-ITEM units (head = #1), so a 15-link '
            f'chain reports 16; got {verify_rows[0]["chain_items"]!r}'
        )
