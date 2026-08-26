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
      |   (step-07/08, class       |   a POSITIONAL pass/fail script)
      |   TestRow3HalvingIsolates- |
      |   TheBadItem)              |
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
  _capture_verify_timeouts       | the ``_run_cmd`` recorder is the shape of
                                 |   test_verify.py::
                                 |   TestRunVerificationColdFirstUse::
                                 |   _make_success_mock.  The second patch it
                                 |   installs (restoring the REAL
                                 |   run_scoped_verification over conftest's
                                 |   autouse stub) is ORIGINAL here — no
                                 |   merge-queue test had needed to reach
                                 |   BELOW that stub before.
  _timed_out_verify_result       | ORIGINAL; the timed_out=True verdict shape
                                 |   is test_merge_queue.py::
                                 |   _mock_verify_timeout's, narrowed to a
                                 |   real VerifyResult rather than a mock.
  _verdict_from_tree             | ORIGINAL.  Nothing upstream keys a verdict
                                 |   on TREE CONTENT — both deep modules use a
                                 |   POSITIONAL pass/fail script, which is
                                 |   blind to which items were in the tree and
                                 |   therefore cannot state Row 3's isolation
                                 |   claim.  Installed over the same public
                                 |   `run_scoped_verification` name conftest's
                                 |   autouse stub uses.
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


def _timed_out_verify_result():
    """A :class:`VerifyResult` that TIMED OUT — a red that names no culprit.

    Distinct from :func:`_fail_verify_result` on exactly one axis, and the
    whole of Row 11 leg (c) turns on it: ``timed_out=True`` is the ONLY thing
    that advances ``_post_merge_verify_timeouts`` (a real test/lint/type red
    deliberately does not feed the loop-breaker — those bubble to the steward
    instead of oscillating).  Everything else is empty rather than
    plausible-looking: a timeout produced no verdicts, so a populated
    ``test_output`` here would be fiction, and an ENOSPC-looking string in one
    of these fields would route the result down the transient-infra branch
    instead of the timeout one.

    ``category=''`` is load-bearing for the same reason — a category in
    ``INFRA_TRANSIENT_CATEGORIES`` would make ``_run_post_merge_verify`` RETRY
    the dispatch rather than conclude, and the leg would be measuring the retry
    loop rather than the timeout path.
    """
    from orchestrator.verify import VerifyResult

    return VerifyResult(
        passed=False, test_output='', lint_output='', type_output='',
        summary='verify timed out', category='', timed_out=True,
    )


def _verdict_from_tree(bad_file: str):
    """A ``run_scoped_verification``-shaped verdict keyed on TREE CONTENT.

    Returns red iff *bad_file* is present in the worktree the verify was
    actually handed, green otherwise.  That one property is what turns Row 3
    from a policy replay into an ISOLATION claim: the same physical item is
    red at every depth and in EVERY chain that contains it, and green nowhere,
    so the bisection's shape is a consequence of the item rather than of the
    fixture.  A positional pass/fail script (the ``script=`` vocabulary, and
    what test_merge_queue_deep_dispatch.py's halving walk uses) cannot make
    that claim at all — it is blind to which items were in the tree, and its
    round-3 pass is an INPUT rather than a derived fact.

    WHICH WORKTREE, per arm — both are real, and the stub does not need to
    know which it is looking at:

      * a DEEP round hands it ``chain.lane``, the ``_spec-`` scratch lane that
        ``build_chain`` merged the whole chain into, so the file is present iff
        the culprit is somewhere in the chain (or in its BASE — a chain built
        ON the culprit is red too, which is exactly why rounds 7–8 of the walk
        stay red);
      * a FLOOR round hands it the warm-swapped ``_spec-`` lane checked out at
        the item's OWN merge commit, so the file is present iff the culprit is
        the dispatching item itself.

    SEAM.  ``orchestrator.merge_queue.run_scoped_verification`` — the same name
    conftest's autouse ``_mock_merge_queue_verification`` replaces with a
    ``passed=True`` stub, so this is a documented, public patch point and not a
    ``merge_queue.<private>`` reach-back (which
    test_merge_queue_reachback_patch_guard.py freezes).  Patching HERE rather
    than replacing ``_run_post_merge_verify`` is what keeps the row honest:
    the real ``_run_post_merge_verify`` still runs, so a red on the ordinary
    arm is rendered into a genuine blocked :class:`MergeOutcome` by production
    code, and its ``timeouts``/``enospc_retries`` bookkeeping stays in the
    loop.  A stub in its place would be asserting the test's own arithmetic —
    and, because that function returns ``MergeOutcome | None`` rather than a
    ``VerifyResult``, would take the ordinary arm down a path it cannot handle.

    FAILS LOUDLY, never green, when the worktree it is handed does not exist.
    That guard is not defensive noise: an item's ephemeral ``_merge-<uuid>``
    is a bare path in this module (see :func:`_ephemeral_merge_wt`), so a round
    that never reached the warm swap would hand over a directory that is not
    there — and a stub that answered "no bad file, therefore green" for it
    would make the whole row pass vacuously, on a tree that was never read.
    """
    from orchestrator.verify import VerifyResult

    async def _verdict(worktree, config, module_configs, task_files=None, **kwargs):
        wt = Path(worktree)
        assert wt.is_dir(), (
            f'_verdict_from_tree({bad_file!r}) was handed a worktree that does '
            f'not exist: {wt}. A verdict derived from an absent tree would be '
            f'green for the wrong reason and make the whole row vacuous — the '
            f'usual cause is a round that never reached the warm swap, leaving '
            f'the item\'s bare ephemeral _merge-<uuid> path in place.'
        )
        if (wt / bad_file).exists():
            return VerifyResult(
                passed=False, test_output=f'{bad_file} is present in this tree',
                lint_output='', type_output='', summary='fail', category='',
            )
        return VerifyResult(
            passed=True, test_output='ok', lint_output='', type_output='',
            summary='ok', category='',
        )

    _verdict.bad_file = bad_file  # type: ignore[attr-defined]
    return _verdict



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


def _capture_verify_timeouts(monkeypatch) -> list[float]:
    """Record the per-command ``timeout`` a merge verify hands down, in order.

    Row 11's budget claim has to be about the number the code THREADS, never
    about elapsed wall clock: the budget under test is 7200 seconds, and there
    is no fake-clock facility for verify in this repo (by design — the verify
    path's deadlines are real subprocess deadlines).  So this captures at the
    bottom of the stack, at the subprocess launcher, and lets everything above
    it run for real.

    TWO patches, and BOTH are required for the capture to be non-vacuous:

      * ``verify._run_cmd`` -> a recorder returning ``(0, '', False)`` (rc 0,
        no output, not timed out).  The recorder is the reason no configured
        command ever actually executes — the gate's config carries the SHIPPED
        ``test_command``/``lint_command``/``type_check_command``, which would
        otherwise launch the whole repo's suite.  Same shape as
        test_verify.py::TestRunVerificationColdFirstUse.
      * ``merge_queue.run_scoped_verification`` -> the REAL
        ``verify.run_scoped_verification``.  The conftest autouse
        ``_mock_merge_queue_verification`` replaces this with a passed=True
        stub for every test in the suite, and that stub sits ABOVE
        ``run_verification`` — i.e. above ``_resolve_verify_timeout``.  Without
        this restore the recorder is never called at all and the assertion
        passes on an empty list.  (Hence the ``assert captured`` vacuity guard
        every caller carries.)

    Returns the live list, appended to as commands dispatch.
    """
    from orchestrator import verify as _verify

    captured: list[float] = []

    async def _recording_run_cmd(
        cmd, cwd, timeout, env=None, log_path=None, **kwargs,
    ):
        captured.append(timeout)
        return 0, '', False

    monkeypatch.setattr('orchestrator.verify._run_cmd', _recording_run_cmd)
    monkeypatch.setattr(
        'orchestrator.merge_queue.run_scoped_verification',
        _verify.run_scoped_verification,
    )
    return captured


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


# ── the n-item gate scene (step-4) ───────────────────────────────────────────


def _gate_followers(n: int, *, start: int = 102) -> tuple[str, ...]:
    """*n* consecutive follower task ids, each destined for a DISJOINT file.

    Disjointness is load-bearing at depth: a 16-deep build performs 15 real
    sequential merges, and a single textual conflict anywhere in that run would
    truncate the chain and turn a DEPTH claim into a truncation claim.  One
    file per task (``f<tid>.txt``) makes every follower chainable by
    construction, so a short chain can only ever be the code's doing.
    """
    return tuple(str(start + i) for i in range(n))


class _StubRemoteAllocator:
    """Minimal HostAllocator stand-in handing out ONE remote lease at a time.

    REMOTE deliberately (cloned from test_merge_queue_deep_dispatch.py): a
    remote lease makes ``_run_post_merge_verify`` build its pool from the
    injected runner instead of a real ``LocalRunner``, so
    ``VerifyRunnerPool.dispatch`` — the single ``merge_verify`` emission site —
    genuinely runs and genuinely emits.  Stubbing ``_run_post_merge_verify``
    wholesale (what ``remote=False`` does) emits no event at all and would make
    every ``chain_items`` telemetry assertion vacuous.
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


def _verify_rows(db_path: Path) -> list[dict]:
    """Return every ``merge_verify`` row's parsed ``data`` dict, in order.

    The durable-tier twin of :func:`_finalized_rows`.  η1's predicate reads
    these rows out of reify's runs.db, so the gate reads them the same way —
    through real sqlite, after ``json.dumps`` — rather than through a capturing
    fake that would pass even for a value the real emit path cannot serialise.
    """
    import json
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT data FROM events WHERE event_type = 'merge_verify' "
            'ORDER BY rowid'
        ).fetchall()
    finally:
        conn.close()
    return [json.loads(r[0]) for r in rows]


class _GateScene:
    """One repo + worker + queue, driven round after round THROUGH finalize.

    The n-follower generalisation of
    test_merge_queue_deep_landing.py::_DeltaScene, which is hard-wired to three
    links plus one truncator and therefore cannot reach the PRD's stated depth
    of 16.  What this adds beyond a wider fixture:

      * ``n_followers`` is a parameter, so a row states its own queue depth;
      * a REMOTE mode (``remote=True``) that drives the real
        ``_run_post_merge_verify`` and therefore the real ``merge_verify``
        emission site, for the rows whose claim is about TELEMETRY or about the
        verify BUDGET rather than about landing;
      * per-round FACT RECORDING onto :attr:`rounds`, so a multi-round row
        asserts against what each round actually did instead of against
        whatever the last round left behind.

    Every spy is PASSTHROUGH (``advance_main``, ``release_chain_build_lane``,
    ``build_chain``, ``_run_post_merge_verify`` in remote mode), so on-disk and
    pool state stay REAL while the call ledgers stay observable.
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
        self.verdicts: list[dict] = []
        self.built: list[dict] = []
        self.lane_releases: list[tuple] = []
        self.advance_calls: list[tuple] = []
        self.reqs: dict[str, MergeRequest] = {}
        self.rounds: list[dict] = []
        self._round_no = 0

    @property
    def depths(self) -> list[int | None]:
        """The DISPATCHED chain depth of each round, in order.

        ``None`` for a round that built no chain at all (kill switch, floor, or
        a declined placement) — deliberately distinct from ``1``, which is a
        chain the policy sized at the floor.
        """
        return [
            None if r['chain'] is None else 1 + len(r['chain'].links)
            for r in self.rounds
        ]

    async def enqueue(self, task_ids) -> None:
        """Put *task_ids* on the queue through the REAL enqueue chokepoint.

        ``enqueue_merge_request`` is what registers ``_on_finalized``, and
        ``merge_finalized`` has no other emit site — so a scene that stuffed
        ``_lane_buffers`` directly would make every landing assertion blind to
        the payload this gate exists to read.
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
        """Drive ONE round: dispatch → verify → finalize; record its facts.

        *req* re-dispatches an EXISTING request (the one a previous round put
        back) instead of picking the next one off the lane buffers — which is
        how a scenario asserts that the very same request lands on a later
        round's own verdict.
        """
        self._round_no += 1
        worker = self.worker
        # ── slot-1 steady state, restored before EVERY dispatch ─────────────
        # `_n_failed` and `_remerge_occurred` are facts about SLOT 1 and about
        # the enclosing `_verifier_loop` ITERATION.  This scene models neither:
        # it drives `_dispatch_item` directly, one slot-2 item per round, with
        # a structurally empty `_inflight`.
        #
        # A REAL slot-2 dispatch never reads either flag.  `_dispatch_item`
        # gates the whole chain-invalidation / Mechanism-2 re-merge block on
        # `not _has_inflight_verify`, and a genuine speculative dispatch
        # happens precisely WHILE the head verify is in flight — so the block
        # is skipped by construction.  Driving dispatch with an empty
        # `_inflight` re-opens it, and two artifacts the pipeline itself never
        # produces then appear:
        #
        #   * any round that requeued (i.e. every red tip) leaves
        #     `_n_failed=True`, so the NEXT round's speculative item is
        #     re-merged into a NON-speculative one and `_deep_chain_placement`
        #     declines at its `item.speculative` guard.  The halving walk could
        #     never take a second step — round 2 of every bisection would
        #     silently be an ordinary verify.
        #   * `_remerge_occurred` is self-sustaining once set (it is assigned
        #     from the same dispatch's own `iteration_did_remerge`), so ONE
        #     re-merge would disable chaining for the rest of the scene.
        #
        # Resetting both restores exactly the state a healthy head slot leaves
        # behind, which is the premise every row in this file is written
        # against.  It is a fidelity choice about what this scene models, not a
        # workaround: nothing in the deep path reads these flags.
        worker._n_failed = False
        worker._remerge_occurred = False
        n_built_before = len(self.built)
        n_releases_before = len(self.lane_releases)
        n_advances_before = len(self.advance_calls)
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
        outcome = popped.result.result() if popped.result.done() else None
        rec.update({
            'round': self._round_no,
            'tag': tag,
            'item': item,
            'req': popped,
            'advanced': advanced,
            'main_before': main_before,
            'main_after': await _rev_parse(self.repo, 'main'),
            'head_mc': head_mc,
            'halving_state': worker._chain_halving_state,
            'outcome': outcome,
            # Per-round SLICES of the cumulative ledgers, so a multi-round row
            # reads what THIS round did rather than the running total.
            'built': self.built[n_built_before:],
            'lane_releases': self.lane_releases[n_releases_before:],
            'advance_calls': self.advance_calls[n_advances_before:],
            'landed': [
                tid for tid, r in self.reqs.items()
                if r.result.done() and not r.result.cancelled()
                and r.result.result().status == 'done'
            ],
        })
        self.rounds.append(rec)
        if entry.lease is not None:
            await worker._host_allocator.release(entry.lease)
        return rec


def _scripted_remote_runner(scene: _GateScene, script, name='gate-runner'):
    """A RemoteRunner-shaped fake whose verdicts come from *script*, in order.

    Used only in ``remote=True`` mode.  The verdict is decided HERE — below
    ``VerifyRunnerPool.dispatch`` — rather than by replacing
    ``_run_post_merge_verify``, so the whole dispatch/emit/timeout-resolution
    path above it genuinely runs and the ``merge_verify`` row is genuinely
    emitted.  A script that runs out returns PASS, matching δ's oracle.

    Three script vocabularies, the third added for Row 11 leg (c):

      * ``True``  — a green real-suite verdict;
      * ``False`` — an ordinary red (:func:`_fail_verify_result`);
      * a :class:`VerifyResult` — returned VERBATIM.  This is the injector for
        a verdict whose SHAPE is the claim rather than its polarity — a
        ``timed_out=True`` result, say — and injecting it here rather than by
        replacing ``_run_post_merge_verify`` (the ``outcome=`` vocabulary
        :func:`_spy_post_merge_verify` uses) is what leaves the real
        function's ``timeouts``/``enospc_retries`` bookkeeping in the loop.
        A stub in its place would be asserting the test's own arithmetic.
    """
    from unittest.mock import AsyncMock, MagicMock

    from orchestrator.verify import VerifyResult

    verdicts = list(script or [])

    async def _run_merge_verify(*args, **kwargs):
        scene.posted.append({'args': args, 'kwargs': kwargs})
        passed = verdicts.pop(0) if verdicts else True
        if isinstance(passed, VerifyResult):
            return passed
        if passed:
            return VerifyResult(
                passed=True, test_output='ok', lint_output='', type_output='',
                summary='ok', category='',
            )
        return _fail_verify_result()

    fake = MagicMock()
    fake.name = name
    fake.is_local = False
    fake.run_merge_verify = AsyncMock(side_effect=_run_merge_verify)
    fake.cancel_verify = AsyncMock(return_value=0)
    fake.probe_clean = AsyncMock(return_value=True)
    return fake


async def _make_gate_scene(
    repo: Path, tmp_path: Path, monkeypatch, *,
    chain_cap: int,
    n_followers: int,
    db_name: str,
    script: list[bool] | None = None,
    verdict=None,
    heads: tuple[str, ...] = ('101',),
    remote: bool = False,
    real_local: bool = False,
) -> _GateScene:
    """Build an n-follower, finalize-capable scene over a REAL git repo.

    *script* is the ordered pass/fail verdict sequence; it runs out into PASS.
    *remote* / *real_local* select WHERE the verdict is injected — three
    mutually exclusive seams, deepest last:

      * neither — replace ``orchestrator.merge_queue._run_post_merge_verify``
        outright (δ's shape).  Cheapest, and the right seam for a row whose
        claim is about LANDING or about queue state.  No ``merge_verify`` event
        is emitted on this path.
      * ``remote=True`` — install a ``_StubRemoteAllocator`` over a REMOTE
        ``HostLease`` whose runner answers the script, and let the real
        ``_run_post_merge_verify`` run.  The right seam for a row whose claim
        is about TELEMETRY (``chain_items``, ``chain_build_ms``), because that
        is produced strictly below that call.
      * ``real_local=True`` — patch NOTHING on the verify path: a LOCAL lease,
        the real ``_run_post_merge_verify``, its real ``LocalRunner``, and (via
        :func:`_capture_verify_timeouts`, which the caller installs) the real
        ``run_scoped_verification`` → ``run_verification``.  The ONLY seam that
        reaches ``verify._resolve_verify_timeout``, so the only one a BUDGET
        claim can be made at.  The caller supplies the recorder; *script* is
        ignored on this arm (the verdict is whatever the real stack produces —
        a pass, since the recorder returns rc 0).

    *verdict* is the CONTENT-KEYED alternative to the positional *script*, and
    it selects a fourth seam — the deepest one that still decides a verdict.
    It is a ``run_scoped_verification``-shaped callable (see
    :func:`_verdict_from_tree`) installed over conftest's autouse stub, with
    everything above it left real: a LOCAL lease (so the ordinary arm really
    warm-swaps into a ``_spec-`` lane and the verdict sees a real tree), the
    real ``_run_post_merge_verify``, and therefore the real rendering of a red
    into a blocked :class:`MergeOutcome`.  Mutually exclusive with *script* —
    a scene that stated its verdicts BOTH by position and by content would
    have two answers for the same round.  Every call is recorded on
    ``scene.verdicts``.

    Module-level monkeypatching is confined to the four names
    test_merge_queue_reachback_patch_guard.py sanctions (``build_chain``,
    ``_run_post_merge_verify``, ``release_chain_build_lane`` and
    ``CHAIN_BUILD_TIMEOUT_SECS``); everything else is patched on the INSTANCE.
    """
    from orchestrator.event_store import EventStore

    followers = _gate_followers(n_followers)
    git_ops = _make_git_ops(repo, size=2)
    config = _make_config(repo, chain_cap=chain_cap)
    for tid in (*heads, *followers):
        await _create_branch_editing(repo, f'task/{tid}', f'f{tid}.txt', f'edit-{tid}\n')
    db_path = tmp_path / db_name
    store = EventStore(db_path, f'run-{db_name}')
    worker = _make_worker(git_ops)
    worker._event_store = store
    scene = _GateScene(git_ops, config, worker, repo, store, db_path)

    if remote:
        from orchestrator.verify_runner import HostLease

        # Installed BEFORE first use so `_ensure_host_allocator`'s cache check
        # short-circuits on it rather than building a real one.
        worker._host_allocator = _StubRemoteAllocator(HostLease(
            name='laptop', runner=_scripted_remote_runner(scene, script),
            is_local=False,
        ))

    await scene.enqueue((*heads, *followers))

    # The round recorder, installed ONCE — re-wrapping per round would capture
    # the previous round's recorder as `real` and nest a wrapper deeper each
    # round.
    real_verify = worker._run_inflight_verify

    async def _recording(_item, _lease, **kwargs):
        rec = dict(kwargs)
        scene.calls.append(rec)
        rec['result'] = await real_verify(_item, _lease, **kwargs)
        return rec['result']

    monkeypatch.setattr(worker, '_run_inflight_verify', _recording)
    scene.lane_releases = _spy_chain_lane_release(monkeypatch)
    scene.advance_calls = _spy_advance_main(git_ops, monkeypatch)

    real_build = merge_queue.build_chain

    async def _recording_build(git_ops_, queue_snapshot, head_merge_commit, **kw):
        record = {
            'queue_snapshot': tuple(queue_snapshot),
            'head_merge_commit': head_merge_commit, **kw,
        }
        scene.built.append(record)
        record['result'] = await real_build(
            git_ops_, queue_snapshot, head_merge_commit, **kw,
        )
        return record['result']

    monkeypatch.setattr(merge_queue, 'build_chain', _recording_build)

    if verdict is not None:
        assert not script, (
            'script= and verdict= are mutually exclusive: a scene cannot state '
            'its verdicts both by position and by tree content'
        )

        async def _recording_verdict(worktree, *args, **kwargs):
            result = await verdict(worktree, *args, **kwargs)
            scene.verdicts.append({
                'worktree': Path(worktree), 'passed': result.passed,
            })
            return result

        # The same public name conftest's autouse `_mock_merge_queue_verification`
        # patches, so this is a documented seam rather than a
        # `merge_queue.<private>` reach-back.  Installed AFTER it, so it wins.
        monkeypatch.setattr(
            'orchestrator.merge_queue.run_scoped_verification', _recording_verdict,
        )
    elif not remote and not real_local:
        verdicts = list(script or [])

        async def _oracle(_git_ops, _req, merge_wt, **kwargs):
            scene.posted.append({'merge_wt': merge_wt, **kwargs})
            passed = verdicts.pop(0) if verdicts else True
            return None if passed else _fail_verify_result()

        monkeypatch.setattr(
            'orchestrator.merge_queue._run_post_merge_verify', _oracle,
        )
    return scene


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


# ═══════════════════════════════════════════════════════════════════════════
# -- step-05 RED: Row 11 — TIMEOUT MARGIN --
#
# The ONE boundary row with zero coverage anywhere in the tree: a repo-wide
# search for the merge-verify cold budget, for a 16-item chain, or for a
# verify timeout on a deep round returns nothing in either upstream deep
# module.  The only build-side deadline that IS tested is the 120 s
# CHAIN_BUILD_TIMEOUT_SECS, and only as a unit.
#
# The row's claim, stated as three separate ones so a failure names WHICH:
#
#   (a) BUDGET  — a deep tip verify is priced by ``is_merge_verify``, NOT by
#       how many items are in the tree.  A 16-item tip and a 1-item always-on
#       verify are handed the SAME merge-verify cold budget; deep chaining
#       does not silently reprice the verify in either direction.
#   (b) MARGIN  — the build that produces that tree finishes inside its own
#       (much smaller) deadline with orders of magnitude to spare, and stamps
#       what it cost where η1's dispatch-stall reader can find it.
#   (c) CLEAN TIMEOUT — when a deep tip verify DOES time out, it degrades
#       through the EXISTING timeout path and no new one: the loop-breaker
#       counter advances, nothing lands, no blocked outcome is rendered for
#       any chained item, and the bisector halves rather than resets.
#
# NONE of the three may elapse real wall clock.  (a) asserts the BUDGET the
# code hands down — captured at the ``_run_cmd`` seam, the shape
# test_verify.py::TestRunVerificationColdFirstUse uses — and (c) INJECTS a
# timed-out verdict rather than waiting for one.  A test that actually waited
# out a 7200 s budget could never run in CI, and one that waited out a short
# substitute budget would be pinning the substitute.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(300)
class TestRow11TimeoutMargin:
    """Row 11: a 16-item chain (cap 32) fits inside the merge-verify budget."""

    async def test_a_depth_sixteen_tip_verify_is_handed_the_merge_cold_budget(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a) Every command in a 16-item tip verify gets the 7200 s cold budget.

        The resolution being pinned is
        ``verify._resolve_verify_timeout``'s cascade **step 0**: with
        ``is_merge_verify=True`` (which ``run_scoped_verification`` also forces
        ``is_cold=True`` for, merge worktrees having no warm cargo cache and no
        ``.task/`` marker to detect one by) the merge-specific knob wins BEFORE
        the general cold knob.  So the negative control matters as much as the
        positive one: 5400.0 — the general cold budget, which is what a
        merge-verify would silently fall back to if the merge knob were ever
        dropped from the threading — must never appear.

        This runs the REAL ``_run_post_merge_verify`` → ``LocalRunner`` →
        ``run_scoped_verification`` → ``run_verification`` stack, with only the
        subprocess launcher stubbed.  Nothing else reaches the resolver: the
        conftest autouse ``_mock_merge_queue_verification`` short-circuits at
        ``run_scoped_verification``, which is above it.
        """
        captured = _capture_verify_timeouts(monkeypatch)
        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=32, n_followers=15, db_name='gate-row11-budget.db',
            real_local=True,
        )
        rec = await scene.round_(tag='budget16', head_tid='101')

        chain = rec['chain']
        assert chain is not None and len(chain.links) == 15, (
            'the budget claim is about a DEPTH-16 tip; got '
            f'{None if chain is None else 1 + len(chain.links)} items'
        )
        # The two knobs this test's whole claim is a routing statement about.
        config = scene.config
        assert config.merge_verify_cold_command_timeout_secs == 7200.0, (
            'the shipped merge-verify cold budget moved: expected 7200.0, got '
            f'{config.merge_verify_cold_command_timeout_secs!r}'
        )
        assert config.verify_cold_command_timeout_secs == 5400.0, (
            'the shipped GENERAL cold budget moved, so the negative control '
            f'below is no longer the right one: got '
            f'{config.verify_cold_command_timeout_secs!r}'
        )

        assert captured, (
            'no verify command was dispatched at all — the capture is vacuous '
            '(the autouse run_scoped_verification stub was probably still in '
            'place, short-circuiting above the resolver)'
        )
        assert set(captured) == {7200.0}, (
            f'every command in a merge verify is handed the merge-verify cold '
            f'budget; got {sorted(set(captured))!r} across {len(captured)} '
            f'commands'
        )
        assert 5400.0 not in captured, (
            f'the GENERAL cold budget leaked into a merge verify: {captured!r}'
        )

    async def test_the_same_budget_is_handed_to_a_one_item_verify(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(a, control) chain_items=1 is priced identically to chain_items=16.

        The hazard this rules out is a repricing that only LOOKS right at one
        end of the range: a deep tip is a bigger tree and takes longer, so a
        future change that scaled the budget with ``chain_items`` (or, worse,
        one that made the deep arm miss the ``is_merge_verify`` threading and
        fall back to the general cold budget) would be invisible to a test that
        only ever measured one depth.

        Same scene, same config, one queued item — ``select_chain_depth``
        declines at ``queue_len < 2``, so this is the ALWAYS-ON arm.
        """
        captured = _capture_verify_timeouts(monkeypatch)
        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=32, n_followers=0, db_name='gate-row11-budget1.db',
            real_local=True,
        )
        rec = await scene.round_(tag='budget1', head_tid='101')

        assert rec['chain'] is None, (
            'a single queued item must not chain (queue_len < 2), but a chain '
            f'of {1 + len(rec["chain"].links)} items was built'
        )
        assert captured, 'no verify command was dispatched at all'
        assert set(captured) == {7200.0}, (
            f'the always-on arm must be priced the same as the deep arm; got '
            f'{sorted(set(captured))!r}'
        )

    async def test_the_sixteen_item_build_finishes_far_inside_its_deadline(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(b) A real 15-link build costs milliseconds against a 120 s deadline.

        ``CHAIN_BUILD_TIMEOUT_SECS`` bounds the WHOLE build, and the build is a
        synchronous stall on the dispatch path — no other item can dispatch
        while it runs — so the margin is the claim, not the wall clock.  The
        assertion is deliberately an ORDER-OF-MAGNITUDE one (30 s against a
        120 s deadline, for a build the replay study measured at ~1–2 s per
        6-chain): a tight bound here would be a flake generator on a loaded
        box, while a loose one still catches the regression that matters — a
        build that went superlinear in depth and started eating the deadline.

        ``build_ms`` is stamped by the builder itself, so this is also the
        assertion that the stamp EXISTS on a real deep build rather than being
        dropped somewhere between the build and the ChainResult.
        """
        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=32, n_followers=15, db_name='gate-row11-margin.db',
            remote=True,
        )
        rec = await scene.round_(tag='margin', head_tid='101')

        chain = rec['chain']
        assert chain is not None and len(chain.links) == 15, (
            f'expected a 15-link build, got '
            f'{None if chain is None else len(chain.links)}'
        )
        assert merge_queue.CHAIN_BUILD_TIMEOUT_SECS == 120.0, (
            f'the build deadline moved: {merge_queue.CHAIN_BUILD_TIMEOUT_SECS!r}'
        )
        assert isinstance(chain.build_ms, int) and chain.build_ms > 0, (
            f'a real build must stamp a positive integer build_ms, got '
            f'{chain.build_ms!r}'
        )
        assert chain.build_ms < 30_000, (
            f'a 15-link build took {chain.build_ms} ms against a '
            f'{merge_queue.CHAIN_BUILD_TIMEOUT_SECS} s deadline — the margin '
            f'this row asserts is gone'
        )

    async def test_chain_build_ms_reaches_the_reader_exactly_when_deep(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(b) The stall stamp is on the deep row and ABSENT on the shallow one.

        η1 reads ``chain_build_ms`` out of the durable ``merge_verify`` row
        alongside drain-time, so the presence rule is part of the contract and
        not an implementation detail: non-None means "this verify paid for a
        build", and ``None`` — not ``0`` — means it paid for none.  A ``0``
        would be a lie rather than an absence, and would land in the reader's
        histogram as a free build.

        Two scenes, because the two arms cannot coexist in one round.
        """
        deep = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=32, n_followers=15, db_name='gate-row11-stampd.db',
            remote=True,
        )
        await deep.round_(tag='stamp16', head_tid='101')
        deep_rows = _verify_rows(deep.db_path)
        assert len(deep_rows) == 1, f'one row per verify, got {len(deep_rows)}'
        assert deep_rows[0]['chain_items'] == 16, (
            f'expected a 16-item row, got {deep_rows[0]["chain_items"]!r}'
        )
        assert deep_rows[0]['chain_build_ms'] is not None, (
            'a deep verify paid for a build and must stamp what it cost; row '
            f'is {deep_rows[0]!r}'
        )
        assert deep_rows[0]['chain_build_ms'] == deep.rounds[0]['chain'].build_ms, (
            f'the emitted stall stamp must be the ChainResult\'s own, not a '
            f're-derivation: row {deep_rows[0]["chain_build_ms"]!r} vs chain '
            f'{deep.rounds[0]["chain"].build_ms!r}'
        )

    async def test_a_one_item_verify_stamps_no_build_cost(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(b, control) The always-on arm reports chain_items=1, build_ms None."""
        shallow = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=32, n_followers=0, db_name='gate-row11-stamps.db',
            remote=True,
        )
        rec = await shallow.round_(tag='stamp1', head_tid='101')
        assert rec['chain'] is None, 'a single queued item must not chain'

        rows = _verify_rows(shallow.db_path)
        assert len(rows) == 1, f'one row per verify, got {len(rows)}'
        assert rows[0]['chain_items'] == 1, (
            f'the always-on arm verifies exactly one item, got '
            f'{rows[0]["chain_items"]!r}'
        )
        assert rows[0]['chain_build_ms'] is None, (
            f'a verify that paid for no build must stamp ABSENCE, not 0; got '
            f'{rows[0]["chain_build_ms"]!r}'
        )

    async def test_a_clean_verify_timeout_degrades_through_the_existing_path(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """(c) A timed-out deep tip takes the ORDINARY timeout path, not a new one.

        Four things must all hold at once, and each is a different mechanism —
        which is why they are asserted together rather than in four scenes:

        1. ``_post_merge_verify_timeouts`` bumps by exactly one.  That counter
           is the loop-breaker's, and it advances ONLY on ``verify.timed_out``
           (a real red must not feed it), so a deep timeout being invisible to
           it would let a deterministically-hanging task re-queue forever.
        2. NOTHING lands.  Main is byte-identical, and the durable ledger has
           no ``merge_finalized`` row at all — the chain's whole prefix stays
           on the queue.
        3. NO blocked ``MergeOutcome`` is rendered for any chained item.  The
           blocked outcome ``_run_post_merge_verify`` builds internally is
           SWALLOWED into a REQUEUED status by the chain arm, precisely because
           a red tip names no culprit; every request future stays pending.
           This is also what keeps a timeout out of workflow.py's thrash ladder
           (Row 8's claim, which this row must not quietly violate).
        4. The bisector HALVES rather than resetting: ``next_halving_state``
           folds a non-pass at the BUILT depth, so a 16-item tree leaves the
           ceiling at 8 — not ``None`` (which would be a reset, i.e. a timeout
           being read as a clean bill of health) and not the floor.

        The verdict is INJECTED below ``_run_post_merge_verify`` so the real
        function does its own ``timeouts``/``enospc`` bookkeeping — a stub in
        its place would be asserting the test's own arithmetic.
        """
        from orchestrator.merge_queue import next_halving_state

        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=32, n_followers=15, db_name='gate-row11-timeout.db',
            remote=True, script=[_timed_out_verify_result()],
        )
        rec = await scene.round_(tag='timeout16', head_tid='101')
        worker = scene.worker

        chain = rec['chain']
        assert chain is not None and len(chain.links) == 15, (
            'the timeout claim is about a DEPTH-16 tip; got '
            f'{None if chain is None else 1 + len(chain.links)} items'
        )

        # 1 — the loop-breaker counter saw it.
        assert dict(worker._post_merge_verify_timeouts) == {'101': 1}, (
            f'a timed-out deep tip must bump the dispatching task\'s timeout '
            f'counter exactly once; got '
            f'{dict(worker._post_merge_verify_timeouts)!r}'
        )

        # 2 — nothing landed, on either tier.
        assert rec['main_after'] == rec['main_before'], (
            f'main moved on a timed-out round: {rec["main_before"][:8]} -> '
            f'{rec["main_after"][:8]}'
        )
        assert rec['advanced'] is False, (
            f'_finalize_inflight must not advance on a timeout, got '
            f'{rec["advanced"]!r}'
        )
        assert _finalized_rows(scene.db_path) == [], (
            f'a timed-out round lands nothing, so it emits no merge_finalized; '
            f'got {_finalized_rows(scene.db_path)!r}'
        )
        assert rec['landed'] == [], (
            f'no request may resolve as done on a timeout; got {rec["landed"]!r}'
        )

        # 3 — no blocked outcome reached ANY chained item.
        assert rec['outcome'] is None, (
            f'the dispatching request must stay unresolved (REQUEUED), not '
            f'carry a blocked outcome; got {rec["outcome"]!r}'
        )
        unresolved = [
            tid for tid, req in scene.reqs.items() if not req.result.done()
        ]
        assert sorted(unresolved, key=int) == sorted(scene.reqs, key=int), (
            f'every chained request stays pending on a red tip; these '
            f'resolved: {sorted(set(scene.reqs) - set(unresolved), key=int)!r}'
        )

        # 4 — the bisector halved off the BUILT depth.
        assert worker._chain_halving_state == next_halving_state(False, 16), (
            f'a timed-out 16-item tip must halve the ceiling to '
            f'{next_halving_state(False, 16)}, got '
            f'{worker._chain_halving_state!r}'
        )
        assert worker._chain_halving_state is not None, (
            'a timeout is not a pass — resetting the bisector would read a '
            'hung verify as a clean bill of health'
        )


# ═══════════════════════════════════════════════════════════════════════════
# -- step-07 RED: Row 3 — THE HALVING WALK ISOLATES THE BAD ITEM --
#
# The row upstream covers only HALFWAY.  test_merge_queue_deep_dispatch.py's
# ``test_dispatch_depths_follow_the_halving_walk`` drives the depths
# ``[6, 3, None, 6]`` off a POSITIONAL script (``[False, False, True, True]``):
# round 1 is red because it is the first element of a list, not because
# anything in the tree it verified is wrong.  That proves the POLICY — halve
# on a fail, reset on a pass — and nothing at all about ISOLATION, which is
# the claim the PRD row actually makes ("item 3 of 6 genuinely red").
#
# What a positional script structurally cannot show:
#
#   * that the SAME physical item is red at every depth and in every chain
#     that contains it (a script is blind to which items were in the tree);
#   * that its innocent successors are green in the very same rounds;
#   * that the bisection TERMINATES on the culprit — items 1 and 2 landing at
#     the floor, item 3 blocking on its own subset verify, and deep resuming
#     once it is gone.  Under a script, "round 3 passes" is an input, so the
#     walk's shape is assumed rather than derived.
#
# So the verdict here is CONTENT-KEYED: red iff the tree that actually ran
# contains item 3's file.  Every number below is then a CONSEQUENCE of that
# one fact plus the shipped policy, and the row's assertions cross-check each
# dispatched depth against ``select_chain_depth`` evaluated on the live queue
# and each halving step against ``next_halving_state`` — so a walk that
# happened to produce the right list for the wrong reason still fails.
#
# THE MEASURED WALK (6 items, item 3 = task 103 genuinely red, cap=6).  The
# head is re-dispatched until it resolves, which is what the real pipeline
# does with a requeued head:
#
#   rnd | head | dispatched | halving after | landed        | why
#   ----+------+------------+---------------+---------------+------------------
#    1  | 101  | 6 items    | 3             | —             | chain holds 103
#    2  | 101  | 3 items    | 1             | —             | chain holds 103
#    3  | 101  | floor      | None (reset)  | 101           | own tree is clean
#    4  | 102  | 5 items    | 2             | —             | chain holds 103
#    5  | 102  | 2 items    | 1             | —             | chain holds 103
#    6  | 102  | floor      | None (reset)  | 102           | own tree is clean
#    7  | 103  | 4 items    | 2             | —             | BASE holds 103
#    8  | 103  | 2 items    | 1             | —             | BASE holds 103
#    9  | 103  | floor      | 1 (unmoved)   | — (BLOCKED)   | own tree is red
#   10  | 104  | floor      | None (reset)  | 104           | state 1 → floor
#   11  | 105  | 2 items    | None          | 105 + 106     | deep RESUMED
#
# Two things in that table are worth stating out loud because a reader's naive
# model gets them wrong:
#
#   (i)  the floor round RESETS the bisector rather than leaving it pinned at
#        1.  A passing slot-2 verify folds ``next_halving_state(True, 1)``
#        (``_run_inflight_verify``'s floor arm) — PRD decision 5's "ANY pass
#        resets".  Without it the walk is a one-way ratchet and two red rounds
#        would disable deep merge-ahead for the life of the process.  The
#        visible cost is rounds 4 and 7: the bisector re-probes deep once per
#        landed item while the bad item is still queued.  That is the design,
#        not a defect — it is the only way the walk climbs back out.
#   (ii) round 9 leaves the bisector UNMOVED at 1.  The floor arm is
#        PASS-ONLY, deliberately: folding a non-chain FAIL there would halve
#        off ordinary red branches and pin the ceiling at the floor without a
#        single deep chain having failed.
#
# RED for the absence of ``_verdict_from_tree`` (step-08).
# ═══════════════════════════════════════════════════════════════════════════

_ROW3_BAD_TASK = '103'
_ROW3_BAD_FILE = f'f{_ROW3_BAD_TASK}.txt'
_ROW3_CAP = 6


@pytest.mark.asyncio
@pytest.mark.timeout(300)
class TestRow3HalvingIsolatesTheBadItem:
    """Row 3: one genuinely-red item, and a bisection that terminates on it."""

    def _assert_conserved(
        self, worker: SpeculativeMergeWorker, main_sha: str,
        permits_before: _PermitCensus, *, where: str,
    ) -> None:
        """The MID-RUN half of the two-way contract, asserted after every round.

        :func:`_assert_two_way_quiescent` cannot be called between rounds of a
        walk like this one, and the reason is a property of the walk rather
        than a gap in the oracle: its surfaces (a), (d) and (f) are
        WHOLE-REGISTRY claims — every request resolved, the request-liveness
        ledger empty, no non-terminal lifecycle entry — and a walk that is
        deliberately mid-flight has four items still queued with pending
        futures at every one of these checkpoints.  Asserting them here would
        either fail honestly or force the caller to pass an empty request list,
        which is the vacuum the oracle's own guard clauses exist to refuse.

        What DOES hold after every single round is conservation, so that is
        what this checks — the three fail-safe audits plus the token-level
        permit census.  The full oracle runs once at the end, after
        :func:`_drain_residue` reports the run left no residue at all.
        """
        spec_violations = worker.speculation_accounting_violations()
        assert spec_violations == [], (
            f'speculation_accounting_violations() non-empty {where}: '
            f'{spec_violations!r}'
        )
        wt_violations = worker.worktree_ledger_violations()
        assert wt_violations == [], (
            f'worktree_ledger_violations() non-empty {where}: {wt_violations!r}'
        )
        tli = worker.two_layer_invariants(main_sha)
        assert tli == [], (
            f'two_layer_invariants({main_sha[:8]!r}) non-empty {where}: {tli!r}'
        )
        now = _permit_census(worker)
        assert now['spec_live'] == permits_before['spec_live'], (
            f'spec_live moved {where}: gained '
            f'{set(now["spec_live"]) - set(permits_before["spec_live"])!r}, lost '
            f'{set(permits_before["spec_live"]) - set(now["spec_live"])!r}'
        )
        assert now['cap_live'] == permits_before['cap_live'], (
            f'cap_live moved {where}: gained '
            f'{set(now["cap_live"]) - set(permits_before["cap_live"])!r}, lost '
            f'{set(permits_before["cap_live"]) - set(now["cap_live"])!r}'
        )

    async def _walk(
        self, git_repo: Path, tmp_path: Path, monkeypatch, *,
        db_name: str, bad_file: str = _ROW3_BAD_FILE, max_rounds: int = 20,
        after_round=None,
    ) -> tuple[_GateScene, list[int], _PermitCensus]:
        """Drive the queue to rest, re-dispatching the head until it resolves.

        Returns the scene, the per-round 1-indexed queue length observed AT
        DISPATCH (the input ``select_chain_depth`` was actually evaluated on),
        and the at-rest permit census taken before the first round.

        *after_round*, when given, is called with each round's record the
        moment it completes — the only way to sample a state that is true
        DURING the bisection and no longer true once the queue has drained.

        The schedule is the pipeline's, not the test's convenience: a red tip
        REQUEUES its dispatching head, and the real merger picks that same
        request again on the next pass.  Popping the NEXT item instead would
        walk a different queue and quietly make the bisection look like it
        terminated when it had merely moved on.
        """
        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=_ROW3_CAP, n_followers=5, db_name=db_name,
            verdict=_verdict_from_tree(bad_file),
        )
        worker = scene.worker
        permits_before = _permit_census(worker)
        queue_lens: list[int] = []
        pending: MergeRequest | None = None
        for _ in range(max_rounds):
            if pending is None:
                nxt = worker._pop_next_pickable()
                if nxt is None:
                    break
            else:
                nxt = pending
            # Sampled AFTER the pop, so it is exactly the 1-indexed count
            # `_deep_chain_placement` computes (`1 + len(chain_snapshot())`).
            queue_lens.append(1 + len(worker.chain_snapshot()))
            rec = await scene.round_(tag='row3', head_tid=nxt.task_id, req=nxt)
            pending = None if nxt.result.done() else nxt
            self._assert_conserved(
                worker, rec['main_after'], permits_before,
                where=f'after round {rec["round"]} (head {nxt.task_id})',
            )
            if after_round is not None:
                after_round(scene, rec)
        else:  # pragma: no cover - a walk that never drains is the bug
            pytest.fail(
                f'the queue never drained in {max_rounds} rounds; depths so far '
                f'were {scene.depths!r}'
            )
        return scene, queue_lens, permits_before

    async def test_the_bisection_walks_down_and_terminates_on_the_red_item(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """The whole row, in one continuous run against one red ITEM.

        Asserted as four linked claims, in the order the walk produces them,
        because each is a different mechanism and a failure should name which:

          1. DEPTHS.  Every round's dispatched depth is exactly
             ``select_chain_depth(cap, queue_len_at_dispatch, state_before)``.
             Cross-checking against the pure policy — rather than against a
             list of literals alone — is what distinguishes "the bisector ran"
             from "the fixture happened to produce these numbers".
          2. HALVING LADDER.  Every state transition is
             ``next_halving_state(passed, built_depth)``, with the floor arm's
             pass-only reset folded at depth 1.
          3. THE CLEAN PREFIX LANDS AT THE FLOOR.  Items 1 and 2 land on
             floor rounds, in order, one main commit each, and neither carries
             ``landed_via_chain`` — a floor round chains nothing, so attributing
             its landing to a chain would inflate η1's items-per-deep-verify.
          4. THE CULPRIT BLOCKS, THEN DEEP RESUMES.  Item 3 fails its OWN
             un-chained subset verify and blocks through the ordinary path;
             the very next pass resets the bisector and the round after it
             chains again at ``min(queue, cap)``.
        """
        from orchestrator.merge_queue import next_halving_state, select_chain_depth

        scene, queue_lens, _permits = await self._walk(
            git_repo, tmp_path, monkeypatch, db_name='gate-row3-walk.db',
        )
        rounds = scene.rounds
        heads = [r['req'].task_id for r in rounds]
        depths = scene.depths
        states = [r['halving_state'] for r in rounds]

        assert heads == [
            '101', '101', '101', '102', '102', '102',
            '103', '103', '103', '104', '105',
        ], f'the requeued head must be re-dispatched until it resolves; got {heads!r}'

        # ── 1. depths ────────────────────────────────────────────────────────
        assert depths == [6, 3, None, 5, 2, None, 4, 2, None, None, 2], (
            f'the bisection walk moved: {depths!r}'
        )
        states_before = [None, *states[:-1]]
        predicted = [
            select_chain_depth(_ROW3_CAP, qlen, state)
            for qlen, state in zip(queue_lens, states_before, strict=True)
        ]
        assert predicted == depths, (
            f'a dispatched depth disagreed with select_chain_depth evaluated on '
            f'the live queue: policy said {predicted!r}, dispatch did {depths!r} '
            f'(queue_len {queue_lens!r}, state before each round {states_before!r})'
        )

        # ── 2. the halving ladder, every step derived from the policy ────────
        assert states == [3, 1, None, 2, 1, None, 2, 1, 1, None, None], (
            f'the halving ladder moved: {states!r}'
        )
        for i, (rec, before, after) in enumerate(
            zip(rounds, states_before, states, strict=True), start=1,
        ):
            passed = rec['outcome'] is not None and rec['outcome'].status == 'done'
            if rec['chain'] is not None:
                expected = next_halving_state(passed, 1 + len(rec['chain'].links))
            elif passed and before is not None:
                # The floor arm's reset, folded at depth 1: at the floor the
                # tree that ran IS a one-item chain.
                expected = next_halving_state(True, 1)
            else:
                # A red floor round folds NOTHING — the arm is pass-only.
                expected = before
            assert after == expected, (
                f'round {i} (head {rec["req"].task_id}, chain '
                f'{None if rec["chain"] is None else 1 + len(rec["chain"].links)}, '
                f'passed={passed}) left the bisector at {after!r}; the policy '
                f'says {expected!r}'
            )

        # ── 3. the clean prefix lands at the floor, unchained ────────────────
        floor_landings = [
            r for r in rounds if r['advanced'] and r['chain'] is None
        ]
        assert [r['req'].task_id for r in floor_landings] == ['101', '102', '104'], (
            f'expected the three clean items to land on FLOOR rounds; got '
            f'{[(r["req"].task_id, r["round"]) for r in floor_landings]!r}'
        )
        first_two = floor_landings[:2]
        assert [r['round'] for r in first_two] == [3, 6], (
            f'items 1 and 2 must land on the floor round that ENDS each '
            f'bisection, got rounds {[r["round"] for r in first_two]!r}'
        )
        counts = []
        for rec in first_two:
            _rc, out, _err = await _run(
                ['git', 'rev-list', '--count',
                 f'{rec["main_before"]}..{rec["main_after"]}'],
                cwd=git_repo,
            )
            counts.append(int(out.strip()))
        assert counts == [2, 2], (
            f'each floor landing is ONE --no-ff merge commit plus the branch '
            f'commit it brings in; got {counts!r} new commits on main'
        )
        finalized = {r['branch']: r for r in _finalized_rows(scene.db_path)}
        for tid in ('101', '102', '104'):
            assert finalized[tid]['state'] == 'done', (
                f'task {tid} must land, got {finalized[tid]!r}'
            )
            assert finalized[tid]['landed_via_chain'] is None, (
                f'task {tid} landed on a FLOOR round, which chains nothing, so '
                f'landed_via_chain must be absent; got '
                f'{finalized[tid]["landed_via_chain"]!r}'
            )

        # ── 4. the culprit blocks, then deep resumes ─────────────────────────
        blocking = rounds[8]
        assert blocking['req'].task_id == _ROW3_BAD_TASK
        assert blocking['chain'] is None, (
            'the culprit must be isolated by its OWN un-chained subset verify, '
            f'but round 9 dispatched a chain of '
            f'{1 + len(blocking["chain"].links)} items'
        )
        assert blocking['outcome'] is not None
        assert blocking['outcome'].status == 'blocked', (
            f'the red item must block through the ordinary path; got '
            f'{blocking["outcome"].status!r}'
        )
        assert finalized[_ROW3_BAD_TASK]['state'] == 'blocked'
        resumed = rounds[10]
        assert resumed['chain'] is not None and len(resumed['chain'].links) == 1, (
            'deep must RESUME once the bad item is gone; round 11 dispatched '
            f'{None if resumed["chain"] is None else 1 + len(resumed["chain"].links)} '
            'items'
        )
        assert [tid for tid, _ in resumed['chain'].links] == ['106']
        assert sorted(resumed['landed'], key=int) == ['101', '102', '104', '105', '106'], (
            f'the resumed chain must land its whole prefix; got {resumed["landed"]!r}'
        )
        for tid in ('105', '106'):
            assert finalized[tid]['landed_via_chain'] == 1, (
                f'task {tid} landed VIA the resumed chain, so it must carry '
                f'landed_via_chain=1 (one per landed item, summed by η1); got '
                f'{finalized[tid]["landed_via_chain"]!r}'
            )

    async def test_the_innocent_successors_are_never_smeared(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """Items 4–6 pay NOTHING for being chained behind a red item.

        This is the isolation claim stated negatively, and it is the one a
        positional script cannot make at all: rounds 1, 4, 7 and 8 all built
        chains CONTAINING tasks 104–106 and all came back red, so if a red tip
        were ever attributed down the chain it would land on exactly these
        three.  Two surfaces are checked, because they fail differently:

          * the REQUEST surface — no blocked/failed ``MergeOutcome`` for any of
            them; each resolves ``done`` on its own verdict, later;
          * the DURABLE surface — through the whole bisection (rounds 1–9)
            their event streams stay at ``['merge_queued']``.  A ``merge_attempt``
            or ``merge_finalized`` row appearing while they were merely chain
            LINKS would mean the walk rendered per-item outcomes for a tree
            whose verdict names no culprit — β's "never emit per-item outcomes"
            contract, read from the tier η1 actually reads.

        Ends with the FULL two-way oracle: the run must come to rest with no
        residue at all, so the whole-registry surfaces the mid-run checkpoints
        cannot assert are asserted here exactly once.
        """
        innocents = ('104', '105', '106')
        streams: list[dict[str, list[str]]] = []

        def _snapshot(scene: _GateScene, _rec: dict) -> None:
            streams.append({
                tid: _events_for_task(scene.db_path, tid) for tid in innocents
            })

        scene, _queue_lens, permits_before = await self._walk(
            git_repo, tmp_path, monkeypatch, db_name='gate-row3-smear.db',
            after_round=_snapshot,
        )

        during_bisection = scene.rounds[8]
        assert during_bisection['round'] == 9, 'round 9 ends the bisection'
        chained_at_least_once = {
            tid
            for rec in scene.rounds[:9] if rec['chain'] is not None
            for tid, _sha in rec['chain'].links
        }
        assert set(innocents) <= chained_at_least_once, (
            f'the claim is vacuous unless 104-106 were really chained behind '
            f'the red item; chains held {sorted(chained_at_least_once)!r}'
        )

        for tid in innocents:
            outcome = scene.reqs[tid].result.result()
            assert outcome.status == 'done', (
                f'task {tid} was only ever a LINK in a red chain, never a '
                f'culprit; got {outcome.status!r} ({outcome.reason!r})'
            )
        blocked_rows = [
            r for r in _finalized_rows(scene.db_path)
            if r['state'] == 'blocked'
        ]
        assert [r['branch'] for r in blocked_rows] == [_ROW3_BAD_TASK], (
            f'exactly ONE item may block in this walk — the red one; got '
            f'{[(r["branch"], r["state"]) for r in blocked_rows]!r}'
        )

        # The durable-tier half.  Sampled from the per-round snapshot taken
        # while the bisection was still running, so it cannot be satisfied by
        # rows the innocents legitimately earned once they became heads.
        for tid in innocents:
            during = streams[8][tid]
            assert during == ['merge_queued'], (
                f'task {tid} was a chain LINK for the whole bisection and must '
                f'have earned no per-item row; its stream after round 9 was '
                f'{during!r}'
            )

        drained = _drain_residue(scene.worker)
        assert drained == set(), (
            f'this walk resolves every request, so nothing may be left queued; '
            f'drained {drained!r}'
        )
        _assert_two_way_quiescent(
            scene.worker,
            await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()),
            permits_before=permits_before,
        )

    async def test_the_content_keyed_oracle_is_green_without_the_bad_file(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """CONTROL: the same fixture, keyed on a file no branch creates.

        Without this the whole row could pass for the wrong reason — an oracle
        that is red for a reason unrelated to its ``bad_file`` argument (a
        typo'd path that never matches, inverted, would be red everywhere)
        produces a plausible-looking bisection too.  Keyed on an absent file
        the very first round must chain all six items, pass, and land the whole
        prefix, so the redness in the row above is demonstrably caused by
        item 3's content and by nothing else in the fixture.
        """
        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=_ROW3_CAP, n_followers=5, db_name='gate-row3-control.db',
            verdict=_verdict_from_tree('f999-no-branch-creates-this.txt'),
        )
        rec = await scene.round_(tag='row3ctl', head_tid='101')

        assert rec['chain'] is not None and len(rec['chain'].links) == 5, (
            f'six queued items at cap=6 must chain all six; got '
            f'{None if rec["chain"] is None else 1 + len(rec["chain"].links)}'
        )
        assert scene.worker._chain_halving_state is None, (
            f'a green tip resets the bisector; got '
            f'{scene.worker._chain_halving_state!r}'
        )
        assert sorted(rec['landed'], key=int) == ['101', '102', '103', '104', '105', '106'], (
            f'a green six-item tip lands the whole prefix; got {rec["landed"]!r}'
        )
        assert scene.verdicts and not any(v['passed'] is False for v in scene.verdicts), (
            f'no verdict may be red when the keyed file exists nowhere; got '
            f'{scene.verdicts!r}'
        )
