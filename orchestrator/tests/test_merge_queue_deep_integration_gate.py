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
    InflightEntry,
    MergeResult,
    QueuedBranch,
    RealMergeItem,
    SpecPermit,
    VerifyWorktreeHandle,
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

    *bump_repo* is the shape Row 6 uses, deliberately: a synthetic
    ``AdvanceOutcome`` states the abort's CLASS, while a real ref move also
    leaves the repository in the state the NEXT round has to rebuild against —
    which is the half of decision #9 the composition sweep exists to reach.

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


def _spy_teardown_pair(
    worker: SpeculativeMergeWorker, monkeypatch,
) -> list[str]:
    """Record the SANCTIONED verify-teardown pair, in call order.

    Hand-rolling ``verify_task.cancel()`` / ``_abort_remote_verify`` is what the
    AST ratchet in
    test_merge_queue_concurrent_verify.py::TestVerifyTeardownChokepoint
    forbids, so "the head verify was cancelled" is not the claim worth making —
    "it went through the chokepoint, in the order the head-failure-cascade
    template uses" is.  Entries read ``teardown:<task_id>`` then
    ``cancel_release:<lease name>``.

    PASSTHROUGH and INSTANCE-level (test_merge_queue_reachback_patch_guard.py
    freezes the ``orchestrator.merge_queue.<private>`` reach-back surface), so
    the real teardown genuinely runs and the lease is genuinely released while
    the call ledger stays observable.

    The EMPTY list is a meaningful reading, not merely the absence of one: the
    adoption gates its whole teardown block on ``not head.verify_task.done()``,
    so a head whose verdict already arrived records nothing here by design.
    """
    calls: list[str] = []
    real_teardown = worker._teardown_verify_task
    real_cancel = worker._cancel_and_release_tracked

    async def _recording_teardown(lease, verify_task, task_id, **kwargs):
        calls.append(f'teardown:{task_id}')
        return await real_teardown(lease, verify_task, task_id, **kwargs)

    async def _recording_cancel(lease):
        calls.append(f'cancel_release:{getattr(lease, "name", None)}')
        return await real_cancel(lease)

    monkeypatch.setattr(worker, '_teardown_verify_task', _recording_teardown)
    monkeypatch.setattr(worker, '_cancel_and_release_tracked', _recording_cancel)
    return calls


def _spy_spec_lane_acquire(git_ops: GitOps, monkeypatch) -> list[tuple]:
    """Record every ``acquire_spec_lane`` call as ``(base_commit, lane, warm)``.

    Row 7's golden states the kill switch's absences BY NAME, and "no scratch
    lane was ever claimed" is one of them.  Counting RELEASES would be the
    inferential version of that claim — release is exactly-once-per-build, so
    zero releases with zero builds does imply zero acquisitions — but a leak is
    precisely the case where the two diverge, and a golden that could not tell
    "never acquired" from "acquired and never given back" would be asserting
    the wrong absence.

    Spied on the GitOps INSTANCE (test_merge_queue_reachback_patch_guard.py
    freezes the ``orchestrator.merge_queue.<private>`` reach-back surface) and
    PASSTHROUGH, so the lane pool stays real.
    """
    calls: list[tuple] = []
    real = git_ops.acquire_spec_lane

    async def _recording(base_commit, *args, **kwargs):
        lane, warm = await real(base_commit, *args, **kwargs)
        calls.append((base_commit, lane, warm))
        return lane, warm

    monkeypatch.setattr(git_ops, 'acquire_spec_lane', _recording)
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


def _lane_of(worker: SpeculativeMergeWorker, task_id: str) -> str | None:
    """Return the lane whose buffer currently holds *task_id*, or ``None``.

    MEMBERSHIP, not position — the composition rows that care about position
    compare the whole ``[r.task_id for r in buffer]`` list instead, because
    submission order inside a lane IS the next round's ``chain_snapshot`` and a
    membership check cannot see a reorder.  This is for the weaker claim that
    still needs saying: the item is on the queue at all, in the lane it was
    submitted to, rather than having been silently promoted or dropped.
    """
    for lane in ('high', 'normal'):
        if any(r.task_id == task_id for r in worker._lane_buffers[lane]):
            return lane
    return None


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


def _rows_of_type(
    db_path: Path, event_type: EventType, *, task_id: str | None = None,
) -> list[dict]:
    """Return every row of *event_type*'s parsed ``data`` dict, in order.

    *task_id*, when given, restricts the read to that task's rows — which is
    how :func:`_gate_round_transcript` attributes a ``merge_verify`` /
    ``merge_finalized`` row to the round whose head owns it, without the
    extractor having to track rowid watermarks the scene never recorded.

    The un-specialised sibling of :func:`_finalized_rows` / :func:`_verify_rows`,
    for the rows a claim is about the ABSENCE of.  Reading the durable tier
    rather than a capturing fake matters most for an absence claim: an
    in-memory recorder can only miss an emit the real store would have
    persisted, so a fake would make "no row" the easy answer.
    """
    import json
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        if task_id is None:
            rows = conn.execute(
                'SELECT data FROM events WHERE event_type = ? ORDER BY rowid',
                (event_type.value,),
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT data FROM events WHERE event_type = ? AND task_id = ? '
                'ORDER BY rowid',
                (event_type.value, task_id),
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


def _blocked_signature(outcome) -> str:
    """The ladder key workflow.py would compute for *outcome*, via the SHIPPED fn.

    workflow.py stashes ``result.reason`` / ``result.failure_category`` /
    ``result.failure_cause_hint`` off a blocked :class:`MergeOutcome` and later
    fingerprints exactly that triple.  Transcribing the fingerprint here would
    make the gate pass through a format change; delegating to
    ``workflow._compute_merge_outcome_signature`` makes it fail on one.
    """
    from orchestrator.workflow import _compute_merge_outcome_signature

    return _compute_merge_outcome_signature(
        outcome.failure_category, outcome.failure_cause_hint, outcome.reason or '',
    )


_MERGE_THRASH_THRESHOLD = 2
"""``OrchestratorConfig.max_consecutive_merge_thrash``'s shipped default.

Restated as a literal rather than read off a config, because the threshold is
what the ROW is about: two identical rendered merge failures in a row is
exactly 3003's signature class, and this file's whole claim is that a deep
round can never produce that pair.  A test that read the live default would
still pass if the default were raised to 5 — and the pair it drives would then
no longer be the trip input it is written to be.  Kept pinned to the shipped
value, with :class:`TestLadderDriverIsNotInert` proving the fold reaches it.
"""


def _ladder_after(
    rounds: list[dict],
    *,
    ledger: RetryLedger | None = None,
    requests: list[MergeRequest] | None = None,
    threshold: int = _MERGE_THRASH_THRESHOLD,
) -> RetryLedger:
    """Fold *rounds*' REAL rendered outcomes through workflow.py's OWN ladder.

    Row 8's other half.  The merge-queue side of the row is a silence claim;
    this is the driver that carries that silence to the consumer 3003's
    signature class actually lives in, so the row can assert the ladder is
    UNMOVED rather than merely that no event was written.

    WHAT WORKFLOW.PY ACTUALLY CONSUMES, and therefore what this reproduces —
    two inputs, both taken off a rendered blocked :class:`MergeOutcome`:

      1. WHETHER there is one at all.  ``_submit_to_merge_queue``'s generic
         blocked path stashes ``result.reason`` into
         ``TaskWorkflow._last_merge_block_reason``, and the merge-phase loop
         gates the ENTIRE thrash check on that field being non-None.  Every
         other outcome status — ``done``, and the REQUEUED arm a red deep tip
         produces — leaves it None and the ladder untouched.  A deep fail
         requeues WITHOUT rendering a block reason, which is precisely why the
         row's claim holds.
      2. Its SIGNATURE, if there is one: ``_compute_merge_outcome_signature``
         over ``(failure_category, failure_cause_hint, reason)`` — the same
         triple ``_submit_to_merge_queue`` stashes alongside the reason.

    Both are taken through the SHIPPED functions
    (:func:`orchestrator.workflow._compute_merge_outcome_signature`,
    :func:`orchestrator.workflow._evaluate_merge_thrash`) rather than
    transcribed.  A transcription would let a signature-format change or a
    counter-arithmetic change pass this gate silently, which for a row whose
    entire content is "the shipped ladder does not move" would be the one
    failure mode that matters.

    ONE LEDGER, NOT ONE PER TASK.  workflow.py's ladder is per-task state
    (``metadata.retry_ledger``), and a scene's rounds are all dispatches of ONE
    head, so a single ledger IS the head's.  Any additional blocked outcome
    rendered for a chained LINK is folded in too — *requests*, when given, is
    the scene's whole registry and is scanned for exactly that.  Folding a
    link's outcome into the head's ledger is not what production would do (the
    link has its own workflow and its own ledger); it is deliberately STRICTER,
    because this row's claim is that no link renders one at all.  A leak
    anywhere in the queue therefore moves the counter and fails the assertion,
    instead of being filed to a per-task ledger nobody looks at.

    ORDER.  Rounds first, in round order — the only real chronology available —
    then any link outcome not already folded, in *requests* order.  The
    ordering is deterministic rather than arbitrary because the ladder's reset
    arm is order-sensitive: it compares each signature against the PREVIOUS
    one, so a driver that shuffled its inputs could turn a genuine
    increment-to-threshold into a reset and report "unmoved" for the wrong
    reason.

    Args:
        rounds: ``_GateScene.rounds`` records (or fabricated ones — the fold
            reads only ``rec['outcome']``, which is what makes
            :class:`TestLadderDriverIsNotInert` able to control it).
        ledger: the starting ledger.  Defaults to a virgin
            :class:`RetryLedger`; pass a MID-RUN one to make the claim against
            a task that genuinely blocked sequentially earlier, where a leaked
            signature would show up as either an increment or a reset.
        requests: the scene's request registry, scanned for a blocked outcome
            rendered for any item the rounds do not already account for.
        threshold: what ``_evaluate_merge_thrash`` escalates at.  Affects only
            the verdict's ``escalate`` flag, which this helper drops — the
            counter arithmetic is threshold-independent — but is threaded so
            the control test can state the number it is asserting against.

    Returns:
        The ledger that came back out.  Hand it to
        :func:`_assert_two_way_quiescent` as ``ladder={'before': ..., 'after':
        ...}`` to make the unmoved claim part of the round-by-round oracle.
    """
    from orchestrator.workflow import _evaluate_merge_thrash

    current = RetryLedger() if ledger is None else ledger

    outcomes = [rec.get('outcome') for rec in rounds]
    if requests:
        seen = {id(o) for o in outcomes if o is not None}
        for req in requests:
            if not req.result.done() or req.result.cancelled():
                continue
            outcome = req.result.result()
            if id(outcome) not in seen:
                seen.add(id(outcome))
                outcomes.append(outcome)

    for outcome in outcomes:
        # Gate 1: workflow.py's `_last_merge_block_reason is not None`.  A
        # REQUEUED round renders no outcome at all (None here), and a landed
        # one renders `status='done'` — neither reaches the thrash check.
        if outcome is None or getattr(outcome, 'status', None) != 'blocked':
            continue
        signature = _blocked_signature(outcome)
        current = _evaluate_merge_thrash(
            current, current.last_merge_outcome_signature, signature, threshold,
        ).ledger
    return current


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


def _assert_midrun_conserved(
    worker: SpeculativeMergeWorker,
    main_sha: str,
    permits_before: _PermitCensus,
    *,
    where: str,
) -> None:
    """The MID-RUN half of the two-way contract, callable BETWEEN rounds.

    :func:`_assert_two_way_quiescent` cannot be called between the rounds of a
    run that is deliberately mid-flight, and the reason is a property of the
    run rather than a gap in the oracle: its surfaces (a), (d) and (f) are
    WHOLE-REGISTRY claims — every request resolved, the request-liveness ledger
    empty, no non-terminal lifecycle entry — and a mid-flight run has queued
    items with pending futures at every checkpoint.  Asserting them there would
    either fail honestly or force the caller to pass an empty request list,
    which is the vacuum the full oracle's guard clauses exist to refuse.

    What DOES hold at every instant is CONSERVATION, so that is what this
    checks: the three fail-safe audits plus the token-level permit census.  The
    full oracle then runs once, at the end, after :func:`_drain_residue` has
    confirmed the run left exactly the residue it promised.

    *where* is interpolated into every failure message, because this is called
    several times inside one test and "which round" is the first thing a reader
    needs.

    NOTE the same ``_running`` fail-open hazard the full oracle guards:
    ``speculation_accounting_violations`` and ``worktree_ledger_violations``
    both short-circuit to ``[]`` on a stopped worker.  A mid-run caller has a
    live worker by construction (it is between rounds of a run it is driving),
    so this is asserted rather than merely documented.
    """
    assert worker._running is True, (
        f'_assert_midrun_conserved requires a worker with _running=True {where}, '
        f'got {worker._running!r}: both audits below short-circuit to [] on a '
        f'stopped worker and would pass vacuously'
    )
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


def _gate_head_verify_task(
    order: list[str] | None = None,
    *,
    lease_ctx=None,
    started: asyncio.Event | None = None,
):
    """A live SLOT-1 head verify task that never finishes on its own.

    Cloned from test_merge_queue_deep_landing.py::_head_verify_task.  The
    never-finishing part is the point: production's head is still verifying
    when the speculative tip's verdict arrives, and δ's whole head-cancel
    contract is about what happens to a verify that has NOT concluded.  A task
    that resolved on its own would model the uninteresting case.

    *lease_ctx*, when given, is an async context manager entered for the whole
    (never-ending) span — hand it ``git_ops.merge_verify_lease()`` and the head
    holds the REAL ``_merge-verify`` lane lock, which is what makes Row 9's
    staging check ("both axes read BUSY beforehand") measure a genuinely-held
    lane rather than an empty one.

    *started*, when given, is set once the lease is actually held, so a caller
    waits on the fact instead of sleeping for it.  *order*, when given, records
    ``'cancelled'`` at teardown — the remote-abort-before-cancel ordering claim
    reads it alongside the lease's own recorder.
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
        self.lane_acquires: list[tuple] = []
        self.advance_calls: list[tuple] = []
        self.reqs: dict[str, MergeRequest] = {}
        self.rounds: list[dict] = []
        self._round_no = 0
        # Slot 1, when a scenario opts into the two-slot topology via
        # `attach_head`.  `None` for the single-slot rounds every other row
        # drives — `round_` is untouched by this, deliberately: four landed
        # rows depend on its exact behaviour.
        self.head_entry: InflightEntry | None = None
        self.head_mc: str | None = None

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
        n_acquires_before = len(self.lane_acquires)
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
            'lane_acquires': self.lane_acquires[n_acquires_before:],
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

    # ── the TWO-SLOT topology (step-14): rows 5 and 9 ───────────────────────

    async def attach_head(
        self, head_tid: str, *, task_factory=None, lease=None,
        speculative: bool = False,
    ) -> InflightEntry:
        """Register *head_tid* as a live SLOT-1 verify, the way dispatch would.

        Production's topology, which :meth:`round_` alone cannot express::

            main <- I0 (head, non-speculative, base = REAL main)
                      <- I1 (speculative, base = I0's merge commit, chains)
                         <- I2 ... Ik (links, built on I1's merge commit)

        The head is appended through the real ``_inflight_append`` chokepoint
        after a real ``_note_transition``, so it is genuinely registered at
        VERIFYING and genuinely visible to ``_finalizing_head_entry`` /
        ``_inflight[0]`` — which is what ``_adopt_head_on_tip_authority``
        reads.  Stuffing the deque directly would leave the ItemLifecycle
        registry disagreeing with it, and the conservation oracle would then be
        auditing a state production never produces.

        *task_factory* is called with the scene's OWN ``GitOps`` and returns the
        head's verify task; the default never finishes on its own (see
        :func:`_gate_head_verify_task`).  Passing the scene's GitOps rather
        than letting the factory build one is what lets a factory hold the same
        ``_merge-verify`` lane lock the worker will later release, with no
        module-attribute patching.

        *lease* defaults to a LOCAL one, because ``lease.is_local`` is the axis
        δ's cleanliness argument splits on: a local head frees BOTH lease axes
        by construction when its task is cancelled, while a remote one SIGKILLs
        and must clear the holder-pgid rendezvous explicitly.
        """
        from orchestrator.merge_types import ItemLifecycleState

        assert self.head_entry is None, (
            'a scene has ONE slot-1 head; attaching a second would leave the '
            'first unfinalized in the deque and every ordering claim below it '
            'ambiguous'
        )
        worker = self.worker
        main_sha = await _rev_parse(self.repo, 'main')
        head_mc = await _merge_commit_off_main(
            self.repo, f'task/{head_tid}', f'head-{head_tid}',
        )
        popped = worker._pop_next_pickable()
        assert popped is not None and popped.task_id == head_tid, (
            f'expected {head_tid} to be the pickable head, got '
            f'{None if popped is None else popped.task_id}'
        )
        wt = _ephemeral_merge_wt(self.git_ops, f'head-{head_tid}')
        head_item = RealMergeItem(
            request=popped,
            merge_result=MergeResult(
                success=True, merge_commit=head_mc, merge_worktree=wt,
            ),
            merge_wt=wt,
            base_sha=main_sha,      # slot 1 CASes against REAL main
            speculative=speculative,    # the trust anchor — never chained
        )
        entry = InflightEntry(
            item=head_item,
            lease=lease if lease is not None else _local_lease(),
            verify_task=(
                task_factory(self.git_ops) if task_factory is not None
                else _gate_head_verify_task()
            ),  # type: ignore[arg-type]
            merge_wt=wt,
            was_speculative=speculative,
            # Mirrors `_dispatch_item`: the box is built alongside the entry and
            # seeded with the item's own ephemeral, so a head running the REAL
            # `_run_inflight_verify` has somewhere for a warm swap to publish.
            verify_wt=VerifyWorktreeHandle(merge_wt=wt),
        )
        worker._note_transition(
            popped.request_id, ItemLifecycleState.MERGING,
            ItemLifecycleState.DISPATCHING, live_obj=head_item,
        )
        worker._inflight_append(entry)
        self.head_entry = entry
        self.head_mc = head_mc
        return entry

    async def adopting_round_(self, *, tag: str, spec_tid: str) -> dict:
        """Drive one round STACKED on the attached head, then finalize both.

        The two-slot twin of :meth:`round_`.  Three differences, all forced by
        the topology rather than chosen:

          * the dispatching item is merged onto the HEAD's merge commit
            (``base_sha=self.head_mc``), not onto main — that is what makes the
            head's tree a subset of the tip's;
          * the head is finalized FIRST.  That order is the deque's, not this
            method's: the head was appended before the spec entry, so
            FINALIZE-HEAD reaches it first, which is exactly what makes the
            spec item's ``expected_main`` (= the head's merge commit) correct
            without any coordination;
          * the head finalize is BOUNDED, and the bound is part of the
            contract.  The head's verify never completes on its own in these
            scenes, so an adoption that failed to tear it down would leave
            ``_finalize_inflight`` parked on ``await entry.verify_task``
            FOREVER, with the whole queue stalled behind it on a verdict it no
            longer needs.  A timeout makes that a fast, legible RED instead of
            a hung suite.

        Records the same per-round facts :meth:`round_` does, plus
        ``'head_advanced'`` and ``'head_entry'``, so the composition rows read
        one uniform ``scene.rounds`` regardless of which driver produced it.
        """
        assert self.head_entry is not None, (
            'adopting_round_ requires attach_head() first — without a slot-1 '
            'entry there is nothing to adopt and this is just round_'
        )
        assert self.head_mc is not None
        self._round_no += 1
        worker = self.worker
        head_entry = self.head_entry
        # Slot-1 steady state, for the reason documented at length in `round_`.
        worker._n_failed = False
        worker._remerge_occurred = False
        n_built_before = len(self.built)
        n_acquires_before = len(self.lane_acquires)
        n_releases_before = len(self.lane_releases)
        n_advances_before = len(self.advance_calls)
        main_before = await _rev_parse(self.repo, 'main')
        spec_mc = await _merge_commit_onto(
            self.repo, f'task/{spec_tid}', self.head_mc,
            f'{tag}-r{self._round_no}',
        )
        popped = worker._pop_next_pickable()
        assert popped is not None and popped.task_id == spec_tid, (
            f'expected {spec_tid} to be the pickable spec item, got '
            f'{None if popped is None else popped.task_id}'
        )
        wt = _ephemeral_merge_wt(self.git_ops, f'{tag}-r{self._round_no}')
        item = RealMergeItem(
            request=popped,
            merge_result=MergeResult(
                success=True, merge_commit=spec_mc, merge_worktree=wt,
            ),
            merge_wt=wt,
            base_sha=self.head_mc,  # stacked on the head's merge commit
            speculative=True,       # slot 2 — the only kind that chains
        )
        entry = await worker._dispatch_item(item)
        assert entry is not None, 'dispatch must not decline: a host is free'
        assert entry.verify_task is not None
        await entry.verify_task
        # `_finalize_inflight` does NOT pop — the real `_verifier_loop` calls
        # `_inflight_popleft()` and hands it the entry it took.  Mirror that,
        # or the head stays in `_inflight` forever and `frozen_prefix()` keeps
        # reporting a landed item against a base main has long since passed.
        # Conditional because the ADOPTING arm may already have removed it:
        # a head that was still verifying is torn down and retired by the
        # adoption itself, and popping again would take the wrong entry.
        if worker._inflight and worker._inflight[0] is head_entry:
            assert worker._inflight_popleft() is head_entry
        head_advanced = await asyncio.wait_for(
            worker._finalize_inflight(head_entry), timeout=60,
        )
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
            'head_advanced': head_advanced,
            'head_entry': head_entry,
            'main_before': main_before,
            'main_after': await _rev_parse(self.repo, 'main'),
            'head_mc': self.head_mc,
            'halving_state': worker._chain_halving_state,
            'outcome': outcome,
            'built': self.built[n_built_before:],
            'lane_acquires': self.lane_acquires[n_acquires_before:],
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

    # ── the hot-reload seam (step-14): row 10 ───────────────────────────────

    def reload_cap(self, chain_cap: int) -> dict:
        """Flip ``merge_deep.chain_cap`` through the REAL ``config.apply_reload``.

        Returns the reload disposition dict verbatim, so the caller asserts on
        ``reloaded`` / ``applied`` / ``restart_required`` rather than on this
        helper's summary of them — "reloaded is not everything took effect" is
        an operational rule in this repo, and a helper that collapsed the three
        would hide exactly the tier regression Row 10 is written to catch.

        Mutates the scene's LIVE config object in place, which is what makes
        the flip observable to already-enqueued requests: ``OrchestratorConfig``
        and ``MergeDeepConfig`` are plain mutable ``BaseModel``s, every
        ``MergeRequest`` holds ``self.config`` by REFERENCE, and the worker
        reads the cap off the dispatching request's config at dispatch time.
        Rebuilding the config instead would leave every queued request pointing
        at the old one and the row would silently be testing nothing.
        """
        from orchestrator.config import apply_reload

        result = apply_reload(self.config, _make_config(self.repo, chain_cap=chain_cap))
        assert isinstance(result, dict), (
            f'apply_reload must return a disposition dict; got {result!r}'
        )
        return result


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
    advance_hook=None,
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

    *advance_hook* is threaded to :func:`_spy_advance_main`, and it is the ONLY
    way a scene can disrupt the CAS walk from the inside: the walk's abort is a
    VALUE ``advance_main`` returns rather than an exception it raises, so a
    scenario about the abort has to answer that one call differently.  Build it
    with :func:`_abort_hook_at`; the call ordinal it arms on is cumulative
    across the whole scene, so a hook armed inside round 1's walk leaves every
    later round running clean.

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
    scene.lane_acquires = _spy_spec_lane_acquire(git_ops, monkeypatch)
    scene.lane_releases = _spy_chain_lane_release(monkeypatch)
    scene.advance_calls = _spy_advance_main(git_ops, monkeypatch, hook=advance_hook)

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

        Kept as a method purely for call-site brevity inside this row's walk;
        the body it delegates to is the module-level
        :func:`_assert_midrun_conserved`, which step-13's composition sweep
        also drives.  ONE implementation, because a second copy of a
        conservation oracle is exactly the kind of duplicate that goes quietly
        out of step with the surfaces it audits.
        """
        _assert_midrun_conserved(worker, main_sha, permits_before, where=where)

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


# ═══════════════════════════════════════════════════════════════════════════
# -- step-09 RED: Row 8 — DEEP FAILS NEVER FEED THE THRASH GUARD --
#
# The row upstream is ONE-WAY.  test_merge_queue_deep_landing.py pins the
# merge-queue half as event SILENCE — ``test_the_abort_feeds_the_thrash_ladder
# _nothing`` and ``test_two_consecutive_tip_fails_render_nothing_for_any_link``
# assert that an unlanded link emits nothing but its own ``merge_queued`` and
# renders no outcome.  Both are statements about merge_queue.py.
#
# But the row's CLAIM is about workflow.py: task 3003's signature class is a
# ``consecutive_merge_thrash`` ladder that trips on two identical rendered
# merge failures in a row and escalates a human.  "Deep fails never feed the
# thrash guard" is therefore a claim about the CONSUMER, and nobody anywhere
# in the tree drives that consumer with a deep round's real outputs.  Event
# silence is the PREMISE; the conclusion — that the silence actually reaches
# the ladder and leaves it unmoved — is unasserted.
#
# So this row is driven TWO-WAY:
#
#   WORKER SIDE (the premise, re-asserted at composition level across a PAIR
#   of rounds rather than as two independent single-round scenes): zero blocked
#   ``MergeOutcome``s, zero ``merge_attempt`` rows, every chained future still
#   pending, every chained request still at its original lane INDEX, main
#   unmoved.
#
#   LEDGER SIDE (original here): the pair of rounds' real observable outputs
#   are folded through the SHIPPED ladder — ``workflow._evaluate_merge_thrash``
#   over a real ``shared.task_metadata.RetryLedger``, keyed by the SHIPPED
#   ``workflow._compute_merge_outcome_signature`` — and the ladder comes back
#   byte-identical.  Deriving the signature THROUGH the production function
#   rather than restating a string is the point: a signature-format change
#   would otherwise slip past this gate silently.
#
# TWO rounds, and both at depth >= 4, is the load-bearing shape.
# ``max_consecutive_merge_thrash`` defaults to 2, so a PAIR of identical
# rendered failures is exactly the input that trips the ladder — one round
# could not distinguish "never feeds it" from "has not fed it twice yet".
#
# The inertness control is mandatory.  An "unmoved" assertion over a driver
# that folds nothing proves only that the driver is inert, so the same helper
# is handed a genuine repeated blocked signature and must advance the counter
# to the escalation threshold.
# ═══════════════════════════════════════════════════════════════════════════

_ROW8_CAP = 8
"""Chain cap for the row-8 pair.  With 8 followers the walk is 8 -> 4: round 1
targets ``min(queue_len=9, cap=8)`` and round 2 targets ``min(9, 8,
next_halving_state(False, 8)=4)``.  Both are >= 4, so NEITHER round of the
pair is near the ``< 2`` floor where no chain code runs at all — a pair that
halved into the floor would be asserting silence about a round that never
chained."""

_ROW8_FOLLOWERS = 8


@pytest.mark.asyncio
@pytest.mark.timeout(300)
class TestRow8DeepFailsNeverFeedTheThrashLadder:
    """Row 8: a red deep tip renders nothing, so the ladder has nothing to eat."""

    async def _pair(
        self, git_repo: Path, tmp_path: Path, monkeypatch, *, db_name: str,
    ) -> _GateScene:
        """Two consecutive RED deep tips over the same head, depths 8 then 4.

        The head is re-dispatched by OBJECT on round 2 rather than re-popped:
        a red tip REQUEUES its dispatching request onto ``_queue`` (deliberately
        NOT back into a lane buffer — draining it would make the head a member
        of its own next chain), so ``_pop_next_pickable`` would hand back a
        FOLLOWER and the second round would be a different scenario entirely.
        """
        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=_ROW8_CAP, n_followers=_ROW8_FOLLOWERS, db_name=db_name,
            script=[False, False],
        )
        head = scene.reqs['101']
        await scene.round_(tag='row8', head_tid='101')
        await scene.round_(tag='row8', head_tid='101', req=head)
        return scene

    async def test_two_red_deep_tips_render_nothing_on_either_tier(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """WORKER SIDE — the premise, across the PAIR that would trip the ladder.

        Five absences, each a different mechanism, so a failure names which:
        the depths really were deep, no outcome was rendered, no durable row
        was written, no future resolved, and no request moved in its lane.
        """
        from orchestrator.merge_queue import next_halving_state, select_chain_depth

        scene = await self._pair(
            git_repo, tmp_path, monkeypatch, db_name='gate-row8-pair.db',
        )
        worker = scene.worker

        # 0 — the pair really was deep, and deep at the depths the policy picks.
        queue_len = 1 + _ROW8_FOLLOWERS
        expected = [
            select_chain_depth(_ROW8_CAP, queue_len, None),
            select_chain_depth(
                _ROW8_CAP, queue_len, next_halving_state(False, _ROW8_CAP),
            ),
        ]
        assert scene.depths == expected, (
            f'the pair must chain at the policy-selected depths {expected!r}; '
            f'got {scene.depths!r}'
        )
        assert all(d is not None and d >= 4 for d in scene.depths), (
            f'both rounds must be >= 4 deep — a round that halved into the '
            f'``< 2`` floor runs no chain code and its silence proves nothing; '
            f'got {scene.depths!r}'
        )

        # 1 — no blocked outcome for ANY item, on either round.
        for rec in scene.rounds:
            assert rec['outcome'] is None, (
                f'round {rec["round"]} rendered an outcome for its dispatching '
                f'request: {rec["outcome"]!r}'
            )
        resolved = {
            tid: req.result for tid, req in scene.reqs.items() if req.result.done()
        }
        assert resolved == {}, (
            f'every chained request stays pending across a red pair; these '
            f'resolved: {sorted(resolved, key=int)!r}'
        )

        # 2 — the durable tier is silent for every task in the queue.
        for tid in scene.reqs:
            assert _events_for_task(scene.db_path, tid) == ['merge_queued'], (
                f'task {tid} emitted more than its own enqueue event: '
                f'{_events_for_task(scene.db_path, tid)!r}'
            )
        attempts = _rows_of_type(scene.db_path, EventType.merge_attempt)
        assert attempts == [], (
            f'a red deep tip must write no merge_attempt row — that row is '
            f'what a rendered failure looks like on the durable tier, and it '
            f'carries the very (category, cause_hint) pair the thrash ladder '
            f'keys on; got {attempts!r}'
        )
        assert _finalized_rows(scene.db_path) == [], (
            f'nothing landed, so nothing may be finalized; got '
            f'{_finalized_rows(scene.db_path)!r}'
        )

        # 3 — main never moved, on either round.
        for rec in scene.rounds:
            assert rec['main_after'] == rec['main_before'], (
                f'main moved on red round {rec["round"]}: '
                f'{rec["main_before"][:8]} -> {rec["main_after"][:8]}'
            )
            assert rec['advanced'] is False, (
                f'round {rec["round"]} advanced on a red tip: {rec["advanced"]!r}'
            )

        # 4 — every follower is still at its ORIGINAL lane index.  Order, not
        # membership: a chain rebuild that reordered the buffer would leave the
        # next round chaining a different prefix while every membership
        # assertion above still passed.
        assert [r.task_id for r in worker._lane_buffers['normal']] == list(
            _gate_followers(_ROW8_FOLLOWERS)
        ), (
            f'the followers left their submission order: '
            f'{[r.task_id for r in worker._lane_buffers["normal"]]!r}'
        )
        assert list(worker._lane_buffers['high']) == [], (
            f'nothing was ever enqueued high; got '
            f'{[r.task_id for r in worker._lane_buffers["high"]]!r}'
        )

    async def test_the_pair_leaves_the_shipped_thrash_ladder_byte_identical(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """LEDGER SIDE — the conclusion, folded through the REAL ladder.

        The pair's own observable outputs are handed to
        ``workflow._evaluate_merge_thrash``.  It has nothing to eat, because the
        ladder is fed ONLY from a rendered blocked ``MergeOutcome`` (workflow.py
        stashes ``result.reason`` into ``_last_merge_block_reason`` and gates
        the whole thrash check on it being non-None), and a deep red tip
        REQUEUES without rendering one.

        Asserted from BOTH a virgin ledger and a MID-RUN one.  A mid-run ledger
        — a task that genuinely blocked sequentially earlier — is the case that
        actually matters: it already carries a signature, so a deep fail that
        leaked ANY signature would either increment the counter (identical
        signature) or reset it to 1 (a different one), and both are visible
        only against a non-zero starting point.
        """
        scene = await self._pair(
            git_repo, tmp_path, monkeypatch, db_name='gate-row8-ladder.db',
        )

        virgin = RetryLedger()
        after_virgin = _ladder_after(
            scene.rounds, ledger=virgin, requests=list(scene.reqs.values()),
        )
        assert after_virgin.consecutive_merge_thrash == 0, (
            f'a red deep pair fed the ladder a signature from nothing: counter '
            f'{virgin.consecutive_merge_thrash!r} -> '
            f'{after_virgin.consecutive_merge_thrash!r}'
        )
        assert after_virgin.last_merge_outcome_signature is None, (
            f'a red deep pair rendered a signature: '
            f'{after_virgin.last_merge_outcome_signature!r}'
        )

        mid_run = RetryLedger(
            consecutive_merge_thrash=1,
            last_merge_outcome_signature='sig-from-an-earlier-sequential-block',
        )
        after_mid = _ladder_after(
            scene.rounds, ledger=mid_run, requests=list(scene.reqs.values()),
        )
        assert (
            after_mid.consecutive_merge_thrash == mid_run.consecutive_merge_thrash
        ), (
            f'the deep pair moved a mid-run counter: '
            f'{mid_run.consecutive_merge_thrash!r} -> '
            f'{after_mid.consecutive_merge_thrash!r}'
        )
        assert (
            after_mid.last_merge_outcome_signature
            == mid_run.last_merge_outcome_signature
        ), (
            f'the deep pair overwrote a mid-run signature: '
            f'{mid_run.last_merge_outcome_signature!r} -> '
            f'{after_mid.last_merge_outcome_signature!r}'
        )

    async def test_the_pair_is_conserved_and_two_way_quiescent(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """The oracle, over the pair, with the ladder as its CAS/ledger half.

        The residue is retired explicitly and CHECKED against what the pair
        promised to leave behind before the oracle runs, so a follower that
        wrongly landed shows up as a residue mismatch rather than being quietly
        drained away.  The head is retired separately because a red tip parks it
        on ``_queue``, which ``_drain_residue`` (a LANE-buffer drain) cannot
        reach.
        """
        scene = await self._pair(
            git_repo, tmp_path, monkeypatch, db_name='gate-row8-quiesce.db',
        )
        worker = scene.worker
        main_sha = scene.rounds[-1]['main_after']
        permits_before = _permit_census(worker)

        head = scene.reqs['101']
        assert not head.result.done(), (
            'the head must still be unresolved — it is requeued, not blocked'
        )
        head.result.cancel()
        worker._retire_item(head.request_id)

        residue = _drain_residue(worker)
        assert residue == set(_gate_followers(_ROW8_FOLLOWERS)), (
            f'the pair promised to leave every follower buffered and nothing '
            f'else; drained {sorted(residue, key=int)!r}'
        )

        before = RetryLedger()
        _assert_two_way_quiescent(
            worker, main_sha, list(scene.reqs.values()),
            permits_before=permits_before,
            ladder={
                'before': before,
                'after': _ladder_after(
                    scene.rounds, ledger=before,
                    requests=list(scene.reqs.values()),
                ),
            },
        )


class TestLadderDriverIsNotInert:
    """The inertness control for :func:`_ladder_after`, and its keying.

    Sync by design: the control is about the FOLD, not about the pipeline, so
    it fabricates the round records rather than driving a scene.  (Note this
    class carries no ``@pytest.mark.asyncio`` — pytest-asyncio is STRICT here,
    and a sync test inside a class-level ``asyncio`` mark is a hard ERROR.)
    """

    def _blocked_round(
        self, *, reason: str, category: str = 'gui_tsc',
        cause_hint: str = 'src/App.tsx: TS2322',
    ) -> dict:
        """A round record whose dispatching request RESOLVED blocked."""
        from orchestrator.merge_types import MergeOutcome

        return {
            'round': 1,
            'tag': 'control',
            'outcome': MergeOutcome(
                status='blocked', reason=reason,
                failure_category=category, failure_cause_hint=cause_hint,
            ),
        }

    def test_a_repeated_blocked_signature_reaches_the_escalation_threshold(
        self,
    ) -> None:
        """Without this, "unmoved" would prove only that the driver is inert.

        Two rounds carrying the SAME rendered failure is precisely 3003's
        signature class, and ``max_consecutive_merge_thrash`` defaults to 2, so
        the counter must arrive exactly AT the threshold — not one short of it
        (which would mean the second round was dropped) and not past it (which
        would mean a round was folded twice).
        """
        rounds = [
            self._blocked_round(reason='post-merge verification failed'),
            self._blocked_round(reason='post-merge verification failed'),
        ]
        after = _ladder_after(rounds, ledger=RetryLedger())

        assert after.consecutive_merge_thrash == _MERGE_THRASH_THRESHOLD, (
            f'two identical rendered failures must reach the threshold '
            f'{_MERGE_THRASH_THRESHOLD}; got '
            f'{after.consecutive_merge_thrash!r} — the driver folded nothing, '
            f'so every "unmoved" assertion in this row is vacuous'
        )
        assert after.last_merge_outcome_signature == _blocked_signature(
            rounds[-1]['outcome'],
        ), (
            f'the ladder must key on the SHIPPED signature function; got '
            f'{after.last_merge_outcome_signature!r}'
        )

    def test_two_different_blocked_signatures_reset_rather_than_accumulate(
        self,
    ) -> None:
        """The driver must reproduce the ladder's RESET arm too.

        A driver that only ever incremented would report thrash for a task
        making genuine progress between blocks, and would still pass the
        threshold control above.
        """
        rounds = [
            self._blocked_round(reason='a', cause_hint='src/App.tsx: TS2322'),
            self._blocked_round(reason='b', cause_hint='src/Other.tsx: TS7006'),
        ]
        after = _ladder_after(rounds, ledger=RetryLedger())

        assert after.consecutive_merge_thrash == 1, (
            f'a DIFFERING signature resets the counter to 1 (one occurrence of '
            f'something new observed); got {after.consecutive_merge_thrash!r}'
        )

    def test_a_round_that_landed_feeds_the_ladder_nothing(self) -> None:
        """Only 'blocked' feeds the ladder — a DONE outcome is not a failure.

        The row's silence claim would be trivially true if the driver ignored
        every outcome, so it must be shown to ignore exactly the right ones.
        """
        from orchestrator.merge_types import MergeOutcome

        landed = {
            'round': 1, 'tag': 'control',
            'outcome': MergeOutcome(status='done', merge_sha='deadbeef'),
        }
        after = _ladder_after([landed], ledger=RetryLedger())

        assert after.consecutive_merge_thrash == 0, (
            f'a landed round is not a merge failure and must not feed the '
            f'ladder; got {after.consecutive_merge_thrash!r}'
        )


# ── the ROUND-SEQUENCE transcript extractor (step-12) ────────────────────────


def _gate_round_transcript(scene: _GateScene, idx: int) -> dict:
    """Normalise ONE round into a repo-independent, comparable dict.

    test_merge_queue_deep_landing.py::_delta_round_transcript widened for a
    SEQUENCE golden.  EXTRACTOR ONLY — every value is read back off facts
    ``_GateScene`` already recorded, or off the durable tier; nothing here
    re-drives the round, so the golden compares ONE run rather than two.

    KEY SET, alphabetical and explicit (adding a field must force a visible
    golden edit, never a silent widening of the dict):

      ``advanced``          — did ``_finalize_inflight`` advance main.
      ``build_chain_calls`` — ``build_chain`` invocations in THIS round.
      ``chain``             — ``None``, or the chained task ids in link order.
      ``chain_build_ms``    — normalised: ``None``, or ``'<stamped>'`` for any
                              real cost.  The raw ms is a wall-clock reading and
                              can never appear in a golden; what the row asserts
                              is presence/absence.
      ``chain_items``       — off the round's ``merge_verify`` row, i.e. the
                              value η1's reader actually sees.  NOT the dispatch
                              kwarg, which is a literal floor of 1 on every path
                              and would make the field vacuous.
      ``events``            — the DISPATCHING task's ordered event-type stream.
      ``halving_state``     — ``_chain_halving_state`` after the round.
      ``landed_via_chain``  — off the head's ``merge_finalized`` row.
      ``main_moved``        — did main's sha change across the round.
      ``outcome_status``    — the head request's rendered ``MergeOutcome``.
      ``probe_base``        — normalised to ``'<sha>'``; the raw sha is
                              repo-dependent.
      ``result_has_outcome``/``result_has_worktree``/``result_status`` — the
                              ``InflightVerifyResult`` triple.
      ``spec_lane_acquisitions`` — ``acquire_spec_lane`` calls in THIS round.
      ``verified_the_items_own_merge_commit`` — did the verify run against the
                              item's OWN merge commit (the floor path) rather
                              than a chain tip.
    """
    rec = scene.rounds[idx]
    item = rec['item']
    head_tid = rec['req'].task_id
    chain = rec['chain']
    verify = _rows_of_type(scene.db_path, EventType.merge_verify, task_id=head_tid)
    finalized = _rows_of_type(
        scene.db_path, EventType.merge_finalized, task_id=head_tid,
    )
    build_ms = verify[0].get('chain_build_ms') if verify else None
    outcome = rec['outcome']
    return {
        'advanced': rec['advanced'],
        'build_chain_calls': len(rec['built']),
        'chain': None if chain is None else [tid for tid, _ in chain.links],
        'chain_build_ms': None if build_ms is None else '<stamped>',
        'chain_items': verify[0]['chain_items'] if verify else None,
        'events': _events_for_task(scene.db_path, head_tid),
        'halving_state': rec['halving_state'],
        'landed_via_chain': finalized[0].get('landed_via_chain') if finalized else None,
        'main_moved': rec['main_after'] != rec['main_before'],
        'outcome_status': None if outcome is None else outcome.status,
        'probe_base': None if rec.get('probe_base') is None else '<sha>',
        'result_has_outcome': rec['result'].outcome is not None,
        'result_has_worktree': rec['result'].merge_wt is not None,
        'result_status': None if rec['result'].status is None else rec['result'].status.value,
        'spec_lane_acquisitions': len(rec.get('lane_acquires', ())),
        'verified_the_items_own_merge_commit':
            bool(verify) and verify[0]['merge_sha'] == item.merge_result.merge_commit,
    }


def _gate_sequence_transcript(scene: _GateScene) -> list[dict]:
    """Every round of *scene*, in order, as :func:`_gate_round_transcript` dicts."""
    return [_gate_round_transcript(scene, i) for i in range(len(scene.rounds))]


# ═══════════════════════════════════════════════════════════════════════════
# -- step-11 RED: Row 7 — KILL-SWITCH BYTE-IDENTITY, over a ROUND SEQUENCE --
#
# Both upstream forms of this row compare ONE round.
# test_merge_queue_deep_dispatch.py compares a single-round transcript against
# the same round re-run; test_merge_queue_deep_landing.py compares a single
# round against a golden dict literal.  The PRD's claim is bigger than either:
# at cap=0 "dispatch/behaviour is identical to the pre-PRD golden transcript",
# which is a claim about a RUN.
#
# Why a SEQUENCE and not one round.  Every piece of deep state that could leak
# into the kill-switched path is state that survives ACROSS rounds —
# ``_chain_halving_state``, the ``_spec-`` lane pool, the speculation permit
# ledger, ``_n_failed`` / ``_remerge_occurred``.  A one-round comparison cannot
# see any of it, because a single round has no previous round to inherit from.
# The three rounds are deliberately MIXED — a pass, a fail, a pass — since the
# fail is the only round that could arm halving state at all, and the pass
# after it is the only round that could then observe it.
#
# Why a GOLDEN LITERAL and not a second run.  The claim is about ABSENCES, and
# a literal names each one: chain None, chain_build_ms None, build_chain_calls
# 0, spec_lane_acquisitions 0, halving_state None, landed_via_chain None,
# chain_items 1, verified_the_items_own_merge_commit True.  A differential
# comparison against a re-run would go green for two runs that were identically
# WRONG.
#
# Why the POSITIVE CONTROL is mandatory.  A golden of absences passes trivially
# over an inert scene.  The SAME script over the SAME seeded queue at cap=6
# must produce a transcript that differs in every one of those fields — which
# is what proves the cap=0 transcript is a kill switch and not a broken fixture.
#
# One deliberate scene choice, and it is NOT about the deep path: the remote
# cross-check of a remote green (``verify_cross_check_remote_green``, default
# True) runs a full LOCAL trust-anchor suite on the merge worktree, and whether
# that produces ``verdict_parity_ok``, ``verify_cross_check_inconclusive`` or
# nothing at all varies run to run in a fixture repo.  Measured: the same
# scene emitted ``verdict_parity_ok`` on one run and not on the next.  That
# variance is unrelated to the kill switch and would make an event-stream
# golden flaky, so the row turns the cross-check OFF and pins a stream that was
# then measured identical across three consecutive runs.
# ═══════════════════════════════════════════════════════════════════════════

_ROW7_FOLLOWERS = 7
"""Eight queued items, so BOTH arms of the row get three real rounds.

At cap=0 the three heads are 101/102/103 and the rest stay queued; at cap=6 the
first round chains six and lands them, leaving 107 and 108 for rounds 2 and 3.
Fewer followers and the cap=6 control would run out of queue after one round,
which would make "the same sequence" a comparison of a 3-round run against a
1-round one — a difference proving nothing about the kill switch."""

_ROW7_SCRIPT = [True, False, True]
"""Pass, FAIL, pass.  The fail is the only round that can arm halving state,
and the pass after it is the only round that can observe it having leaked."""

_ROW7_GOLDEN: list[dict] = [
    # ── round 1 — head 101, GREEN.  Lands by the ordinary adjacent path. ──
    {
        'advanced': True,
        'build_chain_calls': 0,          # no chain was ever built
        'chain': None,                   # ...so none was handed to the verify
        'chain_build_ms': None,          # ...and none was costed for the reader
        'chain_items': 1,                # η1 sees a ONE-item verify
        'events': [
            'merge_queued', 'merge_verify', 'merge_attempt', 'merge_finalized',
        ],
        'halving_state': None,           # nothing to bisect
        'landed_via_chain': None,        # landed by advance_main, not by a walk
        'main_moved': True,
        'outcome_status': 'done',
        'probe_base': None,
        'result_has_outcome': False,     # a green verify renders no MergeOutcome
        'result_has_worktree': True,     # the floor path returns the item's own
        'result_status': None,           # no InflightStatus sentinel
        'spec_lane_acquisitions': 0,     # no scratch lane was ever claimed
        'verified_the_items_own_merge_commit': True,
    },
    # ── round 2 — head 102, RED.  Blocks through the ordinary path. ───────
    #
    # The absence of `merge_attempt` here is not a deep fact: a blocked
    # post-merge verify on the ordinary path writes merge_finalized and no
    # attempt row.  It is pinned because the cap=6 control's red round emits a
    # DIFFERENT stream again (no merge_finalized at all — the chain arm
    # requeues rather than blocking), and that contrast is the row.
    {
        'advanced': False,
        'build_chain_calls': 0,
        'chain': None,
        'chain_build_ms': None,
        'chain_items': 1,
        'events': ['merge_queued', 'merge_verify', 'merge_finalized'],
        'halving_state': None,           # THE round that would arm it, if deep
        'landed_via_chain': None,
        'main_moved': False,
        'outcome_status': 'blocked',     # a named culprit — the ordinary verdict
        'probe_base': None,
        'result_has_outcome': True,
        'result_has_worktree': True,
        'result_status': None,
        'spec_lane_acquisitions': 0,
        'verified_the_items_own_merge_commit': True,
    },
    # ── round 3 — head 103, GREEN.  Byte-identical to round 1. ────────────
    {
        'advanced': True,
        'build_chain_calls': 0,
        'chain': None,
        'chain_build_ms': None,
        'chain_items': 1,
        'events': [
            'merge_queued', 'merge_verify', 'merge_attempt', 'merge_finalized',
        ],
        'halving_state': None,           # the red round left NOTHING behind
        'landed_via_chain': None,
        'main_moved': True,
        'outcome_status': 'done',
        'probe_base': None,
        'result_has_outcome': False,
        'result_has_worktree': True,
        'result_status': None,
        'spec_lane_acquisitions': 0,
        'verified_the_items_own_merge_commit': True,
    },
]
"""The pre-PRD transcript, stated as a literal so every absence is NAMED.

Recorded from a measured run and then justified field by field above; the
positive control below is what makes the recording meaningful rather than
circular.
"""

_ROW7_DEEP_FIELDS = (
    'build_chain_calls',
    'chain',
    'chain_build_ms',
    'chain_items',
    'landed_via_chain',
    'spec_lane_acquisitions',
    'verified_the_items_own_merge_commit',
)
"""Every transcript field the deep path can move.  The control asserts ALL of
them differ at cap=6 — a control that only checked one could pass while the
other six were silently inert."""


@pytest.mark.asyncio
@pytest.mark.timeout(300)
class TestRow7KillSwitchByteIdentity:
    """Row 7: at cap=0 a whole RUN is byte-identical to the pre-PRD transcript."""

    async def _sequence(
        self, git_repo: Path, tmp_path: Path, monkeypatch, *,
        chain_cap: int, db_name: str,
    ) -> _GateScene:
        """Drive the fixed 3-round mixed sequence over one seeded queue.

        ``remote=True`` so the REAL ``_run_post_merge_verify`` runs and the
        ``merge_verify`` row is genuinely emitted — ``chain_items`` and
        ``chain_build_ms`` are produced strictly below that call, so a scene
        that stubbed it out could not state their absence at all.
        """
        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=chain_cap, n_followers=_ROW7_FOLLOWERS, db_name=db_name,
            remote=True, script=list(_ROW7_SCRIPT),
        )
        # See the section banner: the remote cross-check is run-to-run
        # nondeterministic in a fixture repo and has nothing to do with the
        # kill switch.  Mutating the live config object is what an operator
        # reload does too (OrchestratorConfig is a plain mutable BaseModel), and
        # every enqueued request holds THIS object by reference.
        scene.config.verify_cross_check_remote_green = False
        for tag in ('r1', 'r2', 'r3'):
            nxt = scene.worker._pop_next_pickable()
            assert nxt is not None, (
                f'the queue ran dry before round {tag}: a 3-round claim needs '
                f'three real rounds, not a short run that passes vacuously'
            )
            await scene.round_(tag=tag, head_tid=nxt.task_id, req=nxt)
        return scene

    async def test_the_kill_switched_run_matches_the_golden_transcript(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """The whole three-round sequence, compared against the literal.

        Compared as ONE equality on the whole list rather than field by field,
        so a field that stopped being extracted at all fails here instead of
        silently dropping out of the comparison.
        """
        scene = await self._sequence(
            git_repo, tmp_path, monkeypatch,
            chain_cap=0, db_name='gate-row7-golden.db',
        )
        transcript = _gate_sequence_transcript(scene)

        assert len(transcript) == 3, (
            f'the golden is a THREE-round claim; got {len(transcript)} rounds'
        )
        assert transcript == _ROW7_GOLDEN, (
            'the kill-switched run diverged from the pre-PRD golden.\n'
            + '\n'.join(
                f'  round {i + 1} {key}: golden {_ROW7_GOLDEN[i][key]!r} != '
                f'observed {got.get(key, "<MISSING>")!r}'
                for i, got in enumerate(transcript)
                for key in sorted(set(_ROW7_GOLDEN[i]) | set(got))
                if _ROW7_GOLDEN[i].get(key, '<MISSING>') != got.get(key, '<MISSING>')
            )
        )

    async def test_the_same_sequence_at_cap_six_moves_every_deep_field(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """POSITIVE CONTROL — without it the golden could pass over an inert scene.

        The same script over the same seeded queue, with only ``chain_cap``
        changed.  Every field the deep path can move must actually move on the
        very first round, and the whole sequence must differ.
        """
        scene = await self._sequence(
            git_repo, tmp_path, monkeypatch,
            chain_cap=6, db_name='gate-row7-control.db',
        )
        transcript = _gate_sequence_transcript(scene)

        assert transcript != _ROW7_GOLDEN, (
            'cap=6 produced the kill-switched transcript — the scene is inert '
            'and the golden above proves nothing'
        )
        for field in _ROW7_DEEP_FIELDS:
            assert transcript[0][field] != _ROW7_GOLDEN[0][field], (
                f'round 1 field {field!r} did not move at cap=6: still '
                f'{transcript[0][field]!r}'
            )
        # The red round diverges STRUCTURALLY, not just numerically: the chain
        # arm requeues without naming a culprit, so it renders no outcome and
        # writes no merge_finalized at all.
        assert transcript[1]['outcome_status'] is None, (
            f'a red deep tip names no culprit; got '
            f'{transcript[1]["outcome_status"]!r}'
        )
        assert 'merge_finalized' not in transcript[1]['events'], (
            f'a requeued deep round finalizes nothing; got '
            f'{transcript[1]["events"]!r}'
        )
        assert transcript[1]['halving_state'] == 1, (
            f'the red deep round must ARM the bisector (this is exactly what '
            f'the golden proves cap=0 never does); got '
            f'{transcript[1]["halving_state"]!r}'
        )

    async def test_a_restarted_worker_inherits_no_halving_suspicion(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """The golden survives a worker restart, for a STRUCTURAL reason.

        ``halving_state: None`` appears in all three golden rounds, and a
        reader could reasonably suspect that is an artifact of THIS worker
        never having been deep.  It is not: a freshly constructed worker starts
        at ``None``, and at cap=0 ``select_chain_depth`` short-circuits on the
        cap BEFORE halving state is consulted at all — so no state a previous
        process could have left behind can change the transcript.
        """
        from orchestrator.merge_queue import select_chain_depth

        scene = await self._sequence(
            git_repo, tmp_path, monkeypatch,
            chain_cap=0, db_name='gate-row7-restart.db',
        )
        assert scene.worker._chain_halving_state is None, (
            f'the kill-switched run armed the bisector: '
            f'{scene.worker._chain_halving_state!r}'
        )

        restarted = _make_worker(scene.git_ops)
        assert restarted._chain_halving_state is None, (
            f'a fresh worker must start with no suspicion; got '
            f'{restarted._chain_halving_state!r}'
        )
        assert restarted._n_failed is False and restarted._remerge_occurred is False

        # ...and the cap short-circuit makes the point unconditional: NO
        # halving state, inherited or otherwise, can produce a chain at cap=0.
        for queue_len in (1, 2, 8, 32):
            for state in (None, 1, 2, 4, 16):
                assert select_chain_depth(0, queue_len, state) is None, (
                    f'select_chain_depth(0, {queue_len}, {state}) returned '
                    f'{select_chain_depth(0, queue_len, state)!r} — the kill '
                    f'switch must be unconditional'
                )


# ═══════════════════════════════════════════════════════════════════════════
# -- step-13 RED: THE COMPOSITION SWEEP — rows 1, 2, 5, 6, 9 and 10 --
#
# These six rows are the ones the module's SUBSUMPTION table marks FULLY
# SUBSUMED: every one of them already has a unit-level owner in
# test_merge_queue_deep_landing.py, and re-cloning those units here would buy
# nothing but a second copy that rots in step with the first.
#
# What is NOT covered upstream is the COMPOSITION.  Every upstream owner is a
# single-round scene built by a purpose-shaped fixture, torn down before the
# next assertion:
#
#   * ``TestDeepLandingEndToEnd`` drives at most TWO rounds, and its
#     ``_assert_quiescent`` calls are all terminal — nothing samples the
#     conservation surfaces BETWEEN rounds;
#   * ``TestHeadCancelOnAdoption`` / ``TestHeadCancelLeavesTheLaneIdle`` build
#     the two-slot topology by hand and stop at ``_run_inflight_verify`` — the
#     head is never carried through a finalize INSIDE a scene that also has a
#     queue, a permit census and a lane pool to conserve;
#   * ``TestStaleCasAbortLeavesTheRestAlone`` asserts the abort leaves the rest
#     alone, but never runs the NEXT round that has to rebuild on the moved
#     main — which is the half of decision #9 an operator actually feels.
#
# So each test below states its row's claim ONCE, inside a continuous run over
# ONE worker, ONE queue, ONE repo and ONE permit census, and pairs it with the
# conservation oracle: :func:`_assert_midrun_conserved` after every round that
# leaves the pipeline deliberately mid-flight, and the full
# :func:`_assert_two_way_quiescent` once :func:`_drain_residue` has confirmed
# the run left exactly the residue it promised.
#
# WHY THE MID-RUN ORACLE IS A SUBSET, restated here because it is the one
# design choice a reader will question: surfaces (a), (d) and (f) of the full
# oracle are WHOLE-REGISTRY claims (every request resolved, the liveness
# ledger empty, no non-terminal lifecycle entry), and a run that is
# deliberately mid-flight has queued items with pending futures at every
# checkpoint.  Asserting them between rounds would either fail honestly or
# force an empty request list — the exact vacuum the oracle's guard clauses
# refuse.  Conservation, by contrast, holds at EVERY instant, so that is what
# the between-rounds oracle checks.
# ═══════════════════════════════════════════════════════════════════════════

_COMPOSED_CAP = 4
"""The cap rows 1, 2 and 6 share.

Four is the smallest cap that makes all three rows non-degenerate at once: a
4-item chain has a head plus THREE links, so row 6's abort can land a proper
prefix (head + 1) and still leave two unlanded links behind, and row 2's
halving step is ``4 -> 2`` rather than straight to the floor."""

_COMPOSED_FOLLOWERS = 5
"""Five followers, so the cap BINDS rather than the queue running out.

With six queued items and a cap of four, a short chain is the code's doing and
never the fixture's — and rows 1 and 6 are both left with genuine residue,
which is what makes their ``_drain_residue`` assertions load-bearing."""

_HEAD_CAP = 6
"""Rows 5 and 9 chain the WHOLE queue behind the adopted head.

Their claim is about the head — that a red one does not block the prefix, and
that its lease is released — so the prefix must be long enough that "the whole
prefix landed" is a real statement.  Cap 6 over four followers means nothing
truncates and nothing is left over: the chain is exactly the queue."""


@pytest.mark.asyncio
@pytest.mark.timeout(300)
class TestBoundaryRowsComposed:
    """Rows 1, 2, 5, 6, 9 and 10, each inside ONE continuous multi-round run."""

    async def test_row_1_a_green_tip_lands_the_whole_prefix_in_order(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """Row 1: ONE verify pays for a 4-item prefix, and the canary says so.

        The PRD's headline row.  Upstream owns the landing itself; what is
        added here is the READER's view of it — the claim is checked through
        ``scripts/merge-deep-canary-predicate.sh``'s own arithmetic
        (:func:`_canary_predicate_items_per`) over the REAL durable rows, so a
        stamp that landed correctly but encoded the wrong unit still fails.

        ``landed_via_chain`` ships as ONE PER LANDED ITEM, never as the chain
        size k — so the PRD's "landed_via_chain=4" is the SUM over the walk's
        four ``merge_finalized`` rows, which is exactly what the shipped
        predicate computes.  Stated both ways below (the sum, and the ratio the
        predicate derives from it) because they fail differently: a k-per-item
        stamp would give a sum of 16 and a ratio of 16.0, while a positional
        stamp would give 10 and 10.0.
        """
        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=_COMPOSED_CAP, n_followers=_COMPOSED_FOLLOWERS,
            db_name='gate-row1-prefix.db', remote=True, script=[True],
        )
        # The remote cross-check of a remote green is run-to-run
        # nondeterministic in a fixture repo (measured; see Row 7's banner) and
        # has nothing to do with this row's claim, so it is off.
        scene.config.verify_cross_check_remote_green = False
        worker = scene.worker
        permits_before = _permit_census(worker)

        rec = await scene.round_(tag='row1', head_tid='101')

        chain = rec['chain']
        assert chain is not None and len(chain.links) == 3, (
            f'cap={_COMPOSED_CAP} must bind into a 4-item chain; got '
            f'{None if chain is None else 1 + len(chain.links)} items'
        )

        # ── ONE verify for the whole prefix ─────────────────────────────────
        verify_rows = _verify_rows(scene.db_path)
        assert len(verify_rows) == 1, (
            f'a green deep round pays for exactly ONE verify — that is the '
            f'whole PRD — got {len(verify_rows)}: {verify_rows!r}'
        )
        assert verify_rows[0]['chain_items'] == 4, (
            f'the reader must see a FOUR-item verify; got '
            f'{verify_rows[0]["chain_items"]!r}'
        )

        # ── four in-order landings, through the real CAS walk ───────────────
        landed = ['101', *[tid for tid, _mc in chain.links]]
        assert [c[0] for c in rec['advance_calls']] == [
            rec['head_mc'], *[mc for _tid, mc in chain.links],
        ], (
            f'advance_main must run once per chain item, in LAND order; got '
            f'{[c[0][:8] for c in rec["advance_calls"]]!r}'
        )
        assert rec['landed'] == landed, (
            f'expected {landed!r} to land; got {rec["landed"]!r}'
        )
        for tid in landed:
            outcome = scene.reqs[tid].result.result()
            assert outcome.status == 'done', f'task {tid} did not land: {outcome!r}'

        # ── main's first-parent history is LINEAR, in land order ────────────
        _rc, out, _err = await _run(
            ['git', 'rev-list', '--first-parent', 'main'], cwd=git_repo,
        )
        expected = [rec['head_mc'], *[mc for _tid, mc in chain.links]]
        assert out.split()[:len(expected)] == list(reversed(expected)), (
            "main's first-parent history must be exactly the land order"
        )

        # ── the SHIPPED canary's arithmetic, over the REAL rows ─────────────
        finalized = _finalized_rows(scene.db_path)
        assert len(finalized) == 4, (
            f'four items landed, so four merge_finalized rows; got '
            f'{len(finalized)}'
        )
        assert sum(row['landed_via_chain'] for row in finalized) == 4, (
            f'landed_via_chain is ONE PER LANDED ITEM, so a 4-item walk sums '
            f'to 4; got {[row["landed_via_chain"] for row in finalized]!r}'
        )
        assert _canary_predicate_items_per(verify_rows, finalized) == 4.0, (
            f'the shipped predicate must read 4 items landed per deep verify; '
            f'got {_canary_predicate_items_per(verify_rows, finalized)!r}'
        )

        # ── conservation, then the full two-way oracle ──────────────────────
        _assert_midrun_conserved(
            worker, rec['main_after'], permits_before, where='after row 1',
        )
        assert _drain_residue(worker) == {'105', '106'}, (
            'the two items past the cap must still be queued, untouched'
        )
        _assert_two_way_quiescent(
            worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()), permits_before=permits_before,
        )

    async def test_row_2_a_red_tip_leaves_the_queue_intact_and_lands_later(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """Row 2: a red tip is a NON-EVENT, and the same requests land later.

        The composition this adds over the upstream owner is the SECOND round:
        upstream re-dispatches the requeued head and stops, while this run
        carries the halving state the red round armed into the next round's
        depth selection and then drains the whole scene through the two-way
        oracle.  A red tip that quietly consumed a permit, or that left the
        bisector armed after landing, is invisible to a one-round scene.
        """
        from orchestrator.merge_queue import next_halving_state, select_chain_depth

        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=_COMPOSED_CAP, n_followers=_COMPOSED_FOLLOWERS,
            db_name='gate-row2-intact.db', script=[False, True],
        )
        worker = scene.worker
        followers = list(_gate_followers(_COMPOSED_FOLLOWERS))
        permits_before = _permit_census(worker)

        red = await scene.round_(tag='row2', head_tid='101')

        assert red['chain'] is not None and len(red['chain'].links) == 3, (
            f'the red round must really have been deep; got '
            f'{None if red["chain"] is None else 1 + len(red["chain"].links)}'
        )
        # ── zero landings via the chain, on BOTH tiers ──────────────────────
        assert red['main_after'] == red['main_before'], 'a red tip moved main'
        assert red['advanced'] is False, f'advanced on a red tip: {red!r}'
        assert red['landed'] == [], f'a red tip landed something: {red["landed"]!r}'
        assert _finalized_rows(scene.db_path) == [], (
            f'nothing landed, so nothing may be finalized; got '
            f'{_finalized_rows(scene.db_path)!r}'
        )

        # ── every item still queued, at its ORIGINAL lane index ─────────────
        assert [r.task_id for r in worker._lane_buffers['normal']] == followers, (
            f'the chain mutated the queue on a red tip: '
            f'{[r.task_id for r in worker._lane_buffers["normal"]]!r}'
        )
        for tid in followers:
            assert _lane_of(worker, tid) == 'normal', (
                f'task {tid} left its lane on a red tip'
            )
            assert not scene.reqs[tid].result.done(), (
                f'task {tid} was handed a verdict no verify produced'
            )

        # ── the bisector halved to 2, off the DISPATCHED depth ──────────────
        assert worker._chain_halving_state == 2, (
            f'a red 4-item tip must halve the ceiling to 2; got '
            f'{worker._chain_halving_state!r}'
        )
        assert worker._chain_halving_state == next_halving_state(
            False, _COMPOSED_CAP,
        ), 'the halving state must be the policy function\'s own answer'
        _assert_midrun_conserved(
            worker, red['main_after'], permits_before, where='after the red round',
        )

        # ── and the SAME requests land on a later round's own verdict ───────
        assert worker._queue.qsize() == 1, (
            f'the head must go back on _queue unresolved; qsize='
            f'{worker._queue.qsize()}'
        )
        requeued = worker._queue.get_nowait()
        assert requeued is scene.reqs['101'] and not requeued.result.done()

        green = await scene.round_(tag='row2', head_tid='101', req=requeued)
        assert scene.reqs['101'].result.result().status == 'done', (
            'the very same request must land on the next round'
        )
        assert green['main_after'] != green['main_before']
        # The armed ceiling really bound the second round's depth.
        assert green['chain'] is not None and len(green['chain'].links) == (
            select_chain_depth(_COMPOSED_CAP, 1 + len(followers), 2) - 1
        ), (
            f'the second round must chain at the HALVED depth; got '
            f'{None if green["chain"] is None else 1 + len(green["chain"].links)}'
        )
        chained = {tid for tid, _mc in green['chain'].links}
        for tid in chained:
            assert scene.reqs[tid].result.result().landed_via_chain == 1

        assert _drain_residue(worker) == set(followers) - chained
        _assert_two_way_quiescent(
            worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()), permits_before=permits_before,
        )

    async def test_row_5_a_red_head_verdict_never_blocks_a_green_prefix(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """Row 5: head verify RED (a flake), tip GREEN — the prefix still lands.

        A red head verdict is a statement about a SUBSET of the tree the green
        tip has strictly better evidence for.  Acting on it would fail an item
        the tip just proved good and strand the whole prefix behind it, so the
        red verdict resolves no blocked ``MergeOutcome`` at all.

        Composed rather than cloned: this runs the head through a REAL
        ``_finalize_inflight`` inside a scene that also has a queue, a permit
        census and a lane pool — the upstream owner stops at
        ``_run_inflight_verify`` — and it pins the head's landing as an
        ORDINARY advance (``landed_via_chain`` absent), which is what keeps the
        canary's ratio honest.
        """
        from orchestrator.merge_types import InflightVerifyResult, MergeOutcome

        def _red_head_task(_git_ops):
            """A head verify that ALREADY came back red, before δ looks at it."""
            async def _body():
                return InflightVerifyResult(
                    outcome=MergeOutcome('blocked', reason='head verify red (flake)'),
                    merge_wt=None,
                    spec_warm=False,
                )
            return asyncio.get_running_loop().create_task(_body())

        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=_HEAD_CAP, n_followers=4, db_name='gate-row5-headred.db',
            heads=('100', '101'), script=[True],
        )
        worker = scene.worker
        permits_before = _permit_census(worker)

        teardown = _spy_teardown_pair(worker, monkeypatch)
        head_entry = await scene.attach_head('100', task_factory=_red_head_task)
        await asyncio.sleep(0)   # let the red verdict actually land
        assert head_entry.verify_task.done(), (
            'the head must be RED before the tip is adopted, or this row is '
            'just the ordinary head-cancel row'
        )

        rec = await scene.adopting_round_(tag='row5', spec_tid='101')

        # ── the red verdict did not survive the green tip ───────────────────
        head_outcome = scene.reqs['100'].result.result()
        assert head_outcome.status == 'done', (
            f'a red head verdict must not survive a green tip; got '
            f'{head_outcome!r}'
        )
        chain = rec['chain']
        assert chain is not None and len(chain.links) == 4, (
            f'the whole remaining queue must chain behind the head — cap '
            f'{_HEAD_CAP} over four followers means nothing truncates and '
            f'nothing is left over; got '
            f'{None if chain is None else 1 + len(chain.links)} items'
        )
        assert await _rev_parse(git_repo, 'main') == chain.tip, (
            'the full prefix must land, up to and including the verified tip'
        )
        for tid in ('101', *[t for t, _mc in chain.links]):
            assert scene.reqs[tid].result.result().status == 'done', (
                f'task {tid} was stranded behind a red head'
            )
        assert len(scene.posted) == 1, (
            f'ONE verify paid for the head AND the prefix; got '
            f'{len(scene.posted)}'
        )

        # ── the head landed by the ORDINARY advance, not by the walk ────────
        by_branch = {row['branch']: row for row in _finalized_rows(scene.db_path)}
        assert by_branch['100']['landed_via_chain'] is None, (
            f'the head carries no chain of its own — stamping it would inflate '
            f"the canary's ratio by one every deep round; got "
            f'{by_branch["100"]["landed_via_chain"]!r}'
        )
        for tid in ('101', *[t for t, _mc in chain.links]):
            assert by_branch[tid]['landed_via_chain'] == 1

        # ── the DELIVERED-verdict arm tears NOTHING down, and must not ──────
        #
        # Measured, and it is a real distinction rather than an omission: the
        # adoption gates its whole teardown block on
        # ``not head.verify_task.done()``, and this head's verdict had already
        # arrived.  There is no in-flight verify to cancel, no lease to
        # reclaim early, and no worktree to re-publish — the entry's lease and
        # worktree stay on ``_finalize_inflight``'s ORDINARY release path.
        # What still happens is the part the row is about: ``chain_adopted``
        # is set BEFORE that branch, so the red verdict is DECLINED either way.
        #
        # The other arm — a head still verifying when the tip passes, which
        # DOES go through ``_teardown_verify_task`` then
        # ``_cancel_and_release_tracked`` — is asserted in Row 9, where the
        # cancel is the mechanism the lease-release claim rests on.  Splitting
        # them is not a convenience: one scene has ONE head, so a single test
        # cannot state both, and asserting the live-head chokepoint here would
        # be asserting a branch this scene provably does not enter.
        assert head_entry.chain_adopted is True, (
            'the tip must DECLINE the head\'s red verdict — that flag is set '
            'before the done/not-done split, so it holds on both arms'
        )
        assert teardown == [], (
            f'a head whose verdict already arrived has nothing to tear down; '
            f'got {teardown!r}'
        )

        assert _drain_residue(worker) == set(), (
            'cap 6 over four followers chains the whole queue — nothing may '
            'be left behind'
        )
        _assert_two_way_quiescent(
            worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()), permits_before=permits_before,
        )

    async def test_row_6_a_stale_cas_aborts_and_the_next_round_rebuilds(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """Row 6: main moves mid-walk → abort, then REBUILD on the new main.

        The upstream owner proves the abort itself: the walk stops at the first
        non-``'advanced'`` result, the landed prefix stays landed, and every
        unlanded link goes back to its exact lane index unresolved.  What it
        never does is run the next round — and "the unlanded items land on a
        later round, on the moved main" is the half an operator actually feels,
        because it is what makes decision #9 a DEFERRAL rather than a loss.
        """
        from orchestrator.merge_types import ItemLifecycleState

        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=_COMPOSED_CAP, n_followers=_COMPOSED_FOLLOWERS,
            db_name='gate-row6-stale.db', script=[True, True],
            advance_hook=_abort_hook_at(3, bump_repo=git_repo),
        )
        worker = scene.worker
        permits_before = _permit_census(worker)

        first = await scene.round_(tag='row6', head_tid='101')

        chain = first['chain']
        assert chain is not None and len(chain.links) == 3, (
            f'the aborting round must really have been deep; got '
            f'{None if chain is None else 1 + len(chain.links)}'
        )
        # ── the walk stopped at the FIRST non-'advanced' result ─────────────
        assert [c[0] for c in first['advance_calls']] == [
            first['head_mc'], chain.links[0][1], chain.links[1][1],
        ], (
            f'the walk must stop at the first failure and never skip ahead; '
            f'got {[c[0][:8] for c in first["advance_calls"]]!r}'
        )

        # ── the already-landed prefix STAYS landed ──────────────────────────
        for tid in ('101', '102'):
            assert scene.reqs[tid].result.result().status == 'done', (
                f'task {tid} was un-landed by the abort'
            )
        for sha in (first['head_mc'], chain.links[0][1]):
            rc, _out, _err = await _run(
                ['git', 'merge-base', '--is-ancestor', sha, 'main'], cwd=git_repo,
            )
            assert rc == 0, f'{sha[:8]} is no longer on main after the abort'

        # ── every unlanded link is back at its EXACT lane index, unresolved ─
        assert [r.task_id for r in worker._lane_buffers['normal']] == [
            '103', '104', '105', '106',
        ], (
            f'submission order must survive the abort — it is the next round\'s '
            f'chain_snapshot; got '
            f'{[r.task_id for r in worker._lane_buffers["normal"]]!r}'
        )
        for tid in ('103', '104', '105', '106'):
            req = scene.reqs[tid]
            assert not req.result.done(), f'task {tid} was handed a verdict'
            assert worker._lifecycle.current(req.request_id) is (
                ItemLifecycleState.LANE_BUFFERED
            ), f'task {tid} left LANE_BUFFERED'
        _assert_midrun_conserved(
            worker, first['main_after'], permits_before, where='after the abort',
        )

        # ── the NEXT round rebuilds on the MOVED main and lands them ────────
        second = await scene.round_(tag='row6', head_tid='103')
        assert second['main_before'] != first['main_before'], (
            'the second round must build against the EXTERNALLY moved main'
        )
        assert second['chain'] is not None and len(second['chain'].links) == 3, (
            f'the deferred items must rebuild into a full chain; got '
            f'{None if second["chain"] is None else 1 + len(second["chain"].links)}'
        )
        assert second['landed'] == [
            '101', '102', '103', '104', '105', '106',
        ], (
            f'every deferred item must land on the rebuild; got '
            f'{second["landed"]!r}'
        )

        assert _drain_residue(worker) == set(), (
            'the rebuild drains the queue — nothing may be left behind'
        )
        _assert_two_way_quiescent(
            worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()), permits_before=permits_before,
        )

    async def test_row_9_both_lease_axes_read_idle_within_one_round(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """Row 9: after the tip-pass head cancel, BOTH lease axes read IDLE.

        The PRD's postcondition names ``warm-lane-lock-guard.sh``, which is
        REIFY-side and does not exist in this repo; the equivalent in-process
        oracle is the two axes that script and
        ``GitOps._merge_verify_lease_active`` read between them — the kernel
        flock on the ``_merge-verify`` lane, and the fixed-key holder-pgid
        rendezvous file.

        The STAGING check is what makes this non-vacuous: the head's verify
        holds the REAL ``merge_verify_lease`` for the whole span, and both axes
        are measured BUSY before δ acts.  Without it the post-cancel assertions
        would pass over a lane nothing ever touched.
        """
        import os

        from orchestrator.verify_cancel import (
            lane_lock_holder_pids_strict,
            lane_lock_path,
            read_lock_holder_pgid,
        )

        started = asyncio.Event()

        def _leased_head_task(git_ops):
            # The scene hands its OWN GitOps to the factory, so the head's
            # verify holds the same `_merge-verify` lane lock a real local
            # verify would — no module-attribute patching needed.
            return _gate_head_verify_task(
                lease_ctx=git_ops.merge_verify_lease(), started=started,
            )

        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=_HEAD_CAP, n_followers=4, db_name='gate-row9-lease.db',
            heads=('100', '101'), script=[True],
        )
        worker, git_ops = scene.worker, scene.git_ops
        permits_before = _permit_census(worker)

        teardown = _spy_teardown_pair(worker, monkeypatch)
        head_entry = await scene.attach_head('100', task_factory=_leased_head_task)
        await asyncio.wait_for(started.wait(), timeout=30)

        lock_path = lane_lock_path(git_ops.persistent_merge_worktree_path)
        # ── STAGING: the lane really IS busy, on BOTH axes ──────────────────
        assert git_ops._merge_verify_lease_active() is True, (
            'the head must genuinely hold the lease, or the idle assertions '
            'below are vacuous'
        )
        assert os.getpid() in lane_lock_holder_pids_strict(lock_path), (
            f'the kernel flock axis must read BUSY before the cancel; got '
            f'{lane_lock_holder_pids_strict(lock_path)!r}'
        )

        rec = await scene.adopting_round_(tag='row9', spec_tid='101')

        # ── ...and IDLE within the same round, on both axes independently ───
        assert read_lock_holder_pgid(git_ops.worktree_base) is None, (
            'the fixed-key holder rendezvous must be cleared — '
            'reset_persistent_merge_worktree reads it FAIL-CLOSED'
        )
        assert (not lock_path.exists()) or lane_lock_holder_pids_strict(
            lock_path
        ) == [], (
            f'the kernel flock axis must be free after the head cancel; got '
            f'{lane_lock_holder_pids_strict(lock_path)!r}'
        )
        assert git_ops._merge_verify_lease_active() is False, (
            'the combined predicate the admission guard consumes must agree '
            'with both axes'
        )
        # ── ...and it got there through the SANCTIONED chokepoint ───────────
        #
        # The LIVE-head arm, which Row 5's delivered-verdict head provably
        # cannot enter: the head here is still verifying when the tip passes,
        # so the adoption really does tear it down — and the teardown IS the
        # mechanism the two idle readings above rest on.  Asserted as an
        # ordered pair because the order is load-bearing: the verify
        # coroutine's finally clears ``_inflight_request_id`` on cancellation,
        # which would turn a later ``cancel_verify()`` into a silent no-op that
        # orphans a remote verify-merge process.
        assert teardown == ['teardown:100', 'cancel_release:local'], (
            f'the head verify must be released through _teardown_verify_task '
            f'then _cancel_and_release_tracked, in that order; got '
            f'{teardown!r}'
        )
        assert head_entry.lease is None, (
            'entry.lease must be cleared after the release so no later path '
            'double-releases it'
        )
        # And the round really did adopt — an idle lane after a round that
        # never ran would prove nothing.
        assert rec['chain'] is not None and rec['advanced'] is True
        assert scene.reqs['100'].result.result().status == 'done'

        assert _drain_residue(worker) == set()
        _assert_two_way_quiescent(
            worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()), permits_before=permits_before,
        )

    async def test_row_10_flipping_the_cap_mid_run_starts_landing_chains(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """Row 10: cap 0 -> 6 through the REAL ``apply_reload``, no restart.

        ``merge_deep.chain_cap`` is a green-tier ``RELOADABLE_FIELDS`` leaf and
        the worker reads it LIVE off the dispatching request's config, so the
        operator-facing claim is that the round AFTER the flip builds and lands
        a chain against the very same worker, queue and repo.

        Driven through ``config.apply_reload`` rather than by assigning the
        attribute, because the claim is about the RELOAD PATH — an attribute
        assignment would still pass if ``chain_cap`` were demoted to the
        restart-only tier tomorrow, which is precisely the regression this row
        exists to catch.
        """
        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=0, n_followers=_COMPOSED_FOLLOWERS,
            db_name='gate-row10-reload.db', heads=('101', '107'),
            script=[True, True],
        )
        worker = scene.worker
        permits_before = _permit_census(worker)

        cold = await scene.round_(tag='row10', head_tid='101')
        assert cold['chain'] is None and cold['built'] == [], (
            f'the kill switch must hold before the flip; got {cold["chain"]!r}'
        )
        assert cold['advanced'] is True, 'the cold round still lands, ordinarily'
        _assert_midrun_conserved(
            worker, cold['main_after'], permits_before, where='before the reload',
        )

        result = scene.reload_cap(6)
        assert result['reloaded'] is True, f'apply_reload refused: {result!r}'
        assert 'merge_deep.chain_cap' in result['applied'], (
            f'chain_cap must be a GREEN-tier leaf; got {result!r}'
        )
        assert result['restart_required'] == {}, (
            f'no restart may be required for a green-tier leaf; got {result!r}'
        )
        assert scene.config.merge_deep.chain_cap == 6, (
            'the live config object every enqueued request holds by reference '
            'must observe the new value'
        )

        hot = await scene.round_(tag='row10', head_tid='107')
        assert hot['chain'] is not None and len(hot['chain'].links) == 5, (
            f'the very NEXT round must build a chain — no restart; got '
            f'{None if hot["chain"] is None else 1 + len(hot["chain"].links)}'
        )
        landed = ['107', *[tid for tid, _mc in hot['chain'].links]]
        for tid in landed:
            assert scene.reqs[tid].result.result().status == 'done'
            assert scene.reqs[tid].result.result().landed_via_chain == 1
        # SAME worker across the flip — the "no restart" claim, stated as an
        # object identity rather than inferred from the landing.
        assert scene.worker is worker

        assert _drain_residue(worker) == set()
        _assert_two_way_quiescent(
            worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()), permits_before=permits_before,
        )


# ═══════════════════════════════════════════════════════════════════════════
# -- step-15 RED: Row 4 — A CHAIN CONFLICT TRUNCATES SILENTLY --
#
# The PRD row: "item 2 conflicts with item 1 textually -> chain = [item 1], no
# conflict outcome for item 2, item 2 handled sequentially later".
#
# The SUBSUMPTION table marks this row FULLY SUBSUMED at unit level, and it is:
# test_merge_queue_deep_landing.py builds a 4-item chain whose FIFTH item (105)
# conflicts with LINK 102, and pins that the truncator renders no outcome and
# emits no event.  Two things that scene structurally cannot say, both of which
# are the composition this class owns:
#
#   * TRUNCATION AT POSITION ZERO.  Upstream's truncator sits past three clean
#     links, so `links` is non-empty and the round is deep either way.  A
#     conflict with the DISPATCHING item truncates at position 0, which is a
#     different code path in `_deep_chain_placement` — the `if not result.links`
#     decline, whose two documented obligations (do NOT release the lane a
#     second time; consume NO halving round) are invisible to a scene that
#     never reaches it.
#   * NO SKIP-AHEAD.  `build_chain` must produce a contiguous PREFIX, so a
#     conflict at position 0 must NOT be followed by the clean items behind it.
#     Upstream's fixture has nothing queued past its truncator, so "did it skip
#     ahead?" is unanswerable there; here 103 and 104 are both clean, both in
#     the snapshot the build was handed, and both must be left alone.
#
# WHY THE SECOND TEST NEEDS A RED FIRST ROUND, since it is the one design
# choice a reader will question.  The row's last clause — "item 2 handled
# sequentially later" — can only be staged if item 2 is still MERGEABLE later,
# and after item 1 lands it is not: main then carries item 1's line and item 2's
# ordinary merge conflicts for real.  That is not a fixture limitation, it is
# decision #4's own premise restated — a chain conflict is a conflict with an
# UNLANDED predecessor, and it says nothing about item 2 precisely BECAUSE the
# predecessor may never land.  So the second test makes the predecessor's own
# verify red: item 1 blocks, main never moves, and item 2 then merges cleanly
# and lands on its own round's verdict.  Staging it any other way would be
# asserting a scenario the contract does not describe.
# ═══════════════════════════════════════════════════════════════════════════

_ROW4_CONFLICT_LINE = 5
"""The ``shared.txt`` line the conflicting pair both rewrite.

Line 5 of 20 rather than line 1 or line 20: git's three-line diff context
window makes edits near a file boundary conflict with edits that are nominally
several lines away, and a conflict this file did not intend is exactly what
would turn a "no skip-ahead" claim into an accident.  (The 20-line seed and the
reason for it are documented on :func:`_setup_repo`.)
"""


@pytest.mark.asyncio
@pytest.mark.timeout(300)
class TestRow4ConflictTruncatesSilently:
    """Row 4: a chain conflict is a fact about the CHAIN, not a verdict."""

    async def test_a_conflict_at_position_zero_declines_and_skips_nobody(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """The chain is exactly [item 1]; 103 and 104 are offered and refused.

        ``build_chain`` is handed the WHOLE snapshot — 102, 103 and 104 — and
        102 is the only one that cannot merge.  A build that "helpfully"
        skipped past it would produce a chain with a HOLE in it, and δ's CAS
        walk lands ``links`` in order against main: a hole would land 103 onto
        a main that never received 102, breaking the in-order / frozen-prefix
        invariant the whole slice rests on.  So the assertion is not merely
        that the chain is short — it is that the two clean items BEHIND the
        conflict were available and were left alone.

        The decline's two documented obligations are pinned alongside it:
        nothing is released twice (the conservation oracle would see a
        double-release as a permit or lane violation), and the halving
        bisector is NOT advanced — a round that verified no chain has no tip
        verdict to halve on, and halving here would drag the ceiling down for
        rounds that never failed.
        """
        from orchestrator.merge_queue import select_chain_depth
        from orchestrator.merge_types import ItemLifecycleState

        followers = list(_gate_followers(3))
        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=_HEAD_CAP, n_followers=3, db_name='gate-row4-trunc0.db',
            script=[True],
            branch_content=_conflicting_branch_content('101', '102'),
        )
        worker = scene.worker
        permits_before = _permit_census(worker)

        rec = await scene.round_(tag='row4', head_tid='101')

        # ── the chain is exactly [item 1]: a DECLINE, not a short chain ─────
        assert rec['chain'] is None, (
            f'a zero-link build must decline outright — `_deep_chain_placement` '
            f'returns None so the dispatch is byte-identical to the adjacent '
            f'verify; got {rec["chain"]!r}'
        )
        assert len(scene.built) == 1, (
            f'exactly one build was attempted; got {len(scene.built)}'
        )
        built = scene.built[0]
        result = built['result']
        assert result.links == [], (
            f'item 2 conflicts with item 1, so NOTHING chains; got '
            f'{result.links!r}'
        )
        assert result.truncated_at == '102', (
            f'the truncator must be NAMED, or ε has no fault signal to read; '
            f'got {result.truncated_at!r}'
        )
        assert result.truncated_reason == 'conflict', (
            f"a textual conflict is the BENIGN bucket — 'merge_error' is the "
            f'genuine-fault one and would escalate the build log to WARNING; '
            f'got {result.truncated_reason!r}'
        )
        assert result.tip == rec['head_mc'], (
            f'a zero-link result still names a VERIFIABLE tip, and it is the '
            f'base it was handed; got {result.tip[:8]!r}'
        )

        # ── NO SKIP-AHEAD: the clean items were offered and refused ─────────
        assert [r.task_id for r in built['queue_snapshot']] == followers, (
            f'the build must be offered the whole snapshot, or "it did not '
            f'skip ahead" is unfalsifiable; got '
            f'{[r.task_id for r in built["queue_snapshot"]]!r}'
        )
        assert built['cap'] == _HEAD_CAP - 1, (
            f'the caller converts ITEMS to LINKS-BEYOND-THE-BASE; got '
            f'{built["cap"]!r}'
        )
        assert built['target_depth'] == select_chain_depth(
            _HEAD_CAP, 1 + len(followers), None,
        ) - 1, (
            f'the target depth must be the policy function\'s own answer, '
            f'converted; got {built["target_depth"]!r}'
        )

        # ── item 1 lands by the ORDINARY adjacent path ──────────────────────
        assert rec['advanced'] is True, 'the declined round still lands, ordinarily'
        assert [
            (row['branch'], row['landed_via_chain'])
            for row in _finalized_rows(scene.db_path)
        ] == [('101', None)], (
            f'only the dispatching item landed, and NOT via a chain — a stamp '
            f"here would inflate the canary's ratio for a round that verified "
            f'no chain at all; got {_finalized_rows(scene.db_path)!r}'
        )

        # ── the decline consumed NO halving round ───────────────────────────
        assert worker._chain_halving_state is None, (
            f'a declined round has no tip verdict to halve on; halving here '
            f'would drag the ceiling down for rounds that never failed. Got '
            f'{worker._chain_halving_state!r}'
        )

        # ── every offered item is UNTOUCHED, in its original lane position ──
        assert [r.task_id for r in worker._lane_buffers['normal']] == followers, (
            f'submission order must survive a truncation — it is the next '
            f'round\'s chain_snapshot; got '
            f'{[r.task_id for r in worker._lane_buffers["normal"]]!r}'
        )
        for tid in followers:
            req = scene.reqs[tid]
            assert not req.result.done(), (
                f'task {tid} was handed a verdict the chain had no standing '
                f'to render: {req.result!r}'
            )
            assert worker._lifecycle.current(req.request_id) is (
                ItemLifecycleState.LANE_BUFFERED
            ), f'task {tid} left LANE_BUFFERED'
            assert _events_for_task(scene.db_path, tid) == ['merge_queued'], (
                f'task {tid} must emit nothing but its own enqueue — an event '
                f'here is a public claim about an item nothing verified; got '
                f'{_events_for_task(scene.db_path, tid)!r}'
            )

        assert _drain_residue(worker) == set(followers), (
            'the truncator AND the two clean items behind it stay queued'
        )
        _assert_two_way_quiescent(
            worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()), permits_before=permits_before,
        )

    async def test_the_truncated_item_lands_on_its_own_ordinary_round(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """Item 2's verdict is its OWN round's, rendered after item 1 blocked.

        The composition half of row 4, and the reason the class banner spends a
        paragraph on it: the truncation is only meaningful if item 2 is still
        alive afterwards, so this run carries it to the round where it takes
        the ordinary sequential path and lands on a verify of its own tree.

        REMOTE, not the cheap seam, and the reason is a fixture fact worth
        recording: ``_make_gate_scene``'s non-remote ``_oracle`` answers a red
        with a raw :class:`~orchestrator.verify.VerifyResult`, which the CHAIN
        arm consumes as "not None -> red" without reading a field, but which
        the UN-chained arm dereferences (``out.verify_skipped``) — it expects
        the :class:`MergeOutcome` the real ``_run_post_merge_verify`` renders.
        A red on a DECLINED round therefore has to be injected below that
        function, which is exactly what ``remote=True`` does.
        """
        from orchestrator.merge_types import ItemLifecycleState

        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=_HEAD_CAP, n_followers=1, db_name='gate-row4-later.db',
            remote=True, script=[False, True],
            branch_content=_conflicting_branch_content('101', '102'),
        )
        # Off for the reason Row 1 documents: the remote cross-check of a
        # remote green is run-to-run nondeterministic in a fixture repo and has
        # nothing to do with this row's claim.
        scene.config.verify_cross_check_remote_green = False
        worker = scene.worker
        permits_before = _permit_census(worker)

        red = await scene.round_(tag='row4', head_tid='101')

        assert red['chain'] is None and scene.built[0]['result'].links == [], (
            'the round must really have truncated at position 0'
        )
        assert scene.built[0]['result'].truncated_at == '102'
        assert scene.built[0]['result'].truncated_reason == 'conflict'

        # ── item 1 blocked, so main NEVER MOVED — decision #4's premise ─────
        assert scene.reqs['101'].result.result().status == 'blocked', (
            f'item 1 must take the verdict of its OWN tree; got '
            f'{scene.reqs["101"].result.result()!r}'
        )
        assert red['main_after'] == red['main_before'], (
            'a blocked item moves nothing — and that is precisely why item 2 '
            'is still mergeable below'
        )

        # ── item 2 is untouched by the truncation ───────────────────────────
        two = scene.reqs['102']
        assert not two.result.done(), 'the truncator was handed a verdict'
        assert worker._lifecycle.current(two.request_id) is (
            ItemLifecycleState.LANE_BUFFERED
        )
        assert _events_for_task(scene.db_path, '102') == ['merge_queued'], (
            f'the truncator must emit nothing but its own enqueue up to here; '
            f'got {_events_for_task(scene.db_path, "102")!r}'
        )

        # ── ...and then takes the ORDINARY sequential path and lands ────────
        nxt = worker._pop_next_pickable()
        assert nxt is two, (
            f'item 2 must be the next pickable, at its original lane index; '
            f'got {None if nxt is None else nxt.task_id!r}'
        )
        green = await scene.round_(tag='row4', head_tid='102', req=nxt)

        assert green['chain'] is None, (
            f'nothing is left to chain ONTO item 2, so the policy declines and '
            f'the dispatch is pre-PRD; got {green["chain"]!r}'
        )
        assert len(scene.built) == 1, (
            f'the decline is made at the POLICY layer — the second round must '
            f'not even attempt a build; got {len(scene.built)} builds'
        )
        assert green['advanced'] is True and green['main_after'] != green['main_before']
        outcome = two.result.result()
        assert outcome.status == 'done', (
            f'item 2 must land on its OWN round\'s verdict; got {outcome!r}'
        )
        assert outcome.landed_via_chain is None, (
            f'item 2 landed by the ordinary advance, not by a walk; got '
            f'{outcome.landed_via_chain!r}'
        )
        assert [
            (row['branch'], row['landed_via_chain'])
            for row in _finalized_rows(scene.db_path)
        ] == [('101', None), ('102', None)], (
            f'got {_finalized_rows(scene.db_path)!r}'
        )

        # ── the canary counts NEITHER round as deep ─────────────────────────
        verify_rows = _verify_rows(scene.db_path)
        assert [row['chain_items'] for row in verify_rows] == [1, 1], (
            f'a declined round verifies ONE item; got '
            f'{[row.get("chain_items") for row in verify_rows]!r}'
        )
        assert _canary_predicate_items_per(
            verify_rows, _finalized_rows(scene.db_path),
        ) is None, (
            'a truncation-to-zero contributes nothing to the deep denominator '
            '— counting it would make the shipped ratio read low forever'
        )

        assert _drain_residue(worker) == set(), 'the run drained the whole queue'
        _assert_two_way_quiescent(
            worker, await _rev_parse(git_repo, 'main'),
            list(scene.reqs.values()), permits_before=permits_before,
        )


# ═══════════════════════════════════════════════════════════════════════════
# -- step-15 RED: THE CONSERVATION CAPSTONE — one long mixed run --
#
# The CROSS-CUTTING gap the module docstring names, and the gate's single
# user-observable signal.  Every row above states ONE claim inside a run shaped
# to make that claim sharp; nothing anywhere — here or upstream — asserts that
# the conservation surfaces stay green across a MIXED run that actually lands,
# nor that the telemetry an operator reads agrees with what git actually did.
#
# Upstream's two conservation checks are each blind to half of it:
#   * test_merge_queue_deep_dispatch.py's conservation test never finalizes —
#     it lands nothing, so no permit is ever consumed by a walk;
#   * every test_merge_queue_deep_landing.py `_assert_quiescent` call is
#     single- or two-round, so nothing samples the ledgers ACROSS a sequence.
#
# SIX ROUNDS, one worker, one queue, one repo, one permit census, covering
# every phenomenon the slice can produce:
#
#   R1  deep PASS at cap 6            -> six items land through the CAS walk
#   R2  deep FAIL at 6                -> nothing lands; the bisector halves 6->3
#   R3  deep FAIL at 3                -> nothing lands; the bisector halves 3->1
#   R4  the FLOOR round               -> the policy declines (`< 2`); the item
#                                        lands by the ORDINARY adjacent verify,
#                                        and the pass RESETS the bisector — the
#                                        only way the walk ever climbs back out
#   R5  stale-CAS ABORT               -> main moves externally mid-walk; the
#                                        landed prefix stays, the rest requeues
#   R6  hot-reload 6->32 + a CONFLICT -> the very next round chains TWELVE
#                                        items (twice the old cap, so the
#                                        reload cannot be a no-op) and
#                                        truncates on the conflicting tail
#
# WHY THE MID-RUN ORACLE IS THE SUBSET, restated once more because this is the
# one place a reader will expect the full one: surfaces (a), (d) and (f) of
# `_assert_two_way_quiescent` are WHOLE-REGISTRY claims, and a run that is
# deliberately mid-flight has queued items with pending futures at every
# checkpoint.  `_assert_midrun_conserved` is what holds at EVERY instant, so it
# runs after every round; the full oracle runs once at the end, with the thrash
# ladder as its CAS/ledger half, after `_drain_residue` has confirmed the run
# left exactly the residue it promised.
#
# THE GROUND TRUTH IS GIT, not the telemetry's own arithmetic.  The canary an
# operator reads (`scripts/merge-deep-canary-predicate.sh`, transcribed in
# `_canary_predicate_items_per`) divides a `merge_finalized` sum by a
# `merge_verify` count; comparing those two to each other proves only that the
# emitter is self-consistent.  So the run's landings are re-derived from main's
# first-parent history (`_chain_landed_from_git`) and the telemetry is checked
# against THAT — the same discipline ε's reader signal will need.
# ═══════════════════════════════════════════════════════════════════════════

_CAPSTONE_CAP = 6
"""The starting cap: deep enough that R1's chain is a real prefix, and an even
number, so R2/R3's bisection walks 6 -> 3 -> 1 and reaches the floor in exactly
two failures rather than three."""

_CAPSTONE_RELOADED_CAP = 32
"""The PRD's stated maximum, and what η2's promotion ships.  R6 must chain
DEEPER than ``_CAPSTONE_CAP`` for the reload to have been observable at all."""

_CAPSTONE_FOLLOWERS = 21
"""``102``..``122``: exactly enough to feed six rounds and leave ONE residue.

Sized from the round schedule, not chosen round: 6 land in R1, 1 in R4, 2 in
R5 (the abort truncates the walk), 12 in R6, and ``122`` is the conflicting
tail that must still be queued at the end.  A larger fixture would leave
untouched items that make the final residue assertion weaker."""

_CAPSTONE_SCRIPT = [True, False, False, True, True, True]
"""One verdict per round, in order — the mixed sequence R1..R6 above."""

_CAPSTONE_ABORT_AT = 10
"""The CUMULATIVE ``advance_main`` ordinal R5's external main-bump fires on.

Cumulative across the scene, which is what makes a fixed literal safe here
(see :func:`_abort_hook_at`): R1 advances six times (calls 1-6), R2 and R3 land
nothing at all, R4 advances once (call 7), so R5's walk owns calls 8, 9, 10.
Arming at 10 lets the head and its first link land and aborts on the second,
which is the shape that leaves a genuinely PARTIAL prefix behind.  Every one of
those counts is asserted below, so a schedule change surfaces as a legible
round-by-round failure rather than as a mysteriously mis-armed hook."""


@pytest.mark.asyncio
@pytest.mark.timeout(300)
class TestDeepGateCapstone:
    """One continuous six-round run: conservation, and telemetry vs git."""

    async def test_a_mixed_run_conserves_and_the_canary_agrees_with_git(
        self, git_repo: Path, tmp_path: Path, monkeypatch,
    ) -> None:
        """Every round conserves, and the shipped canary matches git's own count."""
        from orchestrator.merge_queue import next_halving_state, select_chain_depth

        scene = await _make_gate_scene(
            git_repo, tmp_path, monkeypatch,
            chain_cap=_CAPSTONE_CAP, n_followers=_CAPSTONE_FOLLOWERS,
            db_name='gate-capstone.db', remote=True, script=_CAPSTONE_SCRIPT,
            advance_hook=_abort_hook_at(_CAPSTONE_ABORT_AT, bump_repo=git_repo),
            branch_content=_conflicting_branch_content('121', '122'),
        )
        # Off for the reason Row 1 documents (run-to-run nondeterminism in a
        # fixture repo).  Re-pinned after the reload below, which rebuilds the
        # comparison config from defaults and therefore turns it back on.
        scene.config.verify_cross_check_remote_green = False
        worker = scene.worker
        permits_before = _permit_census(worker)
        main_at_start = await _rev_parse(git_repo, 'main')

        def _conserved(rec: dict) -> None:
            _assert_midrun_conserved(
                worker, rec['main_after'], permits_before,
                where=f'after round {rec["round"]} ({rec["tag"]})',
            )

        # ── R1: a deep PASS lands a six-item prefix ─────────────────────────
        r1 = await scene.round_(tag='deep-pass', head_tid='101')
        assert r1['chain'] is not None and 1 + len(r1['chain'].links) == _CAPSTONE_CAP
        assert len(r1['advance_calls']) == _CAPSTONE_CAP, (
            f'the walk advances once per chain item; got '
            f'{len(r1["advance_calls"])}'
        )
        assert r1['landed'] == ['101', *[t for t, _mc in r1['chain'].links]]
        _conserved(r1)

        # ── R2: a deep FAIL halves 6 -> 3 and lands nothing ─────────────────
        r2 = await scene.round_(tag='deep-fail', head_tid='107')
        assert r2['chain'] is not None and 1 + len(r2['chain'].links) == _CAPSTONE_CAP
        assert r2['advanced'] is False and r2['advance_calls'] == [], (
            f'a red tip must not reach the CAS walk at all; got '
            f'{r2["advance_calls"]!r}'
        )
        assert worker._chain_halving_state == next_halving_state(
            False, _CAPSTONE_CAP,
        ) == 3
        _conserved(r2)

        # ── R3: the SAME request fails again, halving 3 -> 1 (the floor) ────
        again = worker._queue.get_nowait()
        assert again is scene.reqs['107'] and not again.result.done(), (
            'a red tip requeues its head UNRESOLVED — the pipeline re-picks it'
        )
        r3 = await scene.round_(tag='deep-fail-2', head_tid='107', req=again)
        assert r3['chain'] is not None and 1 + len(r3['chain'].links) == 3, (
            f'the halved ceiling must bind the second attempt; got '
            f'{None if r3["chain"] is None else 1 + len(r3["chain"].links)}'
        )
        assert r3['advanced'] is False and r3['advance_calls'] == []
        assert worker._chain_halving_state == next_halving_state(False, 3) == 1
        _conserved(r3)

        # ── R4: the FLOOR declines, the item lands ordinarily, the walk resets
        floored = worker._queue.get_nowait()
        assert floored is scene.reqs['107']
        n_builds_before_floor = len(scene.built)
        r4 = await scene.round_(tag='floor', head_tid='107', req=floored)
        assert select_chain_depth(
            _CAPSTONE_CAP, 1 + len(_gate_followers(_CAPSTONE_FOLLOWERS)), 1,
        ) is None, "the policy's `< 2` floor is what declines here"
        assert r4['chain'] is None, (
            f'at the floor the gate must decline; got {r4["chain"]!r}'
        )
        assert len(scene.built) == n_builds_before_floor, (
            'the floor declines at the POLICY layer — no build may be attempted'
        )
        assert r4['advanced'] is True and len(r4['advance_calls']) == 1
        assert scene.reqs['107'].result.result().landed_via_chain is None, (
            'the floor round landed by the ordinary adjacent verify'
        )
        assert worker._chain_halving_state is None, (
            'ANY pass resets the bisector — this arm is the only way the walk '
            'climbs back out once the floor stops producing tip verdicts'
        )
        _conserved(r4)

        # ── R5: main moves externally mid-walk; a PARTIAL prefix lands ──────
        r5 = await scene.round_(tag='stale-cas', head_tid='108')
        assert r5['chain'] is not None and 1 + len(r5['chain'].links) == _CAPSTONE_CAP
        assert len(r5['advance_calls']) == 3, (
            f'the walk must stop at the FIRST non-advanced result and never '
            f'skip ahead; got {len(r5["advance_calls"])} calls'
        )
        assert r5['landed'][-2:] == ['108', '109'], (
            f'the head and its first link landed before the bump; got '
            f'{r5["landed"]!r}'
        )
        assert [r.task_id for r in worker._lane_buffers['normal']][:4] == [
            '110', '111', '112', '113',
        ], (
            f'every unlanded link returns to its EXACT lane index; got '
            f'{[r.task_id for r in worker._lane_buffers["normal"]][:4]!r}'
        )
        _conserved(r5)

        # ── R6: hot-reload 6 -> 32, then a twelve-item chain that truncates ──
        reload_result = scene.reload_cap(_CAPSTONE_RELOADED_CAP)
        assert reload_result['reloaded'] is True, f'apply_reload refused: {reload_result!r}'
        assert 'merge_deep.chain_cap' in reload_result['applied'], (
            f'chain_cap must be a GREEN-tier leaf; got {reload_result!r}'
        )
        assert reload_result['restart_required'] == {}, (
            f'no restart may be required for a green-tier leaf; got '
            f'{reload_result!r}'
        )
        # The comparison config `reload_cap` builds carries stock defaults, so
        # the reload also flips the remote-green cross-check back ON.  Re-pin
        # it for the same determinism reason it was pinned at the top.
        scene.config.verify_cross_check_remote_green = False

        nxt = worker._pop_next_pickable()
        assert nxt is scene.reqs['110']
        r6 = await scene.round_(tag='reload+truncate', head_tid='110', req=nxt)
        assert r6['chain'] is not None and 1 + len(r6['chain'].links) == 12, (
            f'the round AFTER the flip must chain at the NEW cap — twelve '
            f'items, twice the old ceiling, on the same worker and queue; got '
            f'{None if r6["chain"] is None else 1 + len(r6["chain"].links)}'
        )
        assert 1 + len(r6['chain'].links) > _CAPSTONE_CAP, (
            'a chain no deeper than the old cap would leave the reload unproven'
        )
        assert scene.built[-1]['result'].truncated_at == '122', (
            f'the conflicting tail must truncate the chain; got '
            f'{scene.built[-1]["result"].truncated_at!r}'
        )
        assert scene.built[-1]['result'].truncated_reason == 'conflict'
        assert len(r6['advance_calls']) == 12
        assert scene.worker is worker, (
            'the same worker across the flip — the "no restart" claim, as an '
            'object identity rather than an inference from the landing'
        )
        _conserved(r6)

        # ── the durable tier, read back through real sqlite ─────────────────
        finalized = _finalized_rows(scene.db_path)
        verify_rows = _verify_rows(scene.db_path)
        assert len(scene.rounds) == 6, f'six rounds; got {len(scene.rounds)}'
        assert len(verify_rows) == 6, (
            f'one verify per round, deep or not; got {len(verify_rows)}'
        )
        assert [row['chain_items'] for row in verify_rows] == [6, 6, 3, 1, 6, 12], (
            f'the reader must see each round\'s real breadth; got '
            f'{[row.get("chain_items") for row in verify_rows]!r}'
        )

        # ── GROUND TRUTH #1: git's land order IS the telemetry's ────────────
        main_at_end = await _rev_parse(git_repo, 'main')
        landed_per_git = await _chain_landed_from_git(
            git_repo, main_at_start, main_at_end,
        )
        assert landed_per_git == [row['branch'] for row in finalized], (
            f'the merge_finalized stream must be exactly what main\'s '
            f'first-parent history says landed, in the same order; git says '
            f'{landed_per_git!r}, telemetry says '
            f'{[row["branch"] for row in finalized]!r}'
        )
        assert len(landed_per_git) == 21, (
            f'21 of the 22 items land; got {len(landed_per_git)}'
        )

        # ── GROUND TRUTH #2: sum(landed_via_chain) IS git's chain-round count
        per_round_landings = [
            len(await _chain_landed_from_git(
                git_repo, rec['main_before'], rec['main_after'],
            ))
            for rec in scene.rounds
        ]
        assert per_round_landings == [6, 0, 0, 1, 2, 12], (
            f'per-round landings, counted off git; got {per_round_landings!r}'
        )
        chain_landed_per_git = sum(
            n for rec, n in zip(scene.rounds, per_round_landings, strict=True)
            if rec['chain'] is not None
        )
        assert sum(
            row['landed_via_chain'] or 0 for row in finalized
        ) == chain_landed_per_git == 20, (
            f'``landed_via_chain`` ships as ONE PER LANDED ITEM, so its sum is '
            f'the number of items that landed inside a round that carried a '
            f'chain — git says {chain_landed_per_git}, telemetry says '
            f'{sum(row["landed_via_chain"] or 0 for row in finalized)}'
        )
        assert all(
            row['landed_via_chain'] is None
            for row in finalized if row['branch'] == '107'
        ), 'the floor round landed ordinarily and must carry no chain stamp'

        # ── GROUND TRUTH #3: the deep DENOMINATOR is the rounds that chained ─
        n_deep = len([
            row for row in verify_rows
            if isinstance(row.get('chain_items'), int) and row['chain_items'] >= 2
        ])
        rounds_that_chained = len([r for r in scene.rounds if r['chain'] is not None])
        assert n_deep == rounds_that_chained == 5, (
            f'the canary classifies a verify as deep by `chain_items >= 2`, '
            f'which must be exactly the rounds that really built a chain; '
            f'predicate says {n_deep}, the scene says {rounds_that_chained}'
        )
        assert _canary_predicate_items_per(verify_rows, finalized) == 20 / 5, (
            f'the SHIPPED predicate, over the run\'s real rows; got '
            f'{_canary_predicate_items_per(verify_rows, finalized)!r}'
        )

        # ── the ladder never moved, across two deep fails ───────────────────
        before = RetryLedger()
        after = _ladder_after(
            scene.rounds, ledger=before, requests=list(scene.reqs.values()),
        )

        # ── and the run left EXACTLY the residue it promised ────────────────
        residue = _drain_residue(worker)
        assert residue == {'122'}, (
            f'only the conflicting tail may still be queued; drained '
            f'{sorted(residue, key=int)!r}'
        )
        _assert_two_way_quiescent(
            worker, main_at_end, list(scene.reqs.values()),
            permits_before=permits_before,
            ladder={'before': before, 'after': after},
        )
