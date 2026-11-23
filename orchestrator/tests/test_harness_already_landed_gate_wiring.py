"""Tests for the harness's already-landed pre-dispatch gate (task 2313).

Architecturally parallel to task 2156's landed-outbox dispatch gate
(test_harness_landed_dispatch_gate_wiring.py), but this gate consults LIVE
GIT STATE (ancestry, and patch-id containment of a live tip) rather than the
durable LandedOutbox, so it also catches OUT-OF-BAND landings that never went
through this orchestrator's own merge queue: a sibling direct-merge, a prior
orchestrator run, or a rebase/manual landing.  A multi-commit squash is the
residual the patch-id repoint accepted (task 4648), pinned in
test_already_landed_gate_patch_id.py.

Covers the gate's three arms and the vetoes around them:
  - ancestry arm (live tip an ancestor of main): an accepted
    ``branch_work_landed`` verdict flips the task to done; every refused
    verdict is recorded, never escalated, and the task dispatches.
  - off-main-tip arm (live tip not an ancestor): the same producer decides;
    ``no_attribution`` escalates and every other reject is recorded.
  - branch-deleted merge-marker arm: a marker flips through
    ``validate_landing_evidence`` CANDIDATE; a marker that predates this
    incarnation (an ancestor of ``branch_base_sha``) vetoes the flip.
  - the any-level escalation veto, the arbitration holds, the
    delivered-checks guard and the transition-gated emissions.
The producer is stubbed here (``landing_producer``); real git drives it in
test_already_landed_gate_patch_id.py.

Mirrors test_harness_landed_dispatch_gate_wiring.py's ``_build_harness``
bare-harness construction helper exactly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _landing_evidence_helpers import differential_confirms, landing_reject_rows
from escalation.models import Escalation

from orchestrator.already_landed_tick_view import AlreadyLandedTickView
from orchestrator.config import (
    DeliveredChecksConfig,
    OrchestratorConfig,
    RecoveryEmissionConfig,
)
from orchestrator.delivered_checks import DeliveredChecksBlock
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.harness import Harness
from orchestrator.merge_lane.landing_evidence import (
    LandingEvidenceVerdict,
    LandingMethod,
    LandingReason,
    LandingVerdict,
    format_unattributed_landing_detail,
    validate_landing_evidence,
)
from orchestrator.scheduler import TaskAssignment

_HARNESS_SRC_PATH = Path(__file__).parent.parent / 'src' / 'orchestrator' / 'harness.py'

#: The live branch tip every stub wiring below resolves.
_TIP_SHA = 'f' * 40
#: The commit an accepted landing verdict anchors on.
_LANDED_SHA = 'a' * 40


def _landed_verdict(evidence_sha: str = _LANDED_SHA) -> LandingVerdict:
    """An accepted ``branch_work_landed`` verdict anchored on *evidence_sha*."""
    return _patch_id_verdict(LandingReason.landed, evidence_sha=evidence_sha)


def _refused_verdict(reason: LandingReason, **probe: Any) -> LandingVerdict:
    """A rejected ``branch_work_landed`` verdict for *reason*, *probe* merged in."""
    return _patch_id_verdict(reason, evidence_sha=None, **probe)


def _patch_id_verdict(
    reason: LandingReason, *, evidence_sha: str | None, **probe: Any,
) -> LandingVerdict:
    """A hand-built patch-id verdict whose probe is seeded as the producer seeds it.

    The gate records a reject's probe verbatim, so ``method`` and ``reason``
    must be in the probe as well as on the verdict for a row to name them.
    """
    return LandingVerdict(
        accepted=reason is LandingReason.landed,
        evidence_sha=evidence_sha,
        reason=reason,
        probe={
            'task_id': '42', 'branch': 'task/42', 'branch_tip_sha': _TIP_SHA,
            'method': str(LandingMethod.patch_id), 'reason': str(reason), **probe,
        },
        method=LandingMethod.patch_id,
    )


def _not_contained() -> LandingVerdict:
    return _refused_verdict(
        LandingReason.not_landed,
        upstream_ref='main', patch_id_containment='not_contained',
        containment_recheck='healthy',
    )


@pytest.fixture(autouse=True)
def landing_producer(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """``orchestrator.harness.branch_work_landed``, stubbed for every test here.

    The gate's live-branch arms hand their whole decision to that producer,
    which shells ``git cherry``; a ``MagicMock`` ``git_ops`` cannot answer
    that.  The default verdict accepts, anchored on ``_LANDED_SHA``, so a
    harness wired to reach a live arm flips unless a test sets
    ``return_value`` to the verdict under test.  The real producer is driven
    over real git in ``test_already_landed_gate_patch_id.py``.
    """
    producer = AsyncMock(return_value=_landed_verdict())
    monkeypatch.setattr('orchestrator.harness.branch_work_landed', producer)
    return producer


def _build_harness(mock_orch_config) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    Mirrors test_harness_landed_dispatch_gate_wiring.py's ``_build_harness``.
    """
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateAncestryHappyPath:
    """Ancestry happy-path: the live tip is an ancestor of main and the producer accepts."""

    async def test_ancestry_with_an_accepted_verdict_flips_to_done(
        self, mock_orch_config,
    ) -> None:
        """The gate drives the task to done via _mark_in_progress_done,
        anchored on the verdict's evidence sha, and returns True so the
        scheduler withholds dispatch this tick.
        """
        h = _build_harness(mock_orch_config)
        h.git_ops = MagicMock()
        h.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)
        h.git_ops.is_ancestor = AsyncMock(return_value=True)
        h.git_ops.config.branch_prefix = 'task/'
        h.git_ops.config.main_branch = 'main'
        h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
        mark = AsyncMock()
        h._mark_in_progress_done = mark
        h._escalation_queue = None

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        mark.assert_awaited_once()
        assert mark.await_args is not None
        assert mark.await_args.args[0] == '42'
        assert mark.await_args.args[1] == _LANDED_SHA
        assert mark.await_args.args[3] == 'dispatch-gate-already-on-main'


def _wired_ancestry_harness(mock_orch_config) -> Harness:
    """Bare harness pre-wired so the ancestry path would flip on its own.

    The live tip resolves and is an ancestor of main, and the module's
    ``landing_producer`` accepts; each guard test overrides exactly one input
    to trip its veto.
    """
    h = _build_harness(mock_orch_config)
    h.git_ops = MagicMock()
    h.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)
    h.git_ops.is_ancestor = AsyncMock(return_value=True)
    h.git_ops.config.branch_prefix = 'task/'
    h.git_ops.config.main_branch = 'main'

    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
    h._mark_in_progress_done = AsyncMock()
    h._escalation_queue = None
    return h


#: Every reason ``branch_work_landed`` can refuse with.
_PATCH_ID_REJECTS = [
    LandingReason.degenerate_branch,
    LandingReason.no_op_landing,
    LandingReason.not_landed,
    LandingReason.no_attribution,
    LandingReason.git_error,
]


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateAncestryGuards:
    """Every refused verdict on the ancestry arm vetoes the flip.

    Each case starts from an otherwise-flipping ancestry setup and hands the
    gate one refused verdict: the gate returns False, does not stamp and files
    no escalation, so the task dispatches normally.
    """

    @pytest.mark.parametrize('reason', _PATCH_ID_REJECTS)
    async def test_a_refused_verdict_vetoes_the_flip(
        self, mock_orch_config, landing_producer: AsyncMock, reason: LandingReason,
    ) -> None:
        h = _wired_ancestry_harness(mock_orch_config)
        file_escalation = MagicMock()
        h._file_unattributed_landing_escalation = file_escalation
        landing_producer.return_value = _refused_verdict(reason)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        file_escalation.assert_not_called()

    async def test_the_stamp_anchors_on_the_evidence_not_the_branch_tip(
        self, mock_orch_config, landing_producer: AsyncMock,
    ) -> None:
        """The producer, not the gate, chooses the anchor: a citing merge
        commit or patch-id twin on main, never the live tip itself.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        merge_commit_sha = 'c' * 40
        landing_producer.return_value = _landed_verdict(merge_commit_sha)

        assert await h._already_landed_dispatch_gate('42') is True

        mark = cast(AsyncMock, h._mark_in_progress_done)
        assert mark.await_args is not None
        assert mark.await_args.args[1] == merge_commit_sha
        assert mark.await_args.args[1] != _TIP_SHA


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateAncestryDelegatesToHelper:
    """The ancestry path hands the whole decision to ``branch_work_landed`` (task 4648).

    The escalation veto and the ``is_ancestor(branch, main)`` gate still run
    BEFORE the producer is consulted; everything after them, from the
    degenerate-branch guard to attribution, is the producer's.
    """

    async def test_rejected_verdict_returns_false_no_mark_done(
        self, mock_orch_config, landing_producer: AsyncMock,
    ) -> None:
        h = _wired_ancestry_harness(mock_orch_config)
        landing_producer.return_value = _refused_verdict(LandingReason.not_landed)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()


def _open_escalation(
    esc_id: str, *, level: int, severity: str = 'blocking',
) -> Escalation:
    """A pending escalation row as ``get_by_task(tid, status='pending')`` yields.

    ``filing_claimant_run_id`` is left at its default ``None`` (the shape every
    legacy row carries, and every producer that still does not stamp it — task
    3550 added stamping to TaskWorkflow, Harness and the agent filing tools)
    — irrelevant for L1/L2, and
    for L0 the gate passes ``live_claimant=False``, which classifies
    ``dead_l0`` identity-independently.
    """
    return Escalation(
        id=esc_id,
        task_id='42',
        agent_role='implementer',
        severity=severity,
        category='design_concern',
        summary='an open handoff on task 42',
        level=level,
    )


def _queue_with_open(*records: Escalation) -> MagicMock:
    """An escalation queue whose ANY-LEVEL pending read yields *records*.

    ``has_open_l1`` is pinned False on purpose: after task 3534 the gate's veto
    no longer consults the L1-only read at all, so a sub-case that vetoes here
    proves it went through ``get_by_task(..., status='pending')``.  The stub
    stays because ``landing_evidence._file_unattributed_landing_escalation``
    still consumes ``has_open_l1`` for escalation dedup.
    """
    queue = MagicMock()
    queue.has_open_l1 = MagicMock(return_value=False)
    queue.get_by_task = MagicMock(return_value=list(records))
    queue.make_id = MagicMock(return_value='esc-42-1')
    return queue


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateAnyLevelVeto:
    """The escalation veto is ANY-LEVEL, not L1-only (task 3534 / PRD eta-0).

    Spec ``docs/task-escalation-state-spec.md`` E8: the gate's veto read was
    ``has_open_l1(task_id)``, which silently missed an L2-only (or L0-only)
    open escalation while its own docstring claimed parity with
    ``_reconcile_one_stranded``, which went any-level in W10.  The veto now
    asks the shared ``escalation.pins.classify_pins(...).vetoes_done_flip``
    predicate over the same ``get_by_task(tid, status='pending')`` rows the
    resolver's row (f) reads.

    Every sub-case starts from an otherwise-flipping ancestry setup and
    overrides only ``_escalation_queue``.
    """

    async def test_open_l2_only_vetoes_flip(self, mock_orch_config) -> None:
        """PRD boundary row #10: an L2-ONLY open escalation must veto the
        auto-done flip.  This is the exact record the old L1-only read missed.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        cast(MagicMock, h._escalation_queue).get_by_task.assert_called_with(
            '42', status='pending',
        )
        # "Neither layer costs an extra store read": the durable arbitration
        # hold and this veto both decide from the SAME single read.
        assert cast(MagicMock, h._escalation_queue).get_by_task.call_count == 1

    async def test_open_l0_only_vetoes_flip(self, mock_orch_config) -> None:
        """An L0-ONLY open escalation also vetoes the flip.

        Under ``live_claimant=False`` it classifies ``dead_l0`` — which does
        NOT pin recovery (spec S4) but DOES veto a done-flip (PRD D3: "any
        non-info open record still vetoes MARK_DONE").  Pins the liveness-
        independence this call site relies on: both ``dead_l0`` and
        ``queue_handoff`` set ``vetoes_done_flip``, so the gate never has to
        resolve the task's live claimant.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-3', level=0),
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    async def test_open_l1_still_vetoes_flip(self, mock_orch_config) -> None:
        """An open L1 escalation is a deliberate handoff — never second-guessed,
        even though is_ancestor is True.  Migrated off the ``has_open_l1``
        stub (dead wiring once the gate stops reading it) onto the any-level
        pending read, so "existing L1 behaviour unchanged" stays verified.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-7', level=1),
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    async def test_veto_logs_the_pinning_escalation_ids(
        self, mock_orch_config, caplog,
    ) -> None:
        """The veto is LOUD: it names the task and the pinning escalation ids.

        The "not silently" half of boundary row #10 — a task that stops
        auto-flipping must say which record pinned it.  (Task beta/3535
        replaces this log line with the structured ``recovery_vetoed``
        emission plus streak dedup.)
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        messages = [
            r.getMessage() for r in caplog.records
            if r.name == 'orchestrator.harness'
        ]
        # ``'task 42'``, not ``'42'``: the bare id is already a substring of
        # ``'esc-42-9'``, so asserting on it verified nothing about task
        # attribution — a line that dropped the task entirely would pass.
        assert any('esc-42-9' in m and 'task 42' in m for m in messages), messages

    async def test_info_severity_only_does_not_veto(
        self, mock_orch_config,
    ) -> None:
        """PRD D3 / boundary row #8: an ``info``-severity record NEVER pins.

        Guards the choice of predicate rather than the change itself: green
        both before and after, but RED against a naive ``bool(pending_rows)``
        veto — under which a genuinely-landed task carrying one
        ``escalate_info`` annotation would never auto-flip and would
        re-dispatch every tick forever.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-5', level=0, severity='info'),
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()

    async def test_no_open_records_flips_as_today(self, mock_orch_config) -> None:
        """A read that succeeded and found nothing is not a veto — the flip
        happens exactly as it does today.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._escalation_queue = _queue_with_open()

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()

    async def test_escalation_store_read_failure_vetoes_flip(
        self, mock_orch_config, caplog,
    ) -> None:
        """A FAILED read is the third state, not "no open escalations".

        The caller-side half of ``classify_pins``'s STORE-CORRECTNESS
        CONTRACT (obligation 2: "Pass ``records=None`` when that read could
        not be performed or failed. Never substitute ``[]``") — substituting
        ``[]`` here would be the exact esc-3163 collapse the contract exists
        to forbid.  So an unreadable store must veto, must not raise, and
        must say so.

        Letting the exception propagate instead would make the disposition
        invisible: the scheduler's generic ``_consult_already_landed``
        catch-all logs a traceback and fails open for the WHOLE gate, so
        "never phantom-done on an unreadable store" would be incidental
        rather than local and testable.

        Note this is distinct from ``_escalation_queue is None``, which keeps
        failing OPEN — a deliberately-absent queue (escalations disabled) is
        not the same thing as a failed read.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._escalation_queue = MagicMock()
        h._escalation_queue.has_open_l1 = MagicMock(return_value=False)
        h._escalation_queue.get_by_task = MagicMock(
            side_effect=OSError('escalation queue dir unreadable'),
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        messages = [
            r.getMessage() for r in caplog.records
            if r.name == 'orchestrator.harness' and r.levelno >= logging.WARNING
        ]
        # ``'task 42'``, not ``'42'``: the bare id is satisfied by any message
        # merely mentioning an escalation id or the queue dir path, so the
        # task-attribution half of this assertion could never fail.
        assert any(
            'task 42' in m and ('store' in m.lower() or 'read' in m.lower())
            for m in messages
        ), messages


def _wired_marker_harness(
    mock_orch_config, *, marker_sha, branch_base_sha, marker_is_ancestor_of_base,
) -> Harness:
    """Bare harness with is_ancestor(branch, main) False, so the ancestry
    path never engages and the marker path is what's under test.

    ``is_ancestor`` is mocked with a side_effect function because it is
    called with two DIFFERENT argument pairs in this path: once for
    ``(branch, main_branch)`` (must be False to reach the marker path) and
    once for ``(marker_sha, branch_base_sha)`` (the stale-marker check).

    ``commit_effect_present_in_main`` defaults to True (a healthy marker
    landing) and ``_escalation_queue`` is a wired MagicMock with
    ``has_open_l1`` defaulting False — callers flip either to exercise the
    CANDIDATE-mode reject-and-escalate path (task 2678).

    ``get_by_task`` returns ``[]`` EXPLICITLY (task 3534): the gate's veto now
    reads the any-level pending rows, and leaving that to
    ``MagicMock.__iter__``'s empty default would make "no open escalations"
    an accident of the mock rather than a stated precondition.
    """
    h = _build_harness(mock_orch_config)
    h.git_ops = MagicMock()
    h.git_ops.config.branch_prefix = 'task/'
    h.git_ops.config.main_branch = 'main'

    # Branch ref does not exist (deleted post-merge) — this is what routes
    # the gate past the ancestry check and into the marker path at all.
    h.git_ops.resolve_branch_sha = AsyncMock(return_value=None)

    async def _is_ancestor(a, b):
        if a == marker_sha and b == branch_base_sha:
            return marker_is_ancestor_of_base
        raise AssertionError(f'unexpected is_ancestor call: {a!r}, {b!r}')

    h.git_ops.is_ancestor = AsyncMock(side_effect=_is_ancestor)
    h.git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)
    h.git_ops.find_task_citation_commit = AsyncMock(return_value=None)
    h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=True)

    h.scheduler.get_task = AsyncMock(
        return_value={'id': '42', 'metadata': {'branch_base_sha': branch_base_sha}},
    )
    h._branch_is_degenerate = AsyncMock(return_value=False)
    h._mark_in_progress_done = AsyncMock()
    h._escalation_queue = MagicMock()
    h._escalation_queue.has_open_l1 = MagicMock(return_value=False)
    h._escalation_queue.get_by_task = MagicMock(return_value=[])
    h._escalation_queue.make_id = MagicMock(return_value='esc-42-1')
    # Explicit (task 4499): the filer's auto-dismiss guard reads a TERMINAL
    # record on this citation, so "nothing was previously adjudicated" must be
    # STATED, not inherited from MagicMock's truthy auto-child — which would
    # silently suppress every filing these tests assert on.
    h._escalation_queue.find_terminal_by_citation = MagicMock(return_value=None)
    return h


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateMarkerPath:
    """Branch-deleted merge-marker path (RED until step-8 for the
    effect-absent/escalation sub-case; the flip and stale-marker sub-cases
    are already green from step-8's predecessor and are extended here with
    escalation-queue assertions).

    resolve_branch_sha(branch) is None in all sub-cases (the branch ref is
    gone), so the ancestry path never engages — only the marker path can
    produce a result.
    """

    async def test_marker_found_and_not_stale_flips_to_done(
        self, mock_orch_config,
    ) -> None:
        """A merge marker on main, not an ancestor of branch_base_sha (i.e.
        it postdates this incarnation's creation point), with its effect
        PRESENT at main HEAD (the CANDIDATE-mode helper accepts) -> flips
        to done, anchored on the marker sha, with no escalation filed.

        Also pins the citation-laziness fix (task 2313 review): the marker
        path doesn't consume a citation, so find_task_citation_commit must
        never be called here.
        """
        marker_sha = 'b' * 40
        branch_base_sha = 'e' * 40
        h = _wired_marker_harness(
            mock_orch_config,
            marker_sha=marker_sha,
            branch_base_sha=branch_base_sha,
            marker_is_ancestor_of_base=False,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        call_args = cast(AsyncMock, h._mark_in_progress_done).await_args
        assert call_args is not None
        assert call_args.args[0] == '42'
        assert call_args.args[1] == marker_sha
        assert call_args.args[3] == 'dispatch-gate-marker-found'
        cast(AsyncMock, h.git_ops.find_task_citation_commit).assert_not_called()
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()

    async def test_marker_found_not_stale_effect_absent_escalates_no_mark_done(
        self, mock_orch_config,
    ) -> None:
        """The task-1175 shape: a merge marker on main, not stale, but its
        effect has been REVERTED at current main HEAD
        (commit_effect_present_in_main False) — a later commit on main
        undid exactly the paths the marker's landing touched. The
        CANDIDATE-mode FIX 1' guard (task 2678, closing the 1175 clobber in
        this path) must reject it: no _mark_in_progress_done call, the gate
        returns False, and exactly one dedup-guarded 'provenance_unattributed'
        L1 escalation is filed carrying the branch, the marker sha, and the
        'effect_absent' reason. has_open_l1('42') is consulted for dedup
        before the escalation is filed.

        RED: the marker path has no effect check / escalation yet, so it
        would flip to done regardless of commit_effect_present_in_main.
        """
        marker_sha = 'b' * 40
        branch_base_sha = 'e' * 40
        h = _wired_marker_harness(
            mock_orch_config,
            marker_sha=marker_sha,
            branch_base_sha=branch_base_sha,
            marker_is_ancestor_of_base=False,
        )
        h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=False)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

        # Since task 3534 the gate's own veto reads get_by_task(..., pending)
        # instead of has_open_l1, so the escalation helper's dedup check is
        # the ONLY has_open_l1 caller left — but assert on the most recent
        # call rather than an exact count, so a future dedup site added
        # elsewhere in the path doesn't turn this into a brittle failure.
        # Since task 3116 that dedup is deliberately CATEGORY-SCOPED: passing a
        # bare task_id matched any open L1, so an unrelated escalation silently
        # suppressed a provenance_unattributed filing (the task-4105 two-way
        # blindfold). The kwarg here is the point of that change, not drift.
        cast(MagicMock, h._escalation_queue).has_open_l1.assert_called_with(
            '42', category='provenance_unattributed',
        )
        cast(MagicMock, h._escalation_queue).submit.assert_called_once()
        esc = cast(MagicMock, h._escalation_queue).submit.call_args[0][0]
        assert esc.category == 'provenance_unattributed'
        assert esc.task_id == '42'
        assert 'task/42' in esc.detail
        assert marker_sha in esc.detail
        assert 'effect_absent' in esc.detail

    async def test_the_marker_arm_keeps_the_merge_marker_policy(
        self, mock_orch_config, landing_producer: AsyncMock,
    ) -> None:
        """The marker arm stays on ``validate_landing_evidence`` CANDIDATE (task 4648).

        A deleted ref leaves no live tip for ``branch_work_landed`` to judge,
        so the escalated verdict carries ``merge_marker``, not ``patch_id``,
        and the producer the live-branch arms use is never asked.  The ruling
        is recorded in
        ``docs/prds/landed-not-done-recovery.effect-present-caller-enumeration.md`` §7.
        """
        h = _wired_marker_harness(
            mock_orch_config,
            marker_sha='b' * 40,
            branch_base_sha='e' * 40,
            marker_is_ancestor_of_base=False,
        )
        h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=False)
        file_escalation = MagicMock()
        h._file_unattributed_landing_escalation = file_escalation

        assert await h._already_landed_dispatch_gate('42') is False

        file_escalation.assert_called_once()
        _task_id, _branch, verdict = file_escalation.call_args.args
        assert verdict.reason is LandingReason.effect_absent
        assert verdict.method is LandingMethod.merge_marker
        landing_producer.assert_not_awaited()

    async def test_stale_marker_vetoes_flip(self, mock_orch_config) -> None:
        """A marker that IS an ancestor of branch_base_sha predates this
        incarnation (branch was deleted + re-created under the same task
        id) -> vetoes the flip, no _mark_in_progress_done call, and no
        escalation (this veto predates the incarnation entirely — it is
        not an unattributed-landing signal, so it must not be treated as
        one).
        """
        marker_sha = 'b' * 40
        branch_base_sha = 'e' * 40
        h = _wired_marker_harness(
            mock_orch_config,
            marker_sha=marker_sha,
            branch_base_sha=branch_base_sha,
            marker_is_ancestor_of_base=True,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()


def _wired_off_main_tip_harness(
    mock_orch_config, tmp_path: Path | None = None,
) -> Harness:
    """A harness whose live tip resolves but is NOT an ancestor of main.

    Only the off-main-tip arm can decide: the branch exists, so the
    merge-marker search must never run, and it is wired to fail loudly if it
    does; ``get_main_sha`` is wired too and must never be awaited, since the
    stamp anchors on the verdict's evidence and never on main HEAD.  The
    module's ``landing_producer`` decides.  The escalation queue is a
    recording ``MagicMock`` stating "no pending records" and nothing
    previously adjudicated.  With *tmp_path*, the event store and the
    recovery-emission config are real, as in :func:`_emitting_harness`.
    """
    h = _build_harness(mock_orch_config)
    h.git_ops = MagicMock()
    h.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)
    h.git_ops.is_ancestor = AsyncMock(return_value=False)
    h.git_ops.find_merge_marker = AsyncMock(
        side_effect=AssertionError('a live branch never reaches the marker search'),
    )
    h.git_ops.get_main_sha = AsyncMock(return_value='c' * 40)
    h.git_ops.config.branch_prefix = 'task/'
    h.git_ops.config.main_branch = 'main'
    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
    h._mark_in_progress_done = AsyncMock()
    queue = MagicMock()
    queue.has_open_l1 = MagicMock(return_value=False)
    queue.get_by_task = MagicMock(return_value=[])
    queue.make_id = MagicMock(return_value='esc-42-1')
    # Explicit (task 4499): the filer's auto-dismiss guard reads a TERMINAL
    # record on this citation, so "nothing was previously adjudicated" must be
    # STATED, not inherited from MagicMock's truthy auto-child — which would
    # silently suppress every filing these tests assert on.
    queue.find_terminal_by_citation = MagicMock(return_value=None)
    h._escalation_queue = queue
    if tmp_path is not None:
        h.event_store = EventStore(tmp_path / 'runs.db', 'run-test')
        h.config.recovery_emission = RecoveryEmissionConfig()
    return h


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateOffMainTipArmDispositions:
    """The off-main-tip arm: a rebase or manual landing, or none at all.

    One test per disposition: escalate an unattributed landing, stamp an
    attributed one, dispatch a branch that did not land.  A multi-commit
    squash reads as not landed (test_already_landed_gate_patch_id.py).

    ``is_ancestor(branch, main)`` is False and the branch exists in every
    sub-case, so only the off-main-tip arm can produce a result.  It decides
    by the patch-id producer, so the decaying content-equivalence entry
    condition it replaced is never consulted.
    """

    async def test_an_unattributed_landing_escalates_no_mark_done(
        self, mock_orch_config, landing_producer: AsyncMock,
    ) -> None:
        """The work is on main but nothing attributes it to the task: the
        gate must not fabricate an anchor from main HEAD, must not mark done,
        and files exactly one 'provenance_unattributed' escalation carrying
        the branch and the reason.
        """
        h = _wired_off_main_tip_harness(mock_orch_config)
        mark = cast(AsyncMock, h._mark_in_progress_done)
        queue = cast(MagicMock, h._escalation_queue)
        landing_producer.return_value = _refused_verdict(
            LandingReason.no_attribution, evidence_source='none',
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        mark.assert_not_awaited()
        cast(AsyncMock, h.git_ops.get_main_sha).assert_not_awaited()
        queue.submit.assert_called_once()
        esc = queue.submit.call_args[0][0]
        assert esc.category == 'provenance_unattributed'
        assert esc.task_id == '42'
        assert 'task/42' in esc.detail
        assert 'no_attribution' in esc.detail
        assert "'evidence_source': 'none'" in esc.detail

    @pytest.mark.parametrize('evidence_sha', [
        pytest.param('d' * 40, id='cited'),
        pytest.param('7' * 40, id='uncited-patch-id-twin'),
    ])
    async def test_an_attributed_landing_marks_done_on_its_evidence(
        self, mock_orch_config, landing_producer: AsyncMock, evidence_sha: str,
    ) -> None:
        """Stamped done anchored on the evidence that attributes it (a
        citing commit, else the tip's patch-id twin), never main HEAD, and no
        escalation is filed.
        """
        h = _wired_off_main_tip_harness(mock_orch_config)
        mark = cast(AsyncMock, h._mark_in_progress_done)
        landing_producer.return_value = _landed_verdict(evidence_sha)

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        mark.assert_awaited_once()
        assert mark.await_args is not None
        assert mark.await_args.args[0] == '42'
        assert mark.await_args.args[1] == evidence_sha
        assert mark.await_args.args[3] == 'dispatch-gate-content-equivalent'
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()
        cast(AsyncMock, h.git_ops.get_main_sha).assert_not_awaited()

    async def test_a_not_landed_branch_dispatches_normally(
        self, mock_orch_config, landing_producer: AsyncMock,
    ) -> None:
        """No landing at all: the gate returns False so the task dispatches
        normally this tick, with no escalation and no mark_done.
        """
        h = _wired_off_main_tip_harness(mock_orch_config)
        landing_producer.return_value = _not_contained()

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()


class TestAlreadyLandedDispatchGateGetMainShaFallbackGrepGuard:
    """Source-level guard (task 2678): the silent
    ``citation or await self.git_ops.get_main_sha()`` fallback must be
    fully deleted from the off-main-tip path, not merely made
    unreachable — a future edit must not resurrect it as dead code.
    """

    def test_get_main_sha_fallback_expression_absent(self) -> None:
        content = _HARNESS_SRC_PATH.read_text()
        assert 'or await self.git_ops.get_main_sha()' not in content, (
            'harness.py still contains the silent get_main_sha() fallback '
            'expression; task 2678 replaces it with '
            'validate_landing_evidence + escalate-instead-of-stamp.'
        )


# ---------------------------------------------------------------------------
# task 2677 step-7/step-8 — a done_evidence_stale rejection surfacing from
# _mark_in_progress_done's scheduler.mark_done call must gate the dispatch
# (the contested task must never dispatch) on the tick it is first observed,
# and a later tick at the SAME reopen_at must NOT re-attempt the doomed
# write nor file a duplicate escalation. Unlike every other test in this
# file, _mark_in_progress_done is NOT replaced by a bare AsyncMock here —
# these tests exercise the real StaleEvidenceRejection catch (step-6) plus
# the gate's own should_skip pre-check (step-8), so scheduler.mark_done
# itself is the stale-rejecting double and _mark_in_progress_done runs for
# real.
# ---------------------------------------------------------------------------

_STALE_REOPEN_AT = '2026-07-15T00:00:00+00:00'


def _wire_stale_evidence_mark_done(h: Harness, *, evidence_commit: str) -> None:
    """Replace scheduler.mark_done with one that always rejects as stale.

    Mirrors test_reconcile_stranded.py's identically-named helper (kept
    local rather than imported since that module is a test file, not a
    shared fixture library).
    """
    from orchestrator.scheduler import StaleEvidenceRejection

    async def _stale_reject(tid, *, kind, sha, note=None):
        raise StaleEvidenceRejection(
            task_id=tid,
            evidence_commit=evidence_commit,
            evidence_committed_at='2026-07-10T00:00:00+00:00',
            reopen_at=_STALE_REOPEN_AT,
            agent_id='claude-recon-x',
            raw="success=False payload={'error': 'done_evidence_stale'}",
        )
    h.scheduler.mark_done = AsyncMock(side_effect=_stale_reject)


def _let_mark_in_progress_done_run_for_real(h: Harness, tmp_path) -> None:
    """Undo the blanket ``h._mark_in_progress_done = AsyncMock()`` the
    ``_wired_*_harness`` builders apply, and give ``_resolve_task_worktree``
    a real (nonexistent) tmp path so the worktree-cleanup branch is a clean
    no-op (``worktree_path.exists()`` is False) rather than tripping over
    ``h.git_ops``'s auto-mocked ``warm_lane_pool``/``cleanup_worktree``.
    """
    del h._mark_in_progress_done
    h.git_ops.warm_lane_pool = None
    h.git_ops.worktree_base = tmp_path / 'worktrees'


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateStaleEvidenceConflict:
    """RED until step-8 adds the should_skip pre-check to the gate."""

    async def test_ancestry_path_gates_and_does_not_reattempt(
        self, mock_orch_config, tmp_path,
    ) -> None:
        from escalation.queue import EscalationQueue

        from orchestrator.provenance_conflict import ProvenanceConflictSink

        h = _wired_ancestry_harness(mock_orch_config)
        _let_mark_in_progress_done_run_for_real(h, tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )
        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {'reopen_at': _STALE_REOPEN_AT}},
        )
        _wire_stale_evidence_mark_done(h, evidence_commit='a' * 40)

        result_1 = await h._already_landed_dispatch_gate('42')

        assert result_1 is True, 'a contested task must not dispatch'
        assert cast(AsyncMock, h.scheduler.mark_done).await_count == 1
        conflicts = [
            e for e in h._escalation_queue.get_by_task('42', status='pending')
            if e.category == 'provenance_conflict'
        ]
        assert len(conflicts) == 1, f'expected exactly one pending L2, got {len(conflicts)}'
        assert conflicts[0].level == 2
        assert conflicts[0].severity == 'urgent'

        result_2 = await h._already_landed_dispatch_gate('42')

        assert result_2 is True, 'must stay gated on a repeat tick'
        assert cast(AsyncMock, h.scheduler.mark_done).await_count == 1, (
            'a repeat tick at the same reopen_at must not re-attempt the '
            'already-rejected write'
        )
        conflicts_after = [
            e for e in h._escalation_queue.get_by_task('42', status='pending')
            if e.category == 'provenance_conflict'
        ]
        assert len(conflicts_after) == 1, 'must not file a duplicate escalation'

    async def test_off_main_tip_path_gates_and_does_not_reattempt(
        self, mock_orch_config, tmp_path, landing_producer: AsyncMock,
    ) -> None:
        from escalation.queue import EscalationQueue

        from orchestrator.provenance_conflict import ProvenanceConflictSink

        # The off-main-tip arm must accept the landing and proceed to the
        # done-write — only then can the stale-evidence rejection this test
        # exercises fire from mark_done.
        landing_producer.return_value = _landed_verdict('c' * 40)
        h = _wired_off_main_tip_harness(mock_orch_config)
        _let_mark_in_progress_done_run_for_real(h, tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )
        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {'reopen_at': _STALE_REOPEN_AT}},
        )
        _wire_stale_evidence_mark_done(h, evidence_commit='c' * 40)

        result_1 = await h._already_landed_dispatch_gate('42')

        assert result_1 is True, 'a contested task must not dispatch'
        assert cast(AsyncMock, h.scheduler.mark_done).await_count == 1
        conflicts = [
            e for e in h._escalation_queue.get_by_task('42', status='pending')
            if e.category == 'provenance_conflict'
        ]
        assert len(conflicts) == 1, f'expected exactly one pending L2, got {len(conflicts)}'

        result_2 = await h._already_landed_dispatch_gate('42')

        assert result_2 is True, 'must stay gated on a repeat tick'
        assert cast(AsyncMock, h.scheduler.mark_done).await_count == 1, (
            'a repeat tick at the same reopen_at must not re-attempt the '
            'already-rejected write'
        )
        conflicts_after = [
            e for e in h._escalation_queue.get_by_task('42', status='pending')
            if e.category == 'provenance_conflict'
        ]
        assert len(conflicts_after) == 1, 'must not file a duplicate escalation'


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateArbitrationHoldIsDurable:
    """A contested task stays withheld even when the sink's memo is COLD.

    ``TestAlreadyLandedDispatchGateStaleEvidenceConflict`` above only ever
    exercises the WARM path: tick 1 files the conflict in the same process,
    so tick 2 short-circuits on ``should_skip``'s in-memory memo.  That memo
    is in-memory ONLY (``provenance_conflict.py``'s class docstring:
    "an in-memory per-task memo ... Lost on restart"), so it is empty on any
    process that did not itself file the conflict — after a fleet
    redeploy / watchdog restart, or when a DIFFERENT writer site filed it
    (``merge_queue.py``'s two ``ProvenanceConflictSink`` call sites).

    With the memo cold, the task-3534 any-level veto is what sees the
    pending ``provenance_conflict`` record — and its answer is ``return
    False``, which means "I abstain; dispatch normally".  For a task under
    arbitration that is precisely the outcome this gate's docstring forbids
    ("a contested task must never dispatch while under arbitration"), so the
    hold has to be decided from DURABLE store state, ordered above that
    veto.  Ordering alone cannot deliver the invariant; durability can.
    """

    async def test_cold_sink_memo_still_withholds_contested_task(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """A restart clears the memo but not the escalation store."""
        from escalation.queue import EscalationQueue

        from orchestrator.provenance_conflict import ProvenanceConflictSink

        h = _wired_ancestry_harness(mock_orch_config)
        _let_mark_in_progress_done_run_for_real(h, tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )
        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {'reopen_at': _STALE_REOPEN_AT}},
        )
        _wire_stale_evidence_mark_done(h, evidence_commit='a' * 40)

        result_1 = await h._already_landed_dispatch_gate('42')

        assert result_1 is True, 'a contested task must not dispatch'
        conflicts = [
            e for e in h._escalation_queue.get_by_task('42', status='pending')
            if e.category == 'provenance_conflict'
        ]
        assert len(conflicts) == 1, f'expected exactly one pending L2, got {len(conflicts)}'
        assert conflicts[0].level == 2
        assert conflicts[0].severity == 'urgent'

        # SIMULATE THE RESTART: a brand-new sink (empty memo) over the SAME
        # durable queue — exactly what a fleet redeploy leaves behind.
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )
        assert h._provenance_conflict_sink.should_skip(
            '42', reopen_at=_STALE_REOPEN_AT,
        ) is False, (
            'the fresh sink must have a COLD memo — otherwise this test '
            'silently degrades into a re-run of the warm-memo case above'
        )

        result_2 = await h._already_landed_dispatch_gate('42')

        assert result_2 is True, (
            'a contested task must never dispatch while under arbitration — '
            'including on a cold memo'
        )
        assert cast(AsyncMock, h.scheduler.mark_done).await_count == 1, (
            'the durable hold must short-circuit BEFORE re-attempting the '
            'already-rejected done-write, not merely re-derive it'
        )
        conflicts_after = [
            e for e in h._escalation_queue.get_by_task('42', status='pending')
            if e.category == 'provenance_conflict'
        ]
        assert len(conflicts_after) == 1, 'must not file a duplicate escalation'

    async def test_conflict_filed_by_a_different_writer_site_still_withholds(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """The hold is decided from store state, with no write attempt at all.

        The second half of the cold-memo finding, which a restart-only test
        would miss: the conflict was filed by a sink instance this harness
        has never owned (mirroring the merge-queue writer sites), so the
        harness's own memo has never held task 42 at any point.
        """
        from escalation.queue import EscalationQueue

        from orchestrator.provenance_conflict import ProvenanceConflictSink

        h = _wired_ancestry_harness(mock_orch_config)
        _let_mark_in_progress_done_run_for_real(h, tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')
        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {'reopen_at': _STALE_REOPEN_AT}},
        )
        _wire_stale_evidence_mark_done(h, evidence_commit='a' * 40)

        # A writer site the harness does not own files the conflict.
        foreign_sink = ProvenanceConflictSink(escalation_queue=h._escalation_queue)
        foreign_sink.record(
            task_id='42',
            evidence_commit='a' * 40,
            evidence_committed_at='2026-07-10T00:00:00+00:00',
            reopen_at=_STALE_REOPEN_AT,
            agent_id='claude-merge-worker',
            gate_source='merge-queue',
        )

        # The harness's own sink has never seen task 42.
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is True, (
            'a conflict filed by another writer site binds this gate too'
        )
        assert cast(AsyncMock, h.scheduler.mark_done).await_count == 0, (
            'the hold is decided purely from durable store state — no doomed '
            'write attempt at all'
        )

    async def test_repeated_hold_logs_info_once_then_debug(
        self, mock_orch_config, tmp_path, caplog,
    ) -> None:
        """The hold is loud ONCE, then quiet (amendment,
        reviewer_comprehensive observability_log_volume).

        The any-level veto does not need this — it abstains, so the task
        dispatches and stops being a gate candidate.  The hold does: it
        re-fires on every dispatch tick for as long as the arbitration
        stands, and on this (cold-memo) path it short-circuits above
        ``_mark_in_progress_done``, so nothing ever re-warms ``should_skip``
        to quiet it.  Unconditional INFO would be one line per scheduler
        tick, indefinitely, until a human resolved the L2 — the log spam
        INV-4 forbids.
        """
        from escalation.queue import EscalationQueue

        from orchestrator.provenance_conflict import ProvenanceConflictSink

        h = _wired_ancestry_harness(mock_orch_config)
        _let_mark_in_progress_done_run_for_real(h, tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')
        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {'reopen_at': _STALE_REOPEN_AT}},
        )
        _wire_stale_evidence_mark_done(h, evidence_commit='a' * 40)

        ProvenanceConflictSink(escalation_queue=h._escalation_queue).record(
            task_id='42',
            evidence_commit='a' * 40,
            evidence_committed_at='2026-07-10T00:00:00+00:00',
            reopen_at=_STALE_REOPEN_AT,
            agent_id='claude-merge-worker',
            gate_source='merge-queue',
        )
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )

        with caplog.at_level(logging.DEBUG, logger='orchestrator.harness'):
            assert await h._already_landed_dispatch_gate('42') is True
            assert await h._already_landed_dispatch_gate('42') is True
            assert await h._already_landed_dispatch_gate('42') is True

        held = [
            r for r in caplog.records
            if r.name == 'orchestrator.harness'
            and 'under provenance arbitration' in r.getMessage()
        ]
        assert len(held) == 3, [r.getMessage() for r in held]
        assert [r.levelno for r in held] == [
            logging.INFO, logging.DEBUG, logging.DEBUG,
        ], [(r.levelname, r.getMessage()) for r in held]
        assert 'task 42' in held[0].getMessage()

    async def test_hold_released_then_returning_is_loud_again(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """Quieting the repeats must not silence a hold that RETURNS.

        The streak state is dropped whenever the hold does not bind (here:
        the operator resolves the arbitration), so a later conflict on the
        same task announces itself at INFO rather than being folded into a
        stale streak.
        """
        from escalation.queue import EscalationQueue

        from orchestrator.provenance_conflict import ProvenanceConflictSink

        h = _wired_ancestry_harness(mock_orch_config)
        _let_mark_in_progress_done_run_for_real(h, tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')
        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {'reopen_at': _STALE_REOPEN_AT}},
        )
        _wire_stale_evidence_mark_done(h, evidence_commit='a' * 40)

        foreign_sink = ProvenanceConflictSink(escalation_queue=h._escalation_queue)
        escalation_id = foreign_sink.record(
            task_id='42',
            evidence_commit='a' * 40,
            evidence_committed_at='2026-07-10T00:00:00+00:00',
            reopen_at=_STALE_REOPEN_AT,
            agent_id='claude-merge-worker',
            gate_source='merge-queue',
        )
        assert escalation_id is not None
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )

        assert await h._already_landed_dispatch_gate('42') is True
        assert h._provenance_conflict_sink.note_hold_observed(
            '42', reopen_at=_STALE_REOPEN_AT,
        ) is False, 'the first tick must have registered the streak'

        # The operator arbitrates: the hold releases, and the streak state
        # goes with it on the next tick that observes the release.
        h._escalation_queue.resolve(escalation_id, 'operator arbitrated')
        await h._already_landed_dispatch_gate('42')

        assert h._provenance_conflict_sink.note_hold_observed(
            '42', reopen_at=_STALE_REOPEN_AT,
        ) is True, 'a returning hold must be loud again, not silently folded'

    async def test_moved_on_reopen_at_dispatches_instead_of_flipping(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """A re-reopened task escapes the hold by DISPATCHING, never by flipping.

        Pins the whole composition.  The hold inherits ``should_skip``'s
        reopen_at invalidation arm, so a task reopened AFTER its conflict was
        recorded releases it — otherwise fixing the cold-memo bug would
        install a permanent wedge (no dispatch, no flip, until a human
        resolves the L2).  But releasing the hold does NOT mean "flip to
        done": the still-pending urgent L2 is then caught by the any-level
        veto, so the gate abstains and the task dispatches.

        Both the pre-3534 behaviour (auto-flip past a pending L2 — defect E8
        itself) and a reopen_at-agnostic hold (wedge forever) are wrong here.
        Only this composition is right.
        """
        from escalation.queue import EscalationQueue

        from orchestrator.provenance_conflict import ProvenanceConflictSink

        later_reopen_at = '2026-07-20T00:00:00+00:00'

        h = _wired_ancestry_harness(mock_orch_config)
        _let_mark_in_progress_done_run_for_real(h, tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')
        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {'reopen_at': later_reopen_at}},
        )
        _wire_stale_evidence_mark_done(h, evidence_commit='a' * 40)

        # The conflict was recorded against the EARLIER reopen_at.
        ProvenanceConflictSink(escalation_queue=h._escalation_queue).record(
            task_id='42',
            evidence_commit='a' * 40,
            evidence_committed_at='2026-07-10T00:00:00+00:00',
            reopen_at=_STALE_REOPEN_AT,
            agent_id='claude-merge-worker',
            gate_source='merge-queue',
        )
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False, (
            'a re-reopened task must escape by dispatching, not by the gate '
            'flipping it past a still-pending L2'
        )
        assert cast(AsyncMock, h.scheduler.mark_done).await_count == 0, (
            'the any-level veto must catch the still-pending L2 before any '
            'done-write is attempted'
        )

    async def test_plain_l2_pin_still_dispatches(self, mock_orch_config) -> None:
        """A pending non-provenance L2 belongs to the any-level VETO, not the
        hold — so the gate abstains (``False``) rather than withholding.

        Regression fence against broadening the hold into the veto's cases.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    async def test_store_unavailable_still_vetoes_without_holding(
        self, mock_orch_config, caplog,
    ) -> None:
        """The hold cannot see an unreadable store, and must not pretend to.

        ``records=None`` is the caller's ``store_unavailable`` disposition
        (task 3534 step-4) — the gate still ABSTAINS with its WARNING rather
        than the hold silently upgrading a failed read into a withhold.
        """
        h = _wired_ancestry_harness(mock_orch_config)
        h._escalation_queue = MagicMock()
        h._escalation_queue.has_open_l1 = MagicMock(return_value=False)
        h._escalation_queue.get_by_task = MagicMock(
            side_effect=OSError('escalation queue dir unreadable'),
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        messages = [
            r.getMessage() for r in caplog.records
            if r.name == 'orchestrator.harness' and r.levelno >= logging.WARNING
        ]
        # ``'task 42'``, not ``'42'``: the bare id is satisfied by any message
        # merely mentioning an escalation id or the queue dir path, so the
        # task-attribution half of this assertion could never fail.
        assert any(
            'task 42' in m and ('store' in m.lower() or 'read' in m.lower())
            for m in messages
        ), messages


def _wired_absent_branch_harness(mock_orch_config) -> Harness:
    """Bare harness with resolve_branch_sha -> None (branch ref does not
    exist, e.g. a fresh task never dispatched).  is_ancestor is wired but
    must NEVER be called, and neither may the module's ``landing_producer`` —
    the cheap pre-filter should route straight to (and stop at) the marker
    check.
    """
    h = _build_harness(mock_orch_config)
    h.git_ops = MagicMock()
    h.git_ops.config.branch_prefix = 'task/'
    h.git_ops.config.main_branch = 'main'

    h.git_ops.resolve_branch_sha = AsyncMock(return_value=None)
    h.git_ops.is_ancestor = AsyncMock(return_value=False)
    h.git_ops.find_merge_marker = AsyncMock(return_value=None)

    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})
    h._mark_in_progress_done = AsyncMock()
    h._escalation_queue = None
    return h


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateBranchExistencePreFilter:
    """Cheap pre-filter (task 2313 review): resolve_branch_sha is resolved
    ONCE up front instead of letting is_ancestor and the landing producer
    each spawn subprocesses that would only fail through to False.  These
    tests pin the reduced-I/O shape of both the branch-absent and the
    branch-exists-but-not-landed common cases.
    """

    async def test_missing_branch_skips_ancestor_and_content_checks(
        self, mock_orch_config, landing_producer: AsyncMock,
    ) -> None:
        """A branch that doesn't exist yet must never reach is_ancestor or
        the landing producer — both would just fail through to False at the
        cost of wasted subprocesses.  Only the marker search (the one check
        meaningful without a live ref) runs.
        """
        h = _wired_absent_branch_harness(mock_orch_config)

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h.git_ops.is_ancestor).assert_not_called()
        landing_producer.assert_not_awaited()
        cast(AsyncMock, h.git_ops.find_merge_marker).assert_awaited_once()
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    async def test_existing_not_landed_branch_skips_marker_and_citation(
        self, mock_orch_config, landing_producer: AsyncMock,
    ) -> None:
        """A branch that exists but isn't landed (not an ancestor, and the
        producer answers not_landed) must never call find_merge_marker — its
        own internal gate would just return None anyway since the branch ref
        still exists — and the gate must not stamp.
        """
        h = _wired_off_main_tip_harness(mock_orch_config)
        marker_search = AsyncMock(return_value=None)
        h.git_ops.find_merge_marker = marker_search
        mark = cast(AsyncMock, h._mark_in_progress_done)
        landing_producer.return_value = _not_contained()

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        marker_search.assert_not_called()
        landing_producer.assert_awaited_once()
        mark.assert_not_awaited()


@pytest.mark.asyncio
class TestAlreadyLandedDispatchGateInstall:
    """Harness.__init__ wires scheduler._already_landed_gate to the bound method."""

    async def test_scheduler_attribute_wired_to_bound_method(
        self, mock_orch_config,
    ) -> None:
        """RED until step-12 installs the wiring in Harness.__init__.

        Same bound-method equality idiom as
        TestHarnessLandedDispatchGateInstall (task 2156) — a freshly-accessed
        bound method is a new wrapper object each time, so ``==`` (not
        ``is``) is the correct comparison; MagicMock retains the exact
        object assigned during __init__, so this is a genuine RED
        (unset/auto-vivified Mock != bound method) before Harness.__init__
        wires the callable.
        """
        h = _build_harness(mock_orch_config)

        assert h.scheduler._already_landed_gate == h._already_landed_dispatch_gate, (
            'Harness must wire scheduler._already_landed_gate = '
            'harness._already_landed_dispatch_gate after construction'
        )


# ---------------------------------------------------------------------------
# TestAlreadyLandedGateDeliveredChecksGuard (task 3057 — step-3 RED / step-4 GREEN)
#
# Seam 2 of the eleven attribution-shaped mark-done seams. This gate's three
# evidence arms (git ancestry / merge marker / off-main tip) each prove
# only that SOMETHING of this branch reached main — never that THIS task's
# declared capability survived to it. All three are covered, in one
# parametrized matrix, so no single arm can regress unguarded.
# ---------------------------------------------------------------------------

_GATE_TARGET = 'orchestrator.harness.gate_mark_done_on_delivered_checks'

#: (arm id, reason/site label, expected evidence sha) for the three stamp arms.
_ARMS = [
    pytest.param('ancestry', id='ancestry-arm'),
    pytest.param('marker', id='marker-arm'),
    pytest.param('off_main_tip', id='off-main-tip-arm'),
]

#: The arms that validate a merge marker in CANDIDATE mode; every other arm
#: judges the live branch tip through ``branch_work_landed`` (task 4648).
_CANDIDATE_ARMS = frozenset({'marker'})
_LIVE_TIP_ARMS = [arm for arm in _ARMS if arm.values[0] not in _CANDIDATE_ARMS]

_ARM_REASON = {
    'ancestry': 'dispatch-gate-already-on-main',
    'marker': 'dispatch-gate-marker-found',
    'off_main_tip': 'dispatch-gate-content-equivalent',
}

_MARKER_SHA = 'b' * 40
_BRANCH_BASE_SHA = '9' * 40

_ARM_EVIDENCE_SHA = {
    'ancestry': _LANDED_SHA,
    'marker': _MARKER_SHA,
    'off_main_tip': _LANDED_SHA,
}

_DC_CHECK = {'name': 'cap-x', 'kind': 'grep', 'pattern': 'SomePattern', 'expect': 'present'}


def _arm_harness(
    mock_orch_config, arm: str, *, metadata: dict | None = None,
) -> tuple[Harness, dict[str, Any]]:
    """Build a harness wired so *arm* is the arm that reaches a stamp.

    Reuses this module's existing per-arm wiring helpers verbatim, then
    overlays the task metadata (each arm's helper hard-codes its own) and a
    real ``delivered_checks`` config section so the guard has live knobs to
    forward.
    """
    if arm == 'ancestry':
        h = _wired_ancestry_harness(mock_orch_config)
        base_meta: dict = {}
    elif arm == 'marker':
        h = _wired_marker_harness(
            mock_orch_config,
            marker_sha=_MARKER_SHA,
            branch_base_sha=_BRANCH_BASE_SHA,
            marker_is_ancestor_of_base=False,
        )
        base_meta = {'branch_base_sha': _BRANCH_BASE_SHA}
    elif arm == 'off_main_tip':
        h = _wired_off_main_tip_harness(mock_orch_config)
        base_meta = {}
    else:  # pragma: no cover - guard against a typo in the parametrization
        raise AssertionError(f'unknown arm {arm!r}')

    meta = dict(base_meta)
    meta.update(metadata if metadata is not None else {'delivered_checks': [_DC_CHECK]})
    h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': meta})
    h.config.delivered_checks = DeliveredChecksConfig(
        enabled=True, check_timeout_secs=11.0,
    )
    return h, meta


@pytest.mark.asyncio
class TestAlreadyLandedGateDeliveredChecksGuard:
    """The delivered-capability guard on the pre-dispatch already-landed gate.

    Task 2794's six-row acceptance matrix, applied to seam 2 and replicated
    across ALL THREE evidence arms. The recovery on every block is this
    gate's OWN existing "no landing evidence" path — ``return False``, i.e.
    deliberately the pre-2313 behavior (dispatch normally) rather than gating.
    A ``False`` return can never wedge a task the way a permanent gate could,
    and the invariant being defended is "never stamp a hollow done".
    """

    # --- row 1: hollow-done regression / FAILED ---------------------------

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_failed_block_withholds_stamp_and_dispatches(
        self, mock_orch_config, arm,
    ) -> None:
        """FAILED: the branch reached main but the declared capability did
        NOT. No stamp, and the gate returns False so the task DISPATCHES and
        an agent actually delivers it."""
        h, meta = _arm_harness(mock_orch_config, arm)
        guard = AsyncMock(return_value=DeliveredChecksBlock(
            reason='failed', main_sha='m' * 40, failed_check=_DC_CHECK,
        ))

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

        guard.assert_awaited_once()
        assert guard.await_args is not None
        assert guard.await_args.args[0] == '42'
        assert guard.await_args.args[1] == meta
        kwargs = guard.await_args.kwargs
        assert kwargs['project_root'] == str(h.config.project_root)
        assert kwargs['check_timeout_secs'] == 11.0
        assert kwargs['enabled'] is True
        assert kwargs['site'] == _ARM_REASON[arm]
        # The git_ops HANDLE, not merely a truthy sentinel: without it the
        # guard cannot resolve a main SHA and every check-carrying task
        # collapses to `main_sha_unresolved`, i.e. a fleet-wide silent
        # disarm-into-wedge. Pinning identity here is what makes
        # `git_ops=self.git_ops` -> `git_ops=None` a failing mutation.
        assert kwargs['git_ops'] is h.git_ops

    # --- row 2: all_delivered -> byte-identical stamp ----------------------

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_all_delivered_stamps_exactly_as_today(
        self, mock_orch_config, arm,
    ) -> None:
        """The capability IS on main -> the arm's existing stamp fires with
        its exact evidence sha / note / reason, and the gate returns True."""
        h, _meta = _arm_harness(mock_orch_config, arm)
        guard = AsyncMock(return_value=None)

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is True
        mark = cast(AsyncMock, h._mark_in_progress_done)
        mark.assert_awaited_once()
        assert mark.await_args is not None
        assert mark.await_args.args[0] == '42'
        assert mark.await_args.args[1] == _ARM_EVIDENCE_SHA[arm]
        assert mark.await_args.args[3] == _ARM_REASON[arm]

    # --- row 3: no delivered_checks -> unchanged, but still DELEGATED -----

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_check_less_task_delegates_and_stamps(
        self, mock_orch_config, arm,
    ) -> None:
        """A check-less task must not gain a new requirement.

        The harness DELEGATES unconditionally (passing the metadata through)
        rather than short-circuiting itself — the inertness lives in the
        helper, pinned at source in test_delivered_check_gate.py. Duplicating
        it here would be a second place for the kill-switch/inertness rule to
        drift.
        """
        h, meta = _arm_harness(mock_orch_config, arm, metadata={})
        guard = AsyncMock(return_value=None)

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is True
        cast(AsyncMock, h._mark_in_progress_done).assert_awaited_once()
        guard.assert_awaited_once()
        assert guard.await_args is not None
        assert 'delivered_checks' not in guard.await_args.args[1]

    # --- rows 4 & 5: fail-safe blocks are handled UNIFORMLY with FAILED ----

    @pytest.mark.parametrize('arm', _ARMS)
    @pytest.mark.parametrize('reason', ['errored', 'main_sha_unresolved'])
    async def test_fail_safe_blocks_also_withhold_and_dispatch(
        self, mock_orch_config, arm, reason,
    ) -> None:
        """ERRORED / main_sha_unresolved take the SAME recovery as FAILED.

        Deliberate: a malformed delivered_checks descriptor ERRORs forever, so
        a "wait and retry" degradation could WEDGE the task permanently. An
        extra dispatch of an already-landed task is the strictly better
        failure — it always terminates.
        """
        h, _meta = _arm_harness(mock_orch_config, arm)
        guard = AsyncMock(return_value=DeliveredChecksBlock(reason=reason))

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    # --- row 6: kill switch is FORWARDED, never re-implemented -------------

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_kill_switch_is_forwarded_not_short_circuited(
        self, mock_orch_config, arm,
    ) -> None:
        """``delivered_checks.enabled=False`` is forwarded to the helper, which
        owns the kill switch. The harness must NOT branch on it locally, or
        one hot reload would stop disarming all eleven seams at once."""
        h, _meta = _arm_harness(mock_orch_config, arm)
        h.config.delivered_checks = DeliveredChecksConfig(
            enabled=False, check_timeout_secs=11.0,
        )
        guard = AsyncMock(return_value=None)

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is True
        guard.assert_awaited_once()
        assert guard.await_args is not None
        assert guard.await_args.kwargs['enabled'] is False

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_kill_switch_with_real_helper_stamps_as_today(
        self, mock_orch_config, arm,
    ) -> None:
        """End-to-end with the REAL helper: disabled -> the stamp is
        byte-identical to today, with zero check work."""
        h, _meta = _arm_harness(mock_orch_config, arm)
        h.config.delivered_checks = DeliveredChecksConfig(
            enabled=False, check_timeout_secs=11.0,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is True
        mark = cast(AsyncMock, h._mark_in_progress_done)
        mark.assert_awaited_once()
        assert mark.await_args is not None
        assert mark.await_args.args[3] == _ARM_REASON[arm]

    # --- ordering: no wasted check work on landings already refused -------

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_rejected_landing_evidence_never_reaches_the_guard(
        self, mock_orch_config, arm, landing_producer: AsyncMock,
    ) -> None:
        """A REJECTED landing verdict, from either producer, already means "no
        usable landing evidence" — the guard must not pay for a git-grep on a
        landing that is being refused anyway."""
        h, _meta = _arm_harness(mock_orch_config, arm)
        guard = AsyncMock(return_value=None)
        landing_producer.return_value = _refused_verdict(LandingReason.no_attribution)
        rejected = LandingEvidenceVerdict(
            accepted=False, evidence_sha=None, reason='no_citation', probe={},
        )

        with patch(_GATE_TARGET, guard), \
                patch(
                    'orchestrator.harness.validate_landing_evidence',
                    AsyncMock(return_value=rejected),
                ):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        guard.assert_not_awaited()
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()

    async def test_degenerate_branch_never_reaches_the_guard(
        self, mock_orch_config, landing_producer: AsyncMock,
    ) -> None:
        """The ancestry arm's degenerate_branch verdict returns False BEFORE
        any capability is checked — the guard must sit downstream of it."""
        h, _meta = _arm_harness(mock_orch_config, 'ancestry')
        landing_producer.return_value = _refused_verdict(LandingReason.degenerate_branch)
        guard = AsyncMock(return_value=None)

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        guard.assert_not_awaited()

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_stale_conflict_skip_never_reaches_the_guard(
        self, mock_orch_config, arm,
    ) -> None:
        """The task-2677 provenance-conflict memo short-circuits BEFORE the
        git work; the guard must not resurrect that cost."""
        h, _meta = _arm_harness(mock_orch_config, arm)
        h._provenance_conflict_sink = MagicMock()
        h._provenance_conflict_sink.should_skip = MagicMock(return_value=True)
        guard = AsyncMock(return_value=None)

        with patch(_GATE_TARGET, guard):
            result = await h._already_landed_dispatch_gate('42')

        assert result is True
        guard.assert_not_awaited()
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()


# ---------------------------------------------------------------------------
# Task beta (3535): the two emissions this gate was PROMISED.
#
# Task 3534 left two forward references in this very method — its docstring
# ("The veto is LOGGED with the pinning escalation ids ...; task beta (3535)
# replaces that log line with the structured ``recovery_vetoed`` emission plus
# its N-consecutive-vetoes dedup escalation") and the comment above the
# arbitration hold ("... task beta (3535) replaces both with the structured
# `recovery_vetoed` emission plus its streak dedup").  These tests are the
# other half of those promises: honour them, or the spine is left carrying two
# stale lies.
#
# The canonical WHY for the whole mechanism lives in
# ``orchestrator/src/orchestrator/recovery_emission.py``'s module docstring and
# is deliberately not restated here.
# ---------------------------------------------------------------------------

#: What this gate renders as its ``shape``.  It resolves neither a claimant
#: (``live_claimant=False is deliberate and free``) nor a branch state, so both
#: elements are ``unknown`` rather than guessed; ``pending`` is a STRUCTURAL
#: fact of the site (it is consulted only over dispatch candidates), and the
#: trailing ``-`` is "no deploy state", read from the metadata already in hand.
_GATE_PINNED_SHAPE = 'pending|unknown|unknown|true|-'
#: Same, for the arm that could not read the store: the escalation element is
#: not merely absent, it is UNKNOWN (the esc-3163 distinction).
_GATE_UNREADABLE_SHAPE = 'pending|unknown|unknown|unknown|-'


def _emitting_harness(mock_orch_config, tmp_path: Path) -> Harness:
    """A wired ancestry harness with a REAL EventStore and a REAL config model.

    ``mock_orch_config`` is a spec'd MagicMock, so ``config.recovery_emission``
    would hand the streak predicate a MagicMock threshold — which the adapter's
    fail-open guard swallows, silently hiding whatever a test meant to pin.  A
    real :class:`RecoveryEmissionConfig` keeps the assertions honest.
    """
    h = _wired_ancestry_harness(mock_orch_config)
    h.event_store = EventStore(tmp_path / 'runs.db', 'run-test')
    h.config.recovery_emission = RecoveryEmissionConfig()
    return h


def _recovery_rows(harness: Harness) -> list[dict]:
    """Every recovery event row, oldest first, with ``data`` JSON-decoded."""
    conn = sqlite3.connect(str(harness.event_store.db_path))  # type: ignore[union-attr]
    try:
        return [
            {'task_id': tid, 'event_type': et, 'data': json.loads(raw or '{}')}
            for tid, et, raw in conn.execute(
                'SELECT task_id, event_type, data FROM events '
                'WHERE event_type IN (?, ?) ORDER BY id',
                (EventType.recovery_vetoed.value, EventType.recovery_left.value),
            )
        ]
    finally:
        conn.close()


def _conflict_escalation(esc_id: str = 'esc-42-conflict') -> Escalation:
    """A pending ``provenance_conflict`` L2 — the record the hold decides on.

    ``detail`` is left empty on purpose: ``arbitration_pending``'s documented
    ambiguity asymmetry ("EVERY ambiguity ... resolves to HOLD") then binds the
    task regardless of the caller's ``reopen_at``, so these tests exercise the
    hold arm without depending on reopen_at serialisation details that
    ``TestAlreadyLandedDispatchGateArbitrationHoldIsDurable`` already pins.
    """
    return Escalation(
        id=esc_id,
        task_id='42',
        agent_role='merge-worker',
        severity='urgent',
        category='provenance_conflict',
        summary='done evidence predates reopen_at',
        level=2,
    )


@pytest.mark.asyncio
class TestAlreadyLandedGatePinVetoEmission:
    """BOUNDARY #10: the any-level pin veto now emits, not just logs.

    A pending task carrying landing evidence and an open L2 is NOT auto-flipped
    (unchanged), and that hold stops being a log-only fact: it becomes a
    structured ``recovery_vetoed`` row naming the record that held it.
    """

    async def test_pin_veto_emits_exactly_one_recovery_vetoed_row(
        self, mock_orch_config, tmp_path,
    ) -> None:
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False, 'the veto disposition is unchanged: abstain'
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        rows = _recovery_rows(h)
        assert len(rows) == 1, f'expected exactly one recovery row, got {rows}'
        assert rows[0]['event_type'] == EventType.recovery_vetoed.value
        assert rows[0]['task_id'] == '42', (
            'task_id must be a first-class column so the row joins against '
            'task_completed / escalation_created'
        )

    async def test_pin_veto_payload_carries_the_full_key_set(
        self, mock_orch_config, tmp_path,
    ) -> None:
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )

        await h._already_landed_dispatch_gate('42')

        data = _recovery_rows(h)[0]['data']
        assert set(data) == {
            'task_id', 'site', 'shape', 'reason', 'escalation_ids',
            'ages_secs', 'measured_at', 'store_unavailable', 'streak',
        }
        assert data['task_id'] == '42'
        assert data['site'] == 'already_landed_gate'
        assert data['reason'] == 'escalation_pinned'
        assert data['store_unavailable'] is False
        assert data['shape'] == _GATE_PINNED_SHAPE, (
            'this gate resolves neither claimant nor branch state — it must '
            'render them unknown rather than guessing a shape it never saw'
        )
        stamp = datetime.fromisoformat(data['measured_at'])
        assert stamp.tzinfo is not None, 'measured_at must be tz-aware'

    async def test_pin_veto_payload_names_exactly_the_ids_the_log_names(
        self, mock_orch_config, tmp_path, caplog,
    ) -> None:
        """The structured row and the human log line must not disagree.

        The log names ``(*queue_handoff, *dead_l0)`` — the records that
        actually pinned.  The payload buckets those SAME ids and additionally
        carries the non-pinning ones, so a consumer can see the whole open set
        without the two records ever contradicting each other about which
        record held the task.
        """
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
            _open_escalation('esc-42-5', level=0, severity='info'),
        )

        with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        buckets = _recovery_rows(h)[0]['data']['escalation_ids']
        pinning = sorted((*buckets['queue_handoff'], *buckets['dead_l0']))
        assert pinning == ['esc-42-9'], (
            'the info-severity record never pins (PRD D3 / boundary row #8)'
        )
        assert buckets['non_pinning'] == ['esc-42-5'], (
            'a non-pinning record is still REPORTED — the operator sees the '
            'whole open set, bucketed, not a filtered half of it'
        )
        veto_lines = [
            r.getMessage() for r in caplog.records
            if r.name == 'orchestrator.harness' and 'esc-42-9' in r.getMessage()
        ]
        assert veto_lines, 'the human log line is KEPT, not replaced'
        assert not any('esc-42-5' in m for m in veto_lines), (
            'the log names the pinning ids only — the payload is the place '
            'the full bucketed set lives'
        )

    async def test_pin_veto_payload_ages_each_pinning_id_by_id(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """Ages join by ID, never by position — and come from the rows the
        gate ALREADY read (an ``Escalation`` spells its birth ``timestamp``)."""
        h = _emitting_harness(mock_orch_config, tmp_path)
        old = _open_escalation('esc-42-9', level=2)
        old.timestamp = (datetime.now(UTC) - timedelta(seconds=3600)).isoformat()
        recent = _open_escalation('esc-42-8', level=1)
        recent.timestamp = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
        h._escalation_queue = _queue_with_open(old, recent)

        await h._already_landed_dispatch_gate('42')

        ages = _recovery_rows(h)[0]['data']['ages_secs']
        assert set(ages) == {'esc-42-9', 'esc-42-8'}
        assert 3500.0 < ages['esc-42-9'] < 3700.0
        assert 60.0 < ages['esc-42-8'] < 300.0
        assert cast(MagicMock, h._escalation_queue).get_by_task.call_count == 1, (
            'ages must come from the single read the gate already pays for — '
            'telemetry may not add a second store read to a per-tick path'
        )

    async def test_repeat_ticks_with_an_identical_signature_emit_once(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """TICK-FREQUENCY GUARD.  This gate runs once per dispatch tick for as
        long as the task keeps being a candidate, so an unconditional emit
        would append one SQLite row per tick per pinned task, forever — the
        INV-4 storm this gate's own hold-logging already dodges."""
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )

        for _ in range(2):
            assert await h._already_landed_dispatch_gate('42') is False

        assert len(_recovery_rows(h)) == 1, (
            'a quiet repeat of a hold already on record must not re-state it'
        )

    async def test_many_ticks_stay_bounded_not_one_row_per_tick(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """Twenty ticks produce TWO rows, not twenty: the transition, plus the
        one deliberate re-statement at the streak threshold crossing.

        The same bound the OTHER per-tick site keeps
        (``test_scheduler_stranded_blocked_redispatch.py``'s
        ``test_many_ticks_stay_bounded_not_one_row_per_tick``) — one adapter,
        one cadence, so an operator reading these rows never has to know which
        site wrote them to know how to read their frequency.  Here the crossing
        marks the moment a sweep-frequency site WOULD have alarmed; this gate
        deliberately files no L1 of its own (see the method docstring).
        """
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )

        for _ in range(20):
            assert await h._already_landed_dispatch_gate('42') is False

        rows = _recovery_rows(h)
        assert len(rows) == 2, f'expected transition + threshold crossing: {rows}'
        assert [r['data']['streak'] for r in rows] == [
            1, h.config.recovery_emission.veto_streak_threshold,
        ]

    async def test_changed_pinning_ids_re_emit(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """A DIFFERENT record pinning the task is a new fact, not a repeat."""
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )

        await h._already_landed_dispatch_gate('42')
        cast(MagicMock, h._escalation_queue).get_by_task = MagicMock(
            return_value=[_open_escalation('esc-42-11', level=1)],
        )
        await h._already_landed_dispatch_gate('42')

        rows = _recovery_rows(h)
        assert len(rows) == 2, f'a changed pin set must re-announce: {rows}'
        assert sorted(
            i for ids in rows[1]['data']['escalation_ids'].values() for i in ids
        ) == ['esc-42-11']

    async def test_healthy_flip_emits_nothing(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """Nothing held the task back, so there is nothing to describe.  The
        ABSENCE of a row is the meaningful signal (see the EventType members)."""
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open()

        assert await h._already_landed_dispatch_gate('42') is True

        assert _recovery_rows(h) == []


@pytest.mark.asyncio
class TestAlreadyLandedGateArbitrationHoldEmission:
    """The arbitration hold describes itself too — on the TRANSITION only.

    The sink's ``note_hold_observed`` / ``clear_hold_observed`` streak state
    stays the authority for this arm (it already keys on
    ``(task_id, reopen_at)``, which the veto signature does not see), so the
    emission rides that transition rather than re-deriving one beside it.
    """

    async def test_hold_emits_recovery_vetoed_with_the_arbitration_reason(
        self, mock_orch_config, tmp_path,
    ) -> None:
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open(_conflict_escalation())
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is True, 'a contested task is still WITHHELD, not flipped'
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        rows = _recovery_rows(h)
        assert len(rows) == 1, f'expected one row, got {rows}'
        assert rows[0]['event_type'] == EventType.recovery_vetoed.value, (
            'an open record actively withheld a dispatch — that is a VETO, '
            'not a no-mapped-action fall-through'
        )
        assert rows[0]['task_id'] == '42'
        assert rows[0]['data']['reason'] == 'provenance_arbitration'
        assert rows[0]['data']['site'] == 'already_landed_gate'
        assert sorted(
            i for ids in rows[0]['data']['escalation_ids'].values() for i in ids
        ) == ['esc-42-conflict'], 'the payload names the contesting record'

    async def test_hold_repeats_emit_nothing_and_keep_the_log_cadence(
        self, mock_orch_config, tmp_path, caplog,
    ) -> None:
        """The hold re-fires every tick; the structured row must not.

        Also pins that the existing INFO-once/DEBUG-after cadence is PRESERVED
        rather than replaced — the event is the machine-readable record, the
        log stays the human one.
        """
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open(_conflict_escalation())
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=h._escalation_queue,
        )

        with caplog.at_level(logging.DEBUG, logger='orchestrator.harness'):
            for _ in range(3):
                assert await h._already_landed_dispatch_gate('42') is True

        assert len(_recovery_rows(h)) == 1, 'one hold, one row — not one per tick'
        held = [
            r for r in caplog.records
            if r.name == 'orchestrator.harness'
            and 'under provenance arbitration' in r.getMessage()
        ]
        assert [r.levelno for r in held] == [
            logging.INFO, logging.DEBUG, logging.DEBUG,
        ], [(r.levelname, r.getMessage()) for r in held]

    async def test_hold_released_then_returning_re_emits(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """A hold that RETURNS is a new fact, exactly as it is for the log.

        Without this the dedup would be strictly worse than the log line it
        was promised to replace: the sink goes loud again on a returning hold,
        and the event stream must not stay silent about it.
        """
        from orchestrator.provenance_conflict import ProvenanceConflictSink

        h = _emitting_harness(mock_orch_config, tmp_path)
        queue = _queue_with_open(_conflict_escalation())
        h._escalation_queue = queue
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=queue,
        )

        assert await h._already_landed_dispatch_gate('42') is True

        # The operator arbitrates: the record is resolved, the hold releases
        # and the task flips as it would have all along.
        queue.get_by_task = MagicMock(return_value=[])
        assert await h._already_landed_dispatch_gate('42') is True
        assert h._provenance_conflict_sink.note_hold_observed('42') is True, (
            'the release must have dropped the sink state — otherwise this '
            'test silently degrades into a repeat of the quiet-repeat case'
        )
        h._provenance_conflict_sink.clear_hold_observed('42')

        # ... and the conflict comes back.
        queue.get_by_task = MagicMock(return_value=[_conflict_escalation()])
        assert await h._already_landed_dispatch_gate('42') is True

        rows = _recovery_rows(h)
        assert len(rows) == 2, f'a returning hold must re-announce: {rows}'
        assert [r['data']['reason'] for r in rows] == [
            'provenance_arbitration', 'provenance_arbitration',
        ]


@pytest.mark.asyncio
class TestAlreadyLandedGateStoreUnavailableEmission:
    """An unreadable store is the THIRD state, and it says so now.

    The gate already vetoes on a failed read (task 3534) and logs a WARNING.
    What it could not do was let a consumer COUNT how often dispatch decisions
    are being made against an unreadable store.
    """

    async def test_read_failure_emits_recovery_left(
        self, mock_orch_config, tmp_path,
    ) -> None:
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = MagicMock()
        h._escalation_queue.has_open_l1 = MagicMock(return_value=False)
        h._escalation_queue.get_by_task = MagicMock(
            side_effect=OSError('escalation queue dir unreadable'),
        )

        result = await h._already_landed_dispatch_gate('42')

        assert result is False, 'never phantom-done on an unreadable store'
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        rows = _recovery_rows(h)
        assert len(rows) == 1, f'expected one row, got {rows}'
        assert rows[0]['event_type'] == EventType.recovery_left.value, (
            'nothing was held back by a RECORD — the site could not read the '
            'store to find out (the recovery_left half of the discriminator)'
        )
        data = rows[0]['data']
        assert data['reason'] == 'escalation_store_unavailable'
        assert data['store_unavailable'] is True
        assert data['site'] == 'already_landed_gate'
        assert data['shape'] == _GATE_UNREADABLE_SHAPE
        assert data['escalation_ids'] == {
            'dead_l0': [], 'queue_handoff': [], 'non_pinning': [],
        }, 'an unreadable store yields NO ids — never a false empty pin set'

    async def test_repeated_read_failure_stays_bounded(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """A store outage lasts many ticks; it is one fact, not one per tick.

        Bounded exactly as the pin veto is — the transition plus the single
        threshold-crossing re-statement — because both arms ride the one
        adapter rather than each inventing a cadence.
        """
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = MagicMock()
        h._escalation_queue.has_open_l1 = MagicMock(return_value=False)
        h._escalation_queue.get_by_task = MagicMock(
            side_effect=OSError('escalation queue dir unreadable'),
        )

        for _ in range(20):
            assert await h._already_landed_dispatch_gate('42') is False

        rows = _recovery_rows(h)
        assert len(rows) == 2, f'expected transition + threshold crossing: {rows}'
        assert all(r['data']['reason'] == 'escalation_store_unavailable' for r in rows)


def _matrix_pinned(h: Harness) -> None:
    h._escalation_queue = _queue_with_open(_open_escalation('esc-42-9', level=2))


def _matrix_info_only(h: Harness) -> None:
    h._escalation_queue = _queue_with_open(
        _open_escalation('esc-42-5', level=0, severity='info'),
    )


def _matrix_clean(h: Harness) -> None:
    h._escalation_queue = _queue_with_open()


def _matrix_no_queue(h: Harness) -> None:
    h._escalation_queue = None


def _matrix_store_failure(h: Harness) -> None:
    h._escalation_queue = MagicMock()
    h._escalation_queue.has_open_l1 = MagicMock(return_value=False)
    h._escalation_queue.get_by_task = MagicMock(
        side_effect=OSError('escalation queue dir unreadable'),
    )


def _matrix_arbitration(h: Harness) -> None:
    from orchestrator.provenance_conflict import ProvenanceConflictSink

    h._escalation_queue = _queue_with_open(_conflict_escalation())
    h._provenance_conflict_sink = ProvenanceConflictSink(
        escalation_queue=h._escalation_queue,
    )


#: name -> (wiring, expected gate return, expected _mark_in_progress_done awaits)
_GATE_MATRIX = {
    'l2_pin_abstains': (_matrix_pinned, False, 0),
    'info_only_still_flips': (_matrix_info_only, True, 1),
    'clean_read_flips': (_matrix_clean, True, 1),
    'absent_queue_flips': (_matrix_no_queue, True, 1),
    'unreadable_store_abstains': (_matrix_store_failure, False, 0),
    'arbitration_withholds': (_matrix_arbitration, True, 0),
}


@pytest.mark.asyncio
class TestAlreadyLandedGateZeroDispositionChange:
    """THE load-bearing test of task 3535 at this site.

    Task 3535's contract is emission BEFORE behavior (PRD
    ``plans/task-escalation-state-graph-prd.md`` D5): an emission that moved a
    disposition would be a defect no amount of correct payload could redeem.
    So every arm of the gate matrix is re-run with ``recovery_emission.enabled``
    flipped and must produce a byte-identical answer and an identical number of
    done-writes.  ``info_only_still_flips`` is in the matrix deliberately: the
    carve-out task 3534 landed must survive this task untouched.
    """

    @pytest.mark.parametrize('scenario', sorted(_GATE_MATRIX))
    @pytest.mark.parametrize('enabled', [True, False])
    async def test_disposition_is_identical_with_and_without_emission(
        self, mock_orch_config, tmp_path, scenario, enabled,
    ) -> None:
        wire, expected_result, expected_marks = _GATE_MATRIX[scenario]
        h = _emitting_harness(mock_orch_config, tmp_path)
        h.config.recovery_emission = RecoveryEmissionConfig(enabled=enabled)
        wire(h)

        result = await h._already_landed_dispatch_gate('42')

        assert result is expected_result, scenario
        assert cast(
            AsyncMock, h._mark_in_progress_done,
        ).await_count == expected_marks, scenario
        if not enabled:
            assert _recovery_rows(h) == [], (
                'the kill switch must actually silence the events it names'
            )

    async def test_emission_failure_never_breaks_the_gate(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """Telemetry is never load-bearing: a broken event store must not turn
        an abstain into a dispatch-gating exception (the scheduler's catch-all
        would fail the WHOLE gate open, losing the veto)."""
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )
        h.event_store = MagicMock()
        h.event_store.emit = MagicMock(side_effect=RuntimeError('db is gone'))

        result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()


@pytest.mark.asyncio
class TestAlreadyLandedGateReleasesItsVetoStreak:
    """The gate pops its tracker entry when the hold ends (amendment, cycle 1).

    This site runs per dispatch TICK and charges no alarm, so its tracker entry
    is pure footprint — and without a release edge a task vetoed here once and
    then gone done would leave that entry behind for the life of the process.
    That is exactly the growth ``RecoveryVetoStreakTracker.clear``'s docstring,
    ``Harness.__init__``'s comment and
    ``test_clear_pops_the_entry_so_the_footprint_stays_bounded`` all say the pop
    exists to prevent — pinned HERE at the wiring level, where the bound is
    actually delivered or lost.
    """

    def _gate_entries(self, h: Harness) -> list[tuple[str, str]]:
        from orchestrator.recovery_emission import RecoverySite

        return [
            (site, tid) for site, tid in h._recovery_veto_tracker.tracked()
            if site == RecoverySite.already_landed_gate
        ]

    async def test_a_hold_that_ends_pops_the_entry(
        self, mock_orch_config, tmp_path,
    ) -> None:
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )

        assert await h._already_landed_dispatch_gate('42') is False
        assert self._gate_entries(h) == [('already_landed_gate', '42')]

        # The escalation is resolved; nothing pins the task any more.
        cast(MagicMock, h._escalation_queue).get_by_task = MagicMock(
            return_value=[],
        )
        assert await h._already_landed_dispatch_gate('42') is True

        assert self._gate_entries(h) == [], (
            'a gate pass that found no hold must pop the streak, not leave it'
        )

    async def test_a_never_held_task_leaves_no_entry_at_all(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """The healthy majority of dispatch ticks: no hold, no footprint."""
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open()

        for _ in range(5):
            assert await h._already_landed_dispatch_gate('42') is True

        assert self._gate_entries(h) == []

    async def test_a_hold_that_returns_starts_a_fresh_streak(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """The pop is a RELEASE, not a mute: the same hold reappearing is a new
        fact and must announce itself again (streak back to 1)."""
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )

        await h._already_landed_dispatch_gate('42')
        cast(MagicMock, h._escalation_queue).get_by_task = MagicMock(
            return_value=[],
        )
        await h._already_landed_dispatch_gate('42')
        cast(MagicMock, h._escalation_queue).get_by_task = MagicMock(
            return_value=[_open_escalation('esc-42-9', level=2)],
        )
        await h._already_landed_dispatch_gate('42')

        rows = _recovery_rows(h)
        assert [r['data']['streak'] for r in rows] == [1, 1], (
            f'a released-and-returned hold must re-announce at streak 1: {rows}'
        )

    async def test_the_release_never_breaks_the_gate(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """Bookkeeping is never load-bearing: a tracker that raises must not
        turn a healthy flip into a gate exception."""
        h = _emitting_harness(mock_orch_config, tmp_path)
        h._escalation_queue = _queue_with_open()
        h._recovery_veto_tracker = MagicMock()
        h._recovery_veto_tracker.clear = MagicMock(
            side_effect=RuntimeError('tracker is gone'),
        )

        assert await h._already_landed_dispatch_gate('42') is True


_ABSENT_VIEW = AlreadyLandedTickView(local_branches=frozenset({'main'}))


class _GateProbe:
    """The already-landed gate as these tests drive it.

    The one place they name the harness's private members, so each test
    asserts on outcomes (docs/code-quality.md: tests reaching a module's
    internals are an interface smell). The gate has no public caller cheaper
    than a whole ``acquire_next`` tick.
    """

    def __init__(self, harness: Harness, *, stub_flips: bool = False) -> None:
        self.harness = harness
        if stub_flips:
            harness._mark_in_progress_done = AsyncMock()
        self.decide = harness._already_landed_dispatch_gate
        self.consult = harness.scheduler._consult_already_landed
        self.read_snapshot = harness._read_pending_escalations_by_task

    @property
    def flips(self) -> AsyncMock:
        return cast(AsyncMock, self.harness._mark_in_progress_done)

    @property
    def queue(self) -> Any:
        return self.harness._escalation_queue

    @queue.setter
    def queue(self, queue: Any) -> None:
        self.harness._escalation_queue = queue


@pytest.mark.asyncio
class TestAlreadyLandedGateTickView:
    """Task 4900: the gate takes the scheduler's candidate dict as ``task=``
    and the per-tick branch snapshot as ``tick_view=``. The snapshot may only
    make the gate abstain (see ``already_landed_tick_view``'s docstring)."""

    async def test_a_supplied_task_replaces_the_get_task_read(
        self, mock_orch_config,
    ) -> None:
        h = _wired_marker_harness(
            mock_orch_config,
            marker_sha='b' * 40,
            branch_base_sha='e' * 40,
            marker_is_ancestor_of_base=True,
        )
        h.scheduler.get_task = AsyncMock(return_value={'id': '42', 'metadata': {}})

        gate = _GateProbe(h)
        result = await gate.decide(
            '42', task={'id': '42', 'metadata': {'branch_base_sha': 'e' * 40}},
        )

        assert result is False, 'the supplied branch_base_sha drives the stale-marker veto'
        cast(AsyncMock, h.scheduler.get_task).assert_not_awaited()
        gate.flips.assert_not_awaited()

    async def test_without_a_supplied_task_get_task_is_read_once(
        self, mock_orch_config,
    ) -> None:
        h = _wired_absent_branch_harness(mock_orch_config)

        gate = _GateProbe(h)
        await gate.decide('42')

        cast(AsyncMock, h.scheduler.get_task).assert_awaited_once_with('42')

    async def test_absent_and_unmarked_abstains_without_a_ref_read(
        self, mock_orch_config, landing_producer: AsyncMock,
    ) -> None:
        h = _wired_absent_branch_harness(mock_orch_config)

        gate = _GateProbe(h)
        result = await gate.decide('42', tick_view=_ABSENT_VIEW)

        assert result is False
        cast(AsyncMock, h.git_ops.resolve_branch_sha).assert_not_awaited()
        cast(AsyncMock, h.git_ops.is_ancestor).assert_not_awaited()
        landing_producer.assert_not_awaited()
        gate.flips.assert_not_awaited()

    async def test_absent_but_marked_rereads_and_follows_a_recreated_branch(
        self, mock_orch_config, landing_producer: AsyncMock,
    ) -> None:
        """The stale-absence hazard: the marker sends the gate back to a fresh
        ref read, which finds the branch, so the live-branch producer decides
        and the marker can never flip it."""
        h = _wired_absent_branch_harness(mock_orch_config)
        h.git_ops.find_merge_marker = AsyncMock(return_value='b' * 40)
        h.git_ops.resolve_branch_sha = AsyncMock(return_value=_TIP_SHA)
        h.git_ops.is_ancestor = AsyncMock(return_value=False)
        landing_producer.return_value = _not_contained()

        gate = _GateProbe(h)
        result = await gate.decide('42', tick_view=_ABSENT_VIEW)

        assert result is False
        cast(AsyncMock, h.git_ops.resolve_branch_sha).assert_awaited_once()
        landing_producer.assert_awaited_once()
        assert landing_producer.await_args is not None
        assert landing_producer.await_args.kwargs['branch_tip_sha'] == _TIP_SHA
        gate.flips.assert_not_awaited()

    async def test_absent_but_marked_and_confirmed_absent_flips_on_the_marker(
        self, mock_orch_config,
    ) -> None:
        marker_sha = 'b' * 40
        h = _wired_marker_harness(
            mock_orch_config,
            marker_sha=marker_sha,
            branch_base_sha='e' * 40,
            marker_is_ancestor_of_base=False,
        )

        gate = _GateProbe(h)
        result = await gate.decide('42', tick_view=_ABSENT_VIEW)

        assert result is True
        cast(AsyncMock, h.git_ops.resolve_branch_sha).assert_awaited_once()
        call_args = gate.flips.await_args
        assert call_args is not None
        assert call_args.args[1] == marker_sha
        assert call_args.args[3] == 'dispatch-gate-marker-found'

    async def test_a_listed_branch_takes_todays_flow(self, mock_orch_config) -> None:
        h = _wired_ancestry_harness(mock_orch_config)
        view = AlreadyLandedTickView(local_branches=frozenset({'main', 'task/42'}))

        gate = _GateProbe(h)
        result = await gate.decide('42', tick_view=view)

        assert result is True
        cast(AsyncMock, h.git_ops.resolve_branch_sha).assert_awaited_once()
        gate.flips.assert_awaited_once()

    async def test_an_unknown_enumeration_takes_todays_flow(
        self, mock_orch_config,
    ) -> None:
        h = _wired_absent_branch_harness(mock_orch_config)

        gate = _GateProbe(h)
        result = await gate.decide(
            '42', tick_view=AlreadyLandedTickView(local_branches=None),
        )

        assert result is False
        cast(AsyncMock, h.git_ops.resolve_branch_sha).assert_awaited_once()
        cast(AsyncMock, h.git_ops.find_merge_marker).assert_awaited_once()

    async def test_the_arbitration_hold_still_wins_over_the_view(
        self, mock_orch_config, tmp_path,
    ) -> None:
        from escalation.queue import EscalationQueue

        from orchestrator.provenance_conflict import ProvenanceConflictSink

        h = _wired_ancestry_harness(mock_orch_config)
        _let_mark_in_progress_done_run_for_real(h, tmp_path)
        h.git_ops.find_merge_marker = AsyncMock(return_value=None)
        gate = _GateProbe(h)
        gate.queue = EscalationQueue(tmp_path / 'esc')
        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {'reopen_at': _STALE_REOPEN_AT}},
        )
        _wire_stale_evidence_mark_done(h, evidence_commit='a' * 40)
        ProvenanceConflictSink(escalation_queue=gate.queue).record(
            task_id='42',
            evidence_commit='a' * 40,
            evidence_committed_at='2026-07-10T00:00:00+00:00',
            reopen_at=_STALE_REOPEN_AT,
            agent_id='claude-merge-worker',
            gate_source='merge-queue',
        )
        h._provenance_conflict_sink = ProvenanceConflictSink(
            escalation_queue=gate.queue,
        )

        result = await gate.decide('42', tick_view=_ABSENT_VIEW)

        assert result is True, 'a contested task stays withheld whatever the view says'

    async def test_the_escalation_veto_still_fires_and_emits_before_the_view(
        self, mock_orch_config, tmp_path,
    ) -> None:
        h = _emitting_harness(mock_orch_config, tmp_path)
        h.git_ops.find_merge_marker = AsyncMock(return_value=None)
        gate = _GateProbe(h)
        gate.queue = _queue_with_open(
            _open_escalation('esc-42-9', level=2),
        )

        result = await gate.decide('42', tick_view=_ABSENT_VIEW)

        assert result is False
        rows = _recovery_rows(h)
        assert [r['event_type'] for r in rows] == [EventType.recovery_vetoed.value]
        assert rows[0]['data']['reason'] == 'escalation_pinned'

    async def test_the_consult_reads_its_view_through_the_late_bound_git_ops(
        self, tmp_path,
    ) -> None:
        h = Harness(OrchestratorConfig(project_root=tmp_path))
        branch = f'{h.git_ops.config.branch_prefix}9'
        git_ops_double = MagicMock(spec=GitOps)
        git_ops_double.config = h.git_ops.config
        git_ops_double.list_local_branches = AsyncMock(return_value=frozenset({'main'}))
        git_ops_double.find_merge_marker = AsyncMock(return_value=None)
        h.git_ops = git_ops_double

        gate = _GateProbe(h)
        gated = await gate.consult([{'id': '9', 'metadata': {}}])

        assert gated == set()
        git_ops_double.list_local_branches.assert_awaited_once_with()
        git_ops_double.find_merge_marker.assert_awaited_once_with(
            branch, gate_on_existing_ref=False,
        )
        git_ops_double.resolve_branch_sha.assert_not_awaited()


# ---------------------------------------------------------------------------
# task 4496 — the ancestry arm's reject is recorded, not silent.
#
# The arm files no escalation by design, and its False means DISPATCH — so a
# false negative here is an agent redoing landed work.  A transition-gated
# landing_evidence_rejected row makes that rate countable.  Canonical WHY:
# orchestrator/src/orchestrator/landing_reject_emission.py.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAlreadyLandedGateAncestryRejectEmission:
    async def test_a_refused_verdict_is_recorded_and_still_dispatches(
        self, mock_orch_config, tmp_path, landing_producer: AsyncMock,
    ) -> None:
        h = _emitting_harness(mock_orch_config, tmp_path)
        landing_producer.return_value = _not_contained()

        assert await h._already_landed_dispatch_gate('42') is False

        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        [row] = landing_reject_rows(h.event_store)
        assert row['task_id'] == '42'
        data = row['data']
        assert data['site'] == 'already_landed_gate_ancestry'
        assert data['task_status'] == 'pending'
        assert data['reason'] == 'not_landed'
        assert data['method'] == 'patch_id'
        assert data['branch_tip_sha'] == _TIP_SHA
        assert data['patch_id_containment'] == 'not_contained'
        assert data['containment_recheck'] == 'healthy'

    async def test_an_unattributed_reject_is_recorded(
        self, mock_orch_config, tmp_path, landing_producer: AsyncMock,
    ) -> None:
        h = _emitting_harness(mock_orch_config, tmp_path)
        landing_producer.return_value = _refused_verdict(
            LandingReason.no_attribution, evidence_source='none',
        )

        assert await h._already_landed_dispatch_gate('42') is False

        [row] = landing_reject_rows(h.event_store)
        assert row['data']['reason'] == 'no_attribution'
        assert row['data']['evidence_source'] == 'none'
        assert row['data']['branch_tip_sha'] == _TIP_SHA

    async def test_a_reject_repeated_every_tick_is_recorded_once(
        self, mock_orch_config, tmp_path, landing_producer: AsyncMock,
    ) -> None:
        h = _emitting_harness(mock_orch_config, tmp_path)
        landing_producer.return_value = _not_contained()

        assert await h._already_landed_dispatch_gate('42') is False
        assert await h._already_landed_dispatch_gate('42') is False

        assert len(landing_reject_rows(h.event_store)) == 1

    async def test_a_dispatch_between_two_identical_rejects_records_both(
        self, mock_orch_config, tmp_path, landing_producer: AsyncMock,
    ) -> None:
        """The counting unit is one row per signature per dispatch the reject let through."""
        h = _emitting_harness(mock_orch_config, tmp_path)
        landing_producer.return_value = _not_contained()
        assert await h._already_landed_dispatch_gate('42') is False

        slot = h._launch_slot(
            TaskAssignment(task_id='42', task={'id': '42'}, modules=[]),
            asyncio.Semaphore(0), [],
        )
        slot.cancel()
        await asyncio.gather(slot, return_exceptions=True)

        assert await h._already_landed_dispatch_gate('42') is False
        assert len(landing_reject_rows(h.event_store)) == 2

    async def test_an_accepted_ancestry_landing_records_nothing(
        self, mock_orch_config, tmp_path,
    ) -> None:
        h = _emitting_harness(mock_orch_config, tmp_path)

        assert await h._already_landed_dispatch_gate('42') is True

        assert landing_reject_rows(h.event_store) == []

    async def test_declared_checks_are_not_consulted_on_a_reject(
        self, mock_orch_config, tmp_path, landing_producer: AsyncMock,
    ) -> None:
        """The patch-id producer has no delivered-checks rescue, so a declared
        capability cannot turn a refused verdict into a stamp, and the reject
        row carries no differential outcome.
        """
        h = _emitting_harness(mock_orch_config, tmp_path)
        h.scheduler.get_task = AsyncMock(
            return_value={'id': '42', 'metadata': {'delivered_checks': [_DC_CHECK]}},
        )
        landing_producer.return_value = _not_contained()
        runner = AsyncMock(side_effect=differential_confirms)

        with patch('orchestrator.delivered_checks.run_delivered_check', runner):
            assert await h._already_landed_dispatch_gate('42') is False

        runner.assert_not_awaited()
        [row] = landing_reject_rows(h.event_store)
        assert 'delivered_checks_outcome' not in row['data']


# ---------------------------------------------------------------------------
# task 4648 — the OFF-MAIN-TIP arm: a live tip that is not an ancestor of main.
#
# It decides by branch_work_landed, as the ancestry arm does.  Only
# no_attribution escalates (the work is on main but cannot be attributed);
# every other reject is recorded at its own site and the task dispatches.
# ---------------------------------------------------------------------------


#: The off-main-tip arm's non-escalating rejects.
_OFF_MAIN_TIP_RECORDED = [
    LandingReason.not_landed,
    LandingReason.degenerate_branch,
    LandingReason.no_op_landing,
    LandingReason.git_error,
]


@pytest.mark.asyncio
class TestAlreadyLandedGateOffMainTipArm:
    async def test_an_unattributed_landing_escalates_and_dispatches(
        self, mock_orch_config, tmp_path, landing_producer: AsyncMock,
    ) -> None:
        h = _wired_off_main_tip_harness(mock_orch_config, tmp_path)
        file_escalation = MagicMock()
        h._file_unattributed_landing_escalation = file_escalation
        mark = cast(AsyncMock, h._mark_in_progress_done)
        verdict = _refused_verdict(LandingReason.no_attribution)
        landing_producer.return_value = verdict

        assert await h._already_landed_dispatch_gate('42') is False

        file_escalation.assert_called_once_with('42', 'task/42', verdict)
        mark.assert_not_awaited()
        assert landing_reject_rows(h.event_store) == []

    async def test_the_escalation_body_renders_the_patch_id_verdict(self) -> None:
        """FAIL BINDING: a no_attribution patch-id verdict renders into a full L1 body.

        Manifest row 'rejected-verdict-carries-structured-facts-for-the-
        escalating-arms': the probe's facts and the reason's own explanation
        both reach the detail, so the escalation stays actionable.
        """
        verdict = _refused_verdict(LandingReason.no_attribution)

        _summary, detail = format_unattributed_landing_detail('42', 'task/42', verdict)

        assert _TIP_SHA in detail
        assert 'the work IS on main, but no commit cites the task' in detail
        assert 'Unrecognized reason code' not in detail

    @pytest.mark.parametrize('reason', _OFF_MAIN_TIP_RECORDED)
    async def test_every_other_reject_is_recorded_not_escalated(
        self, mock_orch_config, tmp_path, landing_producer: AsyncMock,
        reason: LandingReason,
    ) -> None:
        h = _wired_off_main_tip_harness(mock_orch_config, tmp_path)
        file_escalation = MagicMock()
        h._file_unattributed_landing_escalation = file_escalation
        mark = cast(AsyncMock, h._mark_in_progress_done)
        landing_producer.return_value = _refused_verdict(reason)

        assert await h._already_landed_dispatch_gate('42') is False

        file_escalation.assert_not_called()
        mark.assert_not_awaited()
        [row] = landing_reject_rows(h.event_store)
        assert row['data']['site'] == 'already_landed_gate_off_main_tip'
        assert row['data']['reason'] == str(reason)
        assert row['data']['method'] == 'patch_id'

    async def test_git_error_on_the_ancestry_arm_is_recorded_as_itself(
        self, mock_orch_config, tmp_path, landing_producer: AsyncMock,
    ) -> None:
        """git_error is a claim about the detector, so it is never recorded as not_landed."""
        h = _emitting_harness(mock_orch_config, tmp_path)
        file_escalation = MagicMock()
        h._file_unattributed_landing_escalation = file_escalation
        landing_producer.return_value = _refused_verdict(
            LandingReason.git_error, git_error_stage='patch_id_containment',
        )

        assert await h._already_landed_dispatch_gate('42') is False

        file_escalation.assert_not_called()
        [row] = landing_reject_rows(h.event_store)
        assert row['data']['site'] == 'already_landed_gate_ancestry'
        assert row['data']['reason'] == 'git_error'
        assert row['data']['git_error_stage'] == 'patch_id_containment'


@pytest.mark.asyncio
class TestAlreadyLandedGateLiveTipArmsProducerContract:
    """Both live-tip arms hand ``branch_work_landed`` the same contract (task 4648).

    ``escalation_queue`` and ``recovery_emission`` arm the producer's
    ``git_error`` storm escape, so an arm that dropped either would disarm
    that alarm with no other test failing.
    """

    @pytest.mark.parametrize('arm', _LIVE_TIP_ARMS)
    async def test_the_producer_is_called_with_the_arms_contract(
        self, mock_orch_config, landing_producer: AsyncMock, arm: str,
    ) -> None:
        h, metadata = _arm_harness(
            mock_orch_config, arm, metadata={'branch_base_sha': _BRANCH_BASE_SHA},
        )
        queue = MagicMock()
        queue.get_by_task = MagicMock(return_value=[])
        h._escalation_queue = queue
        validate = AsyncMock()

        with patch('orchestrator.harness.validate_landing_evidence', validate):
            result = await h._already_landed_dispatch_gate('42')

        assert result is True
        validate.assert_not_awaited()
        landing_producer.assert_awaited_once()
        call = landing_producer.await_args
        assert call is not None
        assert call.args == (h.git_ops, '42', 'task/42')
        assert call.kwargs['branch_tip_sha'] == _TIP_SHA
        assert call.kwargs['metadata'] == metadata
        assert call.kwargs['pattern_template'] is h.git_ops.config.commit_citation_pattern
        assert call.kwargs['escalation_queue'] is queue
        assert call.kwargs['recovery_emission'] is h.config.recovery_emission


# ---------------------------------------------------------------------------
# task 4496 — the gate arm that still decides by validate_landing_evidence, the
# MARKER arm, passes the task's declared delivered_checks to it (never None,
# which means "unwired"), and a stamp written off a delivered-checks RESCUE
# discloses it in the provenance note.  Both live-branch arms decide by
# branch_work_landed, which has no rescue (task 4648); _delivered_checks_block
# still guards their stamps (TestAlreadyLandedGateDeliveredChecksGuard).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAlreadyLandedGateDeliveredChecksWiring:
    async def test_a_task_declaring_no_checks_reaches_the_validator_as_empty(
        self, mock_orch_config,
    ) -> None:
        h, _meta = _arm_harness(mock_orch_config, 'marker', metadata={})
        spy = AsyncMock(wraps=validate_landing_evidence)

        with patch('orchestrator.harness.validate_landing_evidence', spy), \
             patch(_GATE_TARGET, AsyncMock(return_value=None)):
            await h._already_landed_dispatch_gate('42')

        assert spy.await_args is not None
        assert spy.await_args.kwargs.get('delivered_checks') == []

    @pytest.mark.parametrize(
        ('declaration', 'forwarded'),
        [
            pytest.param(['junk', _DC_CHECK], [_DC_CHECK], id='non-dict-entry-dropped'),
            pytest.param(
                {'name': 'cap', 'kind': 'grep', 'pattern': 'X', 'expect': 'present'},
                [],
                id='bare-dict-value-never-iterated-by-key',
            ),
        ],
    )
    async def test_only_the_declared_check_descriptors_reach_the_validator(
        self, mock_orch_config, declaration, forwarded,
    ) -> None:
        h, _meta = _arm_harness(
            mock_orch_config, 'marker', metadata={'delivered_checks': declaration},
        )
        spy = AsyncMock(wraps=validate_landing_evidence)

        with patch('orchestrator.harness.validate_landing_evidence', spy), \
             patch(_GATE_TARGET, AsyncMock(return_value=None)):
            await _GateProbe(h).decide('42')

        assert spy.await_args is not None
        assert spy.await_args.kwargs.get('delivered_checks') == forwarded

    @pytest.mark.parametrize('arm', _LIVE_TIP_ARMS)
    async def test_a_live_tip_arm_hands_the_recorded_base_to_the_producer(
        self, mock_orch_config, landing_producer: AsyncMock, arm,
    ) -> None:
        """The producer's recorded-base rung is fed from the task's metadata (task 7836)."""
        h, _meta = _arm_harness(
            mock_orch_config, arm,
            metadata={'branch_base_sha': _BRANCH_BASE_SHA, 'delivered_checks': [_DC_CHECK]},
        )

        with patch(_GATE_TARGET, AsyncMock(return_value=None)):
            await _GateProbe(h).decide('42')

        assert landing_producer.await_args is not None
        assert landing_producer.await_args.kwargs['metadata']['branch_base_sha'] == _BRANCH_BASE_SHA

    async def test_a_differential_rescue_stamps_and_discloses_it(
        self, mock_orch_config,
    ) -> None:
        h, _meta = _arm_harness(mock_orch_config, 'marker')
        h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=False)

        with patch('orchestrator.delivered_checks.run_delivered_check', differential_confirms), \
             patch(_GATE_TARGET, AsyncMock(return_value=None)):
            result = await h._already_landed_dispatch_gate('42')

        assert result is True
        mark = cast(AsyncMock, h._mark_in_progress_done)
        mark.assert_awaited_once()
        assert mark.await_args is not None
        assert mark.await_args.args[1] == _MARKER_SHA
        assert 'cap-x' in mark.await_args.args[2]

    async def test_a_disarmed_kill_switch_stops_the_rescue_and_dispatches(
        self, mock_orch_config,
    ) -> None:
        h, _meta = _arm_harness(mock_orch_config, 'marker')
        h.config.delivered_checks = DeliveredChecksConfig(enabled=False, check_timeout_secs=11.0)
        h.git_ops.commit_effect_present_in_main = AsyncMock(return_value=False)
        runner = AsyncMock(side_effect=differential_confirms)
        spy = AsyncMock(wraps=validate_landing_evidence)

        with patch('orchestrator.delivered_checks.run_delivered_check', runner), \
             patch('orchestrator.harness.validate_landing_evidence', spy):
            result = await h._already_landed_dispatch_gate('42')

        assert result is False
        cast(AsyncMock, h._mark_in_progress_done).assert_not_awaited()
        runner.assert_not_awaited()
        assert spy.await_args is not None
        assert spy.await_args.kwargs.get('checks_enabled') is False


def _pending_snapshot_for(gate: _GateProbe) -> dict[str, list[Escalation]]:
    """What a consult's snapshot holds for task 42, read from the wired stub."""
    queue = cast(MagicMock, gate.queue)
    rows = queue.get_by_task('42', status='pending')
    queue.get_by_task.reset_mock()
    return {'42': list(rows)} if rows else {}


def _disposition_trail(h: Harness) -> list[tuple[str, str, Any]]:
    return [
        (r['event_type'], r['data']['reason'], r['data']['escalation_ids'])
        for r in _recovery_rows(h)
    ]


class _CountingQueue:
    """A REAL queue that counts the two pending reads the gate can make."""

    def __init__(self, queue_dir: Path, *, unreadable: bool = False) -> None:
        from escalation.queue import EscalationQueue

        self.queue = EscalationQueue(queue_dir)
        self.unreadable = unreadable
        self.per_task_reads = 0
        self.snapshot_reads = 0

    def get_by_task(self, task_id: str, **kwargs: Any) -> list[Escalation]:
        self.per_task_reads += 1
        if self.unreadable:
            raise OSError('escalation queue dir unreadable')
        return self.queue.get_by_task(task_id, **kwargs)

    def get_pending_by_task(self) -> dict[str, list[Escalation]]:
        self.snapshot_reads += 1
        if self.unreadable:
            raise OSError('escalation queue dir unreadable')
        return self.queue.get_pending_by_task()


def _consulting_gate(tmp_path: Path, queue: _CountingQueue) -> _GateProbe:
    """A real Harness whose consult runs end to end; every branch is absent."""
    h = Harness(OrchestratorConfig(project_root=tmp_path))
    git_ops_double = MagicMock(spec=GitOps)
    git_ops_double.config = h.git_ops.config
    git_ops_double.list_local_branches = AsyncMock(return_value=frozenset({'main'}))
    git_ops_double.find_merge_marker = AsyncMock(return_value=None)
    h.git_ops = git_ops_double
    gate = _GateProbe(h, stub_flips=True)
    gate.queue = queue
    return gate


_SNAPSHOT_SCENARIOS = [
    'l2_pin_abstains', 'info_only_still_flips', 'clean_read_flips', 'arbitration_withholds',
]


@pytest.mark.asyncio
class TestAlreadyLandedGatePendingEscalationSnapshot:
    """Task 8073: the consult reads the pending store once and hands every
    candidate its rows as ``pending_escalations=``. The gate must decide
    exactly as its own ``get_by_task(task_id, status='pending')`` read would."""

    @pytest.mark.parametrize('scenario', _SNAPSHOT_SCENARIOS)
    async def test_a_snapshot_decides_exactly_as_the_per_candidate_read(
        self, mock_orch_config, tmp_path, scenario,
    ) -> None:
        wire, expected_result, expected_marks = _GATE_MATRIX[scenario]
        per_candidate = _emitting_harness(mock_orch_config, tmp_path / 'a')
        wire(per_candidate)
        per_candidate_gate = _GateProbe(per_candidate)
        from_snapshot = _emitting_harness(mock_orch_config, tmp_path / 'b')
        wire(from_snapshot)
        snapshot_gate = _GateProbe(from_snapshot)
        snapshot = _pending_snapshot_for(snapshot_gate)

        expected = await per_candidate_gate.decide('42')
        result = await snapshot_gate.decide('42', pending_escalations=snapshot)

        assert expected is expected_result
        assert result is expected_result
        assert snapshot_gate.flips.await_count == expected_marks
        assert cast(
            MagicMock, snapshot_gate.queue,
        ).get_by_task.call_count == expected_marks, 'one fresh re-read per flip, none to veto'
        assert cast(
            MagicMock, per_candidate_gate.queue,
        ).get_by_task.call_count == 1, 'without a snapshot the one read is the veto read'
        assert _disposition_trail(from_snapshot) == _disposition_trail(per_candidate)

    async def test_a_task_absent_from_the_snapshot_has_no_pending_rows(
        self, mock_orch_config, tmp_path,
    ) -> None:
        """Absent means ``[]``, never ``None``: the store WAS read, so the
        unreadable-store veto must not fire; the one read the store sees is
        the clean pre-flip re-read."""
        h = _emitting_harness(mock_orch_config, tmp_path)
        _matrix_clean(h)
        gate = _GateProbe(h)

        result = await gate.decide(
            '42', pending_escalations={'7': [_open_escalation('esc-7-1', level=2)]},
        )

        assert result is True
        gate.flips.assert_awaited_once()
        cast(MagicMock, gate.queue).get_by_task.assert_called_once_with(
            '42', status='pending',
        )
        assert _recovery_rows(h) == []

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_a_record_filed_after_the_snapshot_stops_every_flip_arm(
        self, mock_orch_config, arm,
    ) -> None:
        h, _meta = _arm_harness(mock_orch_config, arm, metadata={})
        gate = _GateProbe(h)
        gate.queue = _queue_with_open(_open_escalation('esc-42-9', level=2))

        with patch(_GATE_TARGET, AsyncMock(return_value=None)):
            result = await gate.decide('42', pending_escalations={})

        assert result is False
        gate.flips.assert_not_awaited()
        cast(MagicMock, gate.queue).get_by_task.assert_called_once_with(
            '42', status='pending',
        )

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_a_severity_raised_in_place_after_the_snapshot_stops_every_flip_arm(
        self, mock_orch_config, arm,
    ) -> None:
        """Same id, new severity: ``add_members_to_l2`` / ``attach_dedupe_child``
        raise a pending record's severity in place, so an id diff would miss it."""
        h, _meta = _arm_harness(mock_orch_config, arm, metadata={})
        gate = _GateProbe(h)
        gate.queue = _queue_with_open(
            _open_escalation('esc-42-1', level=2, severity='blocking'),
        )
        snapshot = {'42': [_open_escalation('esc-42-1', level=2, severity='info')]}

        with patch(_GATE_TARGET, AsyncMock(return_value=None)):
            result = await gate.decide('42', pending_escalations=snapshot)

        assert result is False
        gate.flips.assert_not_awaited()

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_an_info_record_filed_after_the_snapshot_does_not_stop_the_flip(
        self, mock_orch_config, arm,
    ) -> None:
        h, _meta = _arm_harness(mock_orch_config, arm, metadata={})
        gate = _GateProbe(h)
        gate.queue = _queue_with_open(
            _open_escalation('esc-42-5', level=0, severity='info'),
        )

        with patch(_GATE_TARGET, AsyncMock(return_value=None)):
            result = await gate.decide('42', pending_escalations={})

        assert result is True
        gate.flips.assert_awaited_once()

    async def test_an_arbitration_filed_after_the_snapshot_withholds_dispatch(
        self, mock_orch_config,
    ) -> None:
        """The re-read decides as the first read does: a contested task is
        withheld (``True``), never released to dispatch."""
        h = _wired_ancestry_harness(mock_orch_config)
        _matrix_arbitration(h)
        gate = _GateProbe(h)

        result = await gate.decide('42', pending_escalations={})

        assert result is True
        gate.flips.assert_not_awaited()

    @pytest.mark.parametrize('arm', _ARMS)
    async def test_a_failed_re_read_before_the_flip_abstains_and_says_so(
        self, mock_orch_config, tmp_path, arm, caplog,
    ) -> None:
        h, _meta = _arm_harness(mock_orch_config, arm, metadata={})
        gate = _GateProbe(h)
        h.event_store = EventStore(tmp_path / 'runs.db', 'run-test')
        h.config.recovery_emission = RecoveryEmissionConfig()
        _matrix_store_failure(h)

        with patch(_GATE_TARGET, AsyncMock(return_value=None)), \
                caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            result = await gate.decide('42', pending_escalations={})

        assert result is False
        gate.flips.assert_not_awaited()
        assert any(
            'could not READ the escalation store for task 42' in r.getMessage()
            for r in caplog.records
        )
        rows = _recovery_rows(h)
        assert [r['data']['reason'] for r in rows] == ['escalation_store_unavailable']
        assert rows[0]['data']['store_unavailable'] is True

    async def test_a_record_resolved_after_the_snapshot_still_holds_the_flip_only_once(
        self, mock_orch_config,
    ) -> None:
        """Over-holding is the snapshot's only error: the stale pin vetoes this
        consult, and the next consult's fresh snapshot lets the task flip."""
        h = _wired_ancestry_harness(mock_orch_config)
        _matrix_clean(h)
        gate = _GateProbe(h)
        pinned = _open_escalation('esc-42-9', level=2)

        held = await gate.decide(
            '42', pending_escalations={'42': [pinned]},
        )
        flipped = await gate.decide('42', pending_escalations={})

        assert (held, flipped) == (False, True)
        gate.flips.assert_awaited_once()

    async def test_the_harness_reader_groups_the_live_store_and_skips_an_unwired_one(
        self, tmp_path,
    ) -> None:
        gate = _GateProbe(Harness(OrchestratorConfig(project_root=tmp_path)))
        gate.queue = None
        assert await gate.read_snapshot() is None

        queue = _CountingQueue(tmp_path / 'esc')
        queue.queue.submit(_open_escalation('esc-42-9', level=2))
        gate.queue = queue

        snapshot = await gate.read_snapshot()

        assert snapshot is not None
        assert [e.id for e in snapshot['42']] == ['esc-42-9']

    async def test_one_store_read_per_consult_whatever_the_candidate_count(
        self, tmp_path,
    ) -> None:
        queue = _CountingQueue(tmp_path / 'esc')
        queue.queue.submit(_open_escalation('esc-42-9', level=2))
        gate = _consulting_gate(tmp_path, queue)
        candidates = [{'id': tid, 'metadata': {}} for tid in ('9', '42', '77', '78')]

        gated = await gate.consult(candidates)

        assert gated == set()
        assert (queue.snapshot_reads, queue.per_task_reads) == (1, 0)

    async def test_an_unreadable_snapshot_degrades_to_per_candidate_store_unavailable(
        self, tmp_path, caplog,
    ) -> None:
        queue = _CountingQueue(tmp_path / 'esc', unreadable=True)
        gate = _consulting_gate(tmp_path, queue)
        candidates = [{'id': tid, 'metadata': {}} for tid in ('9', '42')]

        with caplog.at_level(logging.WARNING):
            gated = await gate.consult(candidates)

        assert gated == set()
        assert (queue.snapshot_reads, queue.per_task_reads) == (1, 2)
        gate.flips.assert_not_awaited()
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('pending escalation snapshot' in m for m in messages), messages
        for tid in ('9', '42'):
            assert any(
                f'could not READ the escalation store for task {tid}' in m for m in messages
            ), messages

    async def test_an_undecodable_record_fails_the_snapshot_and_the_fallback_alike(
        self, tmp_path, caplog,
    ) -> None:
        """A scan error outside the queue's parse tuple (here the
        UnicodeDecodeError, a ValueError, of a non-UTF-8 record) propagates
        from both reads, so the snapshot is ``None`` and each candidate lands
        on store_unavailable exactly as before task 8073."""
        queue = _CountingQueue(tmp_path / 'esc')
        (queue.queue.queue_dir / 'esc-9-1.json').write_bytes(b'\xff\xfe not utf-8')
        gate = _consulting_gate(tmp_path, queue)

        with caplog.at_level(logging.WARNING):
            gated = await gate.consult([{'id': '9', 'metadata': {}}])

        assert gated == set()
        assert (queue.snapshot_reads, queue.per_task_reads) == (1, 1)
        gate.flips.assert_not_awaited()
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('pending escalation snapshot' in m for m in messages), messages
        assert any(
            'could not READ the escalation store for task 9' in m for m in messages
        ), messages
