"""Unit tests for `orchestrator.recovery_pins` — THE orchestrator-side pin predicate.

Task 3541 (PRD `plans/task-escalation-state-graph-prd.md` task eta, spec
`docs/task-escalation-state-spec.md` S6/E7, INV-5).

`escalation.pins` owns the pin CLASS (pure, config-free, category-free).
`orchestrator.recovery_pins` owns the orchestrator-side CATEGORY policy — the
merge-remediable relaxation — and composes the two into the one predicate every
recovery/redispatch veto site consumes.  These tests pin that composition
directly, so the harness- and scheduler-side wiring tests never have to
re-derive it.

The module is deliberately PURE and cycle-free: `TestNoOrchestratorCycle`
enforces that as a parsed property of the source, not as a convention.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from escalation.models import Escalation
from escalation.pins import classify_pins

from orchestrator.recovery_pins import (
    MERGE_REMEDIABLE_ESC_CATEGORIES,
    only_merge_remediable,
    records_pin_blocked_done_flip,
    records_pin_blocked_recovery,
    records_pin_recovery,
)
from orchestrator.task_ground_truth import EscalationRef

_TID = '3541'

#: A live incarnation identity in `shared.task_claimant.compose_claimant_run_id`
#: shape (`{run_id}/{session_id}/pid={owner_pid}`).  Built literally rather than
#: through the composer so a change to the composer's FORMAT fails the pins
#: tests loudly instead of silently making every identity read "unknown".
_LIVE_ID = 'run-a/sess-a/pid=101'
_OTHER_ID = 'run-b/sess-b/pid=202'


def _ref(
    category: str = 'stranded_blocked',
    *,
    level: int = 1,
    severity: str = 'blocking',
    esc_id: str = 'esc-3541-1',
    filing_claimant_run_id: str | None = None,
) -> EscalationRef:
    """A resolver-side record (frozen dataclass leg of the `PinRecord` protocol)."""
    return EscalationRef(
        id=esc_id,
        level=level,
        category=category,
        severity=severity,
        filing_claimant_run_id=filing_claimant_run_id,
    )


def _esc(
    category: str = 'stranded_blocked',
    *,
    level: int = 1,
    severity: str = 'blocking',
    esc_id: str = 'esc-3541-1',
    filing_claimant_run_id: str | None = None,
) -> Escalation:
    """A store-side record (mutable leg of the `PinRecord` protocol)."""
    return Escalation(
        id=esc_id,
        task_id=_TID,
        agent_role='harness-test',
        severity=severity,
        category=category,
        summary='fixture',
        level=level,
        filing_claimant_run_id=filing_claimant_run_id,
    )


class TestMergeRemediableCategories:
    """The relaxation's category set and its `all(...)` semantics (PRD leaf delta).

    Mirrors the pre-existing assertions on `Harness._only_merge_remediable`
    (test_stranded_verified_green.py `TestMergeRemediableCategories`) so the
    MOVE into this module is provably behaviour-preserving.
    """

    def test_the_set_is_exactly_stranded_blocked(self) -> None:
        assert isinstance(MERGE_REMEDIABLE_ESC_CATEGORIES, frozenset)
        assert sorted(MERGE_REMEDIABLE_ESC_CATEGORIES) == ['stranded_blocked']

    def test_empty_is_vacuously_true(self) -> None:
        """No open escalation -> True, byte-identical to `not report.open_escalations`."""
        assert only_merge_remediable([]) is True

    def test_lone_remediable_is_true(self) -> None:
        assert only_merge_remediable([_ref('stranded_blocked')]) is True

    @pytest.mark.parametrize(
        'category', ['design_concern', 'stranded_merge_failed', 'task_failure', 'infra_issue'],
    )
    def test_lone_non_remediable_is_false(self, category: str) -> None:
        assert only_merge_remediable([_ref(category)]) is False

    def test_one_non_remediable_among_remediable_is_false(self) -> None:
        """`all(...)` semantics: ONE human-concern record still vetoes the relaxation."""
        refs = [_ref('stranded_blocked'), _ref('task_failure', esc_id='esc-3541-2')]
        assert only_merge_remediable(refs) is False


class TestRecordsPinRecovery:
    """`records_pin_recovery` IS `classify_pins(...).pins` — no local re-derivation.

    Every case asserts the wrapper's answer AND that it equals the classifier's
    own, so the wrapper can never drift from `escalation.pins`.
    """

    def _both(
        self,
        records: list[EscalationRef] | None,
        *,
        live_claimant: bool,
        live_claimant_id: str | None = None,
    ) -> bool:
        answer = records_pin_recovery(
            _TID, records, live_claimant=live_claimant, live_claimant_id=live_claimant_id,
        )
        expected = classify_pins(
            _TID, records, live_claimant=live_claimant, live_claimant_id=live_claimant_id,
        ).pins
        assert answer is expected, 'wrapper diverged from classify_pins(...).pins'
        return answer

    def test_empty_does_not_pin(self) -> None:
        assert self._both([], live_claimant=False) is False

    def test_l1_pins(self) -> None:
        """L1 is a queue-backed handoff: pins regardless of liveness (link 3)."""
        assert self._both([_ref(level=1)], live_claimant=False) is True

    def test_l2_pins(self) -> None:
        assert self._both([_ref(level=2)], live_claimant=True) is True

    def test_l0_with_matching_live_filer_pins(self) -> None:
        """A genuinely live handoff (link 4, identities MATCH)."""
        records = [_ref(level=0, filing_claimant_run_id=_LIVE_ID)]
        assert self._both(records, live_claimant=True, live_claimant_id=_LIVE_ID) is True

    def test_l0_with_dead_filer_does_not_pin(self) -> None:
        """A newer incarnation never keeps a prior one's unconsumed L0 alive."""
        records = [_ref(level=0, filing_claimant_run_id=_OTHER_ID)]
        assert self._both(records, live_claimant=True, live_claimant_id=_LIVE_ID) is False

    def test_l0_with_no_live_claimant_at_all_does_not_pin(self) -> None:
        """Link 4's identity-independent branch — the sweep-site case."""
        assert self._both([_ref(level=0)], live_claimant=False) is False

    def test_l0_with_unknown_identity_fails_safe_to_pinning(self) -> None:
        """Unknown filing identity is not PROOF of death — fail safe (link 4)."""
        assert self._both([_ref(level=0)], live_claimant=True, live_claimant_id=_LIVE_ID) is True

    @pytest.mark.parametrize('level', [0, 1, 2])
    def test_info_never_pins(self, level: int) -> None:
        """Link 1: an info record is an ANNOTATION, not a handoff, at any level."""
        records = [_ref(level=level, severity='info')]
        assert self._both(records, live_claimant=False) is False

    def test_blank_severity_fails_safe_to_pinning(self) -> None:
        """`EscalationRef`'s `severity=''` default is UNKNOWN -> pins (link 2)."""
        assert self._both([_ref(level=0, severity='')], live_claimant=False) is True

    def test_store_unavailable_pins(self) -> None:
        """`records=None` (a read that could not be performed) always pins."""
        assert self._both(None, live_claimant=False) is True


class TestRecordsPinBlockedRecovery:
    """The blocked-arm predicate: pin class AND the merge-remediable relaxation.

    `records_pin_blocked_recovery` == `records_pin_recovery(...) and not
    only_merge_remediable(records)`.  This is the single predicate BOTH
    `Harness._reconcile_one_stranded`'s blocked arm and
    `Scheduler._phase_redispatch_stranded_blocked` consume, which is what
    unifies the drift E7 catalogued.
    """

    def test_lone_stranded_blocked_l1_does_not_pin(self) -> None:
        """The relaxation: the reaper's own re-pend request never vetoes its own remediation."""
        assert records_pin_blocked_recovery(
            _TID, [_ref('stranded_blocked')], live_claimant=False,
        ) is False

    def test_task_failure_l1_pins(self) -> None:
        """A human-concern class names a problem a merge does not fix."""
        assert records_pin_blocked_recovery(
            _TID, [_ref('task_failure')], live_claimant=False,
        ) is True

    def test_mixed_remediable_and_human_concern_pins(self) -> None:
        """`all(...)` semantics, which test_convert_to_blocked.py already calls load-bearing."""
        records = [_ref('stranded_blocked'), _ref('task_failure', esc_id='esc-3541-2')]
        assert records_pin_blocked_recovery(_TID, records, live_claimant=False) is True

    def test_empty_does_not_pin(self) -> None:
        assert records_pin_blocked_recovery(_TID, [], live_claimant=False) is False

    def test_info_only_does_not_pin(self) -> None:
        """Link 1 short-circuits before the relaxation is consulted at all."""
        records = [_ref('task_failure', severity='info')]
        assert records_pin_blocked_recovery(_TID, records, live_claimant=False) is False

    def test_dead_l0_does_not_pin(self) -> None:
        """A dead-filer L0 has no consumer left, whatever its category."""
        records = [_ref('task_failure', level=0)]
        assert records_pin_blocked_recovery(_TID, records, live_claimant=False) is False

    def test_store_unavailable_pins(self) -> None:
        """An unreadable store never relaxes: there are no categories to judge."""
        assert records_pin_blocked_recovery(_TID, None, live_claimant=False) is True

    @pytest.mark.parametrize(
        ('category', 'level', 'severity', 'expected'),
        [
            ('stranded_blocked', 1, 'blocking', False),
            ('task_failure', 1, 'blocking', True),
            ('stranded_blocked', 0, 'blocking', False),
            ('task_failure', 0, 'blocking', False),
            ('task_failure', 2, 'blocking', True),
            ('task_failure', 1, 'info', False),
        ],
    )
    def test_is_exactly_the_documented_composition(
        self, category: str, level: int, severity: str, expected: bool,
    ) -> None:
        """The formula, asserted against its two halves rather than restated."""
        records = [_ref(category, level=level, severity=severity)]
        composed = (
            records_pin_recovery(_TID, records, live_claimant=False)
            and not only_merge_remediable(records)
        )
        answer = records_pin_blocked_recovery(_TID, records, live_claimant=False)
        assert answer is composed
        assert answer is expected


class TestBothRecordTypesSatisfyTheProtocol:
    """`escalation.models.Escalation` AND `task_ground_truth.EscalationRef` both work.

    The `PinRecord` protocol's read-only members exist precisely so a frozen
    dataclass (`EscalationRef`) and a mutable one (`Escalation`) can both
    satisfy it; these wrappers must not narrow that.
    """

    @pytest.mark.parametrize(
        ('category', 'level', 'severity'),
        [
            ('stranded_blocked', 1, 'blocking'),
            ('task_failure', 1, 'blocking'),
            ('task_failure', 0, 'blocking'),
            ('task_failure', 1, 'info'),
        ],
    )
    def test_same_answer_for_both_record_shapes(
        self, category: str, level: int, severity: str,
    ) -> None:
        ref_rows = [_ref(category, level=level, severity=severity)]
        esc_rows = [_esc(category, level=level, severity=severity)]
        assert records_pin_recovery(
            _TID, ref_rows, live_claimant=False,
        ) is records_pin_recovery(_TID, esc_rows, live_claimant=False)
        assert records_pin_blocked_recovery(
            _TID, ref_rows, live_claimant=False,
        ) is records_pin_blocked_recovery(_TID, esc_rows, live_claimant=False)

    def test_only_merge_remediable_reads_escalation_category(self) -> None:
        assert only_merge_remediable([_esc('stranded_blocked')]) is True
        assert only_merge_remediable([_esc('task_failure')]) is False


class TestNoOrchestratorCycle:
    """The module is pure: it imports `escalation.pins` and stdlib, nothing else.

    Enforced by PARSING the source rather than by convention, because the
    no-cycle property is what lets `scheduler.py` (which does not import
    `task_ground_truth`) and `harness.py` share one predicate at all.
    """

    _FORBIDDEN = (
        'orchestrator.harness',
        'orchestrator.scheduler',
        'orchestrator.task_ground_truth',
        'orchestrator.workflow',
    )

    @staticmethod
    def _imported_modules() -> set[str]:
        source = Path(
            __file__,
        ).parent.parent / 'src' / 'orchestrator' / 'recovery_pins.py'
        tree = ast.parse(source.read_text())
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
                names.update(f'{node.module}.{a.name}' for a in node.names)
        return names

    def test_imports_no_orchestrator_god_file(self) -> None:
        imported = self._imported_modules()
        offenders = sorted(
            name for name in imported
            if any(name == f or name.startswith(f + '.') for f in self._FORBIDDEN)
        )
        assert not offenders, (
            f'recovery_pins.py must stay importable from both harness.py and '
            f'scheduler.py without a cycle, but it imports: {offenders}'
        )

    def test_reaches_the_shared_classifier(self) -> None:
        """Positively: the pin CLASS comes from `escalation.pins`, not from here."""
        imported = self._imported_modules()
        assert any(
            name == 'escalation.pins' or name.startswith('escalation.pins.')
            for name in imported
        ), f'recovery_pins.py must consume escalation.pins; imports were {sorted(imported)}'


# ---------------------------------------------------------------------------
# REVIEW FINDING 2 — the blocked arm's DONE-FLIP question (PRD D3, spec §6
# demand 3).
#
# `records_pin_blocked_recovery` expands to `pins and not only_merge_remediable`
# and `pins` deliberately calls a dead-filer L0 NON-pinning — correct for
# "may this task be re-dispatched / re-filed?", and WRONG for "may this task be
# flipped to done?".  D3 is explicit: "Done-flip vetoes stay maximally
# conservative: ANY non-info open record still vetoes MARK_DONE".
#
# So the blocked arm needs the conservative twin, built from the SAME
# classification read for the other question.  The two differ on EXACTLY one
# input class, and the table-driven test below makes that an enumerated,
# checked set rather than an accident.
# ---------------------------------------------------------------------------


class TestRecordsPinBlockedDoneFlip:
    """`vetoes_done_flip` narrowed by the merge-remediable relaxation."""

    def test_empty_does_not_pin(self) -> None:
        """An unpinned task still self-heals — the relaxation is not a veto."""
        assert records_pin_blocked_done_flip(_TID, [], live_claimant=False) is False

    def test_lone_stranded_blocked_l1_does_not_pin(self) -> None:
        """The merge-remediable relaxation is preserved VERBATIM.

        A task whose branch landed while it was still blocked is usually held
        by the reaper's own `stranded_blocked` — the record that ASKED for this
        landing.  Letting it veto the done-flip pins the task forever after its
        work is already on main.
        """
        assert records_pin_blocked_done_flip(
            _TID, [_ref('stranded_blocked')], live_claimant=False,
        ) is False

    def test_task_failure_l1_pins(self) -> None:
        assert records_pin_blocked_done_flip(
            _TID, [_ref('task_failure')], live_claimant=False,
        ) is True

    def test_a_dead_filer_l0_pins(self) -> None:
        """THE NEW ROW, and the whole reason this predicate exists.

        `records_pin_blocked_recovery` returns False here — a dead L0 has no
        consumer left, so conversion and re-dispatch proceed.  A DONE-FLIP is
        different in kind: it is terminal, and completing a task past an
        unconsumed handoff is the phantom-done the dispatch gate already
        closed at the other end.  The record's owner arrives later, when the
        orphan-L0 reaper promotes it.
        """
        records = [_ref('task_failure', level=0)]
        assert records_pin_blocked_done_flip(_TID, records, live_claimant=False) is True
        assert records_pin_blocked_recovery(_TID, records, live_claimant=False) is False

    def test_info_only_in_a_remediable_category_does_not_pin(self) -> None:
        records = [_ref('stranded_blocked', severity='info')]
        assert records_pin_blocked_done_flip(_TID, records, live_claimant=False) is False

    def test_mixed_remediable_l1_and_foreign_class_info_pins(self) -> None:
        """`all(...)` category semantics, unchanged from its twin."""
        records = [
            _ref('stranded_blocked'),
            _ref('design_concern', severity='info', esc_id='esc-3541-2'),
        ]
        assert records_pin_blocked_done_flip(_TID, records, live_claimant=False) is True

    def test_store_unavailable_pins(self) -> None:
        """An unreadable store never relaxes — same ordering as its twin."""
        assert records_pin_blocked_done_flip(_TID, None, live_claimant=False) is True


class TestTheTwoBlockedPredicatesDivergeOnExactlyOneRow:
    """The asymmetry is an ENUMERATED set, not an emergent accident.

    Walking both predicates over the same fixtures and naming the one row they
    may disagree on means a future edit to either cannot silently widen the
    divergence — which is the failure mode that produced this finding.
    """

    _FIXTURES = (
        ('empty', []),
        ('lone-remediable-l1', [_ref('stranded_blocked')]),
        ('human-concern-l1', [_ref('task_failure')]),
        ('dead-filer-l0', [_ref('task_failure', level=0)]),
        ('info-only', [_ref('stranded_blocked', severity='info')]),
        ('mixed-categories', [
            _ref('stranded_blocked'),
            _ref('task_failure', esc_id='esc-3541-2'),
        ]),
    )

    #: The ONLY label at which the two answers may differ.
    _EXPECTED_DIVERGENCE = frozenset({'dead-filer-l0'})

    @pytest.mark.parametrize(
        'label,records', _FIXTURES, ids=[f[0] for f in _FIXTURES],
    )
    def test_they_agree_everywhere_except_the_named_row(
        self, label: str, records: list,
    ) -> None:
        recovery = records_pin_blocked_recovery(_TID, records, live_claimant=False)
        done_flip = records_pin_blocked_done_flip(_TID, records, live_claimant=False)

        if label in self._EXPECTED_DIVERGENCE:
            assert done_flip is True and recovery is False, (
                f'{label} is the one row the two predicates exist to disagree '
                f'about; it must diverge in the CONSERVATIVE direction'
            )
        else:
            assert done_flip is recovery, (
                f'{label} is not an enumerated divergence — the done-flip '
                f'predicate has widened beyond the dead-L0 row it was added '
                f'for (recovery={recovery}, done_flip={done_flip})'
            )

    def test_store_unavailable_agrees_too(self) -> None:
        assert records_pin_blocked_done_flip(_TID, None, live_claimant=False) is (
            records_pin_blocked_recovery(_TID, None, live_claimant=False)
        )
