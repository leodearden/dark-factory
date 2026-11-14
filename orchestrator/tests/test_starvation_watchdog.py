"""The starvation watchdog's predicate, as a known-answer table (task 3821).

Every gate needs the durable eligibility-age floor AND an in-candidacy stuck
signal, so a task that is merely old (backlog depth) never fires.
"""

from __future__ import annotations

import pytest

from orchestrator.config import StarvationWatchdogConfig
from orchestrator.pin_reservation import Blocker, BlockerKind
from orchestrator.starvation_watchdog import (
    AUTO_PIN_CAP_SENTINEL,
    STARVATION_WATCHDOG_MARKER,
    CandidateFacts,
    Finding,
    LockRefusals,
    ParkBlock,
    StarvationGate,
    cap_escalation,
    classify,
    escalation_text,
)

DAY = 86400.0
FLOOR = 259200.0
CFG = StarvationWatchdogConfig(
    skip_threshold=50, idle_secs=FLOOR, idle_only_secs=FLOOR,
    park_blocked_age_secs=14400.0,
)
HOLDER = Blocker(module='orchestrator/src/orchestrator/harness.py', owner='H', kind=BlockerKind.HELD)


def _facts(
    *, age: float = 0.0, skips: int = 0, refusals: int = 0,
    park: ParkBlock | None = None, task_id: str = 'T',
) -> CandidateFacts:
    return CandidateFacts(
        task_id=task_id, eligible_age_secs=age, skip_count=skips,
        lock_refusals=refusals, park_block=park,
    )


def _park(age: float) -> ParkBlock:
    return ParkBlock(blocking_holder='H', blocking_module='orchestrator/src/orchestrator', park_age_secs=age)


@pytest.mark.parametrize(('facts', 'expected'), [
    pytest.param(_facts(age=10 * DAY), None, id='backlog-alone-never-fires'),
    pytest.param(_facts(age=FLOOR - 1, skips=500, refusals=500), None, id='signal-below-age-floor'),
    pytest.param(_facts(age=FLOOR, skips=49, refusals=49), None, id='age-with-sub-threshold-signal'),
    pytest.param(_facts(age=FLOOR, skips=50), StarvationGate.DUAL, id='dual'),
    pytest.param(_facts(age=FLOOR, refusals=50), StarvationGate.LOCK_REFUSED, id='lock-refused'),
    pytest.param(_facts(age=FLOOR, skips=50, refusals=50), StarvationGate.DUAL, id='dual-before-lock-refused'),
    pytest.param(_facts(park=_park(14400.0)), StarvationGate.PARK_BLOCKED, id='park-blocked'),
    pytest.param(
        _facts(age=FLOOR, skips=50, refusals=50, park=_park(14399.0)), None,
        id='young-park-never-routes-to-a-pinning-gate',
    ),
])
def test_classify_known_answers(facts: CandidateFacts, expected: StarvationGate | None) -> None:
    assert classify(facts, CFG) is expected


def test_each_gate_has_its_own_age_floor() -> None:
    cfg = StarvationWatchdogConfig(skip_threshold=5, idle_secs=100.0, idle_only_secs=200.0)

    assert classify(_facts(age=150.0, skips=5), cfg) is StarvationGate.DUAL
    assert classify(_facts(age=150.0, refusals=5), cfg) is None
    assert classify(_facts(age=200.0, refusals=5), cfg) is StarvationGate.LOCK_REFUSED


def test_only_park_blocked_does_not_pin() -> None:
    assert {gate for gate in StarvationGate if not gate.pins} == {StarvationGate.PARK_BLOCKED}


class TestLockRefusals:
    def test_counts_scans_per_task(self) -> None:
        refusals = LockRefusals()
        refusals.note_scan(['a', 'b'])
        refusals.note_scan(['a', 'b'])
        refusals.note_scan(['a'])

        assert refusals.count('a') == 3
        assert refusals.count('b') == 2
        assert refusals.count('never') == 0
        assert refusals.task_ids() == frozenset({'a', 'b'})

    def test_forget_drops_the_count(self) -> None:
        refusals = LockRefusals()
        refusals.note_scan(['a', 'b'])
        refusals.forget('a')

        assert refusals.count('a') == 0
        assert refusals.task_ids() == frozenset({'b'})


class TestEscalationText:
    @pytest.mark.parametrize(('facts', 'gate'), [
        (_facts(age=FLOOR, skips=50), StarvationGate.DUAL),
        (_facts(age=FLOOR, refusals=50), StarvationGate.LOCK_REFUSED),
        (_facts(age=FLOOR, park=_park(20000.0)), StarvationGate.PARK_BLOCKED),
    ])
    def test_summary_carries_marker_and_gate(
        self, facts: CandidateFacts, gate: StarvationGate,
    ) -> None:
        summary, detail = escalation_text(facts, gate, CFG, [HOLDER])

        assert summary.startswith(STARVATION_WATCHDOG_MARKER)
        assert gate.value in summary
        assert len(summary) <= 200
        assert 'T' in summary
        assert detail

    def test_pin_gate_text_names_the_first_blocker(self) -> None:
        summary, detail = escalation_text(
            _facts(age=FLOOR, refusals=50), StarvationGate.LOCK_REFUSED, CFG, [HOLDER],
        )
        assert 'H' in summary
        assert HOLDER.module in detail

    def test_park_blocked_text_names_holder_and_module(self) -> None:
        park = ParkBlock(blocking_holder='H42', blocking_module='escalation/src', park_age_secs=20000.0)
        summary, detail = escalation_text(
            _facts(age=FLOOR, park=park), StarvationGate.PARK_BLOCKED, CFG, [],
        )
        assert 'H42' in summary
        assert 'H42' in detail
        assert 'escalation/src' in detail


def test_cap_escalation_names_every_capped_task() -> None:
    cohort = [
        Finding(_facts(task_id='A', age=FLOOR, skips=50), StarvationGate.DUAL, (HOLDER,)),
        Finding(_facts(task_id='B', age=FLOOR, refusals=50), StarvationGate.LOCK_REFUSED, ()),
    ]

    l1 = cap_escalation(cohort, CFG)

    assert l1.sentinel == AUTO_PIN_CAP_SENTINEL
    assert l1.summary.startswith(STARVATION_WATCHDOG_MARKER)
    for text in ('A', 'B', 'dual', 'lock_refused', HOLDER.module, str(CFG.auto_pin_max_concurrent)):
        assert text in l1.detail
