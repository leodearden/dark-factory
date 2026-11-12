"""Tests for scripts/sitting/audits.py — the executor-close audit population, its draw and its tally (task 8425)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from escalation.authority import COORDINATOR_IDENTITY, EXECUTOR_IDENTITY, policy_row
from escalation.models import Escalation
from orchestrator.session_registry import (
    AuditOutcome,
    AuditVerdict,
    DecisionRecord,
    DecisionState,
    close_decision_with_evidence,
    decision_path_for_id,
    normalize_escalations_dir,
    record_audit_outcome,
    reopen_decision,
    write_decision,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from sitting import audits as mod
from sitting import inventory
from sitting.fleet_state import Window

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

NOW = datetime(2026, 9, 26, 5, 30, tzinfo=UTC)
WINDOW = Window.trailing(NOW, days=mod.AUDIT_WINDOW_DAYS)
SAMPLED_CLASS = 'landing_provenance'
FULL_CLASS = 'b3_low_risk_unblock'
RETIRED_CLASS = 'a_class_no_longer_in_the_table'


def _in_window(n: int) -> str:
    return (NOW - timedelta(hours=n + 1)).isoformat()


OUTSIDE_WINDOW = (NOW - timedelta(days=mod.AUDIT_WINDOW_DAYS + 3)).isoformat()


@dataclass
class Fleet:
    root: Path
    queue: Path

    @property
    def queue_dir(self) -> str:
        return normalize_escalations_dir(self.queue)

    def index(self, *queues: Path) -> dict[str, Mapping[str, Path]]:
        dirs = [normalize_escalations_dir(q) for q in (self.queue, *queues)]
        return {q: inventory.index_queue(q) for q in dirs}

    def escalation(self, esc_id: str, *, queue: Path | None = None, archived: bool = True, **fields) -> Escalation:
        fields.setdefault('task_id', esc_id.split('-')[1])
        fields.setdefault('agent_role', 'escalation-watcher-auto')
        fields.setdefault('severity', 'blocking')
        fields.setdefault('category', 'design_concern')
        fields.setdefault('summary', f'summary of {esc_id}')
        fields.setdefault('level', 2)
        fields.setdefault('timestamp', '2026-09-01T00:00:00+00:00')
        fields.setdefault('status', 'resolved')
        fields.setdefault('resolved_by', EXECUTOR_IDENTITY)
        fields.setdefault('policy_class', SAMPLED_CLASS)
        esc = Escalation(id=esc_id, **fields)
        target = (queue or self.queue) / ('archive/2026-09-25' if archived else '')
        target.mkdir(parents=True, exist_ok=True)
        (target / f'{esc_id}.json').write_text(esc.to_json())
        return esc

    def decision(self, decision_id: str, esc_id: str, *, queue: Path | None = None,
                 closed_at: str | None = None, evidence: str | None = None) -> DecisionRecord:
        """A DecisionRecord over *esc_id*: closed with evidence at *closed_at*, or open when it is None."""
        closed: dict[str, Any] = {} if closed_at is None else {
            'state': 'answered', 'closed_at': closed_at,
            'closing_evidence': evidence or f'quoted evidence for {decision_id}',
        }
        record = DecisionRecord(
            id=decision_id, project='dark_factory', text=f'close {esc_id}?', filed_at='2026-09-01T00:00:00+00:00',
            escalation_id=esc_id, escalations_dir=normalize_escalations_dir(queue or self.queue), **closed,
        )
        assert write_decision(record, root=self.root)
        return record

    def executor_close(self, n: int, *, policy_class: str = SAMPLED_CLASS, closed_at: str | None = None,
                       queue: Path | None = None, archived: bool = True) -> DecisionRecord:
        esc_id = f'esc-{700 + n}-1'
        self.escalation(esc_id, queue=queue, archived=archived, policy_class=policy_class,
                        claimed_by=EXECUTOR_IDENTITY, claimed_at=_in_window(n))
        return self.decision(f'dec-{policy_class}-{n}', esc_id, queue=queue, closed_at=closed_at or _in_window(n))

    def audit(self, record: DecisionRecord, verdict: AuditVerdict, note: str = '') -> None:
        """Audit *record* under the class its escalation record names now, as the audit verb copies it."""
        assert record.escalation_id is not None
        path = inventory.index_queue(record.escalations_dir)[record.escalation_id]
        policy_class = Escalation.from_json(path.read_text()).policy_class
        assert policy_class is not None, record.id
        audited = record_audit_outcome(
            record.id, verdict, note, root=self.root, expected_project=record.project,
            expected_escalations_dir=record.escalations_dir, audited_at=NOW.isoformat(),
            policy_class=policy_class, closed_at=record.closed_at,
        )
        assert audited is not None

    def closes(self, *queues: Path, window: Window = WINDOW) -> mod.ExecutorCloses:
        return mod.executor_closes(self.root, self.index(*queues), window)


@pytest.fixture
def fleet(tmp_path) -> Fleet:
    queue = tmp_path / 'src' / 'dark-factory' / 'data' / 'escalations'
    queue.mkdir(parents=True)
    return Fleet(tmp_path / 'fleet', queue)


def _ids(closes) -> list[str]:
    return [close.decision_id for close in closes]


def _sort_key(close: mod.ExecutorClose) -> tuple[str, str]:
    return close.closed_at, close.decision_id


def _audit_full_until(policy_class: str) -> int:
    row = policy_row(policy_class)
    assert row is not None, policy_class
    return row.audit_full_until


class TestExecutorCloses:
    def test_an_executor_close_joins_its_decision_to_the_record_the_server_stamped(self, fleet):
        record = fleet.executor_close(1)

        found = fleet.closes()

        assert found.in_window == (mod.ExecutorClose(
            decision_id=record.id, escalation_id='esc-701-1', queue_dir=fleet.queue_dir, project='dark_factory',
            policy_class=SAMPLED_CLASS, closed_at=record.closed_at, evidence=record.closing_evidence, audited=None,
        ),)
        assert found.shortfalls == ()

    def test_a_close_is_found_whether_its_record_is_in_the_root_or_the_archive(self, fleet):
        fleet.executor_close(1, archived=False)
        fleet.executor_close(2, archived=True)

        assert sorted(_ids(fleet.closes().in_window)) == [f'dec-{SAMPLED_CLASS}-1', f'dec-{SAMPLED_CLASS}-2']

    def test_a_close_through_the_registry_verb_is_joined(self, fleet):
        fleet.escalation('esc-801-1', claimed_by=EXECUTOR_IDENTITY, claimed_at=NOW.isoformat())
        opened = fleet.decision('dec-verb', 'esc-801-1')
        closed = close_decision_with_evidence(
            opened.id, 'answered', 'quoted evidence', root=fleet.root,
            expected_project=opened.project, expected_escalations_dir=opened.escalations_dir,
        )
        assert closed is not None
        now = datetime.now(UTC)

        found = fleet.closes(window=Window(now - timedelta(days=1), now + timedelta(minutes=5)))

        assert _ids(found.in_window) == ['dec-verb']

    def test_closes_that_are_not_the_executors_are_excluded(self, fleet):
        fleet.executor_close(1)
        fleet.escalation('esc-901-1', resolved_by='sitting-preparer')
        fleet.decision('dec-carveout', 'esc-901-1', closed_at=_in_window(2))
        fleet.escalation('esc-902-1', resolved_by=COORDINATOR_IDENTITY, claimed_by=EXECUTOR_IDENTITY,
                         claimed_at=(NOW - timedelta(days=2)).isoformat())
        fleet.decision('dec-expired-claim', 'esc-902-1', closed_at=_in_window(3))
        fleet.escalation('esc-903-1', policy_class=None)
        fleet.decision('dec-classless', 'esc-903-1', closed_at=_in_window(4))
        fleet.escalation('esc-904-1', status='pending', resolved_by=None, archived=False)
        fleet.decision('dec-open', 'esc-904-1')

        found = fleet.closes()

        assert _ids(found.in_window) == [f'dec-{SAMPLED_CLASS}-1']
        assert found.shortfalls == ()

    def test_a_close_outside_the_window_is_not_in_it_but_its_audit_counts_for_its_class(self, fleet):
        fleet.executor_close(1)
        audited_old = fleet.executor_close(2, closed_at=OUTSIDE_WINDOW)
        fleet.executor_close(3, policy_class=FULL_CLASS, closed_at=OUTSIDE_WINDOW)
        fleet.audit(audited_old, AuditVerdict.OK)

        found = fleet.closes()

        assert _ids(found.in_window) == [f'dec-{SAMPLED_CLASS}-1']
        assert dict(found.audited_by_class) == {SAMPLED_CLASS: 1}

    def test_the_audit_count_is_fleet_wide_by_recorded_class_and_outlives_the_archive(self, fleet, tmp_path):
        elsewhere = tmp_path / 'src' / 'reify' / 'data' / 'escalations'
        audited = [
            fleet.executor_close(1),
            fleet.executor_close(2, closed_at=OUTSIDE_WINDOW),
            fleet.executor_close(3, policy_class=FULL_CLASS, closed_at=OUTSIDE_WINDOW),
            fleet.executor_close(4, queue=elsewhere),
        ]
        fleet.executor_close(5)
        fleet.executor_close(6, closed_at=OUTSIDE_WINDOW)
        for record in audited:
            fleet.audit(record, AuditVerdict.OK)
        expected = {SAMPLED_CLASS: 3, FULL_CLASS: 1}

        assert dict(fleet.closes().audited_by_class) == expected
        for path in [*fleet.queue.rglob('esc-*.json'), *elsewhere.rglob('esc-*.json')]:
            path.unlink()
        assert dict(fleet.closes().audited_by_class) == expected

    def test_closes_outside_the_window_are_never_joined(self, fleet, tmp_path):
        fleet.executor_close(1)
        fleet.executor_close(2, closed_at=OUTSIDE_WINDOW)
        (fleet.queue / 'archive' / '2026-09-25' / 'esc-702-1.json').unlink()
        fleet.executor_close(3, closed_at=OUTSIDE_WINDOW)
        (fleet.queue / 'archive' / '2026-09-25' / 'esc-703-1.json').write_text('{not json')
        fleet.executor_close(4, closed_at=OUTSIDE_WINDOW, queue=tmp_path / 'src' / 'reify' / 'data' / 'escalations')

        found = fleet.closes()

        assert _ids(found.in_window) == [f'dec-{SAMPLED_CLASS}-1']
        assert found.shortfalls == ()
        assert found.aged_out == 0

    def test_an_audited_close_carries_its_outcome(self, fleet):
        record = fleet.executor_close(1)
        fleet.audit(record, AuditVerdict.WRONG, 'the sibling commit did not carry it')

        (close,) = fleet.closes().in_window

        assert close.audited == AuditOutcome(
            AuditVerdict.WRONG, 'the sibling commit did not carry it', NOW.isoformat(), SAMPLED_CLASS,
            closed_at=record.closed_at,
        )

    def test_a_reopened_close_closed_again_is_unaudited_while_its_earlier_audit_still_counts(self, fleet):
        record = fleet.executor_close(1)
        fleet.audit(record, AuditVerdict.WRONG)
        assert reopen_decision(record.id, fleet.root, expected_project=record.project,
                               expected_escalations_dir=record.escalations_dir) is not None
        assert close_decision_with_evidence(record.id, 'answered', 'closed again', root=fleet.root,
                                            expected_project=record.project,
                                            expected_escalations_dir=record.escalations_dir) is not None

        found = fleet.closes(window=Window.trailing(datetime.now(UTC), days=1))

        (close,) = found.in_window
        assert close.audited is None
        assert dict(found.audited_by_class) == {SAMPLED_CLASS: 1}
        drawn = mod.draw(found.in_window, seed='2026-09-26', audited_by_class=found.audited_by_class)
        assert _ids(drawn.sampled) == [record.id]

    def test_a_close_in_an_unscanned_queue_is_a_shortfall(self, fleet, tmp_path):
        elsewhere = tmp_path / 'src' / 'reify' / 'data' / 'escalations'
        fleet.executor_close(1, queue=elsewhere)

        found = fleet.closes()

        assert found.in_window == ()
        (shortfall,) = found.shortfalls
        assert f'dec-{SAMPLED_CLASS}-1' in shortfall.reason
        assert shortfall.path == normalize_escalations_dir(elsewhere)

    def test_an_unreadable_in_window_escalation_record_is_a_shortfall(self, fleet):
        fleet.executor_close(1)
        (fleet.queue / 'esc-950-1.json').write_text('{not json')
        fleet.decision('dec-unreadable', 'esc-950-1', closed_at=_in_window(2))

        found = fleet.closes()

        assert _ids(found.in_window) == [f'dec-{SAMPLED_CLASS}-1']
        (shortfall,) = found.shortfalls
        assert shortfall.path == str(fleet.queue / 'esc-950-1.json')
        assert 'dec-unreadable' in shortfall.reason
        assert found.aged_out == 0

    def test_an_in_window_close_whose_escalation_aged_out_is_counted_not_a_shortfall(self, fleet):
        fleet.executor_close(1)
        pruned = fleet.decision('dec-missing', 'esc-951-1', closed_at=_in_window(3))

        found = fleet.closes()

        assert _ids(found.in_window) == [f'dec-{SAMPLED_CLASS}-1']
        assert found.shortfalls == ()
        assert found.aged_out == 1
        assert mod.executor_close_of(pruned, fleet.index()) == mod.AgedOut(
            decision_id='dec-missing', escalation_id='esc-951-1', queue_dir=fleet.queue_dir,
        )


class TestDraw:
    def _twelve(self, fleet: Fleet, policy_class: str = SAMPLED_CLASS) -> list[DecisionRecord]:
        return [fleet.executor_close(n, policy_class=policy_class) for n in range(1, 13)]

    def test_ten_distinct_closes_are_drawn_from_twelve(self, fleet):
        self._twelve(fleet)
        found = fleet.closes()

        draw = mod.draw(found.in_window, seed='2026-09-26', audited_by_class=found.audited_by_class,
                        sample_size=10)

        assert draw.state == 'drawn'
        assert draw.full == ()
        assert len(draw.sampled) == 10 == len(set(draw.sampled))
        assert set(draw.sampled) <= set(found.in_window)
        assert list(draw.sampled) == sorted(draw.sampled, key=_sort_key)
        assert draw.seed == '2026-09-26'
        assert draw.population == 12

    def test_the_draw_is_a_pure_function_of_seed_and_data(self, fleet):
        self._twelve(fleet)
        found = fleet.closes()

        def drawn(seed: str) -> frozenset[str]:
            closes = tuple(reversed(found.in_window))
            draw = mod.draw(closes, seed=seed, audited_by_class=found.audited_by_class, sample_size=10)
            return frozenset(_ids(draw.sampled))

        assert drawn('2026-09-26') == drawn('2026-09-26')
        assert drawn('2026-09-26') != drawn('2026-10-03')

    def test_an_audited_close_is_never_drawn(self, fleet):
        records = self._twelve(fleet)
        for record in records[:3]:
            fleet.audit(record, AuditVerdict.OK)
        found = fleet.closes()

        draw = mod.draw(found.in_window, seed='2026-09-26', audited_by_class=found.audited_by_class,
                        sample_size=10)

        assert sorted(_ids(draw.sampled)) == sorted(record.id for record in records[3:])
        assert draw.population == 12

    def test_a_class_below_its_full_audit_count_is_drawn_whole_beside_the_ten(self, fleet):
        assert _audit_full_until(FULL_CLASS) > 3
        full = [fleet.executor_close(n, policy_class=FULL_CLASS) for n in range(20, 23)]
        self._twelve(fleet)
        found = fleet.closes()

        draw = mod.draw(found.in_window, seed='2026-09-26', audited_by_class=found.audited_by_class,
                        sample_size=10)

        assert sorted(_ids(draw.full)) == sorted(record.id for record in full)
        assert len(draw.sampled) == 10
        assert {close.policy_class for close in draw.sampled} == {SAMPLED_CLASS}

    def test_a_class_at_its_full_audit_count_is_drawn_as_ordinary(self, fleet):
        for n in range(20, 23):
            fleet.executor_close(n, policy_class=FULL_CLASS)
        found = fleet.closes()
        reached = {FULL_CLASS: _audit_full_until(FULL_CLASS)}

        draw = mod.draw(found.in_window, seed='2026-09-26', audited_by_class=reached, sample_size=10)

        assert draw.full == ()
        assert len(draw.sampled) == 3

    def test_a_class_one_audit_short_of_its_full_count_is_still_drawn_whole(self, fleet):
        full = [fleet.executor_close(n, policy_class=FULL_CLASS) for n in range(20, 23)]
        found = fleet.closes()
        short = {FULL_CLASS: _audit_full_until(FULL_CLASS) - 1}

        draw = mod.draw(found.in_window, seed='2026-09-26', audited_by_class=short, sample_size=10)

        assert sorted(_ids(draw.full)) == sorted(record.id for record in full)
        assert draw.sampled == ()

    def test_unaudited_closes_do_not_advance_a_class_toward_its_full_count(self, fleet):
        for n in range(30, 30 + _audit_full_until(FULL_CLASS)):
            fleet.executor_close(n, policy_class=FULL_CLASS, closed_at=OUTSIDE_WINDOW)
        fresh = fleet.executor_close(1, policy_class=FULL_CLASS)
        found = fleet.closes()

        draw = mod.draw(found.in_window, seed='2026-09-26', audited_by_class=found.audited_by_class)

        assert _ids(draw.full) == [fresh.id]

    def test_the_draw_carries_the_aged_out_count(self, fleet):
        fleet.executor_close(1)
        found = fleet.closes()

        assert mod.draw(found.in_window, seed='2026-09-26', audited_by_class={}, aged_out=2).aged_out == 2
        assert mod.draw(found.in_window, seed='2026-09-26', audited_by_class={}).aged_out == 0

    def test_a_class_no_longer_in_the_table_is_drawn_as_ordinary(self, fleet):
        fleet.executor_close(1, policy_class=RETIRED_CLASS)
        found = fleet.closes()

        draw = mod.draw(found.in_window, seed='2026-09-26', audited_by_class=found.audited_by_class)

        assert draw.full == ()
        assert _ids(draw.sampled) == [f'dec-{RETIRED_CLASS}-1']

    def test_the_default_sample_is_ten(self):
        assert mod.SAMPLE_SIZE == 10


class TestDrawStates:
    def test_no_executor_close_in_the_window(self):
        draw = mod.draw((), seed='2026-09-26', audited_by_class={})

        assert (draw.state, draw.full, draw.sampled, draw.population) == ('no_executor_close', (), (), 0)

    def test_every_close_in_the_window_already_audited(self, fleet):
        fleet.audit(fleet.executor_close(1), AuditVerdict.OK)
        found = fleet.closes()

        draw = mod.draw(found.in_window, seed='2026-09-26', audited_by_class=found.audited_by_class)

        assert (draw.state, draw.full, draw.sampled, draw.population) == ('all_audited', (), (), 1)


class TestAuditOptions:
    def test_the_answer_vocabulary_is_keyed_by_verdict(self):
        assert set(mod.AUDIT_OPTIONS) == {str(verdict) for verdict in AuditVerdict}
        assert all(text.strip() for text in mod.AUDIT_OPTIONS.values())

    def test_the_answer_slot_is_the_bare_number_and_a_verdict(self):
        assert mod.answer_slot(12) == '12=ok|wrong[:note]'


class TestClassTally:
    def test_counts_audited_and_wrong_closes_of_one_class_on_one_queue(self, fleet, tmp_path):
        elsewhere = tmp_path / 'src' / 'reify' / 'data' / 'escalations'
        ok, wrong, _unaudited = (fleet.executor_close(n) for n in (1, 2, 3))
        old_wrong = fleet.executor_close(4, closed_at=OUTSIDE_WINDOW)
        full_wrong = fleet.executor_close(5, policy_class=FULL_CLASS)
        other_queue_wrong = fleet.executor_close(6, queue=elsewhere)
        fleet.audit(ok, AuditVerdict.OK)
        fleet.audit(wrong, AuditVerdict.WRONG)
        fleet.audit(old_wrong, AuditVerdict.WRONG)
        fleet.audit(full_wrong, AuditVerdict.WRONG)
        fleet.audit(other_queue_wrong, AuditVerdict.WRONG)
        table = {
            (fleet.queue_dir, SAMPLED_CLASS): (3, 2),
            (fleet.queue_dir, FULL_CLASS): (1, 1),
            (normalize_escalations_dir(elsewhere), SAMPLED_CLASS): (1, 1),
            (fleet.queue_dir, RETIRED_CLASS): (0, 0),
        }

        def tallies() -> dict[tuple[str, str], tuple[int, int]]:
            recorded = mod.RecordedAudits.read(fleet.root)
            return {(queue, cls): recorded.tally(queue, cls) for queue, cls in table}

        assert tallies() == table
        for path in [*fleet.queue.rglob('esc-*.json'), *elsewhere.rglob('esc-*.json')]:
            path.unlink()
        assert tallies() == table

    def test_the_tally_counts_the_class_the_audit_recorded(self, fleet):
        record = fleet.executor_close(1)
        fleet.audit(record, AuditVerdict.WRONG)
        fleet.escalation('esc-701-1', resolved_by=COORDINATOR_IDENTITY, policy_class=FULL_CLASS)

        assert mod.RecordedAudits.read(fleet.root).tally(fleet.queue_dir, SAMPLED_CLASS) == (1, 1)
        assert mod.RecordedAudits.read(fleet.root).tally(fleet.queue_dir, FULL_CLASS) == (0, 0)
        (fleet.queue / 'archive' / '2026-09-25' / 'esc-701-1.json').unlink()
        assert mod.RecordedAudits.read(fleet.root).tally(fleet.queue_dir, SAMPLED_CLASS) == (1, 1)

    def test_every_audited_close_of_a_reopened_record_counts(self, fleet):
        record = fleet.executor_close(1)
        fleet.audit(record, AuditVerdict.WRONG)
        identity = {'expected_project': record.project, 'expected_escalations_dir': record.escalations_dir}
        assert reopen_decision(record.id, fleet.root, **identity) is not None
        reclosed = close_decision_with_evidence(
            record.id, DecisionState.ANSWERED, 'executor evidence for the re-close', root=fleet.root, **identity,
        )
        assert reclosed is not None and reclosed.closed_at != record.closed_at

        fleet.audit(reclosed, AuditVerdict.OK)

        recorded = mod.RecordedAudits.read(fleet.root)
        assert recorded.tally(fleet.queue_dir, SAMPLED_CLASS) == (2, 1)
        assert recorded.by_class()[SAMPLED_CLASS] == 2

    def test_recording_folds_in_an_audit_written_after_the_read(self, fleet):
        first, second = fleet.executor_close(1), fleet.executor_close(2)
        fleet.audit(first, AuditVerdict.OK)
        recorded = mod.RecordedAudits.read(fleet.root)
        fleet.audit(second, AuditVerdict.WRONG)
        written = DecisionRecord.from_json(decision_path_for_id(second.id, root=fleet.root).read_text())

        folded = recorded.recording(written)

        assert recorded.tally(fleet.queue_dir, SAMPLED_CLASS) == (1, 0)
        assert folded.tally(fleet.queue_dir, SAMPLED_CLASS) == (2, 1)
        assert folded == mod.RecordedAudits.read(fleet.root)
        assert folded.recording(written) == folded
