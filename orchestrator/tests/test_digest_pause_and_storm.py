"""Tests for the digest's EWA storm classification and paused-waiting visibility (task 5311)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from escalation.models import Escalation
from test_digest import _make_digest_inputs

import orchestrator.digest as digest
from orchestrator.digest import (
    CategoryShare,
    EscalationStats,
    OverdueEscalation,
    PauseVisibility,
    RependedWhilePaused,
)
from orchestrator.event_store import EventStore, EventType


def _section(markdown: str, heading: str) -> str:
    start = markdown.index(f'{heading}\n')
    end = markdown.find('\n## ', start + len(heading))
    return markdown[start:] if end == -1 else markdown[start:end]


class TestDominantCategory:
    def test_sums_across_level_and_status(self) -> None:
        stats = EscalationStats(
            category_level_status_counts={
                ('retry_cap_exhausted', 1, 'pending'): 20,
                ('retry_cap_exhausted', 1, 'resolved'): 2,
                ('design_concern', 0, 'pending'): 5,
                ('infra_issue', 1, 'pending'): 2,
            },
        )
        assert stats.dominant_category() == CategoryShare(
            category='retry_cap_exhausted', count=22, total=29,
        )

    def test_string_form(self) -> None:
        share = CategoryShare(category='retry_cap_exhausted', count=22, total=29)
        assert str(share) == '22/29 retry_cap_exhausted'

    def test_empty_window_is_none(self) -> None:
        assert EscalationStats().dominant_category() is None

    def test_tie_breaks_to_smallest_category(self) -> None:
        stats = EscalationStats(
            category_level_status_counts={
                ('b_cat', 1, 'pending'): 3,
                ('a_cat', 1, 'pending'): 3,
            },
        )
        assert stats.dominant_category() == CategoryShare('a_cat', 3, 6)

    def test_share_is_frozen(self) -> None:
        share = CategoryShare('a_cat', 3, 6)
        field_name = 'count'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(share, field_name, 4)


class TestRenderDominantCategory:
    def _render(self, stats: EscalationStats) -> str:
        inputs = dataclasses.replace(_make_digest_inputs(), escalation_stats=stats)
        return digest.render_digest_markdown(inputs)

    def test_storm_window_names_dominant_category_in_ewa_section(self) -> None:
        markdown = self._render(EscalationStats(
            category_level_status_counts={
                ('retry_cap_exhausted', 1, 'pending'): 3,
                ('design_concern', 0, 'pending'): 1,
            },
        ))
        line = '- Dominant escalation category: 3/4 retry_cap_exhausted'
        assert line in _section(markdown, '## EWA').splitlines()
        assert markdown.count(line) == 1

    def test_empty_stats_render_none(self) -> None:
        markdown = self._render(EscalationStats())
        line = '- Dominant escalation category: none'
        assert line in _section(markdown, '## EWA').splitlines()
        assert markdown.count(line) == 1


def _emit_repend(store: EventStore, task_id: str, escalation_id: str) -> None:
    store.emit(
        EventType.repend_while_paused,
        task_id=task_id,
        data={'escalation_id': escalation_id, 'pause_reason': 'ewa_trip_29.1818'},
    )


class TestRependedDuringCurrentPause:
    def test_only_repends_after_the_latest_resume_latest_per_task(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'events.db'
        store = EventStore(db_path, 'run-a')
        _emit_repend(store, '1', 'esc-1-1')
        store.emit(EventType.scheduler_resumed)
        _emit_repend(store, '2', 'esc-2-1')
        _emit_repend(store, '3', 'esc-3-1')
        _emit_repend(store, '2', 'esc-2-4')

        waiting = digest.repended_during_current_pause(db_path)

        assert isinstance(waiting, tuple)
        assert [(w.task_id, w.escalation_id) for w in waiting] == [
            ('2', 'esc-2-4'), ('3', 'esc-3-1'),
        ]
        assert all(isinstance(w, RependedWhilePaused) and w.repended_at for w in waiting)
        field_name = 'task_id'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(waiting[0], field_name, '9')

    def test_without_any_resume_every_repend_counts(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'events.db'
        store = EventStore(db_path, 'run-a')
        _emit_repend(store, '1', 'esc-1-1')
        _emit_repend(store, '2', 'esc-2-1')

        waiting = digest.repended_during_current_pause(db_path)

        assert [w.task_id for w in waiting] == ['1', '2']

    def test_repends_from_another_run_are_included(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'events.db'
        EventStore(db_path, 'run-a').emit(EventType.scheduler_resumed)
        _emit_repend(EventStore(db_path, 'run-b'), '7', 'esc-7-1')

        waiting = digest.repended_during_current_pause(db_path)

        assert [(w.task_id, w.escalation_id) for w in waiting] == [('7', 'esc-7-1')]

    def test_missing_db_is_empty(self, tmp_path: Path) -> None:
        assert digest.repended_during_current_pause(tmp_path / 'absent.db') == ()

    def test_restart_that_declines_to_reassert_ends_the_pause(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'events.db'
        _emit_repend(EventStore(db_path, 'run-a'), '1', 'esc-1-1')
        run_b = EventStore(db_path, 'run-b')
        run_b.emit(
            EventType.scheduler_pause_restored,
            data={'reason': 'ewa_trip_73.5900', 'reasserted': False},
        )
        run_b.emit(EventType.scheduler_paused, data={'reason': 'park_stop_trip: later'})
        _emit_repend(run_b, '2', 'esc-2-1')

        waiting = digest.repended_during_current_pause(db_path)

        assert [(w.task_id, w.escalation_id) for w in waiting] == [('2', 'esc-2-1')]

    def test_restart_that_reasserts_continues_the_pause(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'events.db'
        _emit_repend(EventStore(db_path, 'run-a'), '1', 'esc-1-1')
        run_b = EventStore(db_path, 'run-b')
        run_b.emit(
            EventType.scheduler_pause_restored,
            data={'reason': 'ewa_trip_73.5900', 'reasserted': True},
        )
        _emit_repend(run_b, '2', 'esc-2-1')

        waiting = digest.repended_during_current_pause(db_path)

        assert [w.task_id for w in waiting] == ['1', '2']

    def test_pre_4559_restore_without_reasserted_key_continues_the_pause(
        self, tmp_path: Path,
    ) -> None:
        db_path = tmp_path / 'events.db'
        _emit_repend(EventStore(db_path, 'run-a'), '1', 'esc-1-1')
        run_b = EventStore(db_path, 'run-b')
        run_b.emit(EventType.scheduler_pause_restored, data={'reason': 'park_stop_trip: x'})
        _emit_repend(run_b, '2', 'esc-2-1')

        waiting = digest.repended_during_current_pause(db_path)

        assert [w.task_id for w in waiting] == ['1', '2']

    def test_the_later_pause_end_wins(self, tmp_path: Path) -> None:
        db_path = tmp_path / 'events.db'
        store = EventStore(db_path, 'run-a')
        store.emit(
            EventType.scheduler_pause_restored,
            data={'reason': 'ewa_trip_73.5900', 'reasserted': False},
        )
        _emit_repend(store, '1', 'esc-1-1')
        store.emit(EventType.scheduler_resumed)
        _emit_repend(store, '2', 'esc-2-1')

        waiting = digest.repended_during_current_pause(db_path)

        assert [w.task_id for w in waiting] == ['2']


FIXED_NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def _pending(
    esc_id: str, *, category: str = 'scheduler_paused', age: timedelta | None = None,
    timestamp: str | None = None, status: str = 'pending',
) -> Escalation:
    if timestamp is None:
        assert age is not None
        timestamp = (FIXED_NOW - age).isoformat()
    return Escalation(
        id=esc_id,
        task_id='__scheduler__',
        agent_role='orchestrator',
        severity='blocking',
        category=category,
        summary=f'summary of {esc_id}',
        timestamp=timestamp,
        status=status,
        level=1,
    )


class TestOverduePendingEscalations:
    def _overdue(self, pending: list[Escalation]) -> tuple[OverdueEscalation, ...]:
        return digest.overdue_pending_escalations(
            pending,
            category='scheduler_paused',
            min_age=digest.SCHEDULER_PAUSE_RESURFACE_AGE,
            now=FIXED_NOW,
        )

    def test_only_pending_records_of_the_category_past_the_age(self) -> None:
        overdue = self._overdue([
            _pending('esc-3h', age=timedelta(hours=3)),
            _pending('esc-1h', age=timedelta(hours=1)),
            _pending('esc-stranded', category='stranded_blocked', age=timedelta(hours=5)),
            _pending('esc-resolved', age=timedelta(hours=5), status='resolved'),
            _pending('esc-garbage', timestamp='garbage'),
            _pending('esc-naive', timestamp='2026-10-11T01:00:00'),
        ])

        assert overdue == (
            OverdueEscalation(
                escalation_id='esc-3h', age=timedelta(hours=3), summary='summary of esc-3h',
            ),
        )
        field_name = 'age'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(overdue[0], field_name, timedelta(0))

    def test_oldest_first(self) -> None:
        overdue = self._overdue([
            _pending('esc-3h', age=timedelta(hours=3)),
            _pending('esc-9h', age=timedelta(hours=9)),
        ])

        assert [o.escalation_id for o in overdue] == ['esc-9h', 'esc-3h']


class TestRenderPausedWaiting:
    def _render(self, visibility: PauseVisibility) -> str:
        inputs = dataclasses.replace(_make_digest_inputs(), pause_visibility=visibility)
        return digest.render_digest_markdown(inputs)

    def test_default_inputs_render_not_paused(self) -> None:
        inputs = _make_digest_inputs()
        assert inputs.pause_visibility == PauseVisibility()

        section = _section(digest.render_digest_markdown(inputs), '## Paused, waiting')

        assert '- Not paused' in section.splitlines()

    def test_paused_lists_each_waiting_task(self) -> None:
        markdown = self._render(PauseVisibility(
            pause_reason='ewa_trip_29.1818',
            repended_waiting=(
                RependedWhilePaused('5260', 'esc-5260-3', '2026-10-11T09:00:00+00:00'),
                RependedWhilePaused('5261', 'esc-5261-1', '2026-10-11T10:00:00+00:00'),
            ),
        ))

        lines = _section(markdown, '## Paused, waiting').splitlines()
        header = '- Paused with 2 re-pended tasks waiting (pause reason: ewa_trip_29.1818)'
        at = lines.index(header)
        assert lines[at + 1].startswith('  - task 5260 ')
        assert 'esc-5260-3' in lines[at + 1]
        assert lines[at + 2].startswith('  - task 5261 ')
        assert 'esc-5261-1' in lines[at + 2]

    def test_overdue_escalation_listed_even_when_not_paused(self) -> None:
        markdown = self._render(PauseVisibility(
            overdue_pause_escalations=(
                OverdueEscalation('esc-__scheduler__-4', timedelta(hours=3), 'Scheduler paused'),
            ),
        ))

        section = _section(markdown, '## Paused, waiting')
        assert '- Not paused' in section.splitlines()
        [overdue_line] = [line for line in section.splitlines() if 'esc-__scheduler__-4' in line]
        assert '3.0h' in overdue_line

    def test_section_follows_the_ewa_section(self) -> None:
        markdown = self._render(PauseVisibility())

        ewa_end = markdown.find('\n## ', markdown.index('## EWA\n') + 1)
        assert markdown[ewa_end + 1:].startswith('## Paused, waiting\n')
