"""Tests for the WriteClassifier fallback storm alarm (task 6624).

Driven through the public ``ClassificationFallbackAlarm.record`` surface with a
fake clock, filing into a real escalation queue under ``tmp_path``.
"""

import logging
from pathlib import Path

import pytest
from fused_memory.services.classification_fallback_alarm import (
    _ANCHOR_TASK_ID,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_SECONDS,
    JOURNAL_PARAM_KEY,
    LOG_EVENT,
    ClassificationFallbackAlarm,
)

from fused_memory.middleware import _folded_escalation
from fused_memory.models.enums import ClassificationFallback
from fused_memory.services import classification_fallback_alarm

_ALARM_LOGGER = classification_fallback_alarm.__name__

_needs_escalation = pytest.mark.skipif(
    not _folded_escalation.HAS_ESCALATION,
    reason='escalation package unavailable (minimal env); filing is a logged no-op there',
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> _FakeClock:
    return _FakeClock()


@pytest.fixture
def alarm(clock) -> ClassificationFallbackAlarm:
    return ClassificationFallbackAlarm(time_provider=clock)


def _pending(tmp_path: Path) -> list:
    from escalation.queue import EscalationQueue  # noqa: PLC0415

    return EscalationQueue(tmp_path / 'data' / 'escalations').get_pending()


def _detail_fields(detail: str) -> dict[str, str]:
    return dict(
        line.split('=', 1) for line in detail.splitlines() if '=' in line.split(' ', 1)[0]
    )


async def _record_n(
    alarm: ClassificationFallbackAlarm,
    n: int,
    *,
    project_id: str = 'p1',
    project_root: str | None,
    fallback: ClassificationFallback | None = ClassificationFallback.llm_error,
) -> list[str | None]:
    return [
        await alarm.record(fallback, project_id=project_id, project_root=project_root)
        for _ in range(n)
    ]


class TestCounting:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('fallback', [None, ClassificationFallback.no_confident_match])
    async def test_non_llm_outcomes_never_count(self, alarm, tmp_path, fallback):
        returned = await _record_n(
            alarm, DEFAULT_THRESHOLD + 2, project_root=str(tmp_path), fallback=fallback,
        )

        assert returned == [None] * (DEFAULT_THRESHOLD + 2)
        assert alarm.tracked_projects == frozenset()

    @pytest.mark.asyncio
    async def test_below_threshold_returns_none(self, alarm, tmp_path):
        returned = await _record_n(alarm, DEFAULT_THRESHOLD - 1, project_root=str(tmp_path))

        assert returned == [None] * (DEFAULT_THRESHOLD - 1)
        assert alarm.tracked_projects == frozenset({'p1'})

    @pytest.mark.asyncio
    async def test_projects_never_pool_into_a_fire(self, alarm, tmp_path):
        p1 = await _record_n(alarm, DEFAULT_THRESHOLD - 1, project_root=str(tmp_path))
        p2 = await _record_n(
            alarm, DEFAULT_THRESHOLD - 1, project_id='p2', project_root=str(tmp_path),
        )

        assert p1 + p2 == [None] * (2 * (DEFAULT_THRESHOLD - 1))
        assert alarm.tracked_projects == frozenset({'p1', 'p2'})

    @pytest.mark.asyncio
    async def test_events_older_than_the_window_age_out(self, alarm, clock, tmp_path):
        await _record_n(alarm, DEFAULT_THRESHOLD - 1, project_root=str(tmp_path))
        clock.now += DEFAULT_WINDOW_SECONDS + 1

        returned = await _record_n(alarm, DEFAULT_THRESHOLD - 1, project_root=str(tmp_path))

        assert returned == [None] * (DEFAULT_THRESHOLD - 1)


@_needs_escalation
class TestFiling:
    @pytest.mark.asyncio
    async def test_non_llm_outcomes_file_nothing(self, alarm, tmp_path):
        for fallback in (None, ClassificationFallback.no_confident_match):
            await _record_n(
                alarm, DEFAULT_THRESHOLD + 2, project_root=str(tmp_path), fallback=fallback,
            )

        assert _pending(tmp_path) == []

    @pytest.mark.asyncio
    async def test_below_threshold_files_nothing(self, alarm, tmp_path):
        await _record_n(alarm, DEFAULT_THRESHOLD - 1, project_root=str(tmp_path))

        assert _pending(tmp_path) == []

    @pytest.mark.asyncio
    async def test_the_threshold_crossing_files_one_escalation(self, alarm, tmp_path):
        await _record_n(alarm, DEFAULT_THRESHOLD - 1, project_root=str(tmp_path))

        esc_id = await alarm.record(
            ClassificationFallback.llm_no_json, project_id='p1', project_root=str(tmp_path),
        )

        [esc] = _pending(tmp_path)
        assert esc_id == esc.id
        assert esc.task_id == _ANCHOR_TASK_ID
        assert esc.level == 1
        assert esc.severity == 'blocking'
        assert 'p1' in esc.summary
        assert str(DEFAULT_THRESHOLD) in esc.summary
        assert 'first burst' in esc.summary.lower()
        fields = _detail_fields(esc.detail)
        assert fields['project_id'] == 'p1'
        assert fields['count'] == str(DEFAULT_THRESHOLD)
        assert fields['threshold'] == str(DEFAULT_THRESHOLD)
        assert fields['window_seconds'] == str(DEFAULT_WINDOW_SECONDS)
        assert ClassificationFallback.llm_error.value in fields['reasons']
        assert ClassificationFallback.llm_no_json.value in fields['reasons']
        assert JOURNAL_PARAM_KEY == 'classification_fallback'
        assert JOURNAL_PARAM_KEY in esc.detail
        assert 'not blocked' in esc.detail

    @pytest.mark.asyncio
    async def test_a_continuing_burst_files_no_second_escalation(
        self, alarm, clock, tmp_path,
    ):
        first = await _record_n(alarm, DEFAULT_THRESHOLD, project_root=str(tmp_path))
        in_window = await _record_n(alarm, DEFAULT_THRESHOLD + 3, project_root=str(tmp_path))
        clock.now += DEFAULT_WINDOW_SECONDS + 1
        next_window = await _record_n(alarm, DEFAULT_THRESHOLD, project_root=str(tmp_path))

        [esc] = _pending(tmp_path)
        assert first[-1] == esc.id
        assert in_window == [None] * (DEFAULT_THRESHOLD + 3)
        assert next_window[-1] == esc.id

    @pytest.mark.asyncio
    async def test_each_project_files_into_its_own_queue(self, alarm, tmp_path):
        root_1, root_2 = tmp_path / 'one', tmp_path / 'two'

        await _record_n(alarm, DEFAULT_THRESHOLD, project_root=str(root_1))
        await _record_n(
            alarm, DEFAULT_THRESHOLD - 1, project_id='p2', project_root=str(root_2),
        )

        assert len(_pending(root_1)) == 1
        assert _pending(root_2) == []


class TestFailSoft:
    @pytest.mark.asyncio
    async def test_an_unresolved_project_root_warns_and_files_nothing(
        self, alarm, caplog,
    ):
        caplog.set_level(logging.WARNING, logger=_ALARM_LOGGER)

        returned = await _record_n(alarm, DEFAULT_THRESHOLD, project_root=None)

        assert returned == [None] * DEFAULT_THRESHOLD
        [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
        message = warning.getMessage()
        assert 'p1' in message
        assert str(DEFAULT_THRESHOLD) in message
        assert 'MemoryService.set_known_projects' in message

    @pytest.mark.asyncio
    async def test_a_burst_logs_exactly_one_error_line(self, alarm, caplog):
        caplog.set_level(logging.WARNING, logger=_ALARM_LOGGER)

        await _record_n(alarm, DEFAULT_THRESHOLD + 3, project_root=None)

        [error] = [
            r for r in caplog.records
            if r.levelno == logging.ERROR and r.getMessage().startswith(LOG_EVENT)
        ]
        message = error.getMessage()
        assert "project_id='p1'" in message
        assert str(DEFAULT_THRESHOLD) in message

    @pytest.mark.asyncio
    async def test_a_failing_filer_never_raises(
        self, alarm, tmp_path, caplog, monkeypatch,
    ):
        def _boom(*_args, **_kwargs):
            raise RuntimeError('queue on fire')

        monkeypatch.setattr(
            classification_fallback_alarm,
            'emit_classification_fallback_storm_escalation',
            _boom,
        )
        caplog.set_level(logging.WARNING, logger=_ALARM_LOGGER)

        returned = await _record_n(alarm, DEFAULT_THRESHOLD, project_root=str(tmp_path))

        assert returned == [None] * DEFAULT_THRESHOLD
        assert any(r.exc_info is not None for r in caplog.records)

    @pytest.mark.asyncio
    async def test_an_unwritable_project_root_never_raises(self, alarm, tmp_path):
        regular_file = tmp_path / 'not-a-dir'
        regular_file.write_text('')

        returned = await _record_n(
            alarm, DEFAULT_THRESHOLD, project_root=str(regular_file / 'project'),
        )

        assert returned == [None] * DEFAULT_THRESHOLD
