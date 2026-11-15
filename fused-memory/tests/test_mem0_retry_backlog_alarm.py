"""``services/mem0_retry_backlog_alarm.py``: a Mem0 write left retrying too long is escalated, never dropped."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from _mem0_intent_fixtures import seed_mem0_intent

from fused_memory.services.mem0_retry_backlog_alarm import (
    ANCHOR_TASK_ID,
    alarm_on_stale_mem0_retries,
)
from fused_memory.services.write_journal import (
    Mem0IntentStatus,
    Mem0RetryBacklog,
    WriteJournal,
)

_NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
_MAX_AGE = timedelta(hours=1)
_PROJECT = 'dark_factory'


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'wj')
    await j.initialize()
    yield j
    await j.close()


@pytest.fixture
def root(tmp_path) -> Path:
    project_root = tmp_path / 'project'
    project_root.mkdir()
    return project_root


def _escalations(project_root: Path) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted((project_root / 'data' / 'escalations').glob('esc-*.json'))
    ]


async def _seed_retrying(journal: WriteJournal, *, age: timedelta, project_id: str = _PROJECT) -> str:
    return await seed_mem0_intent(
        journal,
        status=Mem0IntentStatus.RETRYING,
        project_id=project_id,
        created_at=_NOW - age,
        retry_at=_NOW + timedelta(minutes=5),
    )


async def _alarm(journal: WriteJournal, roots: dict[str, str]) -> list[str]:
    return await alarm_on_stale_mem0_retries(
        journal, now=_NOW, max_age=_MAX_AGE, project_roots=roots
    )


class TestAStaleBacklogIsEscalated:
    @pytest.mark.asyncio
    async def test_one_record_names_the_backlog_and_how_to_check_it(self, journal, root):
        await _seed_retrying(journal, age=timedelta(hours=2))

        filed = await _alarm(journal, {_PROJECT: str(root)})

        (record,) = _escalations(root)
        assert filed == [record['id']]
        assert record['task_id'] == ANCHOR_TASK_ID
        assert record['category'] == 'mem0_write_retry_backlog'
        assert record['severity'] == 'blocking'
        detail = record['detail']
        assert f'project_id={_PROJECT}' in detail
        assert 'count=1' in detail
        assert f'oldest_created_at={(_NOW - timedelta(hours=2)).isoformat()}' in detail
        assert 'age_seconds=7200' in detail
        assert 'max_attempts=1' in detail
        assert 'threshold_seconds=3600' in detail
        assert "status='retrying'" in detail
        action = record['suggested_action']
        assert 'OpenAI' in action
        assert 'mem0_intents' in action
        assert "status='retrying'" in action

    @pytest.mark.asyncio
    async def test_a_second_check_folds_into_the_open_record(self, journal, root):
        await _seed_retrying(journal, age=timedelta(hours=2))
        roots = {_PROJECT: str(root)}

        first = await _alarm(journal, roots)
        second = await _alarm(journal, roots)

        assert first == second
        assert len(_escalations(root)) == 1


class TestNothingToEscalate:
    @pytest.mark.asyncio
    async def test_a_retry_younger_than_the_threshold_files_nothing(self, journal, root):
        await _seed_retrying(journal, age=timedelta(minutes=30))

        assert await _alarm(journal, {_PROJECT: str(root)}) == []
        assert _escalations(root) == []

    @pytest.mark.asyncio
    async def test_old_rows_that_are_not_retrying_file_nothing(self, journal, root):
        for status in (
            Mem0IntentStatus.COMPLETED,
            Mem0IntentStatus.FAILED,
            Mem0IntentStatus.DEAD,
        ):
            await seed_mem0_intent(
                journal, status=status, project_id=_PROJECT,
                created_at=_NOW - timedelta(days=2),
            )

        assert await _alarm(journal, {_PROJECT: str(root)}) == []
        assert _escalations(root) == []


class TestTheAlarmNeverMisfilesOrRaises:
    @pytest.mark.asyncio
    async def test_a_project_with_no_known_root_is_warned_not_filed(
        self, journal, root, caplog
    ):
        await _seed_retrying(journal, age=timedelta(hours=2), project_id='reify')

        with caplog.at_level(logging.WARNING):
            filed = await _alarm(journal, {_PROJECT: str(root)})

        assert filed == []
        assert _escalations(root) == []
        assert any(
            r.levelno == logging.WARNING and 'reify' in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_an_unreadable_journal_is_logged_not_raised(self, tmp_path, root, caplog):
        class _UnreadableJournal(WriteJournal):
            async def get_mem0_retry_backlog(self) -> list[Mem0RetryBacklog]:
                raise sqlite3.OperationalError('database is locked')

        with caplog.at_level(logging.ERROR):
            filed = await _alarm(_UnreadableJournal(tmp_path / 'wj2'), {_PROJECT: str(root)})

        assert filed == []
        assert _escalations(root) == []
        assert any(r.levelno >= logging.ERROR for r in caplog.records)
