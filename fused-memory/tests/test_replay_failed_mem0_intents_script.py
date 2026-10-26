"""``scripts/replay_failed_mem0_intents.py``: replay 'failed' Mem0 intents through add_memory (task 7397).

The async core runs against a real ``WriteJournal`` on ``tmp_path`` with an
injected ``add_memory`` callable standing in for the live MCP tool; ``main`` is
run only in its dry-run form, which never opens an MCP client.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from _fm_helpers import load_script_module

from fused_memory.services.write_journal import WriteJournal

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'replay_failed_mem0_intents.py'

_script = load_script_module(SCRIPT_PATH, mod_name='replay_failed_mem0_intents')
Selection = _script.Selection
ReplayResult = _script.ReplayResult

_T = datetime(2026, 10, 22, 17, 40, tzinfo=UTC)
_LANDED_ACK = {
    'memory_ids': ['new-1'],
    'stores_written': ['mem0'],
    'stores_pending_retry': [],
    'message': "Memory queued for ['mem0']",
}
_PENDING_ACK = {
    'memory_ids': [],
    'stores_written': [],
    'stores_pending_retry': ['mem0'],
    'message': 'Memory queued for [] [mem0_pending_retry: rate_limited: RateLimitError: 429]',
}


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'wj')
    await j.initialize()
    yield j
    await j.close()


async def _seed(
    journal: WriteJournal,
    *,
    status: str = 'failed',
    content: str = 'always pin dependencies exactly',
    agent_id: str | None = 'agent-a',
    session_id: str | None = None,
    metadata: dict | None = None,
    created_at: datetime | None = None,
) -> str:
    intent_id = str(uuid.uuid4())
    await journal.log_mem0_intent(
        intent_id=intent_id,
        write_op_id=str(uuid.uuid4()),
        project_id='dark_factory',
        agent_id=agent_id,
        session_id=session_id,
        category='procedural_knowledge',
        content=content,
        metadata=metadata if metadata is not None else {'category': 'procedural_knowledge'},
        payload_digest='d',
    )
    if status == 'retrying':
        await journal.schedule_mem0_retry(intent_id, reason='rate_limited', retry_at=_T)
    elif status != 'pending':
        await journal.resolve_mem0_intent(
            intent_id, status, reason="Error code: 429 - {'error': 'no credits'}"
        )
    if created_at is not None:
        with closing(sqlite3.connect(journal.db_path)) as conn, conn:
            conn.execute(
                'UPDATE mem0_intents SET created_at = ? WHERE id = ?',
                (created_at.isoformat(), intent_id),
            )
    return intent_id


async def _intent(journal: WriteJournal, intent_id: str) -> dict:
    (row,) = [r for r in await journal.get_mem0_intents() if r['id'] == intent_id]
    return row


class _AddMemory:
    """Stands in for the live add_memory tool: replies with queued acks or raises."""

    def __init__(self, *replies: dict | Exception) -> None:
        self.calls: list[dict[str, Any]] = []
        self._replies = list(replies)

    async def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(arguments)
        reply = self._replies.pop(0) if self._replies else _LANDED_ACK
        if isinstance(reply, Exception):
            raise reply
        return reply


def _by_id(outcomes) -> dict:
    return {o.intent_id: o for o in outcomes}


class TestSelection:
    def test_exactly_one_selection_form_is_required(self):
        with pytest.raises(ValueError):
            Selection()
        with pytest.raises(ValueError):
            Selection(intent_ids=('a',), created_from=_T, created_before=_T)

    @pytest.mark.asyncio
    async def test_listed_ids_select_only_failed_rows(self, journal):
        failed = await _seed(journal)
        other_failed = await _seed(journal)
        completed = await _seed(journal, status='completed')

        selected = await _script.select_intents(
            journal, Selection(intent_ids=(failed, completed))
        )
        outcomes = await _script.replay(
            journal,
            _AddMemory(),
            selection=Selection(intent_ids=(failed, completed)),
            apply=False,
        )

        assert [row['id'] for row in selected] == [failed]
        assert other_failed not in _by_id(outcomes)
        assert _by_id(outcomes)[failed].outcome is ReplayResult.DRY_RUN
        assert _by_id(outcomes)[completed].outcome is ReplayResult.NOT_FAILED

    @pytest.mark.asyncio
    async def test_a_window_selects_failed_rows_in_its_half_open_range(self, journal):
        before = await _seed(journal, created_at=_T - timedelta(minutes=1))
        at_start = await _seed(journal, created_at=_T)
        inside = await _seed(journal, created_at=_T + timedelta(minutes=10))
        await _seed(journal, status='completed', created_at=_T + timedelta(minutes=10))
        await _seed(journal, status='retrying', created_at=_T + timedelta(minutes=10))
        at_end = await _seed(journal, created_at=_T + timedelta(minutes=30))

        selected = await _script.select_intents(
            journal,
            Selection(created_from=_T, created_before=_T + timedelta(minutes=30)),
        )

        ids = {row['id'] for row in selected}
        assert ids == {at_start, inside}
        assert before not in ids
        assert at_end not in ids

    @pytest.mark.asyncio
    async def test_skip_ids_exclude_rows_from_either_form(self, journal):
        kept = await _seed(journal, created_at=_T)
        skipped = await _seed(journal, created_at=_T)

        by_window = await _script.select_intents(
            journal,
            Selection(
                created_from=_T,
                created_before=_T + timedelta(minutes=1),
                skip_ids=frozenset({skipped}),
            ),
        )
        by_id = await _script.replay(
            journal,
            _AddMemory(),
            selection=Selection(intent_ids=(kept, skipped), skip_ids=frozenset({skipped})),
            apply=False,
        )

        assert [row['id'] for row in by_window] == [kept]
        assert set(_by_id(by_id)) == {kept}


class TestReplay:
    @pytest.mark.asyncio
    async def test_forwards_the_full_row_and_only_the_identity_it_has(self, journal):
        content = 'the full text of a write that hit a 429, longer than any prefix. ' * 6
        metadata = {'category': 'procedural_knowledge', 'x_ref': {'task': 7397}}
        with_agent = await _seed(journal, content=content, metadata=metadata)
        with_session = await _seed(journal, agent_id=None, session_id='sess-1')
        add_memory = _AddMemory()

        await _script.replay(
            journal,
            add_memory,
            selection=Selection(intent_ids=(with_agent, with_session)),
            apply=True,
        )

        first, second = add_memory.calls
        assert first == {
            'content': content,
            'project_id': 'dark_factory',
            'category': 'procedural_knowledge',
            'metadata': metadata,
            'agent_id': 'agent-a',
        }
        assert 'agent_id' not in second
        assert second['session_id'] == 'sess-1'

    @pytest.mark.asyncio
    async def test_a_landed_replay_resolves_the_intent_completed(self, journal):
        intent_id = await _seed(journal)

        (outcome,) = await _script.replay(
            journal, _AddMemory(), selection=Selection(intent_ids=(intent_id,)), apply=True
        )

        assert outcome.outcome is ReplayResult.LANDED
        row = await _intent(journal, intent_id)
        assert row['status'] == 'completed'
        assert 'replayed' in row['reason']
        assert 'new-1' in row['reason']

    @pytest.mark.asyncio
    async def test_a_replay_pending_server_retry_resolves_the_intent_completed(self, journal):
        intent_id = await _seed(journal)

        (outcome,) = await _script.replay(
            journal,
            _AddMemory(_PENDING_ACK),
            selection=Selection(intent_ids=(intent_id,)),
            apply=True,
        )

        assert outcome.outcome is ReplayResult.PENDING_RETRY
        row = await _intent(journal, intent_id)
        assert row['status'] == 'completed'
        assert 'replayed' in row['reason']
        assert 'pending server' in row['reason']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('ack', 'expected', 'detail_fragment'),
        [
            (
                {
                    'error': 'a near-duplicate of 1234 exists',
                    'error_type': 'ProceduralKnowledgeNearDuplicateWriteRejected',
                },
                'REJECTED',
                'ProceduralKnowledgeNearDuplicateWriteRejected',
            ),
            (
                {
                    'memory_ids': [],
                    'stores_written': [],
                    'stores_pending_retry': [],
                    'message': 'Memory queued for [] [mem0_error: 429 no credits]',
                },
                'NOT_LANDED',
                'mem0_error',
            ),
        ],
        ids=['guard-rejection', 'mem0-error'],
    )
    async def test_a_replay_that_did_not_land_leaves_the_intent_failed(
        self, journal, ack, expected, detail_fragment
    ):
        intent_id = await _seed(journal)

        (outcome,) = await _script.replay(
            journal,
            _AddMemory(ack),
            selection=Selection(intent_ids=(intent_id,)),
            apply=True,
        )

        assert outcome.outcome is ReplayResult[expected]
        assert detail_fragment in outcome.detail
        assert (await _intent(journal, intent_id))['status'] == 'failed'

    @pytest.mark.asyncio
    async def test_a_raising_call_is_reported_and_the_next_row_still_runs(self, journal):
        first = await _seed(journal, created_at=_T)
        second = await _seed(journal, created_at=_T + timedelta(seconds=1))

        outcomes = await _script.replay(
            journal,
            _AddMemory(ConnectionError('server went away'), _LANDED_ACK),
            selection=Selection(intent_ids=(first, second)),
            apply=True,
        )

        assert _by_id(outcomes)[first].outcome is ReplayResult.ERROR
        assert 'ConnectionError' in _by_id(outcomes)[first].detail
        assert _by_id(outcomes)[second].outcome is ReplayResult.LANDED
        assert (await _intent(journal, first))['status'] == 'failed'

    @pytest.mark.asyncio
    async def test_a_second_run_replays_nothing(self, journal):
        await _seed(journal, created_at=_T)
        await _seed(journal, created_at=_T)
        window = Selection(created_from=_T, created_before=_T + timedelta(minutes=1))
        add_memory = _AddMemory()

        first = await _script.replay(journal, add_memory, selection=window, apply=True)
        second = await _script.replay(journal, add_memory, selection=window, apply=True)

        assert [o.outcome for o in first] == [ReplayResult.LANDED] * 2
        assert second == []
        assert len(add_memory.calls) == 2

    @pytest.mark.asyncio
    async def test_a_dry_run_calls_nothing_and_changes_nothing(self, journal):
        intent_id = await _seed(journal)
        before = await _intent(journal, intent_id)
        add_memory = _AddMemory()

        (outcome,) = await _script.replay(
            journal, add_memory, selection=Selection(intent_ids=(intent_id,)), apply=False
        )

        assert outcome.outcome is ReplayResult.DRY_RUN
        assert add_memory.calls == []
        assert await _intent(journal, intent_id) == before


def _seed_journal_dir(data_dir: Path, count: int) -> list[str]:
    async def _seed_all() -> list[str]:
        j = WriteJournal(data_dir)
        await j.initialize()
        try:
            return [await _seed(j) for _ in range(count)]
        finally:
            await j.close()

    return asyncio.run(_seed_all())


def _statuses(data_dir: Path) -> dict[str, str]:
    with closing(sqlite3.connect(data_dir / 'write_journal.db')) as conn:
        return dict(conn.execute('SELECT id, status FROM mem0_intents').fetchall())


class TestMain:
    def test_a_dry_run_prints_each_selected_row_and_opens_no_client(self, tmp_path, capsys):
        data_dir = tmp_path / 'wj'
        first, second, skipped = _seed_journal_dir(data_dir, 3)
        before = _statuses(data_dir)

        code = _script.main([
            '--journal-dir', str(data_dir),
            '--intent-id', first,
            '--intent-id', second,
            '--intent-id', skipped,
            '--skip', skipped,
            # Nothing listens here, so any connection attempt would fail the run.
            '--mcp-url', 'http://127.0.0.1:1/mcp',
        ])

        assert code == 0
        out = capsys.readouterr().out
        row_lines = [line for line in out.splitlines() if line.split('\t')[0] in before]
        assert sorted(line.split('\t')[0] for line in row_lines) == sorted([first, second])
        assert _statuses(data_dir) == before

    def test_no_selection_is_a_usage_error(self, tmp_path):
        data_dir = tmp_path / 'wj'
        _seed_journal_dir(data_dir, 1)

        with pytest.raises(SystemExit) as excinfo:
            _script.main(['--journal-dir', str(data_dir)])

        assert excinfo.value.code == 2

    def test_a_journal_dir_without_a_journal_is_a_usage_error(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            _script.main(['--journal-dir', str(tmp_path / 'nowhere'), '--intent-id', 'x'])

        assert excinfo.value.code == 2
        assert not (tmp_path / 'nowhere').exists()


def test_the_replayed_metadata_round_trips_through_json(tmp_path):
    """The row stores metadata as JSON text; the tool takes a dict."""
    metadata = {'category': 'procedural_knowledge', 'n': [1, 2]}
    row = {
        'content': 'c',
        'project_id': 'dark_factory',
        'category': 'procedural_knowledge',
        'metadata': json.dumps(metadata),
        'agent_id': None,
        'session_id': None,
    }

    assert _script.replay_arguments(row) == {
        'content': 'c',
        'project_id': 'dark_factory',
        'category': 'procedural_knowledge',
        'metadata': metadata,
    }
