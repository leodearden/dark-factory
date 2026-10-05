"""``scripts/telemetry_query.py``: the operator entry point to the per-write telemetry query.

The script runs the journal's own ``OPERATOR_TELEMETRY_QUERY`` read-only. These
tests seed a real ``WriteJournal`` and read its rows back through the script's
``main``, the way an operator would.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio
from _fm_helpers import load_script_module

from fused_memory.services.write_journal import WriteJournal

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'telemetry_query.py'

_script = load_script_module(SCRIPT_PATH, mod_name='telemetry_query')


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'telemetry')
    await j.initialize()
    yield j
    await j.close()


def _journal_path(journal: WriteJournal) -> Path:
    return journal.data_dir / 'write_journal.db'


async def _seed(
    journal: WriteJournal,
    *,
    operation: str,
    backend: str,
    duration_ms: float | None,
    result_summary: dict | str | None = None,
) -> None:
    write_op_id = str(uuid.uuid4())
    await journal.log_write_op(write_op_id=write_op_id, operation=operation, project_id='p')
    await journal.log_backend_op(
        write_op_id=write_op_id,
        backend=backend,
        operation=operation,
        result_summary=result_summary,
        duration_ms=duration_ms,
    )


def _printed_rows(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


@pytest.mark.asyncio
async def test_prints_one_json_row_per_backend_op_most_recent_first(journal, capsys):
    since = datetime.now(UTC).isoformat()
    await _seed(journal, operation='add_memory', backend='mem0', duration_ms=3.25)
    await _seed(
        journal,
        operation='add_episode',
        backend='graphiti',
        duration_ms=812.5,
        result_summary={
            'result': 'x',
            'tokens': {
                'input_tokens': 120,
                'output_tokens': 45,
                'total_tokens': 165,
                'llm_calls': 1,
            },
        },
    )

    rc = _script.main(['--journal', str(_journal_path(journal)), '--since', since])

    assert rc == 0
    graphiti, mem0 = _printed_rows(capsys)
    assert (graphiti['operation'], graphiti['duration_ms'], graphiti['total_tokens']) == (
        'add_episode',
        812.5,
        165,
    )
    assert (mem0['operation'], mem0['duration_ms'], mem0['total_tokens']) == (
        'add_memory',
        3.25,
        None,
    )


@pytest.mark.asyncio
async def test_limit_caps_the_rows_printed(journal, capsys):
    since = datetime.now(UTC).isoformat()
    for duration_ms in (1.0, 2.0, 3.0):
        await _seed(journal, operation='add_memory', backend='mem0', duration_ms=duration_ms)

    rc = _script.main(
        ['--journal', str(_journal_path(journal)), '--since', since, '--limit', '2']
    )

    assert rc == 0
    assert [row['duration_ms'] for row in _printed_rows(capsys)] == [3.0, 2.0]


@pytest.mark.asyncio
async def test_a_z_suffixed_since_bound_still_includes_rows_in_its_own_second(
    journal, capsys
):
    """The journal compares ``created_at`` as text in its ``+00:00`` form, where
    ``...:00Z`` would sort after ``...:00.5+00:00`` and silently drop the row."""
    whole_second = datetime.now(UTC).replace(microsecond=0)
    await _seed(journal, operation='add_memory', backend='mem0', duration_ms=1.5)

    rc = _script.main(
        [
            '--journal',
            str(_journal_path(journal)),
            '--since',
            whole_second.strftime('%Y-%m-%dT%H:%M:%SZ'),
        ]
    )

    assert rc == 0
    assert [row['duration_ms'] for row in _printed_rows(capsys)] == [1.5]


def test_a_missing_journal_is_refused_and_never_created(tmp_path, capsys):
    missing = tmp_path / 'absent' / 'write_journal.db'
    missing.parent.mkdir()

    rc = _script.main(['--journal', str(missing)])

    assert rc != 0
    assert not missing.exists()
    assert str(missing) in capsys.readouterr().err
