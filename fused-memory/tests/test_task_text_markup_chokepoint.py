"""The task-store chokepoint refuses task text carrying a leaked tool-call envelope.

``SqliteTaskBackend.add_task`` and ``SqliteTaskBackend.update_task`` hold the
only INSERT INTO tasks and the only UPDATE that can set task text, so every
writer, whether MCP-dispatched or internal, passes through them (task 4419).
These tests drive the backend directly with no MCP call in the loop, and read
the stored rows back from ``<project_root>/.taskmaster/tasks/tasks.db``.

Every specimen is composed from ``shared.toolcall_markup``'s spellings rather
than written raw: a raw envelope literal in this file would truncate the
authoring agent's own tool call (see the Sentinel-literal hazard in
``shared/src/shared/toolcall_markup.py``).
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import pytest
import pytest_asyncio
from shared.toolcall_markup import (
    CANONICAL_OPENER_PREFIX,
    INVOKE_CLOSER,
    closer_for,
    detect,
)

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_backend_errors import (
    LeakedEnvelopeMarkupError,
    TaskmasterError,
)
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.utils.toolcall_xml_leak import SCANNED_COLUMNS, detect_leak

# Task 4358 (an agent-followup filed from task 3842): the description closer
# swallowed the priority argument, so main stored priority='medium' where the
# caller had asked for 'low'.
TASK_4358_FRAGMENT = (
    closer_for('description') + '\n' + CANONICAL_OPENER_PREFIX + '"priority">low'
)
TASK_4358_PROSE = 'real description prose\n'
TASK_4358_DESCRIPTION = TASK_4358_PROSE + TASK_4358_FRAGMENT

# Tasks 2938/2939: prose QUOTING the leak shape with an escaped backslash-n,
# which is not whitespace, and continuing afterwards. It must be stored verbatim.
PROSE_MENTION_OF_THE_LEAK = (
    "Stage 1 found that task 2865's description had a leaked fragment appended: `"
    + closer_for('description') + '\\n' + CANONICAL_OPENER_PREFIX + '"priority">low`. '
    'Stage 2 verified it via get_task and stripped it via update_task.'
)

ORDINARY_TEXT = 'Add a regression test for the orphaned-commit path.'

# The shape of recon's own remediation record, which quotes the stripped
# fragment inside metadata. JSON escapes both a newline and the opener's
# quotes, and either escape alone would let the blob pass the detector
# whichever columns were scanned. A space and the bare invoke closer survive
# encoding, so this blob does trip the detector and only the column scope can
# let it through.
REMEDIATION_METADATA = json.dumps({
    'stage2_description_corruption_fix': {
        'stripped_fragment': closer_for('description') + ' ' + INVOKE_CLOSER,
        'intended_priority': 'low',
        'stored_priority': 'medium',
    },
})

# test_strategy is scanned, but neither sink accepts it as an argument.
SINK_WRITABLE_COLUMNS = ('title', 'description', 'details')


@pytest_asyncio.fixture
async def backend(tmp_path):
    b = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await b.start()
    yield b
    await b.close()


@pytest.fixture
def project_root(tmp_path) -> str:
    return str(tmp_path / 'proj')


def _stored_rows(project_root: str) -> list[dict]:
    db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(row) for row in conn.execute(
                'SELECT id, title, description, details, test_strategy, '
                'priority, metadata FROM tasks ORDER BY id'
            )
        ]
    finally:
        conn.close()


def _add_task_text(column: str, value: str) -> dict[str, str]:
    text = {'title': 'clean title', 'description': 'clean description'}
    text[column] = value
    return text


def test_specimens_sit_on_the_intended_side_of_the_precise_detector():
    assert detect_leak(TASK_4358_DESCRIPTION) == TASK_4358_FRAGMENT
    assert detect_leak(PROSE_MENTION_OF_THE_LEAK) is None
    assert detect(PROSE_MENTION_OF_THE_LEAK) is not None
    assert detect_leak(REMEDIATION_METADATA) is not None


def test_every_scanned_column_but_test_strategy_is_writable_at_the_sinks():
    add_params = inspect.signature(SqliteTaskBackend.add_task).parameters
    update_params = inspect.signature(SqliteTaskBackend.update_task).parameters

    assert set(SCANNED_COLUMNS) - set(SINK_WRITABLE_COLUMNS) == {'test_strategy'}
    assert set(SINK_WRITABLE_COLUMNS) <= set(add_params)
    assert set(SINK_WRITABLE_COLUMNS) <= set(update_params)
    assert 'test_strategy' not in add_params
    assert 'test_strategy' not in update_params


@pytest.mark.asyncio
@pytest.mark.parametrize('column', SINK_WRITABLE_COLUMNS)
async def test_add_task_refuses_leaked_text_and_inserts_nothing(
    backend, project_root, column,
):
    with pytest.raises(LeakedEnvelopeMarkupError) as excinfo:
        await backend.add_task(
            project_root, priority=None, **_add_task_text(column, TASK_4358_DESCRIPTION),
        )

    assert isinstance(excinfo.value, TaskmasterError)
    assert excinfo.value.column == column
    assert excinfo.value.fragment == TASK_4358_FRAGMENT
    assert _stored_rows(project_root) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('column', SINK_WRITABLE_COLUMNS)
async def test_add_task_stores_prose_quoting_the_leak_verbatim(
    backend, project_root, column,
):
    await backend.add_task(
        project_root, **_add_task_text(column, PROSE_MENTION_OF_THE_LEAK),
    )

    [row] = _stored_rows(project_root)
    assert row[column] == PROSE_MENTION_OF_THE_LEAK


@pytest.mark.asyncio
async def test_add_task_stores_ordinary_text(backend, project_root):
    await backend.add_task(
        project_root, title='Ordinary title', description=ORDINARY_TEXT,
        details=ORDINARY_TEXT, priority='low',
    )

    [row] = _stored_rows(project_root)
    assert (row['title'], row['description'], row['details'], row['priority']) == (
        'Ordinary title', ORDINARY_TEXT, ORDINARY_TEXT, 'low',
    )


@pytest.mark.asyncio
async def test_add_task_stores_metadata_quoting_the_fragment(backend, project_root):
    await backend.add_task(
        project_root, title='Remediated task', description=ORDINARY_TEXT,
        metadata=REMEDIATION_METADATA,
    )

    [row] = _stored_rows(project_root)
    assert json.loads(row['metadata'])['stage2_description_corruption_fix'] == (
        json.loads(REMEDIATION_METADATA)['stage2_description_corruption_fix']
    )
