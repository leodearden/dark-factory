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
from collections.abc import Mapping
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

# The 4358 specimen followed by residue the tail parser cannot place, so no
# boundary between the caller's text and the swallowed arguments is provable.
UNPLACEABLE_RESIDUE_DESCRIPTION = (
    TASK_4358_DESCRIPTION + closer_for('parameter')
    + '\nand then prose the tail parser cannot place'
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


@pytest_asyncio.fixture
async def backend(tmp_path):
    b = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await b.start()
    yield b
    await b.close()


@pytest.fixture
def project_root(tmp_path) -> str:
    return str(tmp_path / 'proj')


def _db_path(project_root: str) -> Path:
    return Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'


def _stored_rows(project_root: str) -> list[dict]:
    conn = sqlite3.connect(f'file:{_db_path(project_root)}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute('SELECT * FROM tasks ORDER BY id')]
    finally:
        conn.close()


def _corrupt_description_as_a_pre_gate_writer_did(project_root: str, task_id: str) -> None:
    conn = sqlite3.connect(_db_path(project_root))
    try:
        with conn:
            conn.execute(
                'UPDATE tasks SET description = ? WHERE id = ?',
                (TASK_4358_DESCRIPTION, int(task_id)),
            )
    finally:
        conn.close()


async def _seed_clean_task(backend, project_root: str) -> str:
    result = await backend.add_task(
        project_root, title='clean title', description='clean description',
        details='clean details', priority='low',
    )
    return result['id']


def _description_swallowing(swallowed: Mapping[str, str]) -> str:
    """Task 4358's prose, then a fragment that swallowed *swallowed*, in order."""
    return TASK_4358_PROSE + closer_for('description') + closer_for('parameter').join(
        '\n' + CANONICAL_OPENER_PREFIX + f'"{name}">{value}'
        for name, value in swallowed.items()
    )


def _skip_unless_the_sink_writes(sink, column: str) -> None:
    if column not in inspect.signature(sink).parameters:
        pytest.skip(f'{sink.__name__} takes no {column!r} argument, so no write can carry one')


def _add_task_text(column: str, value: str) -> dict[str, str]:
    text = {'title': 'clean title', 'description': 'clean description'}
    text[column] = value
    return text


def test_specimens_sit_on_the_intended_side_of_the_precise_detector():
    assert detect_leak(TASK_4358_DESCRIPTION) == TASK_4358_FRAGMENT
    assert detect_leak(PROSE_MENTION_OF_THE_LEAK) is None
    assert detect(PROSE_MENTION_OF_THE_LEAK) is not None
    assert detect_leak(REMEDIATION_METADATA) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('column', SCANNED_COLUMNS)
async def test_add_task_refuses_leaked_text_and_inserts_nothing(
    backend, project_root, column,
):
    _skip_unless_the_sink_writes(SqliteTaskBackend.add_task, column)

    with pytest.raises(LeakedEnvelopeMarkupError) as excinfo:
        await backend.add_task(
            project_root, priority=None, **_add_task_text(column, TASK_4358_DESCRIPTION),
        )

    assert isinstance(excinfo.value, TaskmasterError)
    assert excinfo.value.column == column
    assert excinfo.value.fragment == TASK_4358_FRAGMENT
    assert _stored_rows(project_root) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('column', SCANNED_COLUMNS)
async def test_add_task_stores_prose_quoting_the_leak_verbatim(
    backend, project_root, column,
):
    _skip_unless_the_sink_writes(SqliteTaskBackend.add_task, column)

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


@pytest.mark.asyncio
@pytest.mark.parametrize('column', SCANNED_COLUMNS)
async def test_update_task_refuses_leaked_text_and_rolls_back_the_whole_write(
    backend, project_root, column,
):
    _skip_unless_the_sink_writes(SqliteTaskBackend.update_task, column)
    task_id = await _seed_clean_task(backend, project_root)
    before = _stored_rows(project_root)

    with pytest.raises(LeakedEnvelopeMarkupError) as excinfo:
        await backend.update_task(
            task_id, project_root, priority='high',
            metadata=json.dumps({'note': 'written alongside the leak'}),
            **{column: TASK_4358_DESCRIPTION},
        )

    assert excinfo.value.column == column
    assert excinfo.value.fragment == TASK_4358_FRAGMENT
    assert _stored_rows(project_root) == before


@pytest.mark.asyncio
async def test_update_task_refuses_a_leaked_prompt_fed_into_details(backend, project_root):
    task_id = await _seed_clean_task(backend, project_root)
    before = _stored_rows(project_root)

    with pytest.raises(LeakedEnvelopeMarkupError) as excinfo:
        await backend.update_task(task_id, project_root, prompt=TASK_4358_DESCRIPTION)

    assert excinfo.value.column == 'details'
    assert _stored_rows(project_root) == before


@pytest.mark.asyncio
async def test_update_task_refuses_appending_a_leaked_tail_to_clean_details(
    backend, project_root,
):
    task_id = await _seed_clean_task(backend, project_root)
    before = _stored_rows(project_root)

    with pytest.raises(LeakedEnvelopeMarkupError) as excinfo:
        await backend.update_task(
            task_id, project_root, details=TASK_4358_DESCRIPTION, append=True,
        )

    assert excinfo.value.column == 'details'
    assert excinfo.value.fragment == TASK_4358_FRAGMENT
    assert _stored_rows(project_root) == before


@pytest.mark.asyncio
async def test_update_task_judges_an_append_on_the_value_that_would_be_stored(
    backend, project_root,
):
    """Neither half leaks alone; the concatenation the append would persist does."""
    stray_closer_at_end = 'details quoting a stray ' + closer_for('details')
    swallowed_opener = CANONICAL_OPENER_PREFIX + '"priority">low'
    assert detect_leak(stray_closer_at_end) is None
    assert detect_leak(swallowed_opener) is None

    task_id = await _seed_clean_task(backend, project_root)
    await backend.update_task(task_id, project_root, details=stray_closer_at_end)
    before = _stored_rows(project_root)

    with pytest.raises(LeakedEnvelopeMarkupError) as excinfo:
        await backend.update_task(
            task_id, project_root, details=swallowed_opener, append=True,
        )

    assert excinfo.value.column == 'details'
    assert excinfo.value.fragment == (
        closer_for('details') + '\n\n' + swallowed_opener
    )
    assert _stored_rows(project_root) == before


@pytest.mark.asyncio
async def test_update_task_leaves_a_pre_gate_corrupt_row_writable(backend, project_root):
    task_id = await _seed_clean_task(backend, project_root)
    _corrupt_description_as_a_pre_gate_writer_did(project_root, task_id)

    await backend.update_task(
        task_id, project_root, metadata=REMEDIATION_METADATA, metadata_mode='merge',
    )
    await backend.update_task(task_id, project_root, priority='low')
    await backend.update_task(task_id, project_root, title='retitled')
    await backend.set_task_status(task_id, 'in-progress', project_root)

    [row] = _stored_rows(project_root)
    assert row['description'] == TASK_4358_DESCRIPTION
    assert row['title'] == 'retitled'
    assert row['status'] == 'in-progress'
    assert 'stage2_description_corruption_fix' in json.loads(row['metadata'])


@pytest.mark.asyncio
async def test_update_task_lets_the_remediation_of_a_corrupt_description_land(
    backend, project_root,
):
    task_id = await _seed_clean_task(backend, project_root)
    _corrupt_description_as_a_pre_gate_writer_did(project_root, task_id)

    await backend.update_task(
        task_id, project_root, description=TASK_4358_PROSE, priority='low',
    )

    [row] = _stored_rows(project_root)
    assert (row['description'], row['priority']) == (TASK_4358_PROSE, 'low')


@pytest.mark.asyncio
async def test_add_task_refusal_names_the_priority_main_would_default_to_medium(
    backend, project_root,
):
    with pytest.raises(LeakedEnvelopeMarkupError) as excinfo:
        await backend.add_task(
            project_root, title='clean title', description=TASK_4358_DESCRIPTION,
        )

    assert isinstance(excinfo.value.recovered, Mapping)
    assert excinfo.value.recovered == {'priority': 'low'}
    assert excinfo.value.clean_value == TASK_4358_PROSE


@pytest.mark.asyncio
async def test_update_task_refusal_names_the_swallowed_priority(backend, project_root):
    task_id = await _seed_clean_task(backend, project_root)

    with pytest.raises(LeakedEnvelopeMarkupError) as excinfo:
        await backend.update_task(
            task_id, project_root, description=TASK_4358_DESCRIPTION,
        )

    assert excinfo.value.recovered == {'priority': 'low'}
    assert excinfo.value.clean_value == TASK_4358_PROSE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'swallowed',
    [{'priority': 'low', 'metadata': '{"a": 1}'}, {'priority': 'low', 'tag': 'master'}],
    ids=['metadata-the-pending-stamp-fills-in', 'tag-the-sink-defaults'],
)
async def test_add_task_refusal_recovers_arguments_the_sink_would_fill_in(
    backend, project_root, swallowed,
):
    with pytest.raises(LeakedEnvelopeMarkupError) as excinfo:
        await backend.add_task(
            project_root, title='clean title',
            description=_description_swallowing(swallowed),
        )

    assert excinfo.value.recovered == swallowed
    assert excinfo.value.clean_value == TASK_4358_PROSE
    assert _stored_rows(project_root) == []


@pytest.mark.asyncio
async def test_update_task_refusal_recovers_a_tag_the_sink_would_default(
    backend, project_root,
):
    task_id = await _seed_clean_task(backend, project_root)
    swallowed = {'priority': 'low', 'tag': 'master'}

    with pytest.raises(LeakedEnvelopeMarkupError) as excinfo:
        await backend.update_task(
            task_id, project_root, description=_description_swallowing(swallowed),
        )

    assert excinfo.value.recovered == swallowed
    assert excinfo.value.clean_value == TASK_4358_PROSE


@pytest.mark.asyncio
async def test_refusal_without_a_provable_boundary_recovers_nothing_rather_than_guessing(
    backend, project_root,
):
    assert detect_leak(UNPLACEABLE_RESIDUE_DESCRIPTION) is not None

    with pytest.raises(LeakedEnvelopeMarkupError) as excinfo:
        await backend.add_task(
            project_root, title='clean title',
            description=UNPLACEABLE_RESIDUE_DESCRIPTION,
        )

    assert excinfo.value.column == 'description'
    assert excinfo.value.recovered == {}
    assert excinfo.value.clean_value is None
    assert _stored_rows(project_root) == []
