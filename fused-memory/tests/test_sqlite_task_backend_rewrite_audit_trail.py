"""SqliteTaskBackend.rewrite_audit_trail — the audit-trail rotation's privileged writer (task 5771).

A whole-blob replace that keeps the machine-authored wait anchor, passes a
stored ``done_provenance`` through verbatim only, never moves ``files`` and
never repairs a corrupt row.  Driven against a real backend on a temp store.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_backend_errors import (
    DoneProvenanceWriteAuthorityError,
    LeakedEnvelopeMarkupError,
    TaskmasterError,
)
from fused_memory.config.schema import TaskmasterConfig

SEED_METADATA = {
    'files': ['docs/task-authoring.md'],
    'task_kind': 'deterministic',
    'x_relay_2026_09_20': {'relay': 'first'},
    'x_relay_2026_09_21': {'relay': 'second'},
}
PROVENANCE = {'kind': 'merged', 'commit': 'a' * 40}
LEAK = 'prose\n' + closer_for('description') + '\n' + CANONICAL_OPENER_PREFIX + '"priority">low'


@pytest_asyncio.fixture
async def seeded(tmp_path):
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await backend.start()
    root = str(tmp_path)
    await backend.add_task(
        project_root=root,
        title='Gate: decide the thing',
        description='old description',
        details='Ruling: none yet.',
        metadata=json.dumps(SEED_METADATA),
    )
    try:
        yield backend, root
    finally:
        await backend.close()


def _db(root: str) -> Path:
    return Path(root) / '.taskmaster' / 'tasks' / 'tasks.db'


def _stored_row(root: str) -> dict[str, Any]:
    conn = sqlite3.connect(f'file:{_db(root)}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return dict(conn.execute('SELECT * FROM tasks WHERE id = 1').fetchone())
    finally:
        conn.close()


def _set_stored_metadata(root: str, raw: str) -> None:
    conn = sqlite3.connect(_db(root))
    try:
        conn.execute('UPDATE tasks SET metadata = ? WHERE id = 1', (raw,))
        conn.commit()
    finally:
        conn.close()


def _rotated_blob(stored: dict[str, Any], **changes: Any) -> dict[str, Any]:
    blob = {
        key: value
        for key, value in stored.items()
        if not key.startswith('x_relay_2026') and key not in ('pending_since',)
    }
    blob['x_relay_history'] = [{'source_key': 'x_relay_2026_09_21', 'value': 'second'}]
    blob.update(changes)
    return blob


@pytest.mark.asyncio
async def test_rewrite_replaces_blob_and_description_but_keeps_the_wait_anchor(seeded):
    backend, root = seeded
    stored = json.loads(_stored_row(root)['metadata'])
    _set_stored_metadata(root, json.dumps({**stored, 'pending_since_backfilled': True}))
    before = await backend.get_task('1', root)
    assert before['metadata']['pending_since']
    await asyncio.sleep(0.005)

    result = await backend.rewrite_audit_trail(
        '1', root, description='new', metadata=_rotated_blob(before['metadata'])
    )

    after = await backend.get_task('1', root)
    assert after['description'] == 'new'
    assert 'x_relay_2026_09_20' not in after['metadata']
    assert after['metadata']['x_relay_history'][0]['source_key'] == 'x_relay_2026_09_21'
    for key in ('pending_since', 'pending_since_backfilled'):
        assert after['metadata'][key] == before['metadata'][key]
    for column in ('title', 'details', 'status', 'candidate_key'):
        assert after[column] == before[column]
    assert after['updatedAt'] > before['updatedAt']
    assert result['updated'] is True
    assert result['id'] == '1'
    assert result['updated_task'] == after


@pytest.mark.asyncio
async def test_description_none_leaves_the_description_column_alone(seeded):
    backend, root = seeded
    before = await backend.get_task('1', root)
    await backend.rewrite_audit_trail(
        '1', root, description=None, metadata=_rotated_blob(before['metadata'])
    )
    after = await backend.get_task('1', root)
    assert after['description'] == 'old description'
    assert 'x_relay_history' in after['metadata']


@pytest.mark.asyncio
async def test_stored_done_provenance_passes_through_verbatim(seeded):
    backend, root = seeded
    await backend.stamp_audit_metadata('1', root, {'done_provenance': PROVENANCE})
    before = await backend.get_task('1', root)
    await backend.rewrite_audit_trail(
        '1', root, description=None, metadata=_rotated_blob(before['metadata'])
    )
    after = await backend.get_task('1', root)
    assert after['metadata']['done_provenance'] == PROVENANCE
    assert 'x_relay_history' in after['metadata']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'provenance_change',
    [
        {'done_provenance': {**PROVENANCE, 'commit': 'b' * 40}},
        {'done_provenance': None},
    ],
    ids=['changed', 'omitted'],
)
async def test_changing_or_dropping_done_provenance_is_refused(seeded, provenance_change):
    backend, root = seeded
    await backend.stamp_audit_metadata('1', root, {'done_provenance': PROVENANCE})
    before_row = _stored_row(root)
    blob = _rotated_blob(json.loads(before_row['metadata']), **provenance_change)
    if blob['done_provenance'] is None:
        del blob['done_provenance']
    with pytest.raises(DoneProvenanceWriteAuthorityError):
        await backend.rewrite_audit_trail('1', root, description='new', metadata=blob)
    assert _stored_row(root) == before_row


@pytest.mark.asyncio
async def test_adding_done_provenance_is_refused(seeded):
    backend, root = seeded
    before_row = _stored_row(root)
    blob = _rotated_blob(json.loads(before_row['metadata']), done_provenance=PROVENANCE)
    with pytest.raises(DoneProvenanceWriteAuthorityError):
        await backend.rewrite_audit_trail('1', root, description='new', metadata=blob)
    assert _stored_row(root) == before_row


@pytest.mark.asyncio
async def test_changing_files_is_refused(seeded):
    backend, root = seeded
    before_row = _stored_row(root)
    blob = _rotated_blob(json.loads(before_row['metadata']), files=['other.py'])
    with pytest.raises(ValueError):
        await backend.rewrite_audit_trail('1', root, description='new', metadata=blob)
    assert _stored_row(root) == before_row


@pytest.mark.asyncio
async def test_corrupt_stored_blob_is_refused_not_repaired(seeded):
    backend, root = seeded
    _set_stored_metadata(root, '{not json')
    before_row = _stored_row(root)
    with pytest.raises((TaskmasterError, ValueError)):
        await backend.rewrite_audit_trail(
            '1', root, description='new', metadata={'files': SEED_METADATA['files']}
        )
    assert _stored_row(root) == before_row


@pytest.mark.asyncio
async def test_leaked_envelope_markup_in_the_description_is_refused(seeded):
    backend, root = seeded
    before_row = _stored_row(root)
    with pytest.raises(LeakedEnvelopeMarkupError):
        await backend.rewrite_audit_trail(
            '1', root, description=LEAK, metadata=_rotated_blob(json.loads(before_row['metadata']))
        )
    assert _stored_row(root) == before_row


@pytest.mark.asyncio
async def test_caller_supplied_pending_since_is_ignored(seeded):
    backend, root = seeded
    before = await backend.get_task('1', root)
    forged = _rotated_blob(before['metadata'], pending_since='2000-01-01T00:00:00.000Z')
    await backend.rewrite_audit_trail('1', root, description=None, metadata=forged)
    after = await backend.get_task('1', root)
    assert after['metadata']['pending_since'] == before['metadata']['pending_since']
