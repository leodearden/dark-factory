"""Boundary row 1, minus the hold, for metadata.followup_items at the submit boundary (task 8414).

plans/follow-up-filing-shape-prd.md §4.1 and §3 decisions 1 and 3: a bundle's R4 keys are derived
server-side on both creation paths, and a bundle breaking a create invariant is refused before a
ticket or a row is written. Each row drives the public MCP tools against a real SQLite store, ticket
store and interceptor.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from typing import Any

import pytest
from _submit_stack import Stack
from _submit_stack import curator_creates as curator_creates
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack
from shared.followup_items import (
    FOLLOWUP_ITEMS_METADATA_KEY,
    FOLLOWUP_ITEMS_REFUSED_ERROR,
    MAX_FOLLOWUP_ITEMS,
    UNPARENTED_ORIGIN,
    FollowupItem,
    item_key,
    with_derived_bundle_keys,
)

from fused_memory.middleware.curator_types import CandidateTask, PreparedCandidate
from fused_memory.middleware.task_curator import TaskCurator

_HEX16 = re.compile(r'[0-9a-f]{16}')


_AREA_FILES = ['src/app/a.py', 'src/app/b.py', 'src/app/c.py']
_FIRST = {'summary': 'The retry loop swallows the timeout', 'files': ['src/app/a.py']}
_SECOND = {'summary': 'The cache key ignores the tenant', 'files': ['src/app/c.py', 'src/app/b.py']}
_FORGED_IDENTITY = {'key': 'forged0000000000', 'origin_task': '1', 'origin_escalation_id': 'esc-forged'}


def _bundle(*items: dict[str, Any], **top: Any) -> dict[str, Any]:
    return {
        'source': 'agent-followup',
        'spawn_context': 'agent_followup',
        'spawned_from': '8414',
        'files': list(_AREA_FILES),
        FOLLOWUP_ITEMS_METADATA_KEY: list(items),
        **top,
    }


# (a) and (b): an implementer-shaped bundle lands as one task with server-derived keys.


@pytest.mark.asyncio
async def test_an_implementer_bundle_lands_with_server_derived_keys(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    task_id = await stack.file_normal(
        title='Follow-ups in src/app', description='Two observations in one area.',
        metadata=_bundle(
            {**_FIRST, **_FORGED_IDENTITY}, {**_SECOND, **_FORGED_IDENTITY},
            escalation_id='esc-forged', suggestion_hash='forged',
        ),
    )

    metadata = await stack.metadata(task_id)
    stored = metadata[FOLLOWUP_ITEMS_METADATA_KEY]
    assert len(stored) == 2
    for item, sent in zip(stored, (_FIRST, _SECOND), strict=True):
        assert item['origin_task'] == '8414'
        assert item['key'] == item_key(origin_task='8414', files=sent['files'], summary=sent['summary'])
    assert metadata['escalation_id'] == 'agent-followup-8414'
    assert metadata['suggestion_hash'] != 'forged'
    assert _HEX16.fullmatch(metadata['suggestion_hash'])
    assert metadata['declared_scope']['basis'] == 'submit'

    assert [bundle.candidate.items for bundle in curator_creates] == [
        tuple(FollowupItem.model_validate(item) for item in stored),
    ]


# (c): server-derived keys make a re-filed bundle an R4 hit, whatever hashes the caller sent.


@pytest.mark.asyncio
async def test_re_filing_the_same_observations_is_an_idempotency_hit(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    first = await stack.file_normal(
        title='Follow-ups in src/app', description='Two observations in one area.',
        metadata=_bundle(_FIRST, _SECOND, suggestion_hash='forged-first'),
    )

    submitted = await stack.call(
        'submit_task', title='A later agent re-files them', description='The same two, reordered.',
        metadata=_bundle(_SECOND, _FIRST, escalation_id='esc-other', suggestion_hash='forged-second'),
    )
    assert 'ticket' in submitted, submitted
    resolved = await stack.call('resolve_ticket', ticket=submitted['ticket'], timeout_seconds=30)

    assert (resolved['status'], resolved['reason'], str(resolved['task_id'])) == (
        'combined', 'idempotency_hit', first,
    )
    assert await stack.task_count() == 1


# (d): the planning-mode path derives the same keys on its deferred row.


@pytest.mark.asyncio
async def test_a_planning_mode_bundle_carries_the_derived_keys(stack: Stack) -> None:
    task_id = await stack.file_planned(
        title='Planned follow-ups in src/app', description='Two observations in one area.',
        metadata=_bundle(_FIRST, _SECOND, escalation_id='esc-forged', suggestion_hash='forged'),
    )

    task = await stack.call('get_task', id=task_id)
    expected = with_derived_bundle_keys(_bundle(_FIRST, _SECOND))
    assert task['status'] == 'deferred'
    for key in (FOLLOWUP_ITEMS_METADATA_KEY, 'escalation_id', 'suggestion_hash'):
        assert task['metadata'][key] == expected[key]


# (e): an unparented bundle has the unparented origin and no top-level escalation_id.


@pytest.mark.asyncio
async def test_an_unparented_bundle_has_no_escalation_id(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    metadata = _bundle(_FIRST, escalation_id='esc-forged')
    metadata.pop('spawned_from')

    task_id = await stack.file_normal(
        title='An unparented follow-up', description='Filed by no task.', metadata=metadata,
    )

    stored = await stack.metadata(task_id)
    assert [item['origin_task'] for item in stored[FOLLOWUP_ITEMS_METADATA_KEY]] == [UNPARENTED_ORIGIN]
    assert 'escalation_id' not in stored


# (f): a bundle breaking a create invariant is refused synchronously, on both paths.

_REFUSALS = [
    pytest.param(
        [{'summary': f'observation {index}', 'files': ['src/app/a.py']} for index in range(MAX_FOLLOWUP_ITEMS + 1)],
        'item_count', None, id='too-many-items',
    ),
    pytest.param(
        [{'summary': 'one', 'files': ['src/app/a.py', 'src/app/undeclared.py']}],
        'undeclared_files', None, id='undeclared-file',
    ),
    pytest.param([_FIRST, dict(_FIRST)], 'duplicate_key', None, id='identical-items'),
    pytest.param([_FIRST, {**_SECOND, 'title': 'not a field'}], 'invalid_item', 1, id='unknown-field'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('planning_mode', [False, True])
@pytest.mark.parametrize('as_json', [False, True], ids=['object', 'json-string'])
@pytest.mark.parametrize(('items', 'reason', 'index'), _REFUSALS)
async def test_a_bundle_breaking_a_create_invariant_is_refused(
    stack: Stack, planning_mode: bool, as_json: bool,
    items: list[dict[str, Any]], reason: str, index: int | None,
) -> None:
    metadata = _bundle(*items)
    before = await stack.task_count()

    result = await stack.call(
        'submit_task', title='A refused bundle', description='Breaks a create invariant.',
        planning_mode=planning_mode, metadata=json.dumps(metadata) if as_json else metadata,
    )

    assert result['success'] is False, result
    assert (result['error'], result['error_type'], result['path']) == (
        FOLLOWUP_ITEMS_REFUSED_ERROR, 'FollowupItemsRefusedError', f'metadata.{FOLLOWUP_ITEMS_METADATA_KEY}',
    ), result
    assert (result['reason'], result['index']) == (reason, index), result
    assert result['detail'], result
    assert 'ticket' not in result
    assert await stack.task_count() == before


# (g): a submission without followup_items keeps its caller-supplied R4 keys.


@pytest.mark.asyncio
async def test_a_steward_shaped_submission_keeps_its_keys(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    task_id = await stack.file_normal(
        title='A steward follow-up', description='Filed from an escalation.',
        metadata={'escalation_id': 'esc-1', 'suggestion_hash': 'h', 'files': ['src/app/a.py']},
    )

    stored = await stack.metadata(task_id)
    assert (stored['escalation_id'], stored['suggestion_hash']) == ('esc-1', 'h')
    assert FOLLOWUP_ITEMS_METADATA_KEY not in stored


# (h): the registration reaches the fused-memory process, so a bundle is never censused.


@pytest.mark.asyncio
async def test_filing_a_bundle_emits_no_census_line_for_it(
    stack: Stack, curator_creates: list[PreparedCandidate], caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        caplog.clear()
        await stack.file_normal(
            title='Follow-ups in src/app', description='Two observations in one area.',
            metadata=_bundle(_FIRST, _SECOND),
        )

    census = [
        record.getMessage() for record in caplog.records
        if 'task_metadata.schema_warning' in record.getMessage()
        and FOLLOWUP_ITEMS_METADATA_KEY in record.getMessage()
    ]
    assert census == []


# (i): indexing a stored row whose bundle is malformed never aborts; the candidate loses its items.


@pytest.mark.asyncio
async def test_indexing_a_malformed_stored_bundle_records_the_candidate_without_items(
    stack: Stack, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    recorded: list[tuple[str, CandidateTask]] = []

    async def _record(self: TaskCurator, task_id: str, candidate: CandidateTask, project_id: str) -> None:
        recorded.append((task_id, candidate))

    monkeypatch.setattr(TaskCurator, 'record_task', _record)
    keyless = {'summary': 'An element without a key', 'files': ['src/app/a.py'], 'origin_task': '8414'}
    row = {'id': '7', 'title': 'A stored bundle', 'metadata': {FOLLOWUP_ITEMS_METADATA_KEY: [keyless]}}

    with caplog.at_level(logging.WARNING, logger='fused_memory.middleware.task_interceptor'):
        caplog.clear()
        await stack.interceptor.index_committed_tasks([row], stack.root)

    assert [(task_id, candidate.title, candidate.items) for task_id, candidate in recorded] == [
        ('7', 'A stored bundle', ()),
    ]
    census = [
        record.getMessage() for record in caplog.records
        if 'task_metadata.schema_warning' in record.getMessage()
        and FOLLOWUP_ITEMS_METADATA_KEY in record.getMessage()
    ]
    assert len(census) == 1, census


def test_the_interceptor_import_chain_registers_the_slice() -> None:
    """``-I`` isolates the interceptor's own import closure from what this pytest process loaded."""
    probe = (
        'import fused_memory.middleware.task_interceptor\n'
        'from shared.task_metadata import is_known_metadata_key\n'
        f'print(is_known_metadata_key({FOLLOWUP_ITEMS_METADATA_KEY!r}))\n'
    )
    result = subprocess.run(
        [sys.executable, '-I', '-c', probe], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'True'
