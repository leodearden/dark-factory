"""Boundary rows 2, 3 and 3b: R4 at the grain of a follow-up item, through the real submit stack (task 8415).

plans/follow-up-filing-shape-prd.md §4.1: an item whose key a live task already carries is dropped
from a submission; a submission left with no items resolves onto the first owner, and a partial one
is filed with only its kept items. Only live tasks own anything, for the top-level
(escalation_id, suggestion_hash) conjunction too.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from _submit_stack import Stack
from _submit_stack import curator_creates as curator_creates
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack
from shared.followup_items import (
    FOLLOWUP_ITEMS_METADATA_KEY,
    item_key,
    observed_item,
    with_derived_bundle_keys,
)

from fused_memory.middleware.curator_types import PreparedCandidate
from fused_memory.middleware.r4_idempotency import LiveOwner, R4Lookup, check_r4

_INTERCEPTOR_LOGGER = 'fused_memory.middleware.task_interceptor'
_R4_LOOKUP_PAGE = 100
_PARENT = '8414'
_AREA_FILES = ['src/app/a.py', 'src/app/b.py', 'src/app/c.py', 'src/app/d.py']
_I1 = {'summary': 'The retry loop swallows the timeout', 'files': ['src/app/a.py']}
_I2 = {'summary': 'The cache key ignores the tenant', 'files': ['src/app/b.py']}
_I3 = {'summary': 'The lock is never released on error', 'files': ['src/app/c.py']}
_I4 = {'summary': 'The log line leaks the token', 'files': ['src/app/d.py']}
_I5 = {'summary': 'The parser drops trailing commas', 'files': ['src/app/a.py', 'src/app/b.py']}


def _bundle(*items: dict[str, Any], spawned_from: str = _PARENT) -> dict[str, Any]:
    return {
        'source': 'agent-followup',
        'spawn_context': 'agent_followup',
        'spawned_from': spawned_from,
        'files': list(_AREA_FILES),
        FOLLOWUP_ITEMS_METADATA_KEY: list(items),
    }


def _key(item: dict[str, Any], origin: str = _PARENT) -> str:
    return item_key(origin_task=origin, files=item['files'], summary=item['summary'])


def _filing(metadata: dict[str, Any], *, title: str) -> dict[str, Any]:
    """One submission; the title is distinct per filing, since title and files make the candidate key."""
    return {'title': title, 'description': 'Observations in one area.', 'metadata': metadata}


def _idempotency_hit(task_id: str) -> dict[str, Any]:
    return {'status': 'combined', 'task_id': task_id, 'reason': 'idempotency_hit'}


async def _file_bundle(stack: Stack) -> str:
    return await stack.file_normal(**_filing(_bundle(_I1, _I2, _I3), title='Follow-ups in src/app'))


@pytest.mark.asyncio
async def test_an_exhausted_bundle_resolves_onto_the_owner_and_records_the_dropped_key(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    owner = await _file_bundle(stack)
    tasks_before, handed_before = await stack.task_count(), len(curator_creates)

    ticket, resolved = await stack.submit(**_filing(_bundle(_I1), title='The retry loop'))

    assert {key: resolved.get(key) for key in ('status', 'task_id', 'reason')} == _idempotency_hit(owner)
    assert await stack.task_count() == tasks_before
    assert len(curator_creates) == handed_before
    outcome = await stack.ticket_outcome(ticket)
    assert outcome is not None
    assert outcome['idempotency_dropped_items'] == [{'key': _key(_I1), 'task_id': owner}]


@pytest.mark.asyncio
async def test_an_item_owned_past_the_first_lookup_page_still_resolves_onto_its_owner(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    for n in range(_R4_LOOKUP_PAGE):
        filler = {'summary': f'An unrelated observation {n}', 'files': ['src/app/a.py']}
        await stack.seed(f'Filler bundle {n}', with_derived_bundle_keys(_bundle(filler)))
    owner = await stack.seed('Follow-ups in src/app', with_derived_bundle_keys(_bundle(_I1, _I2)))

    _ticket, resolved = await stack.submit(**_filing(_bundle(_I1), title='The retry loop'))

    assert {key: resolved.get(key) for key in ('status', 'task_id', 'reason')} == _idempotency_hit(owner)


@pytest.mark.asyncio
async def test_a_partial_bundle_is_filed_with_only_its_kept_items(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    owner = await _file_bundle(stack)

    ticket, resolved = await stack.submit(**_filing(_bundle(_I1, _I4), title='Retry and token'))

    assert resolved.get('status') == 'created', resolved
    filed = str(resolved['task_id'])
    metadata = await stack.metadata(filed)
    assert [item['key'] for item in metadata[FOLLOWUP_ITEMS_METADATA_KEY]] == [_key(_I4)]
    assert metadata['suggestion_hash'] == with_derived_bundle_keys(_bundle(_I4))['suggestion_hash']
    outcome = await stack.ticket_outcome(ticket)
    assert outcome is not None
    assert outcome['idempotency_dropped_items'] == [{'key': _key(_I1), 'task_id': owner}]
    assert [item.key for item in curator_creates[-1].candidate.items] == [_key(_I4)]


@pytest.mark.asyncio
async def test_the_same_observation_from_another_origin_is_new_work(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    owner = await _file_bundle(stack)

    _ticket, resolved = await stack.submit(
        **_filing(_bundle(_I1, spawned_from=owner), title='The retry loop, seen by the owner'),
    )

    assert resolved.get('status') == 'created', resolved


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['done', 'cancelled'])
async def test_a_finished_owner_owns_no_item(
    stack: Stack, curator_creates: list[PreparedCandidate], status: str,
) -> None:
    owner = await _file_bundle(stack)
    await stack.backend.set_task_status(owner, status, stack.root)

    _ticket, resolved = await stack.submit(**_filing(_bundle(_I1), title='The retry loop again'))

    assert resolved.get('status') == 'created', resolved


@pytest.mark.asyncio
async def test_refiling_a_done_bundle_exactly_creates(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    owner = await _file_bundle(stack)
    await stack.backend.set_task_status(owner, 'done', stack.root)

    _ticket, resolved = await stack.submit(**_filing(_bundle(_I1, _I2, _I3), title='Follow-ups again'))

    assert resolved.get('status') == 'created', resolved


@pytest.mark.asyncio
async def test_an_item_keyed_like_a_live_legacy_single_is_owned_by_it(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    legacy = await stack.seed('A legacy follow-up', {
        'escalation_id': 'esc-legacy', 'suggestion_hash': _key(_I5), 'files': _I5['files'],
    })

    _ticket, resolved = await stack.submit(**_filing(_bundle(_I5), title='The parser and commas'))

    assert {key: resolved.get(key) for key in ('status', 'task_id', 'reason')} == _idempotency_hit(legacy)


@pytest.mark.asyncio
async def test_a_legacy_single_keyed_like_a_live_item_is_owned_by_its_bundle(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    item = observed_item(summary=_I5['summary'], files=_I5['files'], origin_task=_PARENT)
    bundle = await stack.seed('A bundle', {
        'files': list(_AREA_FILES), FOLLOWUP_ITEMS_METADATA_KEY: [item.model_dump(mode='json')],
    })

    _ticket, resolved = await stack.submit(**_filing(
        {'escalation_id': 'esc-x', 'suggestion_hash': item.key, 'files': _I5['files']},
        title='The parser drops trailing commas',
    ))

    assert {key: resolved.get(key) for key in ('status', 'task_id', 'reason')} == _idempotency_hit(bundle)


@pytest.mark.asyncio
async def test_a_legacy_hash_under_another_escalation_still_creates(
    stack: Stack, curator_creates: list[PreparedCandidate],
) -> None:
    await stack.seed('A legacy follow-up', {
        'escalation_id': 'esc-a', 'suggestion_hash': 'a1b2c3d4e5f60718', 'files': ['src/app/a.py'],
    })

    _ticket, resolved = await stack.submit(**_filing(
        {'escalation_id': 'esc-b', 'suggestion_hash': 'a1b2c3d4e5f60718', 'files': ['src/app/a.py']},
        title='Another escalation, the same hash',
    ))

    assert resolved.get('status') == 'created', resolved


@pytest.mark.asyncio
async def test_a_failed_item_scan_files_every_item(
    stack: Stack, curator_creates: list[PreparedCandidate],
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    await _file_bundle(stack)
    lookup = stack.backend.find_tasks_by_metadata

    async def _items_scan_fails(project_root: str, key: str, *args: Any, **kwargs: Any) -> Any:
        if key == FOLLOWUP_ITEMS_METADATA_KEY:
            raise RuntimeError('followup_items scan unavailable')
        return await lookup(project_root, key, *args, **kwargs)

    monkeypatch.setattr(stack.backend, 'find_tasks_by_metadata', _items_scan_fails)
    caplog.set_level(logging.WARNING, logger=_INTERCEPTOR_LOGGER)

    _ticket, resolved = await stack.submit(**_filing(_bundle(_I1, _I4), title='Retry and token'))

    assert resolved.get('status') == 'created', resolved
    metadata = await stack.metadata(str(resolved['task_id']))
    assert [item['key'] for item in metadata[FOLLOWUP_ITEMS_METADATA_KEY]] == [_key(_I1), _key(_I4)]
    assert any(
        record.name == _INTERCEPTOR_LOGGER and record.levelno == logging.WARNING
        and record.getMessage().startswith('r4:')
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_one_lookup_scans_the_live_items_once_for_every_check_it_serves(
    stack: Stack, monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await stack.seed('Follow-ups in src/app', with_derived_bundle_keys(_bundle(_I1, _I2)))
    lookup_by_metadata = stack.backend.find_tasks_by_metadata
    item_scans: list[str] = []

    async def _counting_item_scans(project_root: str, key: str, *args: Any, **kwargs: Any) -> Any:
        if key == FOLLOWUP_ITEMS_METADATA_KEY:
            item_scans.append(project_root)
        return await lookup_by_metadata(project_root, key, *args, **kwargs)

    monkeypatch.setattr(stack.backend, 'find_tasks_by_metadata', _counting_item_scans)
    lookup = R4Lookup(stack.backend)

    exhausted = await check_r4(lookup, stack.root, with_derived_bundle_keys(_bundle(_I1)))
    partial = await check_r4(lookup, stack.root, with_derived_bundle_keys(_bundle(_I2, _I4)))

    owned = LiveOwner(task_id=int(owner), title='Follow-ups in src/app')
    assert exhausted.hit == owned
    assert [(dropped.key, dropped.owner) for dropped in partial.dropped] == [(_key(_I2), owned)]
    assert [item.key for item in partial.kept] == [_key(_I4)]
    assert item_scans == [stack.root]


@pytest.mark.asyncio
async def test_a_lookup_whose_scan_raised_scans_again_on_its_next_check(
    stack: Stack, monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await stack.seed('Follow-ups in src/app', with_derived_bundle_keys(_bundle(_I1, _I2)))
    lookup_by_metadata = stack.backend.find_tasks_by_metadata
    failures = ['followup_items scan unavailable']

    async def _first_item_scan_fails(project_root: str, key: str, *args: Any, **kwargs: Any) -> Any:
        if key == FOLLOWUP_ITEMS_METADATA_KEY and failures:
            raise RuntimeError(failures.pop())
        return await lookup_by_metadata(project_root, key, *args, **kwargs)

    monkeypatch.setattr(stack.backend, 'find_tasks_by_metadata', _first_item_scan_fails)
    lookup = R4Lookup(stack.backend)
    refiled = with_derived_bundle_keys(_bundle(_I1))

    with pytest.raises(RuntimeError, match='followup_items scan unavailable'):
        await check_r4(lookup, stack.root, refiled)
    verdict = await check_r4(lookup, stack.root, refiled)

    assert verdict.hit == LiveOwner(task_id=int(owner), title='Follow-ups in src/app')
