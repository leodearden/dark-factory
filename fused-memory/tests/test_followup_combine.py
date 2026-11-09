"""Boundary row 12's combine half: an item-preserving curator combine, through the real submit stack (task 8415).

plans/follow-up-filing-shape-prd.md §4.1-§4.3, decisions 1, 7 and 10: a combine folds the candidate's
items into the target's by the one union, re-stamps the target's declared scope from the rewrite,
records what it folded in the combine audit, and never targets a follow-up holder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from _submit_stack import CuratorScript, Stack
from _submit_stack import curator_script as curator_script
from _submit_stack import passthrough_main_checkout as passthrough_main_checkout
from _submit_stack import stack as stack
from shared.declared_scope import DECLARED_SCOPE_METADATA_KEY, derive_declared_scope
from shared.followup_items import FOLLOWUP_ITEMS_METADATA_KEY, item_key, legacy_item, observed_item

from fused_memory.middleware.curator_types import CuratorDecision, RewrittenTask
from fused_memory.middleware.followup_hold import FOLLOWUP_HOLDER_METADATA_KEY

_FILES = ['src/app/a.py', 'src/app/b.py', 'src/app/c.py']
_A = {'summary': 'The retry loop swallows the timeout', 'files': ['src/app/a.py']}
_B = {'summary': 'The cache key ignores the tenant', 'files': ['src/app/b.py']}
_C = {'summary': 'The lock is never released on error', 'files': ['src/app/c.py']}
_REWRITE = RewrittenTask(
    title='Harden the src/app request path',
    description='Fold the request-path follow-ups into one task.',
    details='- retry timeout\n- tenant cache key\n- lock release',
    files_to_modify=['src/app/a.py', 'src/app/b.py', 'src/app/c.py', 'src/app/d.py'],
    priority='medium',
)
_AUDIT_FIELDS = (
    'candidate_ticket_id', 'candidate_source', 'candidate_spawned_from', 'batch_internal',
    'items_before', 'items_after',
)
_HISTORICAL_FIELDS = (
    'ts', 'project_id', 'target_id', 'curator_decision_id', 'old', 'new', 'justification_truncated',
)


def _bundle(parent: str, *items: dict[str, Any]) -> dict[str, Any]:
    return {
        'source': 'agent-followup',
        'spawn_context': 'agent_followup',
        'spawned_from': parent,
        'files': list(_FILES),
        FOLLOWUP_ITEMS_METADATA_KEY: list(items),
    }


def _key(parent: str, item: dict[str, Any]) -> str:
    return item_key(origin_task=parent, files=item['files'], summary=item['summary'])


def _combine_into(script: CuratorScript, target_id: str, target_title: str) -> None:
    script.decisions.append(CuratorDecision(
        action='combine', target_id=target_id, target_fingerprint=target_title,
        rewritten_task=_REWRITE, justification='the same request path',
    ))


def _audit_rows() -> list[dict[str, Any]]:
    path = Path(os.environ['DARK_FACTORY_DATA_DIR']) / 'combine_audit.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


async def _row(stack: Stack, task_id: str) -> dict[str, Any]:
    result = await stack.call('get_task', id=task_id)
    assert 'error' not in result, result
    return result


def _item_keys(metadata: dict[str, Any]) -> list[str]:
    return [item['key'] for item in metadata[FOLLOWUP_ITEMS_METADATA_KEY]]


def _combine_stamp() -> dict[str, Any]:
    return derive_declared_scope(
        _REWRITE.files_to_modify, _REWRITE.description, _REWRITE.details, basis='combine',
    ).model_dump()


@pytest.mark.asyncio
async def test_a_combine_keeps_every_item_restamps_and_audits_what_it_folded(
    stack: Stack, curator_script: CuratorScript,
) -> None:
    target_title = 'Follow-ups from 101'
    target = await stack.file_normal(title=target_title, description='Two.', metadata=_bundle('101', _A, _B))
    target_before = await stack.metadata(target)
    _combine_into(curator_script, target, target_title)

    ticket, resolved = await stack.submit(
        title='Follow-ups from 202', description='Two more.', metadata=_bundle('202', _B, _C),
    )

    assert {key: resolved.get(key) for key in ('status', 'task_id')} == {'status': 'combined', 'task_id': target}
    after = await stack.metadata(target)
    before_keys = [_key('101', _A), _key('101', _B)]
    after_keys = [*before_keys, _key('202', _B), _key('202', _C)]
    assert _item_keys(after) == after_keys
    assert after[DECLARED_SCOPE_METADATA_KEY] == _combine_stamp()
    assert after['files'] == target_before['files']
    assert (after['spawned_from'], after['source']) == ('101', 'agent-followup')
    [audit] = _audit_rows()
    assert {field: audit[field] for field in _AUDIT_FIELDS} == {
        'candidate_ticket_id': ticket,
        'candidate_source': 'agent-followup',
        'candidate_spawned_from': '202',
        'batch_internal': False,
        'items_before': before_keys,
        'items_after': after_keys,
    }
    assert set(_HISTORICAL_FIELDS) <= set(audit)
    assert set(audit['old']) == {'title', 'description_truncated', 'status'}
    assert set(audit['new']) == {'title', 'description_truncated'}
    assert 'refused_reason' not in audit


@pytest.mark.asyncio
async def test_a_legacy_target_keeps_its_single_as_the_first_item(
    stack: Stack, curator_script: CuratorScript,
) -> None:
    target_title = 'A legacy follow-up'
    target_meta = {'escalation_id': 'esc-legacy', 'suggestion_hash': 'a1b2c3d4e5f60718', 'files': ['src/app/a.py']}
    target = await stack.seed(target_title, target_meta)
    _combine_into(curator_script, target, target_title)

    _ticket, resolved = await stack.submit(
        title='Follow-ups from 202', description='Two more.', metadata=_bundle('202', _B, _C),
    )

    assert resolved.get('status') == 'combined', resolved
    legacy = legacy_item(title=target_title, files=['src/app/a.py'], metadata=target_meta)
    assert legacy is not None
    after = await stack.metadata(target)
    assert after[FOLLOWUP_ITEMS_METADATA_KEY][0] == legacy.model_dump(mode='json')
    assert _item_keys(after) == ['a1b2c3d4e5f60718', _key('202', _B), _key('202', _C)]


@pytest.mark.asyncio
async def test_a_legacy_single_absorbed_by_a_combine_is_owned_by_its_target(
    stack: Stack, curator_script: CuratorScript,
) -> None:
    target_title = 'Follow-ups from 101'
    target = await stack.file_normal(title=target_title, description='Two.', metadata=_bundle('101', _A, _B))
    _combine_into(curator_script, target, target_title)
    legacy = {
        'title': 'The lock is never released on error',
        'description': 'One legacy observation.',
        'metadata': {'escalation_id': 'esc-e', 'suggestion_hash': 'b2c3d4e5f6071829', 'files': ['src/app/c.py']},
    }
    _first, combined = await stack.submit(**legacy)
    assert combined.get('status') == 'combined', combined

    _again, resolved = await stack.submit(**legacy)

    assert {key: resolved.get(key) for key in ('status', 'task_id', 'reason')} == {
        'status': 'combined', 'task_id': target, 'reason': 'idempotency_hit',
    }


@pytest.mark.asyncio
async def test_a_combine_of_two_itemless_tasks_writes_no_items(
    stack: Stack, curator_script: CuratorScript,
) -> None:
    target_title = 'Tidy the request path'
    target = await stack.seed(target_title, {'files': ['src/app/a.py']}, description='Plain work.')
    _combine_into(curator_script, target, target_title)

    _ticket, resolved = await stack.submit(
        title='Tidy the cache path', description='More plain work.', metadata={'files': ['src/app/b.py']},
    )

    assert resolved.get('status') == 'combined', resolved
    after = await stack.metadata(target)
    assert FOLLOWUP_ITEMS_METADATA_KEY not in after
    assert after[DECLARED_SCOPE_METADATA_KEY] == _combine_stamp()
    [audit] = _audit_rows()
    assert (audit['items_before'], audit['items_after']) == ([], [])


@pytest.mark.asyncio
async def test_a_holder_target_is_refused_and_audited(
    stack: Stack, curator_script: CuratorScript,
) -> None:
    holder_title = 'Follow-up holder'
    holder = await stack.seed(holder_title, {
        FOLLOWUP_HOLDER_METADATA_KEY: {'project_id': 'p', 'minted_at': '2026-11-08T00:00:00+00:00'},
        'files': ['src/app/a.py'],
    })
    holder_before = await _row(stack, holder)
    _combine_into(curator_script, holder, holder_title)

    ticket, resolved = await stack.submit(
        title='Follow-ups from 202', description='Two more.', metadata=_bundle('202', _B, _C),
    )

    assert resolved.get('status') == 'created', resolved
    assert await _row(stack, holder) == holder_before
    [audit] = _audit_rows()
    assert audit['refused_reason'] == 'target_is_holder'
    assert {field: audit[field] for field in _AUDIT_FIELDS} == {
        'candidate_ticket_id': ticket,
        'candidate_source': 'agent-followup',
        'candidate_spawned_from': '202',
        'batch_internal': False,
        'items_before': [],
        'items_after': [_key('202', _B), _key('202', _C)],
    }


@pytest.mark.asyncio
async def test_a_target_whose_bundle_repeats_a_key_is_not_combined(
    stack: Stack, curator_script: CuratorScript,
) -> None:
    repeated = observed_item(summary=_A['summary'], files=_A['files'], origin_task='101').model_dump(mode='json')
    target_title = 'A bundle the registry could not refuse'
    target = await stack.seed(target_title, {
        'files': list(_FILES), FOLLOWUP_ITEMS_METADATA_KEY: [repeated, repeated],
    })
    target_before = await _row(stack, target)
    _combine_into(curator_script, target, target_title)

    _ticket, resolved = await stack.submit(
        title='Follow-ups from 202', description='Two more.', metadata=_bundle('202', _B, _C),
    )

    assert resolved.get('status') == 'created', resolved
    assert await _row(stack, target) == target_before
    assert _audit_rows() == []


@pytest.mark.asyncio
async def test_an_ineligible_target_whose_bundle_repeats_a_key_is_still_audited_as_refused(
    stack: Stack, curator_script: CuratorScript,
) -> None:
    repeated = observed_item(summary=_A['summary'], files=_A['files'], origin_task='101').model_dump(mode='json')
    target_title = 'A finished bundle the registry could not refuse'
    target = await stack.seed(target_title, {
        'files': list(_FILES), FOLLOWUP_ITEMS_METADATA_KEY: [repeated, repeated],
    }, status='done')
    target_before = await _row(stack, target)
    _combine_into(curator_script, target, target_title)

    ticket, resolved = await stack.submit(
        title='Follow-ups from 202', description='Two more.', metadata=_bundle('202', _B, _C),
    )

    assert resolved.get('status') == 'created', resolved
    assert await _row(stack, target) == target_before
    [audit] = _audit_rows()
    assert audit['refused_reason'] == 'target_not_pending'
    assert {field: audit[field] for field in _AUDIT_FIELDS} == {
        'candidate_ticket_id': ticket,
        'candidate_source': 'agent-followup',
        'candidate_spawned_from': '202',
        'batch_internal': False,
        'items_before': [],
        'items_after': [],
    }


@pytest.mark.asyncio
async def test_an_unstamped_target_gains_the_combine_stamp(
    stack: Stack, curator_script: CuratorScript,
) -> None:
    target_title = 'Filed before the stamp existed'
    target = await stack.seed(target_title, {'files': ['src/app/a.py']})
    assert DECLARED_SCOPE_METADATA_KEY not in await stack.metadata(target)
    _combine_into(curator_script, target, target_title)

    _ticket, resolved = await stack.submit(
        title='Follow-ups from 202', description='Two more.', metadata=_bundle('202', _B, _C),
    )

    assert resolved.get('status') == 'combined', resolved
    stamp = (await stack.metadata(target))[DECLARED_SCOPE_METADATA_KEY]
    assert stamp == _combine_stamp()
    assert stamp['basis'] == 'combine'
