"""Contract tests for :class:`SqliteTaskBackend`."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import pytest
import pytest_asyncio

from fused_memory.backends.sqlite_task_backend import (
    SqliteTaskBackend,
    _format_task_id,
    _merge_metadata,
    _normalize_legacy_memory_hints_value,
    _parse_task_id,
)
from fused_memory.backends.task_backend_errors import TaskmasterError
from fused_memory.config.schema import TaskmasterConfig


@pytest_asyncio.fixture
async def backend(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    yield b
    await b.close()


@pytest_asyncio.fixture
async def project_root(tmp_path):
    return str(tmp_path / 'proj')


@pytest.fixture(autouse=True)
def _clear_malformed_metadata_warning_dedup():
    """Reset the module-level dedup set so each test sees a clean WARN gate."""
    from fused_memory.backends import sqlite_task_backend as _sb
    if hasattr(_sb, '_warned_malformed_task_ids'):
        _sb._warned_malformed_task_ids.clear()
    yield
    if hasattr(_sb, '_warned_malformed_task_ids'):
        _sb._warned_malformed_task_ids.clear()


# ── ID parsing ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    'raw,expected_id,expected_parent',
    [
        ('5', 5, None),
        ('  10 ', 10, None),
        (7, 7, None),
        ('292.1', 1, 292),
        ('100.42', 42, 100),
    ],
)
def test_parse_task_id_accepts_dotted_and_bare(raw, expected_id, expected_parent):
    tid, parent = _parse_task_id(raw)
    assert tid == expected_id
    assert parent == expected_parent


@pytest.mark.parametrize('raw', ['', 'abc', '1.2.3', '5.x', 'x.5'])
def test_parse_task_id_rejects_malformed(raw):
    with pytest.raises(TaskmasterError) as exc:
        _parse_task_id(raw)
    assert exc.value.code == 'INVALID_TASK_ID'


def test_format_task_id_round_trips():
    assert _format_task_id(7, None) == '7'
    assert _format_task_id(2, 7) == '7.2'


# ── Lifecycle ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_close_idempotent(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    b = SqliteTaskBackend(cfg)
    await b.start()
    await b.start()  # idempotent
    assert b.connected is True
    assert b.restart_count == 1
    await b.close()
    await b.close()  # idempotent
    assert b.connected is False


@pytest.mark.asyncio
async def test_is_alive_reports_state(backend, project_root):
    alive, err = await backend.is_alive()
    assert alive is True
    assert err is None
    await backend.close()
    alive, err = await backend.is_alive()
    assert alive is False


# ── add_task / get_task / get_tasks ────────────────────────────────


@pytest.mark.asyncio
async def test_add_task_then_get_returns_dto(backend, project_root):
    dto = await backend.add_task(
        project_root=project_root, title='First', description='desc',
        details='details', priority='high',
    )
    assert dto['id'] == '1'
    assert 'Successfully added' in dto['message']

    one = await backend.get_task('1', project_root=project_root)
    assert one['id'] == 1  # singular get returns int per Taskmaster wire
    assert one['title'] == 'First'
    assert one['priority'] == 'high'
    assert one['status'] == 'pending'
    assert one['subtasks'] == []

    listing = await backend.get_tasks(project_root=project_root)
    assert isinstance(listing['tasks'], list)
    assert listing['tasks'][0]['id'] == '1'  # plural get_tasks returns string


@pytest.mark.asyncio
async def test_add_task_status_param_creates_row_in_given_status(backend, project_root):
    """add_task(status='deferred') lands the row directly in deferred — one INSERT."""
    dto = await backend.add_task(
        project_root=project_root, title='Deferred task', status='deferred',
    )
    one = await backend.get_task(dto['id'], project_root=project_root)
    assert one['status'] == 'deferred'


@pytest.mark.asyncio
async def test_add_task_status_defaults_to_pending(backend, project_root):
    """Omitting status preserves the historical default of 'pending'."""
    dto = await backend.add_task(project_root=project_root, title='Default task')
    one = await backend.get_task(dto['id'], project_root=project_root)
    assert one['status'] == 'pending'


@pytest.mark.asyncio
async def test_add_task_increments_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='one')
    await backend.add_task(project_root=project_root, title='two')
    listing = await backend.get_tasks(project_root=project_root)
    assert sorted(t['id'] for t in listing['tasks']) == ['1', '2']


@pytest.mark.asyncio
async def test_add_task_promotes_prompt_to_title(backend, project_root):
    dto = await backend.add_task(
        project_root=project_root,
        prompt='Build a frobinator that does X\n\nDetails here',
    )
    one = await backend.get_task(dto['id'], project_root=project_root)
    assert one['title'].startswith('Build a frobinator')
    assert 'Details here' in one['description']


@pytest.mark.asyncio
async def test_add_task_without_title_or_prompt_raises(backend, project_root):
    with pytest.raises(TaskmasterError) as exc:
        await backend.add_task(project_root=project_root)
    assert 'prompt' in exc.value.message


@pytest.mark.asyncio
async def test_get_task_not_found_raises(backend, project_root):
    await backend.add_task(project_root=project_root, title='one')
    with pytest.raises(TaskmasterError) as exc:
        await backend.get_task('999', project_root=project_root)
    assert exc.value.code == 'TASKMASTER_TOOL_ERROR'
    assert 'No tasks found' in exc.value.message


# ── set_task_status ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_task_status_returns_per_id_payload(backend, project_root):
    await backend.add_task(project_root=project_root, title='x')
    result = await backend.set_task_status(
        '1', 'done', project_root=project_root,
    )
    assert 'done' in result['message']
    assert result['tasks'] == [{
        'taskId': '1',
        'oldStatus': 'pending',
        'newStatus': 'done',
    }]


@pytest.mark.asyncio
async def test_set_task_status_unknown_id_raises(backend, project_root):
    with pytest.raises(TaskmasterError):
        await backend.set_task_status('99', 'done', project_root=project_root)


# ── update_task ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_task_appends_metadata(backend, project_root):
    await backend.add_task(
        project_root=project_root,
        title='x',
        metadata=json.dumps({'prd': 'old.md'}),
    )
    dto = await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'files': ['src']}),
        append=True,
    )
    assert dto['updated'] is True
    one = await backend.get_task('1', project_root=project_root)
    assert one['metadata']['prd'] == 'old.md'
    assert one['metadata']['files'] == ['src']


@pytest.mark.asyncio
async def test_update_task_overwrites_metadata_without_append(backend, project_root):
    await backend.add_task(
        project_root=project_root, title='x',
        metadata=json.dumps({'prd': 'old.md'}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'prd': 'new.md'}),
    )
    one = await backend.get_task('1', project_root=project_root)
    assert one['metadata'] == {'prd': 'new.md'}


# ── _merge_metadata: new additive-merge semantics ─────────────────


def test_merge_metadata_list_collision_appends():
    """(a) Top-level list collision under append=True concatenates."""
    result = json.loads(_merge_metadata('{"tags":["a"]}', '{"tags":["b"]}', append=True))
    assert result == {"tags": ["a", "b"]}


def test_merge_metadata_list_collision_dedupes_stable_order():
    """(b) Duplicate items are deduped in stable old-then-new order."""
    result = json.loads(
        _merge_metadata('{"tags":["a","b"]}', '{"tags":["b","c"]}', append=True)
    )
    assert result == {"tags": ["a", "b", "c"]}


def test_merge_metadata_scalar_collision_old_wins_under_append():
    """(c) Regression: scalar collision still resolves OLD-wins under append=True."""
    result = json.loads(
        _merge_metadata('{"prd":"old.md"}', '{"prd":"new.md"}', append=True)
    )
    assert result == {"prd": "old.md"}


def test_merge_metadata_append_false_replaces_verbatim():
    """(d) Regression: append=False replaces the metadata verbatim."""
    result = json.loads(
        _merge_metadata('{"prd":"old.md"}', '{"prd":"new.md"}', append=False)
    )
    assert result == {"prd": "new.md"}


# ── _merge_metadata: recursive dict-merge (memory_hints shape) ────


def test_merge_metadata_nested_dict_lists_union():
    """(a) memory_hints dict shape: inner list values union additively."""
    old_raw = '{"memory_hints":{"entities":["A"],"queries":["q1"]}}'
    new_raw = '{"memory_hints":{"entities":["B"],"queries":["q2"]}}'
    result = json.loads(_merge_metadata(old_raw, new_raw, append=True))
    assert result == {"memory_hints": {"entities": ["A", "B"], "queries": ["q1", "q2"]}}


def test_merge_metadata_nested_dict_lists_dedup():
    """(b) Overlap within inner lists is deduped in stable order."""
    old_raw = '{"memory_hints":{"entities":["A","B"],"queries":[]}}'
    new_raw = '{"memory_hints":{"entities":["B","C"],"queries":[]}}'
    result = json.loads(_merge_metadata(old_raw, new_raw, append=True))
    assert result == {"memory_hints": {"entities": ["A", "B", "C"], "queries": []}}


def test_merge_metadata_nested_scalar_collision_old_wins():
    """(c) Nested scalar collision resolves OLD-wins."""
    old_raw = '{"audit":{"created_by":"x"}}'
    new_raw = '{"audit":{"created_by":"y","updated_by":"z"}}'
    result = json.loads(_merge_metadata(old_raw, new_raw, append=True))
    assert result == {"audit": {"created_by": "x", "updated_by": "z"}}


@pytest.mark.asyncio
async def test_update_task_memory_hints_union(backend, project_root):
    """(d) End-to-end through update_task: memory_hints union via append=True."""
    await backend.add_task(
        project_root=project_root,
        title='hinted',
        metadata=json.dumps({'memory_hints': {'entities': ['A'], 'queries': ['q1']}}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['B'], 'queries': ['q2']}}),
        append=True,
    )
    task = await backend.get_task('1', project_root=project_root)
    hints = task['metadata']['memory_hints']
    assert hints == {'entities': ['A', 'B'], 'queries': ['q1', 'q2']}


@pytest.mark.asyncio
async def test_update_task_preserves_sibling_keys_during_memory_hints_append(backend, project_root):
    """Regression: stage2 prompt promises siblings (`files`, `spawned_from`, audit dicts)
    survive an additive merge whose incoming payload supplies only `memory_hints`. Lock
    that promise end-to-end through `update_task`."""
    await backend.add_task(
        project_root=project_root,
        title='audit-row',
        metadata=json.dumps({
            'files': ['src/a.py', 'src/b.py'],
            'spawned_from': 'task-100',
            'audit': {'created_by': 'x'},
        }),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['E1'], 'queries': ['q1']}}),
        append=True,
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata'] == {
        'files': ['src/a.py', 'src/b.py'],
        'spawned_from': 'task-100',
        'audit': {'created_by': 'x'},
        'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
    }


def test_merge_metadata_list_of_dicts_concatenates_without_dedup():
    """Unhashable list items (dicts) fall back to plain concatenation — no dedup."""
    old_raw = '{"x":[{"k":1}]}'
    new_raw = '{"x":[{"k":1}]}'
    result = json.loads(_merge_metadata(old_raw, new_raw, append=True))
    # Both dicts present; unhashable items are NOT deduped (plain concat).
    assert result == {"x": [{"k": 1}, {"k": 1}]}


def test_merge_metadata_type_mismatch_old_wins_for_non_hint_keys():
    """Type mismatch (old=list, new=dict) resolves to OLD wins for arbitrary keys.

    This audit-field-protection rule is intentional for generic keys (e.g. ``x``,
    ``done_provenance``) where a malformed/unexpected write should not be allowed
    to overwrite a structured value.

    Note: ``memory_hints`` is the only key that receives special treatment — it is
    normalised from legacy list-of-dicts shape to canonical dict shape before
    _merge_values runs, so the dict-vs-dict recursive union path handles the merge
    instead.  See test_merge_metadata_legacy_list_hints_coerce_and_union_with_new_dict.
    """
    old_raw = '{"x":[1,2]}'
    new_raw = '{"x":{"a":1}}'
    result = json.loads(_merge_metadata(old_raw, new_raw, append=True))
    assert result["x"] == [1, 2]


# ── _merge_metadata: legacy memory_hints migration ───────────────────


def test_merge_metadata_legacy_list_hints_coerce_and_union_with_new_dict():
    """Legacy list-of-dicts memory_hints is coerced to dict shape and union-merged.

    When an existing row carries the legacy memory_hints shape
    ``[{"entity": ..., "query": ...}, ...]`` and the incoming payload carries the
    canonical shape ``{"entities": [...], "queries": [...]}`` with append=True, the
    legacy list must be coerced to dict shape BEFORE _merge_values runs — so the
    merge falls into the dict-vs-dict recursive path (which unions the inner lists)
    rather than the type-mismatch OLD-wins path (which silently discards the new dict).

    Old-then-new stable order is preserved (same policy as the dict-vs-dict union).

    Also covers symmetric cases to ensure normalization is applied to both sides:
    * old=canonical dict, new=legacy list → union (new side is also normalised)
    * old=legacy list,     new=legacy list → both coerced then unioned
    """
    # Primary case: old=legacy list, new=canonical dict
    old_raw = '{"memory_hints":[{"entity":"E1","query":"q1"},{"entity":"E2","query":"q2"}]}'
    new_raw = '{"memory_hints":{"entities":["E3"],"queries":["q3"]}}'
    result = json.loads(_merge_metadata(old_raw, new_raw, append=True))
    assert result == {"memory_hints": {"entities": ["E1", "E2", "E3"], "queries": ["q1", "q2", "q3"]}}

    # Symmetric case 1: old=canonical dict, new=legacy list → union
    old_raw_sym = '{"memory_hints":{"entities":["E1"],"queries":["q1"]}}'
    new_raw_sym = '{"memory_hints":[{"entity":"E2","query":"q2"}]}'
    result_sym = json.loads(_merge_metadata(old_raw_sym, new_raw_sym, append=True))
    assert result_sym == {"memory_hints": {"entities": ["E1", "E2"], "queries": ["q1", "q2"]}}

    # Symmetric case 2: old=legacy list, new=legacy list → both coerced, then unioned
    old_raw_ll = '{"memory_hints":[{"entity":"E1","query":"q1"}]}'
    new_raw_ll = '{"memory_hints":[{"entity":"E2","query":"q2"}]}'
    result_ll = json.loads(_merge_metadata(old_raw_ll, new_raw_ll, append=True))
    assert result_ll == {"memory_hints": {"entities": ["E1", "E2"], "queries": ["q1", "q2"]}}


def test_merge_metadata_legacy_hints_not_normalized_on_one_sided_write():
    """Normalization is scoped to the collision path: one-sided writes do not migrate.

    When only the *old* side carries ``memory_hints`` (and the incoming write
    does not touch that key), the stored legacy list shape is left unchanged.
    Normalization only fires when BOTH sides carry ``memory_hints``, keeping
    the special case strictly scoped to the merge-collision path and avoiding
    any implicit side-effect on unrelated writes.
    """
    old_raw = '{"tag":"old","memory_hints":[{"entity":"E1","query":"q1"}]}'
    new_raw = '{"tag":"new"}'  # does not carry memory_hints
    result = json.loads(_merge_metadata(old_raw, new_raw, append=True))
    # scalar collision on "tag" → OLD wins
    assert result["tag"] == "old"
    # memory_hints was NOT in the incoming write, so normalization does not fire;
    # the legacy list shape is preserved verbatim in the merged result.
    assert result["memory_hints"] == [{"entity": "E1", "query": "q1"}]


def test_normalize_legacy_memory_hints_handles_partial_and_malformed_entries():
    """_normalize_legacy_memory_hints_value correctly handles edge cases in the list.

    Proves:
    * dict entries with only entity → entity extracted, no query
    * dict entries with only query → query extracted, no entity
    * dict entries with both → both extracted
    * empty dicts → skipped
    * non-dict items (str, None) → skipped
    * empty-string entity/query → skipped
    * None-valued entity/query → skipped
    * duplicate entity/query values → deduplicated in stable (first-seen) order
    * already-canonical dict input → returned unchanged (pass-through)
    * None input → returned unchanged (pass-through)

    Cross-reference: test_merge_metadata_legacy_list_hints_coerce_and_union_with_new_dict
    covers the full _merge_metadata path; this test locks the helper's semantics.
    """
    # Mixed/malformed list
    malformed = [
        {"entity": "E1"},           # only entity — ok
        {"query": "q1"},            # only query — ok
        {"entity": "E2", "query": "q2"},  # both — ok
        {},                         # empty dict — skip
        "not-a-dict",               # non-dict — skip
        None,                       # non-dict — skip
        {"entity": ""},             # empty string — skip
        {"query": None},            # None value — skip
    ]
    result = _normalize_legacy_memory_hints_value(malformed)
    assert result == {"entities": ["E1", "E2"], "queries": ["q1", "q2"]}

    # Duplicates — deduplicated in stable first-seen order
    duped = [
        {"entity": "E1", "query": "q1"},
        {"entity": "E1", "query": "q2"},  # duplicate entity — skip entity, keep query
        {"entity": "E2", "query": "q1"},  # duplicate query — keep entity, skip query
    ]
    result_dedup = _normalize_legacy_memory_hints_value(duped)
    assert result_dedup == {"entities": ["E1", "E2"], "queries": ["q1", "q2"]}

    # Already-canonical dict — pass-through
    canonical = {"entities": ["X"], "queries": ["q"]}
    assert _normalize_legacy_memory_hints_value(canonical) is canonical

    # None — pass-through
    assert _normalize_legacy_memory_hints_value(None) is None


@pytest.mark.asyncio
async def test_update_task_legacy_list_hints_coerce_under_append_true(backend, project_root):
    """End-to-end: legacy list-shape memory_hints row + append=True dict write → union.

    Locks the Stage-2 LLM call path: update_task(append=True) with a canonical-dict
    memory_hints payload now correctly merges with a row that was seeded in legacy
    list-of-dicts shape, rather than silently discarding the incoming dict.
    """
    await backend.add_task(
        project_root=project_root,
        title='legacy-row',
        metadata=json.dumps({'memory_hints': [{'entity': 'E1', 'query': 'q1'}]}),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['E2'], 'queries': ['q2']}}),
        append=True,
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata']['memory_hints'] == {
        'entities': ['E1', 'E2'],
        'queries': ['q1', 'q2'],
    }


@pytest.mark.asyncio
async def test_update_task_legacy_hints_migration_preserves_sibling_metadata(backend, project_root):
    """Sibling metadata keys are untouched when a legacy-list hints row is migrated.

    Mirrors test_update_task_preserves_sibling_keys_during_memory_hints_append but
    proves the no-collateral-damage promise still holds when the row starts in
    legacy list-of-dicts shape rather than canonical dict shape.
    """
    await backend.add_task(
        project_root=project_root,
        title='sibling-row',
        metadata=json.dumps({
            'files': ['src/a.py'],
            'spawned_from': 'task-100',
            'audit': {'created_by': 'x'},
            'memory_hints': [{'entity': 'E1', 'query': 'q1'}],
        }),
    )
    await backend.update_task(
        '1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['E2'], 'queries': ['q2']}}),
        append=True,
    )
    task = await backend.get_task('1', project_root=project_root)
    assert task['metadata'] == {
        'files': ['src/a.py'],
        'spawned_from': 'task-100',
        'audit': {'created_by': 'x'},
        'memory_hints': {'entities': ['E1', 'E2'], 'queries': ['q1', 'q2']},
    }


# ── add_subtask / nested IDs ───────────────────────────────────────


@pytest.mark.asyncio
async def test_add_subtask_returns_dotted_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='parent')
    dto = await backend.add_subtask(
        '1', project_root=project_root, title='Sub one',
    )
    assert dto['id'] == '1.1'
    assert dto['parent_id'] == '1'
    assert 'created' in dto['message']
    assert dto['subtask']['title'] == 'Sub one'


@pytest.mark.asyncio
async def test_add_subtask_increments_local_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='parent')
    a = await backend.add_subtask('1', project_root=project_root, title='A')
    b = await backend.add_subtask('1', project_root=project_root, title='B')
    assert (a['id'], b['id']) == ('1.1', '1.2')

    parent = await backend.get_task('1', project_root=project_root)
    assert [s['id'] for s in parent['subtasks']] == [1, 2]


@pytest.mark.asyncio
async def test_add_subtask_unknown_parent_raises(backend, project_root):
    with pytest.raises(TaskmasterError) as exc:
        await backend.add_subtask('99', project_root=project_root, title='x')
    assert 'Parent task not found' in exc.value.message


@pytest.mark.asyncio
async def test_set_status_on_subtask_via_dotted_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='S')
    result = await backend.set_task_status(
        '1.1', 'done', project_root=project_root,
    )
    assert result['tasks'][0]['taskId'] == '1.1'
    parent = await backend.get_task('1', project_root=project_root)
    assert parent['subtasks'][0]['status'] == 'done'


# ── update_task on subtasks ───────────────────────────────────────


@pytest.mark.asyncio
async def test_update_task_persists_metadata_on_dotted_subtask_id(backend, project_root):
    """metadata written to a subtask via update_task round-trips through get_task.

    Regression: _row_to_task's subtask branch previously dropped the metadata
    column. Both the direct get_task('1.1') path and the parent['subtasks'][0]
    path must now return the persisted metadata dict.
    """
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='Sub')
    await backend.update_task(
        '1.1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['E1'], 'queries': ['q1']}}),
        append=True,
    )

    # (a) Direct get_task on the subtask must carry the metadata.
    sub = await backend.get_task('1.1', project_root=project_root)
    assert sub['metadata'] == {'memory_hints': {'entities': ['E1'], 'queries': ['q1']}}

    # (b) Parent's subtasks[0] must also carry the metadata.
    parent = await backend.get_task('1', project_root=project_root)
    assert parent['subtasks'][0]['metadata'] == {'memory_hints': {'entities': ['E1'], 'queries': ['q1']}}


@pytest.mark.asyncio
async def test_get_tasks_listing_surfaces_subtask_metadata(backend, project_root):
    """get_tasks listing also surfaces metadata on subtask dicts.

    Locks both the populated-metadata path and the None→{} default so
    the reconciliation context_assembler (which reads via get_tasks) sees
    the same metadata as the direct get_task path.
    """
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='Sub')
    await backend.update_task(
        '1.1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['X']}}),
        append=False,
    )

    listing = await backend.get_tasks(project_root=project_root)

    # (a) Populated-metadata path: get_tasks listing carries the persisted value.
    assert listing['tasks'][0]['subtasks'][0]['metadata'] == {'memory_hints': {'entities': ['X']}}

    # (b) Empty-default path: a freshly added subtask (no update_task) has metadata={}.
    await backend.add_subtask('1', project_root=project_root, title='Fresh')
    listing2 = await backend.get_tasks(project_root=project_root)
    fresh_sub = next(s for s in listing2['tasks'][0]['subtasks'] if s['title'] == 'Fresh')
    assert 'metadata' in fresh_sub
    assert fresh_sub['metadata'] == {}


@pytest.mark.asyncio
async def test_update_task_memory_hints_union_on_subtask(backend, project_root):
    """memory_hints union semantics (append=True) work end-to-end on subtasks.

    Models the top-level test_update_task_memory_hints_union but for the
    dotted-id subtask path. Locks the reconciliation behavior: stage2 attaches
    hints across multiple cycles and the lists grow via union (no duplicates).
    """
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='hinted')

    # First cycle: write initial hints.
    await backend.update_task(
        '1.1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['A'], 'queries': ['q1']}}),
        append=True,
    )
    # Second cycle: union new hints.
    await backend.update_task(
        '1.1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['B'], 'queries': ['q2']}}),
        append=True,
    )

    sub = await backend.get_task('1.1', project_root=project_root)
    hints = sub['metadata']['memory_hints']
    assert hints == {'entities': ['A', 'B'], 'queries': ['q1', 'q2']}


@pytest.mark.asyncio
async def test_update_task_legacy_hints_coerce_on_subtask_dotted_id(backend, project_root):
    """Legacy list-shape memory_hints is coerced on the dotted-id subtask path.

    Stage-2 attaches hints to subtasks via the dotted-id path; this locks parity
    with the top-level test_update_task_legacy_list_hints_coerce_under_append_true.
    Both the direct get_task('1.1') and the parent['subtasks'][0] paths must
    return the unioned dict-shape after the migration write.
    """
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='hinted')

    # Seed the subtask with the legacy list-of-dicts shape.
    await backend.update_task(
        '1.1', project_root=project_root,
        metadata=json.dumps({'memory_hints': [{'entity': 'E1', 'query': 'q1'}]}),
        append=False,
    )
    # Stage-2 writes canonical dict shape with append=True.
    await backend.update_task(
        '1.1', project_root=project_root,
        metadata=json.dumps({'memory_hints': {'entities': ['E2'], 'queries': ['q2']}}),
        append=True,
    )

    expected = {'entities': ['E1', 'E2'], 'queries': ['q1', 'q2']}

    # (a) Direct get_task on the subtask.
    sub = await backend.get_task('1.1', project_root=project_root)
    assert sub['metadata']['memory_hints'] == expected

    # (b) Parent's subtasks[0] also reflects the migration.
    parent = await backend.get_task('1', project_root=project_root)
    assert parent['subtasks'][0]['metadata']['memory_hints'] == expected


@pytest.mark.asyncio
async def test_row_to_task_returns_empty_dict_for_malformed_metadata(backend, project_root):
    """_row_to_task coerces malformed metadata JSON to {} for both top-level and subtask rows.

    Regression guard: if a legacy row holds a non-JSON string in the metadata
    column, the except branch in _row_to_task must surface {} rather than the
    raw string, so downstream `(task.get('metadata') or {}).get(...)` callers
    never receive a str and raise AttributeError.
    """
    # Set up a parent task and a subtask via the normal API.
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='child')

    # Directly corrupt both rows' metadata column with a non-JSON string.
    conn = await backend._get_connection(project_root)
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON' WHERE parent_id = 0 AND id = 1"
    )
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON' WHERE parent_id = 1 AND id = 1"
    )
    await conn.commit()

    # Top-level task: malformed metadata must surface as {}, not 'NOT_JSON'.
    parent = await backend.get_task('1', project_root=project_root)
    assert parent['metadata'] == {}

    # Subtask (via direct get_task): same contract.
    sub = await backend.get_task('1.1', project_root=project_root)
    assert sub['metadata'] == {}

    # Subtask (via parent['subtasks']): same contract.
    assert parent['subtasks'][0]['metadata'] == {}


@pytest.mark.asyncio
async def test_row_to_task_warns_on_malformed_metadata(backend, project_root, caplog):
    """_row_to_task emits a WARNING when it coerces malformed metadata JSON to {}.

    The warning must include the row's tag, id, parent_id (raw DB int), and a
    truncated preview of the bad metadata_raw value so an operator can locate
    and repair the offending row.  The {}-coercion contract must also hold.

    Both the top-level and subtask paths are exercised — _row_to_task is the
    shared converter for both shapes, so the warning must fire regardless of
    whether the corrupted row is a root task or a child.
    """
    # Create a top-level task and a subtask so both row shapes exist.
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='child')

    # Directly corrupt both rows' metadata column with a non-JSON string.
    conn = await backend._get_connection(project_root)
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_GARBAGE_xyz' WHERE parent_id = 0 AND id = 1"
    )
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_GARBAGE_xyz' WHERE parent_id = 1 AND id = 1"
    )
    await conn.commit()

    # Capture WARNING-level records for both top-level and subtask reads.
    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        task = await backend.get_task('1', project_root=project_root)
        sub = await backend.get_task('1.1', project_root=project_root)

    # The {}-coercion contract still holds for both shapes.
    assert task['metadata'] == {}
    assert sub['metadata'] == {}

    # At least one WARNING record must mention tag, id, parent_id, and the payload preview.
    # Use labeled tokens (e.g. 'id=1') rather than bare digits to prevent false positives
    # from incidental numeric matches elsewhere in the log text.
    warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_msgs, 'Expected at least one WARNING log record; got none'
    combined = ' '.join(warning_msgs)

    # Top-level row: tag=master, id=1, parent_id=0 (the top-level sentinel).
    assert 'master' in combined, f'Expected tag "master" in warning; got: {combined!r}'
    assert re.search(r'\bid=1\b', combined), (
        f'Expected word-bounded labeled token "id=1" in warning (not the substring '
        f'inside "parent_id=1"); got: {combined!r}'
    )
    assert 'parent_id=0' in combined, (
        f'Expected labeled token "parent_id=0" in warning; got: {combined!r}'
    )
    assert 'NOT_JSON_GARBAGE' in combined, (
        f'Expected metadata_raw preview in warning; got: {combined!r}'
    )

    # Subtask row: parent_id=1 (the DB int of the parent, not the sentinel).
    assert 'parent_id=1' in combined, (
        f'Expected labeled token "parent_id=1" for subtask warning; got: {combined!r}'
    )

    # The warning must carry a labeled project_root= token so an operator can
    # identify which DB is corrupt (added by task 1263).
    assert 'project_root=' in combined, (
        f'Expected labeled token "project_root=" in warning; got: {combined!r}'
    )


@pytest.mark.asyncio
async def test_row_to_task_warning_deduplicated_per_id_per_process(
    backend, project_root, caplog,
):
    """Repeated reads of the same malformed-metadata row emit at most one WARNING.

    `_get_tasks_internal` invokes `_row_to_task` on every row of every `get_tasks`
    call. A project DB with many corrupted rows would otherwise flood the log
    with one WARNING per row per call. The dedup gate caches `(tag, parent_id,
    id)` triples already warned about and skips subsequent emissions for the
    lifetime of the process.
    """
    await backend.add_task(project_root=project_root, title='parent')
    conn = await backend._get_connection(project_root)
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_DEDUP' "
        "WHERE parent_id = 0 AND id = 1"
    )
    await conn.commit()

    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ):
        first = await backend.get_task('1', project_root=project_root)
        second = await backend.get_task('1', project_root=project_root)
        listing = await backend.get_tasks(project_root=project_root)

    # Coercion contract still holds for every read.
    assert first['metadata'] == {}
    assert second['metadata'] == {}
    assert listing['tasks'][0]['metadata'] == {}

    malformed_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING
        and 'malformed metadata' in r.message
    ]
    assert len(malformed_msgs) == 1, (
        f'Expected exactly one malformed-metadata WARNING across three reads '
        f'of the same row; got {len(malformed_msgs)}: {malformed_msgs}'
    )


@pytest.mark.asyncio
async def test_row_to_task_warning_dedup_key_includes_parent_id(
    backend, project_root, caplog,
):
    """Top-level (parent_id=0, id=1) and subtask (parent_id=1, id=1) dedup independently.

    They share the short int id but represent different rows; the WARNING gate
    must key on the full (tag, parent_id, id) triple so both surface once.
    """
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='child')
    conn = await backend._get_connection(project_root)
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_KEYS' "
        "WHERE parent_id = 0 AND id = 1"
    )
    await conn.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_KEYS' "
        "WHERE parent_id = 1 AND id = 1"
    )
    await conn.commit()

    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ):
        await backend.get_task('1', project_root=project_root)
        await backend.get_task('1.1', project_root=project_root)

    malformed_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING
        and 'malformed metadata' in r.message
    ]
    assert len(malformed_msgs) == 2, (
        f'Expected two distinct dedup keys (top-level vs subtask); got '
        f'{len(malformed_msgs)}: {malformed_msgs}'
    )


@pytest.mark.asyncio
async def test_row_to_task_warning_dedup_distinguishes_project_roots(
    backend, tmp_path, caplog,
):
    """Two project_roots sharing the same (tag, parent_id, id) row emit distinct WARNs.

    A single SqliteTaskBackend instance services all project_roots.  Before the
    fix, the dedup key was (tag, parent_id, id), so both project_roots' corrupted
    (master, 0, 1) rows collided on the same key — the second WARN was silently
    swallowed.  The fix prepends project_root to the tuple, making each project
    DB's WARNING independent.  Each WARNING must also carry a ``project_root=``
    labeled token so an operator can pin the WARN to its DB.
    """
    proj_a = str(tmp_path / 'proj_a')
    proj_b = str(tmp_path / 'proj_b')

    # Each project_root gets a canonical (tag=master, parent_id=0, id=1) row.
    await backend.add_task(project_root=proj_a, title='parent_a')
    await backend.add_task(project_root=proj_b, title='parent_b')

    # Corrupt both DBs' metadata column with a non-JSON string.
    conn_a = await backend._get_connection(proj_a)
    await conn_a.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_PROJ' WHERE parent_id = 0 AND id = 1"
    )
    await conn_a.commit()

    conn_b = await backend._get_connection(proj_b)
    await conn_b.execute(
        "UPDATE tasks SET metadata = 'NOT_JSON_PROJ' WHERE parent_id = 0 AND id = 1"
    )
    await conn_b.commit()

    # Read from both project_roots and capture warnings.
    with caplog.at_level(
        logging.WARNING, logger='fused_memory.backends.sqlite_task_backend',
    ):
        task_a = await backend.get_task('1', project_root=proj_a)
        task_b = await backend.get_task('1', project_root=proj_b)

    # The {}-coercion contract holds for both.
    assert task_a['metadata'] == {}
    assert task_b['metadata'] == {}

    # Both project_roots must produce their own WARNING — the dedup tuple
    # now distinguishes (proj_a, master, 0, 1) from (proj_b, master, 0, 1).
    malformed_msgs = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING
        and 'malformed metadata' in r.message
    ]
    assert len(malformed_msgs) == 2, (
        f'Expected exactly two malformed-metadata WARNs (one per project_root); '
        f'got {len(malformed_msgs)}: {malformed_msgs}'
    )

    # Each individual warning must contain its respective project_root path.
    msgs_containing_proj_a = [m for m in malformed_msgs if proj_a in m]
    msgs_containing_proj_b = [m for m in malformed_msgs if proj_b in m]
    assert msgs_containing_proj_a, (
        f'Expected a WARNING containing {proj_a!r}; messages: {malformed_msgs}'
    )
    assert msgs_containing_proj_b, (
        f'Expected a WARNING containing {proj_b!r}; messages: {malformed_msgs}'
    )

    # Every WARNING must carry the labeled project_root= token.
    for msg in malformed_msgs:
        assert 'project_root=' in msg, (
            f'Expected "project_root=" token in WARNING message; got: {msg!r}'
        )


# ── remove_tasks with cascade ──────────────────────────────────────


@pytest.mark.asyncio
async def test_remove_tasks_cascades_to_subtasks(backend, project_root):
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='A')
    await backend.add_subtask('1', project_root=project_root, title='B')

    dto = await backend.remove_tasks(['1'], project_root=project_root)

    assert dto['successful'] == 3
    assert dto['failed'] == 0
    assert sorted(dto['removed_ids']) == ['1', '1.1', '1.2']

    listing = await backend.get_tasks(project_root=project_root)
    assert listing['tasks'] == []


@pytest.mark.asyncio
async def test_remove_tasks_unknown_id_returns_failure_dto(backend, project_root):
    dto = await backend.remove_tasks(['99'], project_root=project_root)
    assert dto['successful'] == 0
    assert dto['failed'] == 1
    assert dto['removed_ids'] == []


@pytest.mark.asyncio
async def test_remove_tasks_batch_mixed_existing_missing(backend, project_root):
    # Two top-levels exist (1, 2); 3 and 99 do not.
    await backend.add_task(project_root=project_root, title='alpha')
    await backend.add_task(project_root=project_root, title='beta')

    dto = await backend.remove_tasks(
        ['1', '2', '3', '99'], project_root=project_root,
    )

    assert dto['successful'] == 2
    assert dto['failed'] == 2
    assert sorted(dto['removed_ids']) == ['1', '2']
    assert '3' in dto['message']
    assert '99' in dto['message']

    listing = await backend.get_tasks(project_root=project_root)
    assert listing['tasks'] == []


@pytest.mark.asyncio
async def test_remove_tasks_cascades_with_explicit_subtask(backend, project_root):
    # Caller asks to remove parent 1 AND its subtask 1.1; cascade naturally
    # pulls in 1.1 as well — must report once, not twice.
    await backend.add_task(project_root=project_root, title='parent')
    await backend.add_subtask('1', project_root=project_root, title='A')
    await backend.add_subtask('1', project_root=project_root, title='B')

    dto = await backend.remove_tasks(
        ['1', '1.1'], project_root=project_root,
    )

    # 1, 1.1, 1.2 — three rows actually deleted; no double-count of 1.1.
    assert dto['successful'] == 3
    assert dto['failed'] == 0
    assert sorted(dto['removed_ids']) == ['1', '1.1', '1.2']
    assert dto['removed_ids'].count('1.1') == 1


@pytest.mark.asyncio
async def test_remove_tasks_atomicity_on_malformed_id(backend, project_root):
    await backend.add_task(project_root=project_root, title='alpha')
    await backend.add_task(project_root=project_root, title='beta')

    with pytest.raises(TaskmasterError):
        # 'oops' is not a parseable id — the whole batch fails before any
        # delete runs. Verify state is unchanged afterwards.
        await backend.remove_tasks(
            ['1', 'oops', '2'], project_root=project_root,
        )

    listing = await backend.get_tasks(project_root=project_root)
    assert sorted(t['id'] for t in listing['tasks']) == ['1', '2']


@pytest.mark.asyncio
async def test_remove_tasks_rejects_nested_subtask_id_atomically(backend, project_root):
    """remove_tasks raises INVALID_TASK_ID for 3+-level nested ids and rolls back.

    Regression contract: the nested-subtask-id path propagates through the
    public ``remove_tasks`` API surface (not just the private ``_parse_task_id``
    helper) and the entire batch is aborted — no rows are deleted.
    """
    await backend.add_task(project_root=project_root, title='alpha')
    await backend.add_task(project_root=project_root, title='beta')

    with pytest.raises(TaskmasterError) as exc_info:
        # '1.2.3' is a 3-level nested id — not supported; the whole batch
        # must fail before any delete runs.
        await backend.remove_tasks(
            ['1', '1.2.3', '2'], project_root=project_root,
        )

    assert exc_info.value.code == 'INVALID_TASK_ID'
    # Key off the offending id repr rather than pinning the prose — proves the
    # right path fired without coupling to exact wording in _parse_task_id.
    assert "'1.2.3'" in exc_info.value.message

    # State must be unchanged — both tasks still present.
    listing = await backend.get_tasks(project_root=project_root)
    assert sorted(t['id'] for t in listing['tasks']) == ['1', '2']


@pytest.mark.asyncio
async def test_remove_tasks_empty_list_is_noop(backend, project_root):
    dto = await backend.remove_tasks([], project_root=project_root)
    assert dto['successful'] == 0
    assert dto['failed'] == 0
    assert dto['removed_ids'] == []


# ── Dependencies ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_add_and_remove_dependency_round_trip(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_task(project_root=project_root, title='b')

    add = await backend.add_dependency('2', '1', project_root=project_root)
    assert add['id'] == '2' and add['dependency_id'] == '1'

    listing = await backend.get_tasks(project_root=project_root)
    by_id = {t['id']: t for t in listing['tasks']}
    assert by_id['2']['dependencies'] == [1]

    remove = await backend.remove_dependency(
        '2', '1', project_root=project_root,
    )
    assert remove['id'] == '2'
    listing = await backend.get_tasks(project_root=project_root)
    by_id = {t['id']: t for t in listing['tasks']}
    assert by_id['2']['dependencies'] == []


@pytest.mark.asyncio
async def test_add_dependency_self_loop_raises(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    with pytest.raises(TaskmasterError):
        await backend.add_dependency('1', '1', project_root=project_root)


@pytest.mark.asyncio
async def test_validate_dependencies_reports_dangling(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    await backend.add_task(project_root=project_root, title='b')
    await backend.add_dependency('2', '1', project_root=project_root)
    # Remove the target so the dependency on it dangles.
    await backend.remove_tasks(['1'], project_root=project_root)
    res = await backend.validate_dependencies(project_root=project_root)
    assert 'Dangling dependencies' in res['message']
    assert '2 -> 1' in res['message']


@pytest.mark.asyncio
async def test_validate_dependencies_clean_returns_success(backend, project_root):
    await backend.add_task(project_root=project_root, title='a')
    res = await backend.validate_dependencies(project_root=project_root)
    assert res['message'] == 'Dependencies validated successfully'


# ── Persistence on disk ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_db_file_lives_at_taskmaster_tasks_dir(backend, project_root):
    await backend.add_task(project_root=project_root, title='x')
    expected = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
    assert expected.exists()


@pytest.mark.asyncio
async def test_state_survives_close_and_reopen(tmp_path):
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    project_root = str(tmp_path / 'proj')
    b1 = SqliteTaskBackend(cfg)
    await b1.start()
    await b1.add_task(project_root=project_root, title='persisted')
    await b1.close()

    b2 = SqliteTaskBackend(cfg)
    await b2.start()
    listing = await b2.get_tasks(project_root=project_root)
    assert [t['title'] for t in listing['tasks']] == ['persisted']
    await b2.close()


@pytest.mark.asyncio
async def test_checkpoint_all_reports_per_project_result(tmp_path):
    """``checkpoint_all`` returns ``{root: {busy, log, checkpointed}}`` for
    every open project, and an empty dict when no project has been touched."""
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg)
    await backend.start()

    # No projects opened yet → empty result.
    assert await backend.checkpoint_all() == {}

    root_a = str(tmp_path / 'a')
    root_b = str(tmp_path / 'b')
    await backend.add_task(project_root=root_a, title='a')
    await backend.add_task(project_root=root_b, title='b')

    results = await backend.checkpoint_all()
    assert set(results.keys()) == {root_a, root_b}
    for root, r in results.items():
        # busy=0 with no concurrent readers; log/checkpointed are non-negative.
        assert r['busy'] == 0, f'{root}: unexpected busy {r}'
        assert r['log'] >= 0
        assert r['checkpointed'] >= 0
    await backend.close()


@pytest.mark.asyncio
async def test_close_runs_final_truncate_checkpoint(tmp_path):
    """``close()`` should run a final TRUNCATE checkpoint so the next open
    sees an empty WAL and the main DB file is fully up-to-date — minimises
    recovery work on the next start."""
    cfg = TaskmasterConfig(project_root=str(tmp_path))
    backend = SqliteTaskBackend(cfg)
    await backend.start()
    project_root = str(tmp_path / 'proj')
    await backend.add_task(project_root=project_root, title='one')
    await backend.close()

    wal_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db-wal'
    # After clean close, the WAL file either does not exist or has been
    # truncated to its 32-byte header (= 0-frame state). Either is acceptable.
    if wal_path.exists():
        # 32 bytes is the WAL header with zero frames.
        assert wal_path.stat().st_size <= 32, (
            f'WAL not truncated on close: {wal_path.stat().st_size} bytes'
        )


# ── Concurrency ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_add_task_yields_unique_ids(backend, project_root):
    """The per-project write lock must serialise id allocation."""
    coros = [
        backend.add_task(project_root=project_root, title=f't{i}')
        for i in range(20)
    ]
    results = await asyncio.gather(*coros)
    ids = sorted(int(r['id']) for r in results)
    assert ids == list(range(1, 21))


# ── Monotonic id allocation (id-recycling regression) ───────────────
#
# Fix A: ``add_task`` / ``add_subtask`` allocate
# ``max(MAX(tasks.id), id_counters.max_id) + 1`` so a deleted id is NEVER
# reissued.  Without this, deleting the top task frees its id, the next
# ``add_task`` re-mints it, and an orphaned worktree keyed on that numeric id
# gets misadopted for unrelated work (reify task 3770).


@pytest.mark.asyncio
async def test_top_level_id_not_reused_after_delete(backend, project_root):
    """Core regression: deleting the top task must NOT free its id for reuse."""
    one = await backend.add_task(project_root=project_root, title='first')
    assert one['id'] == '1'
    await backend.remove_tasks(['1'], project_root=project_root)
    two = await backend.add_task(project_root=project_root, title='second')
    assert two['id'] == '2'  # NOT '1'


@pytest.mark.asyncio
async def test_subtask_id_not_reused_after_delete(backend, project_root):
    """A deleted subtask id is not reissued within the same parent."""
    await backend.add_task(project_root=project_root, title='parent')
    a = await backend.add_subtask('1', project_root=project_root, title='A')
    assert a['id'] == '1.1'
    await backend.remove_tasks(['1.1'], project_root=project_root)
    b = await backend.add_subtask('1', project_root=project_root, title='B')
    assert b['id'] == '1.2'  # NOT '1.1'


@pytest.mark.asyncio
async def test_id_monotonic_across_delete_add_cycles(backend, project_root):
    """Repeated create+delete of the trailing task keeps bumping the id."""
    ids = []
    for _ in range(5):
        dto = await backend.add_task(project_root=project_root, title='cycle')
        ids.append(int(dto['id']))
        await backend.remove_tasks([dto['id']], project_root=project_root)
    assert ids == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_delete_current_max_still_bumps(backend, project_root):
    """Deleting the current MAX row still advances past it (counter holds)."""
    await backend.add_task(project_root=project_root, title='a')  # 1
    await backend.add_task(project_root=project_root, title='b')  # 2
    await backend.remove_tasks(['2'], project_root=project_root)  # max row gone
    three = await backend.add_task(project_root=project_root, title='c')
    assert three['id'] == '3'  # NOT '2'


@pytest.mark.asyncio
async def test_id_counter_per_tag_isolation(backend, project_root):
    """Counters are scoped per tag — a delete in one tag never affects another."""
    await backend.add_task(project_root=project_root, title='m1', tag='master')
    await backend.add_task(project_root=project_root, title='f1', tag='feature')
    await backend.remove_tasks(['1'], project_root=project_root, tag='master')
    m2 = await backend.add_task(project_root=project_root, title='m2', tag='master')
    f2 = await backend.add_task(project_root=project_root, title='f2', tag='feature')
    assert m2['id'] == '2'  # master counter held past the delete
    assert f2['id'] == '2'  # feature sequence independent, unaffected


@pytest.mark.asyncio
async def test_id_counter_survives_close_reopen(tmp_path):
    """The counter persists across a connection close/reopen.

    Mirrors a ``systemctl restart fused-memory`` cycle: the high-water mark
    must outlive the process so a delete-then-restart-then-add can't recycle.
    """
    proot = str(tmp_path / 'proj')
    cfg = TaskmasterConfig(project_root=str(tmp_path))

    b1 = SqliteTaskBackend(cfg)
    await b1.start()
    await b1.add_task(project_root=proot, title='one')   # id 1
    await b1.remove_tasks(['1'], project_root=proot)
    await b1.close()

    b2 = SqliteTaskBackend(cfg)
    await b2.start()
    try:
        two = await b2.add_task(project_root=proot, title='two')
        assert two['id'] == '2'  # counter survived the reopen — NOT '1'
    finally:
        await b2.close()


@pytest.mark.asyncio
async def test_id_counter_self_heals_when_empty_but_tasks_present(backend, project_root):
    """A legacy DB (tasks present, id_counters empty) honours the row high-water.

    Simulates an upgrade onto a DB that predates the counter: the first
    post-upgrade alloc must be ``MAX(tasks.id) + 1``, then the counter is
    seeded so it holds the line on subsequent deletes.
    """
    await backend.add_task(project_root=project_root, title='a')  # 1
    await backend.add_task(project_root=project_root, title='b')  # 2

    # Wipe the counter to mimic a pre-Fix-A DB.
    conn = await backend._get_connection(project_root)
    await conn.execute('DELETE FROM id_counters')
    await conn.commit()

    three = await backend.add_task(project_root=project_root, title='c')
    assert three['id'] == '3'  # self-healed from MAX(tasks.id)

    # And the counter now holds across a delete of the current max.
    await backend.remove_tasks(['3'], project_root=project_root)
    four = await backend.add_task(project_root=project_root, title='d')
    assert four['id'] == '4'


# ── Cancellation hardening ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_status_cancellation_leaves_connection_clean(
    backend, project_root,
):
    """A cancellation arriving while ``set_task_status`` is queued behind
    the write_lock must not leave the connection mid-transaction.

    Reproduces the soak-cancel signature: hold the per-project write lock,
    queue a ``set_task_status`` against it, cancel the awaiter via
    ``wait_for(timeout=0.001)``, then assert the next ``set_task_status``
    applies cleanly. Pre-fix (Exception-only suppress + unshielded
    rollback) the connection could end up holding an open BEGIN, which
    surfaces as ``cannot start a transaction within a transaction``
    on the next mutation.
    """
    # Seed: one task to flip.
    await backend.add_task(project_root=project_root, title='t0')
    assert (await backend.get_task('1', project_root))['status'] == 'pending'

    # Acquire the per-project write lock so the next set_task_status blocks.
    lock = backend._write_lock(project_root)
    await lock.acquire()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                backend.set_task_status('1', 'in-progress', project_root),
                timeout=0.001,
            )
    finally:
        lock.release()

    # Connection state must be clean: the next mutation succeeds.
    res = await backend.set_task_status('1', 'done', project_root)
    assert res['tasks'][0]['newStatus'] == 'done'
    assert (await backend.get_task('1', project_root))['status'] == 'done'
