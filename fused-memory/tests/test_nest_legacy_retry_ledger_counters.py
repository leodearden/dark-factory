"""Tests for ``scripts/nest_legacy_retry_ledger_counters.py`` (task 7135).

The script moved the two legacy top-level infra-resume counters under
``metadata.retry_ledger`` in the stored blobs, so task 7135 could retire the
parse-time lift in ``shared.task_metadata``. It is driven here only through its
public seams: the pure transform, the payload builder, and ``run`` over an
MCP-shaped adapter whose tools are a REAL started ``SqliteTaskBackend`` — the
raw storage, the whole-blob replace and the metadata-lookup paging are the
behaviours the live writes depend on.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from _fm_helpers import load_script_module
from pydantic import ValidationError
from shared.task_metadata import RetryLedger, parse_metadata

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'nest_legacy_retry_ledger_counters.py'

_mod = load_script_module(SCRIPT_PATH, mod_name='nest_legacy_retry_ledger_counters')
COUNTER_KEYS = _mod.COUNTER_KEYS
MalformedRetryLedgerError = _mod.MalformedRetryLedgerError
NestingPlan = _mod.NestingPlan
nest_counters = _mod.nest_counters
build_update_payload = _mod.build_update_payload
run = _mod.run

FAILURES = 'consecutive_infra_resume_failures'
ITERATION = 'last_infra_resume_iteration_count'
DONE_PROVENANCE = {'kind': 'merged', 'commit': 'a' * 40}
ENQUEUED_AT = '2026-01-01T00:00:00+00:00'

BOTH_COUNTERS_META = {
    FAILURES: 3,
    ITERATION: 7,
    'done_provenance': DONE_PROVENANCE,
    'files': ['a.py'],
}
CONFLICTING_LEDGER_META = {
    FAILURES: 9,
    ITERATION: 4,
    'retry_ledger': {FAILURES: 2, 'merge_first_enqueued_at': ENQUEUED_AT},
}
CLEAN_META = {'files': ['b.py'], 'retry_ledger': {FAILURES: 1}}
# The shape of reify #4880: a null counter, a non-null sibling, no retry_ledger.
NULL_COUNTER_META = {FAILURES: None, ITERATION: 6, 'done_provenance': DONE_PROVENANCE}


class TestNestCounters:
    def test_blob_without_top_level_counter_needs_no_plan(self):
        assert nest_counters(copy.deepcopy(CLEAN_META)) is None

    def test_both_counters_move_into_a_new_ledger(self):
        meta = copy.deepcopy(BOTH_COUNTERS_META)
        plan = nest_counters(meta)

        assert isinstance(plan, NestingPlan)
        assert FAILURES not in plan.after
        assert ITERATION not in plan.after
        assert plan.after['retry_ledger'] == {FAILURES: 3, ITERATION: 7}
        assert {k: v for k, v in plan.after.items() if k != 'retry_ledger'} == {
            'done_provenance': DONE_PROVENANCE,
            'files': ['a.py'],
        }
        assert plan.lifted == (FAILURES, ITERATION)
        assert plan.shadowed == ()
        assert plan.dropped == ()
        assert meta == BOTH_COUNTERS_META

    def test_nested_value_wins_and_the_shadowed_copy_is_dropped(self):
        meta = copy.deepcopy(CONFLICTING_LEDGER_META)
        plan = nest_counters(meta)

        assert plan is not None
        assert plan.after == {
            'retry_ledger': {FAILURES: 2, ITERATION: 4, 'merge_first_enqueued_at': ENQUEUED_AT},
        }
        assert plan.lifted == (ITERATION,)
        assert plan.shadowed == (FAILURES,)
        assert plan.dropped == ()
        assert meta == CONFLICTING_LEDGER_META

    def test_null_counter_is_dropped_and_its_non_null_sibling_nested(self):
        meta = copy.deepcopy(NULL_COUNTER_META)
        plan = nest_counters(meta)

        assert plan is not None
        assert plan.after == {'done_provenance': DONE_PROVENANCE, 'retry_ledger': {ITERATION: 6}}
        assert plan.dropped == (FAILURES,)
        assert plan.lifted == (ITERATION,)
        assert plan.shadowed == ()
        assert meta == NULL_COUNTER_META
        parse_metadata(plan.after, direction='write', enforce=True)

    def test_only_null_counters_leave_no_ledger_behind(self):
        plan = nest_counters({FAILURES: None, ITERATION: None, 'files': ['a.py']})

        assert plan is not None
        assert plan.after == {'files': ['a.py']}
        assert plan.dropped == (FAILURES, ITERATION)
        assert plan.lifted == ()

    def test_null_counter_is_dropped_even_where_a_nested_value_exists(self):
        plan = nest_counters({FAILURES: None, 'retry_ledger': {FAILURES: 2}})

        assert plan is not None
        assert plan.after == {'retry_ledger': {FAILURES: 2}}
        assert plan.dropped == (FAILURES,)
        assert plan.shadowed == ()

    @pytest.mark.parametrize('ledger', ['oops', [1, 2], 5])
    def test_non_dict_ledger_is_refused_naming_the_value(self, ledger):
        with pytest.raises(MalformedRetryLedgerError, match=re.escape(repr(ledger))):
            nest_counters({FAILURES: 1, 'retry_ledger': ledger})

    def test_counter_keys_are_retry_ledger_fields(self):
        assert set(COUNTER_KEYS) <= set(RetryLedger.model_fields)


class TestBuildUpdatePayload:
    def test_payload_is_a_whole_blob_replace_of_metadata_only(self):
        meta = {'retry_ledger': {FAILURES: 3}, 'files': ['a.py']}
        payload = build_update_payload('42', '/proj', meta)

        assert set(payload) == {'id', 'project_root', 'metadata', 'metadata_mode'}
        assert payload['id'] == '42'
        assert payload['project_root'] == '/proj'
        assert payload['metadata_mode'] == 'replace'
        assert json.loads(payload['metadata']) == {**meta, 'allow_mcp_markup': True}
        assert 'description' not in payload
        assert 'details' not in payload


class BackendToolClient:
    """An MCP-shaped ``call_tool`` over a real ``SqliteTaskBackend``.

    A raised backend exception comes back as an ``{'error', 'error_type'}``
    reply, the way ``@mcp_tool_errors`` answers on the server. Every call is
    recorded.
    """

    def __init__(self, backend: SqliteTaskBackend):
        self.backend = backend
        self.calls: list[tuple[str, dict]] = []

    @property
    def tool_names(self) -> list[str]:
        return [name for name, _ in self.calls]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, copy.deepcopy(arguments)))
        handlers = {
            'find_tasks_by_metadata': self.find_tasks_by_metadata,
            'get_task': self.get_task,
            'update_task': self.update_task,
        }
        if name not in handlers:
            raise AssertionError(f'unexpected tool call: {name}')
        try:
            return await handlers[name](arguments)
        except Exception as exc:
            return {'error': str(exc), 'error_type': type(exc).__name__}

    async def find_tasks_by_metadata(self, arguments: dict) -> dict:
        offset = arguments.get('offset', 0)
        page = await self.backend.find_tasks_by_metadata(
            arguments['project_root'], arguments['key'],
            limit=arguments['limit'], offset=offset,
        )
        results = page['results']
        return {
            'results': results,
            'pagination': {
                'has_more': offset + len(results) < page['total'],
                'returned': len(results),
            },
        }

    async def get_task(self, arguments: dict) -> dict:
        return await self.backend.get_task(arguments['id'], project_root=arguments['project_root'])

    async def update_task(self, arguments: dict) -> dict:
        metadata = json.loads(arguments['metadata'])
        metadata.pop('allow_mcp_markup', None)
        return dict(await self.backend.update_task(
            arguments['id'], project_root=arguments['project_root'],
            metadata=json.dumps(metadata), metadata_mode=arguments['metadata_mode'],
        ))


class LookupErrorClient(BackendToolClient):
    async def find_tasks_by_metadata(self, arguments: dict) -> dict:
        return {'error': 'store unavailable', 'error_type': 'TaskBackendUnavailable'}


class RefusingWriteClient(BackendToolClient):
    async def update_task(self, arguments: dict) -> dict:
        return {'error': 'refused', 'error_type': 'LockCharterViolation'}


class RaisingWriteClient(BackendToolClient):
    """Refuses one task's write by raising, as the live client does on an ``isError`` reply."""

    def __init__(self, backend: SqliteTaskBackend, refused_id: str):
        super().__init__(backend)
        self.refused_id = refused_id

    async def call_tool(self, name: str, arguments: dict) -> dict:
        if name == 'update_task' and arguments['id'] == self.refused_id:
            raise RuntimeError('update_task failed: mcp_markup_unrepairable')
        return await super().call_tool(name, arguments)


class SilentlyDroppedWriteClient(BackendToolClient):
    async def update_task(self, arguments: dict) -> dict:
        return {'id': arguments['id'], 'message': 'updated'}


@pytest_asyncio.fixture
async def backend(tmp_path):
    started = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await started.start()
    yield started
    await started.close()


@pytest_asyncio.fixture
async def enforcing_backend(tmp_path):
    """A second backend over the same store, refusing a malformed write as the live server does."""
    started = SqliteTaskBackend(
        TaskmasterConfig(project_root=str(tmp_path)), task_metadata_enforce=True,
    )
    await started.start()
    yield started
    await started.close()


@pytest.fixture
def root(tmp_path) -> str:
    return str(tmp_path / 'proj')


@pytest.fixture
def client(backend) -> BackendToolClient:
    return BackendToolClient(backend)


async def _seed(
    backend: SqliteTaskBackend, root: str, meta: dict[str, Any], *, status: str = 'done',
) -> str:
    """Every live target is done or cancelled, and a done row carries no ``pending_since``.

    Titles are unique because the store refuses a duplicate (title, files) key.
    """
    added = await backend.add_task(
        project_root=root, title=f'seed {uuid.uuid4().hex}', metadata=json.dumps(meta),
        status=status,
    )
    return str(added['id'])


async def _stored(backend: SqliteTaskBackend, root: str, task_id: str) -> dict[str, Any]:
    return (await backend.get_task(task_id, project_root=root))['metadata']


@pytest_asyncio.fixture
async def seeded(backend, root) -> dict[str, str]:
    return {
        'both': await _seed(backend, root, BOTH_COUNTERS_META),
        'conflicting': await _seed(backend, root, CONFLICTING_LEDGER_META),
        'clean': await _seed(backend, root, CLEAN_META),
    }


@pytest.mark.asyncio
async def test_seeded_blob_is_stored_raw(backend, root, seeded):
    stored = await _stored(backend, root, seeded['both'])
    assert stored[FAILURES] == 3
    assert stored[ITERATION] == 7
    assert 'retry_ledger' not in stored


@pytest.mark.asyncio
async def test_dry_run_writes_nothing(backend, root, seeded, client):
    before = {name: await _stored(backend, root, tid) for name, tid in seeded.items()}

    assert await run(client, [root], apply=False, backup_path=None) == 0

    assert 'update_task' not in client.tool_names
    assert {name: await _stored(backend, root, tid) for name, tid in seeded.items()} == before


@pytest.mark.asyncio
async def test_apply_nests_counters_backs_up_and_is_idempotent(
    backend, root, seeded, client, tmp_path,
):
    before = {name: await _stored(backend, root, tid) for name, tid in seeded.items()}
    backup_path = tmp_path / 'backup.json'

    assert await run(client, [root], apply=True, backup_path=backup_path) == 0

    after = {name: await _stored(backend, root, tid) for name, tid in seeded.items()}
    for name in ('both', 'conflicting'):
        assert FAILURES not in after[name]
        assert ITERATION not in after[name]
        assert after[name] == nest_counters(before[name]).after
    assert after['both']['retry_ledger'] == {FAILURES: 3, ITERATION: 7}
    assert after['both']['done_provenance'] == DONE_PROVENANCE
    assert after['both']['files'] == ['a.py']
    assert after['conflicting']['retry_ledger'] == {
        FAILURES: 2, ITERATION: 4, 'merge_first_enqueued_at': ENQUEUED_AT,
    }
    assert after['clean'] == before['clean']

    assert json.loads(backup_path.read_text()) == {
        root: {seeded['both']: before['both'], seeded['conflicting']: before['conflicting']},
    }

    rerun_client = BackendToolClient(backend)
    assert await run(
        rerun_client, [root], apply=True, backup_path=tmp_path / 'rerun-backup.json',
    ) == 0
    assert 'update_task' not in rerun_client.tool_names


@pytest.mark.asyncio
async def test_every_page_of_targets_is_migrated(backend, root, client, tmp_path):
    target_ids = [await _seed(backend, root, BOTH_COUNTERS_META) for _ in range(3)]

    assert await run(
        client, [root], apply=True, backup_path=tmp_path / 'backup.json', page_size=1,
    ) == 0

    for task_id in target_ids:
        stored = await _stored(backend, root, task_id)
        assert stored['retry_ledger'] == {FAILURES: 3, ITERATION: 7}
        assert FAILURES not in stored


@pytest.mark.asyncio
async def test_every_named_root_is_migrated(backend, root, client, tmp_path):
    other_root = str(tmp_path / 'other')
    first = await _seed(backend, root, BOTH_COUNTERS_META)
    second = await _seed(backend, other_root, CONFLICTING_LEDGER_META)
    backup_path = tmp_path / 'backup.json'

    assert await run(client, [root, other_root], apply=True, backup_path=backup_path) == 0

    assert FAILURES not in await _stored(backend, root, first)
    assert FAILURES not in await _stored(backend, other_root, second)
    assert set(json.loads(backup_path.read_text())) == {root, other_root}


@pytest.mark.asyncio
async def test_apply_without_a_backup_path_is_refused(backend, root, seeded, client):
    assert await run(client, [root], apply=True, backup_path=None) != 0

    assert 'update_task' not in client.tool_names
    assert FAILURES in await _stored(backend, root, seeded['both'])


@pytest.mark.asyncio
async def test_apply_onto_an_existing_backup_is_refused(
    backend, root, seeded, client, tmp_path,
):
    backup_path = tmp_path / 'backup.json'
    backup_path.write_text('earlier run')

    assert await run(client, [root], apply=True, backup_path=backup_path) != 0

    assert 'update_task' not in client.tool_names
    assert backup_path.read_text() == 'earlier run'
    assert FAILURES in await _stored(backend, root, seeded['both'])


@pytest.mark.asyncio
async def test_malformed_ledger_is_reported_without_blocking_the_rest(
    backend, root, client, tmp_path,
):
    malformed_meta = {FAILURES: 1, 'retry_ledger': 'oops'}
    malformed = await _seed(backend, root, malformed_meta)
    healthy = await _seed(backend, root, BOTH_COUNTERS_META)

    assert await run(client, [root], apply=True, backup_path=tmp_path / 'backup.json') != 0

    assert await _stored(backend, root, malformed) == malformed_meta
    assert (await _stored(backend, root, healthy))['retry_ledger'] == {FAILURES: 3, ITERATION: 7}


@pytest.mark.asyncio
async def test_blob_a_replace_write_would_strip_is_refused_before_writing(
    backend, root, client, tmp_path,
):
    anchored = await _seed(backend, root, BOTH_COUNTERS_META, status='pending')
    anchored_before = await _stored(backend, root, anchored)
    assert 'pending_since' in anchored_before
    healthy = await _seed(backend, root, BOTH_COUNTERS_META)

    assert await run(client, [root], apply=True, backup_path=tmp_path / 'backup.json') != 0

    assert [args['id'] for name, args in client.calls if name == 'update_task'] == [healthy]
    assert await _stored(backend, root, anchored) == anchored_before
    assert FAILURES not in await _stored(backend, root, healthy)


@pytest.mark.asyncio
async def test_lookup_error_reply_is_never_read_as_no_match(backend, root, seeded, tmp_path):
    client = LookupErrorClient(backend)

    assert await run(client, [root], apply=True, backup_path=tmp_path / 'backup.json') != 0

    assert 'update_task' not in client.tool_names


@pytest.mark.asyncio
async def test_refused_write_is_a_failure(backend, root, seeded, tmp_path):
    client = RefusingWriteClient(backend)

    assert await run(client, [root], apply=True, backup_path=tmp_path / 'backup.json') != 0


@pytest.mark.asyncio
async def test_write_that_did_not_land_fails_the_read_back_naming_the_backup(
    backend, root, seeded, tmp_path, capsys,
):
    client = SilentlyDroppedWriteClient(backend)
    backup_path = tmp_path / 'backup.json'

    assert await run(client, [root], apply=True, backup_path=backup_path) != 0

    captured = capsys.readouterr()
    assert str(backup_path) in captured.out + captured.err


@pytest.mark.asyncio
async def test_raised_write_is_a_problem_and_the_rest_still_migrate(
    backend, root, seeded, tmp_path, capsys,
):
    client = RaisingWriteClient(backend, seeded['both'])

    assert await run(client, [root], apply=True, backup_path=tmp_path / 'backup.json') != 0

    assert FAILURES in await _stored(backend, root, seeded['both'])
    conflicting = await _stored(backend, root, seeded['conflicting'])
    assert FAILURES not in conflicting and ITERATION not in conflicting
    assert f"#{seeded['both']}: update_task raised" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_null_counter_row_migrates_through_an_enforcing_store(
    backend, enforcing_backend, root, tmp_path, capsys,
):
    null_row = await _seed(backend, root, NULL_COUNTER_META)
    with pytest.raises(ValidationError):
        await enforcing_backend.update_task(
            null_row, project_root=root, metadata_mode='replace', metadata=json.dumps({
                'done_provenance': DONE_PROVENANCE,
                'retry_ledger': {FAILURES: None, ITERATION: 6},
            }),
        )

    client = BackendToolClient(enforcing_backend)
    assert await run(client, [root], apply=True, backup_path=tmp_path / 'backup.json') == 0

    assert await _stored(enforcing_backend, root, null_row) == {
        'done_provenance': DONE_PROVENANCE, 'retry_ledger': {ITERATION: 6},
    }
    assert f'dropped={[FAILURES]}' in capsys.readouterr().out
