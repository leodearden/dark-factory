"""The link-heal executor's port onto the live store (task 6181, PRD H1).

Most cases drive ``LinkHealStore`` through a scripted ToolCaller, the same
seam production fills with ``shared.mcp_post.call_mcp_tool``. One integration
class runs ``QdrantLinkCensus`` against a throwaway per-worker collection.
"""

from __future__ import annotations

import contextlib
import hashlib
import uuid
from typing import Any

import pytest
from _fm_helpers import QDRANT_URL, ensure_fresh_collection, qdrant_skipif

from fused_memory.maintenance.link_heal_store import (
    LinkHealStore,
    LiveRecord,
    MetadataChange,
    QdrantLinkCensus,
    StoreReadFailed,
    StoreUnreachable,
    WriteReply,
    text_sha256,
)

PROJECT = 'dark_factory'
CHILD = '11111111-1111-1111-1111-111111111111'
NIL_UUID = '00000000-0000-0000-0000-000000000000'


class ScriptedCaller:
    """A ToolCaller answering from a script; an exception in it is raised."""

    def __init__(self, *replies: Any) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, tool: str, arguments: dict[str, Any]) -> dict | None:
        self.calls.append((tool, arguments))
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _found(text: str, **meta: Any) -> dict[str, Any]:
    return {'found': True, 'memory_id': CHILD, 'content': text, 'metadata': meta}


def test_text_sha256_hashes_the_utf8_bytes():
    assert text_sha256('é') == hashlib.sha256('é'.encode()).hexdigest()


class TestMetadataChange:
    def test_patch_only_constructs(self):
        change = MetadataChange.patch_only({'kind': 'amendment'})
        assert change.patch == {'kind': 'amendment'}
        assert change.delete_keys is None

    def test_delete_only_constructs(self):
        change = MetadataChange.delete_only(['parent_id', 'kind'])
        assert change.delete_keys == ('parent_id', 'kind')
        assert change.patch is None

    def test_both_arms_is_rejected(self):
        with pytest.raises(ValueError, match='both'):
            MetadataChange(patch={'kind': 'amendment'}, delete_keys=('parent_id',))

    def test_neither_arm_is_rejected(self):
        with pytest.raises(ValueError, match='neither'):
            MetadataChange(patch=None, delete_keys=None)

    def test_empty_patch_is_rejected(self):
        with pytest.raises(ValueError, match='empty'):
            MetadataChange.patch_only({})

    def test_empty_delete_list_is_rejected(self):
        with pytest.raises(ValueError, match='empty'):
            MetadataChange.delete_only([])

    def test_a_none_patch_value_is_rejected(self):
        with pytest.raises(ValueError, match='None'):
            MetadataChange.patch_only({'kind': 'amendment', 'parent_id': None})


class TestRead:
    @pytest.mark.asyncio
    async def test_a_found_reply_is_a_live_record(self):
        caller = ScriptedCaller(_found('child text', parent_id='p', kind='amendment'))

        record = await LinkHealStore(caller).read(PROJECT, CHILD)

        assert record == LiveRecord(
            memory_id=CHILD,
            text='child text',
            metadata={'parent_id': 'p', 'kind': 'amendment'},
        )
        assert record.text_sha256 == text_sha256('child text')
        assert caller.calls == [
            ('get_memory_by_id', {'project_id': PROJECT, 'memory_id': CHILD}),
        ]

    @pytest.mark.asyncio
    async def test_a_genuine_miss_is_none(self):
        caller = ScriptedCaller({'found': False, 'memory_id': CHILD})
        assert await LinkHealStore(caller).read(PROJECT, CHILD) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('reply', 'named'),
        [
            ({'error': 'read timed out', 'error_type': 'TimeoutError'}, 'TimeoutError'),
            (None, 'get_memory_by_id'),
            ({'memory_id': CHILD}, 'get_memory_by_id'),
            (ConnectionError('refused'), 'ConnectionError'),
        ],
        ids=['error-reply', 'no-reply', 'no-found-key', 'caller-raises'],
    )
    async def test_an_unknown_answer_is_a_failed_read_never_a_miss(self, reply, named):
        with pytest.raises(StoreReadFailed) as excinfo:
            await LinkHealStore(ScriptedCaller(reply)).read(PROJECT, CHILD)

        failure = excinfo.value
        assert failure.tool == 'get_memory_by_id'
        assert failure.memory_id == CHILD
        message = str(failure)
        assert 'get_memory_by_id' in message
        assert CHILD in message
        assert named in message


class TestCountChildren:
    @pytest.mark.asyncio
    async def test_counts_records_pointing_at_the_id(self):
        caller = ScriptedCaller({'count': 2, 'project_id': PROJECT})

        assert await LinkHealStore(caller).count_children(PROJECT, CHILD) == 2
        assert caller.calls == [
            (
                'count_memories_by_metadata',
                {'project_id': PROJECT, 'filters': {'parent_id': CHILD}},
            ),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'reply',
        [
            {'error': 'x', 'error_type': 'TimeoutError'},
            {'count': '2'},
            {'count': True},
            None,
        ],
        ids=['error-reply', 'string-count', 'bool-count', 'no-reply'],
    )
    async def test_an_unusable_count_is_a_failed_read(self, reply):
        with pytest.raises(StoreReadFailed) as excinfo:
            await LinkHealStore(ScriptedCaller(reply)).count_children(PROJECT, CHILD)
        assert excinfo.value.tool == 'count_memories_by_metadata'
        assert excinfo.value.memory_id == CHILD


class TestWrite:
    RUN_ID = 'a' * 32

    async def _write(self, caller, change):
        return await LinkHealStore(caller).write(
            PROJECT, CHILD, change,
            agent_id='link-heal-aaaaaaaa',
            causation_id=self.RUN_ID,
            reason='link-heal r=aaaaaaaa a=detach',
        )

    @pytest.mark.asyncio
    async def test_a_delete_only_change_sends_exactly_one_delete_call(self):
        caller = ScriptedCaller({'status': 'updated', 'id': CHILD})

        reply = await self._write(caller, MetadataChange.delete_only(['parent_id', 'kind']))

        assert reply == WriteReply(landed=True, error_type=None, error=None)
        assert caller.calls == [
            (
                'update_memory',
                {
                    'memory_id': CHILD,
                    'store': 'mem0',
                    'project_id': PROJECT,
                    'metadata_delete_keys': ['parent_id', 'kind'],
                    'reason': 'link-heal r=aaaaaaaa a=detach',
                    'agent_id': 'link-heal-aaaaaaaa',
                    'metadata': {'_causation_id': self.RUN_ID},
                },
            ),
        ]

    @pytest.mark.asyncio
    async def test_a_patch_only_change_sends_exactly_one_patch_call(self):
        caller = ScriptedCaller({'status': 'updated', 'id': CHILD})

        await self._write(caller, MetadataChange.patch_only({'kind': 'amendment'}))

        (tool, arguments), = caller.calls
        assert tool == 'update_memory'
        assert arguments['metadata_patch'] == {'kind': 'amendment'}
        assert 'metadata_delete_keys' not in arguments

    @pytest.mark.asyncio
    async def test_an_error_type_reply_is_a_failed_write(self):
        caller = ScriptedCaller(
            {'error': 'not authorized', 'error_type': 'Mem0UpdateNotAuthorized'},
        )

        reply = await self._write(caller, MetadataChange.patch_only({'kind': 'amendment'}))

        assert reply == WriteReply(
            landed=False, error_type='Mem0UpdateNotAuthorized', error='not authorized',
        )

    @pytest.mark.asyncio
    async def test_no_reply_is_a_failed_write(self):
        reply = await self._write(
            ScriptedCaller(None), MetadataChange.patch_only({'kind': 'amendment'}),
        )
        assert reply.landed is False
        assert reply.error_type

    @pytest.mark.asyncio
    async def test_a_raising_caller_is_a_failed_write_naming_the_exception(self):
        reply = await self._write(
            ScriptedCaller(ConnectionError('refused')),
            MetadataChange.patch_only({'kind': 'amendment'}),
        )
        assert reply.landed is False
        assert reply.error_type == 'ConnectionError'


class TestProbe:
    @pytest.mark.asyncio
    async def test_a_miss_on_the_nil_uuid_proves_the_server_answers(self):
        caller = ScriptedCaller({'found': False, 'memory_id': NIL_UUID})

        await LinkHealStore(caller).probe(PROJECT)

        assert caller.calls == [
            ('get_memory_by_id', {'project_id': PROJECT, 'memory_id': NIL_UUID}),
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'reply',
        [ConnectionError('refused'), None, {'error': 'x', 'error_type': 'ValidationError'}],
        ids=['caller-raises', 'no-reply', 'error-reply'],
    )
    async def test_anything_else_is_unreachable(self, reply):
        with pytest.raises(StoreUnreachable):
            await LinkHealStore(ScriptedCaller(reply)).probe(PROJECT)


class _Point:
    def __init__(self, point_id: Any) -> None:
        self.id = point_id


class _FakeScrollBackend:
    def __init__(self, points: list[_Point]) -> None:
        self._points = points
        self.scrolls: list[tuple[str, Any]] = []

    async def scroll_collection_pages(self, collection_name: str, *, scroll_filter: Any = None):
        self.scrolls.append((collection_name, scroll_filter))
        for point in self._points:
            yield point


class TestQdrantLinkCensus:
    @pytest.mark.asyncio
    async def test_yields_linked_ids_as_strings_in_order(self):
        as_uuid = uuid.UUID('22222222-2222-2222-2222-222222222222')
        backend = _FakeScrollBackend([_Point(as_uuid), _Point(CHILD)])

        ids = await QdrantLinkCensus(backend, 'fused').linked_ids(PROJECT)

        assert ids == [str(as_uuid), CHILD]

    @pytest.mark.asyncio
    async def test_scrolls_the_projects_collection_for_records_with_a_parent(self):
        from qdrant_client.models import IsEmptyCondition

        backend = _FakeScrollBackend([])

        await QdrantLinkCensus(backend, 'fused').linked_ids(PROJECT)

        (collection, scroll_filter), = backend.scrolls
        assert collection == 'fused_dark_factory'
        (condition,) = scroll_filter.must_not
        assert isinstance(condition, IsEmptyCondition)
        assert condition.is_empty.key == 'parent_id'


@qdrant_skipif()
@pytest.mark.integration
@pytest.mark.timeout(60)
class TestQdrantLinkCensusAgainstQdrant:
    VECTOR_DIM = 8
    PROJECT = 'census_probe'

    @pytest.fixture
    def prefix(self, worker_id: str) -> str:
        return f'_test_link_heal_census_{worker_id}'

    @pytest.fixture
    def seeded(self, prefix: str):
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct

        collection = f'{prefix}_{self.PROJECT}'
        client = QdrantClient(url=QDRANT_URL, timeout=10)
        ensure_fresh_collection(client, collection, size=self.VECTOR_DIM)
        linked = str(uuid.uuid4())
        client.upsert(
            collection_name=collection,
            points=[
                PointStruct(id=linked, vector=[0.1] * self.VECTOR_DIM,
                            payload={'data': 'a', 'parent_id': str(uuid.uuid4())}),
                PointStruct(id=str(uuid.uuid4()), vector=[0.1] * self.VECTOR_DIM,
                            payload={'data': 'b'}),
                PointStruct(id=str(uuid.uuid4()), vector=[0.1] * self.VECTOR_DIM,
                            payload={'data': 'c', 'parent_id': None}),
            ],
        )
        yield linked
        with contextlib.suppress(Exception):
            client.delete_collection(collection)
        client.close()

    @pytest.mark.asyncio
    async def test_only_records_carrying_a_parent_id_are_linked(
        self, mock_config, prefix, seeded,
    ):
        from fused_memory.backends.mem0_client import Mem0Backend

        mock_config.mem0.qdrant_url = QDRANT_URL
        backend = Mem0Backend(mock_config)
        try:
            ids = await QdrantLinkCensus(backend, prefix).linked_ids(self.PROJECT)
        finally:
            await backend.close()

        assert ids == [seeded]
