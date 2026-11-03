"""The live store as the link-heal executor sees it (plans/write-triage-link-healing-prd.md H1).

Every record value the executor hashes or compares is read through the running
server's ``get_memory_by_id`` tool, and every heal is written through its
``update_memory`` tool, so the server's authorization gate and write journal
apply unchanged. :class:`LinkHealStore` is that port, over an injected
:data:`ToolCaller`: production passes :func:`mcp_tool_caller`, tests pass an
in-process transport. :func:`server_tool_caller` opens the production one for
a server URL, for every script that reads the live store.

One read bypasses the server: no tool enumerates the records that carry a
``parent_id``, so :class:`QdrantLinkCensus` scrolls each project's collection
directly for their ids, and only their ids.

This is the bottom layer of the link-heal modules and imports none of them.
"""

from __future__ import annotations

import contextlib
import hashlib
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from qdrant_client.models import Filter, IsEmptyCondition, PayloadField
from shared.mcp_post import call_mcp_tool, open_mcp_client

from fused_memory.models.scope import Scope
from fused_memory.server.grouped_read import PARENT_ID_KEY

if TYPE_CHECKING:
    from fused_memory.backends.mem0_client import Mem0Backend

#: An MCP ``tools/call``: the tool's reply dict, or ``None`` when none arrived.
ToolCaller = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | None]]

READ_TOOL = 'get_memory_by_id'
COUNT_TOOL = 'count_memories_by_metadata'
WRITE_TOOL = 'update_memory'

#: The id :meth:`LinkHealStore.probe` reads: no record has it, so any answer
#: carrying ``found`` proves the server reads the project.
PROBE_MEMORY_ID = str(uuid.UUID(int=0))


def text_sha256(text: str) -> str:
    """The hash the hand-link corpus took: sha256 over the UTF-8 bytes of a
    record's ``get_memory_by_id`` ``content``."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class LiveRecord:
    """One record as the server returned it just now."""

    memory_id: str
    text: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))

    @property
    def text_sha256(self) -> str:
        return text_sha256(self.text)


@dataclass(frozen=True)
class MetadataChange:
    """One ``update_memory`` metadata write: a patch or a delete, never both.

    A combined call takes the server's read-modify-overwrite route, and a
    ``None`` value stores a null rather than removing the key, so both are
    refused at construction.
    """

    patch: Mapping[str, Any] | None = None
    delete_keys: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.patch is not None and self.delete_keys is not None:
            raise ValueError('a MetadataChange carries both a patch and a delete list')
        if self.patch is None and self.delete_keys is None:
            raise ValueError('a MetadataChange carries neither a patch nor a delete list')
        if self.patch is not None:
            self._check_patch(self.patch)
            object.__setattr__(self, 'patch', MappingProxyType(dict(self.patch)))
        else:
            keys = tuple(self.delete_keys or ())
            if not keys:
                raise ValueError('a MetadataChange delete list is empty')
            object.__setattr__(self, 'delete_keys', keys)

    @staticmethod
    def _check_patch(patch: Mapping[str, Any]) -> None:
        if not patch:
            raise ValueError('a MetadataChange patch is empty')
        none_keys = sorted(key for key, value in patch.items() if value is None)
        if none_keys:
            raise ValueError(
                f'a MetadataChange patch sets None for {none_keys}; None is never '
                'patched (it stores a null) — remove the key with a delete instead',
            )

    @classmethod
    def patch_only(cls, patch: Mapping[str, Any]) -> MetadataChange:
        return cls(patch=patch)

    @classmethod
    def delete_only(cls, keys: Iterable[str]) -> MetadataChange:
        return cls(delete_keys=tuple(keys))

    def as_update_arguments(self) -> dict[str, Any]:
        """The one ``update_memory`` argument this change fills."""
        if self.patch is not None:
            return {'metadata_patch': dict(self.patch)}
        return {'metadata_delete_keys': list(self.delete_keys or ())}


@dataclass(frozen=True)
class WriteReply:
    """What one ``update_memory`` call came to. ``landed`` False is a failed write."""

    landed: bool
    error_type: str | None = None
    error: str | None = None


class StoreCallFailed(Exception):
    """A tool call that produced no usable answer. Unknown is never absent."""

    def __init__(self, tool: str, memory_id: str, error_type: str, detail: str = '') -> None:
        self.tool = tool
        self.memory_id = memory_id
        self.error_type = error_type
        self.detail = detail
        super().__init__(f'{tool} for {memory_id} failed with {error_type}: {detail}')


class StoreReadFailed(StoreCallFailed):
    """A read that cannot say whether the record exists."""


class StoreUnreachable(StoreCallFailed):
    """The server did not answer the start-of-run probe."""


class LinkHealStore:
    """Reads and writes record state only through the server's tools."""

    def __init__(self, call_tool: ToolCaller) -> None:
        self._call_tool = call_tool

    async def probe(self, project_id: str) -> None:
        """Raise :class:`StoreUnreachable` unless the server reads *project_id*."""
        try:
            await self.read(project_id, PROBE_MEMORY_ID)
        except StoreReadFailed as failure:
            raise StoreUnreachable(
                failure.tool, failure.memory_id, failure.error_type, failure.detail,
            ) from failure

    async def read(self, project_id: str, memory_id: str) -> LiveRecord | None:
        """The record, ``None`` on a genuine miss, :class:`StoreReadFailed` otherwise."""
        reply = await self._ask(
            READ_TOOL, {'project_id': project_id, 'memory_id': memory_id}, memory_id,
        )
        if 'found' not in reply:
            raise StoreReadFailed(READ_TOOL, memory_id, 'MalformedReply', 'no found key')
        if not reply['found']:
            return None
        text, metadata = reply.get('content'), reply.get('metadata')
        if not isinstance(text, str) or not isinstance(metadata, Mapping):
            raise StoreReadFailed(
                READ_TOOL, memory_id, 'MalformedReply', 'content or metadata missing',
            )
        return LiveRecord(memory_id=memory_id, text=text, metadata=metadata)

    async def count_children(self, project_id: str, memory_id: str) -> int:
        """How many records carry ``parent_id`` = *memory_id*."""
        reply = await self._ask(
            COUNT_TOOL,
            {'project_id': project_id, 'filters': {PARENT_ID_KEY: memory_id}},
            memory_id,
        )
        count = reply.get('count')
        if not isinstance(count, int) or isinstance(count, bool):
            raise StoreReadFailed(COUNT_TOOL, memory_id, 'MalformedReply', f'count={count!r}')
        return count

    async def write(
        self,
        project_id: str,
        memory_id: str,
        change: MetadataChange,
        *,
        agent_id: str,
        causation_id: str,
        reason: str,
    ) -> WriteReply:
        """Send *change* as exactly one ``update_memory`` call."""
        arguments = {
            'memory_id': memory_id,
            'store': 'mem0',
            'project_id': project_id,
            **change.as_update_arguments(),
            'reason': reason,
            'agent_id': agent_id,
            'metadata': {'_causation_id': causation_id},
        }
        try:
            reply = await self._call_tool(WRITE_TOOL, arguments)
        except Exception as exc:  # noqa: BLE001 -- a raised write is a failed write
            return WriteReply(landed=False, error_type=type(exc).__name__, error=str(exc))
        return _write_reply(reply)

    async def _ask(
        self, tool: str, arguments: dict[str, Any], memory_id: str,
    ) -> dict[str, Any]:
        try:
            reply = await self._call_tool(tool, arguments)
        except Exception as exc:  # noqa: BLE001 -- unknown is not absent
            raise StoreReadFailed(tool, memory_id, type(exc).__name__, str(exc)) from exc
        if reply is None:
            raise StoreReadFailed(tool, memory_id, 'NoReply', 'the server sent no tool reply')
        if reply.get('error_type'):
            raise StoreReadFailed(
                tool, memory_id, str(reply['error_type']), str(reply.get('error', '')),
            )
        return reply


def _write_reply(reply: dict[str, Any] | None) -> WriteReply:
    if reply is None:
        return WriteReply(
            landed=False, error_type='NoReply', error='the server sent no tool reply',
        )
    if reply.get('error_type'):
        return WriteReply(
            landed=False,
            error_type=str(reply['error_type']),
            error=str(reply.get('error', '')),
        )
    if reply.get('status') != 'updated':
        return WriteReply(landed=False, error_type='UnexpectedReply', error=repr(reply)[:200])
    return WriteReply(landed=True)


def mcp_tool_caller(client: Any, base_url: str) -> ToolCaller:
    """The production :data:`ToolCaller`: one ``call_mcp_tool`` per call."""

    async def call(tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        return await call_mcp_tool(
            client, base_url, tool, arguments, context=f'link-heal {tool}',
        )

    return call


@contextlib.asynccontextmanager
async def server_tool_caller(server_url: str) -> AsyncIterator[ToolCaller]:
    """A :func:`mcp_tool_caller` over a client open for the context's lifetime."""
    async with open_mcp_client() as client:
        yield mcp_tool_caller(client, server_url)


class CensusFailed(Exception):
    """A census that could not enumerate a project's linked records."""

    def __init__(self, project_id: str, error_type: str, detail: str = '') -> None:
        self.project_id = project_id
        self.error_type = error_type
        self.detail = detail
        super().__init__(f'the link census of {project_id} failed with {error_type}: {detail}')


class LinkCensus(Protocol):
    """Enumerates the ids of a project's records that carry a ``parent_id``.

    One that cannot raises :class:`CensusFailed`.
    """

    async def linked_ids(self, project_id: str) -> list[str]: ...


class QdrantLinkCensus:
    """A read-only scroll of a project's collection for linked record ids."""

    def __init__(self, backend: Mem0Backend, collection_prefix: str) -> None:
        self._backend = backend
        self._collection_prefix = collection_prefix

    async def linked_ids(self, project_id: str) -> list[str]:
        collection = Scope(project_id=project_id).mem0_collection_name(
            self._collection_prefix,
        )
        has_parent = Filter(
            must_not=[IsEmptyCondition(is_empty=PayloadField(key=PARENT_ID_KEY))],
        )
        points: AsyncIterator[Any] = self._backend.scroll_collection_pages(
            collection, scroll_filter=has_parent,
        )
        try:
            return [str(point.id) async for point in points]
        except Exception as exc:  # noqa: BLE001 -- a partial census is no census
            raise CensusFailed(project_id, type(exc).__name__, str(exc)) from exc
