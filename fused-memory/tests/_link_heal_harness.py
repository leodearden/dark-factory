"""The link-heal executor's in-process server (task 6181, PRD H1 boundary tests).

A REAL ``MemoryService`` and MCP server over :class:`FakeMem0`, a stateful
stand-in for ``Mem0Backend``, with a real ``WriteJournal`` on ``tmp_path``.
:func:`in_process_tool_caller` is the fake MCP transport: it calls the server's
own tools, so the ``update_memory`` authorization gate, its payload routes, the
grouped reads and the write journal all run for real. No real collection is
ever touched.

``FakeMem0.overwrite_payload`` and ``FakeMem0.update`` raise, which pins at the
store that a patch and a delete are never combined and content is never sent.
The server turns that raise into an error reply, so each also counts its calls
in ``FakeMem0.writes`` under its own name, where a test can see it.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from _fm_helpers import install_identity_mocks

from fused_memory.backends.mem0_client import split_managed_metadata
from fused_memory.config.schema import Mem0UpdateConfig
from fused_memory.maintenance.link_heal import LINK_KEYS, BasisSource, LinkBasis, Verdict
from fused_memory.maintenance.link_heal_executor import (
    Escape,
    EscapeFiler,
    RunLimits,
    RunReport,
    run_apply,
    run_plan,
)
from fused_memory.maintenance.link_heal_ledger import LinkHealLedger, RunSource
from fused_memory.maintenance.link_heal_store import LinkHealStore, ToolCaller, text_sha256
from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.models.scope import Scope
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CONTESTED_METADATA_KEY,
    PARENT_ID_KEY,
)
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import MemoryService, SearchResults
from fused_memory.services.write_journal import WriteJournal

ALL_LINK_HEAL_PREFIXES = ['recon-stage-', 'curator-', 'link-heal-']
WITHOUT_LINK_HEAL_PREFIX = ['recon-stage-', 'curator-']

WRITE_ROUTES = ('set_payload', 'delete_payload')
FORBIDDEN_ROUTES = ('overwrite_payload', 'update')

DF = 'dark_factory'
REIFY = 'reify'
PROJECTS = (DF, REIFY)

CHILD = '11111111-1111-4111-8111-111111111111'
PARENT = '22222222-2222-4222-8222-222222222222'
CHILD_TEXT = 'the child note about link healing'
PARENT_TEXT = 'the parent note about link healing'

DEFAULT_LIMITS = RunLimits(max_actions_per_run=25, backlog_multiplier=5, write_failure_streak=3)


def corpus_basis(
    verdict: str,
    *,
    child: str = CHILD,
    parent: str = PARENT,
    child_text: str = CHILD_TEXT,
    parent_text: str = PARENT_TEXT,
    key: str = 'H001',
    project: str = DF,
) -> LinkBasis:
    """A hand-link corpus verdict on (child, parent) at these texts' hashes."""
    return LinkBasis(
        project_id=project,
        child_id=child,
        parent_id=parent,
        verdict=Verdict(verdict),
        child_sha256=text_sha256(child_text),
        parent_sha256=text_sha256(parent_text),
        source=BasisSource.CORPUS,
        key=key,
    )


def _matches(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(key in payload and payload[key] == value for key, value in filters.items())


class FakeMem0:
    """Mem0 points keyed by (project, memory id); also a ``LinkCensus``."""

    def __init__(self) -> None:
        self.points: dict[tuple[str, str], dict[str, Any]] = {}
        self.read_failures: set[str] = set()
        self.writes: Counter[str] = Counter()

    @property
    def write_count(self) -> int:
        return sum(self.writes[route] for route in WRITE_ROUTES)

    def payload(self, project_id: str, memory_id: str) -> dict[str, Any]:
        return self.points[(project_id, memory_id)]

    async def get_point_by_id(self, memory_id: str, scope: Scope) -> dict[str, Any] | None:
        if memory_id in self.read_failures:
            raise TimeoutError(f'qdrant read timed out for {memory_id}')
        payload = self.points.get((scope.project_id, memory_id))
        return None if payload is None else dict(payload)

    async def set_payload(self, memory_id: str, payload: dict[str, Any], scope: Scope) -> None:
        self.writes['set_payload'] += 1
        self.points[(scope.project_id, memory_id)].update(payload)

    async def delete_payload(self, memory_id: str, keys: list[str], scope: Scope) -> None:
        self.writes['delete_payload'] += 1
        point = self.points[(scope.project_id, memory_id)]
        for key in keys:
            point.pop(key, None)

    async def overwrite_payload(self, *args: Any, **kwargs: Any) -> None:
        self.writes['overwrite_payload'] += 1
        raise AssertionError('link-heal combined a patch and a delete (overwrite_payload)')

    async def update(self, *args: Any, **kwargs: Any) -> None:
        self.writes['update'] += 1
        raise AssertionError('link-heal sent content (mem0.update)')

    def _rows(self, scope: Scope, filters: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        return [
            (memory_id, payload)
            for (project_id, memory_id), payload in self.points.items()
            if project_id == scope.project_id and _matches(payload, filters)
        ]

    async def count_by_metadata(self, scope: Scope, filters: dict[str, Any]) -> int:
        return len(self._rows(scope, filters))

    async def scroll_by_metadata(
        self, scope: Scope, filters: dict[str, Any], limit: int = 1000, *, with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        rows = [
            {'id': memory_id, 'created_at': payload.get('created_at'), 'metadata': dict(payload)}
            for memory_id, payload in self._rows(scope, filters)
        ]
        return rows[:limit]

    async def linked_ids(self, project_id: str) -> list[str]:
        return [
            memory_id
            for (project, memory_id), payload in self.points.items()
            if project == project_id and payload.get(PARENT_ID_KEY) not in (None, [])
        ]

    def search_results(self, query: str, project_id: str, limit: int) -> SearchResults:
        """Every point in *project_id* whose text contains *query*, as mem0 hits."""
        needle = query.lower()
        hits = [
            _memory_result(memory_id, payload)
            for (project, memory_id), payload in self.points.items()
            if project == project_id and needle in str(payload.get('data', '')).lower()
        ]
        return SearchResults(hits[:limit])


def _memory_result(memory_id: str, payload: dict[str, Any]) -> MemoryResult:
    _managed, custom = split_managed_metadata(dict(payload))
    return MemoryResult(
        id=memory_id,
        content=str(payload.get('data', '')),
        category=MemoryCategory(payload['category']),
        source_store=SourceStore.mem0,
        relevance_score=0.9,
        metadata=custom,
        created_at=payload.get('created_at'),
    )


def parse_tool_reply(result: Any) -> dict[str, Any]:
    """A FastMCP ``call_tool`` result as the JSON a wire client would decode."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return json.loads(json.dumps(result))


def in_process_tool_caller(server: Any) -> ToolCaller:
    """The fake MCP transport: one call of the server's own tool per call."""

    async def call(tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        return parse_tool_reply(await server._tool_manager.call_tool(tool, arguments))

    return call


@dataclass
class LinkHealHarness:
    service: MemoryService
    mem0: FakeMem0
    journal: WriteJournal
    server: Any
    journal_dir: Path

    def tool_caller(self) -> ToolCaller:
        return in_process_tool_caller(self.server)

    def store(self) -> LinkHealStore:
        return LinkHealStore(self.tool_caller())

    async def call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        reply = await self.tool_caller()(tool, arguments)
        assert reply is not None
        return reply

    def seed(self, project_id: str, memory_id: str, text: str, **meta: Any) -> None:
        sequence = len(self.mem0.points)
        self.mem0.points[(project_id, memory_id)] = {
            'data': text,
            'hash': f'hash-{memory_id}',
            'created_at': f'2026-01-01T00:00:{sequence % 60:02d}+00:00',
            'user_id': project_id,
            'category': MemoryCategory.observations_and_summaries.value,
            **meta,
        }

    def seed_link(
        self,
        *,
        kind: str | None = AMENDMENT_KIND,
        child: str = CHILD,
        parent: str = PARENT,
        child_text: str = CHILD_TEXT,
        parent_text: str | None = PARENT_TEXT,
        project: str = DF,
        contested: bool = False,
    ) -> None:
        """A child linked to a parent; ``parent_text=None`` seeds no parent."""
        if parent_text is not None:
            self.seed(project, parent, parent_text)
        meta: dict[str, object] = {PARENT_ID_KEY: parent}
        if kind is not None:
            meta['kind'] = kind
        if contested:
            meta[CONTESTED_METADATA_KEY] = True
        self.seed(project, child, child_text, **meta)

    def journal_rows(self, kind: str = 'write') -> list[dict[str, Any]]:
        """The journal's ``write_ops`` rows of *kind*, oldest first, params decoded."""
        with sqlite3.connect(self.journal_dir / 'write_journal.db') as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM write_ops WHERE kind = ? ORDER BY created_at', (kind,),
            ).fetchall()
        return [{**dict(row), 'params': json.loads(row['params'] or '{}')} for row in rows]


def _mock_graphiti() -> MagicMock:
    graphiti = MagicMock()
    graphiti.search = AsyncMock(return_value=[])
    graphiti.search_nodes = AsyncMock(return_value=[])
    install_identity_mocks(graphiti)
    return graphiti


async def build_harness(
    mock_config: Any, tmp_path: Path, *, metadata_patch_prefixes: list[str],
) -> LinkHealHarness:
    """A real MemoryService and MCP server over a fresh :class:`FakeMem0`.

    The ``update_memory`` allowlist is set explicitly, never read from the YAML.
    """
    mock_config.mem0_update = Mem0UpdateConfig(
        enabled=True, metadata_patch_allowed_agent_prefixes=list(metadata_patch_prefixes),
    )
    service = MemoryService(mock_config)
    mem0 = FakeMem0()
    service.mem0 = mem0  # type: ignore[assignment]
    service.graphiti = _mock_graphiti()

    async def search(query: str, project_id: str, limit: int = 10, **_ignored: Any):
        return mem0.search_results(query, project_id, limit)

    service.search = search  # type: ignore[method-assign]
    journal_dir = tmp_path / 'journal'
    journal = WriteJournal(journal_dir)
    await journal.initialize()
    service.set_write_journal(journal)
    return LinkHealHarness(
        service=service,
        mem0=mem0,
        journal=journal,
        server=create_mcp_server(service),
        journal_dir=journal_dir,
    )


async def run_corpus_plan(
    harness: LinkHealHarness,
    ledger: LinkHealLedger,
    plan_path: Path,
    bases: tuple[LinkBasis, ...],
    limits: RunLimits = DEFAULT_LIMITS,
) -> RunReport:
    """``run_plan`` over the harness's store and census, for both projects."""
    return await run_plan(
        bases,
        store=harness.store(),
        census=harness.mem0,
        ledger=ledger,
        limits=limits,
        projects=PROJECTS,
        source=RunSource.CORPUS,
        plan_path=plan_path,
    )


class RecordingFiler:
    """An ``EscapeFiler`` that files nowhere and remembers every escape."""

    def __init__(self) -> None:
        self.escapes: list[Escape] = []

    def __call__(self, escape: Escape) -> str | None:
        self.escapes.append(escape)
        return f'esc-recorded-{len(self.escapes)}'


async def run_corpus_apply(
    harness: LinkHealHarness,
    ledger: LinkHealLedger,
    *,
    limits: RunLimits = DEFAULT_LIMITS,
    filer: EscapeFiler | None = None,
    approved_plan_sha256: str | None = None,
) -> RunReport:
    """``run_apply`` of the corpus-planned heals over the harness's store."""
    return await run_apply(
        store=harness.store(),
        ledger=ledger,
        limits=limits,
        filer=filer or RecordingFiler(),
        source=RunSource.CORPUS,
        approved_plan_sha256=approved_plan_sha256,
    )


def assert_store_invariants(harness: LinkHealHarness) -> None:
    """No heal combines a patch and a delete, sends content, or stores a null link key."""
    for route in FORBIDDEN_ROUTES:
        assert harness.mem0.writes[route] == 0, route
    for (project_id, memory_id), payload in harness.mem0.points.items():
        for key in LINK_KEYS:
            assert key not in payload or payload[key] is not None, (project_id, memory_id, key)
