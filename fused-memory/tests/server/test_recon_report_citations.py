"""Tests for the recon_report cite_* citation tools (task β).

Covers:
- TestCiteEntity  — cite_entity happy path, entity_not_found, finding_unknown, accumulation
- TestCiteEdge    — cite_edge UUID shape gate, edge_not_found, happy path, finding_unknown
- TestCiteTask    — cite_task happy path, unknown_project, task_not_found, cross-finding P5
- TestCiteMemory  — cite_memory UUID shape gate, memory_not_found, happy paths (both stores)
- TestCiteToolsViaFastMCP — tools registered, end-to-end via call_tool, P4 schema rejection
- TestReconReportComponentsWiring — _build_recon_report_components service injection
"""

from __future__ import annotations

import pytest
from mcp.server.fastmcp.exceptions import ToolError

# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeMemoryService:
    """Lightweight async stub for MemoryService (cite_entity / cite_edge / cite_memory)."""

    def __init__(
        self,
        *,
        entity_nodes: list[dict] | None = None,
        edge_result: dict | None = None,
        edge_raises=None,
        memory_result: dict | None = None,
        memory_raises=None,
    ) -> None:
        # get_entity configuration
        self._entity_nodes: list[dict] = entity_nodes if entity_nodes is not None else []

        # get_edge configuration
        self._edge_result = edge_result
        self._edge_raises = edge_raises  # exception instance to raise

        # get_memory configuration
        self._memory_result = memory_result
        self._memory_raises = memory_raises

        # Call tracking
        self.get_entity_calls: list[tuple[str, str]] = []
        self.get_edge_calls: list[tuple[str, str]] = []
        self.get_memory_calls: list[tuple[str, str, str]] = []

    async def get_entity(self, name: str, project_id: str) -> dict:
        self.get_entity_calls.append((name, project_id))
        return {'nodes': self._entity_nodes, 'edges': []}

    async def get_edge(self, edge_uuid: str, project_id: str) -> dict:
        self.get_edge_calls.append((edge_uuid, project_id))
        if self._edge_raises is not None:
            raise self._edge_raises
        return self._edge_result  # type: ignore[return-value]

    async def get_memory(self, memory_id: str, store: str, project_id: str) -> dict:
        self.get_memory_calls.append((memory_id, store, project_id))
        if self._memory_raises is not None:
            raise self._memory_raises
        return self._memory_result  # type: ignore[return-value]


class _FakeTaskInterceptor:
    """Lightweight async stub for TaskInterceptor.get_task."""

    def __init__(self, *, results: dict[tuple[str, str], dict] | None = None) -> None:
        # results keyed by (task_id, project_root)
        self._results: dict[tuple[str, str], dict] = results or {}
        self.get_task_calls: list[tuple[str, str]] = []

    async def get_task(self, task_id: str, project_root: str, tag=None) -> dict:
        self.get_task_calls.append((task_id, project_root))
        return self._results.get((task_id, project_root), {'error': 'task_not_found'})


# ---------------------------------------------------------------------------
# Shared helper — builds a state with one open report + one finding
# ---------------------------------------------------------------------------


def _make_state_with_finding(
    memory_service=None,
    task_interceptor=None,
    known_projects: dict | None = None,
) -> tuple:
    """Return (state, run_id, finding_id) with one started report + one finding."""
    from fused_memory.server.recon_report import ReconReportState

    t = [0.0]
    state = ReconReportState(
        ttl_seconds=300,
        clock=lambda: t[0],
        memory_service=memory_service,
        task_interceptor=task_interceptor,
    )
    if known_projects is not None:
        state.known_projects = known_projects

    run_id = 'run-1'
    state.start_report(run_id=run_id, stage='reconciler', project_id='dark_factory')
    result = state.add_finding(
        run_id=run_id,
        severity='moderate',
        category='memory_stale',
        description='d',
        suggested_action='a',
        actionable=True,
        task_id='42',
        flag_type='orphaned_knowledge',
    )
    finding_id = result['finding_id']
    return state, run_id, finding_id


# ---------------------------------------------------------------------------
# step-1: TestCiteEntity — RED until step-2 adds cite_entity to ReconReportState
# ---------------------------------------------------------------------------


class TestCiteEntity:
    """Drive state.cite_entity() directly (no FastMCP spin-up needed)."""

    def _fake(self, nodes=None):
        if nodes is None:
            nodes = [{'uuid': 'a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'name': 'Foo'}]
        return _FakeMemoryService(entity_nodes=nodes)

    def _state_and_finding(self, nodes=None):
        fake = self._fake(nodes)
        state, run_id, finding_id = _make_state_with_finding(memory_service=fake)
        return state, run_id, finding_id, fake

    @pytest.mark.asyncio
    async def test_happy_path_returns_entity_uuid_and_canonical_name(self):
        """cite_entity with a known entity returns {entity_uuid, canonical_name}."""
        state, run_id, finding_id, _ = self._state_and_finding()

        result = await state.cite_entity(run_id, finding_id, 'foo')

        assert result.get('entity_uuid') == 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
        assert result.get('canonical_name') == 'Foo'

    @pytest.mark.asyncio
    async def test_happy_path_mutates_cited_entities(self):
        """The returned dict must also appear in the finding's cited_entities."""
        state, run_id, finding_id, _ = self._state_and_finding()

        await state.cite_entity(run_id, finding_id, 'foo')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        item = report['flagged_items'][0]
        assert len(item['cited_entities']) == 1
        assert item['cited_entities'][0]['entity_uuid'] == 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
        assert item['cited_entities'][0]['canonical_name'] == 'Foo'

    @pytest.mark.asyncio
    async def test_entity_not_found_when_nodes_empty(self):
        """Empty nodes list → {error:'entity_not_found', error_type:'ReconReportEntityNotFound'}."""
        state, run_id, finding_id, _ = self._state_and_finding(nodes=[])

        result = await state.cite_entity(run_id, finding_id, 'ghost')

        assert result.get('error') == 'entity_not_found'
        assert result.get('error_type') == 'ReconReportEntityNotFound'

    @pytest.mark.asyncio
    async def test_entity_not_found_leaves_cited_entities_unchanged(self):
        """entity_not_found must NOT mutate cited_entities."""
        state, run_id, finding_id, _ = self._state_and_finding(nodes=[])

        await state.cite_entity(run_id, finding_id, 'ghost')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert report['flagged_items'][0]['cited_entities'] == []

    @pytest.mark.asyncio
    async def test_finding_unknown_for_bogus_finding_id(self):
        """Bogus finding_id → {error:'finding_unknown', error_type:'ReconReportFindingUnknown'}."""
        state, run_id, _finding_id, _ = self._state_and_finding()

        result = await state.cite_entity(run_id, 'bogus-finding-id', 'foo')

        assert result.get('error') == 'finding_unknown'
        assert result.get('error_type') == 'ReconReportFindingUnknown'

    @pytest.mark.asyncio
    async def test_accumulation_two_citations_produce_len_two(self):
        """Two cite_entity calls on the same finding → len(cited_entities)==2 (append, not replace)."""
        nodes = [
            {'uuid': 'aaaaaaaa-0000-0000-0000-000000000001', 'name': 'Alpha'},
            {'uuid': 'bbbbbbbb-0000-0000-0000-000000000002', 'name': 'Beta'},
        ]
        # First call returns Alpha, second call returns Beta
        fake1 = _FakeMemoryService(entity_nodes=[nodes[0]])
        state, run_id, finding_id = _make_state_with_finding(memory_service=fake1)

        # First citation
        r1 = await state.cite_entity(run_id, finding_id, 'alpha')
        assert r1.get('entity_uuid') == nodes[0]['uuid']

        # Swap to Beta nodes
        state._memory_service = _FakeMemoryService(entity_nodes=[nodes[1]])  # type: ignore[attr-defined]

        # Second citation
        r2 = await state.cite_entity(run_id, finding_id, 'beta')
        assert r2.get('entity_uuid') == nodes[1]['uuid']

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert len(report['flagged_items'][0]['cited_entities']) == 2

    @pytest.mark.asyncio
    async def test_run_id_unknown_returned_for_bad_run(self):
        """Passing an unregistered run_id returns run_id_unknown."""
        state, run_id, finding_id, _ = self._state_and_finding()

        result = await state.cite_entity('no-such-run', finding_id, 'foo')

        assert result.get('error') == 'run_id_unknown'


# ---------------------------------------------------------------------------
# step-3: TestCiteEdge — RED until step-4 adds cite_edge to ReconReportState
# ---------------------------------------------------------------------------


class TestCiteEdge:
    """Drive state.cite_edge() directly."""

    _VALID_UUID = 'f47ac10b-58cc-4372-a567-0e02b2c3d479'

    def _fake(self, *, edge_result=None, edge_raises=None):
        return _FakeMemoryService(
            edge_result=edge_result or {'uuid': self._VALID_UUID, 'name': 'test-edge', 'fact': 'some fact'},
            edge_raises=edge_raises,
        )

    def _state_and_finding(self, **fake_kwargs):
        fake = self._fake(**fake_kwargs)
        state, run_id, finding_id = _make_state_with_finding(memory_service=fake)
        return state, run_id, finding_id, fake

    @pytest.mark.asyncio
    async def test_invalid_uuid_shape_short_hex(self):
        """'96cddd4d-edge' (P3) → {error:'invalid_uuid_shape'}, no service call."""
        state, run_id, finding_id, fake = self._state_and_finding()

        result = await state.cite_edge(run_id, finding_id, '96cddd4d-edge')

        assert result.get('error') == 'invalid_uuid_shape'
        assert result.get('error_type') == 'ReconReportInvalidUuid'
        assert fake.get_edge_calls == []

    @pytest.mark.asyncio
    async def test_invalid_uuid_shape_leaves_cited_edges_unchanged(self):
        """UUID shape rejection must NOT mutate cited_edges."""
        state, run_id, finding_id, _ = self._state_and_finding()

        await state.cite_edge(run_id, finding_id, '96cddd4d-edge')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert report['flagged_items'][0]['cited_edges'] == []

    @pytest.mark.asyncio
    async def test_edge_not_found_when_get_edge_raises(self):
        """EdgeNotFoundError from get_edge → {error:'edge_not_found'}."""
        from graphiti_core.errors import EdgeNotFoundError

        state, run_id, finding_id, _ = self._state_and_finding(
            edge_raises=EdgeNotFoundError('edge not found')
        )

        result = await state.cite_edge(run_id, finding_id, self._VALID_UUID)

        assert result.get('error') == 'edge_not_found'
        assert result.get('error_type') == 'ReconReportEdgeNotFound'

    @pytest.mark.asyncio
    async def test_edge_not_found_leaves_cited_edges_unchanged(self):
        from graphiti_core.errors import EdgeNotFoundError

        state, run_id, finding_id, _ = self._state_and_finding(
            edge_raises=EdgeNotFoundError('edge not found')
        )
        await state.cite_edge(run_id, finding_id, self._VALID_UUID)

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert report['flagged_items'][0]['cited_edges'] == []

    @pytest.mark.asyncio
    async def test_happy_path_returns_edge_uuid_and_fact(self):
        """Happy path returns {edge_uuid, fact_text_snapshot}."""
        state, run_id, finding_id, _ = self._state_and_finding(
            edge_result={'uuid': self._VALID_UUID, 'name': 'rel', 'fact': 'some fact'}
        )

        result = await state.cite_edge(run_id, finding_id, self._VALID_UUID)

        assert result.get('edge_uuid') == self._VALID_UUID
        assert result.get('fact_text_snapshot') == 'some fact'

    @pytest.mark.asyncio
    async def test_happy_path_mutates_cited_edges(self):
        state, run_id, finding_id, _ = self._state_and_finding(
            edge_result={'uuid': self._VALID_UUID, 'name': 'rel', 'fact': 'some fact'}
        )

        await state.cite_edge(run_id, finding_id, self._VALID_UUID)

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        edges = report['flagged_items'][0]['cited_edges']
        assert len(edges) == 1
        assert edges[0]['edge_uuid'] == self._VALID_UUID
        assert edges[0]['fact_text_snapshot'] == 'some fact'

    @pytest.mark.asyncio
    async def test_finding_unknown_for_bogus_finding_id(self):
        state, run_id, _, _ = self._state_and_finding()

        result = await state.cite_edge(run_id, 'bogus-finding', self._VALID_UUID)

        assert result.get('error') == 'finding_unknown'
        assert result.get('error_type') == 'ReconReportFindingUnknown'


# ---------------------------------------------------------------------------
# step-5: TestCiteTask — RED until step-6 adds cite_task to ReconReportState
# ---------------------------------------------------------------------------

_KNOWN_PROJECTS = {
    'dark_factory': '/home/leo/src/dark-factory',
    'other_project': '/home/leo/src/other',
}


class TestCiteTask:
    """Drive state.cite_task() directly."""

    def _fake_ti(self, known_roots=None):
        """Build a fake task interceptor that returns T-<task_id> for known roots."""
        if known_roots is None:
            known_roots = {'/home/leo/src/dark-factory', '/home/leo/src/other'}
        results = {}
        for pr in known_roots:
            for tid in ['1', '5', '42']:
                results[(tid, pr)] = {'id': tid, 'title': f'T-{tid}'}
        return _FakeTaskInterceptor(results=results)

    def _state_and_finding(self, known_projects=None, fake_ti=None):
        if fake_ti is None:
            fake_ti = self._fake_ti()
        kp = known_projects if known_projects is not None else dict(_KNOWN_PROJECTS)
        state, run_id, finding_id = _make_state_with_finding(
            task_interceptor=fake_ti,
            known_projects=kp,
        )
        return state, run_id, finding_id, fake_ti

    @pytest.mark.asyncio
    async def test_happy_path_returns_project_task_title(self):
        """Happy path → {project_id, task_id, title}."""
        state, run_id, finding_id, _ = self._state_and_finding()

        result = await state.cite_task(run_id, finding_id, 'dark_factory', '5')

        assert result.get('project_id') == 'dark_factory'
        assert result.get('task_id') == '5'
        assert result.get('title') == 'T-5'

    @pytest.mark.asyncio
    async def test_happy_path_mutates_cited_tasks(self):
        state, run_id, finding_id, _ = self._state_and_finding()

        await state.cite_task(run_id, finding_id, 'dark_factory', '5')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        tasks = report['flagged_items'][0]['cited_tasks']
        assert len(tasks) == 1
        assert tasks[0] == {'project_id': 'dark_factory', 'task_id': '5', 'title': 'T-5'}

    @pytest.mark.asyncio
    async def test_unknown_project_returns_error_no_service_call(self):
        """Unknown project_id → {error:'unknown_project'}, no get_task call."""
        state, run_id, finding_id, fake_ti = self._state_and_finding()

        result = await state.cite_task(run_id, finding_id, 'nope', '1')

        assert result.get('error') == 'unknown_project'
        assert result.get('error_type') == 'ReconReportUnknownProject'
        assert fake_ti.get_task_calls == []

    @pytest.mark.asyncio
    async def test_unknown_project_leaves_cited_tasks_unchanged(self):
        state, run_id, finding_id, _ = self._state_and_finding()

        await state.cite_task(run_id, finding_id, 'nope', '1')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert report['flagged_items'][0]['cited_tasks'] == []

    @pytest.mark.asyncio
    async def test_task_not_found_when_interceptor_returns_error(self):
        """get_task returning error dict → {error:'task_not_found'}."""
        fake_ti = _FakeTaskInterceptor(results={})  # all tasks return error
        state, run_id, finding_id, _ = self._state_and_finding(fake_ti=fake_ti)

        result = await state.cite_task(run_id, finding_id, 'dark_factory', '999')

        assert result.get('error') == 'task_not_found'
        assert result.get('error_type') == 'ReconReportTaskNotFound'

    @pytest.mark.asyncio
    async def test_task_not_found_leaves_cited_tasks_unchanged(self):
        fake_ti = _FakeTaskInterceptor(results={})
        state, run_id, finding_id, _ = self._state_and_finding(fake_ti=fake_ti)

        await state.cite_task(run_id, finding_id, 'dark_factory', '999')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert report['flagged_items'][0]['cited_tasks'] == []

    @pytest.mark.asyncio
    async def test_p5_two_findings_with_same_task_id_different_projects(self):
        """P5: two findings, task_id='1', different project_ids → distinct cited_tasks entries."""
        fake_ti = self._fake_ti()
        state, run_id, finding_id_1, _ = self._state_and_finding(fake_ti=fake_ti)

        # Add a second finding
        r2 = state.add_finding(
            run_id=run_id,
            severity='low',
            category='cat',
            description='d2',
            suggested_action='a',
            task_id='99',  # different sig
            flag_type='other_flag',
        )
        finding_id_2 = r2['finding_id']

        # Cite task_id='1' from two different projects on two different findings
        await state.cite_task(run_id, finding_id_1, 'dark_factory', '1')
        await state.cite_task(run_id, finding_id_2, 'other_project', '1')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        tasks_1 = report['flagged_items'][0]['cited_tasks']
        tasks_2 = report['flagged_items'][1]['cited_tasks']

        assert len(tasks_1) == 1
        assert tasks_1[0]['project_id'] == 'dark_factory'
        assert len(tasks_2) == 1
        assert tasks_2[0]['project_id'] == 'other_project'

    @pytest.mark.asyncio
    async def test_finding_unknown_for_bogus_finding_id(self):
        state, run_id, _, _ = self._state_and_finding()

        result = await state.cite_task(run_id, 'bogus-finding', 'dark_factory', '1')

        assert result.get('error') == 'finding_unknown'
        assert result.get('error_type') == 'ReconReportFindingUnknown'


# ---------------------------------------------------------------------------
# step-7: TestCiteMemory — RED until step-8 adds cite_memory to ReconReportState
# ---------------------------------------------------------------------------


class TestCiteMemory:
    """Drive state.cite_memory() directly."""

    _VALID_UUID = 'd4e5f6a7-b8c9-0123-d456-e78f9a0b1c2d'
    _FINGERPRINT = {
        'category': 'observations_and_summaries',
        'agent_id': 'x',
        'created_at': '2026-05-28T00:00:00Z',
    }

    def _fake(self, *, memory_result=None, memory_raises=None):
        return _FakeMemoryService(
            memory_result=memory_result if memory_result is not None else self._FINGERPRINT,
            memory_raises=memory_raises,
        )

    def _state_and_finding(self, **fake_kwargs):
        fake = self._fake(**fake_kwargs)
        state, run_id, finding_id = _make_state_with_finding(memory_service=fake)
        return state, run_id, finding_id, fake

    @pytest.mark.asyncio
    async def test_happy_path_graphiti_store(self):
        """store='graphiti' → {memory_id, store, metadata_fingerprint}."""
        state, run_id, finding_id, _ = self._state_and_finding()

        result = await state.cite_memory(run_id, finding_id, self._VALID_UUID, 'graphiti')

        assert result.get('memory_id') == self._VALID_UUID
        assert result.get('store') == 'graphiti'
        assert result.get('metadata_fingerprint') == self._FINGERPRINT

    @pytest.mark.asyncio
    async def test_happy_path_mem0_store(self):
        """store='mem0' → {memory_id, store, metadata_fingerprint}."""
        fingerprint = {'category': 'procedural_knowledge', 'agent_id': 'y', 'created_at': '2026-01-01'}
        state, run_id, finding_id, _ = self._state_and_finding(memory_result=fingerprint)

        result = await state.cite_memory(run_id, finding_id, self._VALID_UUID, 'mem0')

        assert result.get('memory_id') == self._VALID_UUID
        assert result.get('store') == 'mem0'
        assert result.get('metadata_fingerprint') == fingerprint

    @pytest.mark.asyncio
    async def test_happy_path_mutates_cited_memories(self):
        state, run_id, finding_id, _ = self._state_and_finding()

        await state.cite_memory(run_id, finding_id, self._VALID_UUID, 'graphiti')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        memories = report['flagged_items'][0]['cited_memories']
        assert len(memories) == 1
        assert memories[0]['memory_id'] == self._VALID_UUID
        assert memories[0]['store'] == 'graphiti'
        assert memories[0]['metadata_fingerprint'] == self._FINGERPRINT

    @pytest.mark.asyncio
    async def test_invalid_uuid_shape_8char_hex_prefix(self):
        """'2531b4d8' (8-char) → {error:'invalid_uuid_shape'}, no service call."""
        state, run_id, finding_id, fake = self._state_and_finding()

        result = await state.cite_memory(run_id, finding_id, '2531b4d8', 'graphiti')

        assert result.get('error') == 'invalid_uuid_shape'
        assert result.get('error_type') == 'ReconReportInvalidUuid'
        assert fake.get_memory_calls == []

    @pytest.mark.asyncio
    async def test_invalid_uuid_shape_leaves_cited_memories_unchanged(self):
        state, run_id, finding_id, _ = self._state_and_finding()

        await state.cite_memory(run_id, finding_id, '2531b4d8', 'graphiti')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert report['flagged_items'][0]['cited_memories'] == []

    @pytest.mark.asyncio
    async def test_memory_not_found_when_service_raises(self):
        """Raises from get_memory → {error:'memory_not_found'}."""
        from graphiti_core.errors import EdgeNotFoundError

        state, run_id, finding_id, _ = self._state_and_finding(
            memory_raises=EdgeNotFoundError('not found')
        )

        result = await state.cite_memory(run_id, finding_id, self._VALID_UUID, 'graphiti')

        assert result.get('error') == 'memory_not_found'
        assert result.get('error_type') == 'ReconReportMemoryNotFound'

    @pytest.mark.asyncio
    async def test_memory_not_found_leaves_cited_memories_unchanged(self):
        from graphiti_core.errors import EdgeNotFoundError

        state, run_id, finding_id, _ = self._state_and_finding(
            memory_raises=EdgeNotFoundError('not found')
        )
        await state.cite_memory(run_id, finding_id, self._VALID_UUID, 'graphiti')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert report['flagged_items'][0]['cited_memories'] == []

    @pytest.mark.asyncio
    async def test_finding_unknown_for_bogus_finding_id(self):
        state, run_id, _, _ = self._state_and_finding()

        result = await state.cite_memory(run_id, 'bogus-finding', self._VALID_UUID, 'graphiti')

        assert result.get('error') == 'finding_unknown'
        assert result.get('error_type') == 'ReconReportFindingUnknown'


# ---------------------------------------------------------------------------
# step-13: TestCiteMemoryExceptionNarrowing — RED until step-14 narrows except
# ---------------------------------------------------------------------------


class TestCiteMemoryExceptionNarrowing:
    """Verifies that unexpected exceptions propagate rather than being misclassified as memory_not_found.

    cite_memory catches only (EdgeNotFoundError, MemoryNotFoundError); KeyError and
    ValueError propagate to the caller unchanged.
    """

    _VALID_UUID = 'd4e5f6a7-b8c9-0123-d456-e78f9a0b1c2d'

    def _state_and_finding(self, memory_raises):
        fake = _FakeMemoryService(memory_raises=memory_raises)
        state, run_id, finding_id = _make_state_with_finding(memory_service=fake)
        return state, run_id, finding_id

    @pytest.mark.asyncio
    async def test_memory_not_found_error_caught_as_memory_not_found(self):
        """MemoryNotFoundError → {error:'memory_not_found'} and cited_memories unchanged."""
        from fused_memory.services.memory_service import MemoryNotFoundError  # noqa: PLC0415

        state, run_id, finding_id = self._state_and_finding(
            memory_raises=MemoryNotFoundError('not-found')
        )

        result = await state.cite_memory(run_id, finding_id, self._VALID_UUID, 'mem0')

        assert result.get('error') == 'memory_not_found'
        assert result.get('error_type') == 'ReconReportMemoryNotFound'
        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert report['flagged_items'][0]['cited_memories'] == []

    @pytest.mark.asyncio
    async def test_edge_not_found_error_caught_as_memory_not_found_graphiti_parity(self):
        """EdgeNotFoundError from get_memory (graphiti path) → memory_not_found (parity)."""
        from graphiti_core.errors import EdgeNotFoundError

        state, run_id, finding_id = self._state_and_finding(
            memory_raises=EdgeNotFoundError('not found')
        )

        result = await state.cite_memory(run_id, finding_id, self._VALID_UUID, 'graphiti')

        assert result.get('error') == 'memory_not_found'
        assert result.get('error_type') == 'ReconReportMemoryNotFound'

    @pytest.mark.asyncio
    async def test_key_error_propagates_not_silenced(self):
        """KeyError from get_memory must propagate — not be caught as memory_not_found."""
        state, run_id, finding_id = self._state_and_finding(
            memory_raises=KeyError('transient mem0 bug')
        )

        with pytest.raises(KeyError):
            await state.cite_memory(run_id, finding_id, self._VALID_UUID, 'mem0')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert report['flagged_items'][0]['cited_memories'] == []

    @pytest.mark.asyncio
    async def test_value_error_propagates_not_silenced(self):
        """ValueError from get_memory must propagate — not be caught as memory_not_found."""
        state, run_id, finding_id = self._state_and_finding(
            memory_raises=ValueError('unrelated graphiti error')
        )

        with pytest.raises(ValueError):
            await state.cite_memory(run_id, finding_id, self._VALID_UUID, 'graphiti')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert report['flagged_items'][0]['cited_memories'] == []


# ---------------------------------------------------------------------------
# step-9: TestCiteToolsViaFastMCP + TestReconReportComponentsWiring
#         RED until step-10 registers the tools and extends _build_recon_report_components
# ---------------------------------------------------------------------------


class TestCiteToolsViaFastMCP:
    """Verify the four cite_* tools are registered and wired correctly in FastMCP."""

    def _make(self):
        from fused_memory.server.recon_report import ReconReportState, create_recon_report_server

        t = [0.0]
        fake_ms = _FakeMemoryService(
            entity_nodes=[{'uuid': 'aaaaaaaa-1111-1111-1111-111111111111', 'name': 'Foo'}]
        )
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: t[0],
            memory_service=fake_ms,
        )
        state.known_projects = {}
        mcp = create_recon_report_server(state)
        return state, mcp

    def test_four_cite_tools_registered(self):
        """All four cite_* tools must be registered in the FastMCP tool manager."""
        _, mcp = self._make()
        tools = set(mcp._tool_manager._tools.keys())
        assert {'cite_entity', 'cite_edge', 'cite_task', 'cite_memory'} <= tools

    @pytest.mark.asyncio
    async def test_end_to_end_cite_entity_via_call_tool(self):
        """Full lifecycle: start_report → add_finding → cite_entity → get_assembled_report."""
        state, mcp = self._make()
        tm = mcp._tool_manager

        # start_report
        await tm.call_tool('start_report', {
            'run_id': 'r1', 'stage': 'test_stage', 'project_id': 'dark_factory',
        })

        # add_finding
        r = await tm.call_tool('add_finding', {
            'run_id': 'r1',
            'severity': 'low',
            'category': 'cat',
            'description': 'd',
            'suggested_action': 'a',
            'task_id': '1',
            'flag_type': 'f',
        })
        finding_id = r['finding_id']

        # cite_entity
        cite_r = await tm.call_tool('cite_entity', {
            'run_id': 'r1',
            'finding_id': finding_id,
            'name': 'Foo',
        })
        assert cite_r.get('entity_uuid') == 'aaaaaaaa-1111-1111-1111-111111111111'

        # complete
        await tm.call_tool('complete', {'run_id': 'r1', 'summary': 'done'})

        # Verify state has citation
        report = state.get_assembled_report('r1', 'test_stage')
        assert report is not None
        assert len(report['flagged_items'][0]['cited_entities']) == 1

    @pytest.mark.asyncio
    async def test_p4_cite_task_without_project_id_raises_validation_error(self):
        """P4: cite_task called with project_id omitted must raise a validation error."""
        state, mcp = self._make()
        # Add a known project and finding first
        state.known_projects = {'dark_factory': '/home/leo/src/dark-factory'}
        state.start_report(run_id='r1', stage='s', project_id='dark_factory')
        fid = state.add_finding(
            run_id='r1', severity='low', category='c',
            description='d', suggested_action='a'
        )['finding_id']
        tm = mcp._tool_manager

        # Omit project_id — FastMCP/Pydantic raises ToolError wrapping ValidationError
        with pytest.raises(ToolError):
            await tm.call_tool('cite_task', {
                'run_id': 'r1',
                'finding_id': fid,
                'task_id': '1',
                # project_id intentionally omitted
            })


# ---------------------------------------------------------------------------
# step-11: TestUuidCaseInsensitive — RED until step-12 adds re.IGNORECASE
# ---------------------------------------------------------------------------


class TestUuidCaseInsensitive:
    """Valid UUIDs with uppercase hex digits must NOT be rejected as invalid_uuid_shape."""

    _UUID_UPPER = 'F47AC10B-58CC-4372-A567-0E02B2C3D479'
    _UUID_MIXED = 'F47AC10B-58cc-4372-A567-0e02b2c3d479'
    _FINGERPRINT = {'category': 'observations_and_summaries', 'agent_id': 'a', 'created_at': '2026-05-28'}

    @pytest.mark.asyncio
    async def test_cite_edge_accepts_uppercase_uuid(self):
        """cite_edge with all-caps UUID must NOT return invalid_uuid_shape."""
        fake = _FakeMemoryService(
            edge_result={'uuid': self._UUID_UPPER, 'name': 'E', 'fact': 'fact-text'},
        )
        state, run_id, finding_id = _make_state_with_finding(memory_service=fake)

        result = await state.cite_edge(run_id, finding_id, self._UUID_UPPER)

        # Must NOT be rejected by the shape gate
        assert 'error' not in result, f"Unexpected error: {result}"
        assert result.get('edge_uuid') == self._UUID_UPPER
        assert result.get('fact_text_snapshot') == 'fact-text'
        # Shape gate must NOT have short-circuited — service was called
        assert len(fake.get_edge_calls) == 1
        assert fake.get_edge_calls[0][0] == self._UUID_UPPER

    @pytest.mark.asyncio
    async def test_cite_edge_cited_edges_gains_citation_for_uppercase_uuid(self):
        """cited_edges must include the citation when UUID is all-caps."""
        fake = _FakeMemoryService(
            edge_result={'uuid': self._UUID_UPPER, 'name': 'E', 'fact': 'fact-text'},
        )
        state, run_id, finding_id = _make_state_with_finding(memory_service=fake)

        await state.cite_edge(run_id, finding_id, self._UUID_UPPER)

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        edges = report['flagged_items'][0]['cited_edges']
        assert len(edges) == 1
        assert edges[0]['edge_uuid'] == self._UUID_UPPER

    @pytest.mark.asyncio
    async def test_cite_memory_accepts_mixed_case_uuid(self):
        """cite_memory with mixed-case UUID must NOT return invalid_uuid_shape."""
        fake = _FakeMemoryService(memory_result=self._FINGERPRINT)
        state, run_id, finding_id = _make_state_with_finding(memory_service=fake)

        result = await state.cite_memory(run_id, finding_id, self._UUID_MIXED, 'graphiti')

        # Must NOT be rejected by the shape gate
        assert 'error' not in result, f"Unexpected error: {result}"
        assert result.get('memory_id') == self._UUID_MIXED
        # Shape gate must NOT have short-circuited — service was called
        assert len(fake.get_memory_calls) == 1

    @pytest.mark.asyncio
    async def test_cite_memory_cited_memories_gains_citation_for_mixed_case_uuid(self):
        """cited_memories must include the citation when UUID has mixed case."""
        fake = _FakeMemoryService(memory_result=self._FINGERPRINT)
        state, run_id, finding_id = _make_state_with_finding(memory_service=fake)

        await state.cite_memory(run_id, finding_id, self._UUID_MIXED, 'graphiti')

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        memories = report['flagged_items'][0]['cited_memories']
        assert len(memories) == 1
        assert memories[0]['memory_id'] == self._UUID_MIXED


# ---------------------------------------------------------------------------
# TestServiceNotConfigured — service_not_configured guard when services are None
# ---------------------------------------------------------------------------


class TestServiceNotConfigured:
    """Guard: cite_* methods return service_not_configured when the injected service is None."""

    _VALID_UUID = 'f47ac10b-58cc-4372-a567-0e02b2c3d479'

    @pytest.mark.asyncio
    async def test_cite_entity_without_memory_service(self):
        """cite_entity returns service_not_configured when memory_service is None."""
        state, run_id, finding_id = _make_state_with_finding()  # memory_service=None

        result = await state.cite_entity(run_id, finding_id, 'SomeName')

        assert result.get('error') == 'service_not_configured'
        assert result.get('error_type') == 'ReconReportServiceUnavailable'

    @pytest.mark.asyncio
    async def test_cite_edge_without_memory_service(self):
        """cite_edge returns service_not_configured after UUID validation when memory_service is None."""
        state, run_id, finding_id = _make_state_with_finding()

        result = await state.cite_edge(run_id, finding_id, self._VALID_UUID)

        assert result.get('error') == 'service_not_configured'
        assert result.get('error_type') == 'ReconReportServiceUnavailable'

    @pytest.mark.asyncio
    async def test_cite_edge_invalid_uuid_still_rejected_before_service_guard(self):
        """invalid_uuid_shape is returned for bad UUIDs even when memory_service is None."""
        state, run_id, finding_id = _make_state_with_finding()

        result = await state.cite_edge(run_id, finding_id, 'not-a-uuid')

        assert result.get('error') == 'invalid_uuid_shape'

    @pytest.mark.asyncio
    async def test_cite_task_without_task_interceptor(self):
        """cite_task returns service_not_configured after project validation when task_interceptor is None."""
        state, run_id, finding_id = _make_state_with_finding()
        state.known_projects = {'dark_factory': '/home/leo/src/dark-factory'}

        result = await state.cite_task(run_id, finding_id, 'dark_factory', '1')

        assert result.get('error') == 'service_not_configured'
        assert result.get('error_type') == 'ReconReportServiceUnavailable'

    @pytest.mark.asyncio
    async def test_cite_task_unknown_project_still_rejected_before_service_guard(self):
        """unknown_project is returned for bad project_id even when task_interceptor is None."""
        state, run_id, finding_id = _make_state_with_finding()
        # known_projects is empty by default

        result = await state.cite_task(run_id, finding_id, 'nonexistent', '1')

        assert result.get('error') == 'unknown_project'

    @pytest.mark.asyncio
    async def test_cite_memory_without_memory_service(self):
        """cite_memory returns service_not_configured after UUID validation when memory_service is None."""
        state, run_id, finding_id = _make_state_with_finding()

        result = await state.cite_memory(run_id, finding_id, self._VALID_UUID, 'graphiti')

        assert result.get('error') == 'service_not_configured'
        assert result.get('error_type') == 'ReconReportServiceUnavailable'


class TestReconReportComponentsWiring:
    """Verify _build_recon_report_components passes services into the state."""

    def _make_config(self):
        from fused_memory.config.schema import (
            FusedMemoryConfig,
            ReconciliationConfig,
            ServerConfig,
        )

        return FusedMemoryConfig(
            server=ServerConfig(recon_report_port=8003, host='127.0.0.1'),
            reconciliation=ReconciliationConfig(recon_report_state_ttl_seconds=300),
        )

    def test_services_injected_when_provided(self):
        """State returned by _build_recon_report_components has injected services."""
        from fused_memory.server.main import _build_recon_report_components

        fake_ms = _FakeMemoryService()
        fake_ti = _FakeTaskInterceptor()
        kp = {'p': '/root'}
        config = self._make_config()

        state, _, _ = _build_recon_report_components(
            config,
            memory_service=fake_ms,
            task_interceptor=fake_ti,
            known_projects=kp,
        )

        assert state._memory_service is fake_ms  # type: ignore[attr-defined]
        assert state._task_interceptor is fake_ti  # type: ignore[attr-defined]
        assert state.known_projects == kp  # type: ignore[attr-defined]

    def test_existing_config_only_call_still_works(self):
        """_build_recon_report_components(config) with no extras stays backwards-compatible."""
        from fused_memory.server.main import _build_recon_report_components

        config = self._make_config()
        state, mcp, uv_cfg = _build_recon_report_components(config)

        # Basic sanity — still returns 3-tuple with correct types
        assert state._ttl_seconds == 300  # type: ignore[attr-defined]
        assert mcp.name == 'Recon Report'
        assert uv_cfg.port == 8003
