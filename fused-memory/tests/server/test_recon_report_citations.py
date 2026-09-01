"""Tests for the recon_report cite_* citation tools (task β).

Covers:
- TestCiteEntity  — cite_entity happy path, entity_not_found, finding_unknown, accumulation
- TestCiteEdge    — cite_edge UUID shape gate, edge_not_found, happy path, finding_unknown
- TestCiteTask    — cite_task happy path, unknown_project, task_not_found, cross-finding P5
- TestCiteMemory  — cite_memory UUID shape gate, memory_not_found, happy paths (both stores)
- TestCiteRun      — cite_run UUID shape gate, run_not_found, happy path, finding_unknown (task 2595)
- TestCiteToolsViaFastMCP — tools registered, end-to-end via call_tool, P4 schema rejection
- TestReconReportComponentsWiring — _build_recon_report_components service injection
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from mcp.server.fastmcp.exceptions import ToolError

# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeMemoryService:
    """Lightweight async stub for MemoryService (cite_entity / cite_edge / cite_memory / cite_run)."""

    def __init__(
        self,
        *,
        entity_nodes: list[dict] | None = None,
        edge_result: dict | None = None,
        edge_raises=None,
        memory_result: dict | None = None,
        memory_raises=None,
        count_result: int | None = None,
    ) -> None:
        # get_entity configuration
        self._entity_nodes: list[dict] = entity_nodes if entity_nodes is not None else []

        # get_edge configuration
        self._edge_result = edge_result
        self._edge_raises = edge_raises  # exception instance to raise

        # get_memory configuration
        self._memory_result = memory_result
        self._memory_raises = memory_raises

        # count_memories_by_metadata configuration (cite_run, task 2595)
        self._count_result: int = count_result if count_result is not None else 0

        # Call tracking
        self.get_entity_calls: list[tuple[str, str]] = []
        self.get_edge_calls: list[tuple[str, str]] = []
        self.get_memory_calls: list[tuple[str, str, str]] = []
        self.count_by_metadata_calls: list[tuple[str, dict]] = []

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

    async def count_memories_by_metadata(self, project_id: str, filters: dict) -> int:
        self.count_by_metadata_calls.append((project_id, filters))
        return self._count_result


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
    async def test_invalid_uuid_shape_trailing_newline(self):
        """A canonical UUID with a trailing newline → invalid_uuid_shape.

        Not hypothetical: the anchored `^...$` regex this gate used to apply
        with `.match()` ACCEPTED it, because Python's `$` matches immediately
        before a trailing newline. Such an id passed the gate and then resolved
        to nothing downstream — the same silent no-op delete_memory now
        hard-errors on (task 3132).
        """
        state, run_id, finding_id, fake = self._state_and_finding()

        result = await state.cite_edge(run_id, finding_id, self._VALID_UUID + '\n')

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
# task-2425 step-1: TestCiteTaskInRunCitedTaskDedup — RED until step-2 adds
# the in-run cited-task fold to ReconReportState.cite_task
# ---------------------------------------------------------------------------


class TestCiteTaskInRunCitedTaskDedup:
    """In-run dedup fold (task-2425): two null-task_id findings that cite the
    SAME primary external task are semantically duplicate regardless of how
    differently they are worded (different category/flag_type/description),
    because they evade BOTH add_finding in-run guards — the (task_id,
    flag_type) signature path (task_id=None on both is not itself a
    collision) and the null-null description-content-hash path (task-1654;
    differently-worded descriptions hash differently). The fold can only
    live in cite_task: it is the first point after add_finding where the
    finding's primary cited external task identity becomes known.
    """

    def _fake_ti(self):
        """Fake task interceptor covering the external task ids these tests cite."""
        known_roots = {'/home/leo/src/dark-factory'}
        results = {}
        for pr in known_roots:
            for tid in ['2405', '2406', '9999']:
                results[(tid, pr)] = {'id': tid, 'title': f'T-{tid}'}
        return _FakeTaskInterceptor(results=results)

    def _make_state(self, fake_ti=None, memory_service=None):
        from fused_memory.server.recon_report import ReconReportState

        if fake_ti is None:
            fake_ti = self._fake_ti()
        t = [0.0]
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: t[0],
            task_interceptor=fake_ti,
            memory_service=memory_service,
        )
        state.known_projects = dict(_KNOWN_PROJECTS)
        return state, t

    @pytest.mark.asyncio
    async def test_core_repro_second_citer_of_same_task_folds_as_duplicate(self):
        """CORE REPRO: two null-task_id findings, worded differently (one
        null-null, one with a distinct non-null flag_type so it evades the
        (task_id, flag_type) signature guard too), both cite the SAME
        external task. The second cite_task must fold: return
        duplicate_finding pointing at the first finding, and the assembled
        report must retain exactly that one finding.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='reconciler', project_id='dark_factory')

        first = state.add_finding(
            run_id='run-1',
            severity='moderate',
            category='cross_project',
            description='dark_factory:2405 still pending, unchanged',
            suggested_action='wait for upstream',
            actionable=True,
            task_id=None,
            flag_type=None,
        )
        assert 'finding_id' in first, f'add_finding failed: {first}'
        finding_id_1 = first['finding_id']

        second = state.add_finding(
            run_id='run-1',
            severity='moderate',
            category='cross_project_routing',
            description='blocked pending dark_factory task 2405 per routing check',
            suggested_action='reroute once unblocked',
            actionable=True,
            task_id=None,
            flag_type='cross_project_routing_stale',
        )
        assert 'finding_id' in second, f'add_finding failed: {second}'
        finding_id_2 = second['finding_id']
        # Sanity: both guards in add_finding were evaded — two distinct
        # findings were allocated (this is the bug's precondition).
        assert finding_id_2 != finding_id_1

        cite_1 = await state.cite_task('run-1', finding_id_1, 'dark_factory', '2405')
        assert cite_1.get('project_id') == 'dark_factory'
        assert cite_1.get('task_id') == '2405'

        cite_2 = await state.cite_task('run-1', finding_id_2, 'dark_factory', '2405')
        assert cite_2.get('error') == 'duplicate_finding'
        assert cite_2.get('error_type') == 'ReconReportDuplicateFinding'
        assert cite_2.get('existing_finding_id') == finding_id_1

        # The folded (second) finding is fully purged, not just hidden.
        assert state._resolve_finding('run-1', finding_id_2) is None

        report = state.get_assembled_report('run-1', 'reconciler')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [finding_id_1]

    @pytest.mark.asyncio
    async def test_different_primary_cited_tasks_both_survive(self):
        """Two null-task_id findings citing DIFFERENT external tasks must
        both keep their citation — no false-positive fold.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='reconciler', project_id='dark_factory')

        f1 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='desc A', suggested_action='a',
            task_id=None, flag_type=None,
        )
        f2 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='desc B', suggested_action='a',
            task_id=None, flag_type='other_flag',
        )
        fid1, fid2 = f1['finding_id'], f2['finding_id']

        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
        cite2 = await state.cite_task('run-1', fid2, 'dark_factory', '2406')

        assert 'error' not in cite1, cite1
        assert 'error' not in cite2, cite2

        report = state.get_assembled_report('run-1', 'reconciler')
        assert report is not None
        ids = {item['finding_id'] for item in report['flagged_items']}
        assert ids == {fid1, fid2}

    @pytest.mark.asyncio
    async def test_non_null_task_id_finding_is_never_a_fold_anchor(self):
        """A finding with a REAL top-level task_id is never folded, even if
        it cites the same external task that a null-task_id finding also
        cites — the fold is gated strictly on finding.task_id is None.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='reconciler', project_id='dark_factory')

        f_null = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='null task_id desc', suggested_action='a',
            task_id=None, flag_type=None,
        )
        f_real = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='real task_id desc', suggested_action='a',
            task_id='9876', flag_type='some_flag',
        )
        fid_null, fid_real = f_null['finding_id'], f_real['finding_id']

        cite_null = await state.cite_task('run-1', fid_null, 'dark_factory', '2405')
        cite_real = await state.cite_task('run-1', fid_real, 'dark_factory', '2405')

        assert 'error' not in cite_null, cite_null
        assert 'error' not in cite_real, cite_real

        report = state.get_assembled_report('run-1', 'reconciler')
        assert report is not None
        ids = {item['finding_id'] for item in report['flagged_items']}
        assert ids == {fid_null, fid_real}

    @pytest.mark.asyncio
    async def test_cross_run_isolation_same_primary_task_both_survive(self):
        """The SAME primary external task cited from two different run_ids
        must not fold across runs.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='reconciler', project_id='dark_factory')
        state.start_report(run_id='run-2', stage='reconciler', project_id='dark_factory')

        f1 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='run1 desc', suggested_action='a',
            task_id=None, flag_type=None,
        )
        f2 = state.add_finding(
            run_id='run-2', severity='low', category='cross_project',
            description='run2 desc', suggested_action='a',
            task_id=None, flag_type=None,
        )
        fid1, fid2 = f1['finding_id'], f2['finding_id']

        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
        cite2 = await state.cite_task('run-2', fid2, 'dark_factory', '2405')

        assert 'error' not in cite1, cite1
        assert 'error' not in cite2, cite2

        report1 = state.get_assembled_report('run-1', 'reconciler')
        report2 = state.get_assembled_report('run-2', 'reconciler')
        assert report1 is not None and report2 is not None
        assert [i['finding_id'] for i in report1['flagged_items']] == [fid1]
        assert [i['finding_id'] for i in report2['flagged_items']] == [fid2]

    @pytest.mark.asyncio
    async def test_only_primary_citation_is_a_fold_anchor(self):
        """Citing a SECOND task on a finding must not register a fold
        anchor for that second task, and re-citing the finding's own
        primary task again on the SAME finding must not self-collapse.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='reconciler', project_id='dark_factory')

        f1 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='desc one', suggested_action='a',
            task_id=None, flag_type=None,
        )
        fid1 = f1['finding_id']

        cite_primary = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
        assert 'error' not in cite_primary, cite_primary

        # Secondary citation on the SAME finding — must succeed plainly.
        cite_secondary = await state.cite_task('run-1', fid1, 'dark_factory', '9999')
        assert 'error' not in cite_secondary, cite_secondary

        # A DIFFERENT finding citing '9999' as ITS primary must still
        # succeed — proving the secondary citation above did not register
        # '9999' as a fold anchor.
        f2 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='desc two', suggested_action='a',
            task_id=None, flag_type='another_flag',
        )
        fid2 = f2['finding_id']
        cite_other_primary = await state.cite_task('run-1', fid2, 'dark_factory', '9999')
        assert 'error' not in cite_other_primary, cite_other_primary

        # Re-citing finding 1's own primary task again must not self-collapse.
        cite_primary_again = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
        assert 'error' not in cite_primary_again, cite_primary_again

        report = state.get_assembled_report('run-1', 'reconciler')
        assert report is not None
        ids = {item['finding_id'] for item in report['flagged_items']}
        assert ids == {fid1, fid2}

        # task-2425 amend: re-citing the same primary task must not create a
        # duplicate citation entry either -- fid1's cited_tasks stays at its
        # two distinct citations ('2405' primary + '9999' secondary), not
        # three with '2405' appearing twice.
        item_1 = next(i for i in report['flagged_items'] if i['finding_id'] == fid1)
        cited_pairs = [(c['project_id'], c['task_id']) for c in item_1['cited_tasks']]
        assert cited_pairs == [('dark_factory', '2405'), ('dark_factory', '9999')]

    @pytest.mark.asyncio
    async def test_two_memory_consolidator_findings_citing_same_task_both_survive(self):
        """memory_consolidator (Stage-1) findings are EXEMPT from the in-run
        cited-task fold (esc-2425-1): the fold is gated on
        ``finding_entry.stage != 'memory_consolidator'``, mirroring Fix-1's
        read-time ``stage != 'memory_consolidator'`` carve-out in
        get_assembled_report. Two Stage-1 siblings citing the SAME external
        task as their primary citation must both survive -- neither is
        folded/purged -- exactly like
        test_sibling_stage1_findings_not_mutually_suppressed in
        test_recon_report.py, but exercised at the write-time fold rather
        than the read-time suppression.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='memory_consolidator', project_id='dark_factory')

        f1 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='mc desc one', suggested_action='a',
            task_id=None, flag_type=None,
        )
        f2 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='mc desc two', suggested_action='a',
            task_id=None, flag_type='other_flag',
        )
        fid1, fid2 = f1['finding_id'], f2['finding_id']

        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
        cite2 = await state.cite_task('run-1', fid2, 'dark_factory', '2405')

        assert 'error' not in cite1, cite1
        assert 'error' not in cite2, cite2

        report = state.get_assembled_report('run-1', 'memory_consolidator')
        assert report is not None
        ids = {item['finding_id'] for item in report['flagged_items']}
        assert ids == {fid1, fid2}

    @pytest.mark.asyncio
    async def test_memory_consolidator_finding_does_not_consume_fold_anchor_for_other_stage(self):
        """A memory_consolidator-stage citation must not register a fold
        anchor that a later non-mc-stage finding relies on. Stage 1 cites an
        external task first; Stage 2 (reconciler) then cites the SAME task
        as ITS OWN primary citation -- it must succeed plainly (not fold),
        proving the Stage-1 citation registered nothing in
        ``_run_cited_task_index`` for the non-mc stage to collide with.
        """
        state, _ = self._make_state()

        state.start_report(run_id='run-1', stage='memory_consolidator', project_id='dark_factory')
        f_mc = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='mc desc', suggested_action='a',
            task_id=None, flag_type=None,
        )
        fid_mc = f_mc['finding_id']
        cite_mc = await state.cite_task('run-1', fid_mc, 'dark_factory', '2405')
        assert 'error' not in cite_mc, cite_mc

        state.start_report(run_id='run-1', stage='reconciler', project_id='dark_factory')
        # actionable=True is explicit (task-2432): this null-task_id/
        # cross_project finding would otherwise pick up the new computed
        # default of False, which would trip the unrelated Fix-1 read-time
        # suppression (its lone citation is a subset of f_mc's) and hide it
        # from report_recon below for a reason orthogonal to what this test
        # verifies (write-time fold-anchor isolation across stages).
        f_recon = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='recon desc', suggested_action='a',
            actionable=True, task_id=None, flag_type='recon_flag',
        )
        fid_recon = f_recon['finding_id']
        cite_recon = await state.cite_task('run-1', fid_recon, 'dark_factory', '2405')
        assert 'error' not in cite_recon, cite_recon

        report_mc = state.get_assembled_report('run-1', 'memory_consolidator')
        report_recon = state.get_assembled_report('run-1', 'reconciler')
        assert report_mc is not None and report_recon is not None
        assert [i['finding_id'] for i in report_mc['flagged_items']] == [fid_mc]
        assert [i['finding_id'] for i in report_recon['flagged_items']] == [fid_recon]

    @pytest.mark.asyncio
    async def test_folded_finding_discards_previously_recorded_citations(self):
        """task-2425 amend: _purge_finding removes a folded finding
        wholesale, not just its dedup-index entries. If the folded finding
        had already accumulated a cite_entity/cite_edge/cite_memory citation
        BEFORE the cite_task call that folds it, that citation is discarded
        along with the finding -- it is not merged onto the surviving
        finding. Pins the behavior documented in _purge_finding's and
        cite_task's docstrings so it stays intentional rather than an
        undetected regression.
        """
        fake_ms = _FakeMemoryService(
            entity_nodes=[{'uuid': 'aaaaaaaa-1111-1111-1111-111111111111', 'name': 'Foo'}]
        )
        state, _ = self._make_state(memory_service=fake_ms)
        state.start_report(run_id='run-1', stage='reconciler', project_id='dark_factory')

        first = state.add_finding(
            run_id='run-1', severity='moderate', category='cross_project',
            description='desc one', suggested_action='a',
            task_id=None, flag_type=None,
        )
        fid1 = first['finding_id']
        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
        assert 'error' not in cite1, cite1

        second = state.add_finding(
            run_id='run-1', severity='moderate', category='cross_project_routing',
            description='desc two', suggested_action='a',
            task_id=None, flag_type='other_flag',
        )
        fid2 = second['finding_id']

        # fid2 accumulates a cite_entity citation BEFORE the cite_task call
        # that folds it into fid1.
        cite_entity_result = await state.cite_entity('run-1', fid2, 'Foo')
        assert 'error' not in cite_entity_result, cite_entity_result

        fold_result = await state.cite_task('run-1', fid2, 'dark_factory', '2405')
        assert fold_result.get('error') == 'duplicate_finding'
        assert fold_result.get('existing_finding_id') == fid1

        # fid2 is gone entirely -- its cite_entity citation is not
        # resurrected anywhere, including on the surviving fid1.
        assert state._resolve_finding('run-1', fid2) is None
        report = state.get_assembled_report('run-1', 'reconciler')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [fid1]
        assert report['flagged_items'][0]['cited_entities'] == []


# ---------------------------------------------------------------------------
# task-2432 step-7: TestCiteTaskEntityScopedFold — RED (single-id) until
# step-8 adds an entity-scoped derived-signature fold to cite_task, keyed on
# the finding's PRIMARY cited task rather than its top-level task_id, so a
# null/foreign-single/local-single top-level task_id all collapse onto the
# same cited-task anchor. Comma-joined subset eligibility is step-9/step-10.
# ---------------------------------------------------------------------------


class TestCiteTaskEntityScopedFold:
    """Entity-scoped in-run fold (task-2432 bullets 1b/2/3): two findings
    whose top-level task_id SHAPES differ (None vs a real task_id equal to
    what the null one cites) are semantically duplicate once the null one's
    primary citation names the same task the other's top-level task_id
    already names — regardless of call order or which stage each was filed
    in. Distinct from task-2425's project-scoped null+null fold, which never
    fires once either finding has a non-null top-level task_id.
    """

    def _fake_ti(self):
        """Fake task interceptor covering the external task ids these tests cite."""
        known_roots = {'/home/leo/src/dark-factory'}
        results = {}
        for pr in known_roots:
            for tid in ['2405', '2406', '5040', '5149', '7777', '9999']:
                results[(tid, pr)] = {'id': tid, 'title': f'T-{tid}'}
        return _FakeTaskInterceptor(results=results)

    def _make_state(self, fake_ti=None):
        from fused_memory.server.recon_report import ReconReportState

        if fake_ti is None:
            fake_ti = self._fake_ti()
        t = [0.0]
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: t[0],
            task_interceptor=fake_ti,
        )
        state.known_projects = dict(_KNOWN_PROJECTS)
        return state, t

    # -- CORE (RED) ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_null_cited_then_top_level_task_id_folds(self):
        """(1) occ1 is null-task_id and cites '2405'; occ2's TOP-LEVEL
        task_id is '2405' -- occ2 must fold as a duplicate of occ1 via
        occ1's cite_task-registered derived signature.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        occ1 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ1 desc', suggested_action='a',
            task_id=None, flag_type='cross_project',
        )
        assert 'finding_id' in occ1, occ1
        fid1 = occ1['finding_id']

        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
        assert 'error' not in cite1, cite1

        occ2 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ2 desc, worded differently', suggested_action='a',
            task_id='2405', flag_type='cross_project',
        )
        assert occ2.get('error') == 'duplicate_finding'
        assert occ2.get('existing_finding_id') == fid1

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [fid1]

    @pytest.mark.asyncio
    async def test_top_level_task_id_then_null_cited_folds_reverse_order(self):
        """(2) Reverse order: occ2's TOP-LEVEL task_id ('2405') is filed
        FIRST; occ1 (null-task_id) is filed second and then cites '2405' --
        the cite_task call itself must fold occ1 into occ2 and return
        duplicate_finding, purging occ1.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        occ2 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ2 desc', suggested_action='a',
            task_id='2405', flag_type='cross_project',
        )
        assert 'finding_id' in occ2, occ2
        fid2 = occ2['finding_id']

        occ1 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ1 desc, worded differently', suggested_action='a',
            task_id=None, flag_type='cross_project',
        )
        assert 'finding_id' in occ1, occ1
        fid1 = occ1['finding_id']
        assert fid1 != fid2

        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
        assert cite1.get('error') == 'duplicate_finding'
        assert cite1.get('existing_finding_id') == fid2

        # The folded (null-task_id) finding is fully purged, not just hidden.
        assert state._resolve_finding('run-1', fid1) is None

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [fid2]

    @pytest.mark.asyncio
    async def test_same_project_single_task_id_folds(self):
        """(3) Same shape as (1) with a different task/flag pairing, pinning
        this is not specific to one task id or flag_type value.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        occ_a = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ_a desc', suggested_action='a',
            task_id=None, flag_type='X',
        )
        assert 'finding_id' in occ_a, occ_a
        fid_a = occ_a['finding_id']

        cite_a = await state.cite_task('run-1', fid_a, 'dark_factory', '5040')
        assert 'error' not in cite_a, cite_a

        occ_b = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ_b desc, worded differently', suggested_action='a',
            task_id='5040', flag_type='X',
        )
        assert occ_b.get('error') == 'duplicate_finding'
        assert occ_b.get('existing_finding_id') == fid_a

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [fid_a]

    @pytest.mark.asyncio
    async def test_cross_stage_fold_no_mc_carveout_on_derived_sig(self):
        """(4) The derived-sig fold has NO memory_consolidator carve-out
        (unlike the task-2425 project-scoped fold): an anchor registered by
        a Stage-1 (memory_consolidator) finding's cite_task call still
        collapses a later Stage-2 (task_knowledge_sync) finding whose
        top-level task_id equals that citation.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='memory_consolidator', project_id='dark_factory')

        occ_mc = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='mc desc', suggested_action='a',
            task_id=None, flag_type='Y',
        )
        assert 'finding_id' in occ_mc, occ_mc
        fid_mc = occ_mc['finding_id']

        cite_mc = await state.cite_task('run-1', fid_mc, 'dark_factory', '7777')
        assert 'error' not in cite_mc, cite_mc

        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')
        occ_tks = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='tks desc, worded differently', suggested_action='a',
            task_id='7777', flag_type='Y',
        )
        assert occ_tks.get('error') == 'duplicate_finding'
        assert occ_tks.get('existing_finding_id') == fid_mc

        report_mc = state.get_assembled_report('run-1', 'memory_consolidator')
        assert report_mc is not None
        ids_mc = [item['finding_id'] for item in report_mc['flagged_items']]
        assert ids_mc == [fid_mc]

        report_tks = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report_tks is not None
        assert report_tks['flagged_items'] == []

    # -- REGRESSION GUARDS (stay green) --------------------------------

    @pytest.mark.asyncio
    async def test_different_cited_task_does_not_fold_against_sibling_top_level(self):
        """(5) A finding citing a DIFFERENT task than a sibling's top-level
        task_id must not fold -- both survive.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        occ_top = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='top desc', suggested_action='a',
            task_id='2405', flag_type='Z',
        )
        assert 'finding_id' in occ_top, occ_top
        fid_top = occ_top['finding_id']

        occ_null = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='null desc', suggested_action='a',
            task_id=None, flag_type='Z',
        )
        assert 'finding_id' in occ_null, occ_null
        fid_null = occ_null['finding_id']

        cite_result = await state.cite_task('run-1', fid_null, 'dark_factory', '2406')
        assert 'error' not in cite_result, cite_result

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = {item['finding_id'] for item in report['flagged_items']}
        assert ids == {fid_top, fid_null}

    @pytest.mark.asyncio
    async def test_same_cited_task_different_flag_type_both_survive(self):
        """(6) Same cited task, different flag_type -- the derived sig's
        flag_type component keeps them distinct; both survive.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        occ1 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ1 desc', suggested_action='a',
            task_id=None, flag_type='flag_one',
        )
        assert 'finding_id' in occ1, occ1
        fid1 = occ1['finding_id']

        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
        assert 'error' not in cite1, cite1

        occ2 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ2 desc', suggested_action='a',
            task_id='2405', flag_type='flag_two',
        )
        assert 'finding_id' in occ2, occ2
        fid2 = occ2['finding_id']

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = {item['finding_id'] for item in report['flagged_items']}
        assert ids == {fid1, fid2}

    @pytest.mark.asyncio
    async def test_local_task_id_citing_different_task_is_not_folded(self):
        """(7) NO over-fold: a finding with a LOCAL top-level task_id
        ('100') that cites a DIFFERENT, foreign task ('9999') must not be
        folded against a pure null-task_id finding that cites that same
        foreign task as ITS primary citation -- '9999' is not-in {'100'}.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        occ_null = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='null desc', suggested_action='a',
            task_id=None, flag_type='W',
        )
        assert 'finding_id' in occ_null, occ_null
        fid_null = occ_null['finding_id']

        cite_null = await state.cite_task('run-1', fid_null, 'dark_factory', '9999')
        assert 'error' not in cite_null, cite_null

        occ_local = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='local desc', suggested_action='a',
            task_id='100', flag_type='W',
        )
        assert 'finding_id' in occ_local, occ_local
        fid_local = occ_local['finding_id']

        cite_local = await state.cite_task('run-1', fid_local, 'dark_factory', '9999')
        assert 'error' not in cite_local, cite_local

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = {item['finding_id'] for item in report['flagged_items']}
        assert ids == {fid_null, fid_local}

    # -- CORE (RED, comma-joined — task-2432 step-9) ------------------
    #
    # step-8's fold-eligibility gate is EQUALITY: canonical(finding.task_id)
    # == canonical(cited task_id). A comma-joined top-level task_id like
    # '5040,5149' is never equal to either of its own individual components,
    # so a comma-joined finding is fold-INELIGIBLE on both sides of the
    # check -- it can neither be folded onto an existing single-component
    # anchor, nor register an anchor a later single-component finding could
    # fold onto. step-10 widens the gate to subset (comma-split) membership.

    @pytest.mark.asyncio
    async def test_comma_joined_top_level_folds_via_matching_component_cite(self):
        """(1) occ1 (null) cites '5040' then '5149'; occ_c's TOP-LEVEL
        task_id is the comma-joined '5040,5149' -- occ_c's cite_task('5040')
        must fold onto occ1 via the '5040' component anchor occ1's cite_task
        registered.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        occ1 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ1 desc', suggested_action='a',
            task_id=None, flag_type='X',
        )
        assert 'finding_id' in occ1, occ1
        fid1 = occ1['finding_id']

        cite1_5040 = await state.cite_task('run-1', fid1, 'dark_factory', '5040')
        assert 'error' not in cite1_5040, cite1_5040
        cite1_5149 = await state.cite_task('run-1', fid1, 'dark_factory', '5149')
        assert 'error' not in cite1_5149, cite1_5149

        occ_c = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ_c desc, worded differently', suggested_action='a',
            task_id='5040,5149', flag_type='X',
        )
        assert 'finding_id' in occ_c, occ_c
        fid_c = occ_c['finding_id']

        cite_c = await state.cite_task('run-1', fid_c, 'dark_factory', '5040')
        assert cite_c.get('error') == 'duplicate_finding'
        assert cite_c.get('existing_finding_id') == fid1

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [fid1]

    @pytest.mark.asyncio
    async def test_comma_joined_top_level_folds_via_matching_component_cite_reverse_order(self):
        """(2) Reverse order: occ_c (comma-joined '5040,5149') is filed and
        cites '5040' FIRST; occ1 (null) is filed second and its cite_task
        ('5040') must fold onto occ_c -- occ_c's cite_task call must
        register an anchor for its '5040' component, not just its full
        comma-joined string.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        occ_c = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ_c desc', suggested_action='a',
            task_id='5040,5149', flag_type='X',
        )
        assert 'finding_id' in occ_c, occ_c
        fid_c = occ_c['finding_id']

        cite_c = await state.cite_task('run-1', fid_c, 'dark_factory', '5040')
        assert 'error' not in cite_c, cite_c

        occ1 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ1 desc, worded differently', suggested_action='a',
            task_id=None, flag_type='X',
        )
        assert 'finding_id' in occ1, occ1
        fid1 = occ1['finding_id']

        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '5040')
        assert cite1.get('error') == 'duplicate_finding'
        assert cite1.get('existing_finding_id') == fid_c

        # The folded (null-task_id) finding is fully purged, not just hidden.
        assert state._resolve_finding('run-1', fid1) is None

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [fid_c]

    # -- Purge/refile clears the derived sig (task-2432 step-11) -------
    #
    # _purge_finding (as of step-10) only clears the ORDINARY sig
    # (finding.task_id, finding.flag_type) and the project-scoped
    # _run_cited_task_index anchor -- it does not know about the derived
    # (cited_task_id, flag_type) key the entity-scoped fold registers into
    # _run_sig_index. Deleting a finding that was a derived-sig anchor
    # therefore leaves a stale pointer a later refile bounces off. step-12
    # extends _purge_finding to also clear it.

    @pytest.mark.asyncio
    async def test_delete_then_refile_matching_top_level_task_id_succeeds(self):
        """Deleting a null-task_id finding that cited '5040' (registering
        derived sig ('5040', 'X')) must clear that derived sig so a LATER
        finding whose top-level task_id is '5040' allocates fresh instead of
        bouncing off the deleted finding's stale pointer.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        occ1 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ1 desc', suggested_action='a',
            task_id=None, flag_type='X',
        )
        assert 'finding_id' in occ1, occ1
        fid1 = occ1['finding_id']

        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '5040')
        assert 'error' not in cite1, cite1

        delete_result = state.delete_finding('run-1', fid1)
        assert delete_result == {'status': 'deleted', 'finding_id': fid1}

        occ2 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ2 desc', suggested_action='a',
            task_id='5040', flag_type='X',
        )
        assert 'finding_id' in occ2, occ2

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [occ2['finding_id']]

    @pytest.mark.asyncio
    async def test_delete_then_refile_comma_joined_top_level_task_id_succeeds(self):
        """Comma variant: after deleting a folded finding whose primary cite
        was '5040', a later comma-joined finding ('5040,5149') that cites
        '5040' must allocate and keep fresh, not fold onto (and be purged
        by) the deleted finding's stale derived-sig pointer.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        occ1 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ1 desc', suggested_action='a',
            task_id=None, flag_type='X',
        )
        assert 'finding_id' in occ1, occ1
        fid1 = occ1['finding_id']

        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '5040')
        assert 'error' not in cite1, cite1

        delete_result = state.delete_finding('run-1', fid1)
        assert delete_result == {'status': 'deleted', 'finding_id': fid1}

        occ3 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ3 desc', suggested_action='a',
            task_id='5040,5149', flag_type='X',
        )
        assert 'finding_id' in occ3, occ3
        fid3 = occ3['finding_id']

        cite3 = await state.cite_task('run-1', fid3, 'dark_factory', '5040')
        assert 'error' not in cite3, cite3

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [fid3]


# ---------------------------------------------------------------------------
# task-4185: TestCiteTaskCrossProjectNearCollision — the entity-scoped fold's
# derived signature is PROJECTLESS, so two projects' findings about
# same-numbered tasks can collide. step-1 pins the three folds that must NOT
# change; step-2/4/6 drive the guard itself.
# ---------------------------------------------------------------------------


class TestCiteTaskCrossProjectNearCollision:
    """Cross-project near-collisions on the entity-scoped derived signature
    (task-4185).

    ``_run_sig_index``'s derived key is ``(canonical(cited task_id),
    flag_type)`` — deliberately PROJECTLESS. Operator ruling (2026-08-12):
    the projectless key STAYS, because it is what lets a bare top-level
    task_id (which names no project at all) fold onto a foreign citation,
    and that fold is INTENDED. Only the DETECTABLE half is guarded — a
    cite_task→cite_task collision where the ANCHOR's own primary citation
    pins the same task id to a different project.

    The three tests below are CHARACTERIZATION PINS: each was confirmed to
    fold on unmodified code and must keep folding after the guard lands, so
    the guard cannot silently over-reach.
    """

    def _fake_ti(self):
        """Fake task interceptor covering the cited ids under BOTH known roots.

        These tests cite the SAME numeric task id from two different
        projects, so every id must resolve under each of ``_KNOWN_PROJECTS``'
        roots — otherwise a foreign citation would fail with
        ``task_not_found`` before ever reaching the fold logic.
        """
        results = {}
        for pr in _KNOWN_PROJECTS.values():
            for tid in ['42', '99', '999', '2405', '7777']:
                results[(tid, pr)] = {'id': tid, 'title': f'T-{tid}'}
        return _FakeTaskInterceptor(results=results)

    def _make_state(self, fake_ti=None):
        from fused_memory.server.recon_report import ReconReportState

        if fake_ti is None:
            fake_ti = self._fake_ti()
        t = [0.0]
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: t[0],
            task_interceptor=fake_ti,
        )
        state.known_projects = dict(_KNOWN_PROJECTS)
        return state, t

    # -- CHARACTERIZATION PINS (green on unmodified code) ---------------

    @pytest.mark.asyncio
    async def test_same_project_cite_task_collision_still_folds(self):
        """(b) A genuine SAME-project duplicate must keep folding.

        The anchor's primary citation pins ('dark_factory', '2405'); the
        incoming citation names the same task in the same project, so this
        is an ordinary duplicate, not a near-collision.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        anchor = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='anchor 2405', suggested_action='a',
            task_id='2405', flag_type='X',
        )
        assert 'finding_id' in anchor, anchor
        anchor_id = anchor['finding_id']

        # Self-hit: registers the derived sig ('2405', 'X') AND records
        # cited_tasks[0] == {'dark_factory', '2405'} as the project pin.
        cite_anchor = await state.cite_task('run-1', anchor_id, 'dark_factory', '2405')
        assert 'error' not in cite_anchor, cite_anchor

        incoming = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='incoming 2405, worded differently', suggested_action='a',
            task_id=None, flag_type='X',
        )
        assert 'finding_id' in incoming, incoming
        incoming_id = incoming['finding_id']

        result = await state.cite_task('run-1', incoming_id, 'dark_factory', '2405')
        assert result.get('error') == 'duplicate_finding'
        assert result.get('existing_finding_id') == anchor_id

        assert state._resolve_finding('run-1', incoming_id) is None

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [anchor_id]

    @pytest.mark.asyncio
    async def test_anchor_citing_a_different_task_still_folds(self):
        """(c3) An anchor whose primary citation names a DIFFERENT task
        carries NO project pin for this derived signature — it must keep
        folding.

        The anchor's top-level task_id is '42' (which is what registered the
        ordinary ('42', 'X') signature add_finding consults), but its only
        citation is other_project:999 — '999' is not a member of {'42'}, so
        that cite_task call registered no derived sig at all. A guard that
        compared cited_tasks[0]['project_id'] against the incoming
        project_id WITHOUT first checking the citation names the SAME task
        would see 'other_project' != 'dark_factory' and wrongly break this
        pre-existing fold.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        anchor = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='anchor 42', suggested_action='a',
            task_id='42', flag_type='X',
        )
        assert 'finding_id' in anchor, anchor
        anchor_id = anchor['finding_id']

        cite_anchor = await state.cite_task('run-1', anchor_id, 'other_project', '999')
        assert 'error' not in cite_anchor, cite_anchor

        incoming = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='incoming 42, worded differently', suggested_action='a',
            task_id=None, flag_type='X',
        )
        assert 'finding_id' in incoming, incoming
        incoming_id = incoming['finding_id']

        result = await state.cite_task('run-1', incoming_id, 'dark_factory', '42')
        assert result.get('error') == 'duplicate_finding'
        assert result.get('existing_finding_id') == anchor_id

    @pytest.mark.asyncio
    async def test_foreign_citation_then_bare_add_finding_still_folds(self):
        """(c2) The shape the scope ruling explicitly PROTECTS: a bare
        top-level task_id folding onto a FOREIGN citation.

        A null-task_id finding cites other_project:2405, registering the
        projectless derived sig ('2405', 'X'); a later
        ``add_finding(task_id='2405', flag_type='X')`` — which names no
        project whatsoever — collapses onto it via add_finding's own
        ordinary signature lookup. This path never enters cite_task's guard;
        the test exists so a future attempt to project-namespace the derived
        key fails loudly here.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        citing = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='citing foreign 2405', suggested_action='a',
            task_id=None, flag_type='X',
        )
        assert 'finding_id' in citing, citing
        citing_id = citing['finding_id']

        cite_result = await state.cite_task('run-1', citing_id, 'other_project', '2405')
        assert 'error' not in cite_result, cite_result

        later = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='bare 2405, worded differently', suggested_action='a',
            task_id='2405', flag_type='X',
        )
        assert later.get('error') == 'duplicate_finding'
        assert later.get('existing_finding_id') == citing_id

    # -- THE BUG --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cross_project_near_collision_skips_the_fold(self):
        """Two projects' findings about same-numbered tasks must BOTH survive.

        The anchor's primary citation pins ('dark_factory', '42'); the
        incoming citation names task '42' in other_project — a different
        task entirely that merely shares a number. Before the guard, the
        projectless derived sig ('42', 'X') made the incoming call return
        duplicate_finding and purged the foreign finding WHOLESALE, its
        content surviving only in the task-4184 fold-purge WARNING. With 9
        registered project roots all carrying small-integer Taskmaster ids,
        that numeric overlap is guaranteed-possible.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        local = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='local 42 is stale', suggested_action='a',
            task_id='42', flag_type='X',
        )
        assert 'finding_id' in local, local
        local_id = local['finding_id']

        # Self-hit: registers the derived sig ('42', 'X') and pins
        # cited_tasks[0]['project_id'] == 'dark_factory'.
        cite_local = await state.cite_task('run-1', local_id, 'dark_factory', '42')
        assert 'error' not in cite_local, cite_local

        # Its own add_finding sig is (None, 'X'), so it allocates fresh.
        foreign = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='foreign 42, worded differently', suggested_action='a',
            task_id=None, flag_type='X',
        )
        assert 'finding_id' in foreign, foreign
        foreign_id = foreign['finding_id']

        result = await state.cite_task('run-1', foreign_id, 'other_project', '42')

        # Ordinary citation dict, NOT duplicate_finding.
        assert 'error' not in result, result
        assert result['project_id'] == 'other_project'
        assert result['task_id'] == '42'

        # Not purged.
        assert state._resolve_finding('run-1', foreign_id) is not None

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = {item['finding_id'] for item in report['flagged_items']}
        assert ids == {local_id, foreign_id}

        # Each finding kept its OWN citation.
        local_resolved = state._resolve_finding('run-1', local_id)
        assert local_resolved is not None
        assert [
            (c['project_id'], c['task_id']) for c in local_resolved[1].cited_tasks
        ] == [('dark_factory', '42')]

        foreign_resolved = state._resolve_finding('run-1', foreign_id)
        assert foreign_resolved is not None
        assert [
            (c['project_id'], c['task_id']) for c in foreign_resolved[1].cited_tasks
        ] == [('other_project', '42')]

    @pytest.mark.asyncio
    async def test_skipped_fold_does_not_steal_the_anchor(self):
        """A skipped near-collision must not disable dedup for every
        subsequent same-project duplicate of that task — first registrant
        keeps the derived-sig anchor.

        cite_task's registration tail assigns the derived sig
        unconditionally when entity_fold_eligible, so a foreign finding
        whose fold was SKIPPED would still overwrite the anchor. A later
        genuine same-project duplicate would then compare against the
        FOREIGN citation, see a mismatch, and stop folding — turning a
        cross-project over-fold into a same-project under-fold. `third`
        folds onto `local` on unmodified code, so this is behaviour
        preservation, not a new rule.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        local = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='local 42 is stale', suggested_action='a',
            task_id='42', flag_type='X',
        )
        assert 'finding_id' in local, local
        local_id = local['finding_id']

        cite_local = await state.cite_task('run-1', local_id, 'dark_factory', '42')
        assert 'error' not in cite_local, cite_local

        foreign = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='foreign 42, worded differently', suggested_action='a',
            task_id=None, flag_type='X',
        )
        assert 'finding_id' in foreign, foreign
        foreign_id = foreign['finding_id']

        cite_foreign = await state.cite_task('run-1', foreign_id, 'other_project', '42')
        assert 'error' not in cite_foreign, cite_foreign

        # Comma-joined so its OWN add_finding signature ('42,7777', 'X') is
        # fresh; '42' is a member of its parts, so it is entity-fold-eligible
        # for the ('42', 'X') derived sig.
        third = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='third 42 mention, worded differently', suggested_action='a',
            task_id='42,7777', flag_type='X',
        )
        assert 'finding_id' in third, third
        third_id = third['finding_id']

        result = await state.cite_task('run-1', third_id, 'dark_factory', '42')

        # The ORIGINAL anchor, not the foreign finding that skipped its fold.
        assert result.get('error') == 'duplicate_finding'
        assert result.get('existing_finding_id') == local_id

        assert state._resolve_finding('run-1', third_id) is None

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = {item['finding_id'] for item in report['flagged_items']}
        assert ids == {local_id, foreign_id}

    @pytest.mark.asyncio
    async def test_second_foreign_citation_also_survives(self):
        """CHARACTERIZATION of the ACCEPTED foreign-side under-fold.

        Because a mismatch-skipped finding never takes derived-sig
        ownership, the anchor stays with the FIRST registrant — so a second
        finding citing the same FOREIGN task also finds the
        dark_factory-pinned anchor, also mismatches, and also survives. Two
        genuine other_project duplicates therefore both survive whenever
        fold 1 (the project-scoped one, whose key does carry a project) is
        ineligible for them — here because `foreign_b` has a non-null
        top-level task_id.

        This is intended, not an oversight: the alternative is handing the
        anchor to the foreign finding and under-folding the ORIGINAL
        project instead (see test_skipped_fold_does_not_steal_the_anchor).
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        local = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='local 42 is stale', suggested_action='a',
            task_id='42', flag_type='X',
        )
        assert 'finding_id' in local, local
        local_id = local['finding_id']

        cite_local = await state.cite_task('run-1', local_id, 'dark_factory', '42')
        assert 'error' not in cite_local, cite_local

        foreign_a = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='foreign 42, worded differently', suggested_action='a',
            task_id=None, flag_type='X',
        )
        assert 'finding_id' in foreign_a, foreign_a
        foreign_a_id = foreign_a['finding_id']

        cite_a = await state.cite_task('run-1', foreign_a_id, 'other_project', '42')
        assert 'error' not in cite_a, cite_a

        # Non-null top-level task_id ⇒ NOT fold-1 eligible, so the
        # project-scoped index cannot catch this one; comma-joined so its own
        # add_finding signature ('42,7777', 'X') is fresh while '42' stays a
        # member of its parts (entity-fold eligible).
        foreign_b = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='another foreign 42, worded differently again', suggested_action='a',
            task_id='42,7777', flag_type='X',
        )
        assert 'finding_id' in foreign_b, foreign_b
        foreign_b_id = foreign_b['finding_id']

        result = await state.cite_task('run-1', foreign_b_id, 'other_project', '42')
        assert 'error' not in result, result

        report = state.get_assembled_report('run-1', 'task_knowledge_sync')
        assert report is not None
        ids = {item['finding_id'] for item in report['flagged_items']}
        assert ids == {local_id, foreign_a_id, foreign_b_id}

        # Each kept its own citation; the anchor never moved.
        for fid, expected in (
            (local_id, ('dark_factory', '42')),
            (foreign_a_id, ('other_project', '42')),
            (foreign_b_id, ('other_project', '42')),
        ):
            resolved = state._resolve_finding('run-1', fid)
            assert resolved is not None
            assert [(c['project_id'], c['task_id']) for c in resolved[1].cited_tasks] == [expected]

        assert state._run_sig_index.get('run-1', {}).get(('42', 'X')) == local_id
        # The fold-1 lane the comment points at: the first foreign citation
        # DID anchor 'other_project:42' there (its key carries a project), so
        # a null-task_id foreign duplicate would still fold.
        assert state._run_cited_task_index.get('run-1', {}).get('other_project:42') == foreign_a_id

    # -- THE NEAR-COLLISION LOG ------------------------------------------

    @pytest.mark.asyncio
    async def test_near_collision_emits_operator_legible_warning(self, caplog):
        """A skipped near-collision is the ONLY observable evidence that a run
        contained numerically-colliding cross-project task ids — and the
        UNGUARDABLE add_finding→derived-sig half of that same run may have
        silently folded two projects' findings. The line must therefore be
        legible enough to identify both sides without re-running the stage.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.recon_report'):
            local = state.add_finding(
                run_id='run-1', severity='low', category='memory_stale',
                description='local 42 is stale', suggested_action='a',
                task_id='42', flag_type='X',
            )
            assert 'finding_id' in local, local
            local_id = local['finding_id']

            cite_local = await state.cite_task('run-1', local_id, 'dark_factory', '42')
            assert 'error' not in cite_local, cite_local

            foreign = state.add_finding(
                run_id='run-1', severity='low', category='memory_stale',
                description='foreign 42, worded differently', suggested_action='a',
                task_id=None, flag_type='X',
            )
            assert 'finding_id' in foreign, foreign
            foreign_id = foreign['finding_id']

            result = await state.cite_task('run-1', foreign_id, 'other_project', '42')

        # The log must not be buyable by changing behaviour.
        assert 'error' not in result, result

        warnings = _near_collision_warnings(caplog)
        assert len(warnings) == 1, warnings
        message = warnings[0]

        for expected in (
            foreign_id,          # the finding whose fold was skipped
            local_id,            # the surviving anchor
            'other_project',     # attempted citation's project
            'dark_factory',      # anchor citation's project
            "'42'",              # the colliding task id
            "'X'",               # the flag_type completing the derived sig
            "'run-1'",
            "'task_knowledge_sync'",  # the stage that owns the skipped finding
        ):
            assert expected in message, (expected, message)

    @pytest.mark.asyncio
    async def test_genuine_same_project_fold_emits_no_near_collision_warning(self, caplog):
        """Negative control: an ordinary same-project duplicate is not a
        near-collision. Its own task-4184 purge WARNING must still fire —
        that fold DOES destroy content — but no near-collision line.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.recon_report'):
            anchor = state.add_finding(
                run_id='run-1', severity='low', category='memory_stale',
                description='anchor 2405', suggested_action='a',
                task_id='2405', flag_type='X',
            )
            assert 'finding_id' in anchor, anchor
            anchor_id = anchor['finding_id']

            cite_anchor = await state.cite_task('run-1', anchor_id, 'dark_factory', '2405')
            assert 'error' not in cite_anchor, cite_anchor

            incoming = state.add_finding(
                run_id='run-1', severity='low', category='memory_stale',
                description='incoming 2405, worded differently', suggested_action='a',
                task_id=None, flag_type='X',
            )
            assert 'finding_id' in incoming, incoming
            incoming_id = incoming['finding_id']

            result = await state.cite_task('run-1', incoming_id, 'dark_factory', '2405')

        assert result.get('error') == 'duplicate_finding'
        assert result.get('existing_finding_id') == anchor_id

        assert _near_collision_warnings(caplog) == []
        assert _fold_purge_warnings(caplog) != []

    @pytest.mark.asyncio
    async def test_unpinned_anchor_fold_emits_no_near_collision_warning(self, caplog):
        """The ACCEPTED-AMBIGUITY half, made executable rather than only
        documented.

        The anchor's derived-sig ownership came from
        ``add_finding(task_id='99')``, which names NO project — so there is
        nothing to compare the incoming other_project:99 citation against.
        This fold still fires (it is the intended one per the operator
        ruling) and must NOT be reported as a near-collision: no guard at
        this layer can tell whether the two findings are about the same task.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.recon_report'):
            anchor = state.add_finding(
                run_id='run-1', severity='low', category='memory_stale',
                description='anchor 99, never cited', suggested_action='a',
                task_id='99', flag_type='X',
            )
            assert 'finding_id' in anchor, anchor
            anchor_id = anchor['finding_id']
            # Deliberately NO cite_task on the anchor: cited_tasks stays empty,
            # so it carries no project pin at all.

            incoming = state.add_finding(
                run_id='run-1', severity='low', category='memory_stale',
                description='incoming 99, worded differently', suggested_action='a',
                task_id=None, flag_type='X',
            )
            assert 'finding_id' in incoming, incoming
            incoming_id = incoming['finding_id']

            result = await state.cite_task('run-1', incoming_id, 'other_project', '99')

        assert result.get('error') == 'duplicate_finding'
        assert result.get('existing_finding_id') == anchor_id
        assert state._resolve_finding('run-1', incoming_id) is None

        assert _near_collision_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_project_fold_win_emits_no_near_collision_warning(self, caplog):
        """Negative control for the ONE call where BOTH folds hit.

        The entity-scoped fold is skipped for a project mismatch, but the
        project-scoped fold (checked after the detection, and priority when
        both would hit) purges this very finding and returns
        duplicate_finding. The skip never took effect, so a near-collision
        line claiming 'BOTH findings kept' would be immediately contradicted
        by the task-4184 purge record for the SAME finding_id — a false
        positive in the single channel an operator has for cross-project
        numeric collisions.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory')

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.recon_report'):
            # Pins the derived sig ('42', 'X') to a dark_factory citation.
            local = state.add_finding(
                run_id='run-1', severity='low', category='memory_stale',
                description='local 42 is stale', suggested_action='a',
                task_id='42', flag_type='X',
            )
            assert 'finding_id' in local, local
            local_id = local['finding_id']

            cite_local = await state.cite_task('run-1', local_id, 'dark_factory', '42')
            assert 'error' not in cite_local, cite_local

            # Different flag_type ⇒ no derived-sig contest; this one exists
            # purely to anchor the PROJECT-scoped key 'other_project:42'.
            project_anchor = state.add_finding(
                run_id='run-1', severity='low', category='memory_stale',
                description='foreign 42 under another flag', suggested_action='a',
                task_id=None, flag_type='Y',
            )
            assert 'finding_id' in project_anchor, project_anchor
            project_anchor_id = project_anchor['finding_id']

            cite_project_anchor = await state.cite_task(
                'run-1', project_anchor_id, 'other_project', '42'
            )
            assert 'error' not in cite_project_anchor, cite_project_anchor

            # Hits BOTH: project-scoped on 'other_project:42', entity-scoped
            # on ('42', 'X') — where the anchor is dark_factory-pinned.
            incoming = state.add_finding(
                run_id='run-1', severity='low', category='memory_stale',
                description='foreign 42, worded differently', suggested_action='a',
                task_id=None, flag_type='X',
            )
            assert 'finding_id' in incoming, incoming
            incoming_id = incoming['finding_id']

            result = await state.cite_task('run-1', incoming_id, 'other_project', '42')

        # Fold 1 wins, unchanged by the task-4185 guard.
        assert result.get('error') == 'duplicate_finding'
        assert result.get('existing_finding_id') == project_anchor_id
        assert state._resolve_finding('run-1', incoming_id) is None

        assert _near_collision_warnings(caplog) == []
        # The purge itself is still reported (task-4184).
        purges = [m for m in _fold_purge_warnings(caplog) if incoming_id in m]
        assert len(purges) == 1, purges
        assert 'project_scoped' in purges[0], purges[0]


# ---------------------------------------------------------------------------
# task-4184 step-1/step-3: TestCiteTaskFoldPurgeLogging — RED until step-2/4
# make both cite_task fold branches log the purged finding's content before
# _purge_finding discards it wholesale.
# ---------------------------------------------------------------------------

_FOLD_PURGE_MARKER = 'cite_task fold purged finding'

# task-4185. Deliberately shares no substring with _FOLD_PURGE_MARKER: the two
# filters below must partition this module's cite_task warnings, so each
# family's negative control can assert `== []` without the other's line
# satisfying it. Nothing is purged on this path — a near-collision KEEPS both
# findings — so it is a distinct event, not a variant of the 4184 record.
_NEAR_COLLISION_MARKER = 'cite_task entity-scoped fold SKIPPED'


def _fold_purge_warnings(caplog) -> list[str]:
    """The rendered fold-purge WARNING messages captured so far.

    Filtered on the stable message marker rather than on the payload, so the
    negative control can assert ``== []`` without accidentally matching an
    unrelated warning from the same module.
    """
    return [r.getMessage() for r in caplog.records if _FOLD_PURGE_MARKER in r.getMessage()]


def _near_collision_warnings(caplog) -> list[str]:
    """The rendered cross-project near-collision WARNINGs captured so far
    (task-4185). Marker-filtered, mirroring :func:`_fold_purge_warnings`.
    """
    return [r.getMessage() for r in caplog.records if _NEAR_COLLISION_MARKER in r.getMessage()]


class TestCiteTaskFoldPurgeLogging:
    """Both cite_task in-run folds (task-2425 project-scoped, task-2432
    entity-scoped) purge the losing finding WHOLESALE — its description,
    suggested_action and any citations already attached to it are dropped
    with no trace (see _purge_finding's docstring). A WARNING carrying that
    content is the sole recovery channel, so it must be emitted before the
    purge, on BOTH branches, and never on a non-folding cite_task.
    """

    def _fake_ti(self):
        """Fake task interceptor covering the external task ids these tests cite."""
        known_roots = {'/home/leo/src/dark-factory'}
        results = {}
        for pr in known_roots:
            for tid in ['2405', '2406', '9999']:
                results[(tid, pr)] = {'id': tid, 'title': f'T-{tid}'}
        return _FakeTaskInterceptor(results=results)

    def _make_state(self, fake_ti=None, memory_service=None):
        from fused_memory.server.recon_report import ReconReportState

        if fake_ti is None:
            fake_ti = self._fake_ti()
        t = [0.0]
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: t[0],
            task_interceptor=fake_ti,
            memory_service=memory_service,
        )
        state.known_projects = dict(_KNOWN_PROJECTS)
        return state, t

    @pytest.mark.asyncio
    async def test_project_scoped_fold_logs_purged_finding_content(self, caplog):
        """Project-scoped fold (task-2425): the second null-task_id citer of
        the same external task is purged. The WARNING must carry the purged
        finding's id, owning stage, description and suggested_action, plus
        the surviving anchor's id — everything needed to reconstruct what
        was discarded.
        """
        memory_service = _FakeMemoryService(
            entity_nodes=[{'uuid': 'e' * 32, 'name': 'Widget Service'}]
        )
        state, _ = self._make_state(memory_service=memory_service)
        # NOT memory_consolidator — fold 1 exempts that stage.
        state.start_report(run_id='run-1', stage='reconciler', project_id='dark_factory')

        first = state.add_finding(
            run_id='run-1',
            severity='moderate',
            category='cross_project',
            description='dark_factory:2405 still pending, unchanged',
            suggested_action='wait for upstream',
            actionable=True,
            task_id=None,
            flag_type=None,
        )
        assert 'finding_id' in first, f'add_finding failed: {first}'
        finding_id_1 = first['finding_id']

        second = state.add_finding(
            run_id='run-1',
            severity='moderate',
            category='cross_project_routing',
            description='blocked pending dark_factory task 2405 per routing check',
            suggested_action='reroute once unblocked',
            actionable=True,
            task_id=None,
            flag_type='cross_project_routing_stale',
        )
        assert 'finding_id' in second, f'add_finding failed: {second}'
        finding_id_2 = second['finding_id']
        assert finding_id_2 != finding_id_1

        cite_1 = await state.cite_task('run-1', finding_id_1, 'dark_factory', '2405')
        assert 'error' not in cite_1, cite_1

        # Pre-attach a citation to the finding that is about to be purged, so
        # the discarded-context path _purge_finding warns about is exercised.
        cited = await state.cite_entity('run-1', finding_id_2, 'Widget Service')
        assert 'error' not in cited, cited

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.recon_report'):
            cite_2 = await state.cite_task('run-1', finding_id_2, 'dark_factory', '2405')

        # Return contract is UNCHANGED — the log cannot be bought by altering
        # the fold's behaviour.
        assert cite_2.get('error') == 'duplicate_finding'
        assert cite_2.get('error_type') == 'ReconReportDuplicateFinding'
        assert cite_2.get('existing_finding_id') == finding_id_1
        assert state._resolve_finding('run-1', finding_id_2) is None

        matching = _fold_purge_warnings(caplog)
        assert len(matching) == 1, [r.getMessage() for r in caplog.records]
        message = matching[0]

        # The five MANDATED items.
        assert finding_id_2 in message, message  # purged finding_id
        assert 'reconciler' in message, message  # purged finding's owning stage
        assert 'blocked pending dark_factory task 2405 per routing check' in message, message
        assert 'reroute once unblocked' in message, message
        assert finding_id_1 in message, message  # surviving anchor

        # The pre-attached citation, which the purge destroys along with the
        # finding and which appears NOWHERE else afterwards — the returned
        # duplicate_finding error names only the survivor. Without this the
        # cited_entities/cited_edges/cited_memories/cited_runs half of the log
        # line could be deleted with the whole suite staying green, even
        # though those discarded citations are the primary thing the log
        # exists to make recoverable.
        assert 'e' * 32 in message, message  # cite_entity's recorded entity_uuid
        assert 'Widget Service' in message, message  # ...and its canonical name

        # Discriminates this fold from the entity-scoped one.
        assert 'project_scoped' in message, message

    @pytest.mark.asyncio
    async def test_entity_scoped_fold_logs_purged_finding_content(self, caplog):
        """Entity-scoped fold (task-2432): the null-task_id finding whose
        citation derives a signature already held by a top-level task_id
        finding is purged BY cite_task. Same five mandated fields, and the
        message must name a different fold than the project-scoped one.

        Uses the reverse-order shape (top-level task_id filed FIRST, null
        one cites second) because that is the only ordering whose fold runs
        inside cite_task: in the other ordering add_finding's own signature
        lookup collapses the duplicate before cite_task is ever reached, so
        it never exercises this purge site.
        """
        state, _ = self._make_state()
        state.start_report(
            run_id='run-1', stage='task_knowledge_sync', project_id='dark_factory'
        )

        occ2 = state.add_finding(
            run_id='run-1', severity='low', category='memory_stale',
            description='occ2 desc', suggested_action='a',
            task_id='2405', flag_type='cross_project',
        )
        assert 'finding_id' in occ2, occ2
        fid2 = occ2['finding_id']

        occ1 = state.add_finding(
            run_id='run-1', severity='high', category='memory_stale',
            description='occ1 desc, worded differently and lost on fold',
            suggested_action='reconcile the stale memory by hand',
            task_id=None, flag_type='cross_project',
        )
        assert 'finding_id' in occ1, occ1
        fid1 = occ1['finding_id']
        assert fid1 != fid2

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.recon_report'):
            cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '2405')

        # Return contract unchanged.
        assert cite1.get('error') == 'duplicate_finding'
        assert cite1.get('existing_finding_id') == fid2
        assert state._resolve_finding('run-1', fid1) is None

        matching = _fold_purge_warnings(caplog)
        assert len(matching) == 1, [r.getMessage() for r in caplog.records]
        message = matching[0]

        # The five MANDATED items.
        assert fid1 in message, message  # purged finding_id
        assert 'task_knowledge_sync' in message, message  # purged finding's stage
        assert 'occ1 desc, worded differently and lost on fold' in message, message
        assert 'reconcile the stale memory by hand' in message, message
        assert fid2 in message, message  # surviving anchor

        # Discriminates this fold from the project-scoped one.
        assert 'entity_scoped' in message, message
        assert 'project_scoped' not in message, message

    @pytest.mark.asyncio
    async def test_non_folding_cite_task_emits_no_purge_warning(self, caplog):
        """Negative control: an ordinary cite_task, and a second finding
        citing a DIFFERENT external task, must both succeed silently. A
        purge WARNING here would mean the log fires on the success path.
        """
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='reconciler', project_id='dark_factory')

        f1 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='desc A', suggested_action='a',
            task_id=None, flag_type=None,
        )
        f2 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='desc B', suggested_action='a',
            task_id=None, flag_type='other_flag',
        )
        fid1, fid2 = f1['finding_id'], f2['finding_id']

        with caplog.at_level(logging.WARNING, logger='fused_memory.server.recon_report'):
            cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
            cite2 = await state.cite_task('run-1', fid2, 'dark_factory', '2406')

        assert 'error' not in cite1, cite1
        assert 'error' not in cite2, cite2

        assert _fold_purge_warnings(caplog) == [], [r.getMessage() for r in caplog.records]

        report = state.get_assembled_report('run-1', 'reconciler')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [fid1, fid2]


# ---------------------------------------------------------------------------
# task-2425 step-3: TestCiteTaskFoldKeyClearedOnDelete — RED until step-4
# routes delete_finding through the shared _purge_finding helper
# ---------------------------------------------------------------------------


class TestCiteTaskFoldKeyClearedOnDelete:
    """delete_finding must clear the primary-cited-task fold key (task-2425)
    it registered in ``_run_cited_task_index``, mirroring how it already
    clears the sig/desc dedup indices. Otherwise a deleted finding leaves a
    dangling fold-key pointer, and a LATER finding that cites the same
    external task as its own primary citation wrongly folds into the
    stale, no-longer-resolvable finding_id instead of succeeding.
    """

    def _fake_ti(self):
        known_roots = {'/home/leo/src/dark-factory'}
        results = {}
        for pr in known_roots:
            for tid in ['2405']:
                results[(tid, pr)] = {'id': tid, 'title': f'T-{tid}'}
        return _FakeTaskInterceptor(results=results)

    def _make_state(self):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: t[0],
            task_interceptor=self._fake_ti(),
        )
        state.known_projects = dict(_KNOWN_PROJECTS)
        return state, t

    @pytest.mark.asyncio
    async def test_delete_then_new_primary_citer_of_same_task_succeeds(self):
        state, _ = self._make_state()
        state.start_report(run_id='run-1', stage='reconciler', project_id='dark_factory')

        f1 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='desc one', suggested_action='a',
            task_id=None, flag_type=None,
        )
        fid1 = f1['finding_id']

        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
        assert 'error' not in cite1, cite1

        delete_result = state.delete_finding('run-1', fid1)
        assert delete_result == {'status': 'deleted', 'finding_id': fid1}

        f2 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='desc two, worded differently', suggested_action='a',
            task_id=None, flag_type='some_other_flag',
        )
        fid2 = f2['finding_id']

        cite2 = await state.cite_task('run-1', fid2, 'dark_factory', '2405')
        assert 'error' not in cite2, cite2
        assert cite2 == {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'T-2405'}

        report = state.get_assembled_report('run-1', 'reconciler')
        assert report is not None
        ids = [item['finding_id'] for item in report['flagged_items']]
        assert ids == [fid2]
        assert report['flagged_items'][0]['cited_tasks'] == [
            {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'T-2405'}
        ]


# ---------------------------------------------------------------------------
# task-2425 step-5: TestCiteTaskFoldReleasedAtRunQuiescence — RED until
# step-6 makes tick() pop _run_cited_task_index at run quiescence
# ---------------------------------------------------------------------------


class TestCiteTaskFoldReleasedAtRunQuiescence:
    """tick()'s run-quiescence release (task-2088) must also release the
    primary-cited-task fold index (task-2425) once a run's last entry
    evicts — mirroring the existing _run_sig_index / _run_desc_index /
    _run_finding_index release. Otherwise a stale fold key survives past
    the run it belonged to and wrongly folds a citation on a later,
    unrelated run that happens to cite the same external task.
    """

    def _fake_ti(self):
        known_roots = {'/home/leo/src/dark-factory'}
        results = {}
        for pr in known_roots:
            for tid in ['2405']:
                results[(tid, pr)] = {'id': tid, 'title': f'T-{tid}'}
        return _FakeTaskInterceptor(results=results)

    def _make_state(self, ttl=300):
        from fused_memory.server.recon_report import ReconReportState

        t = [0.0]
        state = ReconReportState(
            ttl_seconds=ttl,
            clock=lambda: t[0],
            task_interceptor=self._fake_ti(),
        )
        state.known_projects = dict(_KNOWN_PROJECTS)
        return state, t

    @pytest.mark.asyncio
    async def test_full_quiescence_releases_cited_task_index(self):
        state, t = self._make_state(ttl=300)

        state.start_report(run_id='run-1', stage='reconciler', project_id='dark_factory')
        f1 = state.add_finding(
            run_id='run-1', severity='low', category='cross_project',
            description='desc one', suggested_action='a',
            task_id=None, flag_type=None,
        )
        fid1 = f1['finding_id']

        cite1 = await state.cite_task('run-1', fid1, 'dark_factory', '2405')
        assert 'error' not in cite1, cite1
        assert state._run_cited_task_index.get('run-1', {}).get('dark_factory:2405') == fid1

        state.complete('run-1', 'stage done')

        # Advance past TTL and evict the run's only entry in one tick().
        t[0] = 301.0
        evicted = state.tick()
        assert evicted == 1

        assert state.get_assembled_report('run-1', 'reconciler') is None
        assert 'run-1' not in state._run_sig_index
        assert 'run-1' not in state._run_desc_index
        assert 'run-1' not in state._run_finding_index
        assert 'run-1' not in state._run_cited_task_index

        # A FRESH run citing the SAME external task as its primary citation
        # must succeed plainly — the released index must not leak across runs.
        state.start_report(run_id='run-2', stage='reconciler', project_id='dark_factory')
        f2 = state.add_finding(
            run_id='run-2', severity='low', category='cross_project',
            description='desc two, unrelated run', suggested_action='a',
            task_id=None, flag_type=None,
        )
        fid2 = f2['finding_id']

        cite2 = await state.cite_task('run-2', fid2, 'dark_factory', '2405')
        assert 'error' not in cite2, cite2
        assert cite2 == {'project_id': 'dark_factory', 'task_id': '2405', 'title': 'T-2405'}


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
# task-2595 step-1: TestCiteRun — RED until step-2 adds cite_run to ReconReportState
# ---------------------------------------------------------------------------


class TestCiteRun:
    """Drive state.cite_run() directly (mirrors TestCiteEdge/TestCiteMemory).

    cite_run closes the gap named by task 2595: a run_id quoted inline in a
    finding's free-text description/suggested_action was previously validated
    by NONE of the cite_entity/cite_edge/cite_task/cite_memory tools, so an
    LLM re-typing a historical run_id from memory (instead of copying it
    verbatim off a fresh tool result) could silently drift by a hex group
    with nothing downstream catching it. cite_run confirms the cited run_id
    actually exists by checking memory_service.count_memories_by_metadata
    (the same mem0 existence signal the originating incident's remediation
    used to self-catch the bug by hand).
    """

    _VALID_CITED_RUN_ID = '30e7dce2-42a7-437d-abe5-a2316faad40a'

    def _fake(self, *, count_result=None):
        return _FakeMemoryService(count_result=count_result)

    def _state_and_finding(self, **fake_kwargs):
        fake = self._fake(**fake_kwargs)
        state, run_id, finding_id = _make_state_with_finding(memory_service=fake)
        return state, run_id, finding_id, fake

    @pytest.mark.asyncio
    async def test_happy_path_returns_run_id_and_match_count(self):
        """count_result=3 → {'run_id': <cited>, 'match_count': 3}; the fake is
        called with ({'run_id': <cited>}) scoped to the finding's project_id."""
        state, run_id, finding_id, fake = self._state_and_finding(count_result=3)

        result = await state.cite_run(run_id, finding_id, self._VALID_CITED_RUN_ID)

        assert result.get('run_id') == self._VALID_CITED_RUN_ID
        assert result.get('match_count') == 3
        assert fake.count_by_metadata_calls == [
            ('dark_factory', {'run_id': self._VALID_CITED_RUN_ID})
        ]

    @pytest.mark.asyncio
    async def test_happy_path_mutates_cited_runs(self):
        """The returned citation must also appear in the finding's cited_runs."""
        state, run_id, finding_id, _ = self._state_and_finding(count_result=3)

        await state.cite_run(run_id, finding_id, self._VALID_CITED_RUN_ID)

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        item = report['flagged_items'][0]
        assert item['cited_runs'] == [
            {'run_id': self._VALID_CITED_RUN_ID, 'match_count': 3}
        ]

    @pytest.mark.asyncio
    async def test_happy_path_projects_in_get_findings_for_run(self):
        """task-2595 step-5: the raw Stage-2 channel (get_findings_for_run)
        must also project cited_runs, not just get_assembled_report."""
        state, run_id, finding_id, _ = self._state_and_finding(count_result=3)

        await state.cite_run(run_id, finding_id, self._VALID_CITED_RUN_ID)

        results = state.get_findings_for_run(run_id)
        by_id = {r['finding_id']: r for r in results}
        assert by_id[finding_id]['cited_runs'] == [
            {'run_id': self._VALID_CITED_RUN_ID, 'match_count': 3}
        ]

    @pytest.mark.asyncio
    async def test_run_not_found_when_count_is_zero(self):
        """count_result=0 → {error:'run_not_found', error_type:'ReconReportRunNotFound'}."""
        state, run_id, finding_id, _ = self._state_and_finding(count_result=0)

        result = await state.cite_run(run_id, finding_id, self._VALID_CITED_RUN_ID)

        assert result.get('error') == 'run_not_found'
        assert result.get('error_type') == 'ReconReportRunNotFound'

    @pytest.mark.asyncio
    async def test_run_not_found_leaves_cited_runs_unchanged(self):
        """run_not_found must NOT mutate cited_runs."""
        state, run_id, finding_id, _ = self._state_and_finding(count_result=0)

        await state.cite_run(run_id, finding_id, self._VALID_CITED_RUN_ID)

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        assert report['flagged_items'][0]['cited_runs'] == []

    @pytest.mark.asyncio
    async def test_invalid_uuid_shape_no_service_call(self):
        """A malformed cited_run_id → {error:'invalid_uuid_shape'}, no service call."""
        state, run_id, finding_id, fake = self._state_and_finding(count_result=3)

        result = await state.cite_run(run_id, finding_id, 'not-a-uuid')

        assert result.get('error') == 'invalid_uuid_shape'
        assert result.get('error_type') == 'ReconReportInvalidUuid'
        assert fake.count_by_metadata_calls == []

    @pytest.mark.asyncio
    async def test_finding_unknown_for_bogus_finding_id(self):
        """Bogus finding_id → {error:'finding_unknown', error_type:'ReconReportFindingUnknown'}."""
        state, run_id, _finding_id, _ = self._state_and_finding(count_result=3)

        result = await state.cite_run(run_id, 'bogus-finding-id', self._VALID_CITED_RUN_ID)

        assert result.get('error') == 'finding_unknown'
        assert result.get('error_type') == 'ReconReportFindingUnknown'

    @pytest.mark.asyncio
    async def test_run_id_unknown_returned_for_bad_run(self):
        """Passing an unregistered run_id returns run_id_unknown."""
        state, run_id, finding_id, _ = self._state_and_finding(count_result=3)

        result = await state.cite_run('no-such-run', finding_id, self._VALID_CITED_RUN_ID)

        assert result.get('error') == 'run_id_unknown'
        assert result.get('error_type') == 'ReconReportRunUnknown'

    @pytest.mark.asyncio
    async def test_service_not_configured_when_memory_service_none(self):
        """cite_run returns service_not_configured after the UUID shape gate
        when memory_service is None."""
        state, run_id, finding_id = _make_state_with_finding()  # memory_service=None

        result = await state.cite_run(run_id, finding_id, self._VALID_CITED_RUN_ID)

        assert result.get('error') == 'service_not_configured'
        assert result.get('error_type') == 'ReconReportServiceUnavailable'

    @pytest.mark.asyncio
    async def test_accumulation_two_citations_produce_len_two(self):
        """Two cite_run calls with different cited_run_ids → two cited_runs
        entries (append-always, no dedup — matches cite_edge/cite_memory)."""
        other_run_id = '11111111-2222-3333-4444-555555555555'
        fake = _FakeMemoryService(count_result=1)
        state, run_id, finding_id = _make_state_with_finding(memory_service=fake)

        r1 = await state.cite_run(run_id, finding_id, self._VALID_CITED_RUN_ID)
        assert 'error' not in r1, r1
        r2 = await state.cite_run(run_id, finding_id, other_run_id)
        assert 'error' not in r2, r2

        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        cited_runs = report['flagged_items'][0]['cited_runs']
        assert len(cited_runs) == 2
        assert {c['run_id'] for c in cited_runs} == {self._VALID_CITED_RUN_ID, other_run_id}


# ---------------------------------------------------------------------------
# step-9: TestCiteToolsViaFastMCP + TestReconReportComponentsWiring
#         RED until step-10 registers the tools and extends _build_recon_report_components
# ---------------------------------------------------------------------------


class TestCiteToolsViaFastMCP:
    """Verify the four cite_* tools are registered and wired correctly in FastMCP."""

    def _make(self, task_interceptor=None, known_projects=None, count_result=None):
        from fused_memory.server.recon_report import ReconReportState, create_recon_report_server

        t = [0.0]
        fake_ms = _FakeMemoryService(
            entity_nodes=[{'uuid': 'aaaaaaaa-1111-1111-1111-111111111111', 'name': 'Foo'}],
            count_result=count_result,
        )
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: t[0],
            memory_service=fake_ms,
            task_interceptor=task_interceptor,
        )
        state.known_projects = known_projects if known_projects is not None else {}
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

    @pytest.mark.asyncio
    async def test_cite_task_duplicate_finding_propagates_through_call_tool(self):
        """task-2425 amend: the in-run cited-task fold's duplicate_finding
        return (ReconReportState.cite_task) must propagate unchanged through
        the registered FastMCP ``cite_task`` tool. Prior fold coverage
        (TestCiteTaskInRunCitedTaskDedup) drives ReconReportState.cite_task
        directly; this exercises the actual MCP invocation path
        (tm.call_tool) end-to-end, matching the tool docstring's promise of
        an ``existing_finding_id`` pointer on fold.
        """
        fake_ti = _FakeTaskInterceptor(
            results={('2405', '/home/leo/src/dark-factory'): {'id': '2405', 'title': 'T-2405'}}
        )
        state, mcp = self._make(
            task_interceptor=fake_ti,
            known_projects={'dark_factory': '/home/leo/src/dark-factory'},
        )
        tm = mcp._tool_manager

        await tm.call_tool('start_report', {
            'run_id': 'r1', 'stage': 'reconciler', 'project_id': 'dark_factory',
        })

        # Two differently-worded null-task_id findings (distinct flag_type
        # so neither add_finding in-run guard collapses them) that will both
        # cite the same external task — mirrors the core repro.
        f1 = await tm.call_tool('add_finding', {
            'run_id': 'r1',
            'severity': 'moderate',
            'category': 'cross_project',
            'description': 'dark_factory:2405 still pending',
            'suggested_action': 'wait for upstream',
        })
        fid1 = f1['finding_id']

        f2 = await tm.call_tool('add_finding', {
            'run_id': 'r1',
            'severity': 'moderate',
            'category': 'cross_project_routing',
            'description': 'blocked pending dark_factory task 2405 per routing check',
            'suggested_action': 'reroute once unblocked',
            'flag_type': 'cross_project_routing_stale',
        })
        fid2 = f2['finding_id']
        assert fid2 != fid1

        cite1 = await tm.call_tool('cite_task', {
            'run_id': 'r1', 'finding_id': fid1,
            'project_id': 'dark_factory', 'task_id': '2405',
        })
        assert 'error' not in cite1, cite1

        cite2 = await tm.call_tool('cite_task', {
            'run_id': 'r1', 'finding_id': fid2,
            'project_id': 'dark_factory', 'task_id': '2405',
        })
        assert cite2.get('error') == 'duplicate_finding'
        assert cite2.get('error_type') == 'ReconReportDuplicateFinding'
        assert cite2.get('existing_finding_id') == fid1

        # The fold purged fid2 server-side too, not just at the response level.
        report = state.get_assembled_report('r1', 'reconciler')
        assert report is not None
        assert [i['finding_id'] for i in report['flagged_items']] == [fid1]

    # -- task-2595 step-3: cite_run via FastMCP — RED until step-4 registers
    #    the cite_run tool wrapper -----------------------------------------

    _CITE_RUN_VALID_UUID = '30e7dce2-42a7-437d-abe5-a2316faad40a'

    async def _start_and_add_finding(self, mcp):
        tm = mcp._tool_manager
        await tm.call_tool('start_report', {
            'run_id': 'r1', 'stage': 'test_stage', 'project_id': 'dark_factory',
        })
        r = await tm.call_tool('add_finding', {
            'run_id': 'r1',
            'severity': 'low',
            'category': 'cat',
            'description': 'd',
            'suggested_action': 'a',
            'task_id': '1',
            'flag_type': 'f',
        })
        return tm, r['finding_id']

    def test_cite_run_registered(self):
        """cite_run must be registered in the FastMCP tool manager."""
        _, mcp = self._make()
        tools = set(mcp._tool_manager._tools.keys())
        assert 'cite_run' in tools

    @pytest.mark.asyncio
    async def test_end_to_end_cite_run_happy_path_via_call_tool(self):
        """Full lifecycle through the MCP boundary: happy path returns
        {'run_id', 'match_count'}."""
        state, mcp = self._make(count_result=3)
        tm, finding_id = await self._start_and_add_finding(mcp)

        cite_r = await tm.call_tool('cite_run', {
            'run_id': 'r1',
            'finding_id': finding_id,
            'cited_run_id': self._CITE_RUN_VALID_UUID,
        })
        assert cite_r.get('run_id') == self._CITE_RUN_VALID_UUID
        assert cite_r.get('match_count') == 3

        report = state.get_assembled_report('r1', 'test_stage')
        assert report is not None
        assert len(report['flagged_items'][0]['cited_runs']) == 1

    @pytest.mark.asyncio
    async def test_end_to_end_cite_run_not_found_via_call_tool(self):
        """count_result=0 surfaces the run_not_found error dict through the MCP boundary."""
        _, mcp = self._make(count_result=0)
        tm, finding_id = await self._start_and_add_finding(mcp)

        cite_r = await tm.call_tool('cite_run', {
            'run_id': 'r1',
            'finding_id': finding_id,
            'cited_run_id': self._CITE_RUN_VALID_UUID,
        })
        assert cite_r.get('error') == 'run_not_found'
        assert cite_r.get('error_type') == 'ReconReportRunNotFound'

    @pytest.mark.asyncio
    async def test_end_to_end_cite_run_invalid_uuid_shape_via_call_tool(self):
        """A malformed cited_run_id surfaces invalid_uuid_shape through the MCP boundary."""
        _, mcp = self._make(count_result=3)
        tm, finding_id = await self._start_and_add_finding(mcp)

        cite_r = await tm.call_tool('cite_run', {
            'run_id': 'r1',
            'finding_id': finding_id,
            'cited_run_id': 'not-a-uuid',
        })
        assert cite_r.get('error') == 'invalid_uuid_shape'
        assert cite_r.get('error_type') == 'ReconReportInvalidUuid'


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


# ---------------------------------------------------------------------------
# task 2979: apply_citation_verification — the verification write-back
# ---------------------------------------------------------------------------


class _WBFakeMemoryService:
    """Minimal memory service for the write-back tests — every id 'exists' at
    cite time, so a phantom has to be created the way production creates one:
    cited successfully, then verified later and found gone."""

    async def get_memory(self, memory_id: str, project_id: str, store: str) -> dict:
        return {'category': 'observations_and_summaries', 'agent_id': 'x', 'created_at': 'now'}


class TestApplyCitationVerification:
    """``verify_cited_memories`` must reach the AUTHORITATIVE recon_report
    record, not just the throwaway projection (task 2979, Gap B).

    ``get_assembled_report`` builds a FRESH dict per finding with
    ``'cited_memories': list(f.cited_memories)`` — a NEW list object. So the
    verification pass in ``BaseStage.run()`` mutates that projection while the
    authoritative ``_Finding``, and the durable SQLite row written inside the
    CLI subprocess at add_finding/cite_memory/complete time (strictly BEFORE
    verification runs), keep the phantom forever. Two stores then permanently
    disagree about the same finding's citations, and which one a consumer reads
    decides whether it sees the phantom. ``apply_citation_verification`` closes
    that divergence.
    """

    _GOOD = 'd4e5f6a7-b8c9-0123-d456-e78f9a0b1c2d'
    _PHANTOM = 'b47ded9b-1111-4222-8333-444444444444'

    def _state_with_two_citations(self, tmp_path=None):
        """A completed report whose single finding cites _GOOD and _PHANTOM."""
        from fused_memory.server.recon_report import ReconReportState

        store = None
        if tmp_path is not None:
            from fused_memory.server.recon_report_store import ReconReportStore

            store = ReconReportStore(tmp_path / 'recon_report_state.db')
            store.open()

        t = [0.0]
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: t[0],
            memory_service=_WBFakeMemoryService(),
            store=store,
        )
        run_id = 'run-wb-1'
        state.start_report(run_id=run_id, stage='memory_consolidator', project_id='dark_factory')
        finding_id = state.add_finding(
            run_id=run_id,
            severity='moderate',
            category='memory_stale',
            description='a finding citing a phantom',
            suggested_action='a',
            actionable=True,
            task_id='42',
            flag_type='orphaned_knowledge',
        )['finding_id']
        return state, store, run_id, finding_id

    async def _cite_both(self, state, run_id, finding_id):
        await state.cite_memory(run_id, finding_id, self._GOOD, 'mem0')
        await state.cite_memory(run_id, finding_id, self._PHANTOM, 'mem0')

    def _verification_result(self, finding_id):
        """What BaseStage.run()'s verification pass concluded: keep _GOOD, drop
        _PHANTOM (get_memory_by_id returned None for it)."""
        return [
            {
                'finding_id': finding_id,
                'cited_memories': [{'memory_id': self._GOOD, 'store': 'mem0'}],
                'citation_failures': [
                    {'memory_id': self._PHANTOM, 'store': 'mem0', 'reason': 'memory_not_found'},
                ],
            },
        ]

    @pytest.mark.asyncio
    async def test_phantom_stripped_from_authoritative_findings(self):
        """After apply_citation_verification, neither get_findings_for_run nor
        get_assembled_report lists the phantom id."""
        state, _store, run_id, finding_id = self._state_with_two_citations()
        await self._cite_both(state, run_id, finding_id)
        state.complete(run_id, summary='s')

        # Pre-condition: both ids are on the authoritative record.
        pre = state.get_findings_for_run(run_id)[0]
        assert [c['memory_id'] for c in pre['cited_memories']] == [self._GOOD, self._PHANTOM]

        state.apply_citation_verification(run_id, self._verification_result(finding_id))

        post = state.get_findings_for_run(run_id)[0]
        assert [c['memory_id'] for c in post['cited_memories']] == [self._GOOD]

        assembled = state.get_assembled_report(run_id, 'memory_consolidator')
        assert assembled is not None
        assembled_finding = assembled['flagged_items'][0]
        assert [c['memory_id'] for c in assembled_finding['cited_memories']] == [self._GOOD]

    @pytest.mark.asyncio
    async def test_citation_failure_marker_is_recorded_and_projected(self):
        """The dropped phantom is recorded on the finding and visible to BOTH
        readers — the marker is what makes the phantom claim surfaced rather
        than silently vanished."""
        state, _store, run_id, finding_id = self._state_with_two_citations()
        await self._cite_both(state, run_id, finding_id)
        state.complete(run_id, summary='s')

        state.apply_citation_verification(run_id, self._verification_result(finding_id))

        expected = [{'memory_id': self._PHANTOM, 'store': 'mem0', 'reason': 'memory_not_found'}]
        assert state.get_findings_for_run(run_id)[0]['citation_failures'] == expected
        assembled = state.get_assembled_report(run_id, 'memory_consolidator')
        assert assembled is not None
        assert assembled['flagged_items'][0]['citation_failures'] == expected

    @pytest.mark.asyncio
    async def test_correction_is_durable_across_a_fresh_hydrate(self, tmp_path):
        """The correction reaches the SQLite row, not just memory.

        This is the assertion that actually closes the divergence: a brand-new
        ReconReportState hydrating from the same db file must see the corrected
        citations. Without the _persist_run write-back, the row still carries
        the pre-verification phantom.
        """
        from fused_memory.server.recon_report import ReconReportState
        from fused_memory.server.recon_report_store import ReconReportStore

        state, store, run_id, finding_id = self._state_with_two_citations(tmp_path)
        assert store is not None  # tmp_path was passed, so a store was built.
        await self._cite_both(state, run_id, finding_id)
        state.complete(run_id, summary='s')

        state.apply_citation_verification(run_id, self._verification_result(finding_id))

        # Re-open the same db in a fresh state object and hydrate.
        store.close()
        reopened = ReconReportStore(tmp_path / 'recon_report_state.db')
        reopened.open()
        rehydrated = ReconReportState(
            ttl_seconds=300,
            clock=lambda: 0.0,
            memory_service=_WBFakeMemoryService(),
            store=reopened,
        )
        rehydrated.hydrate_from_store()

        durable = rehydrated.get_findings_for_run(run_id)
        assert durable, 'the run must survive hydrate_from_store'
        assert [c['memory_id'] for c in durable[0]['cited_memories']] == [self._GOOD], (
            'the persisted row still carries the phantom — the verification '
            'result never reached the durable store'
        )
        assert durable[0]['citation_failures'] == [
            {'memory_id': self._PHANTOM, 'store': 'mem0', 'reason': 'memory_not_found'},
        ]

    @pytest.mark.asyncio
    async def test_accepted_after_complete_unlike_delete_finding(self):
        """The _ERR_ALREADY_COMPLETED guard must NOT apply here.

        This write-back is a harness-side integrity correction that by
        construction runs after the CLI subprocess called complete(), so
        applying delete_finding's guard would reject every legitimate call. The
        asymmetry is deliberate: unlike a retraction it only edits citation
        lists, never adds or removes a finding, so complete()'s cached
        flagged_count is unaffected.
        """
        state, _store, run_id, finding_id = self._state_with_two_citations()
        await self._cite_both(state, run_id, finding_id)
        completed = state.complete(run_id, summary='s')
        flagged_count_before = completed['flagged_count']

        result = state.apply_citation_verification(run_id, self._verification_result(finding_id))

        assert 'error' not in result, (
            f'post-complete write-back must be accepted, got {result!r}'
        )
        # Contrast: delete_finding on the same completed entry IS rejected.
        assert state.delete_finding(run_id, finding_id).get('error') == 'report_already_completed'
        # flagged_count is untouched — only citation lists changed.
        assert len(state.get_findings_for_run(run_id)) == flagged_count_before

    @pytest.mark.asyncio
    async def test_repeat_pass_does_not_duplicate_citation_failure_markers(self):
        """A second pass over the same (run_id, stage) is IDEMPOTENT.

        The caller sends the FULL citation_failures list off the assembled
        projection, not the delta it appended — and get_assembled_report
        projects the already-persisted markers straight back out. A plain
        extend therefore re-appended every prior marker on every repeat. A
        repeat is architecturally reachable: start_report is deliberately
        idempotent and RETAINS prior findings, and the resume path can
        re-invoke stage.run() under the same run_id.
        """
        state, _store, run_id, finding_id = self._state_with_two_citations()
        await self._cite_both(state, run_id, finding_id)
        state.complete(run_id, summary='s')

        state.apply_citation_verification(run_id, self._verification_result(finding_id))
        # Second pass sends exactly what get_assembled_report now projects: the
        # surviving citation AND the marker the first pass persisted.
        state.apply_citation_verification(run_id, self._verification_result(finding_id))

        durable = state.get_findings_for_run(run_id)[0]
        assert durable['citation_failures'] == [
            {'memory_id': self._PHANTOM, 'store': 'mem0', 'reason': 'memory_not_found'},
        ], 'a repeat pass must not duplicate the marker it already persisted'
        assert [c['memory_id'] for c in durable['cited_memories']] == [self._GOOD]

    @pytest.mark.asyncio
    async def test_repeat_verification_error_dedupes_across_error_types(self):
        """The dedupe key is (memory_id, store, reason) — error_type excluded.

        Two verification errors for the same citation record the same fact
        ("this id could not be resolved") whichever exception class surfaced
        it. Keying on error_type would let a flapping backend grow the list
        without bound across repeat passes.
        """
        state, _store, run_id, finding_id = self._state_with_two_citations()
        await self._cite_both(state, run_id, finding_id)
        state.complete(run_id, summary='s')

        def _err_result(error_type):
            return [
                {
                    'finding_id': finding_id,
                    'cited_memories': [
                        {'memory_id': self._GOOD, 'store': 'mem0'},
                        {'memory_id': self._PHANTOM, 'store': 'mem0'},
                    ],
                    'citation_failures': [
                        {
                            'memory_id': self._PHANTOM,
                            'store': 'mem0',
                            'reason': 'verification_error',
                            'error_type': error_type,
                        },
                    ],
                },
            ]

        state.apply_citation_verification(run_id, _err_result('TimeoutError'))
        state.apply_citation_verification(run_id, _err_result('ConnectionError'))

        durable = state.get_findings_for_run(run_id)[0]
        assert len(durable['citation_failures']) == 1, (
            f'error_type must not split the dedupe key, got '
            f'{durable["citation_failures"]!r}'
        )
        assert durable['citation_failures'][0]['error_type'] == 'TimeoutError'

    @pytest.mark.asyncio
    async def test_no_op_result_does_not_repersist_the_run(self):
        """A result that changes nothing must not drive a _persist_run.

        _persist_run re-serialises and upserts EVERY entry of the run, and the
        overwhelmingly common case is a clean verification pass where every
        citation resolved. Recording "nothing moved" at the cost of a full-run
        rewrite, once per stage per cycle, is pure waste.
        """
        state, _store, run_id, finding_id = self._state_with_two_citations()
        await self._cite_both(state, run_id, finding_id)
        state.complete(run_id, summary='s')

        # Echo back exactly what the projection carries — which is what
        # BaseStage.run() sends after a pass that dropped nothing. Built from
        # the projection rather than hand-written so the no-op stays a no-op if
        # cite_memory's entry shape ever gains a field.
        projected = state.get_findings_for_run(run_id)[0]
        no_op = [
            {
                'finding_id': finding_id,
                'cited_memories': projected['cited_memories'],
                'citation_failures': projected['citation_failures'],
            },
        ]
        with patch.object(state, '_persist_run') as persist:
            result = state.apply_citation_verification(run_id, no_op)

        assert result['findings_updated'] == 1, 'the finding still RESOLVED'
        assert result['findings_changed'] == 0, 'but nothing about it moved'
        assert persist.call_count == 0, 'a no-op write-back must not re-persist the run'

    @pytest.mark.asyncio
    async def test_unknown_finding_id_is_skipped_not_raised(self):
        """A post-hoc hygiene pass must never fail a stage: an unresolvable
        finding_id is logged and skipped, and the resolvable ones still apply."""
        state, _store, run_id, finding_id = self._state_with_two_citations()
        await self._cite_both(state, run_id, finding_id)
        state.complete(run_id, summary='s')

        results = [
            {'finding_id': 'no-such-finding', 'cited_memories': [], 'citation_failures': []},
            *self._verification_result(finding_id),
        ]
        state.apply_citation_verification(run_id, results)  # must not raise

        assert [
            c['memory_id'] for c in state.get_findings_for_run(run_id)[0]['cited_memories']
        ] == [self._GOOD]

    def test_unknown_run_id_is_skipped_not_raised(self):
        """Same posture for a run_id that no longer exists (TTL-reaped)."""
        from fused_memory.server.recon_report import ReconReportState

        state = ReconReportState(ttl_seconds=300, clock=lambda: 0.0)
        state.apply_citation_verification('no-such-run', [])  # must not raise
