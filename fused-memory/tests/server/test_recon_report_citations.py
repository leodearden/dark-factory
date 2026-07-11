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
