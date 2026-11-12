"""Unit tests for GraphitiBackend.detect_stale_dry_run, the force=False dry_run stale probe.

The service-level orchestration of the probe stays in
test_rebuild_entity_summaries.py (TestServiceRebuildOrchestrationDryRun,
TestRebuildReportsReadCompleteness). See task 526 for the probe's design.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import COMPLETE_READ, incomplete_paged_read
from _read_policy_logs import node_read_warnings

from fused_memory.backends.graphiti_client import (
    INCOMPLETE_CENSUS_UNAVAILABLE,
    INCOMPLETE_SHORT_READ,
    INCOMPLETE_STRUCTURAL_KINDS,
    IncompleteEnumerationError,
    ReadCompleteness,
    StaleProbeResult,
)


class TestDetectStaleSummariesDryRun:
    """Unit tests for GraphitiBackend.detect_stale_dry_run.

    This method is the memory-cheaper alternative to
    ``detect_stale_with_edges`` used by the force=False, dry_run=True
    code path in ``rebuild_entity_summaries``.

    Unlike the bulk variant, this probe issues up-to-N targeted
    ``get_valid_edges_for_node`` calls (one per non-empty-summary entity)
    and never materialises the O(E) all-edges dict.  The stale dict schema
    it produces is identical to ``detect_stale_with_edges`` so
    that downstream consumers (e.g. the result['details'] field) see no
    surface-level change.

    The node corpus enters at the ``enumerate_entity_nodes`` seam
    (``make_edge_backend``), so the real checked read applies the
    incompleteness policy and its verdict lands on
    ``StaleProbeResult.entities_completeness``.
    """

    @pytest.mark.asyncio
    async def test_returns_stale_list_and_total_count(
        self, mock_config, make_backend, make_edge_backend
    ):
        """Returns a StaleProbeResult whose total_count is the total entity count.

        Three entities:
        - Alice: stale (summary != canonical)
        - Bob: clean (summary matches canonical)
        - Charlie: empty summary (cheap-skipped)

        Expected: result.stale has 1 entry (Alice), result.total_count == 3.
        """
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-alice', 'name': 'Alice', 'summary': 'old fact'},
            {'uuid': 'uuid-bob', 'name': 'Bob', 'summary': 'current fact'},
            {'uuid': 'uuid-charlie', 'name': 'Charlie', 'summary': ''},
        ], edges={})

        def edges_for_node(node_uuid, *, group_id):
            if node_uuid == 'uuid-alice':
                return [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}]
            if node_uuid == 'uuid-bob':
                return [{'uuid': 'e2', 'fact': 'current fact', 'name': 'edge2'}]
            return []

        backend.get_valid_edges_for_node = AsyncMock(side_effect=edges_for_node)

        result = await backend.detect_stale_dry_run(group_id='test')

        assert isinstance(result, StaleProbeResult)
        assert isinstance(result.stale, list)
        assert isinstance(result.total_count, int)
        assert result.total_count == 3
        assert len(result.stale) == 1
        assert result.stale[0]['uuid'] == 'uuid-alice'
        # Charlie's empty summary is cheap-skipped: only Alice and Bob trigger edge queries.
        assert backend.get_valid_edges_for_node.await_count == 2

    @pytest.mark.asyncio
    async def test_flags_stale_entity_with_diagnostic_counts(
        self, mock_config, make_backend, make_edge_backend
    ):
        """Stale entity dict has the same schema as detect_stale_with_edges.

        Keys: uuid, name, summary, duplicate_count, stale_line_count,
              valid_fact_count, summary_line_count.
        """
        # Summary has 'factA' duplicated and 'old' which is not a valid edge fact.
        # valid edges: factA only.
        # duplicate_count=1 ('factA' appears twice → 2-1=1 extra)
        # stale_line_count=1 ('old' is not in valid_fact_set, 'factA' appears twice
        #   but the duplicate is also 'factA' which IS in valid_fact_set, so only 'old' is stale)
        # valid_fact_count=1 ('factA')
        # summary_line_count=3 ('factA', 'factA', 'old')
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'factA\nfactA\nold'},
        ], edges={})
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'factA', 'name': 'edge1'},
        ])

        result = await backend.detect_stale_dry_run(group_id='test')

        assert result.total_count == 1
        assert len(result.stale) == 1
        entity = result.stale[0]

        # Schema keys must exactly match detect_stale_with_edges output
        expected_keys = {
            'uuid', 'name', 'summary',
            'duplicate_count', 'stale_line_count', 'valid_fact_count', 'summary_line_count',
        }
        assert set(entity.keys()) == expected_keys

        assert entity['uuid'] == 'uuid-1'
        assert entity['name'] == 'Alice'
        assert entity['summary'] == 'factA\nfactA\nold'
        assert entity['duplicate_count'] == 1   # 'factA' appears twice → 1 extra
        assert entity['stale_line_count'] == 1  # 'old' not in valid_fact_set
        assert entity['valid_fact_count'] == 1  # deduped: only 'factA'
        assert entity['summary_line_count'] == 3

    @pytest.mark.asyncio
    async def test_skips_clean_entity(self, mock_config, make_backend, make_edge_backend):
        """Entity whose summary exactly matches canonical facts is not flagged as stale.

        Summary 'current fact' matches the single valid edge fact → clean.
        result.stale should be empty, but result.total_count is still 1.
        """
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'current fact'},
        ], edges={})
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'},
        ])

        result = await backend.detect_stale_dry_run(group_id='test')

        assert result.stale == []
        assert result.total_count == 1

    @pytest.mark.asyncio
    async def test_does_not_call_get_all_valid_edges(
        self, mock_config, make_backend, make_edge_backend
    ):
        """The dry_run probe never awaits a whole-graph edge read.

        This is the defining property of the cheap probe: it fetches edges
        per-entity via get_valid_edges_for_node rather than materialising the
        O(E) bulk all-edges dict, through neither get_all_valid_edges nor the
        enumerate_all_valid_edges seam beneath it.  The assertion style follows
        the project's AsyncMock convention (assert_not_awaited, not
        assert_not_called) as enforced in test_graphiti_rebuild_pipeline.py.
        """
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'stale fact'},
        ], edges={})
        backend.get_all_valid_edges = AsyncMock()
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'},
        ])

        await backend.detect_stale_dry_run(group_id='test')

        backend.get_all_valid_edges.assert_not_awaited()
        backend.enumerate_all_valid_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_empty_summary_entity_without_edge_query(
        self, mock_config, make_backend, make_edge_backend
    ):
        """Empty-summary entities are cheap-skipped: no get_valid_edges_for_node call.

        Empty summaries are 'not stale by definition' (matches the semantics
        of detect_stale_with_edges at graphiti_client.py:905-907).
        The probe additionally saves the edge query for those entities.
        """
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Charlie', 'summary': ''},
        ], edges={})
        backend.get_valid_edges_for_node = AsyncMock()

        result = await backend.detect_stale_dry_run(group_id='test')

        # Empty summary is not stale
        assert result.stale == []
        assert result.total_count == 1
        # get_valid_edges_for_node must NOT have been called for the empty-summary entity
        backend.get_valid_edges_for_node.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_accepts_max_concurrency_kwarg(
        self, mock_config, make_backend, make_edge_backend
    ):
        """detect_stale_dry_run accepts an optional max_concurrency keyword parameter.

        Calling with max_concurrency=5 must not raise TypeError.
        Calling with the default (no kwarg) must also still work.
        Both calls return a well-formed StaleProbeResult.
        """
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'fact'},
        ], edges={})
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'fact', 'name': 'edge1'},
        ])

        # Should not raise TypeError with explicit max_concurrency
        result = await backend.detect_stale_dry_run(group_id='test', max_concurrency=5)
        assert result.total_count == 1

        # Default (no kwarg) must still work
        default_result = await backend.detect_stale_dry_run(group_id='test')
        assert default_result.total_count == 1

    @pytest.mark.asyncio
    async def test_raises_when_max_concurrency_invalid(
        self, mock_config, make_backend, make_edge_backend
    ):
        """detect_stale_dry_run raises ValueError for max_concurrency < 1.

        Semaphore(0) would deadlock; Semaphore(-1) raises ValueError from stdlib.
        A clear ValueError with a descriptive message is better than either.
        """
        backend = make_edge_backend(make_backend(mock_config), nodes=[], edges={})

        with pytest.raises(ValueError, match='max_concurrency must be >= 1'):
            await backend.detect_stale_dry_run(group_id='test', max_concurrency=0)

        with pytest.raises(ValueError, match='max_concurrency must be >= 1'):
            await backend.detect_stale_dry_run(group_id='test', max_concurrency=-5)

    @pytest.mark.asyncio
    async def test_edge_fetches_run_concurrently(
        self, mock_config, make_backend, make_edge_backend
    ):
        """Edge fetches overlap rather than running sequentially (peak_concurrent == 5).

        Uses asyncio.Barrier(5) so all 5 coroutines must be in-flight simultaneously
        before any of them can complete — this provides a deterministic proof of
        concurrency (not just a scheduler hint via sleep(0)).
        """
        n = 5
        entities = [
            {'uuid': f'uuid-{i}', 'name': f'Entity{i}', 'summary': 'fact'}
            for i in range(n)
        ]
        backend = make_edge_backend(make_backend(mock_config), nodes=entities, edges={})

        in_flight = 0
        peak_concurrent = 0
        barrier = asyncio.Barrier(n)

        async def tracking_edges(node_uuid, *, group_id):
            nonlocal in_flight, peak_concurrent
            in_flight += 1
            peak_concurrent = max(peak_concurrent, in_flight)
            await barrier.wait()  # all n must arrive before any can continue
            in_flight -= 1
            return [{'uuid': 'e1', 'fact': 'fact', 'name': 'edge1'}]

        backend.get_valid_edges_for_node = AsyncMock(side_effect=tracking_edges)

        await backend.detect_stale_dry_run(group_id='test', max_concurrency=n)

        assert peak_concurrent == n, (
            f"Expected all {n} fetches in-flight simultaneously, got peak={peak_concurrent}"
        )

    @pytest.mark.asyncio
    async def test_concurrency_bounded_by_max_concurrency(
        self, mock_config, make_backend, make_edge_backend
    ):
        """Peak concurrent calls never exceeds max_concurrency.

        With 6 entities and max_concurrency=2, the semaphore must hold all but
        2 calls back — peak_concurrent must be <= 2.
        """
        entities = [
            {'uuid': f'uuid-{i}', 'name': f'Entity{i}', 'summary': 'fact'}
            for i in range(6)
        ]
        backend = make_edge_backend(make_backend(mock_config), nodes=entities, edges={})

        in_flight = 0
        peak_concurrent = 0

        async def tracking_edges(node_uuid, *, group_id):
            nonlocal in_flight, peak_concurrent
            in_flight += 1
            peak_concurrent = max(peak_concurrent, in_flight)
            await asyncio.sleep(0)  # yield to event loop so others can contend
            in_flight -= 1
            return [{'uuid': 'e1', 'fact': 'fact', 'name': 'edge1'}]

        backend.get_valid_edges_for_node = AsyncMock(side_effect=tracking_edges)

        await backend.detect_stale_dry_run(group_id='test', max_concurrency=2)

        assert peak_concurrent <= 2, (
            f"Expected peak concurrent calls <= 2 (bounded), got {peak_concurrent}"
        )

    @pytest.mark.asyncio
    async def test_exception_propagates_from_gather(
        self, mock_config, make_backend, make_edge_backend
    ):
        """RuntimeError raised by one entity's edge fetch propagates out of detect_stale_dry_run.

        Other entities succeed; the first Exception found in the gather results
        must be re-raised (not swallowed or returned).
        """
        entities = [
            {'uuid': 'uuid-ok', 'name': 'OK', 'summary': 'fact'},
            {'uuid': 'uuid-bad', 'name': 'Bad', 'summary': 'fact'},
            {'uuid': 'uuid-ok2', 'name': 'OK2', 'summary': 'fact'},
        ]
        backend = make_edge_backend(make_backend(mock_config), nodes=entities, edges={})

        async def edges_with_error(node_uuid, *, group_id):
            if node_uuid == 'uuid-bad':
                raise RuntimeError('db timeout')
            return [{'uuid': 'e1', 'fact': 'fact', 'name': 'edge1'}]

        backend.get_valid_edges_for_node = AsyncMock(side_effect=edges_with_error)

        with pytest.raises(RuntimeError, match='db timeout'):
            await backend.detect_stale_dry_run(group_id='test')

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_from_gather(
        self, mock_config, make_backend, make_edge_backend
    ):
        """asyncio.CancelledError raised during an edge fetch propagates out immediately.

        CancelledError is a BaseException (not Exception), so Pass 1
        (propagate_cancellations) must re-raise it before Pass 2 even runs.
        """
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'fact'},
        ], edges={})

        async def cancel_on_fetch(node_uuid, *, group_id):
            raise asyncio.CancelledError()

        backend.get_valid_edges_for_node = AsyncMock(side_effect=cancel_on_fetch)

        with pytest.raises(asyncio.CancelledError):
            await backend.detect_stale_dry_run(group_id='test')

    @pytest.mark.asyncio
    async def test_exception_logs_warning_for_each_failure_before_raising_first(
        self, mock_config, make_backend, make_edge_backend, caplog
    ):
        """Each captured Exception is logged at WARNING before the first is raised.

        Task β (fm-cancellederror-convention-prd.md): detect_stale_dry_run routes
        through gather_or_raise (fused_memory.utils.async_utils), which adds
        observability the hand-rolled Pass 2 never had — every captured Exception
        is logged at WARNING (not just the one that gets raised) before the
        positionally-first is re-raised. Raise-first semantics are unchanged (see
        test_exception_propagates_from_gather above); this test pins the *added*
        logging side-effect.

        Three entities: 'uuid-bad-one' and 'uuid-bad-two' both raise distinct
        RuntimeErrors; 'uuid-ok' succeeds. asyncio.gather preserves input order in
        its results list regardless of completion order, so the positionally-first
        Exception (uuid-bad-one's) is the one raised, and both failures are logged.

        The CancelledError-propagates-with-zero-warnings guarantee is not
        re-asserted here: Pass 1 (inside gather_or_raise) re-raises a captured
        BaseException before Pass 2's logging loop even runs (see
        fused_memory/utils/async_utils.py), and test_cancelled_error_propagates_from_gather
        above already pins CancelledError propagation for this call site.
        """
        entities = [
            {'uuid': 'uuid-bad-one', 'name': 'BadOne', 'summary': 'fact'},
            {'uuid': 'uuid-ok', 'name': 'OK', 'summary': 'fact'},
            {'uuid': 'uuid-bad-two', 'name': 'BadTwo', 'summary': 'fact'},
        ]
        backend = make_edge_backend(make_backend(mock_config), nodes=entities, edges={})

        async def edges_with_errors(node_uuid, *, group_id):
            if node_uuid == 'uuid-bad-one':
                raise RuntimeError('first failure')
            if node_uuid == 'uuid-bad-two':
                raise RuntimeError('second failure')
            return [{'uuid': 'e1', 'fact': 'fact', 'name': 'edge1'}]

        backend.get_valid_edges_for_node = AsyncMock(side_effect=edges_with_errors)

        with (
            caplog.at_level(logging.WARNING, logger='fused_memory.backends.graphiti_client'),
            pytest.raises(RuntimeError, match='first failure'),
        ):
            await backend.detect_stale_dry_run(group_id='test')

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2, (
            f'expected one WARNING per captured Exception (2), got {len(warnings)}: '
            f'{[r.message for r in warnings]}'
        )
        assert any('first failure' in r.message for r in warnings)
        assert any('second failure' in r.message for r in warnings)
        assert all('RuntimeError' in r.message for r in warnings)

    @pytest.mark.asyncio
    async def test_edge_fetch_failure_warns_under_the_callers_logger(
        self, mock_config, make_backend, make_edge_backend, caplog
    ):
        """Every warning the probe emits follows ``log``, the per-node edge fetch's too."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'fact'},
        ], edges={})
        backend.get_valid_edges_for_node = AsyncMock(side_effect=RuntimeError('boom'))

        with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match='boom'):
            await backend.detect_stale_dry_run(
                group_id='test', log=logging.getLogger('test.stale_probe')
            )

        edge_warnings = {
            r.name for r in caplog.records
            if r.levelno == logging.WARNING and 'get_valid_edges_for_node' in r.getMessage()
        }
        assert edge_warnings == {'test.stale_probe'}

    @pytest.mark.asyncio
    async def test_result_correctness_under_parallel_execution(
        self, mock_config, make_backend, make_edge_backend
    ):
        """Stale/clean/empty distribution is preserved correctly under parallel execution.

        10 entities with deterministic distribution:
        - 4 stale (summary differs from canonical edge facts)
        - 3 clean (summary matches)
        - 3 empty-summary (cheap-skipped, not stale)

        Assert:
        - result.total_count == 10
        - result.stale contains exactly the 4 stale entity UUIDs (order-independent)
        - result.stale does NOT contain any clean or empty-summary UUIDs
        """
        stale_uuids = {f'uuid-stale-{i}' for i in range(4)}
        clean_uuids = {f'uuid-clean-{i}' for i in range(3)}
        empty_uuids = {f'uuid-empty-{i}' for i in range(3)}

        entities = (
            [{'uuid': u, 'name': u, 'summary': 'old fact'} for u in sorted(stale_uuids)]
            + [{'uuid': u, 'name': u, 'summary': 'current fact'} for u in sorted(clean_uuids)]
            + [{'uuid': u, 'name': u, 'summary': ''} for u in sorted(empty_uuids)]
        )
        backend = make_edge_backend(make_backend(mock_config), nodes=entities, edges={})

        async def edges_by_uuid(node_uuid, *, group_id):
            # stale entities: edge fact differs from their summary ('old fact')
            if node_uuid in stale_uuids:
                return [{'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'}]
            # clean entities: edge fact matches their summary
            if node_uuid in clean_uuids:
                return [{'uuid': 'e2', 'fact': 'current fact', 'name': 'edge2'}]
            return []

        backend.get_valid_edges_for_node = AsyncMock(side_effect=edges_by_uuid)

        # max_concurrency=2 with 10 entities exercises bounding alongside correctness
        result = await backend.detect_stale_dry_run(group_id='test', max_concurrency=2)

        assert result.total_count == 10
        returned_uuids = {e['uuid'] for e in result.stale}
        assert returned_uuids == stale_uuids
        # No clean or empty entities in the stale list
        assert returned_uuids.isdisjoint(clean_uuids)
        assert returned_uuids.isdisjoint(empty_uuids)
        # Empty-summary entities never trigger edge queries
        for empty_uuid in empty_uuids:
            # Verify no call was made for empty-summary entities
            for call in backend.get_valid_edges_for_node.call_args_list:
                assert call.args[0] != empty_uuid, (
                    f"Expected no edge fetch for empty-summary entity {empty_uuid}"
                )

    @pytest.mark.asyncio
    async def test_complete_node_read_reports_a_complete_verdict(
        self, mock_config, make_backend, make_edge_backend
    ):
        """A proven-complete node read is reported as such, read at the enumerate seam."""
        backend = make_edge_backend(make_backend(mock_config), nodes=[
            {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'current fact'},
        ], edges={})
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'},
        ])

        result = await backend.detect_stale_dry_run(group_id='test')

        assert result.entities_completeness == ReadCompleteness(
            complete=True, incomplete_kind=None
        )
        backend.enumerate_entity_nodes.assert_awaited_once_with(group_id='test')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('kind', [INCOMPLETE_SHORT_READ, INCOMPLETE_CENSUS_UNAVAILABLE])
    async def test_empirical_node_read_is_reported_and_the_probe_proceeds(
        self, kind, mock_config, make_backend, make_edge_backend, caplog
    ):
        """An EMPIRICALLY incomplete node read warns under the caller's logger and proceeds."""
        backend = make_edge_backend(
            make_backend(mock_config),
            nodes=[{'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'old fact'}],
            edges={},
            nodes_read=incomplete_paged_read(kind),
        )
        backend.get_valid_edges_for_node = AsyncMock(return_value=[
            {'uuid': 'e1', 'fact': 'current fact', 'name': 'edge1'},
        ])

        with caplog.at_level(logging.WARNING):
            result = await backend.detect_stale_dry_run(
                group_id='test', log=logging.getLogger('test.stale_probe')
            )

        assert result.entities_completeness == ReadCompleteness(
            complete=False, incomplete_kind=kind
        )
        assert [e['uuid'] for e in result.stale] == ['uuid-1']
        assert node_read_warnings(caplog, 'test.stale_probe')
        assert not node_read_warnings(caplog, 'fused_memory.backends.graphiti_client')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('kind', sorted(INCOMPLETE_STRUCTURAL_KINDS))
    async def test_structural_node_read_raises_before_any_edge_fetch(
        self, kind, mock_config, make_backend, make_edge_backend
    ):
        """A STRUCTURALLY incomplete node read is not an answer, so no edge is fetched."""
        backend = make_edge_backend(
            make_backend(mock_config),
            nodes=[
                {'uuid': 'uuid-1', 'name': 'Alice', 'summary': 'a real summary'},
                {'uuid': 'uuid-2', 'name': 'Bob', 'summary': 'another real summary'},
            ],
            edges={},
            nodes_read=incomplete_paged_read(kind),
        )
        backend.get_valid_edges_for_node = AsyncMock(return_value=[])

        with pytest.raises(IncompleteEnumerationError):
            await backend.detect_stale_dry_run(group_id='test')

        backend.get_valid_edges_for_node.assert_not_awaited()

    def test_probe_result_is_frozen_and_keyword_only(self):
        result = StaleProbeResult(stale=[], total_count=0, entities_completeness=COMPLETE_READ)

        for field in dataclasses.fields(StaleProbeResult):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(result, field.name, None)
        pytest.raises(TypeError, StaleProbeResult, [], 0, COMPLETE_READ)
