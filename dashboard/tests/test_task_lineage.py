"""Tests for the task lineage index: the typed projected read under it
(``dashboard.data.tasks.fetch_metadata_projection``) and the per-root store
that serves it (PRD dashboard-prd-lanes §4.2, §5.1, §5.3)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from _canned_mcp import CannedMCP, _raw_row

from dashboard.data import task_lineage
from dashboard.data.datum import DatumState, validate_datum
from dashboard.data.prd_attribution import (
    LINEAGE_METADATA_KEYS,
    PrdKey,
    attribute,
    lineage_entry,
)
from dashboard.data.task_lineage import acquire_lineage, clear_lineage_state
from dashboard.data.tasks import ProjectedTask, TaskReadOffline, fetch_metadata_projection

ROOT = '/proj/lineage'

# ---------------------------------------------------------------------------
# the typed projected read
# ---------------------------------------------------------------------------


def _rows() -> list[dict]:
    return [
        _raw_row(3, 'done', metadata={'prd_path': 'plans/a-prd.md', 'files': ['x.py']}),
        _raw_row(1, 'pending', metadata={'spawned_from': 3, 'prd': 'plans/b-prd.md'}),
        _raw_row(2, 'done', metadata={'files': ['y.py']}),
    ]


class TestCannedMCPFieldsProjection:
    """The substrate emulates α's ``fields`` projection (§5.1)."""

    async def test_fields_keep_id_and_only_the_requested_present_metadata_keys(self):
        canned = CannedMCP(_rows())
        answer = await canned(None, 'u', 'get_tasks', {
            'project_root': ROOT,
            'fields': ['id', 'metadata.prd_path', 'metadata.spawned_from'],
        })
        assert answer['tasks'] == [
            {'id': '1', 'metadata': {'spawned_from': 3}},
            {'id': '2', 'metadata': {}},
            {'id': '3', 'metadata': {'prd_path': 'plans/a-prd.md'}},
        ]

    async def test_no_fields_serves_rows_unchanged(self):
        canned = CannedMCP(_rows())
        answer = await canned(None, 'u', 'get_tasks', {'project_root': ROOT, 'fields': None})
        assert answer['tasks'] == sorted(_rows(), key=lambda row: int(row['id']))

    async def test_fields_compose_with_statuses(self):
        canned = CannedMCP(_rows())
        answer = await canned(None, 'u', 'get_tasks', {
            'project_root': ROOT, 'statuses': ['done'], 'fields': ['id', 'status'],
        })
        assert answer['tasks'] == [{'id': '2', 'status': 'done'}, {'id': '3', 'status': 'done'}]

    async def test_requesting_metadata_itself_keeps_the_whole_dict(self):
        canned = CannedMCP(_rows())
        answer = await canned(None, 'u', 'get_tasks', {
            'project_root': ROOT, 'fields': ['metadata', 'metadata.prd_path'],
        })
        assert answer['tasks'][2] == {
            'id': '3', 'metadata': {'prd_path': 'plans/a-prd.md', 'files': ['x.py']},
        }


class TestFetchMetadataProjection:
    """One unpaged ``get_tasks`` with ``fields``, typed, never a ``_shape_task`` row."""

    async def test_issues_one_unpaged_projected_get_tasks(self, dummy_client, dummy_config):
        canned = CannedMCP(_rows())
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            await fetch_metadata_projection(
                dummy_client, dummy_config, ROOT,
                metadata_keys=('prd_path', 'spawned_from'), timeout=1.5,
            )
        [call] = canned.calls_to('get_tasks')
        assert call['args'] == {
            'project_root': ROOT,
            'fields': ['id', 'metadata.prd_path', 'metadata.spawned_from'],
        }
        assert call['kwargs']['timeout'] == 1.5
        assert len(canned.calls) == 1

    async def test_returns_typed_rows_with_integer_ids(self, dummy_client, dummy_config):
        canned = CannedMCP(_rows())
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            result = await fetch_metadata_projection(
                dummy_client, dummy_config, ROOT,
                metadata_keys=('prd_path', 'spawned_from'), timeout=1.5,
            )
        assert isinstance(result, tuple)
        assert [(row.id, dict(row.metadata)) for row in result] == [
            (1, {'spawned_from': 3}),
            (2, {}),
            (3, {'prd_path': 'plans/a-prd.md'}),
        ]
        assert all(isinstance(row, ProjectedTask) for row in result)

    async def test_skips_non_integer_ids_and_normalises_missing_metadata(
        self, dummy_client, dummy_config,
    ):
        async def _answer(client, url, tool, args, **kwargs):
            return {'tasks': [
                {'id': 'x-9', 'metadata': {'prd_path': 'plans/a-prd.md'}},
                {'id': '4'},
                {'id': 5, 'metadata': ['not', 'a', 'dict']},
                {'id': '6', 'metadata': {'prd_path': 'plans/c-prd.md'}},
            ]}

        with patch('dashboard.data.tasks.mcp_tool_call', new=_answer):
            result = await fetch_metadata_projection(
                dummy_client, dummy_config, ROOT, metadata_keys=('prd_path',),
            )
        assert isinstance(result, tuple)
        assert [(row.id, dict(row.metadata)) for row in result] == [
            (4, {}), (5, {}), (6, {'prd_path': 'plans/c-prd.md'}),
        ]

    async def test_every_url_failing_is_offline_naming_the_failure(
        self, dummy_client, dummy_config,
    ):
        canned = CannedMCP(_rows())
        canned.fail_when = lambda call: call['tool'] == 'get_tasks'
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            result = await fetch_metadata_projection(
                dummy_client, dummy_config, ROOT, metadata_keys=('prd_path',),
            )
        assert isinstance(result, TaskReadOffline)
        assert 'canned get_tasks read timeout' in result.detail

    async def test_a_structured_tool_error_is_a_soft_failure(self, dummy_client, dummy_config):
        async def _answer(client, url, tool, args, **kwargs):
            return {'error': 'cannot project spam', 'error_type': 'ValidationError'}

        with patch('dashboard.data.tasks.mcp_tool_call', new=_answer):
            result = await fetch_metadata_projection(
                dummy_client, dummy_config, ROOT, metadata_keys=('prd_path',),
            )
        assert isinstance(result, TaskReadOffline)
        assert 'cannot project spam' in result.detail


# ---------------------------------------------------------------------------
# acquire_lineage: the per-root, background-refreshed lineage index
# ---------------------------------------------------------------------------

T0 = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
OTHER_ROOT = '/proj/other'


@pytest.fixture(autouse=True)
def _fresh_lineage_state() -> Iterator[None]:
    clear_lineage_state()
    yield
    clear_lineage_state()


@pytest.fixture()
def quick_waits(monkeypatch):
    monkeypatch.setattr(task_lineage, 'LINEAGE_REQUEST_WAIT_SECONDS', 0.05)
    monkeypatch.setattr(task_lineage, 'LINEAGE_READ_TIMEOUT', 0.2)


def _projected_calls(canned: CannedMCP, root: str = ROOT) -> list[dict]:
    return [call for call in canned.calls_to('get_tasks') if call['args']['project_root'] == root]


class _Hanging:
    """A substrate whose projected reads hang until healed, then answer as *inner*."""

    def __init__(self, inner: CannedMCP) -> None:
        self.inner = inner
        self.hanging = True
        self.calls = 0

    async def __call__(self, client, url, tool, args, **kwargs):
        self.calls += 1
        if self.hanging:
            await asyncio.Event().wait()
        return await self.inner(client, url, tool, args, **kwargs)


class TestAcquireLineage:
    async def test_cold_first_read_is_fresh_at_the_request_instant(
        self, dummy_client, dummy_config,
    ):
        canned = CannedMCP(_rows())
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            datum = await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)

        assert datum.state is DatumState.FRESH
        assert datum.as_of == T0
        assert datum.freshness_bound_seconds == task_lineage.LINEAGE_FRESHNESS_BOUND_SECONDS
        assert datum.value is not None
        assert dict(datum.value) == {
            1: lineage_entry({'spawned_from': 3, 'prd': 'plans/b-prd.md'}, ROOT),
            2: lineage_entry({}, ROOT),
            3: lineage_entry({'prd_path': 'plans/a-prd.md'}, ROOT),
        }
        assert attribute(datum.value)[0][3].prd == 'plans/a-prd.md'
        assert datum.value[1].own_prd == PrdKey('plans/b-prd.md')
        [call] = _projected_calls(canned)
        assert call['args']['fields'] == [
            'id', *(f'metadata.{key}' for key in LINEAGE_METADATA_KEYS),
        ]

    async def test_a_value_within_its_ttl_is_served_without_a_read(
        self, dummy_client, dummy_config,
    ):
        canned = CannedMCP(_rows())
        ttl = task_lineage.LINEAGE_TTL_SECONDS
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
            within = await acquire_lineage(
                dummy_client, dummy_config, ROOT, now=T0 + timedelta(seconds=ttl - 1),
            )
            assert len(_projected_calls(canned)) == 1
            assert within.as_of == T0
            past = await acquire_lineage(
                dummy_client, dummy_config, ROOT, now=T0 + timedelta(seconds=ttl + 1),
            )
        assert len(_projected_calls(canned)) == 2
        assert past.state is DatumState.FRESH
        assert past.as_of == T0 + timedelta(seconds=ttl + 1)

    async def test_a_failed_first_read_is_unknown_with_the_failure(
        self, dummy_client, dummy_config,
    ):
        canned = CannedMCP(_rows())
        canned.fail_when = lambda call: call['tool'] == 'get_tasks'
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            datum = await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
        assert datum.state is DatumState.UNKNOWN
        assert datum.value is None
        assert datum.reason is not None
        assert 'canned get_tasks read timeout' in datum.reason

    async def test_a_failure_after_a_success_is_stale_at_the_original_as_of(
        self, dummy_client, dummy_config,
    ):
        canned = CannedMCP(_rows())
        later = T0 + timedelta(seconds=task_lineage.LINEAGE_TTL_SECONDS + 1)
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            good = await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
            canned.fail_when = lambda call: call['tool'] == 'get_tasks'
            datum = await acquire_lineage(dummy_client, dummy_config, ROOT, now=later)
        assert datum.state is DatumState.STALE
        assert datum.as_of == T0
        assert datum.value == good.value
        assert datum.reason is not None
        assert 'canned get_tasks read timeout' in datum.reason

    async def test_a_slow_first_read_answers_within_the_wait_then_lands_single_flight(
        self, dummy_client, dummy_config, quick_waits, monkeypatch,
    ):
        monkeypatch.setattr(task_lineage, 'LINEAGE_READ_TIMEOUT', 5.0)
        canned = CannedMCP(_rows(), metadata_projection_delay=0.5)
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            loop = asyncio.get_running_loop()
            started = loop.time()
            pending = await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
            assert loop.time() - started < 0.4
            assert pending.state is DatumState.UNKNOWN
            assert pending.reason is not None
            assert 'in flight' in pending.reason
            await asyncio.sleep(0.6)
            landed = await acquire_lineage(
                dummy_client, dummy_config, ROOT, now=T0 + timedelta(seconds=1),
            )
        assert landed.state is DatumState.FRESH
        assert landed.as_of == T0
        assert len(_projected_calls(canned)) == 1

    async def test_a_hung_read_ends_on_its_budget_and_frees_the_slot(
        self, dummy_client, dummy_config, quick_waits,
    ):
        hanging = _Hanging(CannedMCP(_rows()))
        retry = task_lineage.LINEAGE_RETRY_SECONDS
        with patch('dashboard.data.tasks.mcp_tool_call', new=hanging):
            await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
            await asyncio.sleep(0.3)
            failed = await acquire_lineage(
                dummy_client, dummy_config, ROOT, now=T0 + timedelta(seconds=1),
            )
            assert failed.state is DatumState.UNKNOWN
            assert failed.reason is not None
            assert 'budget' in failed.reason
            assert hanging.calls == 1
            hanging.hanging = False
            healed = await acquire_lineage(
                dummy_client, dummy_config, ROOT, now=T0 + timedelta(seconds=retry + 1),
            )
        assert hanging.calls == 2
        assert healed.state is DatumState.FRESH
        assert healed.as_of == T0 + timedelta(seconds=retry + 1)

    async def test_a_hung_refresh_over_a_last_good_serves_it_stale(
        self, dummy_client, dummy_config, quick_waits,
    ):
        hanging = _Hanging(CannedMCP(_rows()))
        hanging.hanging = False
        later = T0 + timedelta(seconds=task_lineage.LINEAGE_TTL_SECONDS + 1)
        with patch('dashboard.data.tasks.mcp_tool_call', new=hanging):
            good = await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
            hanging.hanging = True
            in_flight = await acquire_lineage(dummy_client, dummy_config, ROOT, now=later)
            assert in_flight.as_of == T0
            assert in_flight.value == good.value
            await asyncio.sleep(0.3)
            datum = await acquire_lineage(
                dummy_client, dummy_config, ROOT, now=later + timedelta(seconds=1),
            )
        assert datum.state is DatumState.STALE
        assert datum.as_of == T0
        assert datum.value == good.value
        assert datum.reason is not None
        assert 'budget' in datum.reason

    async def test_a_last_good_past_its_bound_is_served_stale_while_its_refresh_is_in_flight(
        self, dummy_client, dummy_config, quick_waits,
    ):
        hanging = _Hanging(CannedMCP(_rows()))
        hanging.hanging = False
        bound = task_lineage.LINEAGE_FRESHNESS_BOUND_SECONDS
        past_bound = T0 + timedelta(seconds=bound + 1)
        with patch('dashboard.data.tasks.mcp_tool_call', new=hanging):
            good = await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
            hanging.hanging = True
            datum = await acquire_lineage(dummy_client, dummy_config, ROOT, now=past_bound)
            await asyncio.sleep(0.3)
        assert datum.state is DatumState.STALE
        assert datum.as_of == T0
        assert datum.value == good.value
        assert datum.reason == (
            f'measured {bound + 1}s before it was served, past the {bound}s freshness bound'
        )
        validate_datum(datum, past_bound)

    @pytest.mark.parametrize('seam', ['fetch_metadata_projection', 'lineage_entry'])
    async def test_an_unexpected_error_is_logged_as_a_bug_and_served_unknown(
        self, dummy_client, dummy_config, monkeypatch, caplog, seam,
    ):
        def _bug(*_args, **_kwargs):
            raise RuntimeError('lineage bug')

        monkeypatch.setattr(task_lineage, seam, _bug)
        with (
            patch('dashboard.data.tasks.mcp_tool_call', new=CannedMCP(_rows())),
            caplog.at_level(logging.WARNING, logger=task_lineage.__name__),
        ):
            datum = await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
        assert datum.state is DatumState.UNKNOWN
        assert datum.reason == 'lineage read raised RuntimeError: lineage bug'
        [record] = [r for r in caplog.records if r.name == task_lineage.__name__]
        assert record.levelno == logging.WARNING
        assert record.exc_info is not None
        assert 'BUG' in record.getMessage()

    async def test_an_outage_is_served_without_a_bug_warning(
        self, dummy_client, dummy_config, caplog,
    ):
        canned = CannedMCP(_rows())
        canned.fail_when = lambda call: call['tool'] == 'get_tasks'
        with (
            patch('dashboard.data.tasks.mcp_tool_call', new=canned),
            caplog.at_level(logging.WARNING, logger=task_lineage.__name__),
        ):
            datum = await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
        assert datum.state is DatumState.UNKNOWN
        assert [r for r in caplog.records if r.name == task_lineage.__name__] == []

    async def test_a_failed_read_is_not_retried_before_the_retry_floor(
        self, dummy_client, dummy_config,
    ):
        canned = CannedMCP(_rows())
        canned.fail_when = lambda call: call['tool'] == 'get_tasks'
        retry = task_lineage.LINEAGE_RETRY_SECONDS
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
            early = await acquire_lineage(
                dummy_client, dummy_config, ROOT, now=T0 + timedelta(seconds=retry - 1),
            )
            assert len(_projected_calls(canned)) == 1
            assert early.state is DatumState.UNKNOWN
            canned.fail_when = lambda call: False
            due = await acquire_lineage(
                dummy_client, dummy_config, ROOT, now=T0 + timedelta(seconds=retry),
            )
        assert len(_projected_calls(canned)) == 2
        assert due.state is DatumState.FRESH

    async def test_roots_keep_separate_values_and_failures(self, dummy_client, dummy_config):
        canned = CannedMCP(_rows())
        canned.fail_when = lambda call: call['args'].get('project_root') == OTHER_ROOT
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            good = await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
            bad = await acquire_lineage(dummy_client, dummy_config, OTHER_ROOT, now=T0)
            again = await acquire_lineage(
                dummy_client, dummy_config, ROOT, now=T0 + timedelta(seconds=1),
            )
        assert good.state is DatumState.FRESH
        assert bad.state is DatumState.UNKNOWN
        assert again.state is DatumState.FRESH
        assert again.value == good.value
        assert len(_projected_calls(canned, ROOT)) == 1

    async def test_a_read_abandoned_by_clear_does_not_repopulate_the_store(
        self, dummy_client, dummy_config, quick_waits,
    ):
        canned = CannedMCP(_rows(), metadata_projection_delay=0.15)
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            await acquire_lineage(dummy_client, dummy_config, ROOT, now=T0)
            clear_lineage_state()
            await asyncio.sleep(0.25)
            canned.metadata_projection_delay = 0.0
            datum = await acquire_lineage(
                dummy_client, dummy_config, ROOT, now=T0 + timedelta(seconds=1),
            )
        assert len(_projected_calls(canned)) == 2
        assert datum.state is DatumState.FRESH
        assert datum.as_of == T0 + timedelta(seconds=1)
