"""Tests for the task lineage index: the typed projected read under it
(``dashboard.data.tasks.fetch_metadata_projection``) and the per-root store
that serves it (PRD dashboard-prd-lanes §4.2, §5.1, §5.3)."""

from __future__ import annotations

from unittest.mock import patch

from _canned_mcp import CannedMCP, _raw_row

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

    async def test_a_projected_row_is_not_a_shaped_task_row(self, dummy_client, dummy_config):
        canned = CannedMCP(_rows())
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            result = await fetch_metadata_projection(
                dummy_client, dummy_config, ROOT, metadata_keys=('prd_path',),
            )
        assert isinstance(result, tuple)
        assert not hasattr(result[0], 'title')
        assert not hasattr(result[0], 'status')

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

