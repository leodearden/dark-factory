"""Tests for /partials/memory-graphs route and template rendering."""

from __future__ import annotations

import json
import re

from unittest.mock import AsyncMock, patch

_TIMESERIES = {
    'labels': [f'{h:02d}:00' for h in range(24)],
    'reads': [0] * 24,
    'writes': [0] * 24,
}
_OPERATIONS = {'labels': ['search', 'add_memory'], 'values': [100, 50]}
_AGENTS = {'labels': ['claude-interactive', 'recon-consolidator'], 'values': [80, 70]}


def _patch_journal():
    return (
        patch(
            'dashboard.data.write_journal.get_memory_timeseries',
            new_callable=AsyncMock, return_value=_TIMESERIES,
        ),
        patch(
            'dashboard.data.write_journal.get_operations_breakdown',
            new_callable=AsyncMock, return_value=_OPERATIONS,
        ),
        patch(
            'dashboard.data.write_journal.get_agent_breakdown',
            new_callable=AsyncMock, return_value=_AGENTS,
        ),
    )


class TestMemoryGraphsPartial:
    def test_returns_200(self, client):
        p1, p2, p3 = _patch_journal()
        with p1, p2, p3:
            resp = client.get('/partials/memory-graphs')
            assert resp.status_code == 200
            assert 'text/html' in resp.headers['content-type']

    def test_contains_chart_canvases(self, client):
        p1, p2, p3 = _patch_journal()
        with p1, p2, p3:
            html = client.get('/partials/memory-graphs').text
            assert 'memoryTimeseriesChart' in html
            assert 'memoryOpsChart' in html
            assert 'memoryAgentChart' in html

    def test_contains_inline_data(self, client):
        p1, p2, p3 = _patch_journal()
        with p1, p2, p3:
            html = client.get('/partials/memory-graphs').text
            assert 'tsData' in html
            assert 'opsData' in html
            assert 'agentData' in html

    def test_contains_section_heading(self, client):
        p1, p2, p3 = _patch_journal()
        with p1, p2, p3:
            html = client.get('/partials/memory-graphs').text
            assert 'last 24' in html

    def test_contains_pie_labels(self, client):
        p1, p2, p3 = _patch_journal()
        with p1, p2, p3:
            html = client.get('/partials/memory-graphs').text
            assert 'By operation' in html
            assert 'By agent' in html


# --- Helper to extract JS variable value from rendered HTML ---

def _extract_js_var(html: str, var_name: str) -> dict:
    """Extract a JSON-assigned JS variable from rendered HTML."""
    pattern = rf'var {var_name}\s*=\s*(\{{.*?\}});'
    match = re.search(pattern, html, re.DOTALL)
    assert match is not None, f'Variable {var_name!r} not found in HTML'
    return json.loads(match.group(1))


_MANY_AGENTS = {
    'labels': ['agent-a', 'agent-b', 'agent-c', 'agent-d', 'agent-e', 'agent-f'],
    'values': [60, 50, 40, 30, 20, 10],
}
_MANY_OPS = {
    'labels': ['search', 'add_memory', 'get_entity', 'add_episode', 'delete_memory', 'replay'],
    'values': [90, 80, 70, 60, 50, 40],
}


class TestTopNGrouping:
    """Tests that group_top_n() is applied in the route handler."""

    def test_agent_data_with_many_entries_has_other(self, client):
        """When agent data has >5 entries, rendered HTML contains 'Other' in agentData."""
        p1 = patch(
            'dashboard.data.write_journal.get_memory_timeseries',
            new_callable=AsyncMock, return_value=_TIMESERIES,
        )
        p2 = patch(
            'dashboard.data.write_journal.get_operations_breakdown',
            new_callable=AsyncMock, return_value=_OPERATIONS,
        )
        p3 = patch(
            'dashboard.data.write_journal.get_agent_breakdown',
            new_callable=AsyncMock, return_value=_MANY_AGENTS,
        )
        with p1, p2, p3:
            html = client.get('/partials/memory-graphs').text
        agent_data = _extract_js_var(html, 'agentData')
        assert 'Other' in agent_data['labels']
        assert len(agent_data['labels']) == 6  # top 5 + Other

    def test_operations_data_with_many_entries_has_other(self, client):
        """When operations data has >5 entries, rendered HTML contains 'Other' in opsData."""
        p1 = patch(
            'dashboard.data.write_journal.get_memory_timeseries',
            new_callable=AsyncMock, return_value=_TIMESERIES,
        )
        p2 = patch(
            'dashboard.data.write_journal.get_operations_breakdown',
            new_callable=AsyncMock, return_value=_MANY_OPS,
        )
        p3 = patch(
            'dashboard.data.write_journal.get_agent_breakdown',
            new_callable=AsyncMock, return_value=_AGENTS,
        )
        with p1, p2, p3:
            html = client.get('/partials/memory-graphs').text
        ops_data = _extract_js_var(html, 'opsData')
        assert 'Other' in ops_data['labels']
        assert len(ops_data['labels']) == 6  # top 5 + Other

    def test_no_other_when_data_within_limit(self, client):
        """When both data sets have <=5 entries, no 'Other' label appears."""
        p1 = patch(
            'dashboard.data.write_journal.get_memory_timeseries',
            new_callable=AsyncMock, return_value=_TIMESERIES,
        )
        p2 = patch(
            'dashboard.data.write_journal.get_operations_breakdown',
            new_callable=AsyncMock, return_value=_OPERATIONS,
        )
        p3 = patch(
            'dashboard.data.write_journal.get_agent_breakdown',
            new_callable=AsyncMock, return_value=_AGENTS,
        )
        with p1, p2, p3:
            html = client.get('/partials/memory-graphs').text
        ops_data = _extract_js_var(html, 'opsData')
        agent_data = _extract_js_var(html, 'agentData')
        assert 'Other' not in ops_data['labels']
        assert 'Other' not in agent_data['labels']


class TestTimeseriesContainer:
    """Tests for the timeseries chart container CSS."""

    def test_timeseries_container_has_position_relative(self, client):
        """Timeseries chart container div must have position:relative for Chart.js responsive mode."""
        p1, p2, p3 = _patch_journal()
        with p1, p2, p3:
            html = client.get('/partials/memory-graphs').text
        # The container div should include position:relative in its style
        assert 'position:relative' in html or 'position: relative' in html


class TestDoughnutDatalabels:
    """Tests for datalabels plugin configuration on doughnut charts."""

    def test_doughnut_script_contains_datalabels_config(self, client):
        """Doughnut chart script must contain datalabels configuration."""
        p1, p2, p3 = _patch_journal()
        with p1, p2, p3:
            html = client.get('/partials/memory-graphs').text
        assert 'datalabels' in html

    def test_doughnut_script_contains_percentage_formatter(self, client):
        """Doughnut chart datalabels must include a percentage formatter function."""
        p1, p2, p3 = _patch_journal()
        with p1, p2, p3:
            html = client.get('/partials/memory-graphs').text
        # Should have a formatter that computes percentage
        assert 'formatter' in html
        assert '%' in html or 'toFixed' in html
