"""Tests for /partials/memory-graphs route and template rendering."""

from __future__ import annotations

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
