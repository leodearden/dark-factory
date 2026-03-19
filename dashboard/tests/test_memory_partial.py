"""Tests for /partials/memory route and memory.html template."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

_ONLINE_STATUS = {
    'graphiti': {'connected': True, 'node_count': 42},
    'mem0': {'connected': True, 'memory_count': 128},
}

_ONLINE_QUEUE = {
    'counts': {'pending': 0, 'retry': 0, 'dead': 0},
    'oldest_pending_age_seconds': None,
}


class TestMemoryPartialOnline:
    def test_route_returns_200_html(self, client):
        with (
            patch(
                'dashboard.data.memory.get_memory_status',
                new_callable=AsyncMock,
                return_value=_ONLINE_STATUS,
            ),
            patch(
                'dashboard.data.memory.get_queue_stats',
                new_callable=AsyncMock,
                return_value=_ONLINE_QUEUE,
            ),
        ):
            resp = client.get('/partials/memory')
            assert resp.status_code == 200
            assert 'text/html' in resp.headers['content-type']

    def test_online_contains_card_labels(self, client):
        with (
            patch(
                'dashboard.data.memory.get_memory_status',
                new_callable=AsyncMock,
                return_value=_ONLINE_STATUS,
            ),
            patch(
                'dashboard.data.memory.get_queue_stats',
                new_callable=AsyncMock,
                return_value=_ONLINE_QUEUE,
            ),
        ):
            html = client.get('/partials/memory').text
            assert 'Graphiti' in html
            assert 'Mem0' in html
            assert 'Taskmaster' in html
            assert 'Write Queue' in html

    def test_online_graphiti_count(self, client):
        with (
            patch(
                'dashboard.data.memory.get_memory_status',
                new_callable=AsyncMock,
                return_value=_ONLINE_STATUS,
            ),
            patch(
                'dashboard.data.memory.get_queue_stats',
                new_callable=AsyncMock,
                return_value=_ONLINE_QUEUE,
            ),
        ):
            html = client.get('/partials/memory').text
            assert '42' in html
            assert 'knowledge graph nodes' in html

    def test_online_mem0_count(self, client):
        with (
            patch(
                'dashboard.data.memory.get_memory_status',
                new_callable=AsyncMock,
                return_value=_ONLINE_STATUS,
            ),
            patch(
                'dashboard.data.memory.get_queue_stats',
                new_callable=AsyncMock,
                return_value=_ONLINE_QUEUE,
            ),
        ):
            html = client.get('/partials/memory').text
            assert '128' in html
            assert 'vector memories' in html

    def test_online_taskmaster_connected(self, client):
        with (
            patch(
                'dashboard.data.memory.get_memory_status',
                new_callable=AsyncMock,
                return_value=_ONLINE_STATUS,
            ),
            patch(
                'dashboard.data.memory.get_queue_stats',
                new_callable=AsyncMock,
                return_value=_ONLINE_QUEUE,
            ),
        ):
            html = client.get('/partials/memory').text
            assert 'Connected' in html
