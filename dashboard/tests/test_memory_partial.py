"""Tests for /partials/memory route and memory.html template."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

_ONLINE_STATUS = {
    'graphiti': {'connected': True, 'node_count': 42},
    'mem0': {'connected': True, 'memory_count': 128},
    'taskmaster': {'connected': True},
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


_OFFLINE_STATUS = {'offline': True, 'error': 'Connection refused'}
_OFFLINE_QUEUE = {'offline': True, 'error': 'Connection refused'}


class TestMemoryPartialOffline:
    def test_offline_shows_error_card(self, client):
        with (
            patch(
                'dashboard.data.memory.get_memory_status',
                new_callable=AsyncMock,
                return_value=_OFFLINE_STATUS,
            ),
            patch(
                'dashboard.data.memory.get_queue_stats',
                new_callable=AsyncMock,
                return_value=_OFFLINE_QUEUE,
            ),
        ):
            resp = client.get('/partials/memory')
            assert resp.status_code == 200
            html = resp.text
            assert 'Fused Memory Server Offline' in html
            assert 'Connection refused' in html
            assert 'knowledge graph nodes' not in html


class TestMemoryPartialDots:
    def test_graphiti_green_when_connected(self, client):
        status = {
            'graphiti': {'connected': True, 'node_count': 10},
            'mem0': {'connected': True, 'memory_count': 5},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'bg-green-500' in html

    def test_graphiti_red_when_disconnected(self, client):
        status = {
            'graphiti': {'connected': False, 'node_count': 0},
            'mem0': {'connected': True, 'memory_count': 5},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'bg-red-500' in html

    def test_mem0_red_when_disconnected(self, client):
        status = {
            'graphiti': {'connected': True, 'node_count': 10},
            'mem0': {'connected': False, 'memory_count': 0},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'bg-red-500' in html

    def test_taskmaster_green_when_online(self, client):
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=_ONLINE_QUEUE),
        ):
            html = client.get('/partials/memory').text
            # All 4 cards should have green dots when everything is connected
            assert html.count('bg-green-500') >= 3

    def test_taskmaster_red_when_disconnected(self, client):
        status = {
            'graphiti': {'connected': True, 'node_count': 10},
            'mem0': {'connected': True, 'memory_count': 5},
            'taskmaster': {'connected': False},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'bg-red-500' in html

    def test_taskmaster_shows_disconnected_text(self, client):
        status = {
            'graphiti': {'connected': True, 'node_count': 10},
            'mem0': {'connected': True, 'memory_count': 5},
            'taskmaster': {'connected': False},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'Disconnected' in html

    def test_taskmaster_defaults_disconnected_when_key_missing(self, client):
        status = {
            'graphiti': {'connected': True, 'node_count': 10},
            'mem0': {'connected': True, 'memory_count': 5},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'bg-red-500' in html
            assert 'Disconnected' in html


class TestWriteQueueCard:
    def test_queue_dot_green_all_zero(self, client):
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            # All dots should be green (4 cards, all connected, all zero counts)
            assert html.count('bg-green-500') == 4

    def test_queue_dot_yellow_pending(self, client):
        queue = {'counts': {'pending': 3, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'bg-yellow-500' in html

    def test_queue_dot_red_dead(self, client):
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 2}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'bg-red-500' in html

    def test_queue_metrics_inline(self, client):
        queue = {'counts': {'pending': 3, 'retry': 1, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert '3 pending' in html
            assert '1 retry' in html
            assert '0 dead' in html


class TestWriteQueueConditionalStyling:
    def test_dead_count_red_text(self, client):
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 2}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'text-red-' in html

    def test_dead_zero_no_red_text(self, client):
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'text-red-400' not in html

    def test_oldest_age_shown_over_60(self, client):
        queue = {'counts': {'pending': 1, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': 120}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'oldest:' in html
            assert '2m ago' in html
            assert 'text-yellow-' in html

    def test_oldest_age_hidden_under_60(self, client):
        queue = {'counts': {'pending': 1, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': 30}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'oldest:' not in html

    def test_oldest_age_hidden_none(self, client):
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'oldest:' not in html


_SPLIT_BRAIN_QUEUE = {'offline': True, 'error': 'Connection reset'}


class TestSplitBrainQueueOffline:
    """Tests for split-brain: status online but queue offline."""

    def test_status_online_queue_offline_returns_200(self, client):
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=_SPLIT_BRAIN_QUEUE),
        ):
            resp = client.get('/partials/memory')
            assert resp.status_code == 200

    def test_status_online_queue_offline_no_oldest_warning(self, client):
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=_SPLIT_BRAIN_QUEUE),
        ):
            html = client.get('/partials/memory').text
            assert 'oldest:' not in html

    def test_status_online_queue_offline_shows_cards(self, client):
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=_SPLIT_BRAIN_QUEUE),
        ):
            html = client.get('/partials/memory').text
            assert 'Graphiti' in html
            assert 'Mem0' in html
            assert 'Taskmaster' in html
            assert 'Write Queue' in html

    def test_status_online_queue_offline_queue_metrics_default(self, client):
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=_SPLIT_BRAIN_QUEUE),
        ):
            html = client.get('/partials/memory').text
            assert '0 pending' in html
            assert '0 retry' in html
            assert '0 dead' in html
