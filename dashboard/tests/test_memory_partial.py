"""Tests for /partials/memory route and memory.html template."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

_ONLINE_STATUS = {
    'graphiti': {'connected': True},
    'mem0': {'connected': True},
    'taskmaster': {'connected': True},
    'projects': {
        'dark_factory': {'graphiti_nodes': 42, 'mem0_memories': 128},
    },
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

    def test_memory_heading_present(self, client):
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
            assert '<h2' in html
            assert 'Memory' in html

    def test_online_contains_infra_labels(self, client):
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
            assert 'Queue' in html

    def test_online_project_graphiti_count(self, client):
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
            assert 'graph nodes' in html

    def test_online_project_mem0_count(self, client):
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

    def test_online_project_name_shown(self, client):
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
            assert 'dark_factory' in html

    def test_multiple_projects_shown(self, client):
        status = {
            'graphiti': {'connected': True},
            'mem0': {'connected': True},
            'projects': {
                'dark_factory': {'graphiti_nodes': 42, 'mem0_memories': 128},
                'reify': {'graphiti_nodes': 100, 'mem0_memories': 50},
            },
        }
        with (
            patch(
                'dashboard.data.memory.get_memory_status',
                new_callable=AsyncMock,
                return_value=status,
            ),
            patch(
                'dashboard.data.memory.get_queue_stats',
                new_callable=AsyncMock,
                return_value=_ONLINE_QUEUE,
            ),
        ):
            html = client.get('/partials/memory').text
            assert 'dark_factory' in html
            assert 'reify' in html
            assert '42' in html
            assert '100' in html


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
            assert 'graph nodes' not in html


class TestMemoryPartialDots:
    def test_graphiti_green_when_connected(self, client):
        status = {
            'graphiti': {'connected': True},
            'mem0': {'connected': True},
            'projects': {'test': {'graphiti_nodes': 10, 'mem0_memories': 5}},
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
            'graphiti': {'connected': False},
            'mem0': {'connected': True},
            'projects': {},
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
            'graphiti': {'connected': True},
            'mem0': {'connected': False},
            'projects': {},
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
            # Graphiti, Mem0, Taskmaster, Queue — all 4 should be green
            assert html.count('bg-green-500') >= 3

    def test_taskmaster_red_when_disconnected(self, client):
        status = {
            'graphiti': {'connected': True},
            'mem0': {'connected': True},
            'taskmaster': {'connected': False},
            'projects': {'test': {'graphiti_nodes': 10, 'mem0_memories': 5}},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'bg-red-500' in html

    def test_taskmaster_defaults_disconnected_when_key_missing(self, client):
        status = {
            'graphiti': {'connected': True},
            'mem0': {'connected': True},
            'projects': {'test': {'graphiti_nodes': 10, 'mem0_memories': 5}},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'bg-red-500' in html


class TestWriteQueueCard:
    def test_queue_dot_green_all_zero(self, client):
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            # All infra dots should be green (4 items in health row)
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
            assert '3p/' not in html
            assert '1r/' not in html

    def test_queue_tooltip_title_attribute(self, client):
        queue = {'counts': {'pending': 2, 'retry': 1, 'dead': 5}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'title="Write Queue: 2 pending / 1 retry / 5 dead letters"' in html


class TestWriteQueueConditionalStyling:
    def test_dead_count_red_text(self, client):
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 2}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'bg-red-500' in html

    def test_oldest_age_shown_over_60(self, client):
        queue = {'counts': {'pending': 1, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': 120}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'oldest' in html
            assert '2m ago' in html
            assert 'text-yellow-' in html

    def test_oldest_age_hidden_under_60(self, client):
        queue = {'counts': {'pending': 1, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': 30}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'oldest' not in html

    def test_oldest_age_hidden_none(self, client):
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
            assert 'oldest' not in html


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
            assert 'oldest' not in html

    def test_status_online_queue_offline_shows_infra_labels(self, client):
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=_SPLIT_BRAIN_QUEUE),
        ):
            html = client.get('/partials/memory').text
            assert 'Graphiti' in html
            assert 'Mem0' in html
            assert 'Taskmaster' in html
            assert 'Queue' in html

    def test_status_online_queue_offline_queue_metrics_default(self, client):
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=_SPLIT_BRAIN_QUEUE),
        ):
            html = client.get('/partials/memory').text
            assert '0 pending' in html
            assert '0 retry' in html
            assert '0 dead' in html
            assert '0p/' not in html
            assert '0r/' not in html


class TestMemoryDotAriaLabels:
    """Tests for ARIA attributes on status indicator dots in memory.html."""

    def test_graphiti_dot_has_role_status(self, client):
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=_ONLINE_QUEUE),
        ):
            html = client.get('/partials/memory').text
        assert 'role="status"' in html

    def test_graphiti_dot_aria_label_connected(self, client):
        status = {
            'graphiti': {'connected': True},
            'mem0': {'connected': True},
            'taskmaster': {'connected': True},
            'projects': {'test': {'graphiti_nodes': 10, 'mem0_memories': 5}},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
        assert 'aria-label="Graphiti connected"' in html

    def test_graphiti_dot_aria_label_disconnected(self, client):
        status = {
            'graphiti': {'connected': False},
            'mem0': {'connected': True},
            'taskmaster': {'connected': True},
            'projects': {},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
        assert 'aria-label="Graphiti disconnected"' in html

    def test_mem0_dot_aria_label_connected(self, client):
        status = {
            'graphiti': {'connected': True},
            'mem0': {'connected': True},
            'taskmaster': {'connected': True},
            'projects': {'test': {'graphiti_nodes': 10, 'mem0_memories': 5}},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
        assert 'aria-label="Mem0 connected"' in html

    def test_mem0_dot_aria_label_disconnected(self, client):
        status = {
            'graphiti': {'connected': True},
            'mem0': {'connected': False},
            'taskmaster': {'connected': True},
            'projects': {},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
        assert 'aria-label="Mem0 disconnected"' in html

    def test_taskmaster_dot_aria_label_connected(self, client):
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=_ONLINE_QUEUE),
        ):
            html = client.get('/partials/memory').text
        assert 'aria-label="Taskmaster connected"' in html

    def test_taskmaster_dot_aria_label_disconnected(self, client):
        status = {
            'graphiti': {'connected': True},
            'mem0': {'connected': True},
            'taskmaster': {'connected': False},
            'projects': {'test': {'graphiti_nodes': 10, 'mem0_memories': 5}},
        }
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=status),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
        assert 'aria-label="Taskmaster disconnected"' in html

    def test_write_queue_dot_aria_label_healthy(self, client):
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
        assert 'aria-label="Write Queue healthy"' in html

    def test_write_queue_dot_aria_label_pending(self, client):
        queue = {'counts': {'pending': 3, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
        assert 'aria-label="Write Queue pending"' in html

    def test_write_queue_dot_aria_label_dead_letters(self, client):
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 2}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
        assert 'aria-label="Write Queue dead letters"' in html

    def test_all_four_dots_have_role_status(self, client):
        queue = {'counts': {'pending': 0, 'retry': 0, 'dead': 0}, 'oldest_pending_age_seconds': None}
        with (
            patch('dashboard.data.memory.get_memory_status', new_callable=AsyncMock, return_value=_ONLINE_STATUS),
            patch('dashboard.data.memory.get_queue_stats', new_callable=AsyncMock, return_value=queue),
        ):
            html = client.get('/partials/memory').text
        assert html.count('role="status"') == 4


class TestMemoryCardLayout:
    def test_cards_container_has_flex_wrap(self, client):
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
            assert 'flex flex-wrap gap-4' in html

    def test_project_cards_have_min_width(self, client):
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
            # One project card = one min-w-[200px]
            assert html.count('min-w-[200px]') == 1

    def test_offline_no_flex_wrap_container(self, client):
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
            html = client.get('/partials/memory').text
            assert 'flex-wrap' not in html
