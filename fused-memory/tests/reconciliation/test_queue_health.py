"""Tests for reconciliation/queue_health.py — summarize_graphiti_queue_health.

Step-5 (RED): module/function does not exist yet.
Step-6 (GREEN): create queue_health.py with the function.
"""

from __future__ import annotations

from fused_memory.reconciliation.queue_health import summarize_graphiti_queue_health


class TestSummarizeGraphitiQueueHealth:
    """summarize_graphiti_queue_health classifies DurableWriteQueue.get_stats() output."""

    def test_dead_letters_mark_unhealthy(self):
        """When dead > 0, healthy=False and dead_count is propagated."""
        stats = {
            'counts': {'completed': 10, 'dead': 2, 'pending': 1},
            'oldest_pending_age_seconds': 42.0,
        }
        result = summarize_graphiti_queue_health(stats)

        assert result['dead_count'] == 2
        assert result['pending_count'] == 1
        assert result['retry_count'] == 0
        assert result['oldest_pending_age_seconds'] == 42.0
        assert result['healthy'] is False, (
            'dead_count=2 must yield healthy=False (silent-drop indicator)'
        )

    def test_all_completed_is_healthy(self):
        """When no dead entries, healthy=True."""
        stats = {
            'counts': {'completed': 20},
            'oldest_pending_age_seconds': None,
        }
        result = summarize_graphiti_queue_health(stats)

        assert result['dead_count'] == 0
        assert result['pending_count'] == 0
        assert result['retry_count'] == 0
        assert result['healthy'] is True

    def test_empty_stats_degrades_gracefully(self):
        """Empty dict degrades to healthy=True with zeroed counts — no KeyError."""
        result = summarize_graphiti_queue_health({})

        assert result['healthy'] is True
        assert result['dead_count'] == 0
        assert result['pending_count'] == 0
        assert result['retry_count'] == 0
        assert result['oldest_pending_age_seconds'] is None

    def test_missing_counts_key_degrades_gracefully(self):
        """Missing 'counts' key degrades to healthy=True with zeroed counts — no KeyError."""
        stats = {'oldest_pending_age_seconds': 5.0}
        result = summarize_graphiti_queue_health(stats)

        assert result['healthy'] is True
        assert result['dead_count'] == 0
        assert result['oldest_pending_age_seconds'] == 5.0

    def test_retry_count_propagated(self):
        """'retry' status count is exposed as retry_count."""
        stats = {
            'counts': {'completed': 5, 'retry': 3},
            'oldest_pending_age_seconds': 10.0,
        }
        result = summarize_graphiti_queue_health(stats)

        assert result['retry_count'] == 3
        assert result['dead_count'] == 0
        assert result['healthy'] is True

    def test_only_dead_unhealthy_not_retry_or_pending(self):
        """healthy=False only when dead > 0; pending or retry alone do not trigger."""
        stats_pending = {
            'counts': {'pending': 5, 'retry': 2},
            'oldest_pending_age_seconds': 30.0,
        }
        result = summarize_graphiti_queue_health(stats_pending)
        assert result['healthy'] is True, (
            'pending+retry without dead must still be healthy'
        )
