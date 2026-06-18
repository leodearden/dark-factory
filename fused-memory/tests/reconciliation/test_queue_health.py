"""Tests for reconciliation/queue_health.py — summarize_graphiti_queue_health.

Step-5 (RED): module/function does not exist yet.
Step-6 (GREEN): create queue_health.py with the function.
"""

from __future__ import annotations

import logging

import pytest

from fused_memory.reconciliation.queue_health import summarize_graphiti_queue_health

_LOGGER = "fused_memory.reconciliation.queue_health"


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

    # --- Step-1 (task 1810): malformed-stats WARNING tests ---

    @pytest.mark.parametrize("bad_stats", [None, "oops", [1, 2, 3]])
    def test_non_dict_stats_yields_unhealthy_and_warning(self, bad_stats, caplog):
        """stats that is not a dict => healthy=False AND a WARNING on the module logger.

        Currently RED: non-dict stats yields healthy=True and no logger exists.
        """
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = summarize_graphiti_queue_health(bad_stats)

        assert result['healthy'] is False, (
            f"non-dict stats {bad_stats!r} must yield healthy=False"
        )
        warns = [
            r for r in caplog.records
            if r.name == _LOGGER and r.levelno >= logging.WARNING
        ]
        assert warns, (
            f"expected a WARNING on {_LOGGER!r} for non-dict stats, got none"
        )

    def test_counts_present_but_non_dict_yields_unhealthy_and_warning(self, caplog):
        """stats with 'counts' present but not a dict => healthy=False + WARNING.

        Currently RED: the 'corrupt' string falls through to counts={} => healthy=True.
        """
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = summarize_graphiti_queue_health({'counts': 'corrupt'})

        assert result['healthy'] is False, (
            "counts='corrupt' (present-but-not-dict) must yield healthy=False"
        )
        warns = [
            r for r in caplog.records
            if r.name == _LOGGER and r.levelno >= logging.WARNING
        ]
        assert warns, (
            f"expected a WARNING on {_LOGGER!r} for present-but-non-dict counts"
        )

    def test_absent_counts_key_is_benign_no_warning(self, caplog):
        """Missing 'counts' key is benign: healthy=True with ZERO warnings.

        Regression: the benign-absent path must stay silent.
        """
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = summarize_graphiti_queue_health({'oldest_pending_age_seconds': None})

        assert result['healthy'] is True
        warns = [
            r for r in caplog.records
            if r.name == _LOGGER and r.levelno >= logging.WARNING
        ]
        assert not warns, (
            f"absent 'counts' key must not emit any WARNING; got {warns!r}"
        )

    def test_healthy_case_no_warning(self, caplog):
        """REGRESSION: healthy stats emit ZERO warnings."""
        stats = {'counts': {'completed': 10}, 'oldest_pending_age_seconds': 0.5}
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = summarize_graphiti_queue_health(stats)

        assert result['healthy'] is True
        warns = [
            r for r in caplog.records
            if r.name == _LOGGER and r.levelno >= logging.WARNING
        ]
        assert not warns, f"healthy stats must not emit WARNINGs; got {warns!r}"

    def test_unhealthy_case_no_extra_warning(self, caplog):
        """REGRESSION: dead-letter unhealthy case emits ZERO warnings (only failure branch warns)."""
        stats = {'counts': {'dead': 1, 'completed': 5}}
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = summarize_graphiti_queue_health(stats)

        assert result['healthy'] is False
        warns = [
            r for r in caplog.records
            if r.name == _LOGGER and r.levelno >= logging.WARNING
        ]
        assert not warns, (
            f"dead-letter unhealthy case must not emit WARNINGs; got {warns!r}"
        )
