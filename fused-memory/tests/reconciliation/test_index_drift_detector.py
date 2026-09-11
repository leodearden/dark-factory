"""Tests for reconciliation/index_drift_detector.py (task 3709, PRD δ).

The I/O half of the FalkorDB index-provisioning drift detector: it turns an
unhealthy record from `summarize_index_health()` into exactly ONE open level-1
`recon_missing_index` escalation per drifted graph.

Driven by a REAL `EscalationQueue` on `tmp_path` with filesystem-level
assertions, so the storm escape (INV-4) is proven through actual on-disk dedup
rather than a mock's return value — the same way the existing recon dedup/storm
tests prove theirs.

HAZARD compliance: no live FalkorDB graph is read or written, no `FalkorDriver`
or `GraphitiBackend` is constructed, and no index is created or dropped — real
graphs' index state is protected evidence for open escalation esc-3375-1.  The
"graph with a deliberately-absent index" is a health record, not a live graph.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

from fused_memory.reconciliation.index_drift_detector import (
    _MISSING_INDEX_ESCALATION_CATEGORY,
    escalate_missing_indices,
)

_LOGGER = 'fused_memory.reconciliation.index_drift_detector'


def _unhealthy(missing=None) -> dict:
    """A health record shaped exactly as summarize_index_health() returns one.

    Deliberately carries NO graph identity: `summarize_index_health()` does not
    produce one, so the graph a filing is about comes from the `group_id`
    argument to `escalate_missing_indices` and nowhere else.  A `group_id`
    parameter here would read as if it varied the graph under test while
    changing nothing — a fixture edit that leaves the test silently passing.
    """
    if missing is None:
        missing = [
            ('Entity', 'NODE', 'name', 'RANGE'),
            ('Episodic', 'NODE', 'uuid', 'RANGE'),
        ]
    return {
        'healthy': False,
        'missing': sorted(missing),
        'unexpected': [],
        'expected_total': 38,
        'actual_total': 38 - len(missing),
    }


def _healthy() -> dict:
    return {
        'healthy': True,
        'missing': [],
        'unexpected': [],
        'expected_total': 38,
        'actual_total': 38,
    }


def _make_queue(tmp_path, subdir: str = 'recon_esc'):
    from escalation.queue import EscalationQueue

    queue_dir = tmp_path / subdir
    return EscalationQueue(queue_dir), queue_dir


class TestEscalateMissingIndices:
    """escalate_missing_indices files one deduped, structured L1 per drifted graph."""

    # --- (a) BOUNDARY TEST 5 / INV-4 storm escape ---

    def test_repeated_drift_files_exactly_one_escalation(self, tmp_path):
        """Two calls with the same unhealthy record file ONE escalation, not two.

        The standing loud signal is the escalation REMAINING OPEN — that is what
        lets a persistently-missing index (every recon cycle, forever) file once
        instead of storming the queue.
        """
        queue, queue_dir = _make_queue(tmp_path)
        health = _unhealthy()

        first = escalate_missing_indices(
            queue, 'dark_factory', health, project_id='dark_factory'
        )
        second = escalate_missing_indices(
            queue, 'dark_factory', health, project_id='dark_factory'
        )

        assert first is not None, 'The first drift observation must file'
        assert second is None, (
            'A second identical observation must be suppressed by the open L1'
        )

        files = list(queue_dir.glob('esc-*.json'))
        assert len(files) == 1, f'Expected exactly one escalation on disk, got {files}'

        pending = [
            e
            for e in queue.get_pending()
            if e.category == _MISSING_INDEX_ESCALATION_CATEGORY
        ]
        assert len(pending) == 1

    # --- (b) the filed record's routing fields ---

    def test_filed_record_routing_fields(self, tmp_path):
        """task_id is the synthetic graph key; filed at level 1, blocking, pending."""
        queue, _ = _make_queue(tmp_path)

        esc_id = escalate_missing_indices(
            queue, 'dark_factory', _unhealthy(), project_id='dark_factory'
        )

        assert esc_id is not None
        (record,) = queue.get_pending()
        assert record.task_id == 'graph:dark_factory'
        assert record.level == 1
        assert record.severity == 'blocking'
        assert record.status == 'pending'
        assert record.category == _MISSING_INDEX_ESCALATION_CATEGORY

    # --- (c) INV-2: structured payload, no prose parsing ---

    def test_structured_payload_is_readable_without_parsing_prose(self, tmp_path):
        """group_id and missing specs are first-class fields on the on-disk record."""
        queue, queue_dir = _make_queue(tmp_path)
        health = _unhealthy()

        escalate_missing_indices(
            queue, 'dark_factory', health, project_id='dark_factory'
        )

        (path,) = list(queue_dir.glob('esc-*.json'))
        record = json.loads(path.read_text())

        assert record['index_health'] is not None
        assert record['index_health']['group_id'] == 'dark_factory'
        assert record['index_health']['missing'] == [
            list(spec) for spec in health['missing']
        ]
        assert record['index_health']['expected_total'] == health['expected_total']
        # Raw measurement stays in the evidence channel, separate from the
        # structured payload.
        assert record['evidence'], 'A raw-observation evidence entry must be present'

    # --- (d) INV-7: a machine-attributable exit owner ---

    def test_suggested_action_names_the_exit_owner(self, tmp_path):
        """suggested_action names the recon-escalation-watcher / port 8103."""
        queue, _ = _make_queue(tmp_path)

        escalate_missing_indices(
            queue, 'dark_factory', _unhealthy(), project_id='dark_factory'
        )

        (record,) = queue.get_pending()
        assert '8103' in record.suggested_action, (
            'The exit owner (reconciliation escalation queue, port 8103) must be '
            f'named: {record.suggested_action!r}'
        )

    # --- (e) a healthy graph is silent ---

    def test_healthy_record_files_nothing_and_never_consults_the_queue(self, tmp_path):
        """A healthy graph files no escalation and does not even touch the queue."""
        queue = MagicMock()

        result = escalate_missing_indices(
            queue, 'dark_factory', _healthy(), project_id='dark_factory'
        )

        assert result is None
        queue.has_open_l1.assert_not_called()
        queue.submit.assert_not_called()

    def test_none_health_files_nothing(self, tmp_path):
        """A None record (unreadable/absent graph) is not drift."""
        queue = MagicMock()

        assert (
            escalate_missing_indices(
                queue, 'dark_factory', None, project_id='dark_factory'
            )
            is None
        )
        queue.submit.assert_not_called()

    # --- (f) category-scoped dedup ---

    def test_dedup_is_category_scoped(self):
        """An unrelated open L1 on the same graph key must not suppress this finding."""
        queue = MagicMock()
        queue.has_open_l1.side_effect = (
            lambda tid, *, category=None: category == 'some_other_category'
        )
        queue.make_id.return_value = 'esc-graph:dark_factory-0001'

        result = escalate_missing_indices(
            queue, 'dark_factory', _unhealthy(), project_id='dark_factory'
        )

        assert result is not None, (
            'An open L1 in a DIFFERENT category must not suppress a genuine '
            'index-drift finding'
        )
        queue.has_open_l1.assert_called_once_with(
            'graph:dark_factory', category=_MISSING_INDEX_ESCALATION_CATEGORY
        )
        queue.submit.assert_called_once()
        submitted = queue.submit.call_args[0][0]
        assert submitted.category == _MISSING_INDEX_ESCALATION_CATEGORY, (
            'The record must be SUBMITTED under the same category it deduped on, '
            'or the next call will not find it and will file again'
        )

    # --- (g) fail-soft ---

    def test_submit_failure_is_fail_soft(self, caplog):
        """A queue whose submit raises returns None, logs a WARNING, raises nothing."""
        queue = MagicMock()
        queue.has_open_l1.return_value = False
        queue.make_id.return_value = 'esc-graph:dark_factory-0001'
        queue.submit.side_effect = RuntimeError('disk full')

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = escalate_missing_indices(
                queue, 'dark_factory', _unhealthy(), project_id='dark_factory'
            )

        assert result is None
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, 'A submit failure must be logged, not swallowed silently'

    def test_dedup_read_failure_is_fail_soft(self, caplog):
        """A failing has_open_l1 returns None and logs — it must never propagate.

        `has_open_l1` → `get_by_task` globs the queue root and parses every
        pending record, so an OSError (fd exhaustion, a permission flap, a disk
        error) is a real failure mode there and not only on `submit`.  Escaping,
        it would reach `_detect_index_drift`, whose caller's guard would discard
        the health record the graph read had ALREADY produced — a queue-read
        hiccup silently erasing a good read from the Stage 1 report.

        It fails CLOSED (files nothing) rather than filing blind: with no dedup
        answer, filing risks the once-per-cycle storm the dedup exists to
        prevent, and the drift is re-observed next cycle anyway.
        """
        queue = MagicMock()
        queue.has_open_l1.side_effect = OSError('too many open files')

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = escalate_missing_indices(
                queue, 'dark_factory', _unhealthy(), project_id='dark_factory'
            )

        assert result is None
        # With no dedup answer, filing blind would risk the storm the dedup
        # exists to prevent.
        queue.submit.assert_not_called()
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, 'A dedup-read failure must be logged, not swallowed silently'
        assert any('dark_factory' in r.getMessage() for r in warnings), (
            'The WARNING must name the graph it failed for'
        )

    def test_malformed_health_record_is_fail_soft(self):
        """Payload construction is inside the try too — nothing escapes.

        `health['missing']` bound to something un-listable is not a shape this
        module validates; it must degrade to "not filed", never to an exception
        that costs the caller its health record.
        """
        queue = MagicMock()
        queue.has_open_l1.return_value = False
        queue.make_id.return_value = 'esc-graph:dark_factory-0001'
        health = _unhealthy()
        health['missing'] = 42  # not iterable — list() raises

        assert (
            escalate_missing_indices(
                queue, 'dark_factory', health, project_id='dark_factory'
            )
            is None
        )
        queue.submit.assert_not_called()

    def test_make_id_failure_is_fail_soft(self):
        """id-generation failures are inside the try too — nothing escapes."""
        queue = MagicMock()
        queue.has_open_l1.return_value = False
        queue.make_id.side_effect = RuntimeError('counter unreadable')

        assert (
            escalate_missing_indices(
                queue, 'dark_factory', _unhealthy(), project_id='dark_factory'
            )
            is None
        )

    # --- suppression is observable ---

    def test_suppressed_call_logs_at_info(self, tmp_path, caplog):
        """The dedup suppression is visible at INFO, not silent."""
        queue, _ = _make_queue(tmp_path)
        health = _unhealthy()
        escalate_missing_indices(
            queue, 'dark_factory', health, project_id='dark_factory'
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER):
            escalate_missing_indices(
                queue, 'dark_factory', health, project_id='dark_factory'
            )

        assert any(
            'suppress' in r.getMessage().lower() for r in caplog.records
        ), 'The suppressed second filing must say so at INFO'

    # --- per-graph isolation ---

    def test_two_graphs_dedup_independently(self, tmp_path):
        """One graph's open escalation must not suppress another graph's finding."""
        queue, queue_dir = _make_queue(tmp_path)

        first = escalate_missing_indices(
            queue, 'dark_factory', _unhealthy(), project_id='dark_factory'
        )
        second = escalate_missing_indices(
            queue, 'autotrade', _unhealthy(), project_id='autotrade'
        )

        assert first is not None and second is not None
        assert len(list(queue_dir.glob('esc-*.json'))) == 2
        task_ids = {e.task_id for e in queue.get_pending()}
        assert task_ids == {'graph:dark_factory', 'graph:autotrade'}
