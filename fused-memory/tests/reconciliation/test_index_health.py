"""Tests for reconciliation/index_health.py — summarize_index_health (task 3709, PRD δ).

The pure half of the FalkorDB index-provisioning drift detector: it takes an
ACTUAL spec set (what `CALL db.indexes()` reported, normalised by α) and an
EXPECTED spec set (α's `expected_index_set()`) and classifies the difference.
All I/O lives in the caller (`harness._check_index_health`).

HAZARD compliance: these are pure unit tests over set arithmetic.  No live
FalkorDB graph is read or written, no `FalkorDriver` or `GraphitiBackend` is
constructed, and no index is created or dropped anywhere — real graphs' index
state is protected evidence for open escalation esc-3375-1.

Fixtures are DERIVED from `expected_index_set()`, never hard-coded to a pinned
count (the `test_ensure_indices.py` convention), so a graphiti upgrade that
changes the index set cannot make these tests pass against a stale expectation.
"""

from __future__ import annotations

import json
import logging

from fused_memory.backends.falkor_indices import expected_index_set
from fused_memory.reconciliation.index_health import summarize_index_health

_LOGGER = 'fused_memory.reconciliation.index_health'


class TestSummarizeIndexHealth:
    """summarize_index_health diffs an actual spec set against the expected set."""

    # --- (a) fully provisioned ---

    def test_fully_provisioned_graph_is_healthy(self):
        """actual == expected → healthy, with both difference lists empty."""
        expected = expected_index_set()
        result = summarize_index_health(set(expected), expected)

        assert result['healthy'] is True
        assert result['missing'] == []
        assert result['unexpected'] == []
        assert result['expected_total'] == len(expected)
        assert result['actual_total'] == len(expected)

    # --- (b) an absent spec is drift ---

    def test_one_absent_spec_is_unhealthy_and_named(self):
        """A single expected-but-absent spec yields healthy=False and names it."""
        expected = expected_index_set()
        dropped = sorted(expected)[0]
        actual = set(expected) - {dropped}

        result = summarize_index_health(actual, expected)

        assert result['healthy'] is False, (
            'An expected index that is absent from the graph must not report healthy'
        )
        assert result['missing'] == [dropped]
        assert result['unexpected'] == []
        assert result['expected_total'] == len(expected)
        assert result['actual_total'] == len(expected) - 1

    # --- (c) deterministic, diffable payloads ---

    def test_missing_and_unexpected_are_sorted_lists(self):
        """Both difference lists are SORTED lists, so the payload is deterministic.

        The escalation record embeds these verbatim; set iteration order varies
        between processes, which would make two identical findings produce two
        different-looking payloads.
        """
        expected = expected_index_set()
        dropped = set(sorted(expected)[:3])
        extras = {
            ('Entity', 'NODE', 'z_custom_field', 'RANGE'),
            ('Entity', 'NODE', 'a_custom_field', 'RANGE'),
        }
        actual = (set(expected) - dropped) | extras

        result = summarize_index_health(actual, expected)

        assert isinstance(result['missing'], list)
        assert isinstance(result['unexpected'], list)
        assert result['missing'] == sorted(dropped)
        assert result['unexpected'] == sorted(extras)

    # --- (d) unexpected is reported, never acted on (D8) ---

    def test_operator_added_index_is_reported_but_still_healthy(self):
        """A present-but-unexpected index is surfaced and does NOT flip healthy.

        D8: `unexpected` is reported, never acted on — an operator-added index
        is not drift to repair, so it must not manufacture an escalation.
        """
        expected = expected_index_set()
        extra = ('Entity', 'NODE', 'operator_added_field', 'RANGE')
        actual = set(expected) | {extra}

        result = summarize_index_health(actual, expected)

        assert result['unexpected'] == [extra]
        assert result['missing'] == []
        assert result['healthy'] is True, (
            'An operator-added index is not drift — it must not report unhealthy'
        )
        assert result['actual_total'] == len(expected) + 1

    # --- (e) a degenerate expected set fails CLOSED ---

    def test_empty_expected_set_fails_closed_with_warning(self, caplog):
        """An empty expected set forces healthy=False and logs a WARNING.

        `missing == []` is vacuously true when `expected` is empty, so without
        this guard a degenerate expected set would certify every graph on the
        server as perfectly healthy — a silent false all-clear, which is this
        PRD's own failure mode recurring inside the detector meant to catch it.
        """
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = summarize_index_health({('Entity', 'NODE', 'name', 'RANGE')}, set())

        assert result['healthy'] is False, (
            'An empty expected set must fail CLOSED, never launder into a green report'
        )
        assert result['missing'] == []
        assert result['expected_total'] == 0
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, 'A degenerate expected set must go loud, not silently pass'

    def test_empty_expected_and_empty_actual_also_fails_closed(self):
        """Both sets empty is still not a clean bill of health."""
        result = summarize_index_health(set(), set())

        assert result['healthy'] is False
        assert result['missing'] == []
        assert result['unexpected'] == []
        assert result['expected_total'] == 0
        assert result['actual_total'] == 0

    # --- (f) purity + serialisability ---

    def test_inputs_are_not_mutated(self):
        """Neither argument set is mutated — callers reuse `expected_index_set()`."""
        expected = expected_index_set()
        dropped = sorted(expected)[0]
        actual = set(expected) - {dropped}

        expected_before = set(expected)
        actual_before = set(actual)

        summarize_index_health(actual, expected)

        assert expected == expected_before
        assert actual == actual_before

    def test_result_is_json_serialisable(self):
        """The record is plain built-ins — it is embedded in an escalation payload."""
        expected = expected_index_set()
        dropped = sorted(expected)[0]
        result = summarize_index_health(set(expected) - {dropped}, expected)

        encoded = json.dumps(result)
        decoded = json.loads(encoded)

        assert decoded['healthy'] is False
        assert decoded['expected_total'] == len(expected)
        assert decoded['missing'] == [list(dropped)]

    def test_record_has_exactly_the_contract_keys(self):
        """The health record's shape is the PRD δ contract — pinned explicitly."""
        result = summarize_index_health(set(), expected_index_set())

        assert set(result) == {
            'healthy',
            'missing',
            'unexpected',
            'expected_total',
            'actual_total',
        }

    def test_wholly_unprovisioned_graph_reports_every_expected_spec_missing(self):
        """The state every graph is in today: nothing provisioned at all."""
        expected = expected_index_set()
        result = summarize_index_health(set(), expected)

        assert result['healthy'] is False
        assert result['missing'] == sorted(expected)
        assert result['actual_total'] == 0
        assert result['expected_total'] == len(expected)
