"""Tests for the resolved-curator-gate Mem0 source sweep (task 3084).

Stage 1 (``MemoryConsolidator``) has no deterministic sweep that notices when
a human-curator gate task (``metadata.operational_mode == 'gate'``) has in
fact been resolved.  The resolution evidence is already deterministic and
already in Mem0 — the reify curator writes its ruling stamped
``metadata.source == f'curator_gate_{task_id}'`` (independently documented at
``fused-memory/tests/fixtures/README.md``) — but nothing reads it back, so
detection is an ad-hoc Stage-3 spot-check that misses roughly a quarter of
cases (reify run ec45eed0: gates 5561 and 5563 were resolved-but-stale and
went undetected).

Covers:
- curator_gate_source / CURATOR_GATE_SOURCE_TEMPLATE: the single owner of the
  ``curator_gate_{task_id}`` source-key spelling — the one load-bearing
  string of this task.
"""

from __future__ import annotations

from fused_memory.reconciliation.curator_gate_resolution_sweep import (
    CURATOR_GATE_SOURCE_TEMPLATE,
    curator_gate_source,
)


class TestCuratorGateSource:
    """curator_gate_source(task_id) owns the ``curator_gate_{task_id}`` spelling.

    Every Mem0 read this module performs filters on exactly this string, so a
    divergence here is silently a zero-recall sweep — the failure mode this
    task exists to fix.  These tests pin the spelling, the int coercion, and
    the template/helper identity (INV-5: one copy in the tree).
    """

    def test_str_task_id_yields_curator_gate_key(self):
        """A str task id formats to the exact observed key (reify gate 5561)."""
        assert curator_gate_source('5561') == 'curator_gate_5561', (
            'source key must be the exact curator-written spelling; '
            f'got {curator_gate_source("5561")!r}'
        )

    def test_int_task_id_is_coerced_to_the_same_key(self):
        """An int task id must not silently produce a different key than the str form."""
        assert curator_gate_source(5561) == 'curator_gate_5561', (
            'int task ids must be coerced to str before formatting, else an '
            f'int-typed caller queries a different key; got {curator_gate_source(5561)!r}'
        )
        assert curator_gate_source(5561) == curator_gate_source('5561'), (
            'int and str spellings of the same task id must collapse to one key'
        )

    def test_template_and_helper_are_one_definition(self):
        """CURATOR_GATE_SOURCE_TEMPLATE.format(...) equals the helper's output (INV-5)."""
        assert CURATOR_GATE_SOURCE_TEMPLATE.format(task_id='5561') == curator_gate_source('5561'), (
            'the template and the helper must be one definition, not two copies '
            'that can drift apart'
        )
        assert CURATOR_GATE_SOURCE_TEMPLATE == 'curator_gate_{task_id}', (
            f'template spelling drifted; got {CURATOR_GATE_SOURCE_TEMPLATE!r}'
        )
