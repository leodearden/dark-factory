"""Tests for fused_memory.reconciliation.recon_self_model — the single-source
self-model of recon's control-plane mechanisms (task 2220, W5-β, PRD
plans/recon-reliability-prd.md §8.4, stream W5 foundations phase).

FOUNDATIONS-FIRST: this task builds ONLY this module + these tests. The
prompt cutover (stage1.py/stage2.py importing the rendered sections) and the
premise-lint wiring at the recon submit path are task ξ.

Assertions are pinned to runtime return values (constants, rendered strings,
predicate bools, Violation lists) and stable load-bearing substrings within
rendered sections — NOT verbatim prompt-text equality, which is ξ's exact
drift invariant to own.
"""

from __future__ import annotations

from fused_memory.reconciliation import recon_self_model as m

# --------------------------------------------------------------------------- #
# Static vocabulary constants (step-1/2)
# --------------------------------------------------------------------------- #


class TestVocabularyConstants:
    """MARKER_KINDS / EXECUTION_CLASSES / MCP_CALL_SIGNATURES are the
    single-sourced static vocabulary (PRD §8.1, §8.5)."""

    def test_marker_kinds_is_full_record_kind_vocabulary(self):
        """MARKER_KINDS is the full 5-value §8.1 record_kind vocabulary, as a tuple."""
        assert m.MARKER_KINDS == (
            'stage1_flag_marker',
            'stage1_flag_suppression',
            'stage2_persistence_marker',
            'flag_for_stage2',
            'cycle_summary',
        )

    def test_execution_classes(self):
        """EXECUTION_CLASSES names the three PRD §8.5 execution classes, in order."""
        assert m.EXECUTION_CLASSES == ('code_tdd', 'operational', 'decision')

    def test_mcp_call_signatures_covers_recon_tool_surface(self):
        """MCP_CALL_SIGNATURES is a non-empty mapping covering the recon tool surface."""
        assert isinstance(m.MCP_CALL_SIGNATURES, dict)
        assert m.MCP_CALL_SIGNATURES
        required_keys = {
            'submit_task',
            'resolve_ticket',
            'add_finding',
            'cite_task',
            'add_memory',
            'search',
        }
        assert required_keys <= m.MCP_CALL_SIGNATURES.keys(), (
            f'Missing MCP_CALL_SIGNATURES keys: '
            f'{required_keys - m.MCP_CALL_SIGNATURES.keys()}'
        )
        for key in required_keys:
            sig = m.MCP_CALL_SIGNATURES[key]
            assert isinstance(sig, str) and sig, (
                f'MCP_CALL_SIGNATURES[{key!r}] must be a non-empty str, got {sig!r}'
            )
