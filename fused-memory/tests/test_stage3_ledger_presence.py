"""Tests for τ2 (task 2437): Stage 3 consults the ReconLedgerStore
authoritatively for cycle-summary presence, via τ1's (task 2436)
``get_cycle_summary_presence`` tool, falling back to the existing best-effort
Mem0 two-path check only when the ledger read is inconclusive.

Two cohesive pieces of the same G2 signal live here (PRD
plans/stage3-ledger-presence-prd.md §12):

- ``TestStage3PromptLedgerAuthoritative`` — a prompt-content contract test.
  The Stage-3 agent's in-loop reasoning is not unit-testable; the LLM prompt
  string IS the observable behaviour surface (the agent cannot call a tool or
  apply a rule it is not told about), so this asserts the new tool is named
  and the ledger-authoritative rule markers are present, plus a closure gate
  that the retired "Known gap" comment is actually gone from the module
  source. Mirrors test_stages.py::TestStage3PromptAlignment.
- ``TestWriteThenReadLedgerSeam`` (added separately) — a write→read
  boundary/seam test proving the mechanical path the rewritten prompt now
  trusts is real, not faked.
"""

from __future__ import annotations

import inspect

import fused_memory.reconciliation.prompts.stage3 as stage3_module
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT


class TestStage3PromptLedgerAuthoritative:
    """STAGE3_SYSTEM_PROMPT names the new ledger-presence tool and carries
    the ledger-authoritative rule; the retired "Known gap" comment is gone
    from the module source."""

    def test_prompt_names_get_cycle_summary_presence_tool(self):
        assert 'get_cycle_summary_presence' in STAGE3_SYSTEM_PROMPT, (
            "STAGE3_SYSTEM_PROMPT must name the get_cycle_summary_presence tool"
        )

    def test_prompt_carries_ledger_authoritative_rule_markers(self):
        assert 'ledger_available' in STAGE3_SYSTEM_PROMPT, (
            "STAGE3_SYSTEM_PROMPT must reference the 'ledger_available' return field"
        )
        assert 'authoritative' in STAGE3_SYSTEM_PROMPT, (
            "STAGE3_SYSTEM_PROMPT must describe the ledger read as authoritative"
        )

    def test_known_gap_comment_removed_from_module_source(self):
        assert 'Known gap' not in inspect.getsource(stage3_module), (
            'The retired "Known gap" comment must be deleted from stage3.py, '
            'not merely absent from the prompt string'
        )
