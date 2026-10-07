"""Wiring contracts for Stage 1's ``## Preserved-Specimen Corroboration`` directive.

The directive tells Stage 1 to read a task's recorded not-actionable
``investigation_outcome`` verdicts before calling it stranded.  These tests pin
that the reader it names is one the stage is TOLD it has and actually HOLDS,
and that the filter it asks for is the one the code-side guard
(``preservation_specimen_guard.filter_preservation_specimen_flags``) runs.

Pinning a prompt is behaviour testing here, not documentation testing: a
reconciliation stage is an LLM agent whose system prompt is its code path (the
convention ``tests/test_stage1_consolidation_guidance.py`` states).  So these
pin WIRING, never prose.
"""

from __future__ import annotations

from _recon_prompt_write_scope import tool_listing

from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
from fused_memory.reconciliation.preservation_specimen_guard import preservation_mem0_filters
from fused_memory.reconciliation.prompts.stage1 import (
    PRESERVED_SPECIMEN_CORROBORATION_HEADING,
    STAGE1_SYSTEM_PROMPT,
)

# MCP-prefixed, as an agent must type it.  The bare name `get_memories_by_metadata`
# already appears in the reconciliation prompts, so a bare-name pin would pass
# without the advertisement this directive depends on.
_READER = 'mcp__fused-memory__get_memories_by_metadata'


def _section() -> str:
    """The directive's slice of the prompt, up to the next top-level heading."""
    start = STAGE1_SYSTEM_PROMPT.index(PRESERVED_SPECIMEN_CORROBORATION_HEADING)
    end = STAGE1_SYSTEM_PROMPT.find(
        '\n## ', start + len(PRESERVED_SPECIMEN_CORROBORATION_HEADING),
    )
    return STAGE1_SYSTEM_PROMPT[start:] if end == -1 else STAGE1_SYSTEM_PROMPT[start:end]


def test_heading_occurs_exactly_once() -> None:
    assert STAGE1_SYSTEM_PROMPT.count(PRESERVED_SPECIMEN_CORROBORATION_HEADING) == 1


def test_reader_is_advertised_in_the_tool_block() -> None:
    assert _READER in tool_listing(STAGE1_SYSTEM_PROMPT)


def test_advertised_reader_is_one_stage1_holds() -> None:
    # The anti-drift companion: --disallowed-tools OMITS a denied tool, so a
    # reader that is advertised but denied would make the directive a lie.
    assert _READER not in STAGE1_DISALLOWED


def test_section_names_the_reader() -> None:
    assert _READER in _section()


def test_section_renders_the_guards_filter() -> None:
    assert repr(preservation_mem0_filters('<id>')) in _section()
