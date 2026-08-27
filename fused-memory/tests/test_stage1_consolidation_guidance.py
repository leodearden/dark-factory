"""Wiring contracts for Stage 1's cluster-fold guidance (task 3134, PRD leaf iota).

These tests pin the WIRING — that an op the stage actually HOLDS is advertised
to it, that the load-bearing parameter and outcome names a caller cannot
succeed without are present, that a shared constant renders exactly once — and
NOT the prose.  Same convention as
``tests/test_recon_gate_closure_guidance.py``, whose module docstring states
it: wording may be reworded freely; the wiring may not silently break.

Pinning a prompt IS behaviour testing here, not documentation testing.  A
reconciliation stage is an LLM agent whose only runtime artifact is its system
prompt: the prompt is the code path.  A stage told to hand-roll a
write-then-delete choreography executes that choreography, and the +1-per-pass
consolidation ratchet ``consolidate_memories`` exists to end comes straight
back.

Division of labour with task 3112 (``## Consolidation Gate``, rendered from
``fused_memory.reconciliation.consolidation_gate``): 3112 owns the TARGET END
STATE (N short single-claim peers sharing ``metadata.topic``, exactly one
``canonical: true``, ``supersedes`` naming only genuinely-deleted ids).  This
task owns HOW the fold is EXECUTED — the ordering, ``run_id``, ``survivors``,
the no-resume rule, and the two escape flags.  The end-state brief is
CROSS-REFERENCED, never restated: a second normative copy inside one assembled
prompt is the INV-5 no-lockstep-duplication failure, and
``TestStage1ExecutionContract`` carries the negative assertion that pins the
boundary.
"""

from __future__ import annotations

from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT

# The MCP-prefixed tool id, as an agent must actually type it.  The BARE name
# `consolidate_memories` already appears in the prompt (inside task 3112's
# `## Consolidation Gate` end-state brief), so asserting on the bare name would
# pass without the advertisement this task adds.
_CONSOLIDATE_TOOL_ID = 'mcp__fused-memory__consolidate_memories'


class TestStage1AdvertisesTheConsolidationOp:
    """`consolidate_memories` must be LISTED in Stage 1's tool block.

    Stage 1 holds the op — ``STAGE1_DISALLOWED`` folds DISALLOW_TASK_WRITES,
    DISALLOW_RECON_REPORT_LEDGER_WRITES, DISALLOW_ESCALATION_READS and
    DISALLOW_BUILTIN, but never DISALLOW_MEMORY_WRITES where the op sits — yet
    ``## Available Tools`` did not name it.  ``--disallowed-tools`` OMITS a
    denied tool from the agent's listing rather than rejecting the call, so a
    held-but-unadvertised op is indistinguishable, from inside the stage, from
    one it does not have.
    """

    def test_the_mcp_prefixed_tool_id_is_present(self) -> None:
        assert _CONSOLIDATE_TOOL_ID in STAGE1_SYSTEM_PROMPT

    def test_it_is_advertised_in_the_tool_block_not_mentioned_later(self) -> None:
        # `## Available Tools` opens the prompt's tool block and
        # `## Your Consolidation Tasks` is the next top-level section after
        # it, so a first occurrence before that heading means the op was
        # ADVERTISED rather than mentioned incidentally further down.
        assert STAGE1_SYSTEM_PROMPT.index(_CONSOLIDATE_TOOL_ID) < STAGE1_SYSTEM_PROMPT.index(
            '## Your Consolidation Tasks'
        )

    def test_the_advertised_op_is_one_the_stage_actually_holds(self) -> None:
        # The anti-drift companion: a stage must never advertise a tool its
        # own disallow list denies.  Passes today; it exists so that a future
        # change to STAGE1_DISALLOWED cannot silently turn the advertisement
        # above into a lie.  cli_stage_runner.py's own comment records the
        # split: the safety classification belongs with the tool, "Stage 1's
        # ADVERTISEMENT of the op is task 3134's".
        assert _CONSOLIDATE_TOOL_ID not in STAGE1_DISALLOWED
