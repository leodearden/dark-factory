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
from fused_memory.reconciliation.prompts import STALE_KNOWLEDGE_ANNOTATION_NORM
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt

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


def _section(heading: str) -> str:
    """Return *heading*'s slice of the prompt, up to the next top-level one.

    Slicing keeps every assertion below scoped to the section under test, so a
    token that happens to appear elsewhere in a 60k-char prompt cannot make a
    section-level claim pass vacuously.
    """
    start = STAGE1_SYSTEM_PROMPT.index(heading)
    rest = STAGE1_SYSTEM_PROMPT[start + len(heading) :]
    end = rest.find('\n## ')
    return rest if end == -1 else rest[:end]


class TestStage1ExecutionContract:
    """`## Executing a Cluster Fold` must carry what 3112's gate section omits.

    Task 3112 owns the TARGET END STATE; this section owns HOW the fold is
    executed.  Every token pinned below is one a caller cannot succeed without
    and which the gate section does not supply: the two id arms, the run
    attribution, and the outcome field that decides whether the fold actually
    closed.
    """

    def test_the_section_exists_exactly_once(self) -> None:
        assert '## Executing a Cluster Fold' in STAGE1_SYSTEM_PROMPT
        assert STAGE1_SYSTEM_PROMPT.count('## Executing a Cluster Fold') == 1

    def test_the_load_bearing_call_and_outcome_names_are_present(self) -> None:
        section = _section('## Executing a Cluster Fold')
        for token in ('supersedes', 'retain', 'run_id', 'survivors'):
            assert token in section, token

    def test_the_canonical_is_written_before_any_delete(self) -> None:
        # The ordering directive is the whole reason this rewrite exists: an
        # unordered canonical-write-plus-deletes with no verification is the
        # +1-per-pass ratchet `consolidate_memories` was built to end.
        section = _section('## Executing a Cluster Fold')
        assert 'before' in section
        assert 'delete' in section

    def test_partial_is_stated_as_a_no_resume_outcome(self) -> None:
        # `server/consolidation.py::_PARTIAL_RECOVERY_HINT` says it on the
        # response; the prompt must say it where the caller reads it BEFORE
        # calling, because re-running for the same (project, topic) writes a
        # SECOND canonical rather than resuming.
        section = _section('## Executing a Cluster Fold')
        assert 'partial' in section
        assert 'consolidate_memories' in section

    def test_the_target_end_state_brief_is_not_restated_here(self) -> None:
        # INV-5 no-lockstep-duplication.  `render_end_state_brief` (task 3112)
        # owns that text and renders it into this same assembled prompt; a
        # second normative copy is a contradiction an inference-time reader
        # resolves arbitrarily.  This section CROSS-REFERENCES it instead.
        assert 'TARGET END STATE' not in _section('## Executing a Cluster Fold')

    def test_no_doubled_brace_survives_into_the_rendered_prompt(self) -> None:
        # `STAGE1_SYSTEM_PROMPT` is an f-string, so literal braces in its
        # source must be doubled — and a doubled brace SURVIVING into the
        # rendered text means the source over-escaped, shipping a malformed
        # payload example to the agent silently.  (A single-brace slip fails
        # loudly at import instead, taking the whole stage down, so only the
        # quiet direction needs a pin.)
        assert '{{' not in STAGE1_SYSTEM_PROMPT
        assert '}}' not in STAGE1_SYSTEM_PROMPT


class TestStage1CitationGateAlignment:
    """`## UUID Resolution Discipline` must teach the POST-3624 citation gate.

    The pre-delete citation-repoint gate is a property of the RECORD, not of
    who is deleting.  The section documented `replacement_memory_id` and the
    two CitationReplacement* refusals but never mentioned the one sanctioned
    bypass, so a Stage-1 agent facing a plain drop with no survivor had no
    stated way forward at all.

    Each assertion is pinned on the SAME vocabulary the server-side refusal
    uses (`server/tools.py::_CITATION_REPOINT_REQUIRED_HINT` and
    ``::_IGNORED_DANGLING_OVERRIDE_HINT``), so the prompt and the refusal the
    agent actually receives draw the same boundary by the same names rather
    than growing a second, drifting explanation of one gate.
    """

    def test_the_sanctioned_bypass_is_named_as_the_agent_must_type_it(self) -> None:
        section = _section('## UUID Resolution Discipline')
        assert "metadata={'allow_dangling_citations': True}" in section

    def test_the_gate_is_stated_to_bind_every_caller(self) -> None:
        # Same words FUSED_MEMORY_INSTRUCTIONS already uses: "a property of
        # the RECORD, not of who is deleting, so it applies to every caller".
        assert 'every caller' in _section('## UUID Resolution Discipline')

    def test_the_bypass_is_scoped_to_a_plain_drop(self) -> None:
        # `_CITATION_REPOINT_REQUIRED_HINT`'s exact scoping vocabulary.  An
        # unscoped advertisement would point Stage 1 at the one posture the
        # field evidence names as destructive — reflexive use across ~88
        # consecutive refusals is how genuine third-party citers get stranded.
        assert 'plain drop' in _section('## UUID Resolution Discipline')

    def test_the_literal_true_rule_is_stated(self) -> None:
        # Per `_IGNORED_DANGLING_OVERRIDE_HINT`: a truthy 'yes'/1/'true' is
        # IGNORED and the refusal stands.  Without this the strictness reads
        # as a dead end — the flag appears to have been passed and the same
        # refusal comes back.
        assert 'literal' in _section('## UUID Resolution Discipline')

    def test_the_bypass_does_not_displace_the_survivor_naming_rule(self) -> None:
        # Anti-over-correction.  `replacement_memory_id` remains the correct
        # answer for a consolidation delete, which has a survivor by
        # definition; the escape is the exception, not the replacement.
        assert 'replacement_memory_id' in _section('## UUID Resolution Discipline')


class TestStage1DoesNotCiteTheRetiredExemption:
    """No recon-stage-scoped story may survive in the delete-discipline section.

    The one remaining `recon-stage-*` mention in stage1.py belongs to the
    `ReconMixedFramingWriteRejected` write gate — a still-live, unrelated
    mechanism — and sits in a different section, deliberately outside the
    slice asserted here.
    """

    def test_the_delete_discipline_section_names_no_recon_stage_exemption(self) -> None:
        assert 'recon-stage' not in _section('## UUID Resolution Discipline')


class TestSharedNormNamesTheSanctionedPath:
    """The SHARED stale-knowledge norm must not prescribe the hand-rolled fold.

    ``STALE_KNOWLEDGE_ANNOTATION_NORM`` renders verbatim into BOTH stage
    prompts, so its clause (d) — "amend the SURVIVOR in place ... and only
    THEN delete the redundant siblings" — is a second normative instruction
    for the very choreography ``## Executing a Cluster Fold`` supersedes.  Two
    contradictory instructions inside one assembled prompt is the INV-5
    failure, and an inference-time reader resolves it arbitrarily.

    The norm may name the op because BOTH stages hold it: neither
    ``STAGE1_DISALLOWED`` nor ``STAGE2_DISALLOWED`` folds
    DISALLOW_MEMORY_WRITES, which is the standing precondition for anything
    this constant says.
    """

    def test_the_norm_names_the_consolidation_op(self) -> None:
        assert 'consolidate_memories' in STALE_KNOWLEDGE_ANNOTATION_NORM

    def test_it_still_renders_exactly_once_into_both_stage_prompts(self) -> None:
        # Same exactly-once shape as
        # tests/test_stages.py::test_boundary_note_rendered_verbatim_into_all_three_stage_prompts.
        # A shared constant that renders twice states its rule twice.
        assert STAGE1_SYSTEM_PROMPT.count(STALE_KNOWLEDGE_ANNOTATION_NORM) == 1
        assert build_stage2_system_prompt('dark_factory').count(STALE_KNOWLEDGE_ANNOTATION_NORM) == 1

    def test_it_keeps_the_must_nots_its_own_comment_block_demands(self) -> None:
        # Each of these is a live wiring constraint on this constant, not
        # style: build_stage2_system_prompt RAISES unless '## Available Tools'
        # appears exactly once in STAGE2_SYSTEM_PROMPT, and
        # test_recon_report_guidance_drift.py requires every
        # `mcp__recon-report__` example in an assembled prompt to carry
        # `run_id=`.
        assert '## Available Tools' not in STALE_KNOWLEDGE_ANNOTATION_NORM
        assert 'mcp__recon-report__' not in STALE_KNOWLEDGE_ANNOTATION_NORM

    def test_it_does_not_cross_reference_a_stage1_only_heading(self) -> None:
        # `## Executing a Cluster Fold` exists in stage1.py only; stage2.py
        # has no such section, so the SHARED norm must refer to the op by
        # tool name — a fact true in both stages — never by that heading.
        # Same rule the constant's comment block already states for
        # "## UUID Resolution Discipline".
        assert '## Executing a Cluster Fold' not in STALE_KNOWLEDGE_ANNOTATION_NORM

    def test_it_keeps_the_survivor_naming_rule(self) -> None:
        # Anti-over-correction: `replacement_memory_id` stays right for the
        # SINGLE-record supersede case the norm still covers, and for
        # hand-finishing a `partial` consolidation.
        assert 'replacement_memory_id' in STALE_KNOWLEDGE_ANNOTATION_NORM
