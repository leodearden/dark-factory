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
prompt is the INV-5 no-lockstep-duplication failure.  That boundary is
carried by the PROMPT's own opening cross-reference ("The section above states
WHAT a folded cluster must look like when you are done. This one states HOW to
get there."), not by a test here: a negative substring pin over the section
(``'TARGET END STATE' not in ...``) was removed because a restatement that
reworded the heading passes it, so it could never detect the duplication it
named.
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
        # `canonical_content` and `topic` are REQUIRED parameters of the
        # shipped op — a caller cannot succeed without naming them — and the
        # rest are the id arms, the run attribution and the outcome field that
        # decides whether the fold actually closed.
        for token in ('canonical_content', 'topic', 'supersedes', 'retain', 'run_id', 'survivors'):
            assert token in section, token
        # Positional pin: the ADVERTISED CALL names the canonical arm first.
        # This pins the advertised signature's argument ORDER and explicitly
        # NOT the runtime write-before-delete property — that one is not a
        # prompt-side property at all.  It is owned by
        # `server/tools.py::consolidate_memories` step (4) and executably
        # covered by tests/test_consolidate_memories_tool.py
        # (::test_a_repeated_supersede_is_refused_before_any_write,
        # ::test_delete_arm_without_run_id_is_refused_before_the_canonical,
        # ::test_unauthorized_agent_is_denied_before_the_canonical_write,
        # ::test_citations_are_repointed_before_any_delete_lands and
        # ::test_each_victim_is_read_before_its_delete).
        assert 'consolidate_memories(canonical_content=' in section

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

    Exactly two things are pinned here, and both are wire-typed: the payload
    literal an agent must type verbatim for the server to honour it
    (``metadata={'allow_dangling_citations': True}``, the spelling
    `server/tools.py::_CITATION_REPOINT_REQUIRED_HINT` advertises and
    ``::_IGNORED_DANGLING_OVERRIDE_HINT`` enforces), and
    ``replacement_memory_id``.

    The section's PROSE — that the gate binds every caller, that the bypass is
    scoped to a plain drop, that only a literal ``True`` counts — is
    deliberately left UNPINNED so it stays freely rewordable.  A substring pin
    over a phrase occurring once in a ~4.8k-char slice goes red on a rewrite
    that changes nothing an agent can act on, which is the drift this file's
    stated convention exists to avoid.
    """

    def test_the_sanctioned_bypass_is_named_as_the_agent_must_type_it(self) -> None:
        section = _section('## UUID Resolution Discipline')
        assert "metadata={'allow_dangling_citations': True}" in section

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


class TestStage1IsToldTheGuardNowAppliesToIt:
    """Stage 1 must be told the near-duplicate guards now bind it too.

    Retiring the `recon-stage-*` exemption at both sites (the `add_memory`
    reject guard and the write-triage force-store arm) changes what Stage 1
    EXPERIENCES at write time, and the system prompt is the stage's only
    channel for that.  Left unstated, a soft-block is illegible: the agent
    sees a write it has always been allowed to make suddenly refused, and the
    refusal itself advertises `allow_near_duplicate` as the way through — so
    the reflex is to set the override and carry on, re-creating the very
    +1-per-pass ratchet this task closes.

    The pins are on the DIRECTIVE as much as the mechanism: naming the flag is
    not enough if the prompt does not, in the same breath, route a soft-block
    to `consolidate_memories` instead of to the override.
    """

    def test_the_override_flag_is_named_in_the_fold_section(self) -> None:
        # It must be named where the fold is executed, not left to be
        # discovered from a refusal envelope — by then the agent is choosing
        # between "set the flag" and "give up", with no third option stated.
        assert 'allow_near_duplicate' in _section('## Executing a Cluster Fold')

    def test_the_soft_block_is_named_by_its_wire_identity(self) -> None:
        # `server/near_duplicate_guard.py` emits this exact `error_type`.
        # Naming it lets the agent recognise the response it actually
        # receives, rather than pattern-matching an English paraphrase.
        assert (
            'ProceduralKnowledgeNearDuplicateWriteRejected'
            in _section('## Executing a Cluster Fold')
        )

    def test_a_soft_block_is_routed_to_the_op_not_to_the_override(self) -> None:
        # The load-bearing assertion: both tokens in the SAME slice, so the
        # prompt cannot name the escape hatch in one place and the sanctioned
        # path in another and leave the agent to connect them.  A soft-block
        # means the cluster already exists — which is a FOLD signal.
        section = _section('## Executing a Cluster Fold')
        assert 'allow_near_duplicate' in section
        assert 'consolidate_memories' in section

    def test_the_override_is_never_advertised_outside_this_section(self) -> None:
        # Whole-prompt companion.  The flag may be shown at most once, and
        # only inside the section that also states when NOT to use it, so the
        # prompt can never come to instruct the override unconditionally
        # somewhere else.  (Zero occurrences is fine — this bounds where it
        # may appear, it does not require the payload example to exist.)
        literal = "metadata={'allow_near_duplicate': True}"
        assert STAGE1_SYSTEM_PROMPT.count(literal) <= 1
        if literal in STAGE1_SYSTEM_PROMPT:
            assert literal in _section('## Executing a Cluster Fold')
