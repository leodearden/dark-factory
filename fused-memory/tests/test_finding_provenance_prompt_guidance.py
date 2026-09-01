"""Contract tests for the canonical finding-provenance metadata keys in the
recon stage prompts (esc-3796-1, task 4373).

The recon prompts mandate CAPTURING a Stage-1 `finding_id` but never said
WHERE to persist it on a task filed from that finding. Neither spelling has a
code writer, so the prompt is the only possible writer — and with no name
stated, the corpus forked into 64 bespoke spellings. esc-3796-1 ruled that
naming the canonical keys in the prompt is the only drift-stopper available.

WHY THIS FILE PINS IDENTIFIERS WHEN ITS SIBLING REFUSES TO PIN PROSE
--------------------------------------------------------------------
`test_duplicate_finding_salvage_guidance.py` records at length why a prose
substring pin inside a prompt string literal is a documentation meta-test, not
a behavioural contract: it is wrong in both directions — a faithful reword that
names the same surface differently goes red on a correct edit, while a garbled
paragraph that happens to retain the tokens goes green.

That reasoning governs SENTENCES. It does not govern `source_finding_id` and
`related_memory_ids`, which are not wording but the INTERFACE an LLM writes to:
they are the literal keys that land in `task.metadata`, and they are consumed
downstream by the `unknown_key` census grep. A prompt that names a different
spelling does not read differently — it writes a 65th bespoke key, which is
precisely the fork esc-3796-1 ruled on. So the identifier is a behavioural
contract while the surrounding sentence is not, and accordingly **no assertion
in this file pins any sentence**.

The identifiers are single-sourced from the module constants and imported here
rather than retyped, so a rename at the source cannot leave this test silently
agreeing with itself. The cross-check against `parse_metadata` is what catches
a rename to a spelling that is not blessed.

TWO VOCABULARIES, NOT ONE
-------------------------
The keys above are TASK METADATA keys, cross-checked against `parse_metadata`.
The relay hop that carries a finding from Stage 1 to Stage 2 uses a DIFFERENT
vocabulary — the field names of the flagged item that
`ReconReportState.get_assembled_report` projects — and it gets its own
cross-check, an in-process round-trip through that projection. Collapsing the
two is not hypothetical: the first shipped Stage-1 clause told Stage 1 to carry
the METADATA key names as flagged-item fields, on a projection that has no
passthrough, which made the whole Stage-1 half a silent no-op. Both
cross-checks exist so that neither vocabulary can be asserted against nothing
but itself.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from shared.task_metadata import parse_metadata

from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_TASK_WRITES,
    STAGE1_DISALLOWED,
)
from fused_memory.reconciliation.prompts import (
    FINDING_ID_METADATA_KEY,
    FINDING_MEMORY_IDS_METADATA_KEY,
    FINDING_PROVENANCE_VOCABULARY_RULE,
    FLAGGED_ITEM_CITED_MEMORIES_FIELD,
    FLAGGED_ITEM_FINDING_ID_FIELD,
    get_recon_report_tool_guidance,
    render_finding_provenance_section,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.server.recon_report import ReconReportState

_STAGE1_SECTION = render_finding_provenance_section(can_file_tasks=False)
_STAGE2_SECTION = render_finding_provenance_section(can_file_tasks=True)

# The two constants, paired with the spelling esc-3796-1 ratified. This is the
# ONE place in this file the literal strings are written down — everything
# below derives from the imported constants.
_RATIFIED_SPELLINGS = [
    (FINDING_ID_METADATA_KEY, 'source_finding_id'),
    (FINDING_MEMORY_IDS_METADATA_KEY, 'related_memory_ids'),
]

_KEY_IDS = ['finding_id_key', 'memory_ids_key']

# A representative value per key, so the parse_metadata cross-check exercises
# each key's real shape (a scalar id vs. a list of memory ids) rather than
# coercing both into one.
_KEY_SAMPLE_VALUES = [
    (FINDING_ID_METADATA_KEY, 'f1'),
    (FINDING_MEMORY_IDS_METADATA_KEY, ['90bd6ecf']),
]

# Each assembled prompt paired with the per-stage provenance section it is
# expected to carry (Stage 3 carries none — it folds DISALLOW_TASK_WRITES and
# gets only the shared block). Pairing them here is what lets the occurrence
# assertions below DERIVE their expected counts from the single-sourced
# renderers instead of hard-coding a magic number that a legitimate reword
# would have to chase.
_ASSEMBLED_PROMPTS = [
    ('STAGE1_SYSTEM_PROMPT', STAGE1_SYSTEM_PROMPT, _STAGE1_SECTION),
    ('STAGE2_SYSTEM_PROMPT', STAGE2_SYSTEM_PROMPT, _STAGE2_SECTION),
    ('STAGE3_SYSTEM_PROMPT', STAGE3_SYSTEM_PROMPT, ''),
]

# Explicit ids are required, not cosmetic: pytest otherwise derives each id from
# the parametrized value, embedding a whole multi-KB system prompt in every id.
_PROMPT_IDS = ['stage1', 'stage2', 'stage3']


class TestCanonicalKeyConstants:
    """The constants name the spellings esc-3796-1 ratified, and only those."""

    @pytest.mark.parametrize('constant, ratified', _RATIFIED_SPELLINGS, ids=_KEY_IDS)
    def test_constant_equals_ratified_spelling(self, constant: str, ratified: str):
        """The ratification, written down once.

        Renaming a constant is a legitimate refactor; renaming the STRING it
        holds is a vocabulary change that forks the corpus, and esc-3796-1
        reserved that decision to itself.
        """
        assert constant == ratified, (
            f'esc-3796-1 ratified {ratified!r} as the canonical spelling; the prompt '
            f'constant holds {constant!r}. Changing the spelling forks the corpus — '
            'it needs a new ruling, not an edit here.'
        )

    @pytest.mark.parametrize('key, sample_value', _KEY_SAMPLE_VALUES, ids=_KEY_IDS)
    def test_canonical_key_is_blessed_metadata(self, key: str, sample_value):
        """The cross-module drift-stopper, and this task's ACCEPTANCE in executable form.

        A prompt that instructs an LLM to write an UNBLESSED key manufactures
        exactly the `unknown_key` census noise esc-3796-1 exists to eliminate —
        it would be actively self-defeating. This binds the two sides together,
        so it goes red both if the prompt adopts a wrong spelling and if the
        allowlist in `shared/src/shared/task_metadata.py` drops one.

        This is the assertion that makes the pair a contract rather than two
        independent conventions that happen to agree today.
        """
        _, warnings = parse_metadata({key: sample_value}, direction='read')
        offending = [w for w in warnings if w.code == 'unknown_key' and w.field == key]
        assert offending == [], (
            f'The prompts name {key!r} as a canonical metadata key, but it is not '
            f'blessed in _BLESSED_METADATA_KEYS — every task filed from a finding '
            f'would emit a fresh unknown_key census line. Got: {offending}'
        )


class TestSharedGuidanceBlockNamesBothKeys:
    """The one shared block names both keys, so all three stages agree.

    `_render_recon_report_tool_guidance()` is interpolated by stage1, stage2 and
    stage3 alike, and feeds both the live-introspection path and the frozen
    fallback. Putting the names there makes "the three stages agree" a
    STRUCTURAL fact rather than a convention maintained by hand across three
    copies (INV-5 `no-lockstep-duplication`).
    """

    @pytest.mark.parametrize('key, _ratified', _RATIFIED_SPELLINGS, ids=_KEY_IDS)
    def test_guidance_names_each_key_exactly_once(self, key: str, _ratified: str):
        """`.count(...) == 1`, not a bare `in`.

        Catches omission AND double-interpolation in one assertion, and
        subsumes non-emptiness — following the rationale recorded in
        `test_duplicate_finding_salvage_guidance.py`.
        """
        guidance = get_recon_report_tool_guidance()
        assert guidance.count(key) == 1, (
            f'The shared recon-report guidance must name {key!r} exactly once; '
            f'found {guidance.count(key)} occurrences.'
        )

    @pytest.mark.parametrize('name, prompt, section', _ASSEMBLED_PROMPTS, ids=_PROMPT_IDS)
    def test_every_stage_prompt_names_both_keys(self, name: str, prompt: str, section: str):
        """All three assembled prompts name both keys, and ONLY via a renderer.

        Stage 3 is included deliberately. It is read-only and cannot file
        tasks, but it renders the same shared block, and "all three stages
        agree on the vocabulary" is the property being pinned — agreement about
        what the keys ARE is separable from the license to write them.

        The count is checked, not just membership, and the expected value is
        DERIVED from the two single-sourced renderers rather than hard-coded.
        So a stage that grows its own hand-typed mention of a key goes red
        (INV-5 `no-lockstep-duplication`) while a faithful reword inside either
        renderer stays green — the count moves on both sides at once.
        """
        shared = get_recon_report_tool_guidance()
        for key, _ratified in _RATIFIED_SPELLINGS:
            assert key in prompt, (
                f'{name} must name the canonical provenance key {key!r}; a stage left '
                'without the name is a stage free to invent a spelling.'
            )
            expected = shared.count(key) + section.count(key)
            assert prompt.count(key) == expected, (
                f'{name} mentions {key!r} {prompt.count(key)}x, but the single-sourced '
                f'renderers account for only {expected}x. Every mention must come from '
                'get_recon_report_tool_guidance() or render_finding_provenance_section() '
                '— a per-stage copy is the lockstep duplication this section exists to '
                'avoid.'
            )

    @pytest.mark.parametrize('name, prompt, section', _ASSEMBLED_PROMPTS, ids=_PROMPT_IDS)
    def test_the_vocabulary_rule_is_stated_at_most_once(
        self, name: str, prompt: str, section: str
    ):
        """The negative rule reaches each stage that can write a key, exactly once.

        It was briefly stated twice per prompt — once in the shared
        recon-report block and once in the provenance section — which is the
        lockstep duplication INV-5 names: a maintainer rewording one copy would
        not know about the other. This pins the interpolation TOPOLOGY (a
        constant counted, never a sentence retyped), following
        `test_stage2_carries_the_task_filing_branch_exactly_once` below.

        Stage 3 expects zero: it folds DISALLOW_TASK_WRITES, so it cannot mint
        a variant key, and this package does not tell a stage about an action
        it is not sanctioned to take.
        """
        expected = 1 if section else 0
        assert prompt.count(FINDING_PROVENANCE_VOCABULARY_RULE) == expected, (
            f'{name} states FINDING_PROVENANCE_VOCABULARY_RULE '
            f'{prompt.count(FINDING_PROVENANCE_VOCABULARY_RULE)}x; expected {expected}x. '
            'A second surface may interpolate the constant, but then this expectation '
            'is the deliberate decision to move — never a re-typed copy.'
        )


class TestCapabilitySplitOfTheActionableRule:
    """The actionable rule is parameterized on what the stage may actually do.

    Stage 1 cannot file tasks: `cli_stage_runner.STAGE1_DISALLOWED` folds in
    `DISALLOW_TASK_WRITES`, which contains `mcp__fused-memory__submit_task`,
    while `STAGE2_DISALLOWED` does not. Naming a tool in a stage's prompt is a
    live positive license, not inert prose, so handing Stage 1 the Stage-2 text
    would instruct it to take an action it cannot take — which this package
    forbids in two separate documented places
    (`render_escalation_boundary_note`: "Never tell a stage about an action it
    is not sanctioned to take"; `render_source_completion_section`: "Never
    instruct Stage 1 to call a tool it does not hold").

    These assertions pin the SPLIT, not the wording of either branch.
    """

    def test_stage2_carries_the_task_filing_branch_exactly_once(self):
        assert STAGE2_SYSTEM_PROMPT.count(_STAGE2_SECTION) == 1, (
            'STAGE2_SYSTEM_PROMPT must interpolate '
            'render_finding_provenance_section(can_file_tasks=True) exactly once.'
        )

    def test_stage1_carries_the_relay_branch_exactly_once(self):
        assert STAGE1_SYSTEM_PROMPT.count(_STAGE1_SECTION) == 1, (
            'STAGE1_SYSTEM_PROMPT must interpolate '
            'render_finding_provenance_section(can_file_tasks=False) exactly once.'
        )

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_survives_build_stage2_system_prompt(self, project_id: str):
        """Both branches of the runtime builder keep the section.

        `autopilot_video` injects an extra guardrail section before
        `## Available Tools`; that injection must not displace this one.
        """
        built = build_stage2_system_prompt(project_id)
        assert built.count(_STAGE2_SECTION) == 1, (
            f'build_stage2_system_prompt({project_id!r}) must carry the '
            'can_file_tasks=True section exactly once.'
        )

    def test_the_two_renderings_differ(self):
        """A split that renders identically is not a split."""
        assert _STAGE1_SECTION != _STAGE2_SECTION, (
            'The two capability branches render identically — the parameterization '
            'is then decorative, and Stage 1 is being told whatever Stage 2 is told.'
        )

    def test_neither_stage_carries_the_other_branch(self):
        """What makes the split structural rather than incidental.

        A copy-paste that hands Stage 1 the Stage-2 text goes red here even
        though both stages would still "have a provenance section".
        """
        assert _STAGE2_SECTION not in STAGE1_SYSTEM_PROMPT, (
            'STAGE1_SYSTEM_PROMPT carries the can_file_tasks=True rendering — that '
            'licenses Stage 1 to file a task it holds no tool for.'
        )
        assert _STAGE1_SECTION not in STAGE2_SYSTEM_PROMPT, (
            'STAGE2_SYSTEM_PROMPT carries the can_file_tasks=False rendering — that '
            'tells Stage 2 to relay work it can do itself, right now.'
        )

    def test_only_the_task_filing_branch_names_submit_task(self):
        """THE capability contract, and the reason the split exists at all.

        The two halves differ in kind. The POSITIVE half is the contract: Stage
        2 must say WHERE the keys get set, which requires naming the tool it
        holds. The NEGATIVE half is a style choice this section makes and
        `render_source_completion_section` does not — see its failure message.

        The tool name is derived from `DISALLOW_TASK_WRITES` at its source
        rather than hard-coded, so this tracks the disallow list as it changes.
        Mirroring an inventory instead of reading it at the source is the exact
        failure mode that hid the Stage-2 escalation-read gap until task 3163,
        recorded in `render_source_completion_section`'s docstring.
        """
        qualified = 'mcp__fused-memory__submit_task'
        assert qualified in DISALLOW_TASK_WRITES, (
            f'{qualified} is no longer in DISALLOW_TASK_WRITES — the premise of this '
            'capability split has changed, so the split itself needs re-deciding, '
            'not this assertion updating.'
        )
        bare = qualified.rsplit('__', 1)[-1]

        assert bare not in _STAGE1_SECTION, (
            f'The can_file_tasks=False rendering names {bare!r}. Stage 1 does not hold '
            'that tool (DISALLOW_TASK_WRITES is folded into STAGE1_DISALLOWED), and '
            'this section deliberately withholds the token even in a NEGATION — a name '
            'surfaced to a model is still a name surfaced. That is a considered style '
            'choice, recorded as a DIVERGENCE in render_finding_provenance_section\'s '
            'docstring, not a rule about what Stage 1 may be told: '
            'render_source_completion_section makes the opposite call for the same '
            'tool. Harmonising the two is a legitimate edit — but decide it there and '
            'update this expectation with it, rather than reading this failure as a '
            'capability violation.'
        )
        assert bare in _STAGE2_SECTION, (
            f'The can_file_tasks=True rendering must name {bare!r} — Stage 2 holds it, '
            'and the whole point is to say WHERE the keys get set.'
        )

    @pytest.mark.parametrize('key, _ratified', _RATIFIED_SPELLINGS, ids=_KEY_IDS)
    def test_both_renderings_name_both_keys(self, key: str, _ratified: str):
        """Neither stage is left to invent a spelling.

        Stage 1 cannot persist the keys itself, but it must still name them
        when relaying to Stage 2 — otherwise the relay hands over a value with
        no agreed field to put it in, which is the original defect.
        """
        assert key in _STAGE1_SECTION, (
            f'The can_file_tasks=False rendering must name {key!r} so the relay to '
            'Stage 2 carries the agreed field name.'
        )
        assert key in _STAGE2_SECTION, (
            f'The can_file_tasks=True rendering must name {key!r} — it is the key '
            'Stage 2 is being told to set.'
        )


# The relay-channel field names, paired with the spelling
# `ReconReportState.get_assembled_report` actually projects. Same discipline as
# `_RATIFIED_SPELLINGS` above: the ONE place in this file these literals are
# written down.
_RELAY_FIELD_SPELLINGS = [
    (FLAGGED_ITEM_FINDING_ID_FIELD, 'finding_id'),
    (FLAGGED_ITEM_CITED_MEMORIES_FIELD, 'cited_memories'),
]

_RELAY_FIELD_IDS = ['flagged_finding_id', 'flagged_cited_memories']


def _names_token(text: str, token: str) -> bool:
    """True when *text* names *token* as a STANDALONE identifier.

    A bare ``in`` is worthless for ``finding_id``: it is a substring of
    ``source_finding_id``, so the shared provenance body would satisfy it
    without the relay clause existing at all. The boundary form is borrowed
    from `test_recon_report_guidance_drift.py`, which guards its tool tokens
    the same way and for the same reason.
    """
    return re.search(rf'(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])', text) is not None


def _state_with_finding() -> tuple[ReconReportState, str, str]:
    """(state, run_id, stage) — one started report holding one actionable finding.

    Construction mirrors `tests/server/test_recon_report_hook_b.py`'s helper of
    the same name, minus its `memory_service`: nothing here cites anything, and
    `cited_memories` is projected unconditionally as `list(f.cited_memories)`,
    so the FIELD is present on a finding with zero citations. That is precisely
    the property under test — the field's existence, not its contents.
    """
    state = ReconReportState(ttl_seconds=300, clock=lambda: 0.0)
    run_id, stage = 'run-1', 'reconciler'
    state.start_report(run_id=run_id, stage=stage, project_id='dark_factory')
    state.add_finding(
        run_id=run_id,
        severity='moderate',
        category='systemic_pattern',
        description='d',
        suggested_action='a',
        actionable=True,
        task_id='42',
        flag_type='orphaned_knowledge',
    )
    return state, run_id, stage


class TestRelayChannelFieldNames:
    """The Stage-1 → Stage-2 hop names fields that survive report assembly.

    This class exists because the first shipped `can_file_tasks=False` clause
    did NOT. It told Stage 1 to carry the finding id and memory ids "in the
    `flagged_items` entry under those same two field names", justified by
    "`flagged_items` entries are free-form objects" — and that justification is
    false on the production path, which `test_the_projection_is_not_a_passthrough`
    below now pins as an executable fact. `stages/base.py` overwrites
    `stage_result.report` with `ReconReportState.get_assembled_report(...)`
    whenever `recon_report_state` is wired (production always wires it), and
    that projection builds each flagged item as a fixed field-by-field dict. So
    every assertion the old clause could have satisfied was a prose assertion,
    and the whole Stage-1 half was a silent no-op.

    The structural guard that was missing is `test_both_relay_fields_are_real_projection_keys`:
    an in-process round-trip that goes red if either field name the prompt
    directs a stage to use stops being a key the assembler emits.
    """

    @pytest.mark.parametrize(
        'constant, projected', _RELAY_FIELD_SPELLINGS, ids=_RELAY_FIELD_IDS
    )
    def test_constant_equals_projected_field_name(self, constant: str, projected: str):
        """The relay vocabulary, written down once.

        Kept separate from `_RATIFIED_SPELLINGS` deliberately: these are
        flagged-item FIELD names, not task-metadata KEYS. The two hops carry
        the same values under different names, and conflating them is the
        defect this class was added to close.
        """
        assert constant == projected, (
            f'The prompt names {constant!r} as a flagged-item field, but '
            f'get_assembled_report projects it as {projected!r}.'
        )

    @pytest.mark.parametrize(
        'constant, _projected', _RELAY_FIELD_SPELLINGS, ids=_RELAY_FIELD_IDS
    )
    def test_both_relay_fields_are_real_projection_keys(self, constant: str, _projected: str):
        """THE drift-stopper — the assertion that would have caught the defect.

        A real in-process round-trip: start a report, add a finding, assemble.
        A prompt that directs a stage at a field the assembler does not emit is
        a no-op instruction, and no prose pin can tell the difference. This can.
        """
        state, run_id, stage = _state_with_finding()
        report = state.get_assembled_report(run_id, stage)
        assert report is not None, 'get_assembled_report returned None for a started run'
        item = report['flagged_items'][0]
        assert constant in item, (
            f'The prompts direct a stage to use the flagged-item field {constant!r}, '
            f'but get_assembled_report projects only {sorted(item)}. An instruction '
            'naming a field the assembler drops is a silent no-op.'
        )

    def test_the_projection_is_not_a_passthrough(self):
        """The NEGATIVE half: there is no mechanism to attach an extra field.

        `add_finding` takes no `**kwargs` at either layer, and the assembler
        builds each flagged item as a fixed field-by-field dict. Pinning that
        behaviourally is what stops a future author re-deriving the refuted
        free-form-relay design from this suite — the two assertions above say
        which fields ARE carried; this one says no others can be.

        Deliberately NOT an assertion on the full projected key SET: that goes
        red on any unrelated field addition, which is a legitimate change.
        """
        state, run_id, _stage = _state_with_finding()
        # Bound as dict[str, Any] rather than splatted inline: a bare
        # `**{CONST: 'f1'}` literal infers dict[str, str], which pyright then
        # checks against EVERY unfilled parameter of add_finding — reporting a
        # spurious reportArgumentType against `actionable: bool | None`. The
        # runtime behaviour under test (add_finding takes no **kwargs, so the
        # call raises TypeError) is identical either way.
        extra_kwargs: dict[str, Any] = {FINDING_ID_METADATA_KEY: 'f1'}
        with pytest.raises(TypeError):
            state.add_finding(
                run_id=run_id,
                severity='moderate',
                category='systemic_pattern',
                description='d2',
                suggested_action='a',
                **extra_kwargs,
            )

    def test_stage1_holds_the_citation_tool_its_clause_names(self):
        """Tool licensing for the corrected Stage-1 channel.

        Memory ids reach Stage 2 ONLY via `cite_memory` — the relay clause is
        inert without it — so naming it is load-bearing, and naming a tool
        Stage 1 does not hold would be the licensing violation this package
        forbids. Membership is checked against the imported `STAGE1_DISALLOWED`
        rather than a hard-coded mirror, matching
        `test_only_the_task_filing_branch_names_submit_task`.

        (`cli_stage_runner.py` documents that the in-process
        `mcp__recon-report__*` tools are intentionally in NO disallow list.)
        """
        qualified = 'mcp__recon-report__cite_memory'
        assert qualified not in STAGE1_DISALLOWED, (
            f'{qualified} is now in STAGE1_DISALLOWED — the relay clause names a tool '
            'Stage 1 no longer holds, so the CHANNEL needs re-deciding, not this '
            'assertion updating.'
        )
        assert qualified in _STAGE1_SECTION, (
            f'The can_file_tasks=False rendering must name {qualified!r}: it is the '
            'only way a finding\'s memory ids reach Stage 2, and a relay clause that '
            'omits it leaves every relayed finding arriving with an empty list.'
        )

    @pytest.mark.parametrize(
        'constant, _projected', _RELAY_FIELD_SPELLINGS, ids=_RELAY_FIELD_IDS
    )
    def test_both_renderings_name_both_relay_fields(self, constant: str, _projected: str):
        """Producer side and reader side agree structurally, not by convention.

        Stage 1 produces the flagged item; Stage 2 reads it and copies the two
        values into task metadata. Both halves must name the same fields, or
        the hop is two independent guesses that happen to line up today.
        """
        assert _names_token(_STAGE1_SECTION, constant), (
            f'The can_file_tasks=False rendering must name the flagged-item field '
            f'{constant!r} — it is the PRODUCER side of the relay.'
        )
        assert _names_token(_STAGE2_SECTION, constant), (
            f'The can_file_tasks=True rendering must name the flagged-item field '
            f'{constant!r} — it is the READER side, and it is where the value is '
            'taken FROM.'
        )
