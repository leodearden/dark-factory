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
"""

from __future__ import annotations

import pytest
from shared.task_metadata import parse_metadata

from fused_memory.reconciliation.prompts import (
    FINDING_ID_METADATA_KEY,
    FINDING_MEMORY_IDS_METADATA_KEY,
    get_recon_report_tool_guidance,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

# The call-opener scanners are imported from the drift-guard module that owns
# them rather than re-implemented, for the same reason step-6 imports
# DISALLOW_TASK_WRITES from cli_stage_runner: a hand-rolled twin of a guard
# tracks the original only until one of them is edited.
from test_recon_report_guidance_drift import (  # noqa: E402
    _AGENT_CALLED_REPORT_TOOLS,
    _extract_call_args_at,
    _iter_call_openers,
)

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

    @pytest.mark.parametrize(
        'name, prompt',
        [
            ('STAGE1_SYSTEM_PROMPT', STAGE1_SYSTEM_PROMPT),
            ('STAGE2_SYSTEM_PROMPT', STAGE2_SYSTEM_PROMPT),
            ('STAGE3_SYSTEM_PROMPT', STAGE3_SYSTEM_PROMPT),
        ],
        # Explicit ids are required, not cosmetic: pytest otherwise derives each
        # id from the parametrized value, embedding a whole multi-KB system
        # prompt in every test id.
        ids=['stage1', 'stage2', 'stage3'],
    )
    def test_every_stage_prompt_names_both_keys(self, name: str, prompt: str):
        """All three assembled prompts carry both canonical names.

        Stage 3 is included deliberately. It is read-only and cannot file
        tasks, but it renders the same shared block, and "all three stages
        agree on the vocabulary" is the property being pinned — agreement about
        what the keys ARE is separable from the license to write them.
        """
        for key, _ratified in _RATIFIED_SPELLINGS:
            assert key in prompt, (
                f'{name} must name the canonical provenance key {key!r}; a stage left '
                'without the name is a stage free to invent a spelling.'
            )


class TestGuidanceClauseDoesNotReopenTheRunIdHole:
    """The new clause must not smuggle in a hand-written call example.

    `_render_recon_report_tool_guidance`'s call examples are GENERATED from live
    tool signatures (task-2559) precisely so they cannot omit a required kwarg.
    A hand-written example added alongside them would bypass that generator and
    reopen the exact drift task-2559 closed — so the clause is prose that NAMES
    the two keys and shows no call.
    """

    def test_every_call_opener_in_the_guidance_still_carries_run_id(self):
        guidance = get_recon_report_tool_guidance()
        for tool_name in _AGENT_CALLED_REPORT_TOOLS:
            for paren_idx in _iter_call_openers(guidance, tool_name):
                args_substr = _extract_call_args_at(guidance, paren_idx)
                assert 'run_id' in args_substr, (
                    f'A `{tool_name}(...)` example in the shared guidance is missing '
                    f'`run_id` — got: {tool_name}({args_substr})'
                )
