"""Anti-drift parity guard: the code-quality LISTS in ``docs/code-quality.md``
vs the copies the reviewer and architect role prompts now carry inline
(task 5225).

A dispatched agent's system prompt cannot follow a cross-reference, so
``orchestrator/src/orchestrator/agents/roles.py`` carries the fourteen
headlines inline. That is a second copy of a list whose single normative home
is ``docs/code-quality.md`` (INV-9 ``one-fact-one-home``; the doc's own
preamble records that a restated copy went stale once already, task 3802).
This module is what keeps the inline copy provably DERIVED rather than a
second source.

WHAT THIS ASSERTS — three things, and no sentence of prose on either side.
(1) The 14 bold HEADLINE TOKENS of a numbered markdown list under the literal
heading ``## The fourteen heuristics``. (2) The bold ITEM LABELS of the two
bullet lists the prompt also duplicates — the two stances and the
do-not-steer-by list. Both are read from BOTH sides by the same parser and
compared as ordered list EQUALITIES rather than as substring pins, so renaming,
reordering, adding or dropping an item on either side fails loudly and names the
divergence, and neither list can drift from the doc unnoticed. (3) The splice
contract — which roles carry ``CODE_QUALITY_GUIDANCE``, how many times, and
into which half of the reviewer's prompt.

What it does NOT assert is any sentence of prose, on either side. The doc's
agreed readings, each bullet's body and the prompt's surrounding instructions
can all be rewritten word for word and this module stays green; only headline
and label TOKENS are compared, and only against the doc. The items INSIDE those
bullets are deliberately unpinned for the reason recorded above
``_LABEL_SECTIONS``. ``TestHeadlineParser`` demonstrates the mechanism's
can-fail executably rather than by assertion.

TASK 5192 — open when this landed — is adjudicating whether prompt-PROSE drift
guards earn their edit friction. This guard is over STRUCTURED lists — numbered
headlines and bullet labels — not prose, which is exactly why rewording every
sentence on either side is a no-op here. But if 5192 rules against structured
pins too, THIS MODULE is in scope for that ruling: delete it and leave the
prompt block in place.

Built in two halves, in that order — the layout of
``orchestrator/tests/test_cited_test_class_drift.py``: unit tests of the pure
parser against synthetic strings and ``tmp_path`` files FIRST, then the wired
guard against the real doc and the real role prompts. The real tree is green on
arrival, so the synthetic half carries the whole burden of proving the
mechanism can FAIL.
"""

from __future__ import annotations

import dataclasses
import re
from typing import NamedTuple

import pytest
from _orch_helpers import make_prompt_resolution_workflow
from _role_splice_contract import SpliceContract, assert_brace_free, assert_nonempty
from code_quality_headlines import (
    DOC_PATH,
    HEADLINE_SECTION_HEADING,
    bold_item_labels,
    doc_bold_item_labels,
    doc_headlines,
    numbered_headlines,
)
from shared.prompt_artifact import PromptArtifactStore

from orchestrator.agents.roles import (
    ARCHITECT_CODE_QUALITY_ADDENDUM,
    CODE_QUALITY_GUIDANCE,
    REVIEWER_COMPREHENSIVE,
    ROLES,
)

_ANCHOR = HEADLINE_SECTION_HEADING

#: The roles that judge or design code, and so are given the heuristics inline.
_CODE_QUALITY_ROLES = frozenset({'architect', 'deep_reviewer', 'reviewer_comprehensive'})

# ---------------------------------------------------------------------------
# The content half — MEASUREMENT RECORD, read this before adding an anchor.
#
# Nothing here is PINNED against a literal written in this file. Every list the
# prompt duplicates is DERIVED from docs/code-quality.md by the same parser that
# reads the prompt, and compared as an ordered equality. That is the whole
# difference between this module and a wording pin: a literal anchor asserts
# that someone once typed a phrase, while a derivation asserts that the two
# copies still agree — which is the only property INV-9 actually wants.
#
# The items INSIDE the bullets are deliberately not compared: the five
# interface-design smells, each bullet's one-clause reason, and the cite-by-name
# instruction. All of them remain MANDATED content of CODE_QUALITY_GUIDANCE;
# what is absent is a test pin, never the prompt text. Do not add pins back.
#
# WHY, measured in both directions against this block's own prose: rewording
# "monkeypatching a private name by dotted path" to "patching private names via
# their import path" is benign and would redden the suite, while turning "report
# a reach-back import as an interface-design finding" into "do NOT report a
# reach-back import as an interface-design finding" INVERTS the instruction and
# leaves every fragment anchor green. A fragment pin therefore fails on a
# legitimate tightening and passes on prose reworded to say the opposite — no
# correctness content in either direction, only a tax on future prompt edits.
# That is the standing rule of _role_splice_contract.py; two pins of this exact
# family were deleted from test_roles_wait_pattern.py under the task 3607
# review, and test_roles_scope_boundary.py records the same lesson from commit
# d794419730. The bodies of these bullets diverge from the doc's BY DESIGN — the
# prompt drops the doc's repo-specific measurements — so only labels are
# comparable, and a body comparison would be a false failure waiting to happen.
# ---------------------------------------------------------------------------


class _LabelSection(NamedTuple):
    """A bullet list the role prompts duplicate from the doc, label for label."""

    prompt_heading: str
    doc_heading: str
    label_count: int


#: The two label lists, with the heading each side files them under. The stances
#: share one heading; the do-not-steer-by list does not, because the doc keeps it
#: inside a wider measurement section (a table of instruments, then the four
#: don'ts) that the prompt deliberately omits. One parser still reads both sides,
#: so the SHAPE compared is identical — only the section locator differs.
_LABEL_SECTIONS = (
    _LabelSection('## Two stances', '## Two stances', 2),
    _LabelSection('## Do not steer by', '## What to measure, and what not to steer by', 4),
)

#: Substrings that would break one of the existing all-roles prompt scanners if
#: a later edit to this one constant introduced them.
_FORBIDDEN_IN_ANY_PROMPT_BLOCK = (
    'mcp__',
    'submit_review_verdict',
    'Output pure JSON',
    'produce a structured JSON review',
    ':!.task',
)

#: Any allowed executor model: nothing is pinned for any of them, so the value
#: only has to be a real resolve() key, not a particular one.
_MODEL = 'opus'

# Structurally real synthetics: a numbered bold item BEFORE the anchor and one
# AFTER the section's terminating ``## `` heading, both of which must be
# excluded, around a short well-formed list.
_WELL_FORMED = (
    '# Code quality\n\n'
    '## Definition\n\n'
    '1. **Not a heuristic.** A numbered bold item in an EARLIER section.\n\n'
    f'{_ANCHOR}\n\n'
    'Prose between the heading and the list, which this parser never reads.\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n'
    '2. **Simple control flows.** Few decision points per unit.\n'
    '3. **SPOT — single point of truth.** Each fact lives in one place.\n\n'
    '## Two stances\n\n'
    '4. **Not a heuristic either.** A numbered bold item in a LATER section.\n'
)

# Mirrors the real doc's heuristic 14, whose continuation line carries its own
# ``**No cheating**`` bold run.
_WITH_CONTINUATION = (
    f'{_ANCHOR}\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n'
    '   **No cheating**: an indented continuation line has its own bold run.\n'
    '2. **Simple control flows.** Few decision points per unit.\n\n'
    '## Two stances\n'
)

_WITH_UNBOLDED_TAIL = (
    f'{_ANCHOR}\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n'
    '2. A numbered line carrying no bold headline at all.\n\n'
    '## Two stances\n'
)

_WITH_UNBOLDED_MIDDLE = (
    f'{_ANCHOR}\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n'
    '2. A numbered line carrying no bold headline at all.\n'
    '3. **Simple control flows.** Few decision points per unit.\n\n'
    '## Two stances\n'
)

_NON_CONTIGUOUS = (
    f'{_ANCHOR}\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n'
    '2. **Simple control flows.** Few decision points per unit.\n'
    '4. **Small function scopes.** A function does one thing.\n\n'
    '## Two stances\n'
)

# The anchor DEMOTED below ``## ``, with the section that follows carrying
# numbered bold items of its own. A substring search would match the demoted
# heading, and the ``^## `` terminator would not close it.
_DEMOTED_ANCHOR = (
    f'#{_ANCHOR}\n\n'
    '1. **Informative names.** A name says what the thing is and does.\n\n'
    '## A later section\n\n'
    '2. **Not a heuristic.** A numbered bold item the slice must never reach.\n'
)

# Bullet-list synthetics. Continuation lines sit at column 0 carrying their own
# inline bold, exactly as the real prompt's stances do.
_BULLET_SECTION = (
    '## Two stances\n\n'
    '- **Comments.** Aim for code that is clear with no or low comments, since\n'
    'a continuation line carries its own **inline bold** and is not an item.\n'
    "- **Tests.** Test access to a module's internals is an interface smell.\n\n"
    '## Do not steer by\n\n'
    '- **Raw line count.** A bullet in a LATER section.\n'
)


class TestHeadlineParser:
    """The pure parser, against synthetic strings and ``tmp_path`` files only.

    No test in this class reads the real doc or the real role prompts; that is
    the wired half's job. These are what prove the mechanism can FAIL.
    """

    def test_headlines_returned_in_document_order(self):
        assert numbered_headlines(_WELL_FORMED, _ANCHOR) == [
            'Informative names.',
            'Simple control flows.',
            'SPOT — single point of truth.',
        ]

    def test_indented_continuation_bold_run_is_not_a_headline(self):
        # The match is anchored at column 0, so heuristic 14's inline
        # ``**No cheating**`` continuation cannot masquerade as a 15th item.
        assert numbered_headlines(_WITH_CONTINUATION, _ANCHOR) == [
            'Informative names.',
            'Simple control flows.',
        ]

    def test_numbered_line_without_a_bold_headline_is_not_picked_up(self):
        assert numbered_headlines(_WITH_UNBOLDED_TAIL, _ANCHOR) == ['Informative names.']

    def test_unbolded_line_inside_the_list_is_caught_by_the_contiguity_check(self):
        # Not picking up an unbolded line is only safe because a DROPPED item
        # inside the list shows up as a hole in the numbering. This pins the
        # interaction between those two rules so neither can be relaxed alone.
        with pytest.raises(ValueError, match=r'\[1, 3\]'):
            numbered_headlines(_WITH_UNBOLDED_MIDDLE, _ANCHOR)

    def test_section_stops_at_the_next_heading(self):
        headlines = numbered_headlines(_WELL_FORMED, _ANCHOR)
        assert 'Not a heuristic.' not in headlines  # before the anchor
        assert 'Not a heuristic either.' not in headlines  # after the terminator

    def test_missing_heading_raises_naming_the_heading_it_looked_for(self):
        # Never ``[]`` and never a skip: a renamed heading is exactly the drift
        # this guard exists to catch (INV-2 structured-facts-at-failure).
        text = _WELL_FORMED.replace(_ANCHOR, '## The fifteen heuristics')
        with pytest.raises(ValueError, match=re.escape(_ANCHOR)):
            numbered_headlines(text, _ANCHOR)

    def test_non_contiguous_numbering_raises_naming_the_observed_numbers(self):
        with pytest.raises(ValueError, match=r'\[1, 2, 4\]'):
            numbered_headlines(_NON_CONTIGUOUS, _ANCHOR)

    def test_one_renamed_headline_makes_the_two_lists_unequal(self):
        # THE CAN-FAIL PROOF. The real tree is green on arrival, so without
        # this the wired half would pass identically if the parser silently
        # returned the same wrong answer (e.g. ``[]``) on both sides.
        drifted = _WELL_FORMED.replace(
            '2. **Simple control flows.**', '2. **Straightforward control flows.**',
        )
        assert numbered_headlines(drifted, _ANCHOR) != numbered_headlines(_WELL_FORMED, _ANCHOR)

    def test_a_demoted_heading_raises_rather_than_matching_as_a_substring(self):
        # Demoting the heading below ``## `` is real structural drift, and the
        # ``^## `` terminator cannot close a ``### `` section — so a substring
        # match would run the slice on into later sections and report success.
        # The heading is matched as a WHOLE LINE for exactly this reason; both
        # parsers here share that slicer.
        with pytest.raises(ValueError, match=re.escape(_ANCHOR)):
            numbered_headlines(_DEMOTED_ANCHOR, _ANCHOR)


class TestBoldItemLabels:
    """The bullet-label parser, against synthetic strings only.

    Same shape as :class:`TestHeadlineParser` and for the same reason: the real
    doc and the real prompts agree on arrival, so only synthetics can show that
    disagreement would be caught.
    """

    def test_labels_returned_in_document_order(self):
        assert bold_item_labels(_BULLET_SECTION, '## Two stances') == ['Comments.', 'Tests.']

    def test_a_continuation_line_s_inline_bold_is_not_a_label(self):
        assert 'inline bold' not in bold_item_labels(_BULLET_SECTION, '## Two stances')

    def test_section_stops_at_the_next_heading(self):
        assert bold_item_labels(_BULLET_SECTION, '## Do not steer by') == ['Raw line count.']

    def test_missing_heading_raises_naming_the_heading_it_looked_for(self):
        with pytest.raises(ValueError, match=re.escape('## Nowhere')):
            bold_item_labels(_BULLET_SECTION, '## Nowhere')

    def test_one_renamed_label_makes_the_two_lists_unequal(self):
        # THE CAN-FAIL PROOF for the label half.
        drifted = _BULLET_SECTION.replace('- **Comments.**', '- **On comments.**')
        assert (
            bold_item_labels(drifted, '## Two stances')
            != bold_item_labels(_BULLET_SECTION, '## Two stances')
        )


class TestDocHeadlines:
    """``doc_headlines`` — the file-reading wrapper around the pure parser."""

    def test_doc_path_resolves_to_a_real_code_quality_doc(self):
        # ``is_file()`` is the assertion with content: it is what catches a wrong
        # ``parents[]`` index in the helper. Re-deriving the expected path here
        # and asserting equality would not — both sides would move together —
        # so it is deliberately not done (heuristic 11).
        #
        # Resolved from ``__file__``, so this holds inside a ``.worktrees/<id>``
        # checkout, which is where this test itself runs.
        assert DOC_PATH.is_file()
        assert DOC_PATH.name == 'code-quality.md'

    def test_missing_doc_raises_loudly_rather_than_skipping(self, tmp_path):
        # tmp_path holds no code-quality.md at all. A caller with no doc to
        # check against has no basis for asserting parity, and treating that as
        # "nothing to check" would disable the guard exactly when its
        # precondition breaks (exemplar:
        # shared/tests/test_architecture_doc_transition_parity.py::TestDocPath).
        with pytest.raises(OSError):
            doc_headlines(tmp_path / 'code-quality.md')

    def test_reads_a_tmp_path_doc_through_the_same_parser(self, tmp_path):
        doc = tmp_path / 'code-quality.md'
        doc.write_text(_WELL_FORMED, encoding='utf-8')
        assert doc_headlines(doc) == numbered_headlines(_WELL_FORMED, _ANCHOR)



@pytest.fixture(scope='module')
def resolved_prompts(tmp_path_factory) -> dict[str, str]:
    """Every role's system prompt as PRODUCTION renders it, with NOTHING pinned.

    Resolved through the single production chokepoint
    ``TaskWorkflow._resolve_role_system_prompt``, which is what the task
    requires: reading ``roles.py`` source, or even ``role.system_prompt``,
    proves only that the text was typed — not that it survived ``str.format``
    interpolation and ``compose_prompt`` assembly.

    WHAT IS AND IS NOT EXERCISED. The artifact store is rooted at a fresh empty
    ``tmp_path``, so ``reviewer_comprehensive`` — the one ``prompt_spec``-
    carrying role — always takes the FALLBACK branch to its in-code constant.
    The pinned branch is not exercised anywhere in this module, and that is the
    intended semantics rather than a gap: a pinned artifact for (reviewer
    prompt_id, model, ``_REVIEWER_PROMPT_HARNESS_VERSION``) replaces the whole
    composed prompt, and the code-quality block lives in the optimizer-owned
    HEURISTICS half (PRD tier1-prompt-optimization D-3) with the harness version
    deliberately NOT bumped for it. So a reviewer resolving a pre-existing pin
    can ship with no code-quality block at all while this module stays green —
    the optimizer owning that half is the mechanism working as specified. If
    that ever stops being intended, the remedy is to bump
    ``_REVIEWER_PROMPT_HARNESS_VERSION`` so old pins stop resolving, not to
    assert here.

    Module-scoped, and built from ``tmp_path_factory`` rather than at import
    time, because resolution needs a real project root and artifacts root. It
    is still constructed exactly once for the module, which is what the
    consuming ``SpliceContract`` needs.
    """
    root = tmp_path_factory.mktemp('prompt_resolution')
    workflow = make_prompt_resolution_workflow(
        tmp_path=root, prompt_store=PromptArtifactStore(root / 'artifacts'),
    )
    return {
        name: workflow._resolve_role_system_prompt(role, _MODEL)
        for name, role in ROLES.items()
    }


@pytest.mark.parametrize('role_name', sorted(_CODE_QUALITY_ROLES))
class TestFourteenHeuristicsReachTheRolePrompts:
    """The wired half: the real doc against the real, production-rendered prompts.

    The assertion is deliberately an ordered list EQUALITY over ONE shared
    anchor parsed by ONE shared parser, not a substring pin. That is what keeps
    ``docs/code-quality.md`` the single point of truth (INV-9, heuristic 11)
    with the prompt derived from it: a headline renamed, reordered, added or
    dropped on EITHER side fails and names the divergence, while rewording any
    prose on either side is a no-op.

    Only this TEST reads ``docs/code-quality.md``. No runtime code path does, so
    an orchestrator operating a project whose checkout lacks the doc is
    unaffected — the prompt carries the headlines inline.
    """

    def test_prompt_headlines_equal_the_doc_headlines(self, role_name, resolved_prompts):
        assert (
            numbered_headlines(resolved_prompts[role_name], HEADLINE_SECTION_HEADING)
            == doc_headlines()
        )

    def test_there_are_exactly_fourteen_of_them(self, role_name, resolved_prompts):
        # Separate from the equality above so a doc edit that deleted items from
        # BOTH sides cannot pass vacuously.
        assert len(numbered_headlines(resolved_prompts[role_name], HEADLINE_SECTION_HEADING)) == 14


@pytest.fixture(scope='module')
def contract(resolved_prompts) -> SpliceContract:
    """The splice contract for ``CODE_QUALITY_GUIDANCE``, over RESOLVED prompts.

    ``all_roles`` is INJECTED rather than left to default to ``ROLES``. The
    resolved prompt and the static ``system_prompt`` attribute are byte-identical
    today only because nothing is pinned, so asserting on the attribute would be
    a coincidence rather than a contract; the helper documents ``all_roles`` as
    the injectable seam for exactly this substitution.

    ``capability`` is a required field of the dataclass and has to be supplied,
    but it is NEVER asserted on here. It is written as an explicit membership
    predicate over ``_CODE_QUALITY_ROLES`` so that no reader can mistake it for
    a derivation.

    TWO ARMS ARE DELIBERATELY NOT CALLED — do not "complete" the contract.

    ``assert_role_set_matches_capability`` needs a machine-derivable property
    that justifies the splice, and none exists. Measured: ``AgentRole`` carries
    no such field; ``mcp_families`` cuts straight across the carrier set
    (``architect`` shares ``plan_tools`` with ``debugger``/``implementer``/
    ``simple_task``, and ``deep_reviewer``'s ``frozenset({'orchestrator'})``
    equals ``steward``'s); and ``prompt_spec is None`` splits it,
    ``reviewer_comprehensive`` being the sole pinned role. The carrier set is an
    EDITORIAL judgement about which roles judge or design code, so deriving it
    from a membership predicate over itself would be exactly the tautology the
    helper's docstring forbids. ``assert_no_other_role_carries`` is the guard
    that replaces it.

    ``assert_placement`` cannot be called either: a
    ``follows=ERROR_REMEDY_HINT_GUIDANCE`` call falls back to the up-front rule
    for any role whose prompt does not contain that predecessor, and fails
    there. ``reviewer_comprehensive`` is such a role — the wait/rejection/remedy
    trio is spliced into ``_UNPINNED_PROMPT_ROLES``, which excludes the one
    pinned role. Placement is covered instead by the ordered list equalities
    above and by ``test_the_reviewer_block_landed_in_the_editable_half``.

    Both records are prose, not assertions. Asserting them would pin incidental
    facts about roles this module has no contract with — a legitimate change to
    ``steward``'s MCP families, or a later task splicing the remedy-hint block
    into the reviewer, would redden a code-quality parity test whose failure
    message could only mislead the editor into "fixing" the wrong thing.
    """
    return SpliceContract(
        constant_name='CODE_QUALITY_GUIDANCE',
        constant=CODE_QUALITY_GUIDANCE,
        roles=_CODE_QUALITY_ROLES,
        role_set_name='_CODE_QUALITY_ROLES',
        capability=lambda role: role.name in _CODE_QUALITY_ROLES,
        capability_description=(
            'the editorial judgement that this role judges or designs code'
        ),
        all_roles={
            name: dataclasses.replace(role, system_prompt=resolved_prompts[name])
            for name, role in ROLES.items()
        },
    )


class TestStancesAndCarrierSet:
    """The rest of what the block must carry, and where it may and may not land."""

    # -- content: the two lists derived from the doc --------------------------

    @pytest.mark.parametrize('role_name', sorted(_CODE_QUALITY_ROLES))
    @pytest.mark.parametrize('section', _LABEL_SECTIONS, ids=lambda s: s.prompt_heading)
    def test_prompt_labels_equal_the_doc_labels(self, role_name, section, resolved_prompts):
        # Same mechanism as the fourteen headlines, applied to the other two
        # lists the prompt duplicates: an ordered equality against the doc, not
        # a literal anchor. A heading missing on either side raises inside the
        # parser, naming it, so the section's presence needs no separate test.
        assert (
            bold_item_labels(resolved_prompts[role_name], section.prompt_heading)
            == doc_bold_item_labels(section.doc_heading)
        )

    @pytest.mark.parametrize('section', _LABEL_SECTIONS, ids=lambda s: s.prompt_heading)
    def test_each_derived_list_has_its_full_membership(self, section):
        # Non-vacuity, exactly as test_there_are_exactly_fourteen_of_them is for
        # the headlines: an equality alone would hold if a list were emptied on
        # both sides at once, or if a renamed heading were fixed on both sides
        # while the items were lost.
        assert len(doc_bold_item_labels(section.doc_heading)) == section.label_count

    # -- the splice contract ------------------------------------------------

    def test_guidance_is_nonempty(self):
        assert_nonempty(
            'CODE_QUALITY_GUIDANCE',
            CODE_QUALITY_GUIDANCE,
            remedy=(
                'Restore the block in roles.py. Every containment assertion in '
                'this module passes vacuously against an emptied constant.'
            ),
        )

    def test_guidance_is_brace_free(self):
        # Load-bearing, not defensive: this constant is spliced into
        # _REVIEWER_HEURISTICS_TEMPLATE, which is run through
        # str.format(specialization=...) by build_reviewer_prompt_spec. A literal
        # brace raises at format time or mangles the rendered reviewer prompt.
        assert_brace_free(
            'CODE_QUALITY_GUIDANCE',
            CODE_QUALITY_GUIDANCE,
            remedy=(
                'Rewrite the offending line without braces — this constant reaches '
                'build_reviewer_prompt_spec\'s str.format() call.'
            ),
        )

    def test_every_carrier_role_carries_the_guidance(self, contract):
        contract.assert_every_role_carries(
            remedy=(
                'A role that judges or designs code must carry '
                'CODE_QUALITY_GUIDANCE; re-splice it, or drop the role from '
                '_CODE_QUALITY_ROLES and say why above the set.'
            ),
        )

    def test_no_other_role_carries_the_guidance(self, contract):
        # This stands in for assert_role_set_matches_capability, which is not
        # called (see the contract fixture): it is the honest guard against an
        # accidental splice into an excluded role, which would ship silently and
        # be paid on every invocation of that role.
        contract.assert_no_other_role_carries(
            remedy=(
                'Either remove the splice from that role, or add it to '
                '_CODE_QUALITY_ROLES and record why it judges or designs code.'
            ),
        )

    def test_guidance_is_spliced_exactly_once_per_carrier(self, contract):
        # absent_ok=True is passed EXPLICITLY and is not the helper's default.
        # The default (False) is reserved for a constant that is a composed HALF
        # whose composition into the splice unit is separately pinned, so that a
        # 0 count proves the whole splice is missing. CODE_QUALITY_GUIDANCE is a
        # whole splice unit, not a half, so that reasoning does not apply here —
        # and skipping the 0 count keeps PRESENCE (the containment test above)
        # and DUPLICATE SPLICE (this test) failing independently for their two
        # distinct root causes, instead of one dropped splice reddening both.
        #
        # Separately: the reviewer's SPECIALIZATION is interpolated into both
        # halves of its prompt and so appears twice BY DESIGN. That is not this
        # constant, and a count of 2 for this constant would be a real
        # duplicate-splice bug.
        contract.assert_spliced_exactly_once(
            absent_ok=True,
            remedy=(
                'Delete the stale duplicate splice. A second copy doubles the '
                'block on every invocation of that role.'
            ),
        )

    # -- invariants with no shared helper ------------------------------------

    @pytest.mark.parametrize('role_name', ['architect', 'deep_reviewer'])
    def test_the_spec_less_carriers_did_not_gain_a_prompt_spec(self, role_name):
        # Both carry the block as a plain concatenated constant, exactly like
        # their four sibling shared blocks. The set-wide equality over every
        # unpinned role is owned by test_roles_tool_call_rejection.py's
        # _UNPINNED_PROMPT_ROLES and deliberately not copied here.
        assert ROLES[role_name].prompt_spec is None

    def test_the_reviewer_block_landed_in_the_editable_half(self):
        # The task mandates the HEURISTICS half — which PRD
        # tier1-prompt-optimization D-3 defines as exactly what an optimizer may
        # rewrite — and forbids the frozen contract. Asserted in BOTH directions
        # so a later editor cannot quietly move it.
        spec = REVIEWER_COMPREHENSIVE.prompt_spec
        assert spec is not None
        assert CODE_QUALITY_GUIDANCE in spec.baseline_heuristics
        assert CODE_QUALITY_GUIDANCE not in spec.contract

    @pytest.mark.parametrize('forbidden', _FORBIDDEN_IN_ANY_PROMPT_BLOCK)
    def test_guidance_cannot_break_the_all_roles_scanners(self, forbidden):
        assert forbidden not in CODE_QUALITY_GUIDANCE

    def test_the_architect_addendum_reaches_only_the_architect(self, resolved_prompts):
        # Heuristics 13 and 14 bind hardest on a plan that splits or extracts a
        # module, which is architect-only content. Forking the shared constant
        # into reviewer and architect variants to carry it would duplicate the
        # fourteen headlines — the exact defect this task exists to prevent — so
        # it is appended at the ARCHITECT call site instead. This pins that the
        # split has not silently collapsed in either direction.
        assert ARCHITECT_CODE_QUALITY_ADDENDUM in resolved_prompts['architect']
        for role_name in sorted(_CODE_QUALITY_ROLES - {'architect'}):
            assert ARCHITECT_CODE_QUALITY_ADDENDUM not in resolved_prompts[role_name]
