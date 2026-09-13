"""Anti-drift parity guard: the fourteen code-quality heuristic HEADLINES in
``docs/code-quality.md`` vs the copies the reviewer and architect role prompts
now carry inline (task 5225).

A dispatched agent's system prompt cannot follow a cross-reference, so
``orchestrator/src/orchestrator/agents/roles.py`` carries the fourteen
headlines inline. That is a second copy of a list whose single normative home
is ``docs/code-quality.md`` (INV-9 ``one-fact-one-home``; the doc's own
preamble records that a restated copy went stale once already, task 3802).
This module is what keeps the inline copy provably DERIVED rather than a
second source.

WHAT THIS ASSERTS, AND NOTHING ELSE: the 14 bold HEADLINE TOKENS of a numbered
markdown list under the literal heading ``## The fourteen heuristics``, parsed
by ONE parser from BOTH sides and compared as an ordered list. It asserts
nothing whatever about the prose around them — the doc's agreed readings and
the prompt's surrounding instructions can both be rewritten word for word and
this module stays green (``TestHeadlineParser`` demonstrates that executably
rather than by assertion). It is an ordered list EQUALITY, not a substring
pin, so renaming, reordering, adding or dropping a headline on either side
fails loudly and names the divergence.

TASK 5192 — open when this landed — is adjudicating whether prompt-PROSE drift
guards earn their edit friction. This guard is over a STRUCTURED numbered list,
not prose, which is exactly why rewording every sentence on either side is a
no-op here. But if 5192 rules against structured pins too, THIS MODULE is in
scope for that ruling: delete it and leave the prompt block in place.

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
from pathlib import Path

import pytest
from _orch_helpers import make_prompt_resolution_workflow
from _role_splice_contract import SpliceContract, assert_brace_free, assert_nonempty
from code_quality_headlines import (
    DOC_PATH,
    HEADLINE_SECTION_HEADING,
    doc_headlines,
    numbered_headlines,
)
from shared.prompt_artifact import PromptArtifactStore

from orchestrator.agents.roles import (
    ARCHITECT_CODE_QUALITY_ADDENDUM,
    CODE_QUALITY_GUIDANCE,
    ERROR_REMEDY_HINT_GUIDANCE,
    REVIEWER_COMPREHENSIVE,
    ROLES,
)

# orchestrator/tests/test_code_quality_guidance_parity.py -> parents[0]=tests,
# parents[1]=orchestrator, parents[2]=repo root. Same idiom as
# orchestrator/tests/conftest.py's REPO_ROOT and test_cited_test_class_drift.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_ANCHOR = HEADLINE_SECTION_HEADING

#: The roles that judge or design code, and so are given the heuristics inline.
_CODE_QUALITY_ROLES = frozenset({'architect', 'deep_reviewer', 'reviewer_comprehensive'})

# ---------------------------------------------------------------------------
# Anchors for the content half. Short NOUN PHRASES and structural HEADINGS, never
# connective prose: a prose pin passes on prose reworded to say the opposite and
# fails on a legitimate tightening, so it only taxes future prompt edits (the
# standing rule of _role_splice_contract.py, and the lesson
# test_roles_scope_boundary.py records from the two literal pins deleted in
# commit d794419730). Every anchor below names a distinct MANDATED ITEM whose
# absence is a real loss of guidance; the sentences carrying them can be
# rewritten freely.
# ---------------------------------------------------------------------------

_STANCE_ANCHORS = (
    '## Two stances',
    '**Comments.**',
    '**Tests.**',
)

#: The five symptoms the tests stance must name, one anchor each.
_INTERFACE_SMELL_ANCHORS = (
    'dotted path',           # monkeypatching a private name by dotted path
    'private attribute',     # reading a private attribute from a test
    'reach-back import',
    'function-local import',  # placed to break an import cycle
    're-export shim',
)

#: ...and the verdict they carry: findings about the interface, not style nits.
_INTERFACE_SMELL_VERDICT_ANCHOR = 'interface-design finding'

#: The four things a reviewer must not steer by, one anchor each.
_DO_NOT_STEER_BY_ANCHORS = (
    '## Do not steer by',
    'Raw line count',
    'Average complexity',
    'autouse stubs',
    'test-to-code ratio',
)

#: The instruction that makes the heuristics usable rather than decorative.
_CITE_BY_NAME_ANCHOR = 'Name the heuristic you are applying'

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


class TestDocHeadlines:
    """``doc_headlines`` — the file-reading wrapper around the pure parser."""

    def test_doc_path_is_the_repo_root_code_quality_md(self):
        # Resolved from ``__file__``, so this holds inside a ``.worktrees/<id>``
        # checkout, which is where this test itself runs.
        assert DOC_PATH == _REPO_ROOT / 'docs' / 'code-quality.md'
        assert DOC_PATH.is_file()

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
    """Every role's system prompt as PRODUCTION renders it.

    Resolved through the single production chokepoint
    ``TaskWorkflow._resolve_role_system_prompt``, which is what the task
    requires: reading ``roles.py`` source, or even ``role.system_prompt``,
    proves only that the text was typed — not that it survived ``str.format``
    interpolation and ``compose_prompt`` assembly. Both production branches are
    exercised with no per-role branching here (``prompt_spec`` set -> the
    artifact store; ``prompt_spec is None`` -> ``system_prompt`` verbatim).

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
    but it is NEVER asserted on here — see
    ``test_role_set_matches_capability_is_deliberately_not_called``. It is
    written as an explicit membership predicate over ``_CODE_QUALITY_ROLES`` so
    that no reader can mistake it for a derivation.
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

    # -- content: the two stances -------------------------------------------

    @pytest.mark.parametrize('role_name', sorted(_CODE_QUALITY_ROLES))
    @pytest.mark.parametrize('anchor', _STANCE_ANCHORS)
    def test_both_stances_are_present(self, role_name, anchor, resolved_prompts):
        assert anchor in resolved_prompts[role_name]

    @pytest.mark.parametrize('role_name', sorted(_CODE_QUALITY_ROLES))
    @pytest.mark.parametrize('anchor', _INTERFACE_SMELL_ANCHORS)
    def test_each_interface_design_smell_is_named(self, role_name, anchor, resolved_prompts):
        assert anchor in resolved_prompts[role_name]

    @pytest.mark.parametrize('role_name', sorted(_CODE_QUALITY_ROLES))
    def test_the_smells_are_reported_as_interface_findings_not_style_nits(
        self, role_name, resolved_prompts,
    ):
        # Without this the five anchors above could all be present in a section
        # that told the reviewer to ignore them.
        assert _INTERFACE_SMELL_VERDICT_ANCHOR in resolved_prompts[role_name]

    # -- content: what not to steer by, and how to cite ---------------------

    @pytest.mark.parametrize('role_name', sorted(_CODE_QUALITY_ROLES))
    @pytest.mark.parametrize('anchor', _DO_NOT_STEER_BY_ANCHORS)
    def test_each_do_not_steer_by_item_is_named(self, role_name, anchor, resolved_prompts):
        assert anchor in resolved_prompts[role_name]

    @pytest.mark.parametrize('role_name', sorted(_CODE_QUALITY_ROLES))
    def test_cite_the_heuristic_by_name_is_instructed(self, role_name, resolved_prompts):
        assert _CITE_BY_NAME_ANCHOR in resolved_prompts[role_name]

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
        # This stands in for assert_role_set_matches_capability (not called, see
        # below): it is the honest guard against an accidental splice into an
        # excluded role, which would ship silently and be paid on every
        # invocation of that role.
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

    def test_role_set_matches_capability_is_deliberately_not_called(self):
        """MEASUREMENT RECORD — do not "complete" the contract by adding it.

        ``assert_role_set_matches_capability`` needs a machine-derivable property
        that justifies the splice, and none exists. Measured: ``AgentRole``
        carries no such field; ``mcp_families`` cuts straight across the carrier
        set (``architect`` shares ``plan_tools`` with ``debugger``/
        ``implementer``/``simple_task``, and ``deep_reviewer``'s
        ``frozenset({'orchestrator'})`` equals ``steward``'s); and
        ``prompt_spec is None`` splits it, ``reviewer_comprehensive`` being the
        sole pinned role. The carrier set is an EDITORIAL judgement about which
        roles judge or design code, so deriving it from a membership predicate
        over itself would be exactly the tautology the helper's docstring
        forbids. ``assert_no_other_role_carries`` is the guard that replaces it.

        This test asserts the two measurements that make the above true, so the
        record cannot silently go stale.
        """
        assert ROLES['deep_reviewer'].mcp_families == ROLES['steward'].mcp_families
        assert {name for name in _CODE_QUALITY_ROLES if ROLES[name].prompt_spec is None} == {
            'architect', 'deep_reviewer',
        }

    def test_placement_is_deliberately_not_asserted_by_the_contract(self, resolved_prompts):
        """MEASUREMENT RECORD — ``assert_placement`` cannot be called here.

        A ``follows=ERROR_REMEDY_HINT_GUIDANCE`` call falls back to the up-front
        rule for any role whose prompt does not contain that predecessor, and
        fails there. ``reviewer_comprehensive`` is such a role: the
        wait/rejection/remedy trio is spliced into ``_UNPINNED_PROMPT_ROLES``,
        which excludes the one pinned role. Placement is instead pinned by the
        ordered list equality of ``TestFourteenHeuristicsReachTheRolePrompts``
        and by the editable-half assertion below.

        This asserts the measurement, so the record cannot go stale.
        """
        assert ERROR_REMEDY_HINT_GUIDANCE not in resolved_prompts['reviewer_comprehensive']

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
