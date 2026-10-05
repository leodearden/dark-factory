"""Splice contract for ``roles.py::CODE_QUALITY_GUIDANCE``.

Asserts which roles carry the code-quality block (architect, deep_reviewer and
reviewer_comprehensive, and no other), that each carries it exactly once, that
the reviewer's copy lands in the optimizer-editable HEURISTICS half, that the
architect-only addendum reaches only the architect, and that the block IS the
render of the packaged normative doc rather than a hand-written copy.

The block's own shape — headings, numbering, list shape, brace-freedom — is tested
against the doc in ``orchestrator/tests/test_code_quality.py``.
"""

from __future__ import annotations

import dataclasses

import pytest
from _orch_helpers import make_prompt_resolution_workflow
from _role_splice_contract import SpliceContract, assert_nonempty
from shared.prompt_artifact import PromptArtifactStore

from orchestrator.agents.code_quality import guidance
from orchestrator.agents.roles import (
    ARCHITECT_CODE_QUALITY_ADDENDUM,
    CODE_QUALITY_GUIDANCE,
    REVIEWER_COMPREHENSIVE,
    ROLES,
)

#: The roles that judge or design code, and so are given the block.
_CODE_QUALITY_ROLES = frozenset({'architect', 'deep_reviewer', 'reviewer_comprehensive'})

#: Any allowed executor model: nothing is pinned for any of them, so the value
#: only has to be a real resolve() key, not a particular one.
_MODEL = 'opus'


@pytest.fixture(scope='module')
def resolved_prompts(tmp_path_factory) -> dict[str, str]:
    """Every role's system prompt as PRODUCTION renders it, with NOTHING pinned.

    Resolved through the single production chokepoint
    ``TaskWorkflow._resolve_role_system_prompt``: reading ``roles.py`` source, or
    even ``role.system_prompt``, proves only that the text was typed — not that
    it survived ``str.format`` interpolation and ``compose_prompt`` assembly.

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
    pinned role. Placement is covered instead by
    ``test_the_reviewer_block_landed_in_the_editable_half``, and by the block
    opening its own ``## `` section, which ``tests/test_code_quality.py``
    asserts.

    Both records are prose, not assertions. Asserting them would pin incidental
    facts about roles this module has no contract with — a legitimate change to
    ``steward``'s MCP families, or a later task splicing the remedy-hint block
    into the reviewer, would redden a code-quality splice test whose failure
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


class TestCodeQualitySplice:
    """What the block is, and where it may and may not land."""

    def test_guidance_is_nonempty(self):
        assert_nonempty(
            'CODE_QUALITY_GUIDANCE',
            CODE_QUALITY_GUIDANCE,
            remedy=(
                'It must be guidance()\'s render of the packaged code_quality.md. '
                'Every containment assertion in this module passes vacuously '
                'against an emptied constant.'
            ),
        )

    def test_the_block_is_the_rendered_normative_doc(self):
        # What keeps the block derived by construction: re-hand-writing it in
        # roles.py, even word for word today, goes red on the next doc edit.
        assert guidance() == CODE_QUALITY_GUIDANCE

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

    @pytest.mark.parametrize('role_name', ['architect', 'deep_reviewer'])
    def test_the_spec_less_carriers_did_not_gain_a_prompt_spec(self, role_name):
        # Both carry the block as a plain concatenated constant, exactly like
        # their four sibling shared blocks. The set-wide equality over every
        # unpinned role is owned by test_roles_tool_call_rejection.py's
        # _UNPINNED_PROMPT_ROLES and deliberately not copied here.
        assert ROLES[role_name].prompt_spec is None

    def test_the_reviewer_block_landed_in_the_editable_half(self):
        # The HEURISTICS half is what PRD tier1-prompt-optimization D-3 defines
        # as exactly what an optimizer may rewrite; the frozen contract is not.
        # Asserted in BOTH directions so a later editor cannot quietly move it.
        spec = REVIEWER_COMPREHENSIVE.prompt_spec
        assert spec is not None
        assert CODE_QUALITY_GUIDANCE in spec.baseline_heuristics
        assert CODE_QUALITY_GUIDANCE not in spec.contract

    def test_the_architect_addendum_reaches_only_the_architect(self, resolved_prompts):
        # Recording a split's heuristic-14 measurement is architect-only work.
        # Forking the shared block into reviewer and architect variants to
        # carry it would duplicate the rendered doc, so it is appended at the
        # ARCHITECT call site instead. This pins that the split has not
        # silently collapsed in either direction.
        assert ARCHITECT_CODE_QUALITY_ADDENDUM in resolved_prompts['architect']
        for role_name in sorted(_CODE_QUALITY_ROLES - {'architect'}):
            assert ARCHITECT_CODE_QUALITY_ADDENDUM not in resolved_prompts[role_name]
