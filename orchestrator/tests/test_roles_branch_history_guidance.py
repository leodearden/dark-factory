"""Guards for where ``agents/branch_history_guidance.py``'s constants reach.

What this module guards: the writer section ``BRANCH_HISTORY_GUIDANCE`` reaches
exactly the roles whose commits the harness attributes to plan steps (derived
from the ``mark_step_done`` grant), once each, and its heading constant
``BRANCH_HISTORY_HEADING`` is the one the section opens with. The reviewer
section ``REVIEWER_BRANCH_HISTORY_RULE`` sits in the reviewer CONTRACT of both
generations, names the WIP subject the harness mints, and reaches no other role.

Every assertion is a containment, count or equality check against a NAMED live
constant: no regex over prose and no literal wording pin. The standing rule and
its rationale live in ``orchestrator/tests/_role_splice_contract.py``.
"""

from __future__ import annotations

import pytest
from _role_splice_contract import SpliceContract, assert_nonempty, assert_opens_own_section

from orchestrator.agents.branch_history_guidance import (
    BRANCH_HISTORY_GUIDANCE,
    BRANCH_HISTORY_HEADING,
    REVIEWER_BRANCH_HISTORY_RULE,
)
from orchestrator.agents.roles import (
    REVIEWER_COMPREHENSIVE,
    REVIEWER_COMPREHENSIVE_CONTEXT_COMPLETE,
    ROLES,
)
from orchestrator.git_ops import WIP_SAFETY_COMMIT_PREFIXES

_MARK_STEP_DONE = 'mcp__plan-tools__mark_step_done'

#: The roles whose commits the harness attributes to plan steps.
_HISTORY_WRITER_ROLES = frozenset({'implementer', 'debugger', 'simple_task'})

_WRITER_CONTRACT = SpliceContract(
    constant_name='BRANCH_HISTORY_GUIDANCE',
    constant=BRANCH_HISTORY_GUIDANCE,
    roles=_HISTORY_WRITER_ROLES,
    role_set_name='_HISTORY_WRITER_ROLES',
    capability=lambda role: _MARK_STEP_DONE in role.allowed_tools,
    capability_description=(
        "a mark_step_done grant (the harness attributes this role's commits to plan steps)"
    ),
)


class TestWriterSectionSplice:
    """The writer section reaches every step-attributed committer, once."""

    def test_guidance_is_nonempty(self) -> None:
        assert_nonempty(
            'BRANCH_HISTORY_GUIDANCE',
            BRANCH_HISTORY_GUIDANCE,
            remedy='Restore the writer section in agents/branch_history_guidance.py.',
        )

    def test_guidance_opens_its_own_section(self) -> None:
        assert_opens_own_section(
            'BRANCH_HISTORY_GUIDANCE',
            BRANCH_HISTORY_GUIDANCE,
            remedy=(
                'Spliced after a role\'s staging block, it would read as more of '
                'the staging rules. Open it with "\\n## " + BRANCH_HISTORY_HEADING.'
            ),
        )

    def test_the_heading_names_the_section(self) -> None:
        assert_nonempty(
            'BRANCH_HISTORY_HEADING',
            BRANCH_HISTORY_HEADING,
            remedy='The amender prompt points at the section by this heading; restore it.',
        )
        assert BRANCH_HISTORY_HEADING in BRANCH_HISTORY_GUIDANCE, (
            'BRANCH_HISTORY_GUIDANCE does not carry BRANCH_HISTORY_HEADING, so the '
            'amender prompt\'s pointer names a section no role prompt has. Build '
            'the section\'s heading line from the constant.'
        )

    def test_role_set_matches_the_mark_step_done_grant(self) -> None:
        _WRITER_CONTRACT.assert_role_set_matches_capability(
            remedy=(
                'A role newly granted mark_step_done has its commits attributed '
                'to plan steps, so a rewrite of its history orphans a recorded '
                'sha: add it to _HISTORY_WRITER_ROLES and splice '
                'BRANCH_HISTORY_GUIDANCE into its prompt. A role that lost the '
                'grant: remove it from the set and drop its splice.'
            ),
        )

    def test_every_writer_role_carries_the_guidance(self) -> None:
        _WRITER_CONTRACT.assert_every_role_carries(
            remedy=(
                'Splice BRANCH_HISTORY_GUIDANCE into the role in roles.py, '
                'immediately before its scope-boundary splice in the '
                '"## CRITICAL: Git Staging Rules" section.'
            ),
        )

    def test_no_other_role_carries_the_guidance(self) -> None:
        _WRITER_CONTRACT.assert_no_other_role_carries(
            remedy=(
                'Remove the splice. A reviewer gets REVIEWER_BRANCH_HISTORY_RULE '
                'in its contract instead; other roles make no step-attributed '
                'commits.'
            ),
        )

    def test_the_guidance_is_spliced_exactly_once(self) -> None:
        # absent_ok=True leaves a missing splice to fail only the containment
        # test above: one root cause, one failing test.
        _WRITER_CONTRACT.assert_spliced_exactly_once(
            absent_ok=True,
            remedy='Keep exactly one BRANCH_HISTORY_GUIDANCE term per writer role.',
        )


#: Both reviewer generations; they share the name ``reviewer_comprehensive``,
#: and only the DIFF_ONLY one is in ROLES.
_REVIEWER_ROLES = pytest.mark.parametrize(
    'role',
    [REVIEWER_COMPREHENSIVE, REVIEWER_COMPREHENSIVE_CONTEXT_COMPLETE],
    ids=['diff_only', 'context_complete'],
)

_REVIEWER_ROLE_NAMES = frozenset(
    {REVIEWER_COMPREHENSIVE.name, REVIEWER_COMPREHENSIVE_CONTEXT_COMPLETE.name}
)


class TestReviewerRule:
    """The reviewer rule sits in the contract of both reviewer generations."""

    def test_rule_is_nonempty(self) -> None:
        assert_nonempty(
            'REVIEWER_BRANCH_HISTORY_RULE',
            REVIEWER_BRANCH_HISTORY_RULE,
            remedy='Restore the reviewer section in agents/branch_history_guidance.py.',
        )

    def test_rule_opens_the_named_section(self) -> None:
        assert_opens_own_section(
            'REVIEWER_BRANCH_HISTORY_RULE',
            REVIEWER_BRANCH_HISTORY_RULE,
            remedy=(
                'Appended after REVIEWER_INV12_BLOCKERS, it would read as more of '
                'them. Open it with "\\n## " + BRANCH_HISTORY_HEADING.'
            ),
        )
        assert BRANCH_HISTORY_HEADING in REVIEWER_BRANCH_HISTORY_RULE, (
            'REVIEWER_BRANCH_HISTORY_RULE does not carry BRANCH_HISTORY_HEADING, '
            'so the reviewer and the writer roles name the rule differently. '
            'Build its heading line from the constant.'
        )

    @_REVIEWER_ROLES
    def test_rule_sits_in_the_contract_a_pinned_artifact_cannot_replace(self, role) -> None:
        spec = role.prompt_spec
        assert spec is not None, f'{role.name} has no prompt_spec to carry the rule.'

        assert REVIEWER_BRANCH_HISTORY_RULE in spec.contract, (
            'REVIEWER_BRANCH_HISTORY_RULE is missing from the reviewer contract. '
            'Append it after REVIEWER_INV12_BLOCKERS in build_reviewer_prompt_spec.'
        )
        assert REVIEWER_BRANCH_HISTORY_RULE not in spec.baseline_heuristics, (
            'REVIEWER_BRANCH_HISTORY_RULE is in the heuristics half, which a pinned '
            'prompt artifact replaces. It is policy: keep it in the contract only.'
        )

    @pytest.mark.parametrize('prefix', WIP_SAFETY_COMMIT_PREFIXES)
    def test_rule_names_each_subject_the_harness_mints(self, prefix: str) -> None:
        # Referential integrity, not a wording pin: the subject the reviewer is
        # told to recognise is the one git_ops mints its WIP commits with.
        assert prefix.strip() in REVIEWER_BRANCH_HISTORY_RULE, (
            f'REVIEWER_BRANCH_HISTORY_RULE does not name the WIP subject prefix '
            f'{prefix.strip()!r}. Render the subject list from '
            'git_ops.WIP_SAFETY_COMMIT_PREFIXES rather than retyping it.'
        )

    def test_no_non_reviewer_role_carries_the_rule(self) -> None:
        offenders = sorted(
            name
            for name, role in ROLES.items()
            if name not in _REVIEWER_ROLE_NAMES
            and REVIEWER_BRANCH_HISTORY_RULE in role.system_prompt
        )
        assert offenders == [], (
            f'Roles carrying REVIEWER_BRANCH_HISTORY_RULE without being a '
            f'reviewer: {offenders}. A writer role gets BRANCH_HISTORY_GUIDANCE '
            'instead; remove the splice.'
        )

    @_REVIEWER_ROLES
    def test_no_reviewer_carries_the_writer_section(self, role) -> None:
        assert BRANCH_HISTORY_GUIDANCE not in role.system_prompt, (
            f'{role.name} carries BRANCH_HISTORY_GUIDANCE, the writer section. '
            'A reviewer gets only REVIEWER_BRANCH_HISTORY_RULE; remove the splice.'
        )
