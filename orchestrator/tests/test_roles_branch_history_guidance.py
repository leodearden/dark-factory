"""Guards for where ``agents/branch_history_guidance.py``'s constants reach.

What this module guards: the writer section ``BRANCH_HISTORY_GUIDANCE`` reaches
exactly the roles whose commits the harness attributes to plan steps (derived
from the ``mark_step_done`` grant), once each, and its heading constant
``BRANCH_HISTORY_HEADING`` is the one the section opens with.

Every assertion is a containment, count or equality check against a NAMED live
constant: no regex over prose and no literal wording pin. The standing rule and
its rationale live in ``orchestrator/tests/_role_splice_contract.py``.
"""

from __future__ import annotations

from _role_splice_contract import SpliceContract, assert_nonempty, assert_opens_own_section

from orchestrator.agents.branch_history_guidance import (
    BRANCH_HISTORY_GUIDANCE,
    BRANCH_HISTORY_HEADING,
)

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
