"""Role-prompt rules for a change that a detector (type checker, repo guard, ratchet) objects to.

`orchestrator/src/orchestrator/agents/roles.py::IMPLEMENTER` appends
`IMPLEMENTER_SUPPRESSION_RULES`. `orchestrator/src/orchestrator/agents/roles.py::STEWARD`
splices `STEWARD_GUARD_INTENT_RULE` into its Rules list as Rule 8, directly after
`orchestrator/src/orchestrator/agents/merge_conflict_guidance.py::STEWARD_MERGE_CONFLICT_RULE`,
and `STEWARD_SUPPRESSION_RULES` right after it.
`orchestrator/src/orchestrator/agents/roles.py::REVIEWER_INV12_BLOCKERS` names
`TYPE_ESCAPES` as a blocker. It is a leaf module, imported downward by roles.py.
The suppression policy is D15 of `plans/inv12-exceptions-owned-or-ratified-prd.md`.

`orchestrator/tests/test_roles_inv12_blocks.py` and
`orchestrator/tests/test_roles_guard_intent_rule.py` check the splices, never
the wording.
"""

from shared.governed_exceptions import DECLARATION_FORMS, INLINE_MARKER_FORMS

SUPPRESSION_SECTION_TITLE: str = 'Silencing a detector'

TYPE_ESCAPES: str = 'Any or cast'
"""The type escapes that must never stand in for a scoped ignore."""

SUPPRESSION_RULES: str = (
    '\n## ' + SUPPRESSION_SECTION_TITLE + """ (INV-12: exceptions are owned or ratified)

Every entry that silences a detector carries a disposition: a named owner who
will remove it, or a recorded operator ruling that it stays. Accepted inline
forms (a trailing comment on the suppressed line):

"""
    + '\n'.join(f'    {form}' for form in INLINE_MARKER_FORMS)
    + """

Accepted forms in a governed-list declaration:

"""
    + '\n'.join(f'    {form}' for form in DECLARATION_FORMS)
    + """

- Fix the type before suppressing it, and never trade an ignore for """ + TYPE_ESCAPES + """.
- File one follow-up ticket per branch and cite it on every marker you add.
- Never cite your own task as the owner of a marker.
- Edit `docs/legibility/exception-ratifications.yaml` only when your brief orders
  it, and quote that order in the commit message"""
)
"""The rules the implementer and the steward share. It stops inside its last
sentence; each role's constant below completes that sentence."""

IMPLEMENTER_SUPPRESSION_RULES: str = SUPPRESSION_RULES + """: the reviewer sees your
  commits, not your brief.
"""

STEWARD_SUPPRESSION_RULES: str = SUPPRESSION_RULES + """. You have no brief, so never
  edit that file. Your changes in the merge phase land without review, so hold
  yourself to the rules a reviewer would enforce.
"""

STEWARD_GUARD_INTENT_RULE: str = f"""\
8. **A repo guard or ratchet fails in the merge phase** (failed post-merge tests, a failed
   fast-forward, a conflict the merger declined). Meet the guard's intent, not just its
   letter:
   - Do not give up a guarantee the task designed (for example, splitting a staged
     two-step writer into separate writes).
   - Do not add a row for the task's own new code to a guard's allowlist, exemption list
     or other governed list. A scoped inline ignore is not such a row: when fixing the
     code is not possible, add one as "{SUPPRESSION_SECTION_TITLE}" below directs.
   - Do not rewrite code into a form the check cannot see.
   Re-keying, removing or lowering an existing exemption that main's change made stale is
   fine. If none of this is possible, do not land the change and do not call
   `resolve_issue`: re-escalate via `escalate_blocker(..., level=1)`, naming the guard,
   quoting its output and stating the guarantee at stake.
"""
