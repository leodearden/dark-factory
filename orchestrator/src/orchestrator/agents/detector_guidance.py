"""Role-prompt rules for a change that a detector (type checker, repo guard, ratchet) objects to.

`orchestrator/src/orchestrator/agents/roles.py::IMPLEMENTER` appends
`IMPLEMENTER_SUPPRESSION_RULES`. `orchestrator/src/orchestrator/agents/roles.py::STEWARD`
splices `STEWARD_GUARD_INTENT_RULE` into its Rules list as Rule 8, directly after
`orchestrator/src/orchestrator/agents/merge_conflict_guidance.py::STEWARD_MERGE_CONFLICT_RULE`,
and `STEWARD_SUPPRESSION_RULES` right after it. It is a leaf module, imported
downward by roles.py. The suppression policy is D15 of
`plans/inv12-exceptions-owned-or-ratified-prd.md`.

`orchestrator/tests/test_roles_inv12_blocks.py` and
`orchestrator/tests/test_roles_guard_intent_rule.py` check the splices, never
the wording.
"""

from shared.governed_exceptions import DECLARATION_FORMS, INLINE_MARKER_FORMS

SUPPRESSION_RULES: str = (
    """
## Silencing a detector (INV-12: exceptions are owned or ratified)

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

- Fix the type before suppressing it, and never trade an ignore for Any or cast.
- File one follow-up ticket per branch and cite it on every marker you add.
- Never cite your own task as the owner of a marker.
- Edit `docs/legibility/exception-ratifications.yaml` only when your brief orders
  it, and quote that order in the commit message"""
)
"""The rules every role shares. It stops inside its last sentence; each role's
constant below completes that sentence."""

IMPLEMENTER_SUPPRESSION_RULES: str = SUPPRESSION_RULES + """: the reviewer sees your
  commits, not your brief.
"""

STEWARD_SUPPRESSION_RULES: str = SUPPRESSION_RULES + """. Your changes in the merge
  phase land without review, so hold yourself to the rules a reviewer would enforce.
"""

STEWARD_GUARD_INTENT_RULE: str = """\
8. **A repo guard or ratchet fails in the merge phase** (failed post-merge tests, a failed
   fast-forward, a conflict the merger declined). Your changes here land without review, so
   meet the guard's intent, not just its letter:
   - Do not give up a guarantee the task designed (for example, splitting a staged
     two-step writer into separate writes).
   - Do not add an exemption or allowlist row for the task's own new code.
   - Do not rewrite code into a form the check cannot see.
   Re-keying, removing or lowering an existing exemption that main's change made stale is
   fine. Any suppression you do add follows "Silencing a detector" below. If none of this
   is possible, do not land the change and do not call `resolve_issue`: re-escalate via
   `escalate_blocker(..., level=1)`, naming the guard, quoting its output and stating the
   guarantee at stake.
"""
