"""Role-prompt rules for a change that a detector (type checker, repo guard, ratchet) objects to.

`orchestrator/src/orchestrator/agents/roles.py::IMPLEMENTER` appends
`IMPLEMENTER_SUPPRESSION_RULES` and `orchestrator/src/orchestrator/agents/roles.py::STEWARD`
splices `STEWARD_SUPPRESSION_RULES` after its Rules list. It is a leaf module,
imported downward by roles.py. The policy is D15 of
`plans/inv12-exceptions-owned-or-ratified-prd.md`.

`orchestrator/tests/test_roles_inv12_blocks.py` checks the splices, never the
wording.
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
