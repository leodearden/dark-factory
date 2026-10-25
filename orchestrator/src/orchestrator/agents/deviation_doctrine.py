"""Role-prompt guidance: what the IMPLEMENTER does when a plan step's text proves wrong.

Renders rung 2 of the amendment-pass precedent ladder, as Leo ratified it on
2026-09-21. `orchestrator/src/orchestrator/agents/roles.py` splices the render
into IMPLEMENTER as `IMPLEMENTER_DEVIATION_DOCTRINE`. It is a leaf module,
imported downward by roles.py, only so that already oversized file does not
grow further.

THIS TEXT IS THE NORMATIVE STATEMENT OF THE DOCTRINE, and code wins over the
memory ladder: a rung that contradicts it is a task against this module, not a
memory write.

`orchestrator/tests/test_roles_deviation_doctrine.py` checks the render and its
splice, never its wording.
"""

from shared.plan_deviation import DEVIATION_KINDS


def render_deviation_doctrine(record_tool: str) -> str:
    """The doctrine, naming ``record_tool`` as the per-step record channel."""
    # Sorted: frozenset iteration order follows the string hash seed, so an
    # unsorted render would give different prompt bytes in different processes.
    kinds = ', '.join('`' + kind + '`' for kind in sorted(DEVIATION_KINDS))
    return (
        """
## When a plan step's text proves wrong

Never edit plan text: not `plan.json`, and not a step's description.

When a step's literal text conflicts with what its stated rationale requires,
and the rationale is clear, apply what the rationale requires, in full.
Self-verify by running the tests the step names. Record the deviation, and
repeat your reasoning in the commit message.

When the rationale itself is wrong or absent, or a frozen design decision is in
conflict, stop and call `escalate_blocker`.

Record each deviation on exactly one of two channels, never both for one fact:

- A deviation attributable to one step: `"""
        + record_tool
        + """(step_id, kind, observed, expected, action_taken, prose_only)`.
  `kind` is one of """
        + kinds
        + """.
  Pass `prose_only=True` when the corrected claim is destined for a docstring,
  a CHANGELOG or a SKILL.md.
- A deviation not attributable to a single step:
  `escalate_info(..., note_to_file=True)`.

Nothing downstream compares your diff to the plan text: no reviewer is given
the plan, and the completion judge is idle. So the record is not for a
reviewer. A nightly census counts these records in aggregate, so write each
one to stand alone without your session.
"""
    )
