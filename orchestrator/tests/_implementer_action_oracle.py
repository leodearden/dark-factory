"""The pinned whole-plan implementer Action paragraph.

Change it together with ``briefing.IMPLEMENTER_ACTIONS`` only when the prompt
change is intended. It is a LITERAL on purpose: the pin needs an oracle
independent of the code under change, so never derive it from production code.
"""

from __future__ import annotations

WHOLE_PLAN_ACTION_PIN = (
    'Execute the next pending steps in TDD order. Commit after each step. '
    'Call `mark_step_done(step_id, commit_sha)` to record progress. '
    'Stop at a logical boundary.\n'
)
ACTION_MARKER = '# Action\n\n'
