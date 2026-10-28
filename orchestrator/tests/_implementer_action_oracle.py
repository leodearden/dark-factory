"""The whole-plan implementer Action paragraph as it rendered before task 5385.

A LITERAL on purpose: "byte-identical to today" needs an oracle independent of
the code under change, so never derive it from production code.
"""

from __future__ import annotations

TODAYS_ACTION = (
    'Execute the next pending steps in TDD order. Commit after each step. '
    'Call `mark_step_done(step_id, commit_sha)` to record progress. '
    'Stop at a logical boundary.\n'
)
ACTION_MARKER = '# Action\n\n'
