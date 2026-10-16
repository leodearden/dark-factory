"""The next action a ``preexisting_main_break`` escalation carries (task 3774).

``MainHealthNextAction`` is the closed vocabulary of those actions.
Invariant: no member asks the reader to wait. The retired
``'await_preexisting_main_hotfix'`` did exactly that on paths where nobody
was producing a fix.

The values are human-facing labels, rendered as "Suggested action" in
briefings and the dashboard; they are not a parsed protocol.
escalation-watcher-auto matches on category and summary, never on
``suggested_action``.

This is a leaf module so that both ``merge_lane/worker.py`` and
``workflow.py`` can import it at module level: worker imports workflow, so
the vocabulary cannot live in either.
"""

from __future__ import annotations

from enum import StrEnum


class MainHealthNextAction(StrEnum):
    AUTO_HEAL_IN_FLIGHT = 'main_health_auto_heal_in_flight'
    """An auto-spawned HIGH-lane fix task owns the fix; this escalation owns the normal-lane halt."""

    NO_FIX_OWNER = 'file_main_fix_or_repair_verify_env'
    """No automated fix exists. Reproduce on a clean environment: if bare main is red, file a fix-main task; if it is green, repair the verify environment."""

    AUTO_HEAL_EXHAUSTED = 'investigate_main_rebreak_after_auto_heal'
    """The auto-heal fix already ran for this signature and the break recurred; automation has stopped."""
