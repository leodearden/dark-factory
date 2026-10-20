"""The closed vocabulary of next actions a ``preexisting_main_break`` escalation carries.

Invariant: no member asks the reader to wait for a fix nothing produces.
"""

from __future__ import annotations

from enum import StrEnum


class MainHealthNextAction(StrEnum):
    AUTO_HEAL_IN_FLIGHT = 'main_health_auto_heal_in_flight'
    """An auto-spawned HIGH-lane fix task owns the fix; this escalation owns the normal-lane halt."""

    FIX_TASK_FILED = 'main_fix_task_owns_fix'
    """A fix-main task named in the detail owns the fix through plan/implement/review; it reproduces on a clean main first."""

    NO_FIX_OWNER = 'file_main_fix_or_repair_verify_env'
    """No fix-main task could be filed or adopted (no task store wired, or the lookup or filing failed). Reproduce on a clean environment: if bare main is red, file a fix-main task; if it is green, repair the verify environment."""

    AUTO_HEAL_EXHAUSTED = 'investigate_main_rebreak_after_auto_heal'
    """The auto-heal fix already ran for this signature and the break recurred; automation has stopped."""
