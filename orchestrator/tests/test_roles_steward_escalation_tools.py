"""Which escalation-store tools the steward role holds (task 3176)."""

from __future__ import annotations

from orchestrator.agents.roles import ROLES, STEWARD

_HISTORY_READ = 'mcp__escalation__get_task_escalation_history'
_DECLARE_PIN = 'mcp__escalation__declare_pin'


def test_steward_holds_the_ever_filed_history_read() -> None:
    """Without it the steward's only task-scoped read is the pending-only
    get_pending_escalations, whose [] never means "none ever filed"."""
    assert _HISTORY_READ in STEWARD.allowed_tools, (
        f'{_HISTORY_READ} must be in STEWARD.allowed_tools; '
        f'current list: {STEWARD.allowed_tools}'
    )


def test_steward_holds_declare_pin() -> None:
    assert _DECLARE_PIN in STEWARD.allowed_tools, (
        f'{_DECLARE_PIN} must be in STEWARD.allowed_tools; '
        f'current list: {STEWARD.allowed_tools}'
    )


def test_steward_is_the_only_role_holding_declare_pin() -> None:
    holders = {
        name for name, role in ROLES.items() if _DECLARE_PIN in role.allowed_tools
    }
    assert holders == {'steward'}, (
        f'{_DECLARE_PIN} must be held by the steward role only; '
        f'extra holders: {sorted(holders - {"steward"})}, '
        f'steward holds it: {"steward" in holders}'
    )
