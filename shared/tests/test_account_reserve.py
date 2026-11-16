"""shared.account_reserve: the reserve rule, and the accounts file's ``reserve:`` block.

Contract: plans/allowance-reserve-by-account-prd.md §6.1, §6.2 and §6.6. Everything goes through
the public names.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from shared.account_reserve import (
    ReserveEvent,
    WorkClass,
    WorkKind,
    resolve_work_class,
)


def prd_block(**overrides: object) -> dict[str, object]:
    """The ``reserve:`` block of PRD §6.2, with any key replaced or added."""
    block: dict[str, object] = {
        'account': 'max-h',
        'weekly_reset_utc': 'mon 21:00',
        'release_hours': 36,
        'admit_tiers': {'default': ['critical', 'high']},
    }
    block.update(overrides)
    return block


class TestResolveWorkClass:
    def test_resolve_returns_a_given_work_class_unchanged(self) -> None:
        work_class = WorkClass(
            kind=WorkKind.TASK,
            project_id='dark_factory',
            tier='high',
            task_id='7',
            role='implementer',
        )

        assert resolve_work_class(work_class) is work_class

    def test_resolve_calls_a_function_at_every_resolve(self) -> None:
        tiers = ['low', 'high']

        def current() -> WorkClass:
            return WorkClass(kind=WorkKind.TASK, project_id='dark_factory', tier=tiers.pop(0))

        first = resolve_work_class(current)
        second = resolve_work_class(current)

        assert first.tier == 'low'
        assert second.tier == 'high'

    def test_resolve_treats_none_as_background(self) -> None:
        assert resolve_work_class(None) == WorkClass(kind=WorkKind.BACKGROUND)

    def test_resolve_rejects_a_function_returning_something_else(self) -> None:
        returns_a_tier_name = Mock(return_value='high')

        with pytest.raises(TypeError, match="'high'"):
            resolve_work_class(returns_a_tier_name)


def test_reserve_event_names() -> None:
    assert {event.value for event in ReserveEvent} == {
        'reserve_config',
        'reserve_wait_started',
        'reserve_wait_ended',
        'reserve_served',
        'reserve_schedule_drift',
        'reserve_account_unavailable',
    }
