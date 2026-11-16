"""shared.account_reserve: the reserve rule, and the accounts file's ``reserve:`` block.

Contract: plans/allowance-reserve-by-account-prd.md §6.1, §6.2 and §6.6. Everything goes through
the public names.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from shared.account_reserve import (
    ReserveConfig,
    ReserveEvent,
    ScheduleDrift,
    WorkClass,
    WorkKind,
    admits,
    resolve_work_class,
    schedule_drift,
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


def reserve_config(**overrides: object) -> ReserveConfig:
    return ReserveConfig.model_validate(prd_block(**overrides))


def utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=UTC)


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


class TestCalendar:
    """PRD §6.2 config; the reference week resets Mon 2026-11-23 21:00 UTC, releasing Sun 09:00."""

    def test_seen_from_wednesday_the_reserve_is_protected(self) -> None:
        config = reserve_config()
        wednesday = utc(2026, 11, 18, 12)

        assert config.next_reset(wednesday) == utc(2026, 11, 23, 21)
        assert config.release_at(wednesday) == utc(2026, 11, 22, 9)
        assert config.in_release_window(wednesday) is False

    def test_the_release_window_opens_at_the_exact_release_instant(self) -> None:
        config = reserve_config()
        release = utc(2026, 11, 22, 9)

        assert config.in_release_window(release) is True
        assert config.in_release_window(release - timedelta(microseconds=1)) is False

    def test_just_before_the_reset_the_window_is_still_open(self) -> None:
        config = reserve_config()
        last_instant = utc(2026, 11, 23, 20, 59, 59, 999999)

        assert config.in_release_window(last_instant) is True
        assert config.next_reset(last_instant) == utc(2026, 11, 23, 21)

    def test_the_protected_window_begins_at_the_exact_reset_instant(self) -> None:
        config = reserve_config()
        reset = utc(2026, 11, 23, 21)

        assert config.next_reset(reset) == utc(2026, 11, 30, 21)
        assert config.release_at(reset) == utc(2026, 11, 29, 9)
        assert config.in_release_window(reset) is False

    def test_a_release_falls_in_the_week_before_its_reset(self) -> None:
        config = reserve_config(weekly_reset_utc='sun 02:00')
        saturday = utc(2026, 11, 28, 12)

        assert config.next_reset(saturday) == utc(2026, 11, 29, 2)
        assert config.release_at(saturday) == utc(2026, 11, 27, 14)
        assert config.in_release_window(saturday) is True

    def test_after_the_reset_the_next_one_is_a_week_on(self) -> None:
        tuesday = utc(2026, 11, 24, 3)

        assert reserve_config().next_reset(tuesday) == utc(2026, 11, 30, 21)

    def test_an_instant_in_another_zone_is_read_as_the_same_instant(self) -> None:
        config = reserve_config()
        india = timezone(timedelta(hours=5, minutes=30))

        seen_from_india = config.next_reset(datetime(2026, 11, 18, 17, 30, tzinfo=india))

        assert seen_from_india == config.next_reset(utc(2026, 11, 18, 12))

    @pytest.mark.parametrize('method', ['next_reset', 'release_at', 'in_release_window'])
    def test_a_naive_instant_is_refused(self, method: str) -> None:
        naive = datetime(2026, 11, 18, 12)

        with pytest.raises(ValueError, match=re.escape(repr(naive))):
            getattr(reserve_config(), method)(naive)


class TestWeeklyResetText:
    @pytest.mark.parametrize('text', ['mon 21:00', 'Mon 21:00', ' MON  21:00 '])
    def test_spellings_of_one_reset_parse_to_one_value(self, text: str) -> None:
        assert (
            reserve_config(weekly_reset_utc=text).weekly_reset_utc
            == reserve_config().weekly_reset_utc
        )

    def test_a_dumped_config_validates_back_to_itself(self) -> None:
        config = reserve_config(weekly_reset_utc='Mon 21:00')

        dumped = config.model_dump()

        assert dumped['weekly_reset_utc'] == 'mon 21:00'
        assert ReserveConfig.model_validate(dumped) == config


class TestAdmitTiers:
    def test_without_admit_tiers_every_project_gets_critical_and_high(self) -> None:
        block = prd_block()
        del block['admit_tiers']
        tiers = ReserveConfig.model_validate(block).admit_tiers

        assert tiers.for_project(None) == ('critical', 'high')
        assert tiers.for_project('reify') == ('critical', 'high')

    def test_a_project_list_applies_to_that_project_only(self) -> None:
        tiers = reserve_config(
            admit_tiers={'dark_factory': ['critical', 'high', 'medium']}
        ).admit_tiers

        assert tiers.for_project('dark_factory') == ('critical', 'high', 'medium')
        assert tiers.for_project('reify') == ('critical', 'high')
        assert tiers.for_project(None) == ('critical', 'high')


PROTECTED = utc(2026, 11, 18, 12)
RELEASE = utc(2026, 11, 22, 10)

ESSENTIAL = WorkClass(kind=WorkKind.ESSENTIAL)
BACKGROUND = WorkClass(kind=WorkKind.BACKGROUND)


def task_work(tier: str | None, project_id: str | None = 'dark_factory') -> WorkClass:
    return WorkClass(kind=WorkKind.TASK, project_id=project_id, tier=tier)


class TestAdmits:
    @pytest.mark.parametrize(
        ('work_class', 'admitted'),
        [
            pytest.param(ESSENTIAL, True, id='essential'),
            pytest.param(BACKGROUND, False, id='background'),
            pytest.param(task_work('critical'), True, id='task-critical'),
            pytest.param(task_work('high'), True, id='task-high'),
            pytest.param(task_work('medium'), False, id='task-medium'),
            pytest.param(task_work('low'), False, id='task-low'),
            pytest.param(task_work(None), False, id='task-without-tier'),
        ],
    )
    def test_protected_window_with_the_default_list(
        self, work_class: WorkClass, admitted: bool
    ) -> None:
        assert admits(reserve_config(), work_class, PROTECTED) is admitted

    @pytest.mark.parametrize(
        ('work_class', 'admitted'),
        [
            pytest.param(task_work('medium', 'dark_factory'), True, id='listed-project'),
            pytest.param(task_work('medium', 'reify'), False, id='other-project'),
            pytest.param(task_work('medium', None), False, id='no-project'),
            pytest.param(task_work(None, 'dark_factory'), True, id='missing-tier-reads-medium'),
        ],
    )
    def test_protected_window_with_a_project_list(
        self, work_class: WorkClass, admitted: bool
    ) -> None:
        config = reserve_config(admit_tiers={'dark_factory': ['critical', 'high', 'medium']})

        assert admits(config, work_class, PROTECTED) is admitted

    def test_an_empty_default_list_still_admits_essential_work(self) -> None:
        config = reserve_config(admit_tiers={'default': []})

        assert admits(config, task_work('high'), PROTECTED) is False
        assert admits(config, ESSENTIAL, PROTECTED) is True

    @pytest.mark.parametrize(
        'work_class',
        [
            pytest.param(BACKGROUND, id='background'),
            pytest.param(task_work('low'), id='task-low'),
            pytest.param(task_work(None), id='task-without-tier'),
            pytest.param(ESSENTIAL, id='essential'),
        ],
    )
    def test_release_window_admits_everything(self, work_class: WorkClass) -> None:
        assert admits(reserve_config(), work_class, RELEASE) is True

    def test_the_kind_decides_not_the_role(self) -> None:
        steward_on_a_low_task = WorkClass(
            kind=WorkKind.TASK,
            project_id='dark_factory',
            tier='low',
            task_id='42',
            role='steward',
        )

        assert admits(reserve_config(), steward_on_a_low_task, PROTECTED) is False


class TestScheduleDrift:
    """PRD §6.2 config; the configured reset nearest these banners is Mon 2026-11-23 21:00 UTC."""

    @pytest.mark.parametrize(
        'observed',
        [
            pytest.param(utc(2026, 11, 23, 22), id='an-hour-late'),
            pytest.param(utc(2026, 11, 23, 20), id='an-hour-early'),
            pytest.param(utc(2026, 11, 24, 0), id='exactly-three-hours-late'),
        ],
    )
    def test_a_banner_off_by_more_than_half_an_hour_is_drift(self, observed: datetime) -> None:
        assert schedule_drift(reserve_config(), observed) == ScheduleDrift(
            configured_reset=utc(2026, 11, 23, 21), observed_reset=observed
        )

    @pytest.mark.parametrize(
        'observed',
        [
            pytest.param(utc(2026, 11, 23, 21), id='on-time'),
            pytest.param(utc(2026, 11, 23, 21, 10), id='ten-minutes-late'),
            pytest.param(utc(2026, 11, 23, 21, 30), id='exactly-half-an-hour-late'),
            pytest.param(utc(2026, 11, 24, 2), id='five-hours-late'),
        ],
    )
    def test_a_banner_near_the_reset_or_far_from_it_is_not_drift(
        self, observed: datetime
    ) -> None:
        assert schedule_drift(reserve_config(), observed) is None

    def test_drift_is_measured_against_the_nearest_reset_in_any_week(self) -> None:
        drift = schedule_drift(reserve_config(), utc(2026, 11, 16, 19, 30))

        assert drift is not None
        assert drift.configured_reset == utc(2026, 11, 16, 21)

    def test_a_naive_banner_time_is_refused(self) -> None:
        with pytest.raises(ValueError, match='timezone-aware'):
            schedule_drift(reserve_config(), datetime(2026, 11, 23, 22))
