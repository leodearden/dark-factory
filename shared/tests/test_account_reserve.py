"""shared.account_reserve: the reserve rule, and the accounts file's ``reserve:`` block.

Contract: plans/allowance-reserve-by-account-prd.md §6.1, §6.2 and §6.6. Everything goes through
the public names.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, time, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from shared.account_reserve import (
    ActiveReserve,
    InvalidReserve,
    ReserveConfig,
    ReserveEvent,
    ReserveRefusal,
    ReserveState,
    ReserveWaitEnd,
    ScheduleDrift,
    WeeklyReset,
    WorkClass,
    WorkKind,
    admits,
    config_event_details,
    first_choice,
    parse_reserve,
    refusal,
    refuses_account,
    resolve_work_class,
    schedule_drift,
    served_details,
    wait_end_for_lease,
    wait_end_without_lease,
    wait_ended_details,
    wait_started_details,
)
from shared.config_models import UsageCapConfig


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

    @pytest.mark.parametrize(
        'off_the_minute', [time(21, 0, 30), time(21, 0, 0, 1)], ids=['second', 'microsecond']
    )
    def test_a_reset_off_the_whole_minute_is_refused(self, off_the_minute: time) -> None:
        with pytest.raises(ValueError, match=re.escape(repr(off_the_minute))):
            WeeklyReset(0, off_the_minute)


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

    @pytest.mark.parametrize('misspelling', ['defaults', 'Default'])
    def test_a_misspelled_default_is_read_as_a_project_and_the_default_stands(
        self, misspelling: str
    ) -> None:
        reserve = parse_reserve(prd_block(admit_tiers={misspelling: ['critical']}), ROSTER)

        assert isinstance(reserve, ActiveReserve)
        assert reserve.config.admit_tiers.for_project(None) == ('critical', 'high')
        assert reserve.config.admit_tiers.for_project(misspelling) == ('critical',)


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


ROSTER = ['max-d', 'max-e', 'max-f', 'max-g', 'max-b', 'max-c', 'max-h']


class TestParseReserve:
    def test_the_prd_block_is_an_active_reserve(self) -> None:
        reserve = parse_reserve(prd_block(), ROSTER)

        assert isinstance(reserve, ActiveReserve)
        assert reserve.state is ReserveState.ACTIVE
        assert reserve.config.account == 'max-h'

    @pytest.mark.parametrize(
        ('block', 'cause'),
        [
            pytest.param(prd_block(colour='red'), 'colour', id='unknown-key'),
            pytest.param(prd_block(account='max-z'), 'max-z', id='unlisted-account'),
            pytest.param(prd_block(release_hours=0), 'release_hours', id='release-hours-0'),
            pytest.param(prd_block(release_hours=168), 'release_hours', id='release-hours-168'),
            pytest.param(
                prd_block(weekly_reset_utc='funday 21:00'), 'weekly_reset_utc', id='unknown-day'
            ),
            pytest.param(
                prd_block(weekly_reset_utc='mon 25:00'), 'weekly_reset_utc', id='hour-25'
            ),
            pytest.param(
                prd_block(weekly_reset_utc='mon 21:60'), 'weekly_reset_utc', id='minute-60'
            ),
            pytest.param(prd_block(weekly_reset_utc='mon'), 'weekly_reset_utc', id='no-time'),
            pytest.param(prd_block(weekly_reset_utc='21:00'), 'weekly_reset_utc', id='no-day'),
            pytest.param(
                prd_block(admit_tiers={'dark_factory': 'high'}),
                'admit_tiers.dark_factory',
                id='project-tiers-as-text',
            ),
            pytest.param(prd_block(admit_tiers=None), 'admit_tiers', id='admit-tiers-null'),
            pytest.param(prd_block(admit_tiers=['high']), 'admit_tiers', id='admit-tiers-list'),
            pytest.param(None, 'reserve: ', id='empty-block'),
            pytest.param([prd_block()], 'reserve: ', id='list-block'),
        ],
    )
    def test_a_bad_block_is_an_invalid_reserve_naming_its_cause(
        self, block: object, cause: str
    ) -> None:
        reserve = parse_reserve(block, ROSTER)

        assert isinstance(reserve, InvalidReserve)
        assert reserve.state is ReserveState.INVALID
        assert cause in reserve.reason

    @pytest.mark.parametrize('reason', ['', '   '])
    def test_an_invalid_reserve_always_carries_its_reason(self, reason: str) -> None:
        with pytest.raises(ValueError, match='reason'):
            InvalidReserve(reason)


def write_accounts_file(path: Path, **beside_accounts: object) -> str:
    """An accounts file listing ROSTER, with any other top-level keys (``reserve=...``) beside it."""
    accounts = [{'name': name, 'oauth_token_env': f'TOKEN_{name.upper()}'} for name in ROSTER]
    path.write_text(yaml.safe_dump({'accounts': accounts, **beside_accounts}))
    return str(path)


class TestAccountsFileLoader:
    def test_without_a_block_there_is_no_reserve_and_the_dump_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / 'accounts.yaml'
        without_block = UsageCapConfig(accounts_file=write_accounts_file(path))
        with_block = UsageCapConfig(accounts_file=write_accounts_file(path, reserve=prd_block()))

        assert without_block.reserve is None
        assert [account.name for account in without_block.accounts] == ROSTER
        assert without_block.model_dump() == with_block.model_dump()

    def test_seen_from_wednesday_the_prd_block_releases_on_sunday_morning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            config = UsageCapConfig(
                accounts_file=write_accounts_file(tmp_path / 'accounts.yaml', reserve=prd_block())
            )

        assert caplog.records == []
        assert isinstance(config.reserve, ActiveReserve)
        assert config.reserve.config.release_at(utc(2026, 11, 18, 12)) == utc(2026, 11, 22, 9)

    @pytest.mark.parametrize(
        ('block', 'cause'),
        [
            pytest.param(prd_block(account='max-z'), 'max-z', id='unlisted-account'),
            pytest.param(prd_block(release_hours=0), 'release_hours', id='release-hours-0'),
            pytest.param(None, 'reserve: ', id='key-without-body'),
        ],
    )
    def test_an_invalid_block_loads_as_an_invalid_reserve_with_a_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        block: object,
        cause: str,
    ) -> None:
        path = write_accounts_file(tmp_path / 'accounts.yaml', reserve=block)

        with caplog.at_level(logging.WARNING):
            config = UsageCapConfig(accounts_file=path)

        assert isinstance(config.reserve, InvalidReserve)
        assert cause in config.reserve.reason
        assert [account.name for account in config.accounts] == ROSTER
        warnings = [record.getMessage() for record in caplog.records]
        assert len(warnings) == 1
        assert path in warnings[0]
        assert config.reserve.reason in warnings[0]

    def test_the_reload_round_trip_reads_the_reserve_from_the_file_again(
        self, tmp_path: Path
    ) -> None:
        config = UsageCapConfig(
            accounts_file=write_accounts_file(tmp_path / 'accounts.yaml', reserve=prd_block())
        )

        reloaded = UsageCapConfig.model_validate(config.model_dump())

        assert reloaded == config
        assert reloaded.reserve == config.reserve

    @pytest.mark.parametrize('beside_a_file', [False, True], ids=['no-file', 'file-without-block'])
    def test_an_inline_reserve_is_ignored_with_a_warning_naming_it(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, beside_a_file: bool
    ) -> None:
        inline = {'account': 'max-h'}
        data: dict[str, object] = {'reserve': inline}
        if beside_a_file:
            data['accounts_file'] = write_accounts_file(tmp_path / 'accounts.yaml')

        with caplog.at_level(logging.WARNING):
            config = UsageCapConfig.model_validate(data)

        assert config.reserve is None
        warnings = [record.getMessage() for record in caplog.records]
        assert len(warnings) == 1
        assert 'usage_cap.reserve' in warnings[0]
        assert repr(inline) in warnings[0]

    def test_no_project_config_gains_a_reserve_key(self) -> None:
        assert 'reserve' not in UsageCapConfig.model_fields

    def test_a_changed_block_is_a_changed_config(self, tmp_path: Path) -> None:
        path = tmp_path / 'accounts.yaml'
        first = UsageCapConfig(accounts_file=write_accounts_file(path, reserve=prd_block()))
        unchanged = UsageCapConfig(accounts_file=write_accounts_file(path, reserve=prd_block()))
        changed = UsageCapConfig(
            accounts_file=write_accounts_file(path, reserve=prd_block(release_hours=24))
        )

        assert unchanged == first
        assert changed != first


GATE_WEDNESDAY = utc(2026, 11, 11, 12)
GATE_RELEASE_AT = utc(2026, 11, 15, 9)
GATE_NEXT_RESET = utc(2026, 11, 16, 21)
GATE_IN_RELEASE = utc(2026, 11, 15, 12)

LOW_TASK = WorkClass(
    kind=WorkKind.TASK, project_id='dark_factory', tier='low', task_id='7', role='implementer'
)
HIGH_TASK = WorkClass(
    kind=WorkKind.TASK, project_id='dark_factory', tier='high', task_id='8', role='implementer'
)


class TestRefusesAccount:
    """Seen from Wed 2026-11-11 12:00 UTC the PRD block releases Sun 11-15 09:00, resets Mon 21:00."""

    @pytest.mark.parametrize(
        'work_class',
        [pytest.param(LOW_TASK, id='task-low'), pytest.param(BACKGROUND, id='background')],
    )
    def test_the_reserve_is_refused_to_an_unadmitted_class_while_protected(
        self, work_class: WorkClass
    ) -> None:
        assert refuses_account(reserve_config(), 'max-h', work_class, GATE_WEDNESDAY) is True

    def test_another_account_is_never_refused(self) -> None:
        assert refuses_account(reserve_config(), 'max-d', LOW_TASK, GATE_WEDNESDAY) is False

    @pytest.mark.parametrize(
        'work_class',
        [pytest.param(HIGH_TASK, id='task-high'), pytest.param(ESSENTIAL, id='essential')],
    )
    def test_an_admitted_class_is_not_refused(self, work_class: WorkClass) -> None:
        assert refuses_account(reserve_config(), 'max-h', work_class, GATE_WEDNESDAY) is False

    def test_nothing_is_refused_in_the_release_window(self) -> None:
        assert refuses_account(reserve_config(), 'max-h', LOW_TASK, GATE_IN_RELEASE) is False


class TestRefusal:
    def test_with_nothing_open_there_is_no_refusal(self) -> None:
        assert refusal(reserve_config(), [], LOW_TASK, GATE_WEDNESDAY) is None

    def test_with_another_account_open_there_is_no_refusal(self) -> None:
        assert refusal(reserve_config(), ['max-d', 'max-h'], LOW_TASK, GATE_WEDNESDAY) is None
        assert refusal(reserve_config(), ['max-d'], LOW_TASK, GATE_WEDNESDAY) is None

    def test_an_admitted_class_is_not_refused(self) -> None:
        assert refusal(reserve_config(), ['max-h'], HIGH_TASK, GATE_WEDNESDAY) is None

    def test_the_release_window_refuses_nobody(self) -> None:
        assert refusal(reserve_config(), ['max-h'], LOW_TASK, GATE_IN_RELEASE) is None

    def test_only_the_reserve_open_and_the_class_refused_is_a_refusal_until_the_release(
        self,
    ) -> None:
        refused = refusal(reserve_config(), ['max-h'], LOW_TASK, GATE_WEDNESDAY)

        assert refused == ReserveRefusal(account='max-h', release_at=GATE_RELEASE_AT)


class TestFirstChoice:
    @pytest.mark.parametrize(
        ('now', 'expected'),
        [
            pytest.param(GATE_WEDNESDAY, None, id='protected'),
            pytest.param(GATE_RELEASE_AT - timedelta(microseconds=1), None, id='before-release'),
            pytest.param(GATE_RELEASE_AT, 'max-h', id='at-release'),
            pytest.param(GATE_IN_RELEASE, 'max-h', id='in-release'),
            pytest.param(
                GATE_NEXT_RESET - timedelta(microseconds=1), 'max-h', id='just-before-reset'
            ),
            pytest.param(GATE_NEXT_RESET, None, id='at-reset'),
        ],
    )
    def test_the_reserve_comes_first_only_in_the_release_window(
        self, now: datetime, expected: str | None
    ) -> None:
        assert first_choice(reserve_config(), now) == expected


class TestWaitEndForLease:
    def test_a_lease_on_another_account_means_an_account_opened(self) -> None:
        assert (
            wait_end_for_lease(reserve_config(), 'max-d', GATE_WEDNESDAY)
            is ReserveWaitEnd.ACCOUNT_OPENED
        )

    def test_a_reserve_lease_in_the_release_window_means_released(self) -> None:
        assert (
            wait_end_for_lease(reserve_config(), 'max-h', GATE_IN_RELEASE)
            is ReserveWaitEnd.RELEASED
        )

    def test_a_reserve_lease_in_the_protected_window_means_admitted(self) -> None:
        assert (
            wait_end_for_lease(reserve_config(), 'max-h', GATE_WEDNESDAY)
            is ReserveWaitEnd.ADMITTED
        )


def test_reserve_wait_end_values() -> None:
    assert {end.value for end in ReserveWaitEnd} == {
        'account_opened',
        'released',
        'admitted',
        'reserve_unusable',
        'deadline',
        'cancelled',
    }


class TestConfigEventDetails:
    def test_an_active_reserve_names_its_calendar_and_tiers(self) -> None:
        details = config_event_details(ActiveReserve(reserve_config()), GATE_WEDNESDAY)

        assert details == {
            'state': 'active',
            'reason': None,
            'account': 'max-h',
            'next_reset': '2026-11-16T21:00:00+00:00',
            'release_at': '2026-11-15T09:00:00+00:00',
            'admit_tiers': {'default': ['critical', 'high']},
        }
        assert json.loads(json.dumps(details)) == details

    def test_a_project_list_appears_under_its_project_id(self) -> None:
        config = reserve_config(admit_tiers={'dark_factory': ['critical', 'high', 'medium']})

        details = config_event_details(ActiveReserve(config), GATE_WEDNESDAY)

        assert details['admit_tiers'] == {
            'default': ['critical', 'high'],
            'dark_factory': ['critical', 'high', 'medium'],
        }

    def test_an_invalid_reserve_gives_its_reason_and_nothing_else(self) -> None:
        details = config_event_details(InvalidReserve('reserve.colour: extra'), GATE_WEDNESDAY)

        assert details == {
            'state': 'invalid',
            'reason': 'reserve.colour: extra',
            'account': None,
            'next_reset': None,
            'release_at': None,
            'admit_tiers': None,
        }


class TestWaitEndWithoutLease:
    """A waiting caller selection could not serve: max-h is the reserve, max-d another account."""

    def test_another_account_open_means_an_account_opened(self) -> None:
        end = wait_end_without_lease(reserve_config(), ['max-d'], [], LOW_TASK, GATE_WEDNESDAY)

        assert end is ReserveWaitEnd.ACCOUNT_OPENED

    @pytest.mark.parametrize(
        'open_accounts',
        [pytest.param([], id='nothing-open'), pytest.param(['max-h'], id='reserve-open')],
    )
    def test_a_refused_caller_stays_while_the_reserve_is_open_or_probing(
        self, open_accounts: list[str]
    ) -> None:
        end = wait_end_without_lease(reserve_config(), open_accounts, [], LOW_TASK, GATE_WEDNESDAY)

        assert end is None

    def test_the_reserve_failed_means_reserve_unusable(self) -> None:
        end = wait_end_without_lease(reserve_config(), [], ['max-h'], LOW_TASK, GATE_WEDNESDAY)

        assert end is ReserveWaitEnd.RESERVE_UNUSABLE

    def test_an_open_account_outranks_a_failed_reserve(self) -> None:
        end = wait_end_without_lease(
            reserve_config(), ['max-d'], ['max-h'], LOW_TASK, GATE_WEDNESDAY
        )

        assert end is ReserveWaitEnd.ACCOUNT_OPENED

    def test_a_caller_no_longer_refused_has_been_admitted_or_released(self) -> None:
        config = reserve_config()

        assert wait_end_without_lease(config, [], [], HIGH_TASK, GATE_WEDNESDAY) is (
            ReserveWaitEnd.ADMITTED
        )
        assert wait_end_without_lease(config, [], [], LOW_TASK, GATE_IN_RELEASE) is (
            ReserveWaitEnd.RELEASED
        )


LOW_TASK_FIELDS: dict[str, object] = {
    'kind': 'task',
    'role': 'implementer',
    'project_id': 'dark_factory',
    'tier': 'low',
    'task_id': '7',
}
REFUSED = ReserveRefusal(account='max-h', release_at=GATE_RELEASE_AT)


class TestEventDetails:
    def test_a_wait_start_names_the_class_the_reserve_and_the_release(self) -> None:
        details = wait_started_details(LOW_TASK, REFUSED, unclassified=False)

        assert details == {
            **LOW_TASK_FIELDS,
            'account': 'max-h',
            'release_at': '2026-11-15T09:00:00+00:00',
            'unclassified': False,
        }
        assert type(details['kind']) is str
        assert json.loads(json.dumps(details)) == details

    def test_an_unclassified_caller_waits_as_background(self) -> None:
        details = wait_started_details(resolve_work_class(None), REFUSED, unclassified=True)

        assert {key: details[key] for key in LOW_TASK_FIELDS} == {
            'kind': 'background',
            'role': None,
            'project_id': None,
            'tier': None,
            'task_id': None,
        }
        assert details['unclassified'] is True

    def test_a_wait_end_adds_the_time_waited_and_the_exit(self) -> None:
        details = wait_ended_details(
            LOW_TASK,
            REFUSED,
            unclassified=False,
            waited_secs=12.345,
            ended_by=ReserveWaitEnd.RELEASED,
        )

        assert details == {
            **wait_started_details(LOW_TASK, REFUSED, unclassified=False),
            'waited_secs': 12.3,
            'ended_by': 'released',
        }
        assert json.loads(json.dumps(details)) == details

    def test_a_served_call_names_the_class_and_its_cost_only(self) -> None:
        assert served_details(LOW_TASK, cost_usd=1.25) == {**LOW_TASK_FIELDS, 'cost_usd': 1.25}
        assert served_details(LOW_TASK, cost_usd=None)['cost_usd'] is None
