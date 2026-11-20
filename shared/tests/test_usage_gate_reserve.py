"""The usage gate applies the allowance reserve: selection, the refusal query and the readers.

Contract: plans/allowance-reserve-by-account-prd.md §6.3, §6.4 and §6.6; the B-numbers are its §7
boundary tests. Every test drives a real UsageGate built from a real accounts file, on a fake
reserve clock, and changes account phases only through public seams.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from _usage_gate_test_helpers import (
    SCOPE,
    make_mock_cost_store,
    ok_result,
    scripted_invoke,
)

from shared.account_reserve import (
    ReserveRefusal,
    ReserveState,
    WorkClass,
    WorkClassSource,
    WorkKind,
)
from shared.cli_invoke import AllAccountsCappedException, invoke_with_cap_retry
from shared.config_models import UsageCapConfig
from shared.cost_store import CostStore
from shared.invocation_outcome import OK, AuthFailed, CapHit
from shared.testing import make_gate_mock
from shared.testing_virtual_clock import virtual_clock_test
from shared.usage_gate import AccountLease, AccountState, InvokeSlot, PoolFrozen, UsageGate

ACCOUNTS = ('max-d', 'max-e', 'max-f', 'max-g', 'max-b', 'max-c', 'max-h')
RESERVE = 'max-h'
DEFAULT_BLOCK: dict[str, object] = {
    'account': RESERVE,
    'weekly_reset_utc': 'mon 21:00',
    'release_hours': 36,
    'admit_tiers': {'default': ['critical', 'high']},
}

WEDNESDAY = datetime(2026, 11, 11, 12, tzinfo=UTC)
RELEASE_AT = datetime(2026, 11, 15, 9, tzinfo=UTC)
NEXT_RESET = datetime(2026, 11, 16, 21, tzinfo=UTC)
IN_RELEASE = datetime(2026, 11, 15, 12, tzinfo=UTC)

LOW = WorkClass(
    kind=WorkKind.TASK, project_id='dark_factory', tier='low', task_id='7', role='implementer'
)
HIGH = WorkClass(
    kind=WorkKind.TASK, project_id='dark_factory', tier='high', task_id='8', role='implementer'
)
ESSENTIAL = WorkClass(kind=WorkKind.ESSENTIAL, role='steward')


class FakeClock:
    """The reserve clock: an aware UTC instant that moves only when told to.

    WEDNESDAY is deliberately in the real past, so a banner reset placed on this calendar is
    already due on the wall clock the gate's cap sweep reads.
    """

    def __init__(self, now: datetime = WEDNESDAY) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, by: timedelta) -> None:
        self.now += by


def token_env(name: str) -> str:
    return f'TEST_TOKEN_{name.upper().replace("-", "_")}'


def write_accounts_file(tmp_path: Path, block: object) -> str:
    accounts = [{'name': name, 'oauth_token_env': token_env(name)} for name in ACCOUNTS]
    data: dict[str, object] = {'accounts': accounts}
    if block is not None:
        data['reserve'] = block
    path = tmp_path / 'accounts.yaml'
    path.write_text(yaml.safe_dump(data))
    return str(path)


def make_reserve_gate(
    tmp_path: Path,
    *,
    block: object = DEFAULT_BLOCK,
    clock: Callable[[], datetime],
    cost_store: CostStore | None = None,
    **cfg: object,
) -> UsageGate:
    settings: dict[str, object] = {'wait_for_reset': False, 'auth_reprobe_enabled': False}
    settings.update(cfg)
    config = UsageCapConfig.model_validate(
        {'accounts_file': write_accounts_file(tmp_path, block), **settings}
    )
    tokens = {token_env(name): f'fake-token-{name}' for name in ACCOUNTS}
    with patch.dict(os.environ, tokens):
        gate = UsageGate(config, cost_store=cost_store, reserve_clock=clock)
    stub_probe(gate, lambda _: True)
    return gate


def stub_probe(gate: UsageGate, answer: Callable[[AccountState], bool]) -> None:
    """Stand *answer* in for the resume probe: the kit's one write of that private seam."""
    gate._run_probe = AsyncMock(side_effect=answer)


def lease_only(gate: UsageGate, name: str) -> AccountLease:
    lease = gate.try_lease(
        exclude=[other for other in ACCOUNTS if other != name], work_class=ESSENTIAL
    )
    assert lease is not None and lease.name == name, f'{name} is not open: {lease!r}'
    return lease


def cap(
    gate: UsageGate, name: str, *, resets_at: datetime | None = None, scope: str | None = None
) -> None:
    slot = InvokeSlot(gate, lease_only(gate, name), scope=scope)
    slot.report(CapHit(reason='test cap', resets_at=resets_at))


def auth_fail(gate: UsageGate, name: str) -> None:
    InvokeSlot(gate, lease_only(gate, name)).report(AuthFailed(status=401))


def cap_all_but(gate: UsageGate, *keep: str) -> None:
    for name in ACCOUNTS:
        if name not in keep:
            cap(gate, name)


def leased_name(gate: UsageGate, *, work_class: WorkClassSource | None = None) -> str | None:
    lease = gate.try_lease(work_class=work_class)
    return None if lease is None else lease.name


class TestNoBlock:
    """B1: without a ``reserve:`` block every caller sees today's walk."""

    def test_every_class_takes_the_first_account(self, tmp_path: Path) -> None:
        gate = make_reserve_gate(tmp_path, block=None, clock=FakeClock())

        assert leased_name(gate, work_class=LOW) == 'max-d'
        assert leased_name(gate, work_class=None) == 'max-d'
        assert leased_name(gate, work_class=HIGH) == 'max-d'

    def test_the_last_account_serves_everyone_when_the_rest_are_capped(
        self, tmp_path: Path
    ) -> None:
        gate = make_reserve_gate(tmp_path, block=None, clock=FakeClock())
        cap_all_but(gate, RESERVE)

        assert leased_name(gate, work_class=LOW) == RESERVE
        assert leased_name(gate) == RESERVE
        assert gate.reserve_state is None
        assert gate.reserve_refusal(LOW) is None


class TestSelection:
    def test_an_active_block_reads_as_active(self, tmp_path: Path) -> None:
        assert make_reserve_gate(tmp_path, clock=FakeClock()).reserve_state is ReserveState.ACTIVE

    def test_b4_an_admitted_caller_keeps_the_roster_order(self, tmp_path: Path) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock())
        cap_all_but(gate, 'max-c', RESERVE)

        assert leased_name(gate, work_class=HIGH) == 'max-c'

    @pytest.mark.parametrize('work_class', [LOW, None], ids=['low', 'unclassified'])
    def test_b5_in_the_release_window_the_reserve_is_tried_first(
        self, tmp_path: Path, work_class: WorkClass | None
    ) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock(IN_RELEASE))

        assert leased_name(gate, work_class=work_class) == RESERVE

    def test_a_refused_class_finds_nothing_when_only_the_reserve_is_open(
        self, tmp_path: Path
    ) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock())
        cap_all_but(gate, RESERVE)

        assert gate.try_lease(work_class=LOW) is None
        assert gate.try_lease() is None

    @pytest.mark.parametrize('work_class', [HIGH, ESSENTIAL], ids=['high', 'essential'])
    def test_an_admitted_class_leases_the_reserve(
        self, tmp_path: Path, work_class: WorkClass
    ) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock())
        cap_all_but(gate, RESERVE)

        assert leased_name(gate, work_class=work_class) == RESERVE


class TestRefusalQuery:
    """B18: the question the scheduler asks changes nothing and claims nothing."""

    def test_a_refused_class_is_told_the_release_time(self, tmp_path: Path) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock())
        cap_all_but(gate, RESERVE)
        capped = gate.capped_account_names

        assert gate.reserve_refusal(LOW) == ReserveRefusal(RESERVE, RELEASE_AT)
        assert gate.reserve_refusal(None) == ReserveRefusal(RESERVE, RELEASE_AT)
        assert gate.reserve_refusal(HIGH) is None
        assert gate.probe_in_flight_account_names == ()
        assert gate.capped_account_names == capped

    def test_with_every_account_capped_nobody_is_refused(self, tmp_path: Path) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock())
        cap_all_but(gate, RESERVE)
        cap(gate, RESERVE)

        assert gate.reserve_refusal(LOW) is None
        assert gate.reserve_refusal(HIGH) is None
        assert gate.capped_account_names == ACCOUNTS


class TestProbingReserve:
    """B17 (selection): a refused caller never takes a probing reserve's one probe slot."""

    async def test_only_an_admitted_caller_claims_the_probe_slot(self, tmp_path: Path) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock())
        cap_all_but(gate, RESERVE)
        cap(gate, RESERVE, resets_at=WEDNESDAY)
        await gate.wait_for_open(timeout=0)

        assert gate.try_lease(work_class=LOW) is None
        assert gate.probe_in_flight_account_names == ()
        assert gate.reserve_refusal(LOW) == ReserveRefusal(RESERVE, RELEASE_AT)

        assert leased_name(gate, work_class=HIGH) == RESERVE
        assert gate.probe_in_flight_account_names == (RESERVE,)
        await gate.shutdown()


class TestPromotedCaller:
    """B19 (selection): a class given as a function is read again at every selection."""

    def test_the_function_is_asked_at_each_selection(self, tmp_path: Path) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock())
        cap_all_but(gate, RESERVE)
        answers = [LOW]
        asked: list[WorkClass] = []

        def current() -> WorkClass:
            asked.append(answers[-1])
            return answers[-1]

        assert gate.try_lease(work_class=current) is None
        answers.append(HIGH)
        assert leased_name(gate, work_class=current) == RESERVE
        assert asked == [LOW, HIGH]


class TestUnusableReserve:
    """B12 (query): a reserve failing authentication protects nothing."""

    def test_with_another_account_open_it_serves_and_nobody_is_refused(
        self, tmp_path: Path
    ) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock())
        auth_fail(gate, RESERVE)

        assert gate.reserve_refusal(LOW) is None
        assert leased_name(gate, work_class=LOW) == 'max-d'

    def test_with_the_rest_capped_nothing_is_open_and_nobody_is_refused(
        self, tmp_path: Path
    ) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock())
        auth_fail(gate, RESERVE)
        cap_all_but(gate, RESERVE)

        assert gate.reserve_refusal(LOW) is None
        assert gate.try_lease(work_class=LOW) is None


INVALID_BLOCKS = [
    pytest.param({**DEFAULT_BLOCK, 'account': 'max-z'}, id='unknown-account'),
    pytest.param({**DEFAULT_BLOCK, 'colour': 'red'}, id='unknown-key'),
    pytest.param({**DEFAULT_BLOCK, 'release_hours': 0}, id='release-hours-0'),
]


class TestInvalidBlock:
    """B11 (selection): an invalid block is inert."""

    @pytest.mark.parametrize('block', INVALID_BLOCKS)
    def test_an_invalid_block_refuses_nobody(self, tmp_path: Path, block: object) -> None:
        gate = make_reserve_gate(tmp_path, block=block, clock=FakeClock())
        cap_all_but(gate, RESERVE)

        assert gate.reserve_state is ReserveState.INVALID
        assert gate.reserve_refusal(LOW) is None
        assert leased_name(gate, work_class=LOW) == RESERVE


async def events(store: AsyncMock, event_type: str) -> list[tuple[str, dict[str, object]]]:
    """The ``(account_name, details)`` of every *event_type* row written so far.

    Gate events are fire-and-forget tasks, so the loop is given a few turns to run them first.
    """
    for _ in range(5):
        await asyncio.sleep(0)
    return [
        (call.kwargs['account_name'], json.loads(call.kwargs['details']))
        for call in store.save_account_event.call_args_list
        if call.kwargs['event_type'] == event_type
    ]


class TestAttachCostStore:
    async def test_attaching_a_store_records_the_reserve_config_once(
        self, tmp_path: Path
    ) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock())
        store = make_mock_cost_store()

        gate.attach_cost_store(store)

        assert await events(store, 'reserve_config') == [
            (
                RESERVE,
                {
                    'state': 'active',
                    'reason': None,
                    'account': RESERVE,
                    'next_reset': NEXT_RESET.isoformat(),
                    'release_at': RELEASE_AT.isoformat(),
                    'admit_tiers': {'default': ['critical', 'high']},
                },
            )
        ]
        cap(gate, 'max-d')
        assert [name for name, _ in await events(store, 'cap_hit')] == ['max-d']

    async def test_a_store_given_to_the_constructor_gets_no_config_row(
        self, tmp_path: Path
    ) -> None:
        store = make_mock_cost_store()
        make_reserve_gate(tmp_path, clock=FakeClock(), cost_store=store)

        assert await events(store, 'reserve_config') == []

    async def test_without_a_block_attaching_records_no_config_row(self, tmp_path: Path) -> None:
        gate = make_reserve_gate(tmp_path, block=None, clock=FakeClock())
        store = make_mock_cost_store()

        gate.attach_cost_store(store)
        cap(gate, 'max-d')

        assert await events(store, 'reserve_config') == []
        assert len(await events(store, 'cap_hit')) == 1

    @pytest.mark.parametrize(
        ('block', 'cause'),
        [
            pytest.param({**DEFAULT_BLOCK, 'account': 'max-z'}, 'max-z', id='unknown-account'),
            pytest.param({**DEFAULT_BLOCK, 'colour': 'red'}, 'colour', id='unknown-key'),
            pytest.param(
                {**DEFAULT_BLOCK, 'release_hours': 0}, 'release_hours', id='release-hours-0'
            ),
        ],
    )
    async def test_b11_an_invalid_block_is_recorded_with_its_reason(
        self, tmp_path: Path, block: object, cause: str
    ) -> None:
        gate = make_reserve_gate(tmp_path, block=block, clock=FakeClock())
        store = make_mock_cost_store()

        gate.attach_cost_store(store)

        [(account_name, details)] = await events(store, 'reserve_config')
        assert account_name == ''
        assert details['state'] == 'invalid'
        assert cause in str(details['reason'])
        assert details['account'] is None
        assert details['release_at'] is None


class TestReserveAccountUnavailable:
    """B12 (event): the reserve failing authentication is recorded, once."""

    async def test_the_reserve_failing_authentication_is_recorded(self, tmp_path: Path) -> None:
        store = make_mock_cost_store()
        gate = make_reserve_gate(tmp_path, clock=FakeClock(), cost_store=store)

        auth_fail(gate, RESERVE)

        assert await events(store, 'reserve_account_unavailable') == [
            (RESERVE, {'account': RESERVE})
        ]

    async def test_another_account_failing_is_not(self, tmp_path: Path) -> None:
        store = make_mock_cost_store()
        gate = make_reserve_gate(tmp_path, clock=FakeClock(), cost_store=store)

        auth_fail(gate, 'max-d')

        assert await events(store, 'reserve_account_unavailable') == []

    async def test_without_a_block_nothing_is_recorded(self, tmp_path: Path) -> None:
        store = make_mock_cost_store()
        gate = make_reserve_gate(tmp_path, block=None, clock=FakeClock(), cost_store=store)

        auth_fail(gate, RESERVE)

        assert await events(store, 'reserve_account_unavailable') == []
        assert len(await events(store, 'auth_failed')) == 1


class TestScheduleDrift:
    """B13: a banner does not move the calendar; drift is recorded at most once per week."""

    async def test_drift_is_recorded_once_and_the_release_time_stays(
        self, tmp_path: Path
    ) -> None:
        store = make_mock_cost_store()
        gate = make_reserve_gate(tmp_path, clock=FakeClock(), cost_store=store)
        cap_all_but(gate, RESERVE)

        cap(gate, RESERVE, resets_at=NEXT_RESET + timedelta(hours=1))
        expected = [
            (
                RESERVE,
                {
                    'configured_reset': NEXT_RESET.isoformat(),
                    'observed_reset': (NEXT_RESET + timedelta(hours=1)).isoformat(),
                },
            )
        ]
        assert await events(store, 'reserve_schedule_drift') == expected

        assert await gate.wait_for_open(timeout=0) is True
        cap(gate, RESERVE, resets_at=NEXT_RESET + timedelta(minutes=50))
        assert await events(store, 'reserve_schedule_drift') == expected

        assert await gate.wait_for_open(timeout=0) is True
        refused = gate.reserve_refusal(LOW)
        assert refused is not None
        assert refused.release_at == RELEASE_AT
        await gate.shutdown()

    async def test_a_reset_five_hours_off_is_not_drift(self, tmp_path: Path) -> None:
        store = make_mock_cost_store()
        gate = make_reserve_gate(tmp_path, clock=FakeClock(), cost_store=store)

        cap(gate, RESERVE, resets_at=NEXT_RESET + timedelta(hours=5))

        assert await events(store, 'reserve_schedule_drift') == []
        assert gate.reserve_refusal(LOW) is None
        cap_all_but(gate, RESERVE)
        assert gate.capped_account_names == ACCOUNTS

    async def test_a_scoped_cap_is_not_drift(self, tmp_path: Path) -> None:
        store = make_mock_cost_store()
        gate = make_reserve_gate(tmp_path, clock=FakeClock(), cost_store=store)

        cap(gate, RESERVE, scope=SCOPE, resets_at=NEXT_RESET + timedelta(hours=1))

        assert len(await events(store, 'cap_hit')) == 1
        assert await events(store, 'reserve_schedule_drift') == []


class TestWithoutAStore:
    async def test_the_reserve_events_record_nothing_and_raise_nothing(
        self, tmp_path: Path
    ) -> None:
        gate = make_reserve_gate(tmp_path, clock=FakeClock())

        cap(gate, RESERVE, resets_at=NEXT_RESET + timedelta(hours=1))
        cap_all_but(gate, RESERVE)
        assert await gate.wait_for_open(timeout=0) is True
        auth_fail(gate, RESERVE)

        assert gate.capped_account_names == ACCOUNTS[:-1]
        assert gate.auth_failed_account_names == (RESERVE,)
        await gate.shutdown()


SIX_CAPPED = ACCOUNTS[:-1]
FROZEN_TODAY = (
    'Usage pool frozen: no account admissible among all 7 account(s) — '
    'capped: max-d, max-e, max-f, max-g, max-b, max-c; auth-failed: none'
)


def reserve_frozen() -> PoolFrozen:
    return PoolFrozen(
        account_count=7,
        capped_account_names=SIX_CAPPED,
        auth_failed_account_names=(),
        reserved_account=RESERVE,
        release_at=RELEASE_AT,
    )


def capped_frozen() -> PoolFrozen:
    return PoolFrozen(
        account_count=7, capped_account_names=SIX_CAPPED, auth_failed_account_names=()
    )


class TestPoolFrozenNamesTheReserve:
    def test_a_reserve_refusal_says_so_and_names_the_release_time(self) -> None:
        frozen = reserve_frozen()

        assert frozen.reserved_account == RESERVE
        assert frozen.release_at == RELEASE_AT
        assert frozen.scope is None
        assert 'refused the reserve' in str(frozen)
        assert RESERVE in str(frozen)
        assert RELEASE_AT.isoformat() in str(frozen)
        assert 'all accounts capped' not in str(frozen)
        assert 'no account admissible' not in str(frozen)

    def test_without_the_reserve_fields_the_message_is_unchanged(self) -> None:
        frozen = capped_frozen()

        assert frozen.reserved_account is None
        assert frozen.release_at is None
        assert str(frozen) == FROZEN_TODAY


class TestCapPatienceMessage:
    def test_a_reserve_refusal_is_not_called_a_cap_wait(self) -> None:
        error = AllAccountsCappedException(
            retries=0, elapsed_secs=1.5, label='Task 7 [impl]', pool_frozen=reserve_frozen()
        )

        assert str(error).startswith('Task 7 [impl]: ')
        assert 'refused the reserve' in str(error)
        assert RELEASE_AT.isoformat() in str(error)
        assert 'all accounts capped' not in str(error).lower()
        assert 'cap-wait' not in str(error).lower()

    def test_a_frozen_pool_without_the_reserve_keeps_its_message(self) -> None:
        error = AllAccountsCappedException(
            retries=0, elapsed_secs=1.5, label='Task 7 [impl]', pool_frozen=capped_frozen()
        )

        assert str(error) == (
            'Task 7 [impl]: cap-wait deadline spent parked after 0 retries (1.5s elapsed) — '
            + FROZEN_TODAY
        )

    def test_without_a_frozen_pool_the_message_is_unchanged(self) -> None:
        error = AllAccountsCappedException(retries=3, elapsed_secs=1.5, label='Task 7 [impl]')

        assert str(error) == 'Task 7 [impl]: all accounts capped after 3 retries (1.5s elapsed)'


def refused_state(
    tmp_path: Path, clock: FakeClock | None = None, **cfg: object
) -> tuple[UsageGate, AsyncMock]:
    """B2's state: a gate with a store, the reserve open and the six others capped for good."""
    store = make_mock_cost_store()
    gate = make_reserve_gate(tmp_path, clock=clock or FakeClock(), cost_store=store, **cfg)
    cap_all_but(gate, RESERVE)
    return gate, store


async def let_run() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def cancel(task: asyncio.Task[object]) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def heartbeats(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, object]]:
    found = []
    for record in caplog.records:
        if record.name == 'shared.usage_gate' and record.getMessage().startswith('{'):
            line = json.loads(record.getMessage())
            if line.get('event') == event:
                found.append(line)
    return found


LOW_WAIT_DETAILS: dict[str, object] = {
    'account': RESERVE,
    'kind': 'task',
    'role': 'implementer',
    'project_id': 'dark_factory',
    'tier': 'low',
    'task_id': '7',
    'release_at': RELEASE_AT.isoformat(),
    'unclassified': False,
}


class TestReserveWait:
    """B2: a refused caller waits without freezing the pool; others are served meanwhile.

    Nothing public reads ``_open`` in a way a reserve wait could disturb: ``is_paused`` and
    ``wait_for_open``'s fast path read account phases, and every wait on ``_open`` follows a
    selection that found no account open. So "the wait never clears ``_open``" holds by
    construction, and these tests assert only the PRD's public signal of it, ``is_paused``.
    """

    async def test_a_refused_caller_waits_and_an_admitted_one_is_served(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        gate, store = refused_state(tmp_path)
        caplog.set_level(logging.WARNING, logger='shared.usage_gate')

        waiter = asyncio.create_task(gate.before_invoke(work_class=LOW))
        await let_run()

        assert not waiter.done()
        assert gate.reserve_waiters == 1
        assert gate.is_paused is False
        assert gate.paused_reason == ''
        lease = await asyncio.wait_for(gate.before_invoke(work_class=HIGH), timeout=1)
        assert lease is not None and lease.name == RESERVE
        assert await events(store, 'reserve_wait_started') == [(RESERVE, LOW_WAIT_DETAILS)]
        [beat] = heartbeats(caplog, 'reserve_wait')
        assert beat['reserve_account'] == RESERVE
        assert beat['release_at'] == RELEASE_AT.isoformat()
        assert (beat['kind'], beat['tier'], beat['task_id']) == ('task', 'low', '7')
        assert {
            'account_count': 7,
            'capped_accounts': list(SIX_CAPPED),
            'auth_failed_accounts': [],
            'probe_in_flight_accounts': [],
            'soonest_open_at': None,
        }.items() <= beat.items()
        assert isinstance(beat['elapsed_s'], float)

        await cancel(waiter)

        assert gate.reserve_waiters == 0
        [(account_name, ended)] = await events(store, 'reserve_wait_ended')
        assert account_name == RESERVE
        assert ended['ended_by'] == 'cancelled'
        assert isinstance(ended['waited_secs'], (int, float))

    async def test_b16_a_caller_without_a_class_waits_unclassified(self, tmp_path: Path) -> None:
        gate, store = refused_state(tmp_path)

        waiter = asyncio.create_task(gate.before_invoke())
        await let_run()

        assert gate.reserve_waiters == 1
        [(_, started)] = await events(store, 'reserve_wait_started')
        assert started['kind'] == 'background'
        assert started['unclassified'] is True
        await cancel(waiter)


class TestReserveWaitDeadline:
    """B9: the deadline names the reserve; no cap is counted."""

    async def test_the_caller_deadline_raises_pool_frozen_naming_the_reserve(
        self, tmp_path: Path
    ) -> None:
        gate, store = refused_state(tmp_path)

        with pytest.raises(PoolFrozen) as raised:
            await asyncio.wait_for(
                gate.before_invoke(work_class=LOW, max_wait_secs=0.05), timeout=2
            )

        assert raised.value.reserved_account == RESERVE
        assert raised.value.release_at == RELEASE_AT
        assert raised.value.scope is None
        [(_, ended)] = await events(store, 'reserve_wait_ended')
        assert ended['ended_by'] == 'deadline'
        assert gate.capped_account_names == SIX_CAPPED
        assert gate.reserve_waiters == 0

    @pytest.mark.parametrize(
        ('park', 'max_wait_secs'),
        [
            pytest.param(False, None, id='park-false'),
            pytest.param(True, 0.0, id='spent-allowance'),
        ],
    )
    async def test_no_allowance_raises_at_once_and_records_no_wait(
        self, tmp_path: Path, park: bool, max_wait_secs: float | None
    ) -> None:
        gate, store = refused_state(tmp_path)

        with pytest.raises(PoolFrozen) as raised:
            await asyncio.wait_for(
                gate.before_invoke(work_class=LOW, park=park, max_wait_secs=max_wait_secs),
                timeout=1,
            )

        assert raised.value.reserved_account == RESERVE
        assert gate.capped_account_names == SIX_CAPPED
        assert gate.reserve_waiters == 0
        assert await events(store, 'reserve_wait_started') == []
        assert await events(store, 'reserve_wait_ended') == []


class TestReserveWaitRouting:
    async def test_b17_a_refused_caller_leaves_a_probing_reserve_unclaimed(
        self, tmp_path: Path
    ) -> None:
        gate, _ = refused_state(tmp_path)
        cap(gate, RESERVE, resets_at=WEDNESDAY)
        await gate.wait_for_open(timeout=0)

        with pytest.raises(PoolFrozen) as raised:
            await asyncio.wait_for(gate.before_invoke(work_class=LOW, park=False), timeout=1)

        assert raised.value.reserved_account == RESERVE
        assert gate.probe_in_flight_account_names == ()
        lease = await asyncio.wait_for(gate.before_invoke(work_class=HIGH), timeout=1)
        assert lease is not None and lease.name == RESERVE
        assert gate.probe_in_flight_account_names == (RESERVE,)
        await gate.shutdown()

    async def test_b12_an_unusable_reserve_sends_the_caller_down_the_all_capped_path(
        self, tmp_path: Path
    ) -> None:
        gate, store = refused_state(tmp_path)
        auth_fail(gate, RESERVE)

        with pytest.raises(PoolFrozen) as raised:
            await asyncio.wait_for(gate.before_invoke(work_class=LOW, park=False), timeout=1)

        assert raised.value.reserved_account is None
        assert gate.reserve_waiters == 0
        assert await events(store, 'reserve_wait_started') == []

    async def test_b15_an_admitted_scoped_caller_takes_the_scope_wait(
        self, tmp_path: Path
    ) -> None:
        gate, store = refused_state(tmp_path)
        cap(gate, RESERVE, scope=SCOPE, resets_at=datetime.now(UTC) + timedelta(hours=1))

        with pytest.raises(PoolFrozen) as raised:
            await asyncio.wait_for(
                gate.before_invoke(SCOPE, work_class=HIGH, park=False), timeout=1
            )

        assert raised.value.scope == SCOPE
        assert raised.value.reserved_account is None
        assert await events(store, 'reserve_wait_started') == []


SUNDAY_10 = datetime(2026, 11, 15, 10, tzinfo=UTC)
NEXT_WEEK_RELEASE_AT = RELEASE_AT + timedelta(days=7)


class TestReserveWaitWakes:
    """B6, B19 and B14 on the virtual clock: re-selection with no account event."""

    @virtual_clock_test
    async def test_b6_the_release_time_ends_the_wait(self, tmp_path: Path) -> None:
        clock = FakeClock()
        gate, store = refused_state(tmp_path, clock)
        loop = asyncio.get_running_loop()

        waiter = asyncio.create_task(gate.before_invoke(work_class=LOW))
        await asyncio.sleep(301)

        assert not waiter.done()
        assert len(await events(store, 'reserve_wait_started')) == 1
        assert await events(store, 'reserve_wait_ended') == []

        clock.advance(SUNDAY_10 - clock.now)
        released_at = loop.time()
        lease = await waiter

        assert lease is not None and lease.name == RESERVE
        assert loop.time() - released_at <= 300
        [(_, ended)] = await events(store, 'reserve_wait_ended')
        assert ended['ended_by'] == 'released'

    @virtual_clock_test
    async def test_b19_a_promoted_caller_leases_the_reserve(self, tmp_path: Path) -> None:
        gate, store = refused_state(tmp_path)
        loop = asyncio.get_running_loop()
        answers = [LOW]

        waiter = asyncio.create_task(gate.before_invoke(work_class=lambda: answers[-1]))
        await asyncio.sleep(301)
        assert not waiter.done()

        answers.append(HIGH)
        promoted_at = loop.time()
        lease = await waiter

        assert lease is not None and lease.name == RESERVE
        assert loop.time() - promoted_at <= 300
        [(_, started)] = await events(store, 'reserve_wait_started')
        [(_, ended)] = await events(store, 'reserve_wait_ended')
        assert started['tier'] == 'low'
        assert ended['tier'] == 'high'
        assert ended['ended_by'] == 'admitted'

    @virtual_clock_test
    async def test_a_clock_past_the_release_waits_for_the_next_one(self, tmp_path: Path) -> None:
        clock = FakeClock()
        gate, store = refused_state(tmp_path, clock)
        selections = 0

        def counted() -> WorkClass:
            nonlocal selections
            selections += 1
            if selections > 20:
                raise RuntimeError('the reserve wait is spinning')
            return LOW

        waiter = asyncio.create_task(gate.before_invoke(work_class=counted))
        await let_run()
        clock.advance(timedelta(days=8))
        await asyncio.sleep(901)

        assert not waiter.done()
        assert selections <= 5
        assert gate.reserve_refusal(LOW) == ReserveRefusal(RESERVE, NEXT_WEEK_RELEASE_AT)
        await cancel(waiter)
        [(_, ended)] = await events(store, 'reserve_wait_ended')
        assert ended['release_at'] == NEXT_WEEK_RELEASE_AT.isoformat()

    @virtual_clock_test
    async def test_b8_a_repoll_during_the_reserves_probe_keeps_the_same_wait(
        self, tmp_path: Path
    ) -> None:
        gate, store = refused_state(tmp_path, probe_slot_hold_secs=3600)
        cap(gate, RESERVE, resets_at=WEDNESDAY)
        await gate.wait_for_open(timeout=0)
        waiter = asyncio.create_task(gate.before_invoke(work_class=LOW))
        await let_run()

        lease = await gate.before_invoke(work_class=HIGH)
        assert lease is not None and lease.name == RESERVE
        await asyncio.sleep(301)

        assert gate.probe_in_flight_account_names == (RESERVE,)
        assert not waiter.done()
        assert gate.reserve_waiters == 1
        assert await events(store, 'reserve_wait_ended') == []
        await cancel(waiter)
        await gate.shutdown()

    @virtual_clock_test
    async def test_b14_a_refused_scoped_caller_takes_the_reserve_wait(
        self, tmp_path: Path
    ) -> None:
        clock = FakeClock()
        gate, _ = refused_state(tmp_path, clock)

        waiter = asyncio.create_task(gate.before_invoke(SCOPE, work_class=LOW))
        await let_run()
        assert gate.reserve_waiters == 1

        clock.advance(SUNDAY_10 - clock.now)
        lease = await waiter

        assert lease is not None and lease.name == RESERVE

    async def test_a_deadline_after_the_clock_passed_the_release_names_the_next_one(
        self, tmp_path: Path
    ) -> None:
        clock = FakeClock()
        gate, _ = refused_state(tmp_path, clock)

        waiter = asyncio.create_task(gate.before_invoke(work_class=LOW, max_wait_secs=0.2))
        await let_run()
        clock.advance(timedelta(days=8))

        with pytest.raises(PoolFrozen) as raised:
            await asyncio.wait_for(waiter, timeout=2)

        assert raised.value.reserved_account == RESERVE
        assert raised.value.release_at == NEXT_WEEK_RELEASE_AT

    async def test_b14_its_deadline_names_the_reserve_not_the_scope(
        self, tmp_path: Path
    ) -> None:
        gate, _ = refused_state(tmp_path)

        with pytest.raises(PoolFrozen) as raised:
            await asyncio.wait_for(
                gate.before_invoke(SCOPE, work_class=LOW, max_wait_secs=0.05), timeout=2
            )

        assert raised.value.reserved_account == RESERVE
        assert raised.value.scope is None


async def eventually(check: Callable[[], bool], *, polls: int = 300) -> None:
    """Poll *check* on a real loop for at most *polls* loop turns 10 ms apart.

    The budget counts loop turns, not wall clock, so a descheduled loop spends none
    of it; at most a few seconds, far below the 300 s re-poll, so a pass proves a wake.
    """
    for _ in range(polls):
        if check():
            return
        await asyncio.sleep(0.01)
    assert check(), f'not within {polls} polls'


def soon() -> datetime:
    return datetime.now(UTC) + timedelta(seconds=0.1)


class TestTransitionsWakeTheWait:
    """PRD §6.3: another account opening, or the reserve failing, wakes a reserve wait."""

    async def test_b7_another_account_reopening_serves_the_waiter(self, tmp_path: Path) -> None:
        store = make_mock_cost_store()
        gate = make_reserve_gate(
            tmp_path, clock=FakeClock(), cost_store=store, wait_for_reset=True
        )
        cap_all_but(gate, 'max-d', RESERVE)
        cap(gate, 'max-d', resets_at=soon())

        waiter = asyncio.create_task(gate.before_invoke(work_class=LOW))
        await let_run()
        assert gate.reserve_waiters == 1

        lease = await asyncio.wait_for(waiter, timeout=3)

        assert lease is not None and lease.name == 'max-d'
        [(_, ended)] = await events(store, 'reserve_wait_ended')
        assert ended['ended_by'] == 'account_opened'
        await gate.shutdown()

    async def test_b8_the_reserves_own_probe_does_not_end_the_wait(
        self, tmp_path: Path
    ) -> None:
        gate, store = refused_state(tmp_path)
        cap(gate, RESERVE, resets_at=WEDNESDAY)
        await gate.wait_for_open(timeout=0)

        waiter = asyncio.create_task(gate.before_invoke(work_class=LOW))
        await let_run()
        lease = await asyncio.wait_for(gate.before_invoke(work_class=HIGH), timeout=1)
        assert lease is not None and lease.name == RESERVE
        await let_run()
        InvokeSlot(gate, lease).confirm()
        await let_run()

        assert not waiter.done()
        assert gate.reserve_waiters == 1
        assert len(await events(store, 'reserve_wait_started')) == 1
        assert await events(store, 'reserve_wait_ended') == []
        await cancel(waiter)

    async def test_b14_a_scope_capped_account_opening_sends_the_waiter_to_the_scope_wait(
        self, tmp_path: Path
    ) -> None:
        store = make_mock_cost_store()
        gate = make_reserve_gate(
            tmp_path, clock=FakeClock(), cost_store=store, wait_for_reset=True
        )
        cap_all_but(gate, 'max-d', RESERVE)
        cap(gate, 'max-d', scope=SCOPE, resets_at=datetime.now(UTC) + timedelta(hours=1))
        cap(gate, 'max-d', resets_at=soon())

        waiter = asyncio.create_task(gate.before_invoke(SCOPE, work_class=LOW))
        await let_run()
        assert gate.reserve_waiters == 1

        await eventually(lambda: gate.reserve_waiters == 0)

        [(_, ended)] = await events(store, 'reserve_wait_ended')
        assert ended['ended_by'] == 'account_opened'
        assert not waiter.done()
        assert gate.is_paused is False
        await cancel(waiter)
        await gate.shutdown()

    async def test_b20_the_reserve_failing_sends_the_waiter_to_the_all_capped_path(
        self, tmp_path: Path
    ) -> None:
        gate, store = refused_state(tmp_path)
        waiter = asyncio.create_task(gate.before_invoke(work_class=LOW))
        await let_run()
        assert gate.reserve_waiters == 1

        auth_fail(gate, RESERVE)
        await eventually(lambda: gate.reserve_waiters == 0, polls=100)

        [(_, ended)] = await events(store, 'reserve_wait_ended')
        assert ended['ended_by'] == 'reserve_unusable'
        assert not waiter.done()
        assert gate.is_paused is True
        await cancel(waiter)

    async def test_b10_when_the_reserve_reopens_only_the_admitted_caller_takes_it(
        self, tmp_path: Path
    ) -> None:
        """``is_paused`` is the PRD's public signal that ``_open`` is untouched (TestReserveWait)."""
        gate = make_reserve_gate(tmp_path, clock=FakeClock(), wait_for_reset=True)
        stub_probe(gate, lambda acct: acct.name == RESERVE)
        cap_all_but(gate, RESERVE)
        cap(gate, RESERVE, resets_at=soon())
        paused_reason = gate.paused_reason
        low = asyncio.create_task(gate.before_invoke(work_class=LOW))
        high = asyncio.create_task(gate.before_invoke(work_class=HIGH))
        await let_run()
        assert gate.is_paused is True

        lease = await asyncio.wait_for(high, timeout=3)
        assert lease is not None and lease.name == RESERVE
        InvokeSlot(gate, lease).confirm()
        await eventually(lambda: gate.reserve_waiters == 1)

        assert not low.done()
        assert gate.is_paused is False
        assert gate.paused_reason == paused_reason
        served = await asyncio.wait_for(gate.before_invoke(work_class=ESSENTIAL), timeout=1)
        assert served is not None and served.name == RESERVE
        await cancel(low)
        await gate.shutdown()


CAP_BANNER = "You've hit your usage limit for Claude Pro. Your plan resets in 3 hours."
HIGH_SERVED: dict[str, object] = {
    'kind': 'task',
    'role': 'implementer',
    'project_id': 'dark_factory',
    'tier': 'high',
    'task_id': '8',
}


class TestReserveServed:
    """B3 and B5: a call on the reserve settling in the protected window is recorded once."""

    async def test_b3_an_admitted_task_is_recorded_with_its_cost(self, tmp_path: Path) -> None:
        gate, store = refused_state(tmp_path)

        async with gate.invoke_slot(work_class=HIGH) as slot:
            assert slot.account_name == RESERVE
            assert slot.work_class is HIGH
            slot.confirm(1.25)

        assert await events(store, 'reserve_served') == [
            (RESERVE, {**HIGH_SERVED, 'cost_usd': 1.25})
        ]

    async def test_b3_essential_work_is_recorded_as_essential(self, tmp_path: Path) -> None:
        gate, store = refused_state(tmp_path)

        async with gate.invoke_slot(work_class=ESSENTIAL) as slot:
            slot.confirm(0.75)

        [(_, served)] = await events(store, 'reserve_served')
        assert served['kind'] == 'essential'
        assert served['role'] == 'steward'
        assert served['cost_usd'] == 0.75

    async def test_b5_a_call_in_the_release_window_is_not_recorded(self, tmp_path: Path) -> None:
        store = make_mock_cost_store()
        gate = make_reserve_gate(tmp_path, clock=FakeClock(IN_RELEASE), cost_store=store)

        async with gate.invoke_slot(work_class=LOW) as slot:
            assert slot.account_name == RESERVE
            slot.confirm(0.5)

        assert await events(store, 'reserve_served') == []

    async def test_a_call_on_another_account_is_not_recorded(self, tmp_path: Path) -> None:
        store = make_mock_cost_store()
        gate = make_reserve_gate(tmp_path, clock=FakeClock(), cost_store=store)

        async with gate.invoke_slot(work_class=LOW) as slot:
            assert slot.account_name == 'max-d'
            slot.confirm(0.5)

        assert await events(store, 'reserve_served') == []

    async def test_one_verdict_one_row_and_a_report_has_no_cost(self, tmp_path: Path) -> None:
        gate, store = refused_state(tmp_path)

        async with gate.invoke_slot(work_class=HIGH) as slot:
            slot.report(OK())
            slot.confirm(2.0)

        assert await events(store, 'reserve_served') == [
            (RESERVE, {**HIGH_SERVED, 'cost_usd': None})
        ]

    async def test_a_slot_left_without_a_verdict_is_not_recorded(self, tmp_path: Path) -> None:
        gate, store = refused_state(tmp_path)

        async with gate.invoke_slot(work_class=HIGH) as slot:
            assert slot.account_name == RESERVE

        assert await events(store, 'reserve_served') == []

    async def test_a_detected_cap_is_recorded_without_a_cost(self, tmp_path: Path) -> None:
        gate, store = refused_state(tmp_path)

        async with gate.invoke_slot(work_class=HIGH) as slot:
            assert slot.detect_cap_hit('', CAP_BANNER) is True

        assert await events(store, 'reserve_served') == [
            (RESERVE, {**HIGH_SERVED, 'cost_usd': None})
        ]

    async def test_the_keyword_reaches_selection(self, tmp_path: Path) -> None:
        gate, _ = refused_state(tmp_path)

        with pytest.raises(PoolFrozen) as raised:
            async with gate.invoke_slot(work_class=LOW, park=False):
                pytest.fail('a refused caller was given a slot')

        assert raised.value.reserved_account == RESERVE


def double_slot() -> MagicMock:
    slot = MagicMock()
    slot.account_name = 'double'
    slot.token = None
    slot.detect_cap_hit = MagicMock(return_value=False)
    return slot


@contextlib.asynccontextmanager
async def todays_invoke_slot(
    scope: str | None = None, *, park: bool = True, max_wait_secs: float | None = None
) -> AsyncIterator[MagicMock]:
    yield double_slot()


@contextlib.asynccontextmanager
async def reserve_aware_invoke_slot(
    scope: str | None = None,
    *,
    park: bool = True,
    max_wait_secs: float | None = None,
    work_class: WorkClassSource | None = None,
) -> AsyncIterator[MagicMock]:
    yield double_slot()


def gate_double(
    invoke_slot: Callable[..., contextlib.AbstractAsyncContextManager[MagicMock]],
) -> MagicMock:
    """A gate whose invoke_slot keeps `invoke_slot`'s signature, so a stray keyword raises."""
    double = MagicMock(spec=UsageGate)
    double.account_count = 1
    double.soonest_resets_at = None
    double.invoke_slot = MagicMock(side_effect=invoke_slot)
    return double


class TestTheKeywordIsForwardedOnlyWhenGiven:
    """B1 (call shape): a caller passing no class calls invoke_slot exactly as today."""

    async def test_without_a_class_the_double_is_called_as_today(self) -> None:
        double = gate_double(todays_invoke_slot)

        result = await invoke_with_cap_retry(
            double,
            'L',
            prompt='p',
            model='m',
            invoke_fn=scripted_invoke(ok_result()),
        )

        assert result.success
        double.invoke_slot.assert_called_once()
        assert double.invoke_slot.call_args.args == ()
        assert set(double.invoke_slot.call_args.kwargs) == {'scope', 'park', 'max_wait_secs'}

    async def test_a_given_class_reaches_invoke_slot_as_the_same_object(self) -> None:
        double = gate_double(reserve_aware_invoke_slot)

        await invoke_with_cap_retry(
            double,
            'L',
            prompt='p',
            model='m',
            work_class=HIGH,
            invoke_fn=scripted_invoke(ok_result()),
        )

        assert double.invoke_slot.call_args.kwargs['work_class'] is HIGH


class TestGateMock:
    async def test_the_mock_forwards_a_given_class_and_mirrors_it_on_the_slot(self) -> None:
        gate = make_gate_mock()

        async with gate.invoke_slot(work_class=HIGH) as slot:
            assert slot.work_class is HIGH

        gate.before_invoke.assert_awaited_once_with(scope=None, work_class=HIGH)

    async def test_without_a_class_before_invoke_is_awaited_as_today(self) -> None:
        gate = make_gate_mock()

        async with gate.invoke_slot() as slot:
            assert slot.work_class is None

        gate.before_invoke.assert_awaited_once_with(scope=None)


class TestThroughTheRetryLoop:
    """B9: a reserve refusal spends the caller's patience without reading as a cap."""

    async def test_spent_patience_names_the_reserve_and_counts_no_cap(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        gate, _ = refused_state(tmp_path)
        invoke = scripted_invoke(ok_result())
        caplog.set_level(logging.ERROR, logger='shared.cli_invoke')

        with pytest.raises(AllAccountsCappedException) as raised:
            await asyncio.wait_for(
                invoke_with_cap_retry(
                    gate,
                    'Task 7 [impl]',
                    work_class=LOW,
                    cap_wait_sanity_secs=0.05,
                    max_cap_retries=1,
                    invoke_fn=invoke,
                    prompt='p',
                    model='claude-opus-4-1',
                ),
                timeout=5,
            )

        assert raised.value.retries == 0
        assert raised.value.pool_frozen is not None
        assert raised.value.pool_frozen.reserved_account == RESERVE
        assert 'refused the reserve' in str(raised.value)
        assert RELEASE_AT.isoformat() in str(raised.value)
        assert 'all accounts capped' not in str(raised.value).lower()
        assert invoke.calls == []
        [line] = [
            record.getMessage()
            for record in caplog.records
            if record.name == 'shared.cli_invoke' and record.levelno == logging.ERROR
        ]
        assert 'refused the reserve' in line
        assert 'cap-wait' not in line.lower()

    async def test_without_parking_the_refusal_propagates(self, tmp_path: Path) -> None:
        gate, _ = refused_state(tmp_path)

        with pytest.raises(PoolFrozen) as raised:
            await asyncio.wait_for(
                invoke_with_cap_retry(
                    gate,
                    'Task 7 [impl]',
                    work_class=LOW,
                    park_on_frozen_pool=False,
                    invoke_fn=scripted_invoke(ok_result()),
                    prompt='p',
                    model='claude-opus-4-1',
                ),
                timeout=5,
            )

        assert raised.value.reserved_account == RESERVE

    async def test_an_admitted_call_is_served_and_recorded_with_its_cost(
        self, tmp_path: Path
    ) -> None:
        gate, store = refused_state(tmp_path)
        invoke = scripted_invoke(ok_result())

        result = await asyncio.wait_for(
            invoke_with_cap_retry(
                gate,
                'Task 8 [impl]',
                work_class=HIGH,
                invoke_fn=invoke,
                prompt='p',
                model='claude-opus-4-1',
            ),
            timeout=5,
        )

        assert result.success
        assert invoke.calls[0]['oauth_token'] == f'fake-token-{RESERVE}'
        assert await events(store, 'reserve_served') == [
            (RESERVE, {**HIGH_SERVED, 'cost_usd': 0.5})
        ]
