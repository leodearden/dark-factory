"""A real UsageGate in a chosen allowance-reserve state, for tests in any suite.

The state most suites need is plans/allowance-reserve-by-account-prd.md §7's B2: every
account but the reserve capped, the reserve open, the clock inside its protected window.
The fused-memory, scripts and orchestrator suites cannot import ``shared/tests``, so the
recipe lives HERE, in the shipped ``shared`` package beside :mod:`shared.testing_virtual_clock`
and :mod:`shared.testing_fake_claude`, rather than being copied into each suite.

Every phase change goes through public seams (``try_lease`` and ``InvokeSlot.report``), and
the gate is built with ``wait_for_reset`` off, so no resume probe ever starts.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import yaml

from shared.account_reserve import WorkClass, WorkKind
from shared.config_models import UsageCapConfig
from shared.cost_store import CostStore
from shared.invocation_outcome import CapHit
from shared.usage_gate import InvokeSlot, UsageGate

__all__ = [
    'DEFAULT_RESERVE_BLOCK',
    'POOL',
    'PROTECTED',
    'RELEASE_AT',
    'RESERVE_ACCOUNT',
    'ReserveClock',
    'cap_account',
    'cap_all_but_reserve',
    'fake_tokens',
    'reserve_gate',
    'write_reserve_accounts_file',
]

RESERVE_ACCOUNT = 'max-h'
POOL: tuple[str, ...] = ('max-d', 'max-e', 'max-f', 'max-g', 'max-b', 'max-c', RESERVE_ACCOUNT)
DEFAULT_RESERVE_BLOCK: Mapping[str, object] = {
    'account': RESERVE_ACCOUNT,
    'weekly_reset_utc': 'mon 21:00',
    'release_hours': 36,
    'admit_tiers': {'default': ['critical', 'high']},
}
PROTECTED = datetime(2026, 11, 11, 12, tzinfo=UTC)
RELEASE_AT = datetime(2026, 11, 15, 9, tzinfo=UTC)

_ESSENTIAL = WorkClass(kind=WorkKind.ESSENTIAL)


class ReserveClock:
    """The reserve calendar's clock: an aware UTC instant that moves only when told to."""

    def __init__(self, now: datetime = PROTECTED) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, by: timedelta) -> None:
        self.now += by


def _token_env(name: str) -> str:
    return f'TEST_TOKEN_{name.upper().replace("-", "_")}'


def fake_tokens(account_names: Sequence[str] = POOL) -> dict[str, str]:
    """The environment that resolves a fake token for each account the accounts file names."""
    return {_token_env(name): f'fake-token-{name}' for name in account_names}


def write_reserve_accounts_file(
    directory: Path,
    account_names: Sequence[str] = POOL,
    block: Mapping[str, object] | None = DEFAULT_RESERVE_BLOCK,
) -> Path:
    """An accounts file naming *account_names* in order, with *block* as its reserve."""
    data: dict[str, object] = {
        'accounts': [{'name': name, 'oauth_token_env': _token_env(name)} for name in account_names]
    }
    if block is not None:
        data['reserve'] = dict(block)
    path = Path(directory) / 'accounts.yaml'
    path.write_text(yaml.safe_dump(data))
    return path


def reserve_gate(
    directory: Path,
    *,
    clock: ReserveClock,
    account_names: Sequence[str] = POOL,
    block: Mapping[str, object] | None = DEFAULT_RESERVE_BLOCK,
    cost_store: CostStore | None = None,
    **usage_cap: object,
) -> UsageGate:
    """A gate over *account_names*, every one open, its reserve calendar read from *clock*.

    The fake tokens are in the environment only while the gate is built, so no real
    credential is ever adopted.
    """
    config = UsageCapConfig.model_validate(
        {
            'accounts_file': str(write_reserve_accounts_file(directory, account_names, block)),
            'wait_for_reset': False,
            'auth_reprobe_enabled': False,
            **usage_cap,
        }
    )
    with patch.dict(os.environ, fake_tokens(account_names)):
        return UsageGate(config, cost_store=cost_store, reserve_clock=clock)


def cap_account(gate: UsageGate, name: str) -> None:
    """Cap *name* with no reset time, as a real CLI cap hit on it would."""
    others = [acct for acct in gate.account_names if acct != name]
    lease = gate.try_lease(exclude=others, work_class=_ESSENTIAL)
    if lease is None or lease.name != name:
        raise AssertionError(f'{name} is not open: {lease!r}')
    InvokeSlot(gate, lease).report(CapHit(reason='test cap', resets_at=None))


def cap_all_but_reserve(gate: UsageGate, reserve: str = RESERVE_ACCOUNT) -> None:
    """Cap every account but *reserve*: the B2 state when the clock is protected."""
    for name in gate.account_names:
        if name != reserve:
            cap_account(gate, name)
