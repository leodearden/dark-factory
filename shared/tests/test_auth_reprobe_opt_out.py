"""A caller can keep an auth-failed account out for the life of its gate (task 6042).

By default an AUTH_FAILED account is re-probed every ``auth_reprobe_secs``, and
each attempt reloads ``.env`` with ``override=True`` before spending a CLI call.
That suits the orchestrator, whose gate lives for days and whose operator fixes
a token by editing ``.env``. It does not suit a nightly oneshot (the legibility
trickle and census): the reload can put back an ``ANTHROPIC_API_KEY`` the
process stripped, and the probe spends a call on a token rejected minutes ago.
``auth_reprobe_enabled=False`` turns the re-probe off.

The interval is the loop's one-second floor and ``_run_probe`` is make_gate's
always-succeeding mock, so the default arm recovers within seconds — which is
what makes the opted-out arm's silence mean something.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from _usage_gate_test_helpers import make_gate

from shared.invocation_outcome import AuthFailed

_REPROBE_INTERVAL_SECS = 1
_RECOVERY_DEADLINE_SECS = 5.0


async def _auth_fail_the_first_account(gate) -> None:
    async with gate.invoke_slot() as slot:
        slot.report(AuthFailed(status=401))


async def _recovered_within(gate, deadline_secs: float) -> bool:
    loop = asyncio.get_running_loop()
    give_up_at = loop.time() + deadline_secs
    while gate.auth_failed_account_names and loop.time() < give_up_at:
        await asyncio.sleep(0.05)
    return not gate.auth_failed_account_names


@pytest.fixture(autouse=True)
def _no_dotenv_reload():
    """A re-probe reloads the nearest ``.env`` into this process; never the repo's."""
    with patch('shared.usage_gate.load_dotenv'):
        yield


async def test_by_default_an_auth_failed_account_is_reprobed_back_in():
    gate = make_gate(['a', 'b'], auth_reprobe_secs=_REPROBE_INTERVAL_SECS)
    try:
        await _auth_fail_the_first_account(gate)
        assert gate.auth_failed_account_names == ('a',)

        assert await _recovered_within(gate, _RECOVERY_DEADLINE_SECS)
    finally:
        await gate.shutdown()


async def test_opted_out_an_auth_failed_account_stays_out():
    gate = make_gate(
        ['a', 'b'], auth_reprobe_secs=_REPROBE_INTERVAL_SECS, auth_reprobe_enabled=False,
    )
    try:
        await _auth_fail_the_first_account(gate)

        await asyncio.sleep(_REPROBE_INTERVAL_SECS * 2)

        assert gate.auth_failed_account_names == ('a',)
    finally:
        await gate.shutdown()
