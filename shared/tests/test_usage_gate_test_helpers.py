"""Contract tests for the shared ``_usage_gate_test_helpers`` module.

Locks the runtime behaviour of the extracted usage-gate test helpers
(``make_gate`` / ``make_mock_cost_store`` / ``set_scope_cap`` /
``scripted_invoke`` / ``cap_result`` / ``ok_result`` / ``SCOPE``) so the
dedup refactor across the scope + cap-retry suites cannot silently drift
their construction semantics. Runtime-behaviour assertions only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from _usage_gate_test_helpers import (
    SCOPE,
    cap_result,
    make_gate,
    make_mock_cost_store,
    ok_result,
    scripted_invoke,
    set_scope_cap,
)

from shared.cli_invoke import AgentResult
from shared.usage_gate import ScopeCap


def test_make_gate_defaults_wait_for_reset_false_and_single_account():
    gate = make_gate(['a'])
    assert gate._config.wait_for_reset is False
    assert len(gate._accounts) == 1


def test_make_gate_threads_cost_store():
    sentinel = object()
    gate = make_gate(['a'], cost_store=sentinel)
    assert gate._cost_store is sentinel


def test_make_mock_cost_store_exposes_async_hooks():
    store = make_mock_cost_store()
    assert isinstance(store.save_account_event, AsyncMock)
    assert isinstance(store.save_invocation, AsyncMock)


def test_set_scope_cap_installs_capped_overlay():
    assert SCOPE == 'claude-fable-5'
    gate = make_gate(['a'])
    acct = gate._accounts[0]
    sc = set_scope_cap(acct)
    assert isinstance(sc, ScopeCap)
    assert acct.scope_caps[SCOPE].capped is True


async def test_scripted_invoke_yields_in_order_and_records_calls():
    r_ok = ok_result()
    r_cap = cap_result('x')
    si = scripted_invoke(r_ok, r_cap)
    first = await si(model='m')
    second = await si(model='m')
    assert first is r_ok
    assert second is r_cap
    assert si.calls == [{'model': 'm'}, {'model': 'm'}]


def test_cap_result_shape():
    r = cap_result('x')
    assert isinstance(r, AgentResult)
    assert r.success is True
    assert r.stderr == 'x'


def test_ok_result_shape():
    r = ok_result()
    assert isinstance(r, AgentResult)
    assert r.success is True
    assert r.output == 'complete'
