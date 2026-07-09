"""§8 two-way boundary gate (B1-B10) for the W4 invocation-outcome seam.

This is task kappa, the SOLE leaf / terminal integration gate for wave W4.
Every producer artifact it exercises (classify_invocation, UsageGate's
AccountPhase state machine, InvokeSlot.report, invoke_with_cap_retry, the
per-(account,pid) probe config dirs, and the _shutting_down guard) already
exists and is green — this module owns NO production code, only the test
file itself.

Each B-scenario below drives a REAL UsageGate (never a MagicMock) through
invoke_with_cap_retry with a scripted fake CLI (invoke_fn=), so the full
production path — account selection -> lease -> classify_invocation ->
InvokeSlot.report -> UsageGate._transition -> cost-store attribution -- runs
end to end. Every scenario asserts BOTH sides of the seam: the producer-side
gate/account state AND the consumer-side observable (retry-loop outcome,
cost-store record, or steward-inherited loop behaviour).

See plans/invocation-outcome-prd.md §8 for the scenario catalogue this
module implements (B1-B10), and this task's plan.json design_decisions for
why a real gate (not a mock) drives every scenario and why B7 is tested at
the shared invoke_with_cap_retry seam rather than by importing
orchestrator.steward (shared/ must not import orchestrator/ — layering
doctrine).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.cli_invoke import (
    AgentResult,
    AllAccountsCappedException,
    invoke_with_cap_retry,
)
from shared.config_models import AccountConfig, UsageCapConfig
from shared.cost_store import CostStore
from shared.invocation_outcome import (
    OK,
    AuthFailed,
    CapHit,
    CliLocalError,
    Failure,
    InvocationOutcome,
    NearCap,
    ZeroOutputWedge,
    classify_invocation,
)
from shared.usage_gate import (
    _LEGAL_TRANSITIONS,
    AccountLease,
    AccountPhase,
    AccountState,
    IllegalTransitionError,
    InvokeSlot,
    UsageGate,
)

# ---------------------------------------------------------------------------
# Harness primitives shared by every B-scenario below (pre-1). Pure
# setup/fixtures -- no behavioural assertions live in this section.
# ---------------------------------------------------------------------------


def make_boundary_gate(
    names: list[str],
    *,
    cost_store: CostStore | None = None,
    wait_for_reset: bool = False,
    **cfg,
) -> UsageGate:
    """Build a REAL UsageGate wired for the two-way boundary harness.

    Mirrors test_cap_retry.make_gate / test_usage_gate.make_gate: fake
    per-account env tokens so ``UsageGate._init_accounts`` resolves real
    ``AccountState`` objects, and ``gate._run_probe`` AsyncMock'd so no real
    ``claude`` subprocess is ever spawned.

    Additionally neutralizes ``_start_account_resume_probe`` /
    ``_start_auth_reprobe`` (instance-level no-op MagicMocks) so entering
    CAPPED/AUTH_FAILED via ``_transition`` never schedules a real background
    asyncio task -- every transition stays synchronous and deterministic,
    which the property (B2) and atomicity (B6) scenarios depend on.

    B10 needs the REAL spawner to exercise the shutdown guard itself; it
    restores the class method post-construction via
    ``del gate._start_account_resume_probe`` (mirrors the established
    ``del gate._run_probe`` idiom in test_usage_gate.py's TestProbeConfigDirIsolation).
    """
    acct_cfgs = []
    env_vars: dict[str, str] = {}
    for name in names:
        env_key = f'TEST_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))

    config = UsageCapConfig(accounts=acct_cfgs, wait_for_reset=wait_for_reset, **cfg)
    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config, cost_store=cost_store)

    gate._run_probe = AsyncMock(return_value=True)
    gate._start_account_resume_probe = MagicMock()
    gate._start_auth_reprobe = MagicMock()
    return gate


def scripted_cli(*results_or_excs):
    """Build an async ``invoke_fn`` standing in for ``invoke_claude_agent``.

    Returns each item from *results_or_excs* in order on successive calls;
    an item that is an exception instance or exception class is raised
    instead of returned, so a single script can drive both success/cap/error
    results and mid-invocation exceptions (B6).

    Every call's kwargs are recorded, in order, on the returned callable's
    ``.calls`` attribute -- consumer-side assertions use this to inspect the
    prompt / resume_session_id / oauth_token each retry attempt actually
    used (B5, B7).

    Raises a loud ``AssertionError`` (not a bare ``StopIteration``, which
    asyncio would swallow/mis-report) if invoked more times than scripted --
    this doubles as the "no infinite retry" bounded-invocation signal (B4).
    """
    remaining = list(results_or_excs)
    calls: list[dict] = []

    async def _invoke(**kwargs):
        calls.append(kwargs)
        if not remaining:
            raise AssertionError(
                f'scripted_cli exhausted after {len(calls)} call(s) -- the retry '
                'loop invoked the fake CLI more times than scripted (unbounded retry?)'
            )
        item = remaining.pop(0)
        if isinstance(item, BaseException) or (
            isinstance(item, type) and issubclass(item, BaseException)
        ):
            raise item
        return item

    _invoke.calls = calls
    return _invoke


class RecordingCostStore:
    """Spy CostStore capturing ``save_invocation`` / ``save_account_event`` calls.

    Duck-types the async write surface of :class:`shared.cost_store.CostStore`
    that ``invoke_with_cap_retry`` and ``UsageGate._transition``'s cost-event
    side effects call, without touching a real sqlite file. Every call's
    keyword arguments are appended, in order, to ``.invocations`` /
    ``.account_events`` for consumer-side attribution assertions (B5).
    """

    def __init__(self) -> None:
        self.invocations: list[dict] = []
        self.account_events: list[dict] = []

    async def save_invocation(self, **kwargs) -> None:
        self.invocations.append(kwargs)

    async def save_account_event(self, **kwargs) -> None:
        self.account_events.append(kwargs)


# -- B3 golden-corpus loader --------------------------------------------------
#
# Reuses (does not duplicate) the checked-in corpus consumed by
# test_invocation_outcome.py's B3 -- see fixtures/cap_strings/README.md for
# the record schema and provenance.

_CORPUS_PATH = Path(__file__).parent / 'fixtures' / 'cap_strings' / 'corpus.json'

_VARIANT_CLASSES: dict[str, type[InvocationOutcome]] = {
    'OK': OK,
    'CapHit': CapHit,
    'NearCap': NearCap,
    'AuthFailed': AuthFailed,
    'CliLocalError': CliLocalError,
    'ZeroOutputWedge': ZeroOutputWedge,
    'Failure': Failure,
}


def load_corpus() -> list[dict]:
    """Load the checked-in cap-string golden corpus records."""
    with _CORPUS_PATH.open() as f:
        return json.load(f)


def agent_result_from_record(record: dict) -> AgentResult:
    """Build an AgentResult from a corpus record.

    Mirrors test_invocation_outcome.py's ``_agent_result_from_record``:
    ``success``/``output`` are AgentResult's no-default positional fields, so
    this helper supplies the README-documented defaults explicitly; every
    other field defaults exactly as AgentResult itself defaults.
    """
    return AgentResult(
        success=record.get('success', False),
        output=record.get('output', ''),
        stderr=record.get('stderr', ''),
        turns=record.get('turns', 0),
        cost_usd=record.get('cost_usd', 0.0),
        timed_out=record.get('timed_out', False),
        transcript_turns=record.get('transcript_turns'),
        api_error_status=record.get('api_error_status'),
    )


# -- DD-5 _open invariant + background-task drain (mirrors test_usage_gate.py) -

def _open_invariant_holds(gate: UsageGate) -> bool:
    """DD-5: gate._open.is_set() <=> any account in {AVAILABLE, PROBING}."""
    expected_open = any(
        a.phase in (AccountPhase.AVAILABLE, AccountPhase.PROBING) for a in gate._accounts
    )
    return gate._open.is_set() == expected_open


async def _drain_task(task: asyncio.Task | None) -> None:
    """Cancel *task* and await its settling so it doesn't leak past the test."""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _drain_bg(gate: UsageGate) -> None:
    """Cancel + await every account's resume/auth-reprobe background task."""
    for acct in gate._accounts:
        await _drain_task(acct.resume_task)
        await _drain_task(acct.auth_reprobe_task)


# ---------------------------------------------------------------------------
# B1 -- an illegal phase edge raises IllegalTransitionError without mutating
# state (producer), and leaves the gate usable for a sibling account
# (consumer): before_invoke() still leases the sibling, DD-5 _open holds.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB1IllegalTransition:
    """AUTH_FAILED -> PROBE_IN_FLIGHT is not in _LEGAL_TRANSITIONS."""

    async def test_illegal_edge_raises_state_unchanged_and_gate_stays_usable(self):
        gate = make_boundary_gate(['acct-a', 'acct-b'])
        auth_failed_acct, sibling = gate._accounts

        # Producer side: reach AUTH_FAILED via a legal edge, then attempt an
        # illegal one.
        gate._transition(auth_failed_acct, AccountPhase.AUTH_FAILED)
        assert auth_failed_acct.phase == AccountPhase.AUTH_FAILED

        with pytest.raises(IllegalTransitionError):
            gate._transition(auth_failed_acct, AccountPhase.PROBE_IN_FLIGHT)

        assert auth_failed_acct.phase == AccountPhase.AUTH_FAILED  # unchanged

        # Consumer side: the gate is not corrupted by the failed transition --
        # the sibling AVAILABLE account is still leasable, and the DD-5 _open
        # invariant holds.
        await _assert_sibling_still_usable(gate, sibling.name)
