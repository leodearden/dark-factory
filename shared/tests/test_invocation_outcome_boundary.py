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
import random
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


async def _assert_sibling_still_usable(gate: UsageGate, expected_name: str) -> None:
    """B1 support: an IllegalTransitionError on one account must not corrupt
    the gate -- a sibling account can still be leased, and the DD-5 _open
    invariant holds."""
    assert _open_invariant_holds(gate)
    lease = await gate.before_invoke()
    assert lease is not None
    assert lease.name == expected_name


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


# ---------------------------------------------------------------------------
# B2 -- the DD-5 _open invariant over seeded random legal-transition walks
# (producer), plus the consumer-facing wait_for_open()/before_invoke()
# observable tracking it (consumer).
# ---------------------------------------------------------------------------

_B2_SAMPLE_EVERY = 10  # how often the walk samples the consumer observable


async def _random_legal_walk(gate: UsageGate, rng: random.Random, steps: int) -> None:
    """B2 walk driver: *steps* random legal ``_transition`` edges.

    After EVERY step, asserts the DD-5 ``_open`` invariant (producer side).
    Every ``_B2_SAMPLE_EVERY`` steps, also asserts the one-directional
    consumer-facing claim this task's plan specifies: whenever ``_open`` is
    set, ``wait_for_open(timeout=0)`` returns True too (consumer side) --
    sampled rather than checked on every step purely to keep a 300-step walk
    fast, not because the invariant is expected to ever lapse between
    samples.

    NOTE -- deliberately one-directional: ``wait_for_open``'s fast path
    short-circuits on ``UsageGate.is_paused`` (``all(capped or auth_failed)``
    accounts), which is a narrower condition than ``not _open.is_set()``
    (``all(phase not in {AVAILABLE, PROBING})``) -- a PROBE_IN_FLIGHT
    account is neither AVAILABLE/PROBING (so ``_open`` is clear) nor
    capped/auth_failed (so ``is_paused`` is False too), so
    ``wait_for_open(0)`` can transiently return True while ``_open.is_set()``
    is False during that in-flight probe window. That gray zone is real but
    self-resolves within one invocation and is out of scope for this
    property (the plan's own B2 language only claims the two unambiguous
    directions: open implies wait_for_open returns, and all-capped implies
    it blocks -- see test_consumer_wait_for_open_false_when_all_capped for
    the second). Asserting full bidirectional equivalence here would be a
    stronger claim than the seam actually guarantees.

    ``wait_for_open(timeout=0)`` is used instead of ``before_invoke()``
    for the mid-walk sample because ``before_invoke()`` blocks
    indefinitely whenever the walk has landed on "every account capped" --
    exactly the state B2 must also exercise.
    """
    assert _open_invariant_holds(gate)
    for i in range(steps):
        acct = rng.choice(gate._accounts)
        legal = sorted(_LEGAL_TRANSITIONS.get(acct.phase, frozenset()))
        if not legal:
            continue
        gate._transition(acct, rng.choice(legal))
        assert _open_invariant_holds(gate)
        if i % _B2_SAMPLE_EVERY == 0 and gate._open.is_set():
            assert await gate.wait_for_open(timeout=0) is True


@pytest.mark.asyncio
class TestB2OpenInvariantProperty:
    """_open.is_set() <=> any account in {AVAILABLE, PROBING} holds after
    every legal _transition edge over a long seeded random walk, and the
    consumer-facing wait_for_open()/before_invoke() observable agrees."""

    async def test_random_walk_two_accounts_preserves_invariant_and_consumer_tracks_it(self):
        gate = make_boundary_gate(['a', 'b'])
        rng = random.Random(20260709)
        await _random_legal_walk(gate, rng, steps=300)
        await _drain_bg(gate)

    async def test_random_walk_three_accounts_preserves_invariant_and_consumer_tracks_it(self):
        gate = make_boundary_gate(['a', 'b', 'c'])
        rng = random.Random(975318642)
        await _random_legal_walk(gate, rng, steps=300)
        await _drain_bg(gate)

    async def test_consumer_before_invoke_returns_promptly_when_open(self):
        gate = make_boundary_gate(['a', 'b'])
        assert gate._open.is_set()

        lease = await asyncio.wait_for(gate.before_invoke(), timeout=1.0)

        assert lease is not None
        assert lease.name in {'a', 'b'}

    async def test_consumer_wait_for_open_false_when_all_capped(self):
        gate = make_boundary_gate(['a', 'b'])
        for acct in gate._accounts:
            gate._transition(acct, AccountPhase.CAPPED)
        assert _open_invariant_holds(gate)

        assert gate._open.is_set() is False
        assert await gate.wait_for_open(timeout=0) is False

        await _drain_bg(gate)


# ---------------------------------------------------------------------------
# B3 -- the checked-in golden corpus classifies as recorded (producer/
# classifier side), and every CapHit record drives the selected account to
# CAPPED end-to-end when replayed through invoke_with_cap_retry (consumer
# side, the boundary value-add beyond test_invocation_outcome.py's B3).
# ---------------------------------------------------------------------------

_CORPUS = load_corpus()
_CORPUS_IDS = [r['id'] for r in _CORPUS]
_CAP_HIT_RECORDS = [r for r in _CORPUS if r['expected'] == 'CapHit']
_CAP_HIT_IDS = [r['id'] for r in _CAP_HIT_RECORDS]


class TestB3ClassifierGoldenCorpus:
    """Producer/classifier side: every corpus record classifies as recorded."""

    @pytest.mark.parametrize('record', _CORPUS, ids=_CORPUS_IDS)
    def test_corpus_record_classifies_as_expected(self, record):
        result = agent_result_from_record(record)
        outcome = classify_invocation(
            result,
            strict_confirm=record['strict_confirm'],
            backend=record.get('backend', 'claude'),
        )

        expected_cls = _VARIANT_CLASSES[record['expected']]
        assert isinstance(outcome, expected_cls), (
            f'{record["id"]}: expected {record["expected"]}, '
            f'got {type(outcome).__name__} ({outcome!r})'
        )

        resets_at_expectation = record.get('resets_at')
        if record['expected'] == 'CapHit' and resets_at_expectation == 'set':
            assert isinstance(outcome, CapHit)
            assert outcome.resets_at is not None, (
                f'{record["id"]}: expected resets_at to be set, got None'
            )
        elif record['expected'] == 'CapHit' and resets_at_expectation == 'none':
            assert isinstance(outcome, CapHit)
            assert outcome.resets_at is None, (
                f'{record["id"]}: expected resets_at to be None, got {outcome.resets_at!r}'
            )


async def _drive_caphit_record_to_capped(record: dict) -> None:
    """B3 consumer-side drive: replay *record* (a CapHit corpus record)
    through invoke_with_cap_retry's fake-CLI harness and assert the
    selected account transitions to AccountPhase.CAPPED end-to-end.

    Uses a 2-account gate so the retry loop can complete: the first
    (scripted) invocation delivers the corpus record's cap output on
    account[0], which invoke_with_cap_retry detects (via the confirmed
    detect_cap_hit path, or the zero-cost heuristic fallback when the
    record's own strict_confirm regime doesn't match the loop's
    hardcoded strict_confirm=True -- every corpus record defaults
    duration_ms/turns/cost_usd to values that trip that heuristic net
    regardless) and transitions to CAPPED; the loop then fails over to
    account[1], whose scripted OK result ends the retry.
    """
    gate = make_boundary_gate(['acct-0', 'acct-1'])
    capped_acct = gate._accounts[0]
    cap_result = agent_result_from_record(record)
    ok_result = AgentResult(success=True, output='done', cost_usd=0.01)

    with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
        result = await invoke_with_cap_retry(
            gate, f'B3[{record["id"]}]',
            invoke_fn=scripted_cli(cap_result, ok_result),
            backend=record.get('backend', 'claude'),
            prompt='hi',
        )

    assert result.success is True, f'{record["id"]}: expected the loop to end in success'
    assert capped_acct.phase == AccountPhase.CAPPED, (
        f'{record["id"]}: expected acct-0 to be CAPPED after the cap-hit '
        f'record, got {capped_acct.phase}'
    )


@pytest.mark.asyncio
class TestB3CapHitDrivesAccountCapped:
    """Consumer side (two-way): every CapHit corpus record, replayed through
    the fake-CLI harness, transitions the selected account to CAPPED."""

    @pytest.mark.parametrize('record', _CAP_HIT_RECORDS, ids=_CAP_HIT_IDS)
    async def test_cap_hit_record_drives_account_to_capped(self, record):
        await _drive_caphit_record_to_capped(record)
