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


# ---------------------------------------------------------------------------
# B4 -- reify-3604: a NON_CAP_CLI_ERROR_MARKERS string in stderr classifies
# as CliLocalError and NEVER CapHit, even combined with cap-like text
# (producer/classifier side); driving that result through the fake-CLI
# harness leaves the account un-capped with a bounded, non-infinite retry
# loop (consumer side).
# ---------------------------------------------------------------------------


async def _drive_non_cap_error_and_assert_bounded(result: AgentResult) -> None:
    """B4 consumer-side drive: replay *result* (a NON_CAP_CLI_ERROR_MARKERS
    failure) through invoke_with_cap_retry's fake-CLI harness and assert (a)
    the selected account is NOT transitioned to CAPPED and (b) the retry
    loop is bounded -- the fake CLI is invoked exactly once, never looped.

    Uses a 2-account gate (mirrors _drive_caphit_record_to_capped) plus a
    low max_cap_retries as a second, independent bound layered on top of
    scripted_cli's own "exhausted" guard: if CliLocalError's precedence
    over CapHit ever regressed (the zero-cost heuristic net in
    invoke_with_cap_retry re-classifying this as a cap hit), either bound
    turns what would be an infinite retry into a loud, immediate failure
    instead of a hang.
    """
    gate = make_boundary_gate(['acct-0', 'acct-1'])
    acct = gate._accounts[0]
    cli = scripted_cli(result)

    with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
        returned = await invoke_with_cap_retry(
            gate, 'B4[non-cap-marker]',
            invoke_fn=cli,
            max_cap_retries=2,
            backend='claude',
            prompt='hi',
        )

    assert returned.success is False, 'expected the local-error result to be returned as-is'
    assert len(cli.calls) == 1, (
        f'expected exactly one invocation, the retry loop made {len(cli.calls)} -- '
        'a NON_CAP marker must never trigger cap-retry'
    )
    assert acct.phase != AccountPhase.CAPPED, (
        f'expected acct-0 to remain un-capped after a NON_CAP_CLI_ERROR_MARKERS '
        f'failure, got {acct.phase}'
    )


@pytest.mark.asyncio
class TestB4Reify3604NonCap:
    """A NON_CAP_CLI_ERROR_MARKERS string always classifies as CliLocalError,
    never CapHit -- reify-3604's structural fix (producer/classifier side) --
    and driving it through invoke_with_cap_retry leaves the account un-capped
    with a bounded, non-infinite retry loop (consumer side)."""

    async def test_non_cap_marker_never_caps_account_and_loop_stays_bounded(self):
        # A --session-id collision (NON_CAP_CLI_ERROR_MARKERS) whose output
        # ALSO carries a cap-hit prefix + confirm keyword -- the adversarial
        # case reify-3604 exists for: CliLocalError must outrank CapHit even
        # when both patterns co-occur in the same output.
        result = AgentResult(
            success=False,
            output="You've hit your usage limit. Your plan resets in 3h.",
            stderr='Error: Session ID abc-123 is already in use.',
        )

        # Producer/classifier side.
        outcome = classify_invocation(result, strict_confirm=True, backend='claude')
        assert isinstance(outcome, CliLocalError)
        assert outcome.marker == 'is already in use'
        assert not isinstance(outcome, CapHit)

        # Consumer side: driven through the fake-CLI harness end-to-end.
        await _drive_non_cap_error_and_assert_bounded(result)


# ---------------------------------------------------------------------------
# B5 -- account[0] forced PROBE_IN_FLIGHT (simulating a concurrent prober
# already holding its slot) must not skew attribution: before_invoke's lease
# (producer side) and invoke_with_cap_retry's cost-store record (consumer
# side) must both name account[1], the account actually leased and invoked
# -- not account[0], closing the pre-delta name-skew (finding 3 / boundary
# test B5).
# ---------------------------------------------------------------------------


async def _assert_probe_in_flight_skew_does_not_skew_attribution(gate: UsageGate) -> None:
    """B5 support: drive account[0] to PROBE_IN_FLIGHT via the legal
    CAPPED -> PROBING -> PROBE_IN_FLIGHT chain (simulating a concurrent
    prober that already claimed its slot), then assert neither
    ``before_invoke``'s lease (producer side) nor ``invoke_with_cap_retry``'s
    cost-store attribution (consumer side) skews toward the PROBE_IN_FLIGHT
    account -- both must name account[1], the account actually leased and
    invoked (finding 3 / task W4-delta's AccountLease fix).
    """
    skewed, sibling = gate._accounts
    gate._transition(skewed, AccountPhase.CAPPED)
    gate._transition(skewed, AccountPhase.PROBING)
    gate._transition(skewed, AccountPhase.PROBE_IN_FLIGHT)
    assert skewed.phase == AccountPhase.PROBE_IN_FLIGHT

    # Producer side: before_invoke must skip the PROBE_IN_FLIGHT account and
    # lease the sibling instead.
    lease = await gate.before_invoke()
    assert lease is not None
    assert lease.name == sibling.name

    # Consumer side: the cost-store attribution recorded by
    # invoke_with_cap_retry must name the account actually invoked (the
    # sibling), not the skewed PROBE_IN_FLIGHT account.
    cost_store = RecordingCostStore()
    ok_result = AgentResult(success=True, output='done', cost_usd=0.02)
    with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
        result = await invoke_with_cap_retry(
            gate, 'B5[attribution]',
            invoke_fn=scripted_cli(ok_result),
            cost_store=cost_store,
            backend='claude',
            prompt='hi',
        )

    assert result.success is True
    assert len(cost_store.invocations) == 1
    assert cost_store.invocations[0]['account_name'] == sibling.name, (
        f'expected cost-store attribution to name {sibling.name!r} (the '
        f'account actually invoked), got '
        f'{cost_store.invocations[0]["account_name"]!r}'
    )


@pytest.mark.asyncio
class TestB5AttributionUnderProbeInFlightSkew:
    """account[0] PROBE_IN_FLIGHT must not skew before_invoke's lease
    identity or invoke_with_cap_retry's cost-store attribution away from
    account[1], the account actually leased and invoked."""

    async def test_probe_in_flight_skew_does_not_skew_lease_or_attribution(self):
        gate = make_boundary_gate(['acct-0', 'acct-1'])
        await _assert_probe_in_flight_skew_does_not_skew_attribution(gate)


# ---------------------------------------------------------------------------
# B6 -- InvokeSlot.report() applies its matching gate transition AND settles
# the slot atomically (producer side), so invoke_slot()'s __aexit__ safety
# net never double-releases/re-transitions once report() has already run.
# When the invoked fn raises BEFORE report() ever gets a chance to run at
# all, that SAME __aexit__ safety net still releases the claimed probe slot
# on the exception exit path (consumer side).
# ---------------------------------------------------------------------------


async def _assert_probe_released_after_exception(gate: UsageGate, acct: AccountState) -> None:
    """B6 support: after an exception aborts an invocation mid-flight --
    before ``report()`` ever gets a chance to run -- the claimed probe slot
    must still be released: the account is back to AVAILABLE (not stuck in
    PROBE_IN_FLIGHT), and the gate's DD-5 ``_open`` invariant holds."""
    assert acct.phase == AccountPhase.AVAILABLE
    assert _open_invariant_holds(gate)


@pytest.mark.asyncio
class TestB6ReportAtomicity:
    """InvokeSlot.report() applies its matching gate transition and settles
    the slot atomically -- so invoke_slot()'s __aexit__ safety net never
    double-releases/re-transitions once report() has run, and still
    releases the claimed probe slot when the invoked fn raises before
    report() ever gets a chance to run at all."""

    async def test_report_settles_and_transitions_atomically(self):
        gate = make_boundary_gate(['acct-a'])
        acct = gate._accounts[0]
        gate._transition(acct, AccountPhase.CAPPED)
        gate._transition(acct, AccountPhase.PROBING)

        async with gate.invoke_slot() as slot:
            assert acct.phase == AccountPhase.PROBE_IN_FLIGHT  # claimed on entry
            slot.report(CapHit(resets_at=None, reason='x'))
            assert slot._settled is True
            assert acct.phase == AccountPhase.CAPPED

        # Producer side: __aexit__ found the slot already settled and did
        # NOT double-release/re-transition -- the CAPPED verdict from
        # report() stands untouched.
        assert acct.phase == AccountPhase.CAPPED

    async def test_exception_before_report_still_releases_probe_slot(self):
        gate = make_boundary_gate(['acct-0', 'acct-1'])
        probing_acct = gate._accounts[0]
        gate._transition(probing_acct, AccountPhase.CAPPED)
        gate._transition(probing_acct, AccountPhase.PROBING)

        with pytest.raises(RuntimeError, match='boom'):
            with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
                await invoke_with_cap_retry(
                    gate, 'B6[exception]',
                    invoke_fn=scripted_cli(RuntimeError('boom')),
                    backend='claude',
                    prompt='hi',
                )

        # Consumer side: the exception propagated (not swallowed), and the
        # account is not left stuck in PROBE_IN_FLIGHT.
        await _assert_probe_released_after_exception(gate, probing_acct)


# ---------------------------------------------------------------------------
# B7 -- the shared invoke_with_cap_retry loop's zero-output-wedge fresh-
# fallback guard -- the exact mechanism the steward inherits via its
# rebuild_prompt + max_cap_retries wiring (task W4-eta), tested at THIS
# shared seam rather than by importing orchestrator.steward (shared/ must
# not import orchestrator/ -- see this task's plan design_decisions). A
# cap-hit resume that then wedges (zero-output timeout on the resumed
# session) triggers the rebuild_prompt(True) fresh-session fallback and
# clears the wedged resume_session_id rather than perpetuating it (consumer
# side); the loop terminates bounded, and the gate is left consistent
# afterward (producer side).
# ---------------------------------------------------------------------------


async def _assert_gate_consistent_after_retry_sequence(gate: UsageGate) -> None:
    """B7 support: after a scripted cap/resume/wedge/success sequence, the
    gate must be left consistent -- no account stuck in PROBE_IN_FLIGHT, and
    the DD-5 ``_open`` invariant holds."""
    assert _open_invariant_holds(gate)
    assert not any(a.phase == AccountPhase.PROBE_IN_FLIGHT for a in gate._accounts)


@pytest.mark.asyncio
class TestB7StewardWedgeGuardInherited:
    """Scripted cap-hit (with session_id) -> resume -> zero-output-wedge ->
    success. Tested at the SHARED invoke_with_cap_retry seam -- the exact
    loop the steward routes every invocation through (task W4-eta)."""

    async def test_cap_resume_then_wedge_triggers_fresh_fallback_and_terminates_bounded(self):
        cap_result = AgentResult(
            success=False,
            output="You've hit your usage limit for Claude Pro. Your plan resets in 3 hours.",
            session_id='sess-cap-123',
        )
        wedge_result = AgentResult(
            success=False,
            output='',
            timed_out=True,
            transcript_turns=0,
        )
        ok_result = AgentResult(success=True, output='done', cost_usd=0.02)

        gate = make_boundary_gate(['acct-0', 'acct-1'])
        cli = scripted_cli(cap_result, wedge_result, ok_result)
        rebuild_calls: list[bool] = []

        async def rebuild_prompt(session_lost: bool) -> str:
            rebuild_calls.append(session_lost)
            return 'rebuilt fresh prompt'

        with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
            result = await invoke_with_cap_retry(
                gate, 'B7[wedge-guard]',
                invoke_fn=cli,
                rebuild_prompt=rebuild_prompt,
                max_cap_retries=16,
                backend='claude',
                prompt='original task prompt',
            )

        # Consumer side: the fresh-session fallback fired exactly once, the
        # wedged session was never re-resumed, and the loop terminated
        # bounded with the eventual success.
        assert result.success is True
        assert len(cli.calls) == 3
        assert rebuild_calls == [True]
        assert cli.calls[1].get('resume_session_id') == 'sess-cap-123'
        assert cli.calls[2].get('resume_session_id') is None
        assert cli.calls[2]['prompt'] == 'rebuilt fresh prompt'

        # Producer side: the gate is left consistent afterward.
        await _assert_gate_consistent_after_retry_sequence(gate)


# ---------------------------------------------------------------------------
# B8 -- with every account CAPPED, _on_sighup_async() force-resets each
# account to AVAILABLE through one _transition(force=True) call apiece (the
# operator-driven SIGHUP hard reset, DD-4) (producer side), reopening the
# gate so before_invoke() returns immediately for callers (consumer side).
# ---------------------------------------------------------------------------


def _cap_every_account(gate: UsageGate) -> None:
    """B8 support: drive every account in *gate* to CAPPED via the real
    AVAILABLE -> CAPPED legal edge, so the gate is fully closed before the
    SIGHUP hard-reset is exercised."""
    for acct in gate._accounts:
        gate._transition(acct, AccountPhase.CAPPED)


@pytest.mark.asyncio
class TestB8SighupUncaps:
    """All accounts CAPPED, then _on_sighup_async() force-resets every
    account to AVAILABLE via one _transition(force=True) call each (producer
    side) and reopens the gate for callers immediately (consumer side)."""

    async def test_sighup_forces_every_account_available_and_reopens_gate(self):
        gate = make_boundary_gate(['a', 'b', 'c'], wait_for_reset=True)
        _cap_every_account(gate)
        assert gate._open.is_set() is False
        assert _open_invariant_holds(gate)

        gate._reprobe_account = AsyncMock(return_value=None)
        with (
            patch.object(gate, '_transition', wraps=gate._transition) as mock_transition,
            patch('shared.usage_gate.load_dotenv'),
        ):
            await gate._on_sighup_async()

        # Producer side: each account reached AVAILABLE through exactly one
        # forced _transition call -- the SIGHUP hard-reset path (DD-4).
        force_calls = [c for c in mock_transition.call_args_list if c.kwargs.get('force')]
        assert len(force_calls) == len(gate._accounts)
        for acct in gate._accounts:
            assert acct.phase == AccountPhase.AVAILABLE

        # Consumer side: the gate is reopened for callers immediately.
        assert gate._open.is_set() is True
        lease = await asyncio.wait_for(gate.before_invoke(), timeout=1.0)
        assert lease is not None

        await _drain_bg(gate)


# ---------------------------------------------------------------------------
# B9 -- per-(account,pid) probe config-dir isolation: every account gets its
# own usage-gate-probe-<name>-<pid> dir (producer/isolation side, fully
# deterministic), and _run_probe injects that account's own token as
# CLAUDE_CODE_OAUTH_TOKEN into the probe subprocess env alongside its own
# CLAUDE_CONFIG_DIR (consumer side, the precedence MECHANISM). The live
# env-vs-config-dir precedence OBSERVATION itself is delegated to the
# existing guarded MUST-OBSERVE test in test_usage_gate.py (task 2139),
# which requires a real claude CLI + a live OAuth account and is therefore
# not re-run here -- see this task's plan design_decisions.
# ---------------------------------------------------------------------------


async def _capture_probe_envs(gate: UsageGate, accts: list[AccountState]) -> list[dict]:
    """B9 support: call the REAL ``UsageGate._run_probe`` for each of *accts*
    (bypassing the harness's AsyncMock) with ``asyncio.create_subprocess_exec``
    faked, and return the subprocess env dict captured for each call, in
    order -- lets the test assert the CLAUDE_CODE_OAUTH_TOKEN/CLAUDE_CONFIG_DIR
    injection without spawning a real claude CLI."""
    del gate._run_probe  # remove the harness's instance-level mock; fall back
    # to the real UsageGate._run_probe under test (mirrors the established
    # `del gate._run_probe` idiom in test_usage_gate.py).
    captured: list[dict] = []

    async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
        captured.append(kwargs.get('env'))
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b'{"ok":true}', b''))
        return proc

    with patch('shared.usage_gate.asyncio.create_subprocess_exec', side_effect=fake_exec):
        for acct in accts:
            await gate._run_probe(acct)

    return captured


class TestB9ProbeDirIsolationAndEnvPrecedence:
    """Deterministic isolation (producer) + env-injection mechanism
    (consumer) halves of B9; the live env-vs-config-dir precedence
    observation is delegated to
    test_usage_gate.TestProbeEnvTokenPrecedenceGuard (task 2139)."""

    def test_probe_config_dirs_keyed_by_name_with_distinct_paths(self):
        gate = make_boundary_gate(['acct-a', 'acct-b'])
        pid = os.getpid()
        acct_a, acct_b = gate._accounts

        assert set(gate._probe_config_dirs.keys()) == {'acct-a', 'acct-b'}
        dir_a, dir_b = gate._config_dir_for(acct_a), gate._config_dir_for(acct_b)
        assert dir_a.path != dir_b.path
        assert f'usage-gate-probe-acct-a-{pid}' in dir_a.path.name
        assert f'usage-gate-probe-acct-b-{pid}' in dir_b.path.name

    @pytest.mark.asyncio
    async def test_run_probe_injects_oauth_token_and_per_account_config_dir(self):
        gate = make_boundary_gate(['acct-a', 'acct-b'])
        acct_a, acct_b = gate._accounts

        envs = await _capture_probe_envs(gate, [acct_a, acct_b])

        env_a, env_b = envs
        assert env_a['CLAUDE_CODE_OAUTH_TOKEN'] == acct_a.token
        assert env_b['CLAUDE_CODE_OAUTH_TOKEN'] == acct_b.token
        assert env_a['CLAUDE_CONFIG_DIR'] == str(gate._config_dir_for(acct_a).path)
        assert env_b['CLAUDE_CONFIG_DIR'] == str(gate._config_dir_for(acct_b).path)
        assert env_a['CLAUDE_CONFIG_DIR'] != env_b['CLAUDE_CONFIG_DIR']


# ---------------------------------------------------------------------------
# B10 -- gate.shutdown() arms _shutting_down FIRST, before cancelling or
# draining anything (its own docstring's ordering guarantee), so a CapHit
# _transition racing -- or arriving after -- that teardown spawns no new
# resume/probe task (producer side). The refusal lives inside the spawner
# method itself (gate-internal): calling it is not something a caller must
# additionally guard with its own cancel-before-shutdown discipline, and it
# still refuses no matter which call path reaches it (consumer side).
# ---------------------------------------------------------------------------


async def _assert_no_resume_task_spawned_via_any_path(gate: UsageGate, acct: AccountState) -> None:
    """B10 support: after gate.shutdown() has armed _shutting_down, neither
    a CAPPED _transition's enter-phase side effect (producer -- the normal
    call path) nor a direct call to the spawner method itself (consumer --
    any caller reaching _start_account_resume_probe by whatever path)
    starts a new resume task. The refusal lives in the spawner
    (gate-internal), not in any cancel-before-shutdown discipline a caller
    must separately observe."""
    assert acct.resume_task is None or acct.resume_task.done()
    gate._start_account_resume_probe(acct)  # direct call, bypassing _transition entirely
    assert acct.resume_task is None or acct.resume_task.done()


@pytest.mark.asyncio
class TestB10ShutdownRefusesProbes:
    """await gate.shutdown() then a CapHit transition spawns no resume
    task -- the refusal is gate-internal, not caller-ordered."""

    async def test_shutdown_then_caphit_spawns_no_resume_task(self):
        gate = make_boundary_gate(['acct-a'], wait_for_reset=True)
        acct = gate._accounts[0]
        # B10 needs the REAL spawner to exercise the shutdown guard itself
        # -- make_boundary_gate neutralizes it with a no-op MagicMock by
        # default so every other scenario's transitions stay synchronous.
        del gate._start_account_resume_probe
        assert acct.phase == AccountPhase.AVAILABLE

        await gate.shutdown()
        assert gate._shutting_down is True

        # Invoked directly, exactly as any caller/consumer would -- no
        # cancel-before-shutdown discipline on our part.
        gate._transition(acct, AccountPhase.CAPPED)
        assert acct.phase == AccountPhase.CAPPED

        await _assert_no_resume_task_spawned_via_any_path(gate, acct)
