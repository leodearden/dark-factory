"""Tests that pin how one architect cell rides the account usage gate.

The cell builds the gate or receives it from a campaign owner, fails over on a
cap, waits out cap patience, and leaves a campaign-owned gate to its owner.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _architect_eval_harness import (
    DEFAULT_JUDGE_PLAN_QUALITY,
    architect_prompt_sent,
    only_record,
    run_architect_eval_hermetic,
)
from _architect_plan_shapes import well_formed_plan
from _campaign_gate_helpers import eval_base_config
from _eval_refusal import REFUSAL_TEXT, refused_agent_result

from orchestrator.evals.metrics import RoleUnderTest

# ---------------------------------------------------------------------------
# UsageGate construction in run_architect_eval (eval-revival φ, task 3630)
#
# The architect eval was the ONE eval entry point invoking an agent with no
# account failover: run_eval (runner.py:401-407) and run_end_to_end
# (:1039-1045) both build a gate, run_architect_eval did not. A capped account
# therefore refused the architect outright, and the cell was recorded
# cap_tainted → EXCLUDED from the reported mean (the 2026-07-28 wave lost ~40%
# of cells, 37 blocked). Exclusion is not neutral: the costlier candidate runs
# longer and is more cap-exposed, so it loses more cells and the comparison is
# biased invisibly.
#
# These pin the gate CONSTRUCTION half — same enabled-guard, same
# warn-and-degrade fallback as the two sibling entry points.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestArchitectEvalUsageGate:
    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_gate_is_constructed_from_usage_cap_when_enabled(self):
        from shared.config_models import UsageCapConfig
        from shared.testing import make_gate_mock

        # A UsageGate-SHAPED gate, not a bare MagicMock: since the call-site
        # swap the gate has a real consumer, and invoke_with_cap_retry reads
        # `account_count` and drives `invoke_slot()` for real. A bare mock
        # returns a Mock for account_count and blows up on the `>` compare.
        # One healthy account is enough — this test pins CONSTRUCTION; failover
        # is TestArchitectEvalCapFailover's job.
        gate = make_gate_mock()
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(), usage_gate=gate,
        )

        # Built exactly ONCE, from the orch config's usage_cap BLOCK — not the
        # whole config, and not a fresh default. Same call shape as
        # run_eval:405 / run_end_to_end:1043.
        mocks['UsageGate'].assert_called_once()
        passed = mocks['UsageGate'].call_args.args[0]
        assert isinstance(passed, UsageCapConfig)
        assert passed.enabled is True

        # …and the cell is otherwise untouched: still one architect invoke,
        # still judged, still a non-sentinel score.
        assert mocks['invoke'].call_count == 1
        assert result.metrics['plan_quality'] == DEFAULT_JUDGE_PLAN_QUALITY
        assert result.metrics['cap_tainted'] is False
        assert result.outcome == 'done'

    async def test_gate_is_not_constructed_when_usage_cap_disabled(self):
        """usage_cap.enabled=False is a real production configuration.

        The gate must stay ABSENT there — the enabled guard is what keeps a
        deployment that opted out of the account pool from paying for probe
        dirs and account state it never asked for.
        """
        gate_factory = MagicMock()
        with patch('orchestrator.evals.eval_gate.UsageGate', gate_factory):
            result, mocks = await run_architect_eval_hermetic(
                self._cfg(), produced_plan=well_formed_plan(),
            )

        assert not gate_factory.called
        # Byte-equivalent to the pre-φ cell.
        assert mocks['invoke'].call_count == 1
        assert result.metrics['plan_quality'] == DEFAULT_JUDGE_PLAN_QUALITY
        assert result.metrics['cap_tainted'] is False
        assert result.outcome == 'done'

    async def test_gate_construction_failure_degrades_to_no_failover(self, caplog):
        """A gate that cannot be built must not take the eval cell down with it.

        Same warn-and-degrade contract as run_eval:406-407: log WHY, then run
        ungated. Losing failover costs cells; raising here would cost the whole
        campaign — and would do it for a REASON that has nothing to do with the
        candidate under test.
        """
        caplog.set_level(logging.WARNING, logger='orchestrator.evals.eval_gate')
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            usage_gate_error=RuntimeError('boom'),
        )

        assert 'without failover' in caplog.text
        assert 'boom' in caplog.text

        # Fully scored cell, not a crash and not an exclusion.
        assert isinstance(result.metrics['plan_quality'], float)
        assert result.metrics['plan_quality'] == DEFAULT_JUDGE_PLAN_QUALITY
        assert result.outcome == 'done'
        assert result.metrics['cap_tainted'] is False
        assert mocks['invoke'].call_count == 1


def _healthy_agent_result():
    """A clean architect answer — the result the SECOND account returns.

    A real AgentResult, not a MagicMock: it has to survive
    ``classify_invocation`` inside ``invoke_with_cap_retry`` (which must see a
    success, not another cap) AND ``classify_unmeasured`` in the runner
    (which must leave the cell unmarked). A MagicMock cannot express either,
    since every attribute access on it returns a truthy Mock.
    """
    from shared.cli_invoke import AgentResult

    return AgentResult(
        success=True,
        output='wrote the plan',
        cost_usd=1.23,
        duration_ms=4567,
        turns=11,
    )


def _two_account_gate():
    """A REAL two-account UsageGate with tokens pre-injected.

    Deliberately not a mock: the point of the failover test is that the genuine
    ``invoke_slot`` / ``detect_cap_hit`` / token-rotation path runs end to end,
    so a mock's scripted answers cannot make it pass. Canonical construction —
    test_usage_gate.py:30-43.
    """
    from _orch_helpers import build_usage_gate
    from shared.config_models import AccountConfig

    return build_usage_gate(
        [
            AccountConfig(name='max-a', oauth_token_env='CLAUDE_OAUTH_A'),
            AccountConfig(name='max-b', oauth_token_env='CLAUDE_OAUTH_B'),
        ],
        ['tok-a', 'tok-b'],
    )


# ---------------------------------------------------------------------------
# Cap FAILOVER through run_architect_eval (eval-revival φ, task 3630)
#
# The user-observable signal of the whole task: a cap on the first account must
# cost the campaign a retry, not a cell. Before φ the 429 was classified by
# the runner's classifier and the cell was persisted cap_tainted with
# plan_quality=None — a measurement the campaign then had to EXCLUDE.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestArchitectEvalCapFailover:
    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_cap_hit_fails_over_to_a_second_account_and_completes_the_cell(self):
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            usage_gate=_two_account_gate(),
            invoke_side_effect=[refused_agent_result(cost_usd=0.0), _healthy_agent_result()],
        )

        # (a) The cap cost a RETRY, not the cell.
        assert mocks['invoke'].await_count == 2

        # (b) …and the retry actually ROTATED the account, rather than
        # re-dispatching onto the same capped token.
        first, second = mocks['invoke'].call_args_list
        assert first.kwargs['oauth_token'] == 'tok-a'
        assert second.kwargs['oauth_token'] == 'tok-b'

        # (c) The cell is HEALTHY — measured, judged, and INCLUDED in the mean.
        # This is the whole point: pre-φ this cell was an exclusion.
        assert result.metrics['cap_tainted'] is False
        assert result.metrics['invocation_errors'] is None
        assert result.outcome == 'done'
        assert result.metrics['plan_quality'] == DEFAULT_JUDGE_PLAN_QUALITY
        assert result.metrics['plan_steps'] == 4

        # (d) The architect is still invoked with EXACTLY today's kwargs — the
        # new seam is a transport change, not a re-specification of the
        # candidate under test. A silent drop here (e.g. backend or
        # env_overrides not reaching invoke_agent) would change WHAT is being
        # measured while every other assertion above stayed green.
        kw = second.kwargs
        assert kw['model'] == 'sonnet'
        assert kw['backend'] == 'claude'
        assert kw['effort'] == 'high'
        assert kw['prompt'] == architect_prompt_sent()
        assert kw['cwd'] == Path('/fake/wt')
        assert kw['max_turns'] == 50
        assert kw['system_prompt']
        assert 'max_budget_usd' in kw
        assert 'allowed_tools' in kw
        assert 'disallowed_tools' in kw
        assert 'env_overrides' in kw
        assert kw['mcp_config'] is not None


def _one_account_gate():
    """A REAL single-account UsageGate — the pool that CANNOT fail over.

    The failover test above is the happy path: a cap costs a retry because a
    healthy account exists. This is the other end of the same story — every
    account capped, nowhere left to rotate to.
    """
    from _orch_helpers import build_usage_gate
    from shared.config_models import AccountConfig

    return build_usage_gate(
        [AccountConfig(name='max-only', oauth_token_env='CLAUDE_OAUTH_ONLY')],
        ['tok-only'],
    )


def _real_cap_retry_with_no_cap_patience():
    """The REAL ``invoke_with_cap_retry``, its public cap-wait bound set to zero.

    The first cap hit exhausts the pool, so the gate, the retry loop and the
    exhaustion path all run for real, in milliseconds rather than the
    inherited bound.
    """
    from shared.cli_invoke import invoke_with_cap_retry

    return functools.partial(invoke_with_cap_retry, cap_wait_sanity_secs=0.0)


# ---------------------------------------------------------------------------
# Shared patience and a LOUD, correctly-labelled exhaustion (eval-revival φ)
#
# Failover is only half the contract. The other half is what happens when it
# runs out: the cell inherits the shared default cap-wait patience, and when
# that patience is exhausted the failure must land in the EXISTING cap_tainted
# backstop stamped as a transport refusal — not as a harness crash of our own.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestArchitectEvalCapPatience:
    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_architect_call_sets_no_cap_bound_of_its_own(self):
        """One cap patience for every eval stage (esc-4002-1, task 7320).

        Waiting is ~free because ``--resume`` carries banked spend across the
        account switch; omitting the kwargs is what inherits the shared default.
        """
        wrapper = AsyncMock(return_value=_healthy_agent_result())
        await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            usage_gate=_one_account_gate(),
            cap_retry=wrapper,
        )

        assert wrapper.await_count == 1
        assert 'cap_wait_sanity_secs' not in wrapper.call_args.kwargs
        assert 'max_cap_retries' not in wrapper.call_args.kwargs

    async def test_fully_capped_pool_fails_loud_into_cap_tainted(self):
        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            usage_gate=_one_account_gate(),
            arch_result=refused_agent_result(cost_usd=0.0),
            cap_retry=_real_cap_retry_with_no_cap_patience(),
        )

        # It RETURNED. That is the requirement being pinned: when the real
        # wrapper gives up, the cell lands in the cap_tainted backstop rather
        # than propagating the exhaustion out of the campaign.
        assert result.outcome == 'blocked'

        # …into the pre-existing backstop, unchanged: no scorable plan landed,
        # so every available number would be fabricated and the cell is
        # excluded rather than scored 0.0.
        assert result.metrics['cap_tainted'] is True
        assert result.metrics['plan_quality'] is None

        # …and labelled as what it IS. Stamping a provider cap `harness_error:`
        # would charge OUR crash for a transport refusal and collapse two rows
        # of the docstring's taint table that exist precisely to stay distinct.
        record = only_record(result.metrics)
        assert (record['stage'], record['cause']) == ('architect', 'cap_exhausted')

    async def test_cap_exhaustion_that_left_a_scorable_plan_is_not_tainted(self):
        from orchestrator.evals.judge import score_plan_structure

        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            usage_gate=_one_account_gate(),
            arch_result=refused_agent_result(cost_usd=0.0),
            cap_retry=_real_cap_retry_with_no_cap_patience(),
        )

        # The architect wrote a real plan through plan-tools BEFORE the pool ran
        # dry — the common shape of a cap landing mid-campaign. Discarding it
        # would turn a scorable cell into an exclusion, re-opening the
        # differential-exclusion hazard from the other side. Cap EXHAUSTION
        # therefore obeys the SAME taint-table row an inline mid-run 429 obeys.
        assert result.metrics['cap_tainted'] is False
        assert result.metrics['plan_quality'] == score_plan_structure(
            well_formed_plan()
        )

        # Still MARKED, so a reader knows why the LLM judge was skipped…
        record = only_record(result.metrics)
        assert (record['stage'], record['cause']) == ('architect', 'cap_exhausted')

        # …and it really WAS skipped, carrying 3629's honest zeros rather than
        # spend for a call that never happened.
        assert mocks['judge'].await_count == 0
        assert result.metrics['judge_invocations'] == 0
        assert result.metrics['judge_cost_usd'] == 0.0


# ---------------------------------------------------------------------------
# The MUST-NOT-CHANGE taint semantics, re-pinned under the NEW seam
# (eval-revival φ). These assert nothing novel — they assert that routing the
# architect through invoke_with_cap_retry changed the TRANSPORT and nothing
# else. If one of them fails, the implementation is wrong; the pin is not.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestPinnedTaintSemanticsSurviveTheCapRetrySeam:
    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_timeout_through_the_gated_seam_is_still_kept_not_tainted(self):
        """A TimeoutError still reaches the runner through the wrapper.

        The pre-existing timeout tests all run gate-ABSENT, where
        ``invoke_with_cap_retry`` degrades to a single pass-through invoke.
        This runs the same scenario down the newly reachable GATED path, where
        the real ``invoke_slot`` context manager wraps the invoke, and pins
        that the wrapper still swallows nothing: a timeout arrives at the
        runner as a ``TimeoutError`` and keeps its deliberate asymmetry —
        MARKED (so the cell is legible) but NOT tainted (candidate-
        attributable, so it keeps scoring on content).

        Note this does NOT pin WHERE the timeout is applied — a mutation that
        bounds the whole retry loop instead of each attempt still passes here,
        because the exception originates in the mocked invoke rather than from
        a real deadline. That property is pinned by
        ``test_the_operator_timeout_bounds_each_attempt_not_the_retry_loop``.
        """
        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            usage_gate=_two_account_gate(),
            invoke_side_effect=TimeoutError(),
        )

        # Unchanged from the gate-absent pins above: a timeout is
        # CANDIDATE-attributable, so it is MARKED but keeps scoring on content.
        assert result.outcome == 'timeout'
        assert result.metrics['cap_tainted'] is False
        assert result.metrics['plan_quality'] is not None
        record = only_record(result.metrics)
        assert (record['stage'], record['cause']) == ('architect', 'timeout')

    async def test_gate_absent_path_is_byte_identical(self):
        """usage_cap disabled is a real deployment, and it must be untouched.

        With no gate the wrapper degrades to num_accounts=1 and a single
        invoke, so a 429 is classified by classify_unmeasured exactly as it
        was pre-φ — one invocation, marked, tainted. No retry, no failover, no
        second call the operator did not opt into.
        """
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            arch_result=refused_agent_result(cost_usd=0.0),
        )

        assert mocks['invoke'].await_count == 1
        assert result.metrics['cap_tainted'] is True
        assert result.metrics['plan_quality'] is None
        record = only_record(result.metrics)
        assert (record['stage'], record['cause']) == ('architect', 'cap_hit')

    async def test_healthy_architect_with_a_stepless_plan_still_scores_a_kept_zero(self):
        """A genuine 0.0 must stay a 0.0 — the anti-fabrication floor is intact.

        Gate present, architect healthy, plan empty. Nothing about the cap seam
        may promote a CONTENT failure into an infra exclusion: the model was
        asked, it answered, and it produced nothing worth scoring. That cell is
        kept and scored 0.0, and the LLM judge is not spent on an unjudgeable
        artifact.
        """
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            usage_gate=_two_account_gate(),
            arch_result=_healthy_agent_result(),
        )

        assert result.metrics['cap_tainted'] is False
        assert result.metrics['invocation_errors'] is None
        assert result.metrics['plan_quality'] == 0.0
        assert mocks['judge'].await_count == 0

    async def test_the_operator_timeout_bounds_each_attempt_not_the_retry_loop(self):
        """The real differential guard on WHERE the operator's --timeout lives.

        The tempting wrong implementation keeps the pre-φ shape —
        ``asyncio.wait_for(invoke_with_cap_retry(...), timeout=...)`` — instead
        of bounding each attempt. It is wrong because the retry loop's wall
        clock is dominated by cap WAITS, which the candidate under test is not
        responsible for: once a wait outlasts --timeout, the cell surfaces as
        ``outcome='timeout'`` → NOT tainted → a cap-starved cell scored a
        fabricated 0.0 and INCLUDED in the mean.

        Every other test in this file passes under that mutation, because they
        mock ``shared.cli_invoke.asyncio.sleep`` — so no wall clock ever
        advances during a cap wait. This one makes the injected gate's slot
        acquisition (``before_invoke``, inside the real wrapper but outside any
        attempt) genuinely outlast --timeout, so only a per-attempt bound
        leaves the cell healthy.
        """
        import asyncio

        from shared.testing import make_gate_mock

        # The REAL sleep, captured before the helper runs: the harness patches
        # ``shared.cli_invoke.asyncio.sleep``, and since that IS the asyncio
        # module the patch is global for its duration, so a plain
        # ``asyncio.sleep`` in here would return instantly.
        real_sleep = asyncio.sleep

        async def _slow_slot(**_kwargs) -> str:
            # ~6x the 30 ms budget below — far outside scheduler jitter.
            await real_sleep(0.2)
            return 'tok'

        gate = make_gate_mock(before_invoke=AsyncMock(side_effect=_slow_slot))
        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            injected_gate=gate,
            arch_result=_healthy_agent_result(),
            task_timeout_minutes=0.0005,  # 30 ms
        )

        # The slot wait outlived --timeout by 6x and the cell is STILL healthy:
        # the operator's budget bounds one attempt, never the retry loop.
        assert result.outcome == 'done'
        assert result.metrics['cap_tainted'] is False
        assert result.metrics['invocation_errors'] is None


# ---------------------------------------------------------------------------
# AMENDMENT PASS — reviewer follow-ups on eval-revival φ (task 3630)
#
# Four independent holes in the first cut of the gated seam: a gate that
# CONSTRUCTS but resolves no accounts, a gate that is never torn down, spend
# from an abandoned attempt vanishing, and the cap-RESUME branch (the common
# shape of a real cap) going entirely unexercised.
# ---------------------------------------------------------------------------

def _zero_account_gate():
    """A UsageGate that CONSTRUCTS successfully but resolves ZERO accounts.

    Not a hypothetical: ``_init_accounts`` warns that the pool is empty and
    returns ``[]`` whenever ``usage_cap.enabled`` is on but no
    ``CLAUDE_OAUTH_*`` env is exported and ``fallback_to_default_credential``
    is off — an ordinary CI/cron shell. The
    constructor SUCCEEDS, so the enabled-guard and the try/except around
    construction both pass it straight through.
    """
    from _orch_helpers import build_usage_gate

    return build_usage_gate([], [])


def _mid_run_cap_agent_result(*, cost_usd: float = 0.9, session_id: str = ''):
    """A cap that landed MID-run, after the architect spent real money.

    ``refused_agent_result(cost_usd=0.0)`` is the OTHER cap shape — an instant, $0.00, zero-
    turn refusal where the account was already capped when the attempt
    started. This one is the shape that makes both remaining amendments
    observable: real ``cost_usd`` to lose, and a real ``session_id`` to resume.
    """
    from shared.cli_invoke import AgentResult

    return AgentResult(
        success=False,
        output=REFUSAL_TEXT,
        cost_usd=cost_usd,
        duration_ms=180_000,
        turns=14,
        subtype='error',
        api_error_status=429,
        session_id=session_id,
    )


@pytest.mark.asyncio
class TestArchitectEvalZeroAccountGate:
    """An EMPTY account pool must degrade to ungated, not refuse every cell."""

    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_the_premise_a_zero_account_gate_refuses_every_invocation(self):
        """Pin WHY the degrade exists, so it cannot be deleted as dead code.

        The failure is deferred, not absent: construction succeeds and the
        first ``before_invoke`` raises. Routed through ``run_architect_eval``
        that RuntimeError lands in the generic handler, is stamped
        ``harness_error:`` → ``arch_unmeasurable`` → tainted → the cell is
        EXCLUDED. Every architect cell in the campaign, for a reason having
        nothing to do with the candidate — a strictly worse version of the
        ~40% cell loss φ exists to fix, since pre-φ the same run worked ungated
        on ambient credentials.
        """
        gate = _zero_account_gate()
        assert gate.account_count == 0
        with pytest.raises(RuntimeError, match='No OAuth accounts available'):
            await gate.before_invoke()

    async def test_zero_account_gate_degrades_to_the_ungated_path(self, caplog):
        caplog.set_level(logging.WARNING, logger='orchestrator.evals.eval_gate')
        gate = _zero_account_gate()

        result, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(), usage_gate=gate,
        )

        # Loud about it — an empty pool where the operator asked for one is a
        # misconfiguration worth seeing in the campaign log.
        assert 'no accounts resolved' in caplog.text
        assert 'UNGATED' in caplog.text

        # …and the cell is a NORMAL cell: one invoke, measured, judged,
        # INCLUDED. An empty pool has nothing to fail over TO, so the ungated
        # path forfeits nothing.
        assert mocks['invoke'].await_count == 1
        assert result.outcome == 'done'
        assert result.metrics['cap_tainted'] is False
        assert result.metrics['invocation_errors'] is None
        assert result.metrics['plan_quality'] == DEFAULT_JUDGE_PLAN_QUALITY

    async def test_the_empty_gate_is_never_dispatched_through(self):
        """The degrade happens at BUILD time, not by catching the refusal.

        Catching the RuntimeError at the call site would work too, but it would
        pay a real ``invoke_slot`` acquisition per cell and leave the gate
        installed for anything else that later reads it. The gate must simply
        never become this cell's transport.
        """
        gate = _zero_account_gate()
        gate.before_invoke = AsyncMock(side_effect=AssertionError('gate was used'))

        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(), usage_gate=gate,
        )
        assert result.outcome == 'done'
        gate.before_invoke.assert_not_awaited()

    async def test_the_discarded_empty_gate_is_torn_down(self):
        """…and it is not merely dropped on the floor.

        Even with zero accounts the constructor allocates a FALLBACK probe
        ``TaskConfigDir`` (the ``or TaskConfigDir(...)`` alias branch) and
        installs a SIGHUP handler. Discarding without teardown would leak one
        temp dir PER CELL for the whole campaign.
        """
        gate = _zero_account_gate()
        gate.shutdown = AsyncMock()

        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(), usage_gate=gate,
        )
        assert result.outcome == 'done'
        gate.shutdown.assert_awaited_once()


@pytest.mark.asyncio
class TestArchitectEvalGateLifecycle:
    """The gate is built PER CELL, so it must be torn down per cell.

    ``cli.py`` loops ``run_architect_eval`` over every config, and a campaign
    loops fixtures × trials, all in ONE process. A gate that is never shut down
    leaves its account-resume probe loop firing real CLI probes for the rest of
    the campaign, and every new gate steals the SIGHUP handler from
    predecessors kept alive by exactly those background tasks.
    """

    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_gate_is_shut_down_when_the_cell_completes(self):
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(), usage_gate=gate,
        )

        assert result.outcome == 'done'
        gate.shutdown.assert_awaited_once()

    async def test_gate_is_shut_down_even_when_the_cell_fails(self):
        """Teardown lives in the ``finally``, so the failure paths get it too.

        A campaign's LOSING cells are the ones most likely to have hit a cap —
        i.e. exactly the ones holding a live probe loop. Leaking on the failure
        path would leak precisely where it costs the most.
        """
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            usage_gate=gate,
            invoke_side_effect=RuntimeError('architect exploded'),
        )

        assert result.outcome == 'blocked'
        gate.shutdown.assert_awaited_once()

    async def test_a_failing_shutdown_never_damages_the_cell(self, caplog):
        """Teardown is best-effort: the cell is already scored by then."""
        from shared.testing import make_gate_mock

        caplog.set_level(logging.WARNING, logger='orchestrator.evals.eval_gate')
        gate = make_gate_mock()
        gate.shutdown = AsyncMock(side_effect=RuntimeError('teardown boom'))

        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(), usage_gate=gate,
        )

        assert result.outcome == 'done'
        assert result.metrics['plan_quality'] == DEFAULT_JUDGE_PLAN_QUALITY
        assert result.metrics['cap_tainted'] is False
        assert 'shutdown failed' in caplog.text

    async def test_no_gate_means_nothing_to_shut_down(self):
        """The ungated path must not NameError or invent a teardown."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
        )
        assert result.outcome == 'done'
        assert mocks['gate'] is None


@pytest.mark.asyncio
class TestArchitectEvalInjectedGate:
    """A gate the CALLER built is used as-is and never torn down (task 4427).

    The per-cell build above stays the default, but a campaign owner that
    already holds a gate must be able to hand it down — and the cell must NOT
    shut it down, because that would destroy the shared cap state mid-campaign,
    which is the entire point of hoisting it.
    """

    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_injected_gate_is_the_one_invoked_with(self):
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        _result, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(), injected_gate=gate,
        )

        # The gate the campaign owns is the gate the invoke actually rides:
        # its slot was taken, and the invoke carried that slot's token.
        gate.before_invoke.assert_awaited()
        assert mocks['invoke'].await_args.kwargs['oauth_token'] == 'tok'

    async def test_an_injected_gate_suppresses_the_per_cell_build(self):
        """The hoist is pointless if the cell builds one anyway."""
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        build = AsyncMock(return_value=make_gate_mock())
        _result, _mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
            injected_gate=gate, gate_factory=build,
        )

        build.assert_not_awaited()

    async def test_an_injected_gate_is_never_shut_down(self):
        """Tearing down a borrowed gate would destroy the campaign's cap state.

        Cell N shutting the shared gate down would take failover away from every
        cell after it — strictly worse than the per-cell gate it replaced.
        """
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(), injected_gate=gate,
        )

        assert result.outcome == 'done'
        gate.shutdown.assert_not_awaited()

    async def test_an_injected_gate_survives_a_failing_cell(self):
        """The failure path is exactly where the shared gate matters most.

        A cell that blew up is the one most likely to have hit a cap, i.e. the
        one carrying the cap knowledge its siblings need.
        """
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan={},
            injected_gate=gate,
            invoke_side_effect=RuntimeError('architect exploded'),
        )

        assert result.outcome == 'blocked'
        gate.shutdown.assert_not_awaited()

    async def test_explicit_none_means_ungated_not_unset(self):
        """``usage_gate=None`` is a DECISION, not a missing argument.

        ``build_eval_usage_gate`` returns None three legitimate ways, so a
        campaign can be deliberately ungated. If the cell read that None as
        "nothing was supplied" it would build its own gate — silently restoring
        the per-cell construction the hoist removes, and re-paying a probe-dir
        alloc plus a SIGHUP-handler steal on the zero-account path.
        """
        build = AsyncMock()
        _result, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
            injected_gate=None, gate_factory=build,
        )

        build.assert_not_awaited()
        # One plain invoke: no slot was taken, so no account token rode it.
        mocks['invoke'].assert_awaited_once()
        assert 'oauth_token' not in mocks['invoke'].await_args.kwargs

    async def test_omitting_the_parameter_keeps_the_owned_path(self):
        """Back-compat: no argument still means build-your-own AND tear it down.

        The four TestArchitectEvalGateLifecycle pins cover the teardown; this
        adds the other half — that the cell still BUILDS when nothing is passed,
        which is what makes the sentinel (rather than a None default) load-bearing.
        """
        from shared.testing import make_gate_mock

        own_gate = make_gate_mock()
        build = AsyncMock(return_value=own_gate)
        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(), gate_factory=build,
        )

        build.assert_awaited_once()
        own_gate.before_invoke.assert_awaited()
        assert result.outcome == 'done'
        # Owned → the cell is still responsible for the teardown.
        own_gate.shutdown.assert_awaited_once()


# ---------------------------------------------------------------------------
# Campaign-level gate ownership (task 4427)
#
# The per-cell gate above is correct but STATELESS across cells: cell N+1
# re-leases the account cell N proved capped, paying a wasted invocation plus a
# cooldown every time. ``campaign_usage_gate`` hoists construction+teardown to
# whoever loops the cells (the μ stage functions and ``evals.cli._run_single_eval``),
# so ONE cap-state view serves the whole campaign.
#
# The hoist rests on one load-bearing premise: the ``usage_cap`` a cell would
# have built its own gate from is the SAME ``usage_cap`` the campaign owner
# reads off ``base_config``. That is an emergent property of two files
# (``build_eval_orch_config`` + ``EVAL_PROFILE``), so it is pinned here as a
# tripwire rather than trusted as a comment.
# ---------------------------------------------------------------------------

class TestCampaignGatePremise:
    """A campaign gate built from ``base_config`` is what every cell expects.

    ``build_eval_usage_gate`` reads exactly one leaf — ``orch_config.usage_cap``
    — and each cell's orch config is
    ``apply_eval_profile(base_config).model_copy(update=...)``. Neither
    ``EVAL_PROFILE`` nor that update dict mentions ``usage_cap``, so the leaf is
    inherited verbatim and the campaign gate is byte-identical to the gate each
    cell builds today. If someone later adds ``usage_cap`` to ``EVAL_PROFILE``,
    the campaign gate would silently stop matching what cells expect — this
    fails loudly instead.
    """


    @pytest.mark.parametrize('role', [RoleUnderTest.ARCHITECT, RoleUnderTest.IMPLEMENTER])
    def test_cell_orch_config_inherits_base_usage_cap_verbatim(
        self, tmp_path: Path, role: RoleUnderTest,
    ):
        from orchestrator.evals.configs import EvalConfig
        from orchestrator.evals.runner import build_eval_orch_config

        base = eval_base_config(tmp_path)
        cfg = EvalConfig(f'{role}-sonnet-high', 'claude', 'sonnet', 'high', role=role)
        task = {'id': 'df_task_4427', 'project_root': str(tmp_path)}

        cell_config = build_eval_orch_config(cfg, task, base)

        assert cell_config.usage_cap == base.usage_cap


@pytest.mark.asyncio
class TestArchitectEvalSpendSurvivesFailover:
    """Spend from an ABANDONED attempt must still be charged to the cell.

    ``invoke_with_cap_retry`` returns only the LAST AgentResult and never
    aggregates cost across attempts. Reading ``result.cost_usd`` therefore
    drops everything the architect spent before a mid-run cap — and φ makes
    that newly consequential, because the cell is now INCLUDED in the
    aggregate. An under-reported cost biases the cost comparison in the same
    direction, and for the same reason, that the quality comparison was biased:
    the more cap-exposed candidate is the one that fails over more often.
    """

    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_spend_from_the_capped_attempt_is_not_dropped(self):
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            usage_gate=_two_account_gate(),
            invoke_side_effect=[
                _mid_run_cap_agent_result(cost_usd=0.9),
                _healthy_agent_result(),
            ],
        )

        assert mocks['invoke'].await_count == 2
        # $0.90 burned before the cap + $1.23 on the account that finished.
        # Reading only the last result would report $1.23 and silently make
        # this candidate look 42% cheaper than it was.
        assert result.metrics['cost_usd'] == pytest.approx(0.9 + 1.23)
        # The judge's share is a SUBSET of that total, not an addend, and the
        # stub verdict spends nothing.
        assert result.metrics['judge_cost_usd'] == 0.0
        assert result.metrics['cap_tainted'] is False

    async def test_duration_stays_the_winning_attempt_not_the_sum(self):
        """Cost accumulates; LATENCY deliberately does not.

        Cap waits and abandoned attempts are INFRA time, not candidate
        latency. Summing them would corrupt the per-candidate duration the
        campaign compares — the same class of error, one column over.
        """
        result, _mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            usage_gate=_two_account_gate(),
            invoke_side_effect=[
                _mid_run_cap_agent_result(cost_usd=0.9),  # duration_ms=180_000
                _healthy_agent_result(),                  # duration_ms=4_567
            ],
        )
        assert result.metrics['workflow_duration_ms'] == 4567

    async def test_single_attempt_spend_is_byte_identical_to_pre_amendment(self):
        """One attempt ⇒ the accumulator equals the old direct read."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            usage_gate=_two_account_gate(),
            arch_result=_healthy_agent_result(),
        )
        assert mocks['invoke'].await_count == 1
        assert result.metrics['cost_usd'] == pytest.approx(1.23)

    async def test_spend_before_a_fully_capped_pool_is_still_reported(self):
        """Even the give-up path reports what it burned.

        The cap-exhaustion handler has no AgentResult to read — the exception
        was raised AT the invoke — so pre-amendment the cell reported $0.00
        however much money the architect had already spent. The accumulator
        makes an EXCLUDED cell's spend honest too, which matters because
        exclusions are still charged to the campaign's budget.
        """
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            usage_gate=_one_account_gate(),
            arch_result=_mid_run_cap_agent_result(cost_usd=0.9),
            cap_retry=_real_cap_retry_with_no_cap_patience(),
        )

        assert mocks['invoke'].await_count == 1
        assert result.outcome == 'blocked'
        assert result.metrics['cap_tainted'] is True
        assert result.metrics['cost_usd'] == pytest.approx(0.9)


@pytest.mark.asyncio
class TestArchitectEvalCapResume:
    """The cap-RESUME branch — the shape a real mid-run cap actually takes.

    ``TestArchitectEvalCapFailover`` exercises the FRESH-retry branch, and only
    because ``refused_agent_result(cost_usd=0.0)`` happens to carry no ``session_id``. A cap
    that lands after the architect has been working DOES carry one, and the
    wrapper then resumes that session on the next account instead of restarting
    it: ``resume_session_id`` is set and the prompt is replaced with
    ``CAP_HIT_RESUME_PROMPT``.

    That branch is where a mis-resume would be most damaging: an architect
    resumed into an EMPTY session receives a two-line "continue" prompt with no
    task context, produces nothing, and is scored as a genuine content 0.0 —
    the φ bias re-created from the inside, with no marker to reveal it.

    Why unchecked resume is nonetheless SOUND for this call site: the wrapper's
    transcript-reachability veto only runs when a ``config_dir`` is supplied,
    and the eval passes none. Claude CLI transcripts are local JSONL at
    ``<config_dir>/projects/<cwd-slug>/<session_id>.jsonl`` (measured
    2026-08-01, cli_invoke.py), and with ``config_dir=None`` every attempt
    inherits the SAME ambient ``CLAUDE_CONFIG_DIR`` in the SAME process with
    the SAME cwd (this cell's eval worktree) — so the transcript attempt 1
    wrote is exactly where attempt 2's ``--resume`` looks. The veto has nothing
    to add here; these tests pin the behaviour so a future change to either
    half (threading a config_dir in, or moving the worktree between attempts)
    fails loudly instead of silently degrading the measurement.
    """

    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_a_capped_session_is_resumed_on_the_next_account(self, caplog):
        from shared.cli_invoke import CAP_HIT_RESUME_PROMPT

        caplog.set_level(logging.WARNING, logger='shared.cli_invoke')
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            usage_gate=_two_account_gate(),
            invoke_side_effect=[
                _mid_run_cap_agent_result(session_id='sess-abc'),
                _healthy_agent_result(),
            ],
        )

        first, second = mocks['invoke'].call_args_list

        # (a) Attempt 1 is a normal fresh dispatch carrying the real prompt.
        assert first.kwargs['prompt'] == architect_prompt_sent()
        assert first.kwargs.get('resume_session_id') is None

        # (b) Attempt 2 RESUMES that session on the OTHER account, rather than
        #     re-running the architect from scratch (which would double the
        #     cell's cost and discard everything already reasoned).
        assert second.kwargs['resume_session_id'] == 'sess-abc'
        assert second.kwargs['prompt'] == CAP_HIT_RESUME_PROMPT
        assert second.kwargs['oauth_token'] == 'tok-b'

        # (c) The candidate SPEC is unchanged across the resume — the seam is
        #     transport, never a re-specification of what is being measured.
        assert second.kwargs['model'] == 'sonnet'
        assert second.kwargs['backend'] == 'claude'
        assert second.kwargs['effort'] == 'high'
        assert second.kwargs['cwd'] == Path('/fake/wt')
        assert second.kwargs['mcp_config'] is not None

        # (d) It took the unchecked-resume branch, and says so in the log —
        #     "unchecked" is the honest word, since with no config_dir the
        #     wrapper never looked for the transcript.
        assert 'transcript unchecked' in caplog.text

        # (e) The cell lands as a normal MEASURED cell, with both attempts'
        #     spend charged to it.
        assert result.outcome == 'done'
        assert result.metrics['cap_tainted'] is False
        assert result.metrics['invocation_errors'] is None
        assert result.metrics['plan_quality'] == DEFAULT_JUDGE_PLAN_QUALITY
        assert result.metrics['cost_usd'] == pytest.approx(0.9 + 1.23)

    async def test_a_sessionless_cap_still_retries_fresh_with_the_real_prompt(self):
        """The contrast case, pinned side by side so the fork stays legible.

        No ``session_id`` ⇒ nothing to resume ⇒ the wrapper restores the
        original architect prompt. Reversing either half of this fork (fresh
        where a session exists, or ``CAP_HIT_RESUME_PROMPT`` where none does)
        silently changes WHAT the second attempt is asked to do.
        """
        _result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            usage_gate=_two_account_gate(),
            invoke_side_effect=[refused_agent_result(cost_usd=0.0), _healthy_agent_result()],
        )

        _first, second = mocks['invoke'].call_args_list
        assert second.kwargs.get('resume_session_id') is None
        assert second.kwargs['prompt'] == architect_prompt_sent()
