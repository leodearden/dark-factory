"""Tests that pin what one architect cell records about its spend.

They cover the plan judge's spend as a subset of the cell cost, the token usage
and proxy signal, and the cost provenance the cell resolves.
"""

from __future__ import annotations

import logging

import pytest
from _architect_eval_harness import (
    DEFAULT_ARCH_INVOKE,
    PROXY_ENV,
    run_architect_eval_hermetic,
    token_agent_result,
)
from _architect_plan_shapes import create_plan_stub, stub_with_steps, well_formed_plan
from _eval_refusal import refused_agent_result

from orchestrator.evals.judge import PlanQualityVerdict
from orchestrator.evals.measurement import Scored
from orchestrator.evals.metrics import RoleUnderTest

# ---------------------------------------------------------------------------
# Architect cell records the plan judge's spend (step-5/6, eval-revival υ)
#
# run_architect_eval built its EvalMetrics from the architect's OWN spend
# only, discarding the plan judge's opus/effort=max invocation cost entirely
# — every architect cell persisted judge_cost_usd=0.0. judge_cost_usd is a
# SUBSET of cost_usd, not disjoint (metrics.py:69-71), so the cell's cost_usd
# must become architect spend + judge spend, with judge_cost_usd as the
# subset breakdown — mirroring the implementer path's existing semantics
# (metrics.py:597-598 / workflow.py:7872-7875).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestArchitectCellRecordsJudgeSpend:
    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    async def test_judge_spend_recorded_and_folded_into_cell_cost(self):
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            judge_return=Scored(
                PlanQualityVerdict(plan_quality=0.77, per_criterion={}, reasoning='good'),
                cost_usd=0.42,
            ),
        )

        # The RETURNED result...
        assert result.metrics['judge_cost_usd'] == pytest.approx(0.42)
        assert result.metrics['judge_invocations'] == 1
        # ...the SUBSET invariant: architect spend (the harness default)
        # PLUS the judge spend, never the architect spend alone.
        assert result.metrics['cost_usd'] == pytest.approx(DEFAULT_ARCH_INVOKE.cost_usd + 0.42)

        # ...and what was actually PERSISTED through the store, not just what
        # the function happens to return.
        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['judge_cost_usd'] == pytest.approx(0.42)
        assert persisted['judge_invocations'] == 1
        assert persisted['cost_usd'] == pytest.approx(DEFAULT_ARCH_INVOKE.cost_usd + 0.42)


@pytest.mark.asyncio
class TestArchitectCellStampsTokenUsageAndProxySignal:
    """Task 3656 step-1: the architect cell must record the run's TOKEN USAGE
    and its PROXY signal, not leave all three at their dataclass defaults.

    These are the three inputs :func:`~orchestrator.evals.metrics.resolve_cost_usd`
    needs to resolve cost provenance per Invariant P5. Without them the
    price-table path is unreachable on the architect side no matter what the
    resolver is asked, so they are pinned BEFORE the resolution itself
    (step-5). Every assertion checks the PERSISTED cell as well as the returned
    result — the persisted JSON is what the report surfaces and an operator read.
    """

    def _cfg(self, **kw):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high',
            role=RoleUnderTest.ARCHITECT, **kw,
        )

    async def test_reported_token_counts_are_stamped(self):
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            arch_result=token_agent_result(12_000, 3_000),
        )

        assert result.metrics['input_tokens'] == 12_000
        assert result.metrics['output_tokens'] == 3_000

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['input_tokens'] == 12_000
        assert persisted['output_tokens'] == 3_000

    async def test_unreported_token_counts_persist_as_zero_not_none(self):
        """``AgentResult`` declares both ``int | None``; a provider that did not
        report must persist an honest ``0``, never a ``None`` that would poison
        the arithmetic in ``resolve_cost_usd`` (and ``json`` a bare null into a
        field the report layer treats as an int)."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            arch_result=token_agent_result(None, None),
        )

        assert result.metrics['input_tokens'] == 0
        assert result.metrics['output_tokens'] == 0
        assert result.metrics['input_tokens'] is not None
        assert result.metrics['output_tokens'] is not None

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['input_tokens'] == 0
        assert persisted['output_tokens'] == 0
        # Same ``int | None`` contract for the cache counts (this AgentResult
        # leaves both at their None default).
        assert persisted['cache_read_tokens'] == 0
        assert persisted['cache_create_tokens'] == 0

    async def test_cache_tokens_and_turns_are_stamped(self):
        """The REST of the token profile ``collect_metrics`` stamps.

        These price nothing, but a native Claude run is cache-read dominated:
        a cell reporting 12k input / 3k output beside a zeroed cache block
        reads as a far smaller run than it was, so the persisted evidence must
        be symmetric with the implementer path's (reviewer: completeness).
        """
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(), produced_plan=well_formed_plan(),
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cache_read_tokens'] == DEFAULT_ARCH_INVOKE.cache_read_tokens
        assert persisted['cache_create_tokens'] == DEFAULT_ARCH_INVOKE.cache_create_tokens
        assert persisted['turns_used'] == DEFAULT_ARCH_INVOKE.turns
        # output_tokens ÷ that leg's own duration — the identical formula
        # collect_metrics uses, over the same duration workflow_duration_ms
        # carries.
        assert persisted['workflow_duration_ms'] == DEFAULT_ARCH_INVOKE.duration_ms
        assert persisted['tokens_per_second'] == pytest.approx(
            round(DEFAULT_ARCH_INVOKE.output_tokens / (DEFAULT_ARCH_INVOKE.duration_ms / 1000), 2),
        )
        assert result.metrics['cache_read_tokens'] == DEFAULT_ARCH_INVOKE.cache_read_tokens

    async def test_zero_duration_cell_reports_zero_tokens_per_second(self):
        """A timed-out cell has no duration to divide by — 0.0, not a
        ZeroDivisionError that would lose the whole cell."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            invoke_side_effect=TimeoutError(),
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['tokens_per_second'] == 0.0
        assert persisted['turns_used'] == 0
        assert result.metrics['tokens_per_second'] == 0.0

    async def test_proxied_candidate_is_flagged_local(self):
        """``ANTHROPIC_BASE_URL`` set == a PROXIED endpoint, the signal Invariant
        P5 keys on to stop trusting the CLI's own cost figure."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(env_overrides={'ANTHROPIC_BASE_URL': 'http://localhost:8000/v1'}),
            produced_plan=well_formed_plan(),
        )

        assert result.metrics['is_local_model'] is True
        assert mocks['save'].call_args.args[0].metrics['is_local_model'] is True

    async def test_native_candidate_is_not_flagged_local(self):
        """Today's shape (``architect-opus-max`` / ``architect-fable-max``):
        native cloud, no proxy, so the CLI cost stays trustworthy."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=well_formed_plan(),
        )

        assert result.metrics['is_local_model'] is False
        assert mocks['save'].call_args.args[0].metrics['is_local_model'] is False


@pytest.mark.asyncio
class TestArchitectCellResolvesCostProvenance:
    """Task 3656 step-5: the architect cell's cost must be RESOLVED through
    :func:`~orchestrator.evals.metrics.resolve_cost_usd`, not copied from the
    raw CLI figure with an UNVERIFIED ``cost_source='cli'`` dataclass default.

    Invariant P5: a PROXIED endpoint's CLI cost figure is untrustworthy, so a
    proxied architect candidate must land on the price table (or the defined,
    LOUD fallback) rather than keeping the proxy's number. ``collect_metrics``
    (the implementer path) has resolved through this seam since task 2820;
    ``run_architect_eval`` bypassed it entirely.

    The plan judge is held to ZERO spend throughout (the harness's default
    ``PlanQualityVerdict`` carries ``cost_usd=0.0``) so these cases isolate the
    ARCHITECT component — composing the two-component label is step-7's.
    """

    def _cfg(self, **kw):
        from orchestrator.evals.configs import EvalConfig

        kw.setdefault('model', 'sonnet')
        return EvalConfig(
            'architect-candidate', 'claude', effort='high', role=RoleUnderTest.ARCHITECT, **kw,
        )

    async def test_native_candidate_keeps_the_cli_figure_verbatim(self):
        """REGRESSION PIN for every architect cell that has already landed.

        ``architect-opus-max`` / ``architect-fable-max`` are native cloud
        (``env_overrides={}``) and unlisted in the price table, so
        ``resolve_cost_usd`` takes its ``not is_local_model`` branch and returns
        ``(cli_cost_usd, 'cli')`` VERBATIM. Both the figure and the label must
        be byte-identical to today's — the change bites only a PROXIED
        candidate, and turns today's UNVERIFIED 'cli' into a DERIVED one.
        """
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=well_formed_plan(),
            orch_prices={},
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_source'] == 'cli'
        assert persisted['cost_usd'] == pytest.approx(DEFAULT_ARCH_INVOKE.cost_usd)
        assert result.metrics['cost_source'] == 'cli'
        assert result.metrics['cost_usd'] == pytest.approx(DEFAULT_ARCH_INVOKE.cost_usd)

    async def test_priced_proxied_candidate_uses_the_price_table(self):
        """A proxied candidate whose model IS listed must be priced from the
        table, NOT from the proxy's own (wrong) CLI figure."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(model='qwen3-coder', env_overrides=dict(PROXY_ENV)),
            produced_plan=well_formed_plan(),
            arch_result=token_agent_result(1_000_000, 500_000, cost_usd=99.0),
            orch_prices={'qwen3-coder': {'input_per_1m': 0.30, 'output_per_1m': 1.20}},
        )

        expected = (1_000_000 * 0.30 + 500_000 * 1.20) / 1_000_000
        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_source'] == 'price_table'
        assert persisted['cost_usd'] == pytest.approx(expected)
        assert persisted['cost_usd'] == pytest.approx(0.90)
        # Emphatically NOT the raw proxied CLI figure.
        assert persisted['cost_usd'] != pytest.approx(99.0)
        assert result.metrics['cost_source'] == 'price_table'

    async def test_priced_proxied_candidate_prices_its_cache_tokens(self):
        """The architect's cache reads and writes are priced at their own
        rates, not dropped from the figure."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(model='qwen3-coder', env_overrides=dict(PROXY_ENV)),
            produced_plan=well_formed_plan(),
            arch_result=token_agent_result(
                1_000_000, 500_000,
                cache_read_tokens=2_000_000, cache_create_tokens=100_000, cost_usd=99.0,
            ),
            orch_prices={'qwen3-coder': {
                'input_per_1m': 0.30, 'output_per_1m': 1.20,
                'cached_input_per_1m': 0.03, 'cache_write_per_1m': 0.30,
            }},
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_usd'] == pytest.approx(0.30 + 0.60 + 0.06 + 0.03)
        assert persisted['cost_usd'] == pytest.approx(0.99)
        assert persisted['cost_source'] == 'price_table'
        assert result.metrics['cost_source'] == 'price_table'

    async def test_unpriced_proxied_candidate_warns_and_uses_defined_fallback(
        self, caplog,
    ):
        """A proxied candidate with NO price entry degrades to the DEFINED
        ``FALLBACK_PRICE`` and says so loudly — never a silent raw-CLI
        number (the loud-over-silent-degradation norm)."""
        from orchestrator.agents.token_pricing import FALLBACK_PRICE

        with caplog.at_level(logging.WARNING, logger='orchestrator.evals.metrics'):
            result, mocks = await run_architect_eval_hermetic(
                self._cfg(model='qwen3-coder', env_overrides=dict(PROXY_ENV)),
                produced_plan=well_formed_plan(),
                arch_result=token_agent_result(1_000_000, 500_000, cost_usd=99.0),
                orch_prices={},
            )

        # Import the constant rather than hardcoding 2.0/8.0, so a future rate
        # change updates this expectation instead of leaving a stale magic
        # number (the single-home discipline of task 2459).
        expected = (
            1_000_000 * FALLBACK_PRICE['input_per_1m']
            + 500_000 * FALLBACK_PRICE['output_per_1m']
        ) / 1_000_000
        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_source'] == 'unpriced_proxy'
        assert persisted['cost_usd'] == pytest.approx(expected)
        assert persisted['cost_usd'] != pytest.approx(99.0)
        assert result.metrics['cost_source'] == 'unpriced_proxy'

        warnings = [
            r.message for r in caplog.records
            if r.levelno >= logging.WARNING
            and 'No configured price for proxied-endpoint model' in r.message
        ]
        assert warnings, (
            'Expected the LOUD unpriced-proxy WARNING from '
            'orchestrator.evals.metrics; got '
            f'{[r.message for r in caplog.records]}'
        )

    async def test_orch_config_never_built_still_lands_a_cell(self):
        """The price table lives on the eval orch config, which is built INSIDE
        the try — so a harness crash before it is bound must leave an explicit
        ``None`` the resolver tolerates, never an ``UnboundLocalError`` /
        ``AttributeError`` that loses the whole cell.

        That path never invoked the architect (0 tokens, $0.00), so no price
        table could have changed the number anyway.
        """
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=well_formed_plan(),
            orch_config_side_effect=RuntimeError('config build exploded'),
        )

        persisted = mocks['save'].call_args.args[0].metrics
        # The cell LANDED — marked and excluded, not lost.
        assert persisted['cap_tainted'] is True
        assert [r['cause'] for r in persisted['invocation_errors']] == ['harness_error']
        assert persisted['cost_usd'] == pytest.approx(0.0)
        # A concrete label, not merely "some member of the vocabulary": nothing
        # was spent, so the resolution is skipped and the documented 'cli'
        # default stands (reviewer: an `in {the whole vocabulary}` check cannot
        # fail, so it pinned nothing).
        assert persisted['cost_source'] == 'cli'
        assert result.metrics['cap_tainted'] is True

    async def test_zero_spend_proxied_cell_does_not_warn_about_pricing(
        self, caplog,
    ):
        """A cell that spent NOTHING has no provenance to resolve.

        The timeout / harness-error / pre-invoke-cap paths never bind
        ``result``, so they reach the resolution with 0 tokens and $0.00.
        Resolving anyway would fire the LOUD unpriced-proxy WARNING for spend
        that never happened on EVERY timed-out proxied cell — training
        operators to ignore the warning that matters — and would label a $0.00
        figure 'unpriced_proxy' as though the degradation were real. The
        loud-over-silent norm is about REAL degradation (reviewer: log-noise).
        """
        with caplog.at_level(logging.WARNING, logger='orchestrator.evals.metrics'):
            result, mocks = await run_architect_eval_hermetic(
                self._cfg(model='qwen3-coder', env_overrides=dict(PROXY_ENV)),
                produced_plan=well_formed_plan(),
                invoke_side_effect=TimeoutError(),
                orch_prices={},
            )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_usd'] == pytest.approx(0.0)
        assert persisted['input_tokens'] == 0
        assert persisted['output_tokens'] == 0
        assert persisted['cost_source'] == 'cli'
        assert result.metrics['cost_source'] == 'cli'
        assert not [
            r for r in caplog.records
            if 'No configured price for proxied-endpoint model' in r.message
        ], (
            'A $0.00 / 0-token cell must not emit the unpriced-proxy '
            f'degradation WARNING; got {[r.message for r in caplog.records]}'
        )


@pytest.mark.asyncio
class TestArchitectCellCostSourceIsComposed:
    """Task 3656 step-7: the cell's ``cost_usd`` is a TWO-COMPONENT sum
    (architect spend + plan-judge spend), so its single ``cost_source`` must
    not silently describe two differently-sourced components.

    Only the ARCHITECT component is resolved. The plan judge is always a
    native-cloud opus call — ``judge_plan_quality`` takes neither the
    candidate's model nor its ``env_overrides`` — so its CLI figure is
    trustworthy by construction, and re-resolving it against the candidate's
    price table would price opus tokens at a vLLM rate. The label therefore
    COMPOSES (``metrics.compose_cost_source``): agreement → that source,
    disagreement with judge dollars actually in the sum → ``'mixed'``.
    """

    def _cfg(self, **kw):
        from orchestrator.evals.configs import EvalConfig

        kw.setdefault('model', 'sonnet')
        return EvalConfig(
            'architect-candidate', 'claude', effort='high', role=RoleUnderTest.ARCHITECT, **kw,
        )

    def _spending_judge(self):
        return Scored(
            PlanQualityVerdict(plan_quality=0.77, per_criterion={}, reasoning='good'),
            cost_usd=0.42,
        )

    async def test_price_table_architect_plus_cli_judge_reads_mixed(self):
        """A price-table architect component beside a CLI judge component: the
        figure is the RESOLVED architect cost plus the judge's CLI cost, and the
        label says the two disagree rather than claiming either one."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(model='qwen3-coder', env_overrides=dict(PROXY_ENV)),
            produced_plan=well_formed_plan(),
            judge_return=self._spending_judge(),
            arch_result=token_agent_result(1_000_000, 500_000, cost_usd=99.0),
            orch_prices={'qwen3-coder': {'input_per_1m': 0.30, 'output_per_1m': 1.20}},
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_usd'] == pytest.approx(0.90 + 0.42)
        assert persisted['judge_cost_usd'] == pytest.approx(0.42)
        assert persisted['cost_source'] == 'mixed'
        assert result.metrics['cost_source'] == 'mixed'

    async def test_todays_native_shape_with_a_spending_judge_stays_cli(self):
        """REGRESSION PIN: both components are CLI-sourced, so a spending judge
        must NOT tip today's architect cells into 'mixed'. Keeps every landed
        result and ``TestArchitectCellRecordsJudgeSpend`` byte-identical."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(env_overrides={}),
            produced_plan=well_formed_plan(),
            judge_return=self._spending_judge(),
            orch_prices={},
        )

        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['cost_source'] == 'cli'
        assert persisted['cost_usd'] == pytest.approx(DEFAULT_ARCH_INVOKE.cost_usd + 0.42)
        assert result.metrics['cost_source'] == 'cli'

    async def test_judge_skipped_labels_the_architect_component_alone(self):
        """An unscorable plan SKIPS the judge (``create_plan_stub`` is the
        header-only artifact ``create_plan`` writes), so no judge dollars enter
        the sum and the label describes the architect component alone — NOT
        'mixed', which would announce a blend that never happened."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(model='qwen3-coder', env_overrides=dict(PROXY_ENV)),
            produced_plan=create_plan_stub(),
            judge_return=self._spending_judge(),
            arch_result=token_agent_result(1_000_000, 500_000, cost_usd=99.0),
            orch_prices={'qwen3-coder': {'input_per_1m': 0.30, 'output_per_1m': 1.20}},
        )

        mocks['judge'].assert_not_called()
        persisted = mocks['save'].call_args.args[0].metrics
        assert persisted['judge_cost_usd'] == pytest.approx(0.0)
        assert persisted['cost_source'] == 'price_table'
        assert persisted['cost_usd'] == pytest.approx(0.90)
        assert result.metrics['cost_source'] == 'price_table'


@pytest.mark.asyncio
class TestJudgeSkippedBranchesRecordNoJudgeSpend:
    """A judge that never ran leaves no judge spend behind on any branch."""

    def _cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    # -- (a) NO FABRICATED SPEND on every judge-skipped branch ----------------
    # Each of these branches already asserts ``mocks['judge'].assert_not_called()``
    # in test_eval_architect.py; here the pin is on the COST fields specifically —
    # a judge that never ran must never leave a nonzero judge_cost_usd/
    # judge_invocations behind, and the cell's cost_usd must equal the
    # ARCHITECT's own spend for that branch, nothing more.

    async def test_cap_tainted_branch_records_architect_spend_only(self):
        """Cap landed right after ``create_plan``: header-only stub, tainted."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=stub_with_steps([]),
            arch_result=refused_agent_result(cost_usd=0.0),
        )
        mocks['judge'].assert_not_called()
        assert result.metrics['judge_cost_usd'] == 0.0
        assert result.metrics['judge_invocations'] == 0
        # refused_agent_result(cost_usd=0.0) itself spends $0.00 — no fabricated judge
        # dollars riding along with a refusal that burned nothing.
        assert result.metrics['cost_usd'] == pytest.approx(0.0)

    async def test_cap_refusal_with_a_plan_records_architect_spend_only(self):
        """Cap landed MID-run, after a real plan with steps had landed."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan=well_formed_plan(),
            arch_result=refused_agent_result(cost_usd=0.0),
        )
        mocks['judge'].assert_not_called()
        assert result.metrics['judge_cost_usd'] == 0.0
        assert result.metrics['judge_invocations'] == 0
        assert result.metrics['cost_usd'] == pytest.approx(0.0)

    async def test_healthy_stepless_plan_records_architect_spend_only(self):
        """Architect ran fine (no cap) but produced nothing worth judging."""
        result, mocks = await run_architect_eval_hermetic(
            self._cfg(),
            produced_plan={},
            arch_success=True,
        )
        mocks['judge'].assert_not_called()
        assert result.metrics['judge_cost_usd'] == 0.0
        assert result.metrics['judge_invocations'] == 0
        # No arch_result override here — the harness's default architect
        # mock reports DEFAULT_ARCH_INVOKE.cost_usd, and that must be the WHOLE cell cost.
        assert result.metrics['cost_usd'] == pytest.approx(DEFAULT_ARCH_INVOKE.cost_usd)
