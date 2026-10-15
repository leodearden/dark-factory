"""Tests for _parse_codex_output, _parse_gemini_output, and pi-backend parsers
in invoke.py."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from shared.cli_invoke import _SubprocessResult

from orchestrator.agents.invoke import (
    _estimate_cost,
    _parse_codex_output,
    _parse_gemini_output,
    _parse_pi_output,
    _pi_thinking,
    _pi_tool_name,
    _warn_if_argv_near_arg_max,
    _write_pi_mcp_config,
)
from orchestrator.config import PriceEntry


def _make_subprocess_result(
    stdout: str = '',
    stderr: str = '',
    returncode: int = 0,
    duration_ms: int = 100,
) -> _SubprocessResult:
    """Construct a _SubprocessResult with sensible defaults."""
    return _SubprocessResult(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        duration_ms=duration_ms,
    )


class TestParseGeminiNullStats:

    def test_null_stats_does_not_raise(self):
        """_parse_gemini_output does not raise AttributeError when stats is JSON null."""
        payload = json.dumps({'response': 'ok', 'stats': None})
        result = _make_subprocess_result(stdout=payload)
        # Must not raise AttributeError
        agent_result = _parse_gemini_output(result, 'gemini-3-flash')
        assert agent_result.success is True
        assert agent_result.output == 'ok'
        assert agent_result.cost_usd == 0.0


class TestParseGeminiAbsentStats:

    def test_absent_stats_key_does_not_raise(self):
        """_parse_gemini_output does not raise when the 'stats' key is absent (not null)."""
        payload = json.dumps({'response': 'world'})  # no 'stats' key
        result = _make_subprocess_result(stdout=payload)
        agent_result = _parse_gemini_output(result, 'gemini-3-flash')
        assert agent_result.success is True
        assert agent_result.output == 'world'
        assert agent_result.cost_usd == 0.0


class TestParseGeminiValidStats:

    def test_valid_stats_computes_nonzero_cost(self):
        """_parse_gemini_output computes non-zero cost when stats contains token counts."""
        payload = json.dumps({
            'response': 'hello',
            'stats': {'input_tokens': 100, 'output_tokens': 50},
        })
        result = _make_subprocess_result(stdout=payload)
        agent_result = _parse_gemini_output(result, 'gemini-3-flash')
        assert agent_result.success is True
        assert agent_result.output == 'hello'
        # gemini-3-flash: input=0.075/1M, output=0.30/1M
        # cost = (100 * 0.075 + 50 * 0.30) / 1_000_000 = 0.0000225
        assert agent_result.cost_usd > 0.0


class TestParseCodexNullUsage:

    def test_null_usage_does_not_raise(self):
        """_parse_codex_output does not raise AttributeError when usage is JSON null."""
        events = [
            {'type': 'thread.started', 'thread_id': 'tid-1'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'hello'}},
            {'type': 'turn.completed', 'usage': None},
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload)
        # Must not raise AttributeError
        agent_result = _parse_codex_output(result, 'o4-mini')
        assert agent_result.success is True
        assert agent_result.output == 'hello'
        assert agent_result.cost_usd == 0.0
        assert agent_result.turns == 1


class TestParseCodexAbsentUsage:

    def test_absent_usage_key_does_not_raise(self):
        """_parse_codex_output does not raise when the 'usage' key is absent (not null)."""
        events = [
            {'type': 'thread.started', 'thread_id': 'tid-3'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'hello'}},
            {'type': 'turn.completed'},  # no 'usage' key
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload)
        agent_result = _parse_codex_output(result, 'o4-mini')
        assert agent_result.success is True
        assert agent_result.output == 'hello'
        assert agent_result.cost_usd == 0.0
        assert agent_result.turns == 1


class TestParseCodexValidUsage:

    def test_valid_usage_computes_nonzero_cost_and_turns(self):
        """_parse_codex_output computes non-zero cost and turns == 1 for real usage."""
        events = [
            {'type': 'thread.started', 'thread_id': 'tid-2'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'done'}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 200, 'output_tokens': 100}},
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload)
        agent_result = _parse_codex_output(result, 'o4-mini')
        assert agent_result.success is True
        assert agent_result.turns == 1
        # o4-mini: input=1.10/1M, output=4.40/1M
        # cost = (200 * 1.10 + 100 * 4.40) / 1_000_000 = 0.00066
        assert agent_result.cost_usd > 0.0


class TestParseCodexOutputConfigPrices:
    """`_parse_codex_output` sources rates from a passed-in config-shaped
    prices map (task 2459) rather than a hardcoded module-level table.
    """

    def test_plain_dict_prices_map_used_for_cost(self):
        events = [
            {'type': 'thread.started', 'thread_id': 'tid-cfg-1'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'done'}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 200, 'output_tokens': 100}},
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload)
        agent_result = _parse_codex_output(
            result, 'o4-mini',
            prices={'o4-mini': {'input_per_1m': 1.10, 'output_per_1m': 4.40}},
        )
        assert agent_result.cost_usd == pytest.approx((200 * 1.10 + 100 * 4.40) / 1_000_000)

    def test_price_entry_valued_prices_map_used_for_cost(self):
        """A `config.prices`-shaped map (PriceEntry values) is directly consumable,
        yielding the identical cost as the equivalent plain-dict map."""
        events = [
            {'type': 'thread.started', 'thread_id': 'tid-cfg-2'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'done'}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 200, 'output_tokens': 100}},
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload)
        agent_result = _parse_codex_output(
            result, 'o4-mini',
            prices={'o4-mini': PriceEntry(input_per_1m=1.10, output_per_1m=4.40)},
        )
        assert agent_result.cost_usd == pytest.approx((200 * 1.10 + 100 * 4.40) / 1_000_000)

    def test_none_prices_resolves_to_packaged_seed_table_without_warning(self, caplog):
        """The unthreaded (prices=None) path resolves to the packaged seed
        table — o4-mini is seeded, so no fallback warning is logged."""
        events = [
            {'type': 'thread.started', 'thread_id': 'tid-cfg-3'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'done'}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 200, 'output_tokens': 100}},
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload)
        with caplog.at_level(logging.WARNING):
            agent_result = _parse_codex_output(result, 'o4-mini')
        assert agent_result.cost_usd == pytest.approx((200 * 1.10 + 100 * 4.40) / 1_000_000)
        assert 'o4-mini' not in caplog.text


class TestParseGeminiOutputConfigPrices:
    """`_parse_gemini_output` sources rates from a passed-in config-shaped
    prices map (task 2459) rather than a hardcoded module-level table.
    """

    def test_plain_dict_prices_map_used_for_cost(self):
        payload = json.dumps({
            'response': 'hello',
            'stats': {'input_tokens': 100, 'output_tokens': 50},
        })
        result = _make_subprocess_result(stdout=payload)
        agent_result = _parse_gemini_output(
            result, 'gemini-3-flash',
            prices={'gemini-3-flash': {'input_per_1m': 0.075, 'output_per_1m': 0.30}},
        )
        assert agent_result.cost_usd == pytest.approx((100 * 0.075 + 50 * 0.30) / 1_000_000)


class TestParseCodexOutputUnknownModelFallback:
    """An un-listed model must never silently fall back — it logs a WARNING
    and uses the single defined `_FALLBACK_PRICE` (task 2459)."""

    def test_unknown_model_logs_warning_and_uses_fallback_price(self, caplog):
        events = [
            {'type': 'thread.started', 'thread_id': 'tid-fallback-1'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'done'}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 200, 'output_tokens': 100}},
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload)
        with caplog.at_level(logging.WARNING):
            agent_result = _parse_codex_output(
                result, 'totally-unknown-model',
                prices={'o4-mini': {'input_per_1m': 1.10, 'output_per_1m': 4.40}},
            )
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('totally-unknown-model' in r.getMessage() for r in warning_records), (
            f'expected a WARNING mentioning the unknown model; got: {[r.getMessage() for r in warning_records]}'
        )
        # _FALLBACK_PRICE: input=2.0/1M, output=8.0/1M
        assert agent_result.cost_usd == pytest.approx((200 * 2.0 + 100 * 8.0) / 1_000_000)


# Measured gpt-6-astra rollout usage (task 5403): codex counts cached tokens
# INSIDE input_tokens.
_ASTRA_INPUT_TOKENS = 311881
_ASTRA_CACHED_INPUT_TOKENS = 277760
_ASTRA_OUTPUT_TOKENS = 1442


class TestEstimateCostCachedInput:
    """The cached share of input_tokens is priced at cached_input_per_1m and
    only the remainder at input_per_1m; an entry with no cached rate prices
    every input token at input_per_1m, exactly as before."""

    @pytest.mark.parametrize('as_price_entry', [True, False])
    def test_measured_astra_rollout_prices_the_cached_share_at_the_cached_rate(
        self, as_price_entry: bool,
    ) -> None:
        rates = {'input_per_1m': 10.0, 'output_per_1m': 50.0, 'cached_input_per_1m': 1.0}
        prices = {'gpt-6-astra': PriceEntry(**rates) if as_price_entry else rates}

        cost = _estimate_cost(
            'gpt-6-astra', _ASTRA_INPUT_TOKENS, _ASTRA_OUTPUT_TOKENS, prices,
            cached_input_tokens=_ASTRA_CACHED_INPUT_TOKENS,
        )

        # ((311881 - 277760) * 10 + 277760 * 1 + 1442 * 50) / 1e6 = 0.69107,
        # against 3.19091 with every input token at the uncached rate.
        assert cost == pytest.approx(0.69107)

    def test_entry_without_cached_rate_prices_cached_tokens_at_input_rate(self) -> None:
        prices = {'gpt-6-astra': PriceEntry(input_per_1m=10.0, output_per_1m=50.0)}

        with_cached = _estimate_cost(
            'gpt-6-astra', _ASTRA_INPUT_TOKENS, _ASTRA_OUTPUT_TOKENS, prices,
            cached_input_tokens=_ASTRA_CACHED_INPUT_TOKENS,
        )

        assert with_cached == _estimate_cost(
            'gpt-6-astra', _ASTRA_INPUT_TOKENS, _ASTRA_OUTPUT_TOKENS, prices,
        )
        # (311881 * 10 + 1442 * 50) / 1e6
        assert with_cached == pytest.approx(3.19091)

    def test_fallback_price_without_cached_key_does_not_raise(self) -> None:
        cost = _estimate_cost('totally-unknown-model', 300, 100, {}, cached_input_tokens=200)

        # _FALLBACK_PRICE has no cached rate: (300 * 2.0 + 100 * 8.0) / 1e6
        assert cost == pytest.approx((300 * 2.0 + 100 * 8.0) / 1_000_000)


# Measured verbatim from codex-cli 0.154.0 on 2026-09-11, carried by both an
# `error` and a `turn.failed` event while `codex exec` exited 0.
_CODEX_USAGE_LIMIT_MESSAGE = (
    "You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), "
    'visit https://chatgpt.com/codex/settings/usage to purchase more credits or try again at 1:07 PM.'
)
# A codex-cli 0.154.0 binary string: an item-level `error` is a warning the
# run continues past ("Defaulting to fallback metadata"), not a failure.
_CODEX_MODEL_METADATA_WARNING = (
    'Model metadata for `gpt-6-astra` not found. Defaulting to fallback metadata; '
    'this can degrade performance and cause issues.'
)


def _codex_stream(*events: dict[str, Any]) -> _SubprocessResult:
    return _make_subprocess_result(stdout='\n'.join(json.dumps(e) for e in events), returncode=0)


class TestParseCodexErrorCarriers:
    """codex exits 0 on a failed turn, so the failure lives only in the JSONL
    stream: `turn.failed` and item-level `error` events must not be dropped."""

    def test_lone_turn_failed_is_a_failure_carrying_its_message(self) -> None:
        agent_result = _parse_codex_output(_codex_stream(
            {'type': 'thread.started', 'thread_id': 'tid-tf-1'},
            {'type': 'turn.started'},
            {'type': 'turn.failed', 'error': {'message': _CODEX_USAGE_LIMIT_MESSAGE}},
        ), 'gpt-6-astra')

        assert agent_result.success is False
        assert _CODEX_USAGE_LIMIT_MESSAGE in agent_result.output

    def test_item_level_error_text_is_surfaced(self) -> None:
        agent_result = _parse_codex_output(_codex_stream(
            {'type': 'thread.started', 'thread_id': 'tid-ie-1'},
            {'type': 'item.completed', 'item': {'type': 'error', 'message': _CODEX_MODEL_METADATA_WARNING}},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'done'}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 5}},
        ), 'gpt-6-astra')

        assert _CODEX_MODEL_METADATA_WARNING in agent_result.output

    def test_item_level_error_alone_does_not_fail_a_completed_turn(self) -> None:
        agent_result = _parse_codex_output(_codex_stream(
            {'type': 'thread.started', 'thread_id': 'tid-ie-2'},
            {'type': 'item.completed', 'item': {'type': 'error', 'message': _CODEX_MODEL_METADATA_WARNING}},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'done'}},
            {'type': 'turn.completed', 'usage': {'input_tokens': 10, 'output_tokens': 5}},
        ), 'gpt-6-astra')

        assert agent_result.success is True

    def test_measured_error_and_turn_failed_pair_reports_the_message_once(self) -> None:
        agent_result = _parse_codex_output(_codex_stream(
            {'type': 'thread.started', 'thread_id': 'tid-pair-1'},
            {'type': 'turn.started'},
            {'type': 'error', 'message': _CODEX_USAGE_LIMIT_MESSAGE},
            {'type': 'turn.failed', 'error': {'message': _CODEX_USAGE_LIMIT_MESSAGE}},
        ), 'gpt-6-astra')

        assert agent_result.success is False
        assert agent_result.output.count(_CODEX_USAGE_LIMIT_MESSAGE) == 1


_ASTRA_PRICES = {
    'gpt-6-astra': PriceEntry(input_per_1m=10.0, output_per_1m=50.0, cached_input_per_1m=1.0),
}


def _codex_turns(*usages: dict[str, int]) -> _SubprocessResult:
    return _codex_stream(
        {'type': 'thread.started', 'thread_id': 'tid-usage'},
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'done'}},
        *({'type': 'turn.completed', 'usage': usage} for usage in usages),
    )


class TestParseCodexCachedUsage:
    """turn.completed usage carries cached_input_tokens INSIDE input_tokens;
    the parser prices that split and reports every token count it read."""

    _MEASURED_USAGE = {
        'input_tokens': _ASTRA_INPUT_TOKENS,
        'cached_input_tokens': _ASTRA_CACHED_INPUT_TOKENS,
        'cache_write_input_tokens': 0,
        'output_tokens': _ASTRA_OUTPUT_TOKENS,
    }

    def test_measured_usage_prices_the_cached_split(self) -> None:
        agent_result = _parse_codex_output(
            _codex_turns(self._MEASURED_USAGE), 'gpt-6-astra', _ASTRA_PRICES,
        )

        # ((311881 - 277760) * 10 + 277760 * 1 + 1442 * 50) / 1e6
        assert agent_result.cost_usd == pytest.approx(0.69107)

    def test_measured_usage_populates_the_token_fields(self) -> None:
        agent_result = _parse_codex_output(
            _codex_turns(self._MEASURED_USAGE), 'gpt-6-astra', _ASTRA_PRICES,
        )

        assert agent_result.input_tokens == _ASTRA_INPUT_TOKENS
        assert agent_result.output_tokens == _ASTRA_OUTPUT_TOKENS
        assert agent_result.cache_read_tokens == _ASTRA_CACHED_INPUT_TOKENS
        assert agent_result.cache_create_tokens == 0

    def test_usage_across_two_turns_adds_field_wise(self) -> None:
        second = {
            'input_tokens': 1000, 'cached_input_tokens': 400,
            'cache_write_input_tokens': 50, 'output_tokens': 20,
        }

        agent_result = _parse_codex_output(
            _codex_turns(self._MEASURED_USAGE, second), 'gpt-6-astra', _ASTRA_PRICES,
        )

        assert agent_result.turns == 2
        assert agent_result.input_tokens == 312881
        assert agent_result.output_tokens == 1462
        assert agent_result.cache_read_tokens == 278160
        assert agent_result.cache_create_tokens == 50
        # ((312881 - 278160) * 10 + 278160 * 1 + 1462 * 50) / 1e6
        assert agent_result.cost_usd == pytest.approx(0.69847)

    def test_cached_above_input_clamps_and_warns_once(self, caplog) -> None:
        usage = {'input_tokens': 1200, 'cached_input_tokens': 1500, 'output_tokens': 10}

        with caplog.at_level(logging.WARNING):
            agent_result = _parse_codex_output(_codex_turns(usage), 'gpt-6-astra', _ASTRA_PRICES)

        # Non-cached count clamped at 0: (0 * 10 + 1500 * 1 + 10 * 50) / 1e6
        assert agent_result.cost_usd == pytest.approx(0.002)
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert '1500' in warnings[0]
        assert '1200' in warnings[0]

    def test_usage_without_cached_keys_prices_as_before(self) -> None:
        agent_result = _parse_codex_output(
            _codex_turns({'input_tokens': 200, 'output_tokens': 100}),
            'gpt-6-astra', _ASTRA_PRICES,
        )

        assert agent_result.cost_usd == (200 * 10.0 + 100 * 50.0) / 1_000_000


class TestPiToolName:
    """`_pi_tool_name` maps a Claude-style tool spec to pi's direct-tool
    name (spike Q3, live-confirmed against pi-mcp-adapter's
    formatToolName/getServerPrefix)."""

    def test_concrete_mcp_tool_uses_server_prefix_formula(self):
        """Concrete `mcp__<server>__<tool>` specs map to
        `<server '-'->'_'>_<tool>` — server hyphens become underscores,
        the tool name is used verbatim."""
        assert _pi_tool_name('mcp__fused-memory__add_memory') == 'fused_memory_add_memory'
        assert _pi_tool_name('mcp__escalation__merge_request') == 'escalation_merge_request'
        assert _pi_tool_name('mcp__plan-tools__create_plan') == 'plan_tools_create_plan'

    def test_builtin_tools_map_to_pi_lowercase_names(self):
        """Claude's built-in tool names map to pi's lowercase built-ins."""
        assert _pi_tool_name('Bash') == 'bash'  # verified spike Q6
        assert _pi_tool_name('Read') == 'read'
        assert _pi_tool_name('Write') == 'write'
        assert _pi_tool_name('Edit') == 'edit'
        assert _pi_tool_name('Glob') == 'glob'
        assert _pi_tool_name('Grep') == 'grep'

    def test_wildcard_mcp_spec_is_unmappable(self):
        """A wildcard `mcp__server__*` spec is not expressible as a single
        direct-tool name — callers drop it from --tools (with a logged
        warning; see TestInvokePiFlags)."""
        assert _pi_tool_name('mcp__jcodemunch__*') is None


class TestPiThinking:
    """`_pi_thinking` maps a Claude-style effort level to pi's
    ``--thinking`` vocabulary (off/minimal/low/medium/high/xhigh/max)."""

    def test_recognized_effort_passes_through_unchanged(self):
        assert _pi_thinking('medium') == 'medium'
        assert _pi_thinking('minimal') == 'minimal'
        assert _pi_thinking('high') == 'high'

    def test_empty_effort_returns_none(self):
        assert _pi_thinking(None) is None
        assert _pi_thinking('') is None

    def test_unrecognized_effort_falls_back_to_high_not_dropped(self):
        """The whole reason _pi_thinking has a fallback branch: an
        unrecognized non-empty effort string must not be silently dropped
        (e.g. via a bare `.get(effort)` regressing to None) — it maps to
        pi's 'high' level instead."""
        assert _pi_thinking('bogus') == 'high'


class TestParsePiEmptyOutput:
    """`_parse_pi_output` on empty stdout mirrors _parse_codex_output's
    empty-output branch (deliverable #1)."""

    def test_empty_stdout_is_error_empty_output(self):
        result = _make_subprocess_result(stdout='', stderr='', returncode=0)
        agent_result = _parse_pi_output(result, 'anthropic/claude-haiku-4-5')
        assert agent_result.success is False
        assert agent_result.subtype == 'error_empty_output'
        assert 'no output' in agent_result.output.lower()

    def test_empty_stdout_propagates_timed_out(self):
        """timed_out propagates on the empty-stdout path too (a timeout
        with nothing captured yet still routes through this branch)."""
        result = _SubprocessResult(
            stdout='', stderr='timeout', returncode=1, duration_ms=100, timed_out=True,
        )
        agent_result = _parse_pi_output(result, 'anthropic/claude-haiku-4-5')
        assert agent_result.timed_out is True


# Two assistant `turn_end`/`agent_end` messages, per the spike's observed
# per-message usage/cost shape (plans/pi-spike-findings.md Q2): 2-turn run,
# turn1 cost.total=0.0009355 + turn2 cost.total=0.000887 == 0.0018225.
_PI_ASSISTANT_MSG_1 = {
    'role': 'assistant',
    'stopReason': 'stop',
    'usage': {'input': 600, 'output': 30, 'totalTokens': 663, 'cost': {'total': 0.0009355}},
    'content': [{'type': 'text', 'text': 'part1'}],
}
_PI_ASSISTANT_MSG_2 = {
    'role': 'assistant',
    'stopReason': 'stop',
    'usage': {'input': 500, 'output': 20, 'totalTokens': 520, 'cost': {'total': 0.000887}},
    'content': [{'type': 'text', 'text': 'part2'}],
}


class TestParsePiSuccess:
    """`_parse_pi_output` happy-path JSONL parse (deliverable #1): native
    per-message cost (NOT the price table), turns = count of `turn_end`
    events, session_id from the `session` event, output = the terminal
    assistant message's joined text."""

    def _build_stdout(self, *, with_deltas: bool = False) -> str:
        events: list[dict[str, Any]] = [
            {'type': 'session', 'id': 'sess-abc'},
            {'type': 'agent_start'},
            {'type': 'turn_start'},
        ]
        if with_deltas:
            events.append({'type': 'message_update', 'delta': {'type': 'text_delta', 'text': 'par'}})
        events.append({'type': 'turn_end', 'message': _PI_ASSISTANT_MSG_1})
        events.append({'type': 'turn_start'})
        if with_deltas:
            events.append({'type': 'tool_execution_update', 'toolName': 'bash', 'delta': 'chunk'})
        events.append({'type': 'turn_end', 'message': _PI_ASSISTANT_MSG_2})
        events.append({
            'type': 'agent_end',
            'messages': [_PI_ASSISTANT_MSG_1, _PI_ASSISTANT_MSG_2],
            'willRetry': False,
        })
        events.append({'type': 'agent_settled'})
        return '\n'.join(json.dumps(e) for e in events)

    def test_success_parses_session_turns_cost_and_output(self):
        result = _make_subprocess_result(stdout=self._build_stdout(), returncode=0)
        agent_result = _parse_pi_output(result, 'anthropic/claude-haiku-4-5')
        assert agent_result.success is True
        assert agent_result.session_id == 'sess-abc'
        assert agent_result.turns == 2
        # NATIVE per-message cost (sum over agent_end.messages[] assistant
        # cost.total), NOT the price table.
        assert agent_result.cost_usd == pytest.approx(0.0018225)
        # output = joined content[type=="text"].text of the *terminal*
        # assistant message (agent_end.messages[-1]) — assert it contains
        # that text rather than pinning a cross-turn join.
        assert 'part2' in agent_result.output

    def test_message_update_and_tool_execution_update_are_ignored(self):
        """Streaming delta events (message_update / tool_execution_update)
        must not affect cost/turns accounting."""
        result = _make_subprocess_result(stdout=self._build_stdout(with_deltas=True), returncode=0)
        agent_result = _parse_pi_output(result, 'anthropic/claude-haiku-4-5')
        assert agent_result.turns == 2
        assert agent_result.cost_usd == pytest.approx(0.0018225)


class TestParsePiStreamError:
    """`_parse_pi_output` must NOT derive success from the process exit code
    alone (spike Q1, LOAD-BEARING): pi's `--mode json -p` exits 0 even on a
    hard API failure (401/400/out-of-quota) — the failure lives IN the
    stream as a terminal assistant message with stopReason=='error' and/or
    errorMessage set, willRetry=false. This is the key divergence from
    _parse_codex_output, which leans on returncode==0 alone."""

    def test_stop_reason_error_is_failure_despite_returncode_zero(self):
        """returncode==0 but the terminal assistant message carries
        stopReason=='error' and errorMessage -> success False, subtype
        'error', even though the process exited cleanly."""
        error_message = {
            'role': 'assistant',
            'stopReason': 'error',
            'errorMessage': '401 invalid x-api-key',
            'usage': {'input': 50, 'output': 0, 'totalTokens': 50, 'cost': {'total': 0.0}},
            'content': [],
        }
        events = [
            {'type': 'session', 'id': 'sess-err-1'},
            {'type': 'turn_end', 'message': error_message},
            {'type': 'agent_end', 'messages': [error_message], 'willRetry': False},
            {'type': 'agent_settled'},
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload, returncode=0)
        agent_result = _parse_pi_output(result, 'anthropic/claude-haiku-4-5')
        assert agent_result.success is False
        assert agent_result.subtype == 'error'

    def test_non_error_stop_reason_is_success(self):
        """A natural terminal stopReason (e.g. 'tool_calls', not 'error')
        with returncode==0 is a success — only 'error'/errorMessage marks
        failure, not any other stopReason value."""
        message = {
            'role': 'assistant',
            'stopReason': 'tool_calls',
            'usage': {'input': 50, 'output': 10, 'totalTokens': 60, 'cost': {'total': 0.0005}},
            'content': [{'type': 'text', 'text': 'done'}],
        }
        events = [
            {'type': 'session', 'id': 'sess-ok-1'},
            {'type': 'turn_end', 'message': message},
            {'type': 'agent_end', 'messages': [message], 'willRetry': False},
            {'type': 'agent_settled'},
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload, returncode=0)
        agent_result = _parse_pi_output(result, 'anthropic/claude-haiku-4-5')
        assert agent_result.success is True
        assert agent_result.subtype == 'success'

    def test_error_message_present_without_error_stop_reason_is_failure(self):
        """A well-formed stopReason=='stop' but with errorMessage present is
        still a failure — errorMessage alone is sufficient to fail it."""
        message = {
            'role': 'assistant',
            'stopReason': 'stop',
            'errorMessage': 'partial failure mid-stream',
            'usage': {'input': 50, 'output': 5, 'totalTokens': 55, 'cost': {'total': 0.0002}},
            'content': [{'type': 'text', 'text': 'partial'}],
        }
        events = [
            {'type': 'session', 'id': 'sess-err-2'},
            {'type': 'turn_end', 'message': message},
            {'type': 'agent_end', 'messages': [message], 'willRetry': False},
            {'type': 'agent_settled'},
        ]
        payload = '\n'.join(json.dumps(e) for e in events)
        result = _make_subprocess_result(stdout=payload, returncode=0)
        agent_result = _parse_pi_output(result, 'anthropic/claude-haiku-4-5')
        assert agent_result.success is False
        assert agent_result.subtype == 'error'


class TestParsePiCostFallback:
    """`_parse_pi_output` falls back to the config price table
    (`_estimate_cost`, task 2459) ONLY when pi's native per-message
    `usage.cost.total` is absent or zero — the exotic custom-provider/
    base-url case the spike calls out (Q2). When native cost IS present
    and non-zero, it wins and the price table is not consulted."""

    def _build_stdout(self, *, cost_total: float | None) -> str:
        usage: dict[str, Any] = {'input': 400, 'output': 200, 'totalTokens': 600}
        if cost_total is not None:
            usage['cost'] = {'total': cost_total}
        message = {
            'role': 'assistant',
            'stopReason': 'stop',
            'usage': usage,
            'content': [{'type': 'text', 'text': 'done'}],
        }
        events = [
            {'type': 'session', 'id': 'sess-fallback'},
            {'type': 'turn_end', 'message': message},
            {'type': 'agent_end', 'messages': [message], 'willRetry': False},
            {'type': 'agent_settled'},
        ]
        return '\n'.join(json.dumps(e) for e in events)

    def test_no_native_cost_falls_back_to_price_table(self):
        """No `cost` sub-object at all on the assistant usage -> the config
        price table (_estimate_cost) is consulted, using a deliberately
        non-seed rate to prove it's not a coincidental match."""
        payload = self._build_stdout(cost_total=None)
        result = _make_subprocess_result(stdout=payload, returncode=0)
        agent_result = _parse_pi_output(
            result, 'o4-mini',
            prices={'o4-mini': {'input_per_1m': 100.0, 'output_per_1m': 100.0}},
        )
        assert agent_result.success is True
        assert agent_result.cost_usd == pytest.approx((400 * 100.0 + 200 * 100.0) / 1_000_000)

    def test_zero_native_cost_falls_back_to_price_table(self):
        """`cost.total`==0.0 (present but zero) is treated the same as
        absent -> the price table fallback still applies."""
        payload = self._build_stdout(cost_total=0.0)
        result = _make_subprocess_result(stdout=payload, returncode=0)
        agent_result = _parse_pi_output(
            result, 'o4-mini',
            prices={'o4-mini': {'input_per_1m': 100.0, 'output_per_1m': 100.0}},
        )
        assert agent_result.cost_usd == pytest.approx((400 * 100.0 + 200 * 100.0) / 1_000_000)

    def test_nonzero_native_cost_wins_over_price_table(self):
        """`cost.total` present and >0 -> the native value is used verbatim;
        the price table (a deliberately different, non-matching rate) is
        NOT consulted."""
        payload = self._build_stdout(cost_total=0.01234)
        result = _make_subprocess_result(stdout=payload, returncode=0)
        agent_result = _parse_pi_output(
            result, 'o4-mini',
            prices={'o4-mini': {'input_per_1m': 100.0, 'output_per_1m': 100.0}},
        )
        assert agent_result.cost_usd == pytest.approx(0.01234)


class TestWritePiMcpConfig:
    """`_write_pi_mcp_config` emits a `.mcp.json`-shaped config with
    `directTools: true` injected per server (spike Q3 — required so each
    MCP tool is individually allowlistable via --tools/--exclude-tools;
    without it pi-mcp-adapter exposes only the proxy `mcp` tool). Only the
    file CONTENT is pinned here — placement/the non-cwd config-path flag
    are resolved by _invoke_pi itself (see its docstring/comments)."""

    def test_servers_get_direct_tools_and_preserve_fields(self, tmp_path):
        config_path = tmp_path / 'pi-mcp-config.json'
        mcp_config = {
            'mcpServers': {
                'fused-memory': {'command': 'x', 'args': ['a'], 'env': {'K': 'V'}},
                'plan-tools': {'command': 'y', 'args': []},
            },
        }

        _write_pi_mcp_config(config_path, mcp_config)

        written = json.loads(config_path.read_text())
        assert set(written.keys()) == {'mcpServers'}
        servers = written['mcpServers']
        assert servers['fused-memory'] == {
            'command': 'x', 'args': ['a'], 'env': {'K': 'V'}, 'directTools': True,
        }
        assert servers['plan-tools'] == {
            'command': 'y', 'args': [], 'directTools': True,
        }

    def test_none_mcp_config_writes_empty_servers_without_raising(self, tmp_path):
        config_path = tmp_path / 'pi-mcp-config.json'

        _write_pi_mcp_config(config_path, None)

        written = json.loads(config_path.read_text())
        assert written == {'mcpServers': {}}


class TestWarnIfArgvNearArgMax:
    """`_warn_if_argv_near_arg_max` is _invoke_pi's guard against the user
    prompt (which, unlike --system-prompt, still travels on argv per the
    spike template) overflowing the OS exec() ARG_MAX and failing with
    E2BIG. `os.sysconf` is patched so the test doesn't depend on the host's
    real ARG_MAX."""

    def test_small_argv_does_not_warn(self, caplog, monkeypatch):
        monkeypatch.setattr(
            'orchestrator.agents.invoke.os.sysconf', lambda name: 2 * 1024 * 1024,
        )
        with caplog.at_level(logging.WARNING):
            _warn_if_argv_near_arg_max(['pi', '--model', 'x', 'hello'])
        assert not [r for r in caplog.records if 'ARG_MAX' in r.getMessage()]

    def test_large_argv_warns(self, caplog, monkeypatch):
        monkeypatch.setattr('orchestrator.agents.invoke.os.sysconf', lambda name: 1024)
        with caplog.at_level(logging.WARNING):
            _warn_if_argv_near_arg_max(['pi', 'x' * 1000])
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('ARG_MAX' in r.getMessage() for r in warning_records), (
            f'expected an ARG_MAX warning; got: {[r.getMessage() for r in warning_records]}'
        )

    def test_sysconf_unavailable_falls_back_without_raising(self, caplog, monkeypatch):
        def _raise(name):
            raise ValueError('unsupported sysconf name')

        monkeypatch.setattr('orchestrator.agents.invoke.os.sysconf', _raise)
        with caplog.at_level(logging.WARNING):
            _warn_if_argv_near_arg_max(['pi', 'hello'])  # must not raise
        assert not [r for r in caplog.records if 'ARG_MAX' in r.getMessage()]

    def test_empty_mcp_config_writes_empty_servers_without_raising(self, tmp_path):
        config_path = tmp_path / 'pi-mcp-config.json'

        _write_pi_mcp_config(config_path, {})

        written = json.loads(config_path.read_text())
        assert written == {'mcpServers': {}}
