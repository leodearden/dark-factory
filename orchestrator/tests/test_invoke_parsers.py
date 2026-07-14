"""Tests for _parse_codex_output, _parse_gemini_output, and pi-backend parsers
in invoke.py."""

from __future__ import annotations

import json
import logging

import pytest
from shared.cli_invoke import _SubprocessResult

from orchestrator.agents.invoke import (
    _parse_codex_output,
    _parse_gemini_output,
    _parse_pi_output,
    _pi_tool_name,
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
        events = [
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
