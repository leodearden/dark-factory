"""Tests for orchestrator.agents.zcode_backend, the ZCode harness arm's translation layer.

Fixtures follow the shapes measured in plans/zcode-harness-probe-2026-11.md.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest
from shared.cli_invoke import _SubprocessResult
from shared.process_cost import ProcessCostBasis

from orchestrator.agents.zcode_backend import (
    ZCodeEndpoint,
    ZCodeEndpointMissing,
    parse_zcode_output,
    zcode_argv,
    zcode_cli_config,
    zcode_provider_config,
)

BASE_URL = 'https://api.z.ai/api/anthropic'
TOKEN = 'k-secret-value'
ENDPOINT = ZCodeEndpoint(base_url=BASE_URL, api_key=TOKEN)
ALWAYS_DISALLOWED = 'AskUserQuestion EnterPlanMode ExitPlanMode'

MEASURED_OUTPUT = {
    'sessionId': 'sess-123',
    'traceId': 'trace-1',
    'turnId': 'turn-1',
    'response': 'probe complete',
    'usage': {
        'source': 'provider',
        'modelRequestCount': 2,
        'inputTokens': 1000,
        'outputTokens': 50,
        'totalTokens': 1050,
        'cacheReadTokens': 300,
        'cacheWriteTokens': 100,
        'reasoningTokens': 0,
    },
    'eventCount': 12,
    'projection': {'status': 'completed', 'turnCount': 1},
}


def _subprocess_result(
    stdout: str = '', stderr: str = '', returncode: int = 0, timed_out: bool = False,
) -> _SubprocessResult:
    return _SubprocessResult(
        stdout=stdout, stderr=stderr, returncode=returncode, duration_ms=1234,
        timed_out=timed_out,
    )


def _recording_cost_of():
    calls: list[tuple[int, int, int]] = []

    def cost_of(uncached_input: int, output: int, cached_input: int) -> float:
        calls.append((uncached_input, output, cached_input))
        return 0.001 * uncached_input + 0.01 * output + 0.0001 * cached_input

    return cost_of, calls


class TestZCodeEndpoint:

    def test_reads_the_endpoint_pair_into_a_frozen_value(self):
        endpoint = ZCodeEndpoint.from_env_overrides(
            {'ANTHROPIC_BASE_URL': BASE_URL, 'ANTHROPIC_AUTH_TOKEN': TOKEN, 'OTHER': 'x'},
        )
        assert endpoint == ZCodeEndpoint(base_url=BASE_URL, api_key=TOKEN)
        for field in dataclasses.fields(endpoint):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(endpoint, field.name, 'other')

    @pytest.mark.parametrize('token', [None, '', '   '])
    def test_a_missing_or_blank_key_is_refused_without_leaking_the_value(self, token):
        env = {'ANTHROPIC_BASE_URL': BASE_URL}
        if token is not None:
            env['ANTHROPIC_AUTH_TOKEN'] = token
        with pytest.raises(ZCodeEndpointMissing) as excinfo:
            ZCodeEndpoint.from_env_overrides(env)
        message = str(excinfo.value)
        assert 'ANTHROPIC_AUTH_TOKEN' in message
        assert 'ANTHROPIC_BASE_URL' not in message
        assert BASE_URL not in message

    def test_a_missing_base_url_is_refused_without_leaking_the_token(self):
        with pytest.raises(ZCodeEndpointMissing) as excinfo:
            ZCodeEndpoint.from_env_overrides({'ANTHROPIC_AUTH_TOKEN': TOKEN})
        message = str(excinfo.value)
        assert 'ANTHROPIC_BASE_URL' in message
        assert TOKEN not in message

    @pytest.mark.parametrize('env', [None, {}])
    def test_no_overrides_names_both_variables(self, env):
        with pytest.raises(ZCodeEndpointMissing) as excinfo:
            ZCodeEndpoint.from_env_overrides(env)
        message = str(excinfo.value)
        assert 'ANTHROPIC_BASE_URL' in message
        assert 'ANTHROPIC_AUTH_TOKEN' in message

    def test_the_refusal_is_a_value_error(self):
        assert issubclass(ZCodeEndpointMissing, ValueError)


class TestZCodeProviderConfig:

    def test_one_anthropic_messages_rule_selected_as_the_default(self):
        doc = zcode_provider_config('glm-5.3', ENDPOINT)
        assert doc['schemaVersion'] == 1
        [rule] = doc['providerRules']
        config = rule['config']
        assert config['api'] == {'type': 'anthropic-messages', 'baseUrl': BASE_URL}
        assert config['access'] == {'type': 'api-key', 'apiKey': TOKEN}
        assert config['personalModelIds'] == ['glm-5.3']
        assert doc['defaultModelSelection'] == {
            'providerId': rule['providerId'], 'modelId': 'glm-5.3',
        }
        assert doc['modelConfigRules'] == {
            'providerModelRules': [], 'manualProviderModelRules': [],
        }

    def test_the_document_is_json_serialisable(self):
        doc = zcode_provider_config('glm-5.3', ENDPOINT)
        assert json.loads(json.dumps(doc)) == doc


class TestZCodeCliConfig:

    @pytest.mark.parametrize('declared_type', [{}, {'type': 'stdio'}])
    def test_a_stdio_server_maps_to_an_enabled_user_scope_server(self, declared_type):
        mcp_config = {'mcpServers': {'plan-tools': {
            **declared_type, 'command': '/venv/bin/python', 'args': ['-m', 'plan_tools'],
            'env': {'PLAN_PATH': '/wt/.task/plan.json'},
        }}}
        doc = zcode_cli_config(mcp_config)
        assert doc['features'] == {'mcp': True}
        assert doc['mcp']['servers'] == {'plan-tools': {
            'type': 'stdio', 'command': '/venv/bin/python', 'args': ['-m', 'plan_tools'],
            'env': {'PLAN_PATH': '/wt/.task/plan.json'}, 'enabled': True,
        }}

    def test_a_stdio_server_without_args_or_env_gets_empty_ones(self):
        doc = zcode_cli_config({'mcpServers': {'bare': {'command': 'srv'}}})
        assert doc['mcp']['servers']['bare'] == {
            'type': 'stdio', 'command': 'srv', 'args': [], 'env': {}, 'enabled': True,
        }

    @pytest.mark.parametrize('transport', ['http', 'sse'])
    def test_a_url_server_keeps_its_transport_and_headers(self, transport):
        doc = zcode_cli_config({'mcpServers': {
            'remote': {'type': transport, 'url': 'http://127.0.0.1:8002/mcp',
                       'headers': {'X-Task': '5399'}},
            'plain': {'type': transport, 'url': 'http://127.0.0.1:8003/mcp'},
        }})
        assert doc['mcp']['servers'] == {
            'remote': {'type': transport, 'url': 'http://127.0.0.1:8002/mcp',
                       'headers': {'X-Task': '5399'}, 'enabled': True},
            'plain': {'type': transport, 'url': 'http://127.0.0.1:8003/mcp', 'enabled': True},
        }

    @pytest.mark.parametrize('mcp_config', [None, {}, {'mcpServers': {}}])
    def test_no_servers_turns_the_mcp_feature_off(self, mcp_config):
        doc = zcode_cli_config(mcp_config)
        assert doc['features'] == {'mcp': False}
        assert doc['mcp']['servers'] == {}

    def test_a_server_with_neither_command_nor_url_is_refused_by_name(self):
        with pytest.raises(ValueError, match='broken-server'):
            zcode_cli_config({'mcpServers': {'broken-server': {'args': ['x']}}})


class TestZCodeArgv:

    def test_the_headless_argv_always_removes_the_interactive_tools(self, tmp_path):
        argv = zcode_argv('do step-1', tmp_path, None)
        assert argv == [
            'zcode', '--prompt', 'do step-1', '--cwd', str(tmp_path), '--mode', 'yolo',
            '--output-format', 'json', '--no-color', '--disallowed-tools', ALWAYS_DISALLOWED,
        ]

    def test_plain_tool_names_are_appended_once(self, tmp_path):
        argv = zcode_argv('p', tmp_path, ['Write', 'mcp__plan-tools__mark_step_done', 'Write',
                                          'AskUserQuestion'])
        assert argv[-2:] == [
            '--disallowed-tools',
            f'{ALWAYS_DISALLOWED} Write mcp__plan-tools__mark_step_done',
        ]

    @pytest.mark.parametrize('spec', ['Bash(git push:*)', 'mcp__x__*'])
    def test_a_pattern_spec_is_dropped_loudly_not_widened_to_the_whole_tool(
        self, tmp_path, caplog, spec,
    ):
        with caplog.at_level(logging.WARNING, logger='orchestrator.agents.zcode_backend'):
            argv = zcode_argv('p', tmp_path, [spec, 'Write'])
        names = argv[argv.index('--disallowed-tools') + 1]
        assert names == f'{ALWAYS_DISALLOWED} Write'
        assert spec not in ' '.join(argv)
        assert 'Bash' not in names.split()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(spec in r.getMessage() for r in warnings)

    def test_the_largest_single_argument_linux_accepts_passes(self, tmp_path):
        prompt = 'x' * 131071
        assert zcode_argv(prompt, tmp_path, None)[2] == prompt

    @pytest.mark.parametrize('n_bytes', [131072, 131073, 200_000])
    def test_a_prompt_beyond_max_arg_strlen_is_refused_with_its_byte_count(
        self, tmp_path, n_bytes,
    ):
        with pytest.raises(ValueError, match=str(n_bytes)):
            zcode_argv('x' * n_bytes, tmp_path, None)

    def test_the_bound_counts_utf8_bytes_not_characters(self, tmp_path):
        prompt = 'é' * 70_000
        with pytest.raises(ValueError, match='140000'):
            zcode_argv(prompt, tmp_path, None)


class TestParseZCodeOutput:

    def test_the_measured_success_json(self):
        cost_of, calls = _recording_cost_of()
        result = parse_zcode_output(
            _subprocess_result(stdout=json.dumps(MEASURED_OUTPUT)), cost_of=cost_of,
        )
        assert result.success is True
        assert result.subtype == 'success'
        assert result.output == 'probe complete'
        assert result.session_id == 'sess-123'
        assert result.turns == 2
        assert result.input_tokens == 600
        assert result.cache_read_tokens == 300
        assert result.cache_create_tokens == 100
        assert result.output_tokens == 50
        assert result.duration_ms == 1234
        assert calls == [(600, 50, 400)]
        assert result.cost_usd == cost_of(600, 50, 400)
        assert result.cost_basis is ProcessCostBasis.USAGE_ESTIMATE
        assert result.timed_out is False

    def test_cached_tokens_above_input_clamp_the_uncached_input_to_zero(self, caplog):
        payload = {**MEASURED_OUTPUT, 'usage': {
            **MEASURED_OUTPUT['usage'], 'inputTokens': 200,
            'cacheReadTokens': 300, 'cacheWriteTokens': 100,
        }}
        cost_of, calls = _recording_cost_of()
        with caplog.at_level(logging.WARNING, logger='orchestrator.agents.zcode_backend'):
            result = parse_zcode_output(
                _subprocess_result(stdout=json.dumps(payload)), cost_of=cost_of,
            )
        assert result.input_tokens == 0
        assert calls == [(0, 50, 400)]
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_an_error_exit_reports_stderr_not_stdout(self):
        cost_of, _ = _recording_cost_of()
        result = parse_zcode_output(
            _subprocess_result(stdout='partial reply text', stderr='Error: boom', returncode=1),
            cost_of=cost_of,
        )
        assert result.success is False
        assert 'boom' in result.output
        assert 'partial reply text' not in result.output
        assert result.stderr == 'Error: boom'

    def test_empty_stdout_is_an_empty_output_failure(self):
        cost_of, _ = _recording_cost_of()
        result = parse_zcode_output(_subprocess_result(stdout='  \n'), cost_of=cost_of)
        assert result.success is False
        assert result.subtype == 'error_empty_output'

    def test_non_json_stdout_on_a_clean_exit_is_a_failure(self):
        cost_of, _ = _recording_cost_of()
        result = parse_zcode_output(_subprocess_result(stdout='not json'), cost_of=cost_of)
        assert result.success is False

    @pytest.mark.parametrize('timed_out', [False, True])
    @pytest.mark.parametrize(('stdout', 'stderr', 'returncode'), [
        (json.dumps(MEASURED_OUTPUT), '', 0),
        ('partial', 'Error: boom', 1),
        ('', '', 0),
        ('not json', '', 0),
        ('', 'Process killed after 1s timeout (SIGTERM+SIGKILL)', 1),
    ])
    def test_timed_out_is_propagated_on_every_path(self, stdout, stderr, returncode, timed_out):
        cost_of, _ = _recording_cost_of()
        result = parse_zcode_output(
            _subprocess_result(stdout=stdout, stderr=stderr, returncode=returncode,
                               timed_out=timed_out),
            cost_of=cost_of,
        )
        assert result.timed_out is timed_out
        if timed_out:
            assert result.success is False
