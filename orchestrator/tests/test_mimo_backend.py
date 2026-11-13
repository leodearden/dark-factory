"""Tests for orchestrator.agents.mimo_backend, the MiMo Code harness arm's translation layer.

Fixtures follow the shapes measured in plans/mimo-code-harness-probe-2026-11.md.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import stat
from pathlib import Path

import pytest
from _mimo_fixture import measured_export
from _plan_tools_mcp_fixture import plan_tools_mcp
from shared.cli_invoke import _SubprocessResult
from shared.process_cost import ProcessCostBasis

from orchestrator.agents import mimo_backend
from orchestrator.agents.mimo_backend import (
    MimoEndpoint,
    MimoEndpointMissing,
    MimoRunHome,
    mimo_config_document,
    mimo_export_argv,
    mimo_permission_rules,
    mimo_run_argv,
    mimo_run_home,
    mimo_session_list_argv,
    mimo_spawn_env,
    mimo_variant,
    parse_mimo_sessions,
    read_mimo_sessions,
)

TOKEN = 'tp-secret-value'
AMS_ANTHROPIC = 'https://token-plan-ams.xiaomimimo.com/anthropic'
AMS_OPENAI = 'https://token-plan-ams.xiaomimimo.com/v1'
ENDPOINT = MimoEndpoint(openai_base_url=AMS_OPENAI, api_key=TOKEN)
MODEL = 'mimo-v2.6-pro'


def _subprocess_result(
    stdout: str = '', stderr: str = '', returncode: int = 0, timed_out: bool = False,
) -> _SubprocessResult:
    return _SubprocessResult(
        stdout=stdout, stderr=stderr, returncode=returncode, duration_ms=1234,
        timed_out=timed_out,
    )


def _cost_of(uncached_input: int, output: int, cached_input: int) -> float:
    return 0.001 * uncached_input + 0.01 * output + 0.0001 * cached_input


class TestMimoEndpoint:

    @pytest.mark.parametrize('host', [
        'https://token-plan-cn.xiaomimimo.com',
        'https://token-plan-sgp.xiaomimimo.com',
        'https://token-plan-ams.xiaomimimo.com',
        'https://api.xiaomimimo.com',
    ])
    @pytest.mark.parametrize('suffix', ['', '/'])
    def test_maps_each_known_anthropic_base_to_its_openai_base(self, host, suffix):
        endpoint = MimoEndpoint.from_env_overrides({
            'ANTHROPIC_BASE_URL': f'{host}/anthropic{suffix}',
            'ANTHROPIC_AUTH_TOKEN': TOKEN,
            'OTHER': 'x',
        })
        assert endpoint == MimoEndpoint(openai_base_url=f'{host}/v1', api_key=TOKEN)

    def test_the_endpoint_is_frozen(self):
        for field in dataclasses.fields(ENDPOINT):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(ENDPOINT, field.name, 'other')

    def test_the_key_stays_out_of_the_repr(self):
        assert TOKEN not in repr(ENDPOINT)

    def test_an_unknown_base_is_refused_naming_it_and_the_known_bases(self):
        unknown = 'https://example.test/anthropic'
        with pytest.raises(MimoEndpointMissing) as excinfo:
            MimoEndpoint.from_env_overrides(
                {'ANTHROPIC_BASE_URL': unknown, 'ANTHROPIC_AUTH_TOKEN': TOKEN},
            )
        message = str(excinfo.value)
        assert unknown in message
        assert AMS_ANTHROPIC in message
        assert 'https://api.xiaomimimo.com/anthropic' in message
        assert TOKEN not in message

    @pytest.mark.parametrize('token', [None, '', '   '])
    def test_a_missing_or_blank_key_is_refused_without_leaking_the_value(self, token):
        env = {'ANTHROPIC_BASE_URL': AMS_ANTHROPIC}
        if token is not None:
            env['ANTHROPIC_AUTH_TOKEN'] = token
        with pytest.raises(MimoEndpointMissing) as excinfo:
            MimoEndpoint.from_env_overrides(env)
        message = str(excinfo.value)
        assert 'ANTHROPIC_AUTH_TOKEN' in message
        assert 'ANTHROPIC_BASE_URL' not in message

    @pytest.mark.parametrize('base', [None, '', '  '])
    def test_a_missing_or_blank_base_is_refused_without_leaking_the_token(self, base):
        env = {'ANTHROPIC_AUTH_TOKEN': TOKEN}
        if base is not None:
            env['ANTHROPIC_BASE_URL'] = base
        with pytest.raises(MimoEndpointMissing) as excinfo:
            MimoEndpoint.from_env_overrides(env)
        message = str(excinfo.value)
        assert 'ANTHROPIC_BASE_URL' in message
        assert TOKEN not in message

    @pytest.mark.parametrize('env', [None, {}])
    def test_no_overrides_names_both_variables(self, env):
        with pytest.raises(MimoEndpointMissing) as excinfo:
            MimoEndpoint.from_env_overrides(env)
        message = str(excinfo.value)
        assert 'ANTHROPIC_BASE_URL' in message
        assert 'ANTHROPIC_AUTH_TOKEN' in message

    def test_the_refusal_is_a_value_error(self):
        assert issubclass(MimoEndpointMissing, ValueError)


class TestMimoPermissionRules:

    def test_an_mcp_tool_denies_the_server_tool_name_verbatim(self):
        assert mimo_permission_rules(['mcp__fused-memory__set_task_status']) == {
            'fused-memory_set_task_status': 'deny',
        }

    def test_a_bash_pattern_denies_its_glob_under_an_allow_all(self):
        rules = mimo_permission_rules(['Bash(git push:*)'])
        assert rules == {'bash': {'*': 'allow', 'git push*': 'deny'}}

    def test_two_bash_patterns_merge_into_one_map_with_the_allow_all_first(self):
        rules = mimo_permission_rules(['Bash(git push:*)', 'Bash(rm -rf:*)'])
        assert rules == {'bash': {'*': 'allow', 'git push*': 'deny', 'rm -rf*': 'deny'}}
        assert next(iter(rules['bash'])) == '*'

    def test_a_plain_tool_name_is_lowercased(self):
        assert mimo_permission_rules(['WebFetch']) == {'webfetch': 'deny'}

    def test_a_wildcard_mcp_spec_is_dropped_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger='orchestrator.agents.mimo_backend'):
            rules = mimo_permission_rules(['mcp__x__*', 'WebFetch'])
        assert rules == {'webfetch': 'deny'}
        assert 'mcp__x__*' in caplog.text

    def test_none_denies_nothing(self):
        assert mimo_permission_rules(None) == {}


class TestMimoConfigDocument:

    def _document(self, **overrides):
        kwargs = {
            'model': MODEL, 'endpoint': ENDPOINT, 'mcp_config': plan_tools_mcp(),
            'permission': {'webfetch': 'deny'}, 'max_turns': 7,
        }
        kwargs.update(overrides)
        return mimo_config_document(**kwargs)

    def test_the_top_level_settings(self):
        doc = self._document()
        assert doc['$schema'] == 'https://mimo.xiaomi.com/mimocode/config.json'
        assert doc['model'] == f'mimo-token-plan/{MODEL}'
        assert doc['autoupdate'] is False
        assert doc['share'] == 'disabled'
        assert doc['permission'] == {'webfetch': 'deny'}
        assert doc['agent'] == {'build': {'steps': 7}}
        assert doc['dream'] == {'auto': False}
        assert doc['distill'] == {'auto': False}

    def test_one_openai_compatible_provider_holding_the_key(self):
        provider = self._document()['provider']
        assert list(provider) == ['mimo-token-plan']
        entry = provider['mimo-token-plan']
        assert entry['name']
        assert entry['only_configured_models'] is True
        assert entry['models'] == {MODEL: {
            'name': MODEL, 'reasoning': True,
            'limit': {'context': 1048576, 'output': 131072},
        }}
        assert entry['options'] == {'baseURL': AMS_OPENAI, 'apiKey': TOKEN}
        assert 'npm' not in entry

    def test_a_stdio_server_becomes_a_local_command(self):
        mcp = self._document()['mcp']
        assert mcp == {'plan-tools': {
            'type': 'local', 'command': ['/venv/bin/python', '-m', 'plan_tools'],
            'environment': {}, 'enabled': True,
        }}

    def test_a_stdio_server_carries_its_env(self):
        config = {'mcpServers': {'s': {
            'type': 'stdio', 'command': 'srv', 'args': [], 'env': {'A': 'b'},
        }}}
        assert self._document(mcp_config=config)['mcp']['s'] == {
            'type': 'local', 'command': ['srv'], 'environment': {'A': 'b'}, 'enabled': True,
        }

    @pytest.mark.parametrize('transport', ['http', 'sse'])
    def test_a_url_server_becomes_remote(self, transport):
        config = {'mcpServers': {
            'h': {'type': transport, 'url': 'http://127.0.0.1:1/mcp', 'headers': {'X': 'y'}},
            'bare': {'type': transport, 'url': 'http://127.0.0.1:2/mcp'},
        }}
        mcp = self._document(mcp_config=config)['mcp']
        assert mcp['h'] == {
            'type': 'remote', 'url': 'http://127.0.0.1:1/mcp', 'headers': {'X': 'y'},
            'enabled': True,
        }
        assert mcp['bare'] == {'type': 'remote', 'url': 'http://127.0.0.1:2/mcp', 'enabled': True}

    def test_a_server_with_neither_shape_is_refused_by_name(self):
        with pytest.raises(ValueError, match='odd-one'):
            self._document(mcp_config={'mcpServers': {'odd-one': {'type': 'ws'}}})

    def test_no_mcp_config_configures_no_servers(self):
        assert self._document(mcp_config=None)['mcp'] == {}


class TestMimoVariantAndArgv:

    @pytest.mark.parametrize('effort', ['low', 'medium', 'high'])
    def test_the_forwarded_variants_pass_through(self, effort):
        assert mimo_variant(effort) == effort

    @pytest.mark.parametrize('effort', [None, ''])
    def test_no_effort_sends_no_variant(self, effort):
        assert mimo_variant(effort) is None

    @pytest.mark.parametrize('effort', ['max', 'xhigh', 'minimal'])
    def test_any_other_effort_becomes_high_with_a_warning(self, effort, caplog):
        with caplog.at_level(logging.WARNING, logger='orchestrator.agents.mimo_backend'):
            assert mimo_variant(effort) == 'high'
        assert effort in caplog.text
        for forwarded in ('low', 'medium', 'high'):
            assert forwarded in caplog.text

    def test_the_run_argv(self, tmp_path):
        assert mimo_run_argv(MODEL, tmp_path, 'high') == [
            'mimo', 'run', '--title', mimo_backend.TITLE, '--format', 'json',
            '--dir', str(tmp_path), '--dangerously-skip-permissions',
            '-m', f'mimo-token-plan/{MODEL}', '--variant', 'high',
        ]

    def test_the_run_argv_without_an_effort_has_no_variant(self, tmp_path):
        argv = mimo_run_argv(MODEL, tmp_path, None)
        assert '--variant' not in argv
        assert argv[-2:] == ['-m', f'mimo-token-plan/{MODEL}']

    def test_the_session_store_argvs(self):
        assert mimo_session_list_argv() == ['mimo', 'session', 'list', '--format', 'json']
        assert mimo_export_argv('ses_a') == ['mimo', 'export', 'ses_a']


MAX_TURNS = 7


def _jsonl(*session_ids: str) -> str:
    """``mimo run --format json`` stdout: one event per id, each carrying its sessionID."""
    return ''.join(
        json.dumps({'type': 'step_finish', 'timestamp': 1, 'sessionID': sid}) + '\n'
        for sid in session_ids
    )


def _parse(exports, run: _SubprocessResult | None = None, *, max_turns: int = MAX_TURNS):
    return parse_mimo_sessions(
        exports, run or _subprocess_result(), max_turns=max_turns, cost_of=_cost_of,
    )


def _with_text(export, text: str):
    export['messages'][-1]['parts'] = [{'type': 'text', 'text': text}]
    return export


class TestParseMimoSessions:

    def test_one_completed_session(self):
        result = _parse([measured_export()])
        assert result.success is True
        assert result.subtype == 'success'
        assert result.output == 'probe complete'
        assert result.session_id == 'ses_x'
        assert result.turns == 2
        assert result.input_tokens == 1400
        assert result.output_tokens == 100
        assert result.cache_read_tokens == 600
        assert result.cache_create_tokens == 0
        assert result.cost_usd == _cost_of(1400, 100, 600)
        assert result.cost_basis is ProcessCostBasis.USAGE_ESTIMATE
        assert result.model_id is None
        assert result.duration_ms == 1234

    def test_sessions_sum_and_the_root_session_is_terminal(self):
        child = _with_text(measured_export('ses_child', parent_id='ses_x'), 'child text')
        result = _parse([child, measured_export()])
        assert result.success is True
        assert result.output == 'probe complete'
        assert result.session_id == 'ses_x'
        assert result.turns == 4
        assert result.input_tokens == 2800
        assert result.output_tokens == 200
        assert result.cache_read_tokens == 1200

    def test_the_root_is_the_session_the_stdout_names(self):
        exports = [measured_export('ses_a'), _with_text(measured_export('ses_b'), 'b text')]
        result = _parse(exports, _subprocess_result(stdout='not json\n' + _jsonl('ses_b')))
        assert result.success is True
        assert result.output == 'b text'
        assert result.session_id == 'ses_b'
        assert result.turns == 4

    def test_a_named_root_whose_export_is_missing_is_a_failure_with_a_warning(self, caplog):
        child = measured_export('ses_child', parent_id='ses_x')
        with caplog.at_level(logging.WARNING, logger='orchestrator.agents.mimo_backend'):
            result = _parse([child], _subprocess_result(stdout=_jsonl('ses_x')))
        assert result.success is False
        assert result.subtype == 'error_no_root_session'
        assert result.session_id == 'ses_x'
        assert result.input_tokens == 1400
        assert 'ses_child' in caplog.text

    def test_without_stdout_a_store_of_child_sessions_is_a_failure_with_a_warning(self, caplog):
        child = measured_export('ses_child', parent_id='ses_x')
        with caplog.at_level(logging.WARNING, logger='orchestrator.agents.mimo_backend'):
            result = _parse([child])
        assert result.success is False
        assert result.subtype == 'error_no_root_session'
        assert result.input_tokens == 1400
        assert 'ses_child' in caplog.text

    def test_without_stdout_two_parentless_sessions_are_a_failure_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger='orchestrator.agents.mimo_backend'):
            result = _parse([measured_export('ses_a'), measured_export('ses_b')])
        assert result.success is False
        assert result.subtype == 'error_no_root_session'
        assert result.input_tokens == 2800
        assert 'ses_a' in caplog.text
        assert 'ses_b' in caplog.text

    def test_a_root_that_reached_the_step_cap_is_error_max_turns(self):
        result = _parse([measured_export()], max_turns=2)
        assert result.success is False
        assert result.subtype == 'error_max_turns'
        assert result.turns == 2
        assert result.input_tokens == 1400
        assert 'probe complete' in result.output

    def test_the_step_cap_counts_only_the_root_sessions_steps(self):
        child = measured_export('ses_child', parent_id='ses_x')
        result = _parse([child, measured_export()], max_turns=3)
        assert result.success is True
        assert result.turns == 4

    def test_an_api_error_is_a_failure_carrying_its_status(self):
        export = measured_export(
            error={'name': 'APIError', 'data': {'statusCode': 401, 'message': 'm'}}, finish=None,
        )
        result = _parse([export])
        assert result.success is False
        assert result.subtype == 'error'
        assert result.api_error_status == 401
        assert 'm' in result.output

    @pytest.mark.parametrize('finish', ['tool-calls', None])
    def test_an_unfinished_terminal_step_is_a_failure(self, finish):
        result = _parse([measured_export(finish=finish)])
        assert result.success is False

    def test_an_export_without_session_info_still_counts_its_steps(self):
        export = measured_export()
        del export['info']
        result = _parse([export])
        assert result.success is True
        assert result.session_id == ''
        assert result.input_tokens == 1400

    def test_no_session_is_a_failure_with_no_tokens(self):
        result = _parse([])
        assert result.success is False
        assert result.subtype == 'error_no_session'
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.cache_read_tokens is None
        assert result.cache_create_tokens is None

    def test_a_timed_out_run_still_reports_its_tokens(self):
        result = _parse([measured_export()], _subprocess_result(timed_out=True, returncode=-9))
        assert result.success is False
        assert result.timed_out is True
        assert result.input_tokens == 1400
        assert result.cost_usd == _cost_of(1400, 100, 600)

    def test_a_nonzero_exit_is_a_failure(self):
        result = _parse([measured_export()], _subprocess_result(returncode=1, stderr='boom'))
        assert result.success is False
        assert result.stderr == 'boom'


class _FakeSessionStore:
    """An injected spawn answering the session-list and export argvs, recording each argv."""

    def __init__(self, ids=('ses_a', 'ses_b'), *, listing=None, exports=None):
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []
        self.listing = listing if listing is not None else _subprocess_result(
            stdout=json.dumps([{'id': sid} for sid in ids]),
        )
        self.exports = exports or {}

    async def __call__(self, argv: list[str], timeout: float) -> _SubprocessResult:
        self.calls.append(argv)
        self.timeouts.append(timeout)
        if argv[1:3] == ['session', 'list']:
            return self.listing
        assert argv[1] == 'export'
        sid = argv[2]
        return self.exports.get(sid) or _subprocess_result(stdout=json.dumps(measured_export(sid)))


@pytest.mark.asyncio
class TestReadMimoSessions:

    async def test_lists_then_exports_every_session_in_order(self):
        spawn = _FakeSessionStore()
        exports = await read_mimo_sessions(spawn)
        assert exports == [measured_export('ses_a'), measured_export('ses_b')]
        assert spawn.calls == [
            mimo_session_list_argv(), mimo_export_argv('ses_a'), mimo_export_argv('ses_b'),
        ]

    async def test_each_read_is_bounded_by_its_own_timeout(self):
        spawn = _FakeSessionStore()
        await read_mimo_sessions(spawn)
        assert spawn.timeouts == [mimo_backend.SESSION_READ_TIMEOUT_SECS] * 3

    @pytest.mark.parametrize('bad_export', [
        _subprocess_result(stdout=json.dumps(measured_export('ses_a')), returncode=1),
        _subprocess_result(stdout='not json'),
        _subprocess_result(stdout='[1, 2]'),
    ])
    async def test_an_unusable_export_is_skipped_with_a_warning(self, bad_export, caplog):
        spawn = _FakeSessionStore(exports={'ses_a': bad_export})
        with caplog.at_level(logging.WARNING, logger='orchestrator.agents.mimo_backend'):
            exports = await read_mimo_sessions(spawn)
        assert exports == [measured_export('ses_b')]
        assert 'ses_a' in caplog.text

    @pytest.mark.parametrize('listing', [
        _subprocess_result(stdout='[{"id": "ses_a"}]', returncode=1),
        _subprocess_result(stdout=''),
        _subprocess_result(stdout='{not json'),
        _subprocess_result(stdout='{"id": "ses_a"}'),
    ])
    async def test_an_unusable_listing_reads_nothing_with_a_warning(self, listing, caplog):
        spawn = _FakeSessionStore(listing=listing)
        with caplog.at_level(logging.WARNING, logger='orchestrator.agents.mimo_backend'):
            exports = await read_mimo_sessions(spawn)
        assert exports == []
        assert spawn.calls == [mimo_session_list_argv()]
        assert caplog.records

    async def test_an_entry_without_an_id_is_skipped(self):
        spawn = _FakeSessionStore(listing=_subprocess_result(
            stdout=json.dumps([{'title': 'no id'}, {'id': 'ses_b'}]),
        ))
        exports = await read_mimo_sessions(spawn)
        assert exports == [measured_export('ses_b')]
        assert spawn.calls == [mimo_session_list_argv(), mimo_export_argv('ses_b')]


OPERATOR_HOME = Path('/home/operator')
CONFIG_DOCUMENT = {'model': f'mimo-token-plan/{MODEL}', 'provider': {'k': TOKEN}}


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _run_home(system_prompt: str = 'SYS PROMPT'):
    return mimo_run_home(
        system_prompt=system_prompt, config_document=CONFIG_DOCUMENT, operator_home=OPERATOR_HOME,
    )


class TestMimoRunHome:

    def test_the_home_holds_the_config_and_instruction_files(self):
        with _run_home() as home:
            assert home.root.name.startswith('mimo-home-')
            assert _mode(home.root) == 0o700
            assert home.config_path == home.root / 'config' / 'mimocode.json'
            assert json.loads(home.config_path.read_text()) == CONFIG_DOCUMENT
            assert _mode(home.config_path) == 0o600
            assert (home.root / 'config' / 'AGENTS.md').read_text() == 'SYS PROMPT'
            assert (home.root / 'tmp').is_dir()
            assert home.operator_home == OPERATOR_HOME

    def test_the_home_value_is_frozen(self):
        with _run_home() as home:
            for field in dataclasses.fields(MimoRunHome):
                with pytest.raises(dataclasses.FrozenInstanceError):
                    setattr(home, field.name, Path('/elsewhere'))

    def test_the_home_is_removed_after_the_block(self):
        with _run_home() as home:
            root = home.root
        assert not root.exists()

    def test_the_home_is_removed_when_the_block_raises(self):
        with pytest.raises(RuntimeError, match='body failed'), _run_home() as home:
            root = home.root
            raise RuntimeError('body failed')
        assert not root.exists()


class TestMimoSpawnEnv:

    def test_the_child_env_points_mimo_at_the_run_home_and_drops_every_secret(self):
        base_env = {
            'PATH': '/usr/local/bin:/usr/bin',
            'HOME': '/home/operator',
            'MIMO_API_KEY': 'mimo-secret',
            'ANTHROPIC_AUTH_TOKEN': TOKEN,
            'ANTHROPIC_BASE_URL': AMS_ANTHROPIC,
            'SOME_TOKEN': 'some-secret',
            'CLAUDE_CODE_OAUTH_TOKEN': 'claude-secret',
        }
        with _run_home() as home:
            env = mimo_spawn_env(base_env, home)
            assert env['HOME'] == str(home.root)
            assert env['MIMOCODE_HOME'] == str(home.root)
            assert env['TMPDIR'] == str(home.root / 'tmp')
        assert env['PATH'] == '/usr/local/bin:/usr/bin'
        for name in ('MIMO_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'SOME_TOKEN',
                     'CLAUDE_CODE_OAUTH_TOKEN'):
            assert name not in env
        assert TOKEN not in env.values()

    def test_mimo_is_told_to_stay_offline_and_ignore_claude_code_mcp(self):
        with _run_home() as home:
            env = mimo_spawn_env({'HOME': '/home/operator'}, home)
        assert env['MIMOCODE_ENABLE_ANALYSIS'] == 'false'
        for flag in ('MIMOCODE_DISABLE_CLAUDE_CODE_MCP', 'MIMOCODE_DISABLE_MODELS_FETCH',
                     'MIMOCODE_DISABLE_AUTOUPDATE', 'MIMOCODE_DISABLE_CRON',
                     'MIMOCODE_DISABLE_SHARE'):
            assert env[flag] == '1'

    def test_no_inherited_mimocode_variable_reaches_mimo(self):
        base_env = {
            'HOME': '/home/operator',
            'MIMOCODE_CONFIG_CONTENT': '{"mcp": {"extra": {"type": "remote", "url": "x"}}}',
            'MIMOCODE_PERMISSION': '{"bash": "allow"}',
            'MIMOCODE_DISABLE_CLAUDE_CODE_MCP': '0',
            'MIMOCODE_HOME': '/home/operator/.mimocode',
        }
        with _run_home() as home:
            env = mimo_spawn_env(base_env, home)
            assert env['MIMOCODE_HOME'] == str(home.root)
        assert {name for name in env if name.startswith('MIMOCODE_')} == {
            'MIMOCODE_HOME', *mimo_backend.MIMOCODE_FLAGS,
        }
        assert env['MIMOCODE_DISABLE_CLAUDE_CODE_MCP'] == '1'

    def test_the_operator_tool_homes_are_kept(self):
        with _run_home() as home:
            env = mimo_spawn_env({'HOME': '/home/operator'}, home)
        assert env['XDG_CACHE_HOME'] == str(OPERATOR_HOME / '.cache')
        assert env['CARGO_HOME'] == str(OPERATOR_HOME / '.cargo')
