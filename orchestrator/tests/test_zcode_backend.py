"""Tests for orchestrator.agents.zcode_backend, the ZCode harness arm's translation layer.

Fixtures follow the shapes measured in plans/zcode-harness-probe-2026-11.md.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest
from _operator_home_fixture import OPERATOR_EMAIL, OPERATOR_IGNORED_FILE, write_operator_home
from _plan_tools_mcp_fixture import plan_tools_mcp
from _zcode_fixture import BASE_URL, measured_output
from shared.cli_invoke import _SubprocessResult
from shared.process_cost import ProcessCostBasis

from orchestrator.agents.zcode_backend import (
    ZCodeEndpoint,
    ZCodeEndpointMissing,
    ZCodeRunHome,
    parse_zcode_output,
    zcode_argv,
    zcode_cli_config,
    zcode_provider_config,
    zcode_run_home,
    zcode_spawn_env,
)

TOKEN = 'k-secret-value'
ENDPOINT = ZCodeEndpoint(base_url=BASE_URL, api_key=TOKEN)
ALWAYS_DISALLOWED = 'AskUserQuestion EnterPlanMode ExitPlanMode'
PROMPT_PREFIX_BYTES = len('--prompt=')


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

    def test_the_key_stays_out_of_the_repr(self):
        assert TOKEN not in repr(ZCodeEndpoint(base_url=BASE_URL, api_key=TOKEN))


class TestZCodeProviderConfig:

    def test_one_anthropic_messages_rule_selected_as_the_default(self):
        doc = zcode_provider_config('glm-5.3', ENDPOINT)
        assert doc['schemaVersion'] == 1
        config = doc['config']
        [rule] = config['providerConfigRules']['providerRules']
        provider = rule['config']
        assert provider['group'] == 'standard-personal'
        assert provider['api'] == {'type': 'anthropic-messages', 'baseUrl': BASE_URL}
        assert provider['access'] == {'type': 'api-key', 'apiKey': TOKEN}
        assert provider['personalModelIds'] == ['glm-5.3']
        assert config['defaultModelSelection'] == {
            'providerId': rule['providerId'], 'modelId': 'glm-5.3',
        }
        assert config['modelConfigRules'] == {
            'providerModelRules': [], 'manualProviderModelRules': [],
        }

    def test_only_keys_the_strict_upstream_codec_accepts(self):
        doc = zcode_provider_config('glm-5.3', ENDPOINT)
        assert set(doc) == {'schemaVersion', 'config'}
        assert set(doc['config']) <= {
            'providerOrder', 'providerConfigRules', 'modelConfigRules', 'defaultModelSelection',
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

    def test_a_url_server_without_a_type_is_http(self):
        doc = zcode_cli_config({'mcpServers': {'remote': {'url': 'http://h/mcp'}}})
        assert doc['mcp']['servers']['remote'] == {
            'type': 'http', 'url': 'http://h/mcp', 'enabled': True,
        }

    @pytest.mark.parametrize('mcp_config', [None, {}, {'mcpServers': {}}])
    def test_no_servers_turns_the_mcp_feature_off(self, mcp_config):
        doc = zcode_cli_config(mcp_config)
        assert doc['features'] == {'mcp': False}
        assert doc['mcp']['servers'] == {}

    def test_a_server_with_neither_command_nor_url_is_refused_by_name(self):
        with pytest.raises(ValueError, match='broken-server'):
            zcode_cli_config({'mcpServers': {'broken-server': {'args': ['x']}}})


def _home_at(path: Path) -> ZCodeRunHome:
    return ZCodeRunHome(
        root=path, provider_config_path=path / 'provider_config.json',
        operator_home=OPERATOR_HOME, workspace=path / 'workspace',
    )


class TestZCodeArgv:

    def test_the_headless_argv_runs_in_the_home_s_workspace_without_the_interactive_tools(
        self, tmp_path,
    ):
        home = _home_at(tmp_path)
        argv = zcode_argv('do step-1', home, None)
        assert argv == [
            'zcode', '--prompt=do step-1', '--cwd', str(home.workspace), '--mode', 'yolo',
            '--output-format', 'json', '--no-color', '--disallowed-tools', ALWAYS_DISALLOWED,
        ]

    @pytest.mark.parametrize('prompt', ['- do step-1', '---\ntitle: x\n---\n', '--cwd=/', ''])
    def test_a_prompt_that_looks_like_an_option_stays_inline_in_one_token(self, tmp_path, prompt):
        home = _home_at(tmp_path)
        argv = zcode_argv(prompt, home, None)
        assert argv[1] == f'--prompt={prompt}'
        assert argv[2:4] == ['--cwd', str(home.workspace)]

    def test_plain_tool_names_are_appended_once(self, tmp_path):
        argv = zcode_argv('p', _home_at(tmp_path), [
            'Write', 'mcp__plan-tools__mark_step_done', 'Write', 'AskUserQuestion',
        ])
        assert argv[-2:] == [
            '--disallowed-tools',
            f'{ALWAYS_DISALLOWED} Write mcp__plan-tools__mark_step_done',
        ]

    @pytest.mark.parametrize('spec', ['Bash(git push:*)', 'mcp__x__*'])
    def test_a_pattern_spec_is_dropped_loudly_not_widened_to_the_whole_tool(
        self, tmp_path, caplog, spec,
    ):
        with caplog.at_level(logging.WARNING, logger='orchestrator.agents.zcode_backend'):
            argv = zcode_argv('p', _home_at(tmp_path), [spec, 'Write'])
        names = argv[argv.index('--disallowed-tools') + 1]
        assert names == f'{ALWAYS_DISALLOWED} Write'
        assert spec not in ' '.join(argv)
        assert 'Bash' not in names.split()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(spec in r.getMessage() for r in warnings)

    def test_the_largest_single_argument_linux_accepts_passes(self, tmp_path):
        prompt = 'x' * (131071 - PROMPT_PREFIX_BYTES)
        assert zcode_argv(prompt, _home_at(tmp_path), None)[1] == f'--prompt={prompt}'

    @pytest.mark.parametrize('n_bytes', [
        131072 - PROMPT_PREFIX_BYTES, 131072, 131073, 200_000,
    ])
    def test_a_prompt_beyond_max_arg_strlen_is_refused_with_its_byte_count(
        self, tmp_path, n_bytes,
    ):
        with pytest.raises(ValueError, match=str(n_bytes)):
            zcode_argv('x' * n_bytes, _home_at(tmp_path), None)

    def test_the_refusal_counts_the_flag_prefix_in_the_argument_size(self, tmp_path):
        n_bytes = 131072 - PROMPT_PREFIX_BYTES
        with pytest.raises(ValueError, match='131072 bytes, but'):
            zcode_argv('x' * n_bytes, _home_at(tmp_path), None)

    def test_the_bound_counts_utf8_bytes_not_characters(self, tmp_path):
        prompt = 'é' * 70_000
        with pytest.raises(ValueError, match='140000'):
            zcode_argv(prompt, _home_at(tmp_path), None)


class TestParseZCodeOutput:

    def test_the_measured_success_json(self):
        cost_of, calls = _recording_cost_of()
        result = parse_zcode_output(
            _subprocess_result(stdout=json.dumps(measured_output())), cost_of=cost_of,
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
        payload = measured_output()
        payload['usage'].update(inputTokens=200, cacheReadTokens=300, cacheWriteTokens=100)
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
        (json.dumps(measured_output()), '', 0),
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


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


OPERATOR_HOME = Path('/home/operator')
WORKTREE = Path('/srv/worktree')


def _run_home(
    system_prompt: str = 'SYS PROMPT', operator_home: Path = OPERATOR_HOME,
    worktree: Path = WORKTREE,
):
    return zcode_run_home(
        system_prompt=system_prompt, model='glm-5.3', endpoint=ENDPOINT,
        mcp_config=plan_tools_mcp(), operator_home=operator_home, worktree=worktree,
    )


class TestZCodeRunHome:

    def test_the_home_holds_the_provider_mcp_and_instruction_files(self, tmp_path):
        with _run_home() as home:
            assert home.root.is_dir()
            assert _mode(home.root) == 0o700
            assert home.root.is_relative_to(Path(tempfile.gettempdir()))
            assert not home.root.is_relative_to(tmp_path)
            assert home.provider_config_path == home.root / '.zcode' / 'v2' / 'provider_config.json'
            assert json.loads(home.provider_config_path.read_text()) == zcode_provider_config(
                'glm-5.3', ENDPOINT,
            )
            assert _mode(home.provider_config_path) == 0o600
            cli_config = home.root / '.zcode' / 'cli' / 'config.json'
            assert json.loads(cli_config.read_text()) == zcode_cli_config(plan_tools_mcp())
            assert (home.root / '.zcode' / 'AGENTS.md').read_text() == 'SYS PROMPT'
            assert (home.root / 'tmp').is_dir()

    def test_the_home_carries_the_operator_home(self, tmp_path):
        with _run_home(operator_home=tmp_path / 'op') as home:
            assert home.operator_home == tmp_path / 'op'

    def test_each_invocation_gets_a_fresh_home(self):
        with _run_home() as first, _run_home() as second:
            assert first.root != second.root

    def test_the_home_value_is_frozen(self):
        with _run_home() as home:
            for field in dataclasses.fields(ZCodeRunHome):
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

    def test_zcode_runs_in_a_link_to_the_worktree_inside_the_home(self, tmp_path):
        wt = tmp_path / 'wt'
        wt.mkdir()
        with _run_home(worktree=wt) as home:
            assert home.workspace.parent == home.root
            assert home.workspace.is_symlink()
            assert home.workspace.resolve() == wt.resolve()

    def test_the_home_holds_an_empty_env_file(self):
        with _run_home() as home:
            sentinel = home.root / '.env'
            assert sentinel.is_file()
            assert not sentinel.is_symlink()
            assert sentinel.read_text() == ''

    def test_removing_the_home_leaves_the_worktree_in_place(self, tmp_path):
        wt = tmp_path / 'wt'
        wt.mkdir()
        (wt / 'kept.txt').write_text('kept\n')
        with _run_home(worktree=wt) as home:
            root = home.root
        assert not root.exists()
        assert (wt / 'kept.txt').read_text() == 'kept\n'


class TestZCodeSpawnEnv:

    def test_the_child_env_points_zcode_at_the_run_home_and_drops_every_secret(self):
        base_env = {
            'PATH': '/usr/local/bin:/usr/bin',
            'ZAI_API_KEY': 'zai-secret',
            'ANTHROPIC_AUTH_TOKEN': TOKEN,
            'ANTHROPIC_BASE_URL': BASE_URL,
            'SOME_TOKEN': 'some-secret',
            'CLAUDE_CODE_OAUTH_TOKEN': 'claude-secret',
            'HOME': '/home/operator',
        }
        with _run_home() as home:
            env = zcode_spawn_env(base_env, home)
            assert env['HOME'] == str(home.root)
            assert env['USERPROFILE'] == str(home.root)
            assert env['ZCODE_DATA_BASE_DIR'] == str(home.root)
            assert env['ZCODE_PERSONAL_PROVIDER_CONFIG_FILE'] == str(home.provider_config_path)
            assert env['TMPDIR'] == str(home.root / 'tmp')
        assert env['PATH'] == '/usr/local/bin:/usr/bin'
        for name in ('ZAI_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'SOME_TOKEN',
                     'CLAUDE_CODE_OAUTH_TOKEN'):
            assert name not in env
        assert TOKEN not in env.values()

    def test_the_operator_tool_homes_default_under_the_operator_home(self):
        with _run_home() as home:
            env = zcode_spawn_env({'PATH': '/usr/bin', 'HOME': '/home/operator'}, home)
            assert env['HOME'] == str(home.root)
        assert env['XDG_CONFIG_HOME'] == '/home/operator/.config'
        assert env['XDG_CACHE_HOME'] == '/home/operator/.cache'
        assert env['XDG_DATA_HOME'] == '/home/operator/.local/share'
        assert env['XDG_STATE_HOME'] == '/home/operator/.local/state'
        assert env['CARGO_HOME'] == '/home/operator/.cargo'
        assert env['RUSTUP_HOME'] == '/home/operator/.rustup'

    def test_an_inherited_tool_home_wins_verbatim(self):
        base_env = {'HOME': '/home/operator', 'CARGO_HOME': '/opt/cargo',
                    'XDG_CACHE_HOME': '/var/cache/op'}
        with _run_home() as home:
            env = zcode_spawn_env(base_env, home)
        assert env['CARGO_HOME'] == '/opt/cargo'
        assert env['XDG_CACHE_HOME'] == '/var/cache/op'
        assert env['RUSTUP_HOME'] == '/home/operator/.rustup'

    def test_a_blank_inherited_tool_home_falls_back_to_the_operator_default(self):
        base_env = {'HOME': '/home/operator', 'CARGO_HOME': '', 'XDG_CONFIG_HOME': ''}
        with _run_home() as home:
            env = zcode_spawn_env(base_env, home)
        assert env['CARGO_HOME'] == '/home/operator/.cargo'
        assert env['XDG_CONFIG_HOME'] == '/home/operator/.config'


def _git(repo: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', *args], cwd=repo, env=env, capture_output=True, text=True, check=False,
    )


def _repo(root: Path) -> Path:
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True, env={
        'PATH': os.environ['PATH'], 'HOME': str(root), 'GIT_CONFIG_NOSYSTEM': '1',
    })
    (root / 'kept.txt').write_text('kept\n')
    (root / OPERATOR_IGNORED_FILE).write_text('dropped\n')
    return root


def _operator_env(operator_home: Path) -> dict[str, str]:
    return {'PATH': os.environ['PATH'], 'HOME': str(operator_home), 'GIT_CONFIG_NOSYSTEM': '1'}


class TestZCodeSpawnEnvGit:

    @pytest.mark.parametrize('home_name', ['operator', 'op#1;x', 'op"q\\z'])
    def test_git_sees_the_operator_identity_and_excludes(self, tmp_path, home_name):
        op = write_operator_home(tmp_path / home_name)
        repo = _repo(tmp_path / 'repo')
        with _run_home(operator_home=op) as home:
            env = zcode_spawn_env(_operator_env(op), home)
            ident = _git(repo, env, 'var', 'GIT_AUTHOR_IDENT')
            staged = _git(repo, env, 'add', '-A', '--dry-run')
        assert ident.returncode == 0, ident.stderr
        assert OPERATOR_EMAIL in ident.stdout
        assert staged.returncode == 0, staged.stderr
        assert 'kept.txt' in staged.stdout
        assert OPERATOR_IGNORED_FILE not in staged.stdout

    def test_an_operator_home_without_a_gitconfig_is_not_an_error(self, tmp_path):
        op = write_operator_home(tmp_path / 'operator', with_gitconfig=False)
        repo = _repo(tmp_path / 'repo')
        with _run_home(operator_home=op) as home:
            status = _git(repo, zcode_spawn_env(_operator_env(op), home), 'status')
        assert status.returncode == 0, status.stderr
