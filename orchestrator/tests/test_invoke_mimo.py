"""invoke_agent(backend='mimo') driven end to end against a fake ``mimo`` executable on PATH.

The fake records what a real MiMo Code process would see (argv, environ, cwd,
stdin and the run-home files) for each of the run, session-list and export
subcommands, and keeps the session store shape measured in
plans/mimo-code-harness-probe-2026-11.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _mimo_fixture import measured_export
from _operator_home_fixture import OPERATOR_ENV_VARS, write_operator_home
from _plan_tools_mcp_fixture import plan_tools_mcp
from shared.process_cost import ProcessCostBasis

from orchestrator.agents import sandbox_dispatch
from orchestrator.agents.invoke import invoke_agent
from orchestrator.agents.mimo_backend import MimoEndpointMissing

SECRET = 'tp-secret'
BASE_URL = 'https://token-plan-ams.xiaomimimo.com/anthropic'
ENDPOINT_OVERRIDES = {'ANTHROPIC_BASE_URL': BASE_URL, 'ANTHROPIC_AUTH_TOKEN': SECRET}
PRICES = {'mimo-v2.6-pro': {'input_per_1m': 1.0, 'output_per_1m': 2.0}}

FAKE_MIMO = '''#!/usr/bin/env python3
import json
import os
import stat
import sys
import time
from pathlib import Path


def snapshot(path):
    path = Path(path)
    if not path.is_file():
        return None
    return {'text': path.read_text(), 'mode': stat.S_IMODE(path.stat().st_mode)}


home = os.environ.get('MIMOCODE_HOME', '')
store = Path(home, 'fake-store.json')
command = sys.argv[1] if len(sys.argv) > 1 else ''
if command == 'run':
    store.write_text(ERROR_EXPORT if os.environ.get('MIMO_FAKE_ERROR') else MEASURED_EXPORT)
record = {'argv': sys.argv[1:], 'environ': dict(os.environ), 'cwd': os.getcwd()}
if command == 'run':
    record['stdin'] = sys.stdin.read()
    record['config'] = snapshot(Path(home, 'config', 'mimocode.json'))
    record['agents_md'] = snapshot(Path(home, 'config', 'AGENTS.md'))
    record['mimocode_home'] = home
with open(os.environ['MIMO_FAKE_CAPTURE'], 'a') as capture:
    capture.write(json.dumps(record) + '\\n')
if command == 'run':
    if not os.environ.get('MIMO_FAKE_LOSE_STDOUT'):
        print(json.dumps({'type': 'step_finish', 'sessionID': 'ses_x'}), flush=True)
    time.sleep(float(os.environ.get('MIMO_FAKE_SLEEP') or 0))
elif command == 'session':
    print(json.dumps([{'id': 'ses_x'}] if store.exists() else []))
elif command == 'export':
    print(store.read_text())
sys.exit(0)
'''


@dataclass(frozen=True)
class FakeMimo:
    capture_path: Path

    def records(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.capture_path.read_text().splitlines()]

    def run_record(self) -> dict[str, Any]:
        [run] = [record for record in self.records() if record['argv'][0] == 'run']
        return run


@pytest.fixture
def fake_mimo(tmp_path, monkeypatch) -> FakeMimo:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    script = bin_dir / 'mimo'
    error_export = measured_export(
        error={'name': 'APIError', 'data': {'statusCode': 401, 'message': 'mock 401'}},
        finish=None,
    )
    script.write_text(
        FAKE_MIMO
        .replace('MEASURED_EXPORT', repr(json.dumps(measured_export('ses_x'))))
        .replace('ERROR_EXPORT', repr(json.dumps(error_export))),
    )
    script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ.get("PATH", "")}')
    capture_path = tmp_path / 'capture.jsonl'
    monkeypatch.setenv('MIMO_FAKE_CAPTURE', str(capture_path))
    return FakeMimo(capture_path=capture_path)


@pytest.fixture
def operator_home(tmp_path, monkeypatch) -> Path:
    op = write_operator_home(tmp_path / 'operator')
    monkeypatch.setenv('HOME', str(op))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    for name in OPERATOR_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return op


@pytest.fixture
def worktree(tmp_path) -> Path:
    wt = tmp_path / 'wt'
    wt.mkdir()
    (wt / 'README.md').write_text('fixture\n')
    return wt


def _listing(directory: Path) -> list[Path]:
    return sorted(path.relative_to(directory) for path in directory.rglob('*'))


async def _invoke(worktree: Path, **overrides: Any):
    kwargs: dict[str, Any] = {
        'prompt': 'do step-1',
        'system_prompt': 'SYS',
        'cwd': worktree,
        'model': 'mimo-v2.6-pro',
        'backend': 'mimo',
        'effort': 'high',
        'max_turns': 7,
        'env_overrides': dict(ENDPOINT_OVERRIDES),
        'mcp_config': plan_tools_mcp(),
        'disallowed_tools': ['mcp__fused-memory__set_task_status'],
        'timeout_seconds': 30,
        'prices': PRICES,
    }
    kwargs.update(overrides)
    return await invoke_agent(**kwargs)


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.mark.asyncio
class TestInvokeMimo:

    async def test_the_result_is_read_from_the_session_store(self, fake_mimo, worktree):
        result = await _invoke(worktree)
        assert result.success is True
        assert result.output == 'probe complete'
        assert result.session_id == 'ses_x'
        assert result.turns == 2
        assert result.input_tokens == 1400
        assert result.cache_read_tokens == 600
        assert result.cache_create_tokens == 0
        assert result.output_tokens == 100
        assert result.cost_usd == pytest.approx((2000 * 1.0 + 100 * 2.0) / 1_000_000)
        assert result.cost_basis is ProcessCostBasis.USAGE_ESTIMATE

    async def test_one_headless_run_then_the_store_read(self, fake_mimo, worktree):
        await _invoke(worktree)
        records = fake_mimo.records()
        assert [record['argv'][0] for record in records] == ['run', 'session', 'export']
        assert records[1]['argv'][:2] == ['session', 'list']
        assert records[2]['argv'][:2] == ['export', 'ses_x']
        run = fake_mimo.run_record()
        argv = run['argv']
        assert _flag_value(argv, '--dir') == str(worktree)
        assert _flag_value(argv, '-m') == 'mimo-token-plan/mimo-v2.6-pro'
        assert _flag_value(argv, '--variant') == 'high'
        assert _flag_value(argv, '--format') == 'json'
        assert '--dangerously-skip-permissions' in argv
        assert 'do step-1' not in argv
        assert run['stdin'] == 'do step-1'

    async def test_the_run_home_carries_the_endpoint_mcp_bounds_and_system_prompt(
        self, fake_mimo, worktree,
    ):
        await _invoke(worktree)
        run = fake_mimo.run_record()
        assert run['config']['mode'] == 0o600
        config = json.loads(run['config']['text'])
        options = config['provider']['mimo-token-plan']['options']
        assert options['baseURL'] == 'https://token-plan-ams.xiaomimimo.com/v1'
        assert options['apiKey'] == SECRET
        assert config['agent']['build']['steps'] == 7
        assert config['permission']['fused-memory_set_task_status'] == 'deny'
        assert config['mcp']['plan-tools']['type'] == 'local'
        assert run['agents_md']['text'] == 'SYS'

    async def test_the_key_reaches_mimo_only_through_its_config_file(self, fake_mimo, worktree):
        await _invoke(worktree)
        for record in fake_mimo.records():
            assert not [name for name, value in record['environ'].items() if SECRET in value]

    async def test_the_run_home_is_gone_and_the_worktree_untouched_afterwards(
        self, fake_mimo, worktree,
    ):
        before = _listing(worktree)
        await _invoke(worktree)
        assert not Path(fake_mimo.run_record()['mimocode_home']).exists()
        assert _listing(worktree) == before

    async def test_the_agent_commands_keep_the_operator_tool_homes(
        self, fake_mimo, worktree, operator_home,
    ):
        await _invoke(worktree)
        run = fake_mimo.run_record()
        environ = run['environ']
        assert environ['HOME'] == environ['MIMOCODE_HOME'] == run['mimocode_home']
        assert environ['XDG_CACHE_HOME'] == str(operator_home / '.cache')

    async def test_a_run_that_printed_nothing_still_succeeds_from_the_store(
        self, fake_mimo, worktree, monkeypatch,
    ):
        monkeypatch.setenv('MIMO_FAKE_LOSE_STDOUT', '1')
        result = await _invoke(worktree)
        assert result.success is True
        assert result.input_tokens == 1400
        assert result.output_tokens == 100
        assert result.cache_read_tokens == 600

    async def test_the_wall_clock_kills_a_run_whose_tokens_are_still_counted(
        self, fake_mimo, worktree, monkeypatch,
    ):
        monkeypatch.setenv('MIMO_FAKE_SLEEP', '60')
        result = await _invoke(worktree, timeout_seconds=3)
        assert result.timed_out is True
        assert result.success is False
        assert result.input_tokens == 1400
        assert not Path(fake_mimo.run_record()['mimocode_home']).exists()

    async def test_a_run_that_reached_its_step_cap_is_error_max_turns(self, fake_mimo, worktree):
        result = await _invoke(worktree, max_turns=2)
        assert json.loads(fake_mimo.run_record()['config']['text'])['agent']['build']['steps'] == 2
        assert result.success is False
        assert result.subtype == 'error_max_turns'
        assert result.input_tokens == 1400

    async def test_an_api_error_is_a_failure_with_its_status(
        self, fake_mimo, worktree, monkeypatch,
    ):
        monkeypatch.setenv('MIMO_FAKE_ERROR', '1')
        result = await _invoke(worktree)
        assert result.success is False
        assert result.api_error_status == 401

    async def test_an_unbounded_invocation_is_refused_before_any_spawn(
        self, fake_mimo, worktree,
    ):
        with pytest.raises(ValueError, match='wall-clock') as excinfo:
            await _invoke(worktree, timeout_seconds=None)
        assert not isinstance(excinfo.value, MimoEndpointMissing)
        assert not fake_mimo.capture_path.exists()

    async def test_a_missing_endpoint_key_is_refused_before_any_spawn(
        self, fake_mimo, worktree,
    ):
        with pytest.raises(MimoEndpointMissing):
            await _invoke(worktree, env_overrides={'ANTHROPIC_BASE_URL': BASE_URL})
        assert not fake_mimo.capture_path.exists()

    async def test_a_missing_home_is_refused_before_any_spawn(
        self, fake_mimo, worktree, monkeypatch,
    ):
        monkeypatch.delenv('HOME')
        with pytest.raises(ValueError, match='HOME'):
            await _invoke(worktree)
        assert not fake_mimo.capture_path.exists()

    async def test_every_spawn_gets_the_run_home_writable_inside_the_sandbox(
        self, fake_mimo, worktree, monkeypatch,
    ):
        wrapped: list[dict[str, Any]] = []

        def record_wrap(inner_cmd, cwd, writable_modules, writable_extras=None):
            wrapped.append({'modules': writable_modules, 'extras': list(writable_extras or [])})
            return inner_cmd

        monkeypatch.setattr(sandbox_dispatch, 'wrap_command', record_wrap)
        await _invoke(worktree, sandbox_modules=['orchestrator'], sandbox_extras=['/x'])
        home = fake_mimo.run_record()['mimocode_home']
        assert len(wrapped) == 3
        for call in wrapped:
            assert call['modules'] == ['orchestrator']
            assert '/x' in call['extras']
            assert home in call['extras']
