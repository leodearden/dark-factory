"""invoke_agent(backend='zcode') driven end to end against a fake ``zcode`` executable on PATH.

The fake records what a real ZCode process would see (argv, environ and the
run-home files) and prints the result shape measured in
plans/zcode-harness-probe-2026-11.md. Its environ includes the ``.env`` file
upstream would load: zai-org/ZCode a9fe401eed
packages/cli/src/env.ts::findDotenv and ::loadCliDotenv walk lexically up
from ``--cwd`` to the first regular ``.env`` and merge it without override.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _operator_home_fixture import (
    OPERATOR_EMAIL,
    OPERATOR_ENV_VARS,
    OPERATOR_IGNORED_FILE,
    write_operator_home,
)
from _plan_tools_mcp_fixture import plan_tools_mcp
from _zcode_fixture import BASE_URL, measured_output
from shared.process_cost import ProcessCostBasis

from orchestrator.agents import sandbox_dispatch
from orchestrator.agents.invoke import invoke_agent
from orchestrator.agents.zcode_backend import ZCodeEndpointMissing

SECRET = 'k-secret'
ENDPOINT_OVERRIDES = {'ANTHROPIC_BASE_URL': BASE_URL, 'ANTHROPIC_AUTH_TOKEN': SECRET}
PRICES = {'glm-5.3': {'input_per_1m': 1.0, 'output_per_1m': 2.0}}

FAKE_ZCODE = '''#!/usr/bin/env python3
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path


def snapshot(path):
    path = Path(path)
    if not path.is_file():
        return None
    return {'text': path.read_text(), 'mode': stat.S_IMODE(path.stat().st_mode)}


def git(repo, *args):
    return subprocess.run(['git', *args], cwd=repo, capture_output=True, text=True)


def git_probe():
    repo = Path(os.environ['TMPDIR'], 'probe-repo')
    repo.mkdir()
    (repo / 'kept.txt').write_text('kept\\n')
    (repo / OPERATOR_IGNORED_FILE).write_text('dropped\\n')
    git(repo, 'init', '-q')
    git(repo, 'add', '-A')
    commit = git(repo, 'commit', '-q', '-m', 'probe')
    return {
        'commit_rc': commit.returncode,
        'commit_stderr': commit.stderr,
        'author_email': git(repo, 'log', '-1', '--format=%ae').stdout.strip(),
        'tracked': git(repo, 'ls-files').stdout.splitlines(),
    }


def find_dotenv(start):
    current = os.path.abspath(start)
    while True:
        candidate = os.path.join(current, '.env')
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def dotenv_loaded_environ(dotenv_path):
    environ = dict(os.environ)
    if dotenv_path is not None:
        for line in Path(dotenv_path).read_text().splitlines():
            key, sep, value = line.partition('=')
            if sep and key.strip():
                environ.setdefault(key.strip(), value.strip())
    return environ


home = os.environ.get('HOME', '')
cwd_argument = sys.argv[sys.argv.index('--cwd') + 1]
dotenv_path = find_dotenv(cwd_argument)
capture = {
    'argv': sys.argv[1:],
    'environ': dotenv_loaded_environ(dotenv_path),
    'dotenv_path': dotenv_path,
    'cwd_realpath': os.path.realpath(cwd_argument),
    'process_cwd': os.getcwd(),
    'home': home,
    'provider_config': snapshot(os.environ.get('ZCODE_PERSONAL_PROVIDER_CONFIG_FILE', '')),
    'cli_config': snapshot(Path(home, '.zcode', 'cli', 'config.json')),
    'agents_md': snapshot(Path(home, '.zcode', 'AGENTS.md')),
}
if os.environ.get('ZCODE_FAKE_GIT_PROBE'):
    capture['git_probe'] = git_probe()
Path(os.environ['ZCODE_FAKE_CAPTURE']).write_text(json.dumps(capture))
time.sleep(float(os.environ.get('ZCODE_FAKE_SLEEP') or 0))
sys.stdout.write(MEASURED_JSON)
'''


@dataclass(frozen=True)
class FakeZCode:
    capture_path: Path

    def capture(self) -> dict[str, Any]:
        return json.loads(self.capture_path.read_text())


@pytest.fixture
def fake_zcode(tmp_path, monkeypatch) -> FakeZCode:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    script = bin_dir / 'zcode'
    script.write_text(
        FAKE_ZCODE
        .replace('MEASURED_JSON', repr(json.dumps(measured_output())))
        .replace('OPERATOR_IGNORED_FILE', repr(OPERATOR_IGNORED_FILE)),
    )
    script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ.get("PATH", "")}')
    capture_path = tmp_path / 'capture.json'
    monkeypatch.setenv('ZCODE_FAKE_CAPTURE', str(capture_path))
    return FakeZCode(capture_path=capture_path)


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
        'model': 'glm-5.3',
        'backend': 'zcode',
        'env_overrides': dict(ENDPOINT_OVERRIDES),
        'mcp_config': plan_tools_mcp(),
        'disallowed_tools': ['Bash(git push:*)', 'Write'],
        'timeout_seconds': 30,
        'prices': PRICES,
    }
    kwargs.update(overrides)
    return await invoke_agent(**kwargs)


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _inline_value(argv: list[str], flag: str) -> str:
    [value] = [arg.removeprefix(f'{flag}=') for arg in argv if arg.startswith(f'{flag}=')]
    return value


@pytest.mark.asyncio
class TestInvokeZCode:

    async def test_the_result_is_parsed_from_the_json_zcode_prints(self, fake_zcode, worktree):
        result = await _invoke(worktree)
        assert result.success is True
        assert result.output == 'probe complete'
        assert result.session_id == 'sess-123'
        assert result.turns == 2
        assert result.input_tokens == 600
        assert result.cache_read_tokens == 300
        assert result.cache_create_tokens == 100
        assert result.output_tokens == 50
        assert result.cost_usd == pytest.approx((600 * 1.0 + 400 * 1.0 + 50 * 2.0) / 1_000_000)
        assert result.cost_basis is ProcessCostBasis.USAGE_ESTIMATE

    async def test_the_argv_runs_one_headless_prompt_in_the_worktree(self, fake_zcode, worktree):
        await _invoke(worktree)
        capture = fake_zcode.capture()
        argv = capture['argv']
        assert _inline_value(argv, '--prompt') == 'do step-1'
        assert Path(_flag_value(argv, '--cwd')).parent == Path(capture['home'])
        assert Path(capture['cwd_realpath']) == worktree.resolve()
        assert Path(capture['process_cwd']) == worktree.resolve()
        assert _flag_value(argv, '--mode') == 'yolo'
        assert _flag_value(argv, '--output-format') == 'json'
        disallowed = _flag_value(argv, '--disallowed-tools').split()
        assert 'Write' in disallowed
        assert 'Bash' not in disallowed

    async def test_an_env_file_above_the_worktree_never_reaches_zcode(self, fake_zcode, tmp_path):
        checkout = tmp_path / 'checkout'
        checkout.mkdir()
        (checkout / '.env').write_text('ZCODE_DOTENV_CANARY=from-the-checkout\n')
        wt = checkout / '.worktrees' / 'wt'
        wt.mkdir(parents=True)
        (wt / 'README.md').write_text('fixture\n')
        await _invoke(wt)
        capture = fake_zcode.capture()
        assert 'ZCODE_DOTENV_CANARY' not in capture['environ']
        assert Path(capture['dotenv_path']) == Path(capture['home']) / '.env'

    async def test_the_run_home_carries_the_endpoint_mcp_and_system_prompt(
        self, fake_zcode, worktree,
    ):
        await _invoke(worktree)
        capture = fake_zcode.capture()
        provider = capture['provider_config']
        assert provider['mode'] == 0o600
        [rule] = json.loads(provider['text'])['config']['providerConfigRules']['providerRules']
        assert rule['config']['api']['baseUrl'] == BASE_URL
        assert rule['config']['access']['apiKey'] == SECRET
        assert rule['config']['personalModelIds'] == ['glm-5.3']
        assert capture['agents_md']['text'] == 'SYS'
        servers = json.loads(capture['cli_config']['text'])['mcp']['servers']
        assert servers['plan-tools']['command'] == '/venv/bin/python'

    async def test_the_key_reaches_zcode_only_through_its_provider_file(
        self, fake_zcode, worktree,
    ):
        await _invoke(worktree)
        environ = fake_zcode.capture()['environ']
        assert not [name for name, value in environ.items() if SECRET in value]

    async def test_the_run_home_is_gone_and_the_worktree_untouched_afterwards(
        self, fake_zcode, worktree,
    ):
        before = _listing(worktree)
        await _invoke(worktree)
        assert not Path(fake_zcode.capture()['home']).exists()
        assert _listing(worktree) == before

    async def test_the_wall_clock_kills_a_run_that_overstays(
        self, fake_zcode, worktree, monkeypatch,
    ):
        monkeypatch.setenv('ZCODE_FAKE_SLEEP', '5')
        result = await _invoke(worktree, timeout_seconds=1)
        assert result.timed_out is True
        assert result.success is False
        assert not Path(fake_zcode.capture()['home']).exists()

    async def test_an_unbounded_invocation_is_refused_before_any_spawn(
        self, fake_zcode, worktree,
    ):
        with pytest.raises(ValueError, match='wall-clock') as excinfo:
            await _invoke(worktree, timeout_seconds=None)
        assert not isinstance(excinfo.value, ZCodeEndpointMissing)
        assert not fake_zcode.capture_path.exists()

    async def test_a_missing_endpoint_key_is_refused_before_any_spawn(
        self, fake_zcode, worktree,
    ):
        with pytest.raises(ZCodeEndpointMissing):
            await _invoke(worktree, env_overrides={'ANTHROPIC_BASE_URL': BASE_URL})
        assert not fake_zcode.capture_path.exists()

    async def test_a_missing_home_is_refused_naming_zcode_before_any_spawn(
        self, fake_zcode, worktree, monkeypatch,
    ):
        monkeypatch.delenv('HOME')
        with pytest.raises(ValueError, match='HOME.*zcode|zcode.*HOME'):
            await _invoke(worktree)
        assert not fake_zcode.capture_path.exists()

    async def test_the_sandbox_is_anchored_on_the_worktree_and_admits_the_run_home(
        self, fake_zcode, worktree, monkeypatch,
    ):
        wrapped: list[dict[str, Any]] = []

        def record_wrap(inner_cmd, cwd, writable_modules, writable_extras=None):
            wrapped.append({
                'cwd': cwd, 'modules': writable_modules, 'extras': list(writable_extras or []),
            })
            return inner_cmd

        monkeypatch.setattr(sandbox_dispatch, 'wrap_command', record_wrap)
        await _invoke(worktree, sandbox_modules=['orchestrator'], sandbox_extras=['/x'])
        [call] = wrapped
        assert call['cwd'] == worktree
        assert call['modules'] == ['orchestrator']
        assert '/x' in call['extras']
        assert fake_zcode.capture()['home'] in call['extras']

    async def test_the_agent_commands_commit_with_the_operator_git_and_tool_homes(
        self, fake_zcode, worktree, operator_home, monkeypatch,
    ):
        monkeypatch.setenv('ZCODE_FAKE_GIT_PROBE', '1')
        await _invoke(worktree)
        capture = fake_zcode.capture()
        probe = capture['git_probe']
        assert probe['commit_rc'] == 0, probe['commit_stderr']
        assert probe['author_email'] == OPERATOR_EMAIL
        assert probe['tracked'] == ['kept.txt']
        environ = capture['environ']
        assert environ['CARGO_HOME'] == str(operator_home / '.cargo')
        assert environ['RUSTUP_HOME'] == str(operator_home / '.rustup')
        assert environ['XDG_CACHE_HOME'] == str(operator_home / '.cache')
        assert environ['HOME'] == capture['home']
