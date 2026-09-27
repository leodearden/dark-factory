"""Tests for scripts/sitting/nightly_prepare.py — the 05:30 headless Fable prepare run (task 5376).

The real ``claude`` is never reached: every test hands ``--claude-bin`` a
recording fake and ``main`` a hand-rolled gate, so no account is leased from
the live pool and no model is billed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from legibility import account_pool
from shared.cli_invoke import build_claude_argv
from shared.usage_gate import AccountLease
from sitting import nightly_prepare as mod
from sitting import prepare_sitting

REPO_ROOT = Path(__file__).resolve().parents[2]

OK_RESULT = json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'recorded 3'})
BANNER = "You've hit your usage limit · resets 5am (Europe/London)"
ENV_KEYS = ('CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_API_KEY', 'SITTING_TEST_MARKER')
TIMEOUT_SECS = '3'
"""Outlasts the fake's interpreter start under ``-n auto`` load (0.5s did not, 2 runs in 10), far short of its sleep."""

APPLY_VERBS = (
    'mcp__escalation__resolve_issue',
    'mcp__escalation__stamp_triage',
    'mcp__escalation__promote_to_l2',
    'mcp__escalation__declare_pin',
    'mcp__fused-memory__update_task',
    'mcp__fused-memory__add_dependency',
    'mcp__fused-memory__submit_task',
    'mcp__fused-memory__add_memory',
    'Edit',
    'Write',
    'NotebookEdit',
)

_FAKE_CLAUDE_SRC = '''#!{python}
import json, os, sys, time
from pathlib import Path

spec = json.loads(Path({spec!r}).read_text())
argv = sys.argv[1:]
record = {{
    'argv': sys.argv,
    'stdin': sys.stdin.read(),
    'cwd': os.getcwd(),
    'env': {{key: os.environ.get(key) for key in spec['env_keys']}},
    'env_has': {{key: key in os.environ for key in spec['env_keys']}},
}}
if '--system-prompt-file' in argv:
    path = argv[argv.index('--system-prompt-file') + 1]
    record['sysprompt_path'] = path
    record['sysprompt'] = Path(path).read_text() if os.path.exists(path) else None
with open(spec['record'], 'a') as sink:
    sink.write(json.dumps(record) + '\\n')
sys.stdout.write(spec['stdout'])
sys.stdout.flush()
sys.stderr.write(spec['stderr'])
sys.stderr.flush()
time.sleep(spec['sleep'])
sys.exit(spec['exit_code'])
'''


@dataclass
class FakeClaude:
    """A recording stand-in for the CLI: one JSON line per invocation in ``calls_file``."""

    bin: Path
    calls_file: Path

    def calls(self) -> list[dict]:
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]

    def only_call(self) -> dict:
        calls = self.calls()
        assert len(calls) == 1, f'expected exactly one claude invocation, got {len(calls)}'
        return calls[0]


def _fake_claude(tmp_path: Path, *, stdout: str = OK_RESULT, stderr: str = '', exit_code: int = 0,
                 sleep: float = 0.0) -> FakeClaude:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    calls_file = tmp_path / 'claude-calls.jsonl'
    spec_file = tmp_path / 'claude-spec.json'
    spec_file.write_text(json.dumps({
        'record': str(calls_file), 'stdout': stdout, 'stderr': stderr,
        'exit_code': exit_code, 'sleep': sleep, 'env_keys': list(ENV_KEYS),
    }))
    fake = bin_dir / 'claude'
    fake.write_text(_FAKE_CLAUDE_SRC.format(python=sys.executable, spec=str(spec_file)))
    fake.chmod(0o755)
    return FakeClaude(fake, calls_file)


@dataclass
class FakeGate:
    """The two gate members ``account_pool.subprocess_env`` calls, over ``(name, capped)`` accounts."""

    accounts: list[tuple[str, bool]]
    released: list[str | None] = field(default_factory=list)
    leases: int = 0

    def try_lease(self, *, scope=None, reverse=False, exclude=None):
        self.leases += 1
        roster = reversed(self.accounts) if reverse else self.accounts
        for name, capped in roster:
            if not capped and not (exclude and name in exclude):
                return AccountLease(name=name, token=f'tok-{name}', generation=0)
        return None

    def release_probe_slot(self, oauth_token):
        self.released.append(oauth_token)


class ExplodingGate:
    def try_lease(self, **_):
        raise RuntimeError('roster unreadable: simulated')

    def release_probe_slot(self, oauth_token):  # pragma: no cover - never leased
        raise AssertionError('nothing was leased')


def _live_gate() -> FakeGate:
    return FakeGate([('max-b', False), ('max-h', False)])


def _main(fake: FakeClaude, *extra: str, gate=None) -> int:
    return mod.main(['--claude-bin', str(fake.bin), *extra], gate=gate if gate is not None else _live_gate())


def _sysprompt_path(call: dict) -> Path:
    return Path(call['sysprompt_path'])


# ---------------------------------------------------------------------------
# 1. The argv is the shared builder's, the prompt goes on stdin, temp files go.
# ---------------------------------------------------------------------------


def test_argv_is_exactly_the_shared_builders_with_the_defaults(tmp_path):
    fake = _fake_claude(tmp_path)

    assert _main(fake) == mod.EXIT_OK

    call = fake.only_call()
    expected, temp_files = build_claude_argv(
        model=mod.DEFAULT_MODEL,
        max_budget_usd=mod.DEFAULT_BUDGET_USD,
        system_prompt=call['sysprompt'],
        max_turns=mod.DEFAULT_MAX_TURNS,
        permission_mode='dontAsk',
        allowed_tools=list(mod.NIGHTLY_ALLOWED_TOOLS),
        disallowed_tools=list(mod.NIGHTLY_DENIED_TOOLS),
        mcp_config=None,
        output_schema=None,
        effort=None,
        resume_session_id=None,
        session_id=None,
    )
    for path in temp_files:
        Path(path).unlink(missing_ok=True)
    expected[0] = str(fake.bin)
    expected[expected.index('--system-prompt-file') + 1] = call['sysprompt_path']
    assert call['argv'] == expected
    assert mod.DEFAULT_MODEL == 'fable', 'the task mandates the prepare-sitting mode run nightly on Fable'
    assert call['sysprompt'], 'the builder always writes a system prompt; it must not be empty'


def test_model_budget_and_turns_seams_reach_the_argv(tmp_path):
    fake = _fake_claude(tmp_path)

    assert _main(fake, '--model', 'opus', '--budget-usd', '2.5', '--max-turns', '17') == mod.EXIT_OK

    argv = fake.only_call()['argv']
    assert argv[argv.index('--model') + 1] == 'opus'
    assert argv[argv.index('--max-budget-usd') + 1] == '2.5'
    assert argv[argv.index('--max-turns') + 1] == '17'


def test_the_prompt_goes_on_stdin_and_names_the_skill_section(tmp_path):
    fake = _fake_claude(tmp_path)

    _main(fake)

    call = fake.only_call()
    assert call['stdin'] == mod.NIGHTLY_PROMPT
    assert 'skills/escalation-watcher/SKILL.md' in mod.NIGHTLY_PROMPT
    assert 'Sitting preparer' in mod.NIGHTLY_PROMPT
    assert f'{mod.PREPARE_COMMAND} record' in mod.NIGHTLY_PROMPT
    assert not any(mod.NIGHTLY_PROMPT in arg for arg in call['argv']), 'the prompt must not ride the argv'


def test_the_child_runs_in_this_checkout(tmp_path):
    """``claude -p`` scopes its tools to its cwd, and the prepare command's path is repo-relative."""
    fake = _fake_claude(tmp_path)

    _main(fake)

    assert Path(fake.only_call()['cwd']).resolve() == REPO_ROOT


@pytest.mark.parametrize(
    ('spec', 'extra'),
    [
        ({'exit_code': 0}, ()),
        ({'exit_code': 3, 'stderr': 'boom'}, ()),
        ({'sleep': 30.0}, ('--timeout-secs', TIMEOUT_SECS)),
    ],
    ids=['success', 'failure', 'timeout'],
)
def test_builder_temp_files_are_unlinked_after_the_run(tmp_path, spec, extra):
    fake = _fake_claude(tmp_path, **spec)

    _main(fake, *extra)

    call = fake.only_call()
    assert call['sysprompt'] is not None, 'the system prompt file must exist while the CLI runs'
    assert not _sysprompt_path(call).exists(), 'the builder temp file must be unlinked after the run'


# ---------------------------------------------------------------------------
# 2. Read-only by construction.
# ---------------------------------------------------------------------------


def test_allow_and_deny_lists_are_disjoint():
    assert not set(mod.NIGHTLY_ALLOWED_TOOLS) & set(mod.NIGHTLY_DENIED_TOOLS)


@pytest.mark.parametrize('verb', APPLY_VERBS)
def test_every_store_writing_apply_verb_is_denied(verb):
    assert verb in mod.NIGHTLY_DENIED_TOOLS


def test_the_only_bash_allowed_is_git_show_git_log_and_the_prepare_script():
    bash = [tool for tool in mod.NIGHTLY_ALLOWED_TOOLS if tool.startswith('Bash')]

    assert sorted(bash) == sorted([
        'Bash(git show:*)',
        'Bash(git log:*)',
        f'Bash({mod.PREPARE_COMMAND}:*)',
    ])
    assert mod.PREPARE_COMMAND.endswith(' scripts/sitting/prepare_sitting.py')


def test_every_allowed_mcp_tool_is_a_read():
    reads = [tool.rsplit('__', 1)[1] for tool in mod.NIGHTLY_ALLOWED_TOOLS if tool.startswith('mcp__')]

    assert reads, 'the run should be able to read escalations and tasks through MCP'
    assert all(verb.startswith(('get_', 'search')) for verb in reads), reads


def test_record_writes_only_under_data_sitting_and_closes_are_never_applied(tmp_path):
    fake = _fake_claude(tmp_path)

    _main(fake)

    assert prepare_sitting.DEFAULT_PREPARATION.parent == REPO_ROOT / 'data' / 'sitting'
    assert '--preparation' not in mod.NIGHTLY_PROMPT
    assert '--apply-closes' not in mod.NIGHTLY_PROMPT
    assert '--apply-closes' not in fake.only_call()['argv']


# ---------------------------------------------------------------------------
# 3. The account comes from the shared pool; the API key never rides along.
# ---------------------------------------------------------------------------


def test_a_leased_account_supplies_the_child_env(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-must-not-survive')
    monkeypatch.setenv('SITTING_TEST_MARKER', 'kept')
    fake = _fake_claude(tmp_path)
    gate = _live_gate()
    expected = account_pool.subprocess_env(_live_gate())
    assert expected is not None

    assert _main(fake, gate=gate) == mod.EXIT_OK

    call = fake.only_call()
    assert call['env']['CLAUDE_CODE_OAUTH_TOKEN'] == expected['CLAUDE_CODE_OAUTH_TOKEN']
    assert call['env']['SITTING_TEST_MARKER'] == 'kept'
    assert not call['env_has']['ANTHROPIC_API_KEY']
    assert gate.released == [expected['CLAUDE_CODE_OAUTH_TOKEN']], 'the lease is handed straight back'


def test_no_lease_inherits_the_parent_env_and_still_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-must-not-survive')
    monkeypatch.setenv('SITTING_TEST_MARKER', 'kept')
    monkeypatch.setenv('CLAUDE_CODE_OAUTH_TOKEN', 'ambient-login')
    fake = _fake_claude(tmp_path)
    gate = FakeGate([('max-b', True), ('max-h', True)])

    assert _main(fake, gate=gate) == mod.EXIT_OK

    call = fake.only_call()
    assert call['env']['CLAUDE_CODE_OAUTH_TOKEN'] == 'ambient-login'
    assert call['env']['SITTING_TEST_MARKER'] == 'kept'
    assert not call['env_has']['ANTHROPIC_API_KEY']
    assert os.environ['ANTHROPIC_API_KEY'] == 'sk-ant-must-not-survive', 'the parent env is not mutated'


# ---------------------------------------------------------------------------
# 4. Exit codes, and a failure that names what both streams said.
# ---------------------------------------------------------------------------


def test_exit_zero_with_a_json_result_is_ok(tmp_path):
    assert _main(_fake_claude(tmp_path)) == mod.EXIT_OK


def test_a_successful_result_that_quotes_a_banner_is_still_ok(tmp_path):
    """The run reads escalations ABOUT usage caps; a verdict quoting one is not a cap."""
    stdout = json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,
                         'result': f'esc-9-1 is about "{BANNER}"; prepared as live'})

    assert _main(_fake_claude(tmp_path, stdout=stdout)) == mod.EXIT_OK


def test_a_non_zero_exit_fails_with_both_stream_tails_labelled(tmp_path, capsys):
    fake = _fake_claude(tmp_path, stdout='partial-out-7Q', stderr='backend-down-9Z', exit_code=3)

    assert _main(fake) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    line = next(line for line in err.splitlines() if 'partial-out-7Q' in line)
    assert 'stdout=' in line and 'stderr=' in line and 'backend-down-9Z' in line
    assert line.startswith('nightly_prepare: ')


def test_a_timeout_fails_with_both_stream_tails_labelled(tmp_path, capsys):
    fake = _fake_claude(tmp_path, stdout='mid-run-out-4K', stderr='mid-run-err-2J', sleep=30.0)

    assert _main(fake, '--timeout-secs', TIMEOUT_SECS) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    line = next(line for line in err.splitlines() if 'timed out' in line)
    assert 'stdout=' in line and 'mid-run-out-4K' in line
    assert 'stderr=' in line and 'mid-run-err-2J' in line


def test_a_usage_limit_banner_on_exit_zero_fails(tmp_path, capsys):
    fake = _fake_claude(tmp_path, stdout=BANNER, exit_code=0)

    assert _main(fake) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    assert 'usage' in err and BANNER in err


def test_an_error_result_on_exit_zero_fails(tmp_path, capsys):
    stdout = json.dumps({'type': 'result', 'subtype': 'error_max_turns', 'is_error': True, 'result': ''})

    assert _main(_fake_claude(tmp_path, stdout=stdout)) == mod.EXIT_FAILED

    assert 'error_max_turns' in capsys.readouterr().err


def test_a_missing_binary_is_a_configuration_error_and_leases_nothing(tmp_path, capsys):
    gate = _live_gate()

    rc = mod.main(['--claude-bin', str(tmp_path / 'no-such-claude')], gate=gate)

    assert rc == mod.EXIT_CONFIG
    assert gate.leases == 0
    assert 'no-such-claude' in capsys.readouterr().err


def test_the_entry_point_runs_standalone_without_the_test_sys_path(tmp_path):
    """The wrapper runs the file directly, so its bootstrap alone must resolve ``legibility.account_pool``'s imports."""
    script = REPO_ROOT / 'scripts' / 'sitting' / 'nightly_prepare.py'
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}

    proc = subprocess.run(
        [sys.executable, str(script), '--claude-bin', str(tmp_path / 'no-such-claude')],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120,
    )

    assert proc.returncode == mod.EXIT_CONFIG, proc.stderr
    assert 'no-such-claude' in proc.stderr


def test_main_never_raises(tmp_path, capsys):
    fake = _fake_claude(tmp_path)

    rc = mod.main(['--claude-bin', str(fake.bin)], gate=ExplodingGate())

    assert rc == mod.EXIT_FAILED
    assert 'roster unreadable: simulated' in capsys.readouterr().err
    assert fake.calls() == []


def test_exit_codes_are_distinct():
    assert (mod.EXIT_OK, mod.EXIT_FAILED, mod.EXIT_CONFIG) == (0, 1, 2)


# ---------------------------------------------------------------------------
# 5. It writes no store, and the fake is the only process it spawns.
# ---------------------------------------------------------------------------


def _snapshot(*roots: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path): (path.read_bytes(), path.stat().st_mtime_ns)
        for root in roots for path in sorted(root.rglob('*')) if path.is_file()
    }


def test_no_store_is_written_and_only_the_fake_is_spawned(tmp_path, monkeypatch):
    queue = tmp_path / 'project' / 'data' / 'escalations'
    decisions = tmp_path / 'fleet' / 'decisions'
    queue.mkdir(parents=True)
    decisions.mkdir(parents=True)
    (queue / 'esc-9-1.json').write_text(json.dumps({'id': 'esc-9-1', 'status': 'pending', 'level': 2}))
    (decisions / 'dec-1.json').write_text(json.dumps({'decision_id': 'dec-1', 'state': 'open'}))
    before = _snapshot(queue, decisions)
    spawned: list[list[str]] = []
    real_popen = subprocess.Popen

    class RecordingPopen(real_popen):  # type: ignore[misc, valid-type]
        def __init__(self, args, *rest, **kwargs):
            spawned.append(list(args))
            super().__init__(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', RecordingPopen)
    fake = _fake_claude(tmp_path)

    assert _main(fake) == mod.EXIT_OK

    assert _snapshot(queue, decisions) == before
    assert not list(queue.parent.rglob('*.lock')) and not list(decisions.rglob('*.lock'))
    assert [argv[0] for argv in spawned] == [str(fake.bin)]
