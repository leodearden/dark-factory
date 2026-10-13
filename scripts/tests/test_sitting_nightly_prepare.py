"""Tests for scripts/sitting/nightly_prepare.py — the 05:30 headless Fable prepare run (task 5376).

The real ``claude`` is never reached: every test hands ``--claude-bin`` a
recording fake and ``main`` a hand-rolled gate, so no account is leased from
the live pool and no model is billed.
"""
from __future__ import annotations

import json
import os
import re
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
ENV_KEYS = ('CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_API_KEY', 'SITTING_TEST_MARKER', 'SITTING_NIGHTLY_CONFINED')
TIMEOUT_SECS = '3'
"""Outlasts the fake's interpreter start under ``-n auto`` load (0.5s did not, 2 runs in 10), far short of its sleep."""

APPLY_VERBS = (
    'mcp__escalation__resolve_issue',
    'mcp__escalation__stamp_triage',
    'mcp__escalation__promote_to_l2',
    'mcp__escalation__declare_pin',
    'mcp__escalation__amend_escalation',
    'mcp__fused-memory__update_task',
    'mcp__fused-memory__add_dependency',
    'mcp__fused-memory__submit_task',
    'mcp__fused-memory__add_memory',
    'NotebookEdit',
)

_FAKE_CLAUDE_SRC = '''#!{python}
import json, os, sys, time
from pathlib import Path

spec = json.loads(Path({spec!r}).read_text())
argv = sys.argv[1:]
root = Path(spec['root'])
inboxes = sorted((root / 'data' / 'sitting' / 'inbox').iterdir())
assert len(inboxes) == 1, inboxes
inbox = inboxes[0]
for name, content in spec['batches'].items():
    (inbox / name).parent.mkdir(parents=True, exist_ok=True)
    (inbox / name).write_text(content if isinstance(content, str) else json.dumps(content))
store_file = root / 'data' / 'sitting' / 'preparation.json'
if spec['recorded']:
    held = json.loads(store_file.read_text())['preparations'] if store_file.exists() else []
    for name in spec['recorded']:
        held.extend(spec['batches'][name])
    store_file.write_text(json.dumps({{'preparations': held}}))
record = {{
    'inbox': str(inbox),
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
if '--mcp-config' in argv:
    path = argv[argv.index('--mcp-config') + 1]
    record['mcp_config_path'] = path
    record['mcp_config'] = json.loads(Path(path).read_text()) if os.path.exists(path) else None
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
    """A recording stand-in for the CLI: one JSON line per invocation in ``calls_file``.

    It acts like an agent working in ``root``: it writes ``batches`` into the run's inbox and
    records the named ones into the preparation store, before it answers.
    """

    bin: Path
    calls_file: Path
    root: Path

    def calls(self) -> list[dict]:
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]

    def only_call(self) -> dict:
        calls = self.calls()
        assert len(calls) == 1, f'expected exactly one claude invocation, got {len(calls)}'
        return calls[0]


def _prep_payload(number: int) -> dict:
    return {
        'item': ['decision', f'dec-{number}'],
        'question': f'Keep the nightly cap on dec-{number}?',
        'options': [
            {'label': 'A', 'text': 'keep it', 'ramification': 'the cap stays'},
            {'label': 'B', 'text': 'lift it', 'ramification': 'the cap goes'},
        ],
        'recommendation': {'option': 'A', 'evidence_chain': 'the last three nights stayed under it'},
        'on_apply': 'the cap is kept',
        'prepared_at': '2026-10-13T05:40:00+00:00',
        'prepared_by': 'nightly-fable',
    }


def _fake_claude(tmp_path: Path, *, stdout: str = OK_RESULT, stderr: str = '', exit_code: int = 0,
                 sleep: float = 0.0, batches: dict | None = None, recorded: tuple[str, ...] | None = None) -> FakeClaude:
    """By default the fake is a productive agent: one batch of one preparation, written and recorded."""
    if batches is None:
        batches, recorded = {'batch-01.json': [_prep_payload(1)]}, ('batch-01.json',)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    calls_file = tmp_path / 'claude-calls.jsonl'
    spec_file = tmp_path / 'claude-spec.json'
    spec_file.write_text(json.dumps({
        'record': str(calls_file), 'stdout': stdout, 'stderr': stderr,
        'exit_code': exit_code, 'sleep': sleep, 'env_keys': list(ENV_KEYS),
        'root': str(tmp_path), 'batches': batches, 'recorded': list(recorded or ()),
    }))
    fake = bin_dir / 'claude'
    fake.write_text(_FAKE_CLAUDE_SRC.format(python=sys.executable, spec=str(spec_file)))
    fake.chmod(0o755)
    return FakeClaude(fake, calls_file, tmp_path)


@dataclass
class FakeGate:
    """The two gate members ``nightly_prepare`` calls to lease an account, over ``(name, capped)`` accounts."""

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


def _run(fake: FakeClaude, *, gate=None, **overrides) -> int:
    """``mod.run`` over the fake's root; *overrides* replace ``Settings`` fields."""
    settings = mod.Settings(**{
        'claude_bin': str(fake.bin), 'model': mod.DEFAULT_MODEL, 'budget_usd': mod.DEFAULT_BUDGET_USD,
        'timeout_secs': mod.DEFAULT_TIMEOUT_SECS, 'max_turns': mod.DEFAULT_MAX_TURNS,
        'mcp_json': mod.DEFAULT_MCP_JSON, 'root': fake.root, **overrides,
    })
    return mod.run(settings, gate=gate if gate is not None else _live_gate())


def _flag_values(argv: list[str], flag: str) -> list[str]:
    values = argv[argv.index(flag) + 1:]
    return values[:next((i for i, value in enumerate(values) if value.startswith('--')), len(values))]


def _sysprompt_path(call: dict) -> Path:
    return Path(call['sysprompt_path'])


# ---------------------------------------------------------------------------
# 1. The argv is the shared builder's, the prompt goes on stdin, temp files go.
# ---------------------------------------------------------------------------


def test_argv_is_exactly_the_shared_builders_with_the_defaults(tmp_path):
    fake = _fake_claude(tmp_path)

    assert _run(fake) == mod.EXIT_OK

    call = fake.only_call()
    expected, temp_files = build_claude_argv(
        model=mod.DEFAULT_MODEL,
        max_budget_usd=mod.DEFAULT_BUDGET_USD,
        system_prompt=call['sysprompt'],
        max_turns=mod.DEFAULT_MAX_TURNS,
        permission_mode='dontAsk',
        allowed_tools=list(mod.nightly_allowed_tools(Path(call['inbox']))),
        disallowed_tools=list(mod.NIGHTLY_DENIED_TOOLS),
        mcp_config=call['mcp_config'],
        output_schema=None,
        effort=None,
        resume_session_id=None,
        session_id=None,
        strict_mcp_config=True,
        setting_sources=[],
    )
    for path in temp_files:
        Path(path).unlink(missing_ok=True)
    expected[0] = str(fake.bin)
    expected[expected.index('--system-prompt-file') + 1] = call['sysprompt_path']
    expected[expected.index('--mcp-config') + 1] = call['mcp_config_path']
    assert call['argv'] == expected
    assert mod.DEFAULT_MODEL == 'fable', 'the task mandates the prepare-sitting mode run nightly on Fable'
    assert call['sysprompt'], 'the builder always writes a system prompt; it must not be empty'


def test_model_budget_and_turns_seams_reach_the_argv(tmp_path):
    fake = _fake_claude(tmp_path)

    assert _run(fake, model='opus', budget_usd=2.5, max_turns=17) == mod.EXIT_OK

    argv = fake.only_call()['argv']
    assert argv[argv.index('--model') + 1] == 'opus'
    assert argv[argv.index('--max-budget-usd') + 1] == '2.5'
    assert argv[argv.index('--max-turns') + 1] == '17'


def test_the_prompt_goes_on_stdin_not_argv(tmp_path):
    fake = _fake_claude(tmp_path)

    _run(fake)

    call = fake.only_call()
    assert call['stdin'].strip()
    assert not any(call['stdin'] in arg for arg in call['argv']), 'the prompt must not ride the argv'


def test_the_child_runs_in_the_root_it_was_given(tmp_path):
    """``claude -p`` scopes its tools to its cwd, and the prepare command's path is root-relative."""
    fake = _fake_claude(tmp_path)

    _run(fake)

    assert Path(fake.only_call()['cwd']).resolve() == tmp_path.resolve()


def test_main_works_in_the_root_it_is_given_and_forwards_the_cli_flags(tmp_path):
    fake = _fake_claude(tmp_path)

    rc = mod.main(['--claude-bin', str(fake.bin), '--model', 'opus', '--max-turns', '17'], gate=_live_gate(), root=tmp_path)

    call = fake.only_call()
    assert rc == mod.EXIT_OK
    assert Path(call['cwd']).resolve() == tmp_path.resolve()
    assert call['argv'][call['argv'].index('--model') + 1] == 'opus'
    assert call['argv'][call['argv'].index('--max-turns') + 1] == '17'


def test_each_run_gets_a_fresh_inbox_under_the_root_and_the_prompt_names_it(tmp_path):
    fake = _fake_claude(tmp_path)

    assert _run(fake) == mod.EXIT_OK

    call = fake.only_call()
    inbox = Path(call['inbox'])
    assert inbox.parent == tmp_path.resolve() / 'data' / 'sitting' / 'inbox'
    assert re.fullmatch(r'\d{8}T\d{6}Z', inbox.name)
    assert inbox.is_dir()
    assert f'{inbox}/' in call['stdin']
    assert f'--from {inbox.relative_to(tmp_path.resolve())}/' in call['stdin']


def test_the_child_reads_no_settings_file_and_may_write_only_its_inbox(tmp_path):
    fake = _fake_claude(tmp_path)

    assert _run(fake) == mod.EXIT_OK

    argv = fake.only_call()['argv']
    assert argv[argv.index('--setting-sources') + 1] == ''
    allowed = _flag_values(argv, '--allowed-tools')
    inbox = fake.only_call()['inbox']
    assert [tool for tool in allowed if tool.startswith('Edit')] == [f'Edit(/{inbox}/**)']
    assert f'Bash({mod.PREPARE_COMMAND} record --from:*)' in allowed
    assert not any('record --from -' in tool for tool in allowed)
    denied = _flag_values(argv, '--disallowed-tools')
    assert 'Edit' not in denied and 'Write' not in denied
    assert 'NotebookEdit' in denied


@pytest.mark.parametrize(
    ('spec', 'extra'),
    [
        ({'exit_code': 0}, {}),
        ({'exit_code': 3, 'stderr': 'boom'}, {}),
        ({'sleep': 30.0}, {'timeout_secs': float(TIMEOUT_SECS)}),
    ],
    ids=['success', 'failure', 'timeout'],
)
def test_builder_temp_files_are_unlinked_after_the_run(tmp_path, spec, extra):
    fake = _fake_claude(tmp_path, **spec)

    _run(fake, **extra)

    call = fake.only_call()
    assert call['sysprompt'] is not None, 'the system prompt file must exist while the CLI runs'
    assert call['mcp_config'] is not None, 'the MCP config file must exist while the CLI runs'
    assert not _sysprompt_path(call).exists(), 'the builder temp file must be unlinked after the run'
    assert not Path(call['mcp_config_path']).exists(), 'the MCP config temp file must be unlinked after the run'


# ---------------------------------------------------------------------------
# 2. Read-only by construction.
# ---------------------------------------------------------------------------


def test_allow_and_deny_lists_are_disjoint(tmp_path):
    assert not set(mod.nightly_allowed_tools(tmp_path)) & set(mod.NIGHTLY_DENIED_TOOLS)


@pytest.mark.parametrize('verb', APPLY_VERBS)
def test_every_store_writing_apply_verb_is_denied(verb):
    assert verb in mod.NIGHTLY_DENIED_TOOLS


def test_the_only_bash_allowed_is_git_show_git_log_and_the_two_prepare_subcommands(tmp_path):
    fake = _fake_claude(tmp_path)

    _run(fake)

    allowed = _flag_values(fake.only_call()['argv'], '--allowed-tools')
    bash = [tool for tool in allowed if tool.startswith('Bash')]
    assert sorted(bash) == sorted([
        'Bash(git show:*)',
        'Bash(git log:*)',
        f'Bash({mod.PREPARE_COMMAND} brief --json:*)',
        f'Bash({mod.PREPARE_COMMAND} record --from:*)',
    ])
    assert mod.PREPARE_COMMAND.endswith(' scripts/sitting/prepare_sitting.py')


def test_every_allowed_mcp_tool_is_a_read():
    reads = [tool.rsplit('__', 1)[1] for tool in mod.NIGHTLY_READ_TOOLS if tool.startswith('mcp__')]

    assert reads, 'the run should be able to read escalations and tasks through MCP'
    assert all(verb.startswith(('get_', 'search')) for verb in reads), reads


def test_the_mcp_servers_are_exactly_the_checkouts_escalation_and_fused_memory_blocks(tmp_path):
    """Explicit and strict: the reads never hang on the ambient .mcp.json being approved headless, and nothing else starts."""
    fake = _fake_claude(tmp_path)

    assert _run(fake) == mod.EXIT_OK

    call = fake.only_call()
    ambient = json.loads((REPO_ROOT / '.mcp.json').read_text())['mcpServers']
    assert call['mcp_config'] == {'mcpServers': {name: ambient[name] for name in ('escalation', 'fused-memory')}}
    argv = call['argv']
    assert argv[argv.index('--mcp-config') + 2] == '--strict-mcp-config'


def test_every_allowed_or_denied_mcp_tool_names_a_configured_server():
    tools = [tool for tool in (*mod.NIGHTLY_READ_TOOLS, *mod.NIGHTLY_DENIED_TOOLS) if tool.startswith('mcp__')]

    assert {tool.split('__')[1] for tool in tools} == set(mod.NIGHTLY_MCP_SERVERS)


@pytest.mark.parametrize(
    ('contents', 'named'),
    [
        (None, 'No such file'),
        ('{not json', 'JSONDecodeError'),
        (json.dumps({'mcpServers': {'escalation': {'type': 'http', 'url': 'http://127.0.0.1:1/mcp'}}}), 'fused-memory'),
        (json.dumps({'servers': {}}), 'mcpServers'),
    ],
    ids=['absent', 'not-json', 'a-server-missing', 'no-mcpServers'],
)
def test_an_unusable_mcp_config_is_a_configuration_error_and_leases_nothing(tmp_path, capsys, contents, named):
    mcp_json = tmp_path / 'mcp.json'
    if contents is not None:
        mcp_json.write_text(contents)
    fake = _fake_claude(tmp_path)
    gate = _live_gate()

    rc = _run(fake, mcp_json=mcp_json, gate=gate)

    assert rc == mod.EXIT_CONFIG
    assert (gate.leases, fake.calls()) == (0, [])
    err = capsys.readouterr().err
    assert str(mcp_json) in err and named in err


def test_the_mcp_json_seam_reaches_the_run(tmp_path):
    servers = {name: {'type': 'http', 'url': f'http://127.0.0.1:9/{name}'} for name in ('escalation', 'fused-memory')}
    mcp_json = tmp_path / 'mcp.json'
    mcp_json.write_text(json.dumps({'mcpServers': {**servers, 'playwright': {'command': 'npx'}}}))
    fake = _fake_claude(tmp_path)

    assert _run(fake, mcp_json=mcp_json) == mod.EXIT_OK

    assert fake.only_call()['mcp_config'] == {'mcpServers': servers}


def test_the_child_runs_the_prepare_script_confined_to_data_sitting(tmp_path):
    """The confinement itself is prepare_sitting's; test_prepare_sitting_cli.py::TestNightlyConfinement covers it."""
    fake = _fake_claude(tmp_path)

    _run(fake)

    call = fake.only_call()
    assert call['env']['SITTING_NIGHTLY_CONFINED'] == call['inbox'], "the env names this run's inbox"
    assert prepare_sitting.DEFAULT_PREPARATION.parent == REPO_ROOT / 'data' / 'sitting'


# ---------------------------------------------------------------------------
# 3. The account comes from the shared pool; the API key never rides along.
# ---------------------------------------------------------------------------


def test_a_leased_account_supplies_the_child_env(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-must-not-survive')
    monkeypatch.setenv('SITTING_TEST_MARKER', 'kept')
    fake = _fake_claude(tmp_path)
    gate = _live_gate()

    assert _run(fake, gate=gate) == mod.EXIT_OK

    call = fake.only_call()
    assert call['env']['CLAUDE_CODE_OAUTH_TOKEN'] == 'tok-max-h', 'leased from the END of the roster'
    assert call['env']['SITTING_TEST_MARKER'] == 'kept'
    assert call['env']['SITTING_NIGHTLY_CONFINED']
    assert not call['env_has']['ANTHROPIC_API_KEY']
    assert gate.released == ['tok-max-h'], 'the lease is handed straight back'


def test_a_real_pool_lease_is_handed_back_before_the_run(tmp_path, monkeypatch, pool_roster):
    """Nothing in the child can settle a slot, so a kept PROBE_IN_FLIGHT claim
    would hold that account out of the pool for the rest of the night."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-must-not-survive')
    accounts_file, env_file = pool_roster('max-p', 'max-q')
    gate = account_pool.build_pool(accounts_file=accounts_file, env_file=env_file)
    fake = _fake_claude(tmp_path)

    assert _run(fake, gate=gate) == mod.EXIT_OK

    assert fake.only_call()['env']['CLAUDE_CODE_OAUTH_TOKEN'] == pool_roster.token('max-q')
    assert not fake.only_call()['env_has']['ANTHROPIC_API_KEY']
    released = gate.try_lease(reverse=True)
    assert released is not None and released.name == 'max-q', (
        'the account the run leased must be leasable again once the run has its env'
    )


def test_no_lease_inherits_the_parent_env_and_still_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-must-not-survive')
    monkeypatch.setenv('SITTING_TEST_MARKER', 'kept')
    monkeypatch.setenv('CLAUDE_CODE_OAUTH_TOKEN', 'ambient-login')
    fake = _fake_claude(tmp_path)
    gate = FakeGate([('max-b', True), ('max-h', True)])

    assert _run(fake, gate=gate) == mod.EXIT_OK

    call = fake.only_call()
    assert call['env']['CLAUDE_CODE_OAUTH_TOKEN'] == 'ambient-login'
    assert call['env']['SITTING_TEST_MARKER'] == 'kept'
    assert not call['env_has']['ANTHROPIC_API_KEY']
    assert os.environ['ANTHROPIC_API_KEY'] == 'sk-ant-must-not-survive', 'the parent env is not mutated'


# ---------------------------------------------------------------------------
# 4. Exit codes, and a failure that names what both streams said.
# ---------------------------------------------------------------------------


def test_exit_zero_with_a_json_result_is_ok(tmp_path):
    assert _run(_fake_claude(tmp_path)) == mod.EXIT_OK


def test_a_successful_result_that_quotes_a_banner_is_still_ok(tmp_path):
    """The run reads escalations ABOUT usage caps; a verdict quoting one is not a cap."""
    stdout = json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,
                         'result': f'esc-9-1 is about "{BANNER}"; prepared as live'})

    assert _run(_fake_claude(tmp_path, stdout=stdout)) == mod.EXIT_OK


def test_a_non_zero_exit_fails_with_both_stream_tails_labelled(tmp_path, capsys):
    fake = _fake_claude(tmp_path, stdout='partial-out-7Q', stderr='backend-down-9Z', exit_code=3)

    assert _run(fake) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    line = next(line for line in err.splitlines() if 'partial-out-7Q' in line)
    assert 'stdout=' in line and 'stderr=' in line and 'backend-down-9Z' in line
    assert line.startswith('nightly_prepare: ')


def test_a_timeout_fails_with_both_stream_tails_labelled(tmp_path, capsys):
    fake = _fake_claude(tmp_path, stdout='mid-run-out-4K', stderr='mid-run-err-2J', sleep=30.0)

    assert _run(fake, timeout_secs=float(TIMEOUT_SECS)) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    line = next(line for line in err.splitlines() if 'timed out' in line)
    assert 'stdout=' in line and 'mid-run-out-4K' in line
    assert 'stderr=' in line and 'mid-run-err-2J' in line


def test_a_usage_limit_banner_on_exit_zero_fails(tmp_path, capsys):
    fake = _fake_claude(tmp_path, stdout=BANNER, exit_code=0)

    assert _run(fake) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    assert 'usage' in err and BANNER in err


def test_an_error_result_on_exit_zero_fails(tmp_path, capsys):
    stdout = json.dumps({'type': 'result', 'subtype': 'error_max_turns', 'is_error': True, 'result': ''})

    assert _run(_fake_claude(tmp_path, stdout=stdout)) == mod.EXIT_FAILED

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

    rc = mod.main(['--claude-bin', str(fake.bin)], gate=ExplodingGate(), root=tmp_path)

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

    assert _run(fake) == mod.EXIT_OK

    assert _snapshot(queue, decisions) == before
    assert not list(queue.parent.rglob('*.lock')) and not list(decisions.rglob('*.lock'))
    assert [argv[0] for argv in spawned] == [str(fake.bin)]


# ---------------------------------------------------------------------------
# 6. The outcome is what the inbox offered and the store holds, not the agent's say-so.
# ---------------------------------------------------------------------------


def test_two_recorded_preparations_are_ok_and_the_done_line_counts_them(tmp_path, capsys):
    batches = {'batch-01.json': [_prep_payload(1)], 'batch-02.json': [_prep_payload(2)]}
    stdout = json.dumps({'type': 'result', 'is_error': False, 'result': 'recorded 2 of 2', 'num_turns': 9,
                         'total_cost_usd': 1.5, 'permission_denials': [{'tool_name': 'Bash'}] * 3})
    fake = _fake_claude(tmp_path, stdout=stdout, batches=batches, recorded=tuple(batches))

    assert _run(fake) == mod.EXIT_OK

    done = next(line for line in capsys.readouterr().out.splitlines() if 'done' in line)
    assert 'recorded=2 of 2' in done and 'undecodable=0' in done and 'permission_denials=3' in done and 'turns=9' in done


def test_a_success_that_recorded_nothing_fails(tmp_path, capsys):
    fake = _fake_claude(tmp_path, batches={})

    assert _run(fake) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    assert 'recorded nothing' in err and '0 item key(s) offered' in err


def test_an_inbox_file_the_store_does_not_hold_fails(tmp_path, capsys):
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)]}, recorded=())

    assert _run(fake) == mod.EXIT_FAILED

    assert '1 item key(s) offered' in capsys.readouterr().err


def test_the_store_holding_a_different_preparation_for_the_key_is_not_a_record(tmp_path, capsys):
    other = {**_prep_payload(1), 'question': 'Lift the nightly cap on dec-1?'}
    store = tmp_path / 'data' / 'sitting' / 'preparation.json'
    store.parent.mkdir(parents=True)
    store.write_text(json.dumps({'preparations': [other]}))
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)]}, recorded=())

    assert _run(fake) == mod.EXIT_FAILED

    assert 'recorded nothing' in capsys.readouterr().err


def test_a_key_offered_twice_is_recorded_when_the_store_holds_either_offer(tmp_path, capsys):
    other = {**_prep_payload(1), 'question': 'Lift the nightly cap on dec-1?'}
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)], 'batch-02.json': [other]},
                        recorded=('batch-02.json',))

    assert _run(fake) == mod.EXIT_OK

    assert 'recorded=1 of 1' in capsys.readouterr().out


def test_undecodable_inbox_content_is_counted_and_never_raises(tmp_path, capsys):
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [{'item': 'not-a-key'}], 'torn.json': '{"item": ['})

    assert _run(fake) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    assert '0 item key(s) offered' in err and 'undecodable=2' in err


def test_a_file_too_deeply_nested_to_parse_is_undecodable_not_fatal(tmp_path, capsys):
    fake = _fake_claude(tmp_path, batches={'deep.json': '[' * 200_000, 'scalars.json': [5, 'x', None]})

    assert _run(fake) == mod.EXIT_FAILED

    assert 'undecodable=4' in capsys.readouterr().err


def test_an_offer_identical_to_an_entry_held_before_the_run_is_not_a_record(tmp_path, capsys):
    store = tmp_path / 'data' / 'sitting' / 'preparation.json'
    store.parent.mkdir(parents=True)
    store.write_text(json.dumps({'preparations': [_prep_payload(1)]}))
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)]}, recorded=('batch-01.json',))

    assert _run(fake) == mod.EXIT_FAILED

    assert 'recorded nothing' in capsys.readouterr().err


def test_a_batch_in_a_subdirectory_or_not_json_is_not_offered(tmp_path, capsys):
    fake = _fake_claude(tmp_path, batches={'notes.txt': [_prep_payload(1)], 'sub/batch-01.json': [_prep_payload(2)]})

    assert _run(fake) == mod.EXIT_FAILED

    assert '0 item key(s) offered' in capsys.readouterr().err


def test_a_decodable_entry_beside_an_undecodable_one_still_counts(tmp_path, capsys):
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)], 'draft.json': [{'item': 'not-a-key'}]},
                        recorded=('batch-01.json',))

    assert _run(fake) == mod.EXIT_OK

    assert 'recorded=1 of 1' in capsys.readouterr().out


def test_a_corrupt_store_counts_zero_recorded(tmp_path, capsys):
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)]}, recorded=())
    store = tmp_path / 'data' / 'sitting' / 'preparation.json'
    store.parent.mkdir(parents=True)
    store.write_text('{not json')

    assert _run(fake) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    assert 'unreadable' in err and 'recorded nothing' in err


def test_a_timeout_after_a_partial_record_still_counts_what_was_recorded(tmp_path, capsys):
    batches = {'batch-01.json': [_prep_payload(1)], 'batch-02.json': [_prep_payload(2)]}
    fake = _fake_claude(tmp_path, batches=batches, recorded=('batch-01.json',), sleep=30.0)

    assert _run(fake, timeout_secs=float(TIMEOUT_SECS)) == mod.EXIT_FAILED

    line = next(line for line in capsys.readouterr().err.splitlines() if 'timed out' in line)
    assert 'recorded=1' in line


def test_a_non_zero_exit_states_how_much_was_recorded(tmp_path, capsys):
    fake = _fake_claude(tmp_path, exit_code=3, stderr='boom')

    assert _run(fake) == mod.EXIT_FAILED

    line = next(line for line in capsys.readouterr().err.splitlines() if 'exited 3' in line)
    assert 'recorded=1' in line
