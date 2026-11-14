"""Tests for scripts/sitting/nightly_prepare.py — the 05:30 headless Fable prepare run (task 5376).

The real ``claude`` is never reached: every test hands ``--claude-bin`` a
recording fake, or sets ``CLAUDE_BINARY`` to one on a PATH stripped of
``claude``, and hands ``main`` a real gate over a hermetic roster
(``conftest.py::pool_roster``), so no account is leased from the live pool and
no model is billed.
"""
from __future__ import annotations

import errno
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from legibility import account_pool
from orchestrator.agents import landlock, sandbox, sandbox_dispatch
from shared.cap_markers import REAL_CLI_CAP_HIT_MESSAGES
from shared.cli_invoke import CAP_HIT_RESUME_PROMPT, build_claude_argv, resolve_claude_binary
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from sitting import nightly_prepare as mod
from sitting import prepare_sitting

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]

OK_RESULT = json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'recorded 3'})
BANNER = "You've hit your usage limit · resets 5am (Europe/London)"
ENV_KEYS = (
    'CLAUDE_CODE_OAUTH_TOKEN', 'ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_AUTH_TOKEN',
    'SITTING_TEST_MARKER', 'SITTING_NIGHTLY_CONFINED', 'CLAUDE_CONFIG_DIR',
)
TIMEOUT_SECS = '3'
"""Outlasts the fake's interpreter start under ``-n auto`` load (0.5s did not, 2 runs in 10), far short of its sleep."""
FAILOVER_TEST_TIMEOUT_SECS = 60
"""A pytest bound for the tests that sit out the shared runner's 5s cap cooldown, with room for ``-n auto`` load."""
SYSTEM_PROMPT_FILE_FLAG = '--system-prompt-file'
MCP_CONFIG_FLAG = '--mcp-config'
BUILDER_TEMP_FILE_FLAGS = (SYSTEM_PROMPT_FILE_FLAG, MCP_CONFIG_FLAG)
"""The argv flags whose values are the paths ``build_claude_argv`` returns as temp files."""

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
spec.update(spec['by_token'].get(os.environ.get('CLAUDE_CODE_OAUTH_TOKEN', ''), {{}}))
time.sleep(spec['stall'])
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
credentials = Path(os.environ.get('CLAUDE_CONFIG_DIR', '/nonexistent'), '.credentials.json')
if credentials.exists():
    record['credentials'] = credentials.read_text()
    record['credentials_mode'] = credentials.stat().st_mode & 0o777
if {system_prompt_file_flag!r} in argv:
    path = argv[argv.index({system_prompt_file_flag!r}) + 1]
    record['sysprompt_path'] = path
    record['sysprompt'] = Path(path).read_text() if os.path.exists(path) else None
if {mcp_config_flag!r} in argv:
    path = argv[argv.index({mcp_config_flag!r}) + 1]
    record['mcp_config_path'] = path
    record['mcp_config'] = json.loads(Path(path).read_text()) if os.path.exists(path) else None
with open(spec['record'], 'a') as sink:
    sink.write(json.dumps(record) + '\\n')
if 'transcript' in spec:
    transcript = Path(os.environ['CLAUDE_CONFIG_DIR'], 'projects', 'fake', spec['transcript']['session_id'] + '.jsonl')
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(''.join(json.dumps(entry) + '\\n' for entry in spec['transcript']['records']))
sys.stdout.write(spec['stdout'])
sys.stdout.flush()
sys.stderr.write(spec['stderr'])
sys.stderr.flush()
time.sleep(spec['sleep'])
sys.exit(spec['exit_code'])
'''


@pytest.fixture(autouse=True)
def _restore_sandbox_backend():
    """The run sets the process-global backend; put it back after each test."""
    saved = sandbox_dispatch.get_backend()
    sandbox_dispatch.reset_escalation_dedupe()
    yield
    sandbox_dispatch.set_backend(saved)
    sandbox_dispatch.reset_escalation_dedupe()


@pytest.fixture
def host_home(tmp_path, monkeypatch):
    """A HOME of the test's own, so the run's ``~/.cache/uv`` grant is observable and no real HOME is touched."""
    home = tmp_path / 'home'
    (home / '.claude').mkdir(parents=True)
    monkeypatch.setenv('HOME', str(home))
    return home


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
                 sleep: float = 0.0, stall: float = 0.0, batches: dict | None = None,
                 recorded: tuple[str, ...] | None = None, by_token: dict[str, dict] | None = None) -> FakeClaude:
    """By default the fake is a productive agent: one batch of one preparation, written and recorded.

    ``stall`` is how long the fake hangs before doing anything, i.e. a CLI that a loaded host has not
    yet started when the timeout fires; ``sleep`` is how long it hangs after its work.

    ``by_token`` maps a leased ``CLAUDE_CODE_OAUTH_TOKEN`` to overrides of any of those fields for the
    calls made on that seat, plus an optional ``transcript`` (``{'session_id', 'records'}``) written
    where the CLI keeps a session, ``$CLAUDE_CONFIG_DIR/projects/<slug>/<session_id>.jsonl``.
    """
    if batches is None:
        batches, recorded = {'batch-01.json': [_prep_payload(1)]}, ('batch-01.json',)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir(exist_ok=True)
    calls_file = tmp_path / 'claude-calls.jsonl'
    spec_file = tmp_path / 'claude-spec.json'
    spec_file.write_text(json.dumps({
        'record': str(calls_file), 'stdout': stdout, 'stderr': stderr,
        'exit_code': exit_code, 'sleep': sleep, 'stall': stall, 'env_keys': list(ENV_KEYS),
        'root': str(tmp_path), 'batches': batches, 'recorded': list(recorded or ()),
        'by_token': by_token or {},
    }))
    fake = bin_dir / 'claude'
    fake.write_text(_FAKE_CLAUDE_SRC.format(
        python=sys.executable, spec=str(spec_file),
        system_prompt_file_flag=SYSTEM_PROMPT_FILE_FLAG, mcp_config_flag=MCP_CONFIG_FLAG,
    ))
    fake.chmod(0o755)
    return FakeClaude(fake, calls_file, tmp_path)


def _cap(sid: str | None = None) -> dict:
    """A seat at its usage limit, as the real CLI answers under ``--output-format json``: no work done.

    The banner's "resets in 3 hours" parses to a FUTURE reset, so the gate cannot reopen the seat mid-test.
    """
    return {
        'stdout': json.dumps({'type': 'result', 'subtype': 'success', 'is_error': True,
                              'result': REAL_CLI_CAP_HIT_MESSAGES[2], 'session_id': sid,
                              'total_cost_usd': 0.0, 'num_turns': 1}),
        'exit_code': 1, 'batches': {}, 'recorded': (),
    }


def _auth_401() -> dict:
    """A seat whose credentials the API rejects, in the real CLI's JSON shape (CLI 2.1.287, task 5947)."""
    return {
        'stdout': json.dumps({'type': 'result', 'subtype': 'success', 'is_error': True, 'api_error_status': 401,
                              'result': 'Failed to authenticate. API Error: 401 OAuth access token is invalid.',
                              'total_cost_usd': 0.0, 'num_turns': 1}),
        'exit_code': 1, 'batches': {}, 'recorded': (),
    }


def _costed_failure(subtype: str, *, result: str = '') -> dict:
    """An ordinary failed run. It costs money and took turns, because the shared runner reads a
    zero-cost failure that took under 5s and at most one turn as an unrecognised cap, and fails over."""
    return {
        'stdout': json.dumps({'type': 'result', 'subtype': subtype, 'is_error': True, 'result': result,
                              'total_cost_usd': 0.5, 'num_turns': 3}),
        'exit_code': 1,
    }


def _progress_transcript(sid: str) -> dict:
    """A session that did work before it was capped: two assistant turns, one of them a tool call."""
    return {'session_id': sid, 'records': [
        {'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'Listing the open items.'}]}},
        {'type': 'assistant', 'message': {'content': [
            {'type': 'tool_use', 'id': 'toolu_01', 'name': 'Bash', 'input': {'command': 'brief --json'}},
        ]}},
    ]}


SEAT_B, SEAT_H = 'max-b', 'max-h'
TOKEN_B, TOKEN_H = f'tok-{SEAT_B}', f'tok-{SEAT_H}'
"""The roster's seats in failover order, and the tokens ``pool_roster`` resolves them to."""


@pytest.fixture
def pool(pool_roster):
    """A real fleet gate over a hermetic two-seat roster; never the repo's ``.env``."""
    accounts_file, env_file = pool_roster(SEAT_B, SEAT_H)
    return account_pool.build_pool(accounts_file=accounts_file, env_file=env_file)


class UnreadableRoster:
    """A gate whose roster cannot be read: its first read raises."""

    @property
    def account_count(self) -> int:
        raise RuntimeError('roster unreadable: simulated')


def _run(fake: FakeClaude, gate, **overrides) -> int:
    """``mod.run`` over the fake's root on *gate*; *overrides* replace ``Settings`` fields."""
    settings = mod.Settings(**{
        'claude_bin': str(fake.bin), 'model': mod.DEFAULT_MODEL, 'budget_usd': mod.DEFAULT_BUDGET_USD,
        'timeout_secs': mod.DEFAULT_TIMEOUT_SECS, 'max_turns': mod.DEFAULT_MAX_TURNS,
        'mcp_json': mod.DEFAULT_MCP_JSON, 'root': fake.root, 'sandbox_backend': 'none', **overrides,
    })
    return mod.run(settings, gate=gate)


def _tokens_called(fake: FakeClaude) -> list[str]:
    return [call['env']['CLAUDE_CODE_OAUTH_TOKEN'] for call in fake.calls()]


def _flag_values(argv: list[str], flag: str) -> list[str]:
    values = argv[argv.index(flag) + 1:]
    return values[:next((i for i, value in enumerate(values) if value.startswith('--')), len(values))]


def _builder_temp_files(argv: tuple[str, ...]) -> tuple[Path, ...]:
    """The path after each ``BUILDER_TEMP_FILE_FLAGS`` flag, in argv order; an absent or valueless flag gives none."""
    return tuple(Path(argv[i + 1]) for i, arg in enumerate(argv[:-1]) if arg in BUILDER_TEMP_FILE_FLAGS)


@dataclass(frozen=True)
class Spawn:
    """A process the run started, as the parent saw it once ``Popen`` had started it."""

    argv: tuple[str, ...]
    present_once_started: frozenset[Path]
    """The builder temp files that existed once the process had been started."""

    def temp_files(self) -> tuple[Path, ...]:
        """The builder temp files this process was handed, per ``_builder_temp_files``."""
        return _builder_temp_files(self.argv)


def _record_spawns(monkeypatch: pytest.MonkeyPatch) -> list[Spawn]:
    """Record every process the run starts, from this side, before ``communicate`` starts its timeout clock.

    Nothing here waits on the child, so the record holds however far the child got before a kill.
    """
    spawns: list[Spawn] = []

    class RecordingPopen(subprocess.Popen):
        def __init__(self, args, *rest, **kwargs):
            argv = tuple(os.fsdecode(arg) for arg in ([args] if isinstance(args, (str, bytes, os.PathLike)) else args))
            super().__init__(args, *rest, **kwargs)
            spawns.append(Spawn(argv, frozenset(path for path in _builder_temp_files(argv) if path.is_file())))

    monkeypatch.setattr(subprocess, 'Popen', RecordingPopen)
    return spawns


def _strip_claude_from_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave no `claude` on PATH, so a bare default can never reach the real CLI."""
    empty_bin = tmp_path / 'empty-bin'
    empty_bin.mkdir()
    monkeypatch.setenv('PATH', f'{empty_bin}{os.pathsep}/usr/bin')
    assert resolve_claude_binary() is None, 'the premise: no claude is reachable'


# ---------------------------------------------------------------------------
# 1. The argv is the shared builder's, the prompt goes on stdin, temp files go.
# ---------------------------------------------------------------------------


def test_argv_is_exactly_the_shared_builders_with_the_defaults(tmp_path, pool):
    fake = _fake_claude(tmp_path)

    assert _run(fake, pool) == mod.EXIT_OK

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
    expected[expected.index(SYSTEM_PROMPT_FILE_FLAG) + 1] = call['sysprompt_path']
    expected[expected.index(MCP_CONFIG_FLAG) + 1] = call['mcp_config_path']
    assert call['argv'] == expected
    assert mod.DEFAULT_MODEL == 'fable', 'the task mandates the prepare-sitting mode run nightly on Fable'
    assert call['sysprompt'], 'the builder always writes a system prompt; it must not be empty'


def test_model_budget_and_turns_seams_reach_the_argv(tmp_path, pool):
    fake = _fake_claude(tmp_path)

    assert _run(fake, pool, model='opus', budget_usd=2.5, max_turns=17) == mod.EXIT_OK

    argv = fake.only_call()['argv']
    assert argv[argv.index('--model') + 1] == 'opus'
    assert argv[argv.index('--max-budget-usd') + 1] == '2.5'
    assert argv[argv.index('--max-turns') + 1] == '17'


def test_the_prompt_goes_on_stdin_not_argv(tmp_path, pool):
    fake = _fake_claude(tmp_path)

    _run(fake, pool)

    call = fake.only_call()
    assert call['stdin'].strip()
    assert not any(call['stdin'] in arg for arg in call['argv']), 'the prompt must not ride the argv'


def test_the_child_runs_in_the_root_it_was_given(tmp_path, pool):
    """``claude -p`` scopes its tools to its cwd, and the prepare command's path is root-relative."""
    fake = _fake_claude(tmp_path)

    _run(fake, pool)

    assert Path(fake.only_call()['cwd']).resolve() == tmp_path.resolve()


def test_main_works_in_the_root_it_is_given_and_forwards_the_cli_flags(tmp_path, pool):
    fake = _fake_claude(tmp_path)

    rc = mod.main(['--claude-bin', str(fake.bin), '--model', 'opus', '--max-turns', '17'], gate=pool, root=tmp_path,
                  sandbox_backend='none')

    call = fake.only_call()
    assert rc == mod.EXIT_OK
    assert Path(call['cwd']).resolve() == tmp_path.resolve()
    assert call['argv'][call['argv'].index('--model') + 1] == 'opus'
    assert call['argv'][call['argv'].index('--max-turns') + 1] == '17'


def test_each_run_gets_a_fresh_inbox_under_the_root_and_the_prompt_names_it(tmp_path, pool):
    fake = _fake_claude(tmp_path)

    assert _run(fake, pool) == mod.EXIT_OK

    call = fake.only_call()
    inbox = Path(call['inbox'])
    assert inbox.parent == tmp_path.resolve() / 'data' / 'sitting' / 'inbox'
    assert re.fullmatch(r'\d{8}T\d{6}Z', inbox.name)
    assert inbox.is_dir()
    assert f'{inbox}/' in call['stdin']
    assert f'--from {inbox.relative_to(tmp_path.resolve())}/' in call['stdin']


def test_the_child_reads_no_settings_file_and_may_write_only_its_inbox(tmp_path, pool):
    fake = _fake_claude(tmp_path)

    assert _run(fake, pool) == mod.EXIT_OK

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
    ('spec', 'extra', 'witness'),
    [
        ({'exit_code': 0}, {}, 'nightly_prepare: done'),
        ({**_costed_failure('error_during_execution'), 'stderr': 'boom'}, {}, 'claude failed'),
        ({'stall': 30.0}, {'timeout_secs': float(TIMEOUT_SECS)}, 'claude timed out'),
    ],
    ids=['success', 'failure', 'timeout'],
)
def test_builder_temp_files_are_unlinked_after_the_run(tmp_path, monkeypatch, capsys, spec, extra, witness, pool):
    spawns = _record_spawns(monkeypatch)
    fake = _fake_claude(tmp_path, **spec)

    _run(fake, pool, **extra)

    assert len(spawns) == 1, f'expected exactly one spawn, got {[spawn.argv for spawn in spawns]}'
    spawn = spawns[0]
    assert len(spawn.temp_files()) == len(BUILDER_TEMP_FILE_FLAGS), f'the CLI must be handed one path per builder temp-file flag: {spawn.argv}'
    assert spawn.present_once_started == frozenset(spawn.temp_files()), 'the builder temp files must exist while the CLI runs'
    assert not any(path.exists() for path in spawn.temp_files()), 'the builder temp files must be unlinked after the run'
    captured = capsys.readouterr()
    assert witness in captured.out + captured.err, f'the run must take the path this case names: {witness!r}'


# ---------------------------------------------------------------------------
# 2. Read-only by construction.
# ---------------------------------------------------------------------------


def test_allow_and_deny_lists_are_disjoint(tmp_path):
    assert not set(mod.nightly_allowed_tools(tmp_path)) & set(mod.NIGHTLY_DENIED_TOOLS)


@pytest.mark.parametrize('verb', APPLY_VERBS)
def test_every_store_writing_apply_verb_is_denied(verb):
    assert verb in mod.NIGHTLY_DENIED_TOOLS


def test_the_only_bash_allowed_is_git_show_git_log_and_the_two_prepare_subcommands(tmp_path, pool):
    fake = _fake_claude(tmp_path)

    _run(fake, pool)

    allowed = _flag_values(fake.only_call()['argv'], '--allowed-tools')
    bash = [tool for tool in allowed if tool.startswith('Bash')]
    assert sorted(bash) == sorted([
        'Bash(git show:*)',
        'Bash(git log:*)',
        f'Bash({mod.PREPARE_COMMAND} brief --json:*)',
        f'Bash({mod.PREPARE_COMMAND} record --from:*)',
    ])
    assert mod.PREPARE_COMMAND.endswith(' scripts/sitting/prepare_sitting.py')


def test_every_allowed_git_read_has_its_output_flag_denied(tmp_path, pool):
    fake = _fake_claude(tmp_path)

    _run(fake, pool)

    argv = fake.only_call()['argv']
    allowed = _flag_values(argv, '--allowed-tools')
    denied = _flag_values(argv, '--disallowed-tools')
    git_reads = {match.group(1) for tool in allowed if (match := re.fullmatch(r'Bash\(git (\S+):\*\)', tool))}
    assert git_reads
    for sub in git_reads:
        assert f'Bash(git {sub} *--output*)' in denied


def test_every_allowed_mcp_tool_is_a_read():
    reads = [tool.rsplit('__', 1)[1] for tool in mod.NIGHTLY_READ_TOOLS if tool.startswith('mcp__')]

    assert reads, 'the run should be able to read escalations and tasks through MCP'
    assert all(verb.startswith(('get_', 'search')) for verb in reads), reads


def test_the_mcp_servers_are_exactly_the_checkouts_escalation_and_fused_memory_blocks(tmp_path, pool):
    """Explicit and strict: the reads never hang on the ambient .mcp.json being approved headless, and nothing else starts."""
    fake = _fake_claude(tmp_path)

    assert _run(fake, pool) == mod.EXIT_OK

    call = fake.only_call()
    ambient = json.loads((REPO_ROOT / '.mcp.json').read_text())['mcpServers']
    assert call['mcp_config'] == {'mcpServers': {name: ambient[name] for name in ('escalation', 'fused-memory')}}
    argv = call['argv']
    assert argv[argv.index(MCP_CONFIG_FLAG) + 2] == '--strict-mcp-config'


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
def test_an_unusable_mcp_config_is_a_configuration_error_and_launches_nothing(tmp_path, capsys, contents, named, pool):
    mcp_json = tmp_path / 'mcp.json'
    if contents is not None:
        mcp_json.write_text(contents)
    fake = _fake_claude(tmp_path)

    rc = _run(fake, pool, mcp_json=mcp_json)

    assert rc == mod.EXIT_CONFIG
    assert fake.calls() == []
    err = capsys.readouterr().err
    assert str(mcp_json) in err and named in err


def test_the_mcp_json_seam_reaches_the_run(tmp_path, pool):
    servers = {name: {'type': 'http', 'url': f'http://127.0.0.1:9/{name}'} for name in ('escalation', 'fused-memory')}
    mcp_json = tmp_path / 'mcp.json'
    mcp_json.write_text(json.dumps({'mcpServers': {**servers, 'playwright': {'command': 'npx'}}}))
    fake = _fake_claude(tmp_path)

    assert _run(fake, pool, mcp_json=mcp_json) == mod.EXIT_OK

    assert fake.only_call()['mcp_config'] == {'mcpServers': servers}


def test_the_child_runs_the_prepare_script_confined_to_data_sitting(tmp_path, pool):
    """The confinement itself is prepare_sitting's; test_prepare_sitting_cli.py::TestNightlyConfinement covers it."""
    fake = _fake_claude(tmp_path)

    _run(fake, pool)

    call = fake.only_call()
    assert call['env']['SITTING_NIGHTLY_CONFINED'] == call['inbox'], "the env names this run's inbox"
    assert prepare_sitting.DEFAULT_PREPARATION.parent == REPO_ROOT / 'data' / 'sitting'


def _landlock_grants(argv: list[str]) -> set[Path]:
    head = argv[:argv.index('--')]
    return {Path(head[i + 1]) for i, token in enumerate(head) if token == '--writable'}


@pytest.mark.skipif(not landlock.is_landlock_available(), reason='needs a Landlock kernel to run the fake confined')
def test_the_cli_runs_under_landlock_writing_only_the_run_grants(tmp_path, monkeypatch, host_home, pool):
    spawned: list[list[str]] = []
    real_popen = subprocess.Popen

    def recording_popen(args, *rest, **kwargs):
        spawned.append(list(args))
        return real_popen(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', recording_popen)
    fake = _fake_claude(tmp_path)

    assert _run(fake, pool, sandbox_backend='landlock') == mod.EXIT_OK

    [argv] = spawned
    assert Path(argv[1]).name == 'landlock_exec.py'
    assert argv[argv.index('--') + 1] == str(fake.bin)
    root = tmp_path.resolve()
    grants = _landlock_grants(argv)
    assert grants == {root / '.task', root / 'data' / 'sitting', host_home / '.cache' / 'uv'}
    config_dir = Path(fake.only_call()['env']['CLAUDE_CONFIG_DIR'])
    assert config_dir.is_absolute()
    assert any(config_dir.is_relative_to(grant) for grant in grants)


def test_without_landlock_the_run_is_refused_even_where_bwrap_is_available(tmp_path, monkeypatch, capsys, pool):
    monkeypatch.setattr(landlock, 'is_landlock_available', lambda: False)
    monkeypatch.setattr(sandbox, 'is_bwrap_available', lambda: True)
    fake = _fake_claude(tmp_path)

    rc = _run(fake, pool, sandbox_backend='landlock')

    assert rc == mod.EXIT_CONFIG
    assert fake.calls() == []
    assert 'refusing to run unconfined' in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 3. The account comes from the shared pool; the API key never rides along.
# ---------------------------------------------------------------------------


def test_a_leased_account_supplies_the_child_env(tmp_path, monkeypatch, pool):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-must-not-survive')
    monkeypatch.setenv('SITTING_TEST_MARKER', 'kept')
    fake = _fake_claude(tmp_path)

    assert _run(fake, pool) == mod.EXIT_OK

    call = fake.only_call()
    assert call['env']['CLAUDE_CODE_OAUTH_TOKEN'] == TOKEN_B, 'the first seat, in the order every other session uses'
    assert call['env']['SITTING_TEST_MARKER'] == 'kept'
    assert call['env']['SITTING_NIGHTLY_CONFINED']
    assert not call['env_has']['ANTHROPIC_API_KEY']


def test_a_proxied_shells_endpoint_selection_does_not_reach_the_child(tmp_path, monkeypatch, pool):
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'http://127.0.0.1:8317')
    monkeypatch.setenv('ANTHROPIC_AUTH_TOKEN', 'proxy-key')
    fake = _fake_claude(tmp_path)

    assert _run(fake, pool) == mod.EXIT_OK

    env_has = fake.only_call()['env_has']
    assert not env_has['ANTHROPIC_BASE_URL']
    assert not env_has['ANTHROPIC_AUTH_TOKEN']


def test_an_empty_pool_skips_the_night_without_launching_the_cli(tmp_path, capsys, pool_roster, sentinel_login):
    accounts_file, env_file = pool_roster(SEAT_B, SEAT_H, resolve_tokens=False)
    gate = account_pool.build_pool(accounts_file=accounts_file, env_file=env_file)
    fake = _fake_claude(tmp_path)

    assert _run(fake, gate) == mod.EXIT_FAILED

    assert fake.calls() == []
    assert 'skipping: the nightly never uses the host login' in capsys.readouterr().err
    assert not (tmp_path / 'data' / 'sitting').exists(), 'a skipped night creates no inbox and no config dir'


def test_the_entry_point_pins_landlock(tmp_path, monkeypatch, capsys, pool):
    monkeypatch.setattr(landlock, 'is_landlock_available', lambda: False)
    monkeypatch.setattr(sandbox, 'is_bwrap_available', lambda: True)
    fake = _fake_claude(tmp_path)

    rc = mod.main(['--claude-bin', str(fake.bin)], gate=pool, root=tmp_path)

    assert rc == mod.EXIT_CONFIG
    assert fake.calls() == []
    assert "backend='landlock'" in capsys.readouterr().err


def test_the_leased_token_is_the_config_dirs_login_for_the_run_only(tmp_path, pool):
    fake = _fake_claude(tmp_path)

    assert _run(fake, pool) == mod.EXIT_OK

    call = fake.only_call()
    assert json.loads(call['credentials']) == {'claudeAiOauth': {'accessToken': TOKEN_B}}
    assert call['credentials_mode'] == 0o600
    assert not (Path(call['env']['CLAUDE_CONFIG_DIR']) / '.credentials.json').exists()


# ---------------------------------------------------------------------------
# 4. Exit codes, and a failure that names the CLI's result and its stderr.
# ---------------------------------------------------------------------------


def test_exit_zero_with_a_json_result_is_ok(tmp_path, pool):
    assert _run(_fake_claude(tmp_path), pool) == mod.EXIT_OK


def test_a_successful_result_that_quotes_a_banner_is_still_ok(tmp_path, pool):
    """The run reads escalations ABOUT usage caps; a verdict quoting one is not a cap."""
    stdout = json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,
                         'result': f'esc-9-1 is about "{BANNER}"; prepared as live'})

    assert _run(_fake_claude(tmp_path, stdout=stdout), pool) == mod.EXIT_OK


def test_a_failed_run_fails_with_its_result_and_stderr_tails_labelled(tmp_path, capsys, pool):
    """The CLI's stdout is a JSON envelope; what the line tails is the ``result`` text inside it."""
    fake = _fake_claude(tmp_path, **_costed_failure('error_during_execution', result='partial-out-7Q'),
                        stderr='backend-down-9Z')

    assert _run(fake, pool) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    line = next(line for line in err.splitlines() if 'claude failed' in line)
    assert "result='partial-out-7Q'" in line
    assert 'stderr=' in line and 'backend-down-9Z' in line
    assert line.startswith('nightly_prepare: ')


def test_a_timeout_fails_naming_its_bound(tmp_path, capsys, pool):
    fake = _fake_claude(tmp_path, sleep=30.0)

    assert _run(fake, pool, timeout_secs=float(TIMEOUT_SECS)) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    line = next(line for line in err.splitlines() if 'timed out' in line)
    assert f'after {float(TIMEOUT_SECS):g}s' in line and 'recorded=' in line
    assert line.startswith('nightly_prepare: ')


def test_a_timeout_error_from_inside_the_runner_is_not_the_runs_own_bound(tmp_path, capsys, monkeypatch, pool):
    """``TimeoutError`` is also ``OSError(ETIMEDOUT)``: only the run's own deadline expiring is a timeout."""
    async def connection_timed_out(*args, **kwargs):
        raise TimeoutError(errno.ETIMEDOUT, 'Connection timed out')

    monkeypatch.setattr(mod, 'invoke_with_cap_retry', connection_timed_out)

    assert _run(_fake_claude(tmp_path), pool) == mod.EXIT_CONFIG

    err = capsys.readouterr().err
    assert 'Connection timed out' in err
    assert 'timed out after' not in err


def test_an_error_result_fails(tmp_path, capsys, pool):
    fake = _fake_claude(tmp_path, **_costed_failure('error_max_turns'))

    assert _run(fake, pool) == mod.EXIT_FAILED

    assert 'error_max_turns' in capsys.readouterr().err


def test_a_missing_binary_is_a_configuration_error(tmp_path, capsys, pool):
    rc = mod.main(['--claude-bin', str(tmp_path / 'no-such-claude')], gate=pool)

    assert rc == mod.EXIT_CONFIG
    assert 'no-such-claude' in capsys.readouterr().err


def test_with_no_flag_the_run_spawns_claude_binary(tmp_path, monkeypatch, pool):
    _strip_claude_from_path(tmp_path, monkeypatch)
    fake = _fake_claude(tmp_path)
    monkeypatch.setenv('CLAUDE_BINARY', str(fake.bin))

    rc = mod.main([], gate=pool, root=fake.root)

    assert rc == mod.EXIT_OK
    assert fake.only_call()['argv'][0] == str(fake.bin)


def test_the_flag_outranks_claude_binary(tmp_path, monkeypatch, pool):
    fake = _fake_claude(tmp_path)
    monkeypatch.setenv('CLAUDE_BINARY', str(tmp_path / 'no-such-claude'))

    rc = mod.main(['--claude-bin', str(fake.bin)], gate=pool, root=fake.root)

    assert rc == mod.EXIT_OK
    assert fake.only_call()['argv'][0] == str(fake.bin)


def test_an_unresolvable_claude_binary_without_the_flag_is_a_configuration_error(tmp_path, monkeypatch, capsys, pool):
    _strip_claude_from_path(tmp_path, monkeypatch)
    monkeypatch.setenv('CLAUDE_BINARY', str(tmp_path / 'no-such-claude'))

    rc = mod.main([], gate=pool, root=tmp_path)

    assert rc == mod.EXIT_CONFIG
    assert str(tmp_path / 'no-such-claude') in capsys.readouterr().err


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

    rc = mod.main(['--claude-bin', str(fake.bin)], gate=UnreadableRoster(), root=tmp_path, sandbox_backend='none')

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


def test_no_store_is_written_and_only_the_fake_is_spawned(tmp_path, monkeypatch, pool):
    queue = tmp_path / 'project' / 'data' / 'escalations'
    decisions = tmp_path / 'fleet' / 'decisions'
    queue.mkdir(parents=True)
    decisions.mkdir(parents=True)
    (queue / 'esc-9-1.json').write_text(json.dumps({'id': 'esc-9-1', 'status': 'pending', 'level': 2}))
    (decisions / 'dec-1.json').write_text(json.dumps({'decision_id': 'dec-1', 'state': 'open'}))
    before = _snapshot(queue, decisions)
    spawns = _record_spawns(monkeypatch)
    fake = _fake_claude(tmp_path)

    assert _run(fake, pool) == mod.EXIT_OK

    assert _snapshot(queue, decisions) == before
    assert not list(queue.parent.rglob('*.lock')) and not list(decisions.rglob('*.lock'))
    assert [spawn.argv[0] for spawn in spawns] == [str(fake.bin)]


def test_the_spawn_recorder_records_every_process_and_never_raises(tmp_path, monkeypatch):
    spawns = _record_spawns(monkeypatch)
    helper = tmp_path / 'helper'
    helper.write_text('#!/bin/sh\nexit 0\n')
    helper.chmod(0o755)

    subprocess.run([sys.executable, '-c', 'pass', MCP_CONFIG_FLAG], check=True)
    subprocess.run(helper, check=True)

    assert [spawn.argv for spawn in spawns] == [(sys.executable, '-c', 'pass', MCP_CONFIG_FLAG), (str(helper),)]
    assert all(spawn.temp_files() == () for spawn in spawns)
    assert all(spawn.present_once_started == frozenset() for spawn in spawns)


# ---------------------------------------------------------------------------
# 6. The outcome is what the inbox offered and the store holds, not the agent's say-so.
# ---------------------------------------------------------------------------


def test_two_recorded_preparations_are_ok_and_the_done_line_counts_them(tmp_path, capsys, pool):
    batches = {'batch-01.json': [_prep_payload(1)], 'batch-02.json': [_prep_payload(2)]}
    stdout = json.dumps({'type': 'result', 'is_error': False, 'result': 'recorded 2 of 2', 'num_turns': 9,
                         'total_cost_usd': 1.5})
    fake = _fake_claude(tmp_path, stdout=stdout, batches=batches, recorded=tuple(batches))

    assert _run(fake, pool) == mod.EXIT_OK

    done = next(line for line in capsys.readouterr().out.splitlines() if 'done' in line)
    assert 'recorded=2 of 2' in done and 'undecodable=0' in done and 'turns=9' in done
    assert f'account={SEAT_B!r}' in done


def test_a_success_that_recorded_nothing_fails(tmp_path, capsys, pool):
    fake = _fake_claude(tmp_path, batches={})

    assert _run(fake, pool) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    assert 'recorded nothing' in err and '0 item key(s) offered' in err


def test_an_inbox_file_the_store_does_not_hold_fails(tmp_path, capsys, pool):
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)]}, recorded=())

    assert _run(fake, pool) == mod.EXIT_FAILED

    assert '1 item key(s) offered' in capsys.readouterr().err


def test_the_store_holding_a_different_preparation_for_the_key_is_not_a_record(tmp_path, capsys, pool):
    other = {**_prep_payload(1), 'question': 'Lift the nightly cap on dec-1?'}
    store = tmp_path / 'data' / 'sitting' / 'preparation.json'
    store.parent.mkdir(parents=True)
    store.write_text(json.dumps({'preparations': [other]}))
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)]}, recorded=())

    assert _run(fake, pool) == mod.EXIT_FAILED

    assert 'recorded nothing' in capsys.readouterr().err


def test_a_key_offered_twice_is_recorded_when_the_store_holds_either_offer(tmp_path, capsys, pool):
    other = {**_prep_payload(1), 'question': 'Lift the nightly cap on dec-1?'}
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)], 'batch-02.json': [other]},
                        recorded=('batch-02.json',))

    assert _run(fake, pool) == mod.EXIT_OK

    assert 'recorded=1 of 1' in capsys.readouterr().out


def test_undecodable_inbox_content_is_counted_and_never_raises(tmp_path, capsys, pool):
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [{'item': 'not-a-key'}], 'torn.json': '{"item": ['})

    assert _run(fake, pool) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    assert '0 item key(s) offered' in err and 'undecodable=2' in err


def test_a_file_too_deeply_nested_to_parse_is_undecodable_not_fatal(tmp_path, capsys, pool):
    fake = _fake_claude(tmp_path, batches={'deep.json': '[' * 200_000, 'scalars.json': [5, 'x', None]})

    assert _run(fake, pool) == mod.EXIT_FAILED

    assert 'undecodable=4' in capsys.readouterr().err


def test_an_offer_identical_to_an_entry_held_before_the_run_is_not_a_record(tmp_path, capsys, pool):
    store = tmp_path / 'data' / 'sitting' / 'preparation.json'
    store.parent.mkdir(parents=True)
    store.write_text(json.dumps({'preparations': [_prep_payload(1)]}))
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)]}, recorded=('batch-01.json',))

    assert _run(fake, pool) == mod.EXIT_FAILED

    assert 'recorded nothing' in capsys.readouterr().err


def test_a_batch_in_a_subdirectory_or_not_json_is_not_offered(tmp_path, capsys, pool):
    fake = _fake_claude(tmp_path, batches={'notes.txt': [_prep_payload(1)], 'sub/batch-01.json': [_prep_payload(2)]})

    assert _run(fake, pool) == mod.EXIT_FAILED

    assert '0 item key(s) offered' in capsys.readouterr().err


def test_a_decodable_entry_beside_an_undecodable_one_still_counts(tmp_path, capsys, pool):
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)], 'draft.json': [{'item': 'not-a-key'}]},
                        recorded=('batch-01.json',))

    assert _run(fake, pool) == mod.EXIT_OK

    assert 'recorded=1 of 1' in capsys.readouterr().out


def test_a_corrupt_store_counts_zero_recorded(tmp_path, capsys, pool):
    fake = _fake_claude(tmp_path, batches={'batch-01.json': [_prep_payload(1)]}, recorded=())
    store = tmp_path / 'data' / 'sitting' / 'preparation.json'
    store.parent.mkdir(parents=True)
    store.write_text('{not json')

    assert _run(fake, pool) == mod.EXIT_FAILED

    err = capsys.readouterr().err
    assert 'unreadable' in err and 'recorded nothing' in err


def test_a_timeout_after_a_partial_record_still_counts_what_was_recorded(tmp_path, capsys, pool):
    batches = {'batch-01.json': [_prep_payload(1)], 'batch-02.json': [_prep_payload(2)]}
    fake = _fake_claude(tmp_path, batches=batches, recorded=('batch-01.json',), sleep=30.0)

    assert _run(fake, pool, timeout_secs=float(TIMEOUT_SECS)) == mod.EXIT_FAILED

    line = next(line for line in capsys.readouterr().err.splitlines() if 'timed out' in line)
    assert 'recorded=1' in line


def test_a_failed_run_states_how_much_was_recorded(tmp_path, capsys, pool):
    fake = _fake_claude(tmp_path, **_costed_failure('error_during_execution'), stderr='boom')

    assert _run(fake, pool) == mod.EXIT_FAILED

    line = next(line for line in capsys.readouterr().err.splitlines() if 'claude failed' in line)
    assert 'recorded=1' in line


# ---------------------------------------------------------------------------
# 7. A seat that cannot take the run hands it to the next one, inside one bound.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(FAILOVER_TEST_TIMEOUT_SECS)
def test_a_cap_on_the_first_seat_fails_over_to_the_next(tmp_path, pool):
    fake = _fake_claude(tmp_path, by_token={TOKEN_B: _cap()})

    assert _run(fake, pool) == mod.EXIT_OK

    assert _tokens_called(fake) == [TOKEN_B, TOKEN_H]


def test_an_auth_rejection_on_the_first_seat_fails_over_to_the_next(tmp_path, pool):
    fake = _fake_claude(tmp_path, by_token={TOKEN_B: _auth_401()})

    assert _run(fake, pool) == mod.EXIT_OK

    assert _tokens_called(fake) == [TOKEN_B, TOKEN_H]
    assert pool.auth_failed_account_names == (SEAT_B,)


@pytest.mark.timeout(FAILOVER_TEST_TIMEOUT_SECS)
def test_a_capped_session_with_progress_resumes_on_the_next_seat(tmp_path, pool):
    sid = '5f0c1d2e-0000-4000-8000-000000000001'
    fake = _fake_claude(tmp_path, by_token={TOKEN_B: {**_cap(sid), 'transcript': _progress_transcript(sid)}})

    assert _run(fake, pool) == mod.EXIT_OK

    first, resumed = fake.calls()
    assert first['env']['CLAUDE_CODE_OAUTH_TOKEN'] == TOKEN_B
    assert resumed['env']['CLAUDE_CODE_OAUTH_TOKEN'] == TOKEN_H
    argv = resumed['argv']
    assert argv[argv.index('--resume') + 1] == sid
    assert resumed['stdin'] == CAP_HIT_RESUME_PROMPT


@pytest.mark.timeout(FAILOVER_TEST_TIMEOUT_SECS)
def test_the_bound_covers_the_whole_run_failovers_included(tmp_path, capsys, pool):
    """The first seat caps fast, and the shared runner's 5s cooldown before the next seat outlasts the bound."""
    fake = _fake_claude(tmp_path, by_token={TOKEN_B: _cap()})

    assert _run(fake, pool, timeout_secs=float(TIMEOUT_SECS)) == mod.EXIT_FAILED

    assert _tokens_called(fake) == [TOKEN_B]
    assert any('timed out' in line for line in capsys.readouterr().err.splitlines())


def _exhaustion_line(err: str) -> str:
    [line] = [line for line in err.splitlines() if 'no pool account could take the run' in line]
    assert line.startswith('nightly_prepare: ')
    return line


def _credentials_left(root: Path) -> list[Path]:
    return list((root / 'data' / 'sitting' / 'config').rglob('.credentials.json'))


@pytest.mark.timeout(FAILOVER_TEST_TIMEOUT_SECS)
def test_every_seat_capped_fails_naming_the_earliest_reset(tmp_path, capsys, pool):
    fake = _fake_claude(tmp_path, by_token={TOKEN_B: _cap(), TOKEN_H: _cap()})

    assert _run(fake, pool) == mod.EXIT_FAILED

    assert _tokens_called(fake) == [TOKEN_B, TOKEN_H], 'one pass over the pool, then stop'
    line = _exhaustion_line(capsys.readouterr().err)
    assert f'capped [{SEAT_B}, {SEAT_H}], auth-failed []' in line
    assert pool.soonest_resets_at is not None
    assert f'earliest reset {pool.soonest_resets_at.isoformat()}' in line
    assert 'recorded=0' in line
    assert _credentials_left(tmp_path) == []


@pytest.mark.timeout(FAILOVER_TEST_TIMEOUT_SECS)
def test_a_pool_both_capped_and_rejecting_names_each_seat_in_its_state(tmp_path, capsys, pool):
    """Ending on a frozen pool rather than on the cap-retry bound logs the same shape, with the capped seat's reset."""
    fake = _fake_claude(tmp_path, by_token={TOKEN_B: _cap(), TOKEN_H: _auth_401()})

    assert _run(fake, pool) == mod.EXIT_FAILED

    assert _tokens_called(fake) == [TOKEN_B, TOKEN_H]
    line = _exhaustion_line(capsys.readouterr().err)
    assert f'capped [{SEAT_B}], auth-failed [{SEAT_H}]' in line
    assert pool.soonest_resets_at is not None
    assert f'earliest reset {pool.soonest_resets_at.isoformat()}' in line
    assert _credentials_left(tmp_path) == []


def test_every_seat_rejecting_its_credentials_fails_and_never_rides_the_host_login(
    tmp_path, capsys, sentinel_login, pool,
):
    fake = _fake_claude(tmp_path, by_token={TOKEN_B: _auth_401(), TOKEN_H: _auth_401()})

    assert _run(fake, pool) == mod.EXIT_FAILED

    assert _tokens_called(fake) == [TOKEN_B, TOKEN_H], 'no third call on a fallback credential'
    assert not any(Path(call['env']['CLAUDE_CONFIG_DIR']).is_relative_to(sentinel_login) for call in fake.calls())
    line = _exhaustion_line(capsys.readouterr().err)
    assert f'capped [], auth-failed [{SEAT_B}, {SEAT_H}]; earliest reset unknown' in line
    assert _credentials_left(tmp_path) == []
