"""The nightly transcript secret sweep redacts live values already on disk.

Driven through the public ``run``: the environment, the clock, the escalation
poster and every root are passed explicitly, so no real HOME, unit dir or HTTP
is touched; ``main`` is driven with explicit roots, an empty unit dir and an
escalation URL nothing listens on. The values are fakes; the placeholder shape
is the 2026-10-11 one.
"""

from __future__ import annotations

import datetime
import json
import os
import stat
from pathlib import Path

import pytest
import transcript_secret_sweep as sweep
from shared.project_layout import agent_transcripts_dir
from shared.secret_env import is_secret_env_name
from shared.secret_redaction import FileChangedDuringRedaction
from shared.transcript_archive import STAGING_SUFFIX

POOL_A = 'sk-ant-oat01-' + 'Lv' * 17 + 'A'
OPENAI = 'sk-proj-' + 'Ok' * 20
NON_LIVE = 'sk-ant-oat01-' + 'Nl' * 17 + 'N'
ENVIRON = {'CLAUDE_OAUTH_TOKEN_A': POOL_A, 'OPENAI_API_KEY': OPENAI, 'PATH': '/usr/bin'}

NOW = datetime.datetime(2026, 10, 21, 2, 30, tzinfo=datetime.UTC)
OLD_MTIME_NS = int((NOW.timestamp() - 3 * 86400) * 1e9)
FRESH_MTIME_NS = int((NOW.timestamp() - 60) * 1e9)


def _line_with(text: str) -> str:
    return json.dumps({'type': 'user', 'message': {'content': [{'type': 'tool_result', 'content': text}]}})


def _write(path: Path, lines: list[str], *, mtime_ns: int = OLD_MTIME_NS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(''.join(f'{line}\n' for line in lines).encode())
    path.chmod(0o664)
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


class Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, arguments: dict) -> dict:
        self.calls.append(arguments)
        return {'status': 'queued'}


class FailingPoster:
    def __call__(self, arguments: dict) -> dict:
        raise ConnectionError('escalation server unreachable')


class Tree:
    def __init__(self, base: Path) -> None:
        self.project_a = base / 'proj-a'
        self.project_b = base / 'proj-b'
        self.claude_projects = base / 'home' / '.claude' / 'projects'
        a_dir = agent_transcripts_dir(self.project_a) / '6000' / '-enc-a'
        b_dir = agent_transcripts_dir(self.project_b) / '6001' / '-enc-b'
        c_dir = self.claude_projects / '-home-x-proj'
        self.pool_main = _write(
            a_dir / 'sid-a.jsonl',
            [_line_with('hi'), _line_with(f'token {POOL_A} here'), _line_with(f'CLAUDE_OAUTH_TOKEN_A={POOL_A}\n')],
        )
        self.pool_subagent = _write(
            a_dir / 'sid-a' / 'subagents' / 'agent-1.jsonl', [_line_with(f'env: {POOL_A}')],
        )
        self.openai = _write(c_dir / 'sid-c.jsonl', [_line_with(f'OPENAI_API_KEY={OPENAI}\n')])
        self.clean = _write(b_dir / 'sid-b.jsonl', [_line_with('nothing to see')])
        self.non_live = _write(c_dir / 'sid-d.jsonl', [_line_with(f'old token {NON_LIVE}')])
        self.fresh = _write(
            b_dir / 'sid-fresh.jsonl', [_line_with(f'live session {POOL_A}')], mtime_ns=FRESH_MTIME_NS,
        )

    def config(self) -> sweep.SweepConfig:
        roots = sweep.transcript_roots([self.project_a, self.project_b], self.claude_projects)
        return sweep.SweepConfig(roots=roots, now=NOW)

    def files(self) -> list[Path]:
        return [self.pool_main, self.pool_subagent, self.openai, self.clean, self.non_live, self.fresh]


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    return Tree(tmp_path)


def _identity(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_ino, st.st_mtime_ns


def _assert_redacted(path: Path, var: str, value: str, before: os.stat_result, line_count: int) -> None:
    data = path.read_bytes()
    after = path.stat()
    assert value.encode() not in data
    assert f'<redacted:{var}>'.encode() in data
    assert after.st_size == before.st_size
    assert len(data.splitlines()) == line_count
    for line in data.splitlines():
        json.loads(line)
    assert stat.S_IMODE(after.st_mode) == 0o664
    assert after.st_mtime_ns == before.st_mtime_ns


def _no_value_in(text: str) -> None:
    for value in (POOL_A, OPENAI):
        assert value not in text


def test_roots_are_each_projects_transcript_archive_plus_the_claude_projects_dir(tree):
    assert sweep.transcript_roots([tree.project_a, tree.project_b], tree.claude_projects) == (
        agent_transcripts_dir(tree.project_a),
        agent_transcripts_dir(tree.project_b),
        tree.claude_projects,
    )


def test_one_sweep_redacts_every_live_value_and_files_one_escalation(tree):
    before = {path: path.stat() for path in tree.files()}
    lines = {path: len(path.read_bytes().splitlines()) for path in tree.files()}
    untouched = {path: (path.read_bytes(), _identity(path)) for path in (tree.clean, tree.non_live, tree.fresh)}
    poster = Recorder()

    result = sweep.run(tree.config(), environ=ENVIRON, post_escalation=poster)

    assert result.ok
    _assert_redacted(tree.pool_main, 'CLAUDE_OAUTH_TOKEN_A', POOL_A, before[tree.pool_main], lines[tree.pool_main])
    _assert_redacted(
        tree.pool_subagent, 'CLAUDE_OAUTH_TOKEN_A', POOL_A, before[tree.pool_subagent], lines[tree.pool_subagent],
    )
    _assert_redacted(tree.openai, 'OPENAI_API_KEY', OPENAI, before[tree.openai], lines[tree.openai])
    for path, (content, identity) in untouched.items():
        assert path.read_bytes() == content
        assert _identity(path) == identity
    assert {hit.path: dict(hit.counts) for hit in result.redacted} == {
        tree.pool_main: {'CLAUDE_OAUTH_TOKEN_A': 2},
        tree.pool_subagent: {'CLAUDE_OAUTH_TOKEN_A': 1},
        tree.openai: {'OPENAI_API_KEY': 1},
    }
    assert result.deferred == (tree.fresh,)
    assert result.clean == 2

    (arguments,) = poster.calls
    assert arguments['category'] == 'risk_identified'
    assert arguments['task_id'] == 'transcript-secret-sweep'
    assert arguments['agent_role'] == 'transcript-secret-sweep'
    assert f'{tree.pool_main}: CLAUDE_OAUTH_TOKEN_A×2' in arguments['detail']
    assert f'{tree.pool_subagent}: CLAUDE_OAUTH_TOKEN_A×1' in arguments['detail']
    assert f'{tree.openai}: OPENAI_API_KEY×1' in arguments['detail']
    assert 'in 3 file(s) (4 occurrence(s))' in arguments['summary']
    assert '0 line(s) in 0 file(s) refused' in arguments['summary']
    (evidence,) = arguments['evidence']
    assert set(evidence) == {'observation', 'measured_at', 'ref'}
    _no_value_in(json.dumps(arguments))
    _no_value_in(json.dumps(result.record()))


def test_a_second_sweep_changes_nothing_and_posts_nothing(tree):
    sweep.run(tree.config(), environ=ENVIRON, post_escalation=Recorder())
    identities = {path: _identity(path) for path in tree.files()}
    poster = Recorder()

    result = sweep.run(tree.config(), environ=ENVIRON, post_escalation=poster)

    assert result.ok
    assert result.redacted == ()
    assert poster.calls == []
    assert {path: _identity(path) for path in tree.files()} == identities


def test_an_environment_without_secrets_cannot_run(tree):
    contents = {path: path.read_bytes() for path in tree.files()}
    poster = Recorder()

    result = sweep.run(tree.config(), environ={'PATH': '/usr/bin'}, post_escalation=poster)

    assert not result.ok
    assert 'no secret-bearing values in the environment' in (result.cause or '')
    assert result.redacted == () and result.clean == 0 and result.deferred == ()
    assert poster.calls == []
    assert {path: path.read_bytes() for path in tree.files()} == contents


def test_a_refused_line_is_escalated_and_the_rest_of_its_file_is_redacted(tmp_path):
    value = '00e9' + 'X' * 44
    project = tmp_path / 'proj'
    held = _write(
        agent_transcripts_dir(project) / '7000' / '-enc' / 'sid.jsonl',
        [json.dumps({'text': f'plain {value}'}), json.dumps({'text': 'café' + 'X' * 44})],
    )
    safe_line, refused_line = held.read_bytes().splitlines()
    config = sweep.SweepConfig(roots=sweep.transcript_roots([project], tmp_path / 'no-claude'), now=NOW)
    poster = Recorder()

    result = sweep.run(config, environ={'CLAUDE_OAUTH_TOKEN_B': value}, post_escalation=poster)

    assert [(line.path, line.var, line.line_number) for line in result.refused] == [
        (held, 'CLAUDE_OAUTH_TOKEN_B', 2),
    ]
    assert {hit.path: dict(hit.counts) for hit in result.redacted} == {held: {'CLAUDE_OAUTH_TOKEN_B': 1}}
    held_safe_line, held_refused_line = held.read_bytes().splitlines()
    assert held_refused_line == refused_line
    assert value.encode() not in held_safe_line
    assert len(held_safe_line) == len(safe_line)
    (arguments,) = poster.calls
    assert '1 line(s) in 1 file(s) refused' in arguments['summary']
    assert f'{held}: refused (CLAUDE_OAUTH_TOKEN_B on line 2), left un-redacted' in arguments['detail']
    assert f'{held}: CLAUDE_OAUTH_TOKEN_B×1' in arguments['detail']
    assert value not in json.dumps(arguments)
    assert value not in json.dumps(result.record())


def test_archive_staging_debris_is_swept_and_other_names_are_not(tmp_path):
    project = tmp_path / 'proj'
    session_dir = agent_transcripts_dir(project) / '7001' / '-enc'
    debris = _write(session_dir / f'sid.jsonl{STAGING_SUFFIX}', [_line_with(f'token {POOL_A}')])
    other = _write(session_dir / 'notes.txt', [_line_with(f'token {POOL_A}')])
    other_content = other.read_bytes()
    before = debris.stat()
    config = sweep.SweepConfig(roots=sweep.transcript_roots([project], tmp_path / 'no-claude'), now=NOW)

    result = sweep.run(config, environ=ENVIRON, post_escalation=Recorder())

    _assert_redacted(debris, 'CLAUDE_OAUTH_TOKEN_A', POOL_A, before, 1)
    assert [hit.path for hit in result.redacted] == [debris]
    assert other.read_bytes() == other_content


@pytest.mark.timeout(60)
def test_a_fifo_named_like_a_transcript_is_passed_over_without_being_opened(tmp_path):
    project = tmp_path / 'proj'
    session_dir = agent_transcripts_dir(project) / '7002' / '-enc'
    leaked = _write(session_dir / 'sid.jsonl', [_line_with(f'token {POOL_A}')])
    fifo = session_dir / 'pipe.jsonl'
    os.mkfifo(fifo)
    config = sweep.SweepConfig(roots=sweep.transcript_roots([project], tmp_path / 'no-claude'), now=NOW)

    result = sweep.run(config, environ=ENVIRON, post_escalation=Recorder())

    assert [hit.path for hit in result.redacted] == [leaked]
    assert fifo not in (*result.deferred, *result.unreadable)
    assert result.clean == 0


@pytest.mark.skipif(os.geteuid() == 0, reason='root reads a mode-000 file')
def test_an_unreadable_file_is_reported_and_its_siblings_are_still_swept(tree):
    tree.openai.chmod(0)
    try:
        result = sweep.run(tree.config(), environ=ENVIRON, post_escalation=Recorder())
    finally:
        tree.openai.chmod(0o664)

    assert result.unreadable == (tree.openai,)
    assert {hit.path for hit in result.redacted} == {tree.pool_main, tree.pool_subagent}
    assert OPENAI.encode() in tree.openai.read_bytes()


def test_a_file_that_changes_while_it_is_redacted_is_deferred(tree, monkeypatch):
    real_redact_in_place = sweep.redact_in_place

    def appended_to_mid_redaction(path, needles, **kwargs):
        if path == tree.pool_subagent:
            raise FileChangedDuringRedaction(path)
        return real_redact_in_place(path, needles, **kwargs)

    monkeypatch.setattr(sweep, 'redact_in_place', appended_to_mid_redaction)

    result = sweep.run(tree.config(), environ=ENVIRON, post_escalation=Recorder())

    assert result.deferred == (tree.pool_subagent, tree.fresh)
    assert {hit.path for hit in result.redacted} == {tree.pool_main, tree.openai}


def test_a_failed_escalation_is_reported_and_the_redaction_still_stands(tree):
    result = sweep.run(tree.config(), environ=ENVIRON, post_escalation=FailingPoster())

    assert result.ok
    assert result.escalation_error == 'ConnectionError: escalation server unreachable'
    assert POOL_A.encode() not in tree.pool_main.read_bytes()
    assert result.record()['escalation_error'] == result.escalation_error


class TestMain:
    @pytest.fixture
    def secretless_environ(self, monkeypatch):
        for name in list(os.environ):
            if is_secret_env_name(name):
                monkeypatch.delenv(name)

    def _argv(self, tree: Tree, tmp_path: Path) -> list[str]:
        return [
            '--project-root', str(tree.project_a),
            '--project-root', str(tree.project_b),
            '--units-dir', str(tmp_path / 'no-units'),
            '--claude-projects-dir', str(tree.claude_projects),
            '--escalation-url', 'http://127.0.0.1:1/mcp',
            '--min-age-secs', '0',
        ]

    def _record(self, capsys) -> dict:
        (line,) = capsys.readouterr().out.splitlines()
        return json.loads(line)

    def test_a_sweep_with_nothing_to_redact_exits_0(self, secretless_environ, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_A', NON_LIVE + 'z' * 8)
        tree = Tree(tmp_path)

        assert sweep.main(self._argv(tree, tmp_path)) == 0

        record = self._record(capsys)
        assert record['ok'] is True
        assert record['redacted'] == []
        assert record['escalation_error'] is None

    def test_an_environment_without_secrets_exits_2(self, secretless_environ, tmp_path, capsys):
        tree = Tree(tmp_path)

        assert sweep.main(self._argv(tree, tmp_path)) == 2

        assert self._record(capsys)['ok'] is False

    def test_an_escalation_that_cannot_be_filed_exits_2(self, secretless_environ, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_A', POOL_A)
        tree = Tree(tmp_path)

        assert sweep.main(self._argv(tree, tmp_path)) == 2

        out, err = capsys.readouterr()
        (line,) = out.splitlines()
        record = json.loads(line)
        assert record['ok'] is True
        assert record['escalation_error']
        assert 'the escalation was not filed' in err
        assert POOL_A not in out + err
