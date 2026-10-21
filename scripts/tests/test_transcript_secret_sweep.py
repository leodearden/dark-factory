"""The nightly transcript secret sweep redacts live values already on disk.

Driven through the public ``run``: the environment, the clock, the escalation
poster and every root are passed explicitly, so no real HOME, unit dir or HTTP
is touched. The values are fakes; the placeholder shape is the 2026-10-11 one.
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
    assert '3' in arguments['summary']
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


def test_a_refused_file_is_escalated_as_left_unredacted(tmp_path):
    value = '00e9' + 'X' * 44
    project = tmp_path / 'proj'
    refused = _write(
        agent_transcripts_dir(project) / '7000' / '-enc' / 'sid.jsonl',
        [json.dumps({'text': 'café' + 'X' * 44})],
    )
    content = refused.read_bytes()
    config = sweep.SweepConfig(roots=sweep.transcript_roots([project], tmp_path / 'no-claude'), now=NOW)
    poster = Recorder()

    result = sweep.run(config, environ={'CLAUDE_OAUTH_TOKEN_B': value}, post_escalation=poster)

    assert [refusal.path for refusal in result.refused] == [refused]
    assert result.redacted == ()
    assert refused.read_bytes() == content
    (arguments,) = poster.calls
    assert str(refused) in arguments['detail']
    assert 'un-redacted' in arguments['detail']
    assert value not in json.dumps(arguments)
    assert value not in json.dumps(result.record())
