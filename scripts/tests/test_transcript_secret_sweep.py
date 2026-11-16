"""The nightly transcript secret sweep redacts live values already on disk.

Driven through the public ``run``: the environment, the clock, the escalation
poster and every root are passed explicitly, so no real HOME, unit dir or HTTP
is touched; ``main`` is driven with explicit roots, an empty unit dir and an
escalation URL nothing listens on. The values are fakes; the placeholder shape
is the 2026-10-11 one.
"""

from __future__ import annotations

import datetime
import getpass
import gzip
import hashlib
import json
import os
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import pytest
import transcript_secret_sweep as sweep
from shared.project_layout import agent_transcripts_dir, offline_lane_dir, verify_logs_dir
from shared.secret_env import is_secret_env_name
from shared.secret_redaction import FileChangedDuringRedaction
from shared.testing_mcp_stub import StubMcpServer
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
    return _write_raw(path, ''.join(f'{line}\n' for line in lines).encode(), mtime_ns=mtime_ns)


def _write_raw(path: Path, data: bytes, *, mtime_ns: int = OLD_MTIME_NS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o664)
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _write_gzip(path: Path, lines: list[str], *, mtime_ns: int = OLD_MTIME_NS) -> Path:
    _write(path, lines, mtime_ns=mtime_ns)
    path.write_bytes(gzip.compress(path.read_bytes()))
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def _junit_lines(*, message: str, text: str) -> list[str]:
    suite = ET.Element('testsuite', name='orchestrator', tests='1', failures='1')
    case = ET.SubElement(suite, 'testcase', classname='tests.test_env', name='test_env_is_scrubbed')
    failure = ET.SubElement(case, 'failure', message=message)
    failure.text = text
    ET.indent(suite)
    return ET.tostring(suite, encoding='unicode', xml_declaration=True).splitlines()


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
        self.project_roots = (self.project_a, self.project_b)
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
        return sweep.SweepConfig(trees=sweep.swept_trees(self.project_roots, self.claude_projects), now=NOW)

    def files(self) -> list[Path]:
        return [self.pool_main, self.pool_subagent, self.openai, self.clean, self.non_live, self.fresh]


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    return Tree(tmp_path)


class LogTree:
    """One project's verify and offline-lane logs, as failing pytest env assertions leave them."""

    def __init__(self, base: Path) -> None:
        self.project = base / 'proj'
        self.project_roots = (self.project,)
        self.claude_projects = base / 'home' / '.claude' / 'projects'
        attempt = verify_logs_dir(self.project) / '6828'
        lane_pass = offline_lane_dir(self.project) / 'qdrant-integration' / '20261020T013000Z'
        self.verify_log = _write(attempt / 'attempt-1.remote-laptop.test-X.log', [
            '=================================== FAILURES ===================================',
            f"E   AssertionError: {{'CLAUDE_OAUTH_TOKEN_A': '{POOL_A}', 'PATH': '/usr/bin'}}",
            '1 failed in 0.42s',
        ])
        self.junit_gz = _write_gzip(
            attempt / 'attempt-1.orchestrator.junit-X.xml.gz',
            _junit_lines(message=f'AssertionError: leaked {POOL_A}', text=f'E   CLAUDE_OAUTH_TOKEN_A={POOL_A}'),
        )
        self.summary = _write(
            attempt / 'attempt-1.summary-X.json',
            [json.dumps({'rc': 1, 'cause_hint': f'env leaked OPENAI_API_KEY={OPENAI}'})],
        )
        self.detect_log = _write(lane_pass / 'detect.log', [f'E   OPENAI_API_KEY={OPENAI}', '1 failed'])
        self.detect_junit = _write(
            lane_pass / 'detect.junit.xml', _junit_lines(message=f'leaked {POOL_A}', text='E   assert False'),
        )
        self.notes = _write(attempt / 'notes.txt', [f'token {POOL_A}'])
        self.fresh = _write(
            verify_logs_dir(self.project) / '6829' / 'attempt-1.test-Y.log',
            [f'E   {POOL_A}'],
            mtime_ns=FRESH_MTIME_NS,
        )

    def config(self) -> sweep.SweepConfig:
        return sweep.SweepConfig(trees=sweep.swept_trees(self.project_roots, self.claude_projects), now=NOW)

    def expected_counts(self) -> dict[Path, dict[str, int]]:
        return {
            self.verify_log: {'CLAUDE_OAUTH_TOKEN_A': 1},
            self.junit_gz: {'CLAUDE_OAUTH_TOKEN_A': 2},
            self.summary: {'OPENAI_API_KEY': 1},
            self.detect_log: {'OPENAI_API_KEY': 1},
            self.detect_junit: {'CLAUDE_OAUTH_TOKEN_A': 1},
        }


@pytest.fixture
def logs(tmp_path: Path) -> LogTree:
    return LogTree(tmp_path)


def _env_minus_zero(environ: Mapping[str, str]) -> bytes:
    return b'\0'.join(f'{name}={value}'.encode() for name, value in environ.items())


class PytestTmpTree:
    """A pytest base temp as env-capturing tests leave it: raw environ dumps and JSON captures."""

    def __init__(self, base: Path) -> None:
        self.root = base / 'pytest-of-u'
        test_dir = self.root / 'pytest-7' / 'popen-gw0' / 'test_x0'
        raw = _env_minus_zero(ENVIRON)
        self.env_dump = _write_raw(test_dir / 'env-dump', raw)
        self.captured_env = _write_raw(test_dir / 'captured_env.bin', raw)
        self.child_env = _write_raw(test_dir / 'child_env.bin', raw)
        self.env_json = _write(test_dir / 'env.json', [json.dumps(ENVIRON)])
        self.env_dump_json = _write(test_dir / 'env-dump.json', [json.dumps({'env': ENVIRON})])
        self.capture_json = _write(test_dir / 'capture.json', json.dumps({'env': ENVIRON}, indent=2).splitlines())
        self.capture_jsonl = _write(test_dir / 'capture.jsonl', [json.dumps({'argv': ['claude'], 'env': ENVIRON})])
        self.runner_state = _write(
            test_dir / 'fake-runner-state.json',
            [json.dumps({'calls': [{'env': {'CLAUDE_OAUTH_TOKEN_A': POOL_A}}]})],
        )
        self.decoys = (
            _write_raw(test_dir / 'seeded.bin', raw),
            _write(test_dir / 'notes.txt', [f'token {POOL_A}']),
            _write(test_dir / '.env', [f'CLAUDE_OAUTH_TOKEN_A={POOL_A}', f'OPENAI_API_KEY={OPENAI}']),
        )
        self.fresh = _write_raw(
            self.root / 'pytest-8' / 'test_y0' / 'env-dump', raw, mtime_ns=FRESH_MTIME_NS,
        )

    def expected_counts(self) -> dict[Path, dict[str, int]]:
        both = {'CLAUDE_OAUTH_TOKEN_A': 1, 'OPENAI_API_KEY': 1}
        return {
            self.env_dump: both,
            self.captured_env: both,
            self.child_env: both,
            self.env_json: both,
            self.env_dump_json: both,
            self.capture_json: both,
            self.capture_jsonl: both,
            self.runner_state: {'CLAUDE_OAUTH_TOKEN_A': 1},
        }

    def config(self, base: Path) -> sweep.SweepConfig:
        return sweep.SweepConfig(
            trees=sweep.swept_trees([], base / 'no-claude', pytest_tmp_dir=self.root), now=NOW,
        )


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


def _decoded(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.name.endswith('.gz') else data


def _readable_xml(content: bytes) -> str:
    root = ET.fromstring(content)
    return ' '.join([*(value for element in root.iter() for value in element.attrib.values()), *root.itertext()])


def _assert_log_redacted(path: Path, var: str, before: os.stat_result, before_content: bytes) -> None:
    content = _decoded(path)
    after = path.stat()
    assert ENVIRON[var].encode() not in content
    assert len(content) == len(before_content)
    assert len(content.splitlines()) == len(before_content.splitlines())
    if '.xml' in path.name:
        assert f'<redacted:{var}>' in _readable_xml(content)
    else:
        assert f'<redacted:{var}>'.encode() in content
    if path.suffix == '.json':
        json.loads(content)
    assert stat.S_IMODE(after.st_mode) == 0o664
    assert after.st_mtime_ns == before.st_mtime_ns


def _no_value_in(text: str) -> None:
    for value in (POOL_A, OPENAI):
        assert value not in text


def test_swept_trees_are_each_projects_transcripts_verify_logs_and_offline_lane_plus_the_claude_projects_dir(tree):
    a, b, claude = tree.project_a, tree.project_b, tree.claude_projects

    assert sweep.swept_trees([a, b], claude) == (
        sweep.SweptTree(agent_transcripts_dir(a), sweep.TRANSCRIPT_SUFFIXES),
        sweep.SweptTree(verify_logs_dir(a), sweep.LOG_SUFFIXES),
        sweep.SweptTree(offline_lane_dir(a), sweep.LOG_SUFFIXES),
        sweep.SweptTree(agent_transcripts_dir(b), sweep.TRANSCRIPT_SUFFIXES),
        sweep.SweptTree(verify_logs_dir(b), sweep.LOG_SUFFIXES),
        sweep.SweptTree(offline_lane_dir(b), sweep.LOG_SUFFIXES),
        sweep.SweptTree(claude, sweep.TRANSCRIPT_SUFFIXES),
    )
    assert {'.log', '.json', '.xml', '.xml.gz'} <= set(sweep.LOG_SUFFIXES)


def test_the_pytest_base_temp_is_swept_last_and_only_for_env_dump_names(tree, tmp_path):
    roots, claude = tree.project_roots, tree.claude_projects
    pytest_tmp = tmp_path / 'pytest-of-u'

    assert sweep.swept_trees(roots, claude, pytest_tmp_dir=pytest_tmp) == (
        *sweep.swept_trees(roots, claude),
        sweep.SweptTree(pytest_tmp, sweep.PYTEST_ENV_DUMP_SUFFIXES),
    )


def test_the_pytest_base_temp_is_pytests_own_per_user_dir():
    assert sweep.pytest_base_temp() == Path(tempfile.gettempdir()) / f'pytest-of-{getpass.getuser()}'


def test_one_sweep_redacts_the_env_dumps_a_pytest_base_temp_holds(tmp_path):
    dumps = PytestTmpTree(tmp_path)
    expected = dumps.expected_counts()
    before = {path: path.stat() for path in expected}
    decoys = {path: (path.read_bytes(), _identity(path)) for path in dumps.decoys}
    poster = Recorder()

    result = sweep.run(dumps.config(tmp_path), environ=ENVIRON, post_escalation=poster)

    assert {hit.path: dict(hit.counts) for hit in result.redacted} == expected
    for path in expected:
        data = path.read_bytes()
        after = path.stat()
        _no_value_in(data.decode())
        assert after.st_size == before[path].st_size
        assert stat.S_IMODE(after.st_mode) == 0o664
        assert after.st_mtime_ns == before[path].st_mtime_ns
        if path.suffix == '.json':
            json.loads(data)
    for path, (content, identity) in decoys.items():
        assert path.read_bytes() == content
        assert _identity(path) == identity
    assert result.deferred == (dumps.fresh,)
    (arguments,) = poster.calls
    for path in expected:
        assert f'{path}: ' in arguments['detail']
    _no_value_in(json.dumps(arguments))
    _no_value_in(json.dumps(result.record()))


def test_a_truncated_value_in_a_verify_log_and_a_gzipped_junit_is_redacted(tmp_path):
    var = 'ANTHROPIC_API_KEY'
    value = 'sk-ant-api03-' + hashlib.sha512(b'a live anthropic key').hexdigest()[:95]
    attempt = verify_logs_dir(tmp_path / 'proj') / '10160'
    log = _write(attempt / 'attempt-1.remote-laptop.test-X.log', [
        f"E   AssertionError: {{'{var}': '{value[:95]}...', 'PATH': '/usr/bin'}}",
        '1 failed in 0.42s',
    ])
    junit = _write_gzip(
        attempt / 'attempt-1.orchestrator.junit-X.xml.gz',
        _junit_lines(message=f'AssertionError: leaked {value[:40]}...', text='E   assert False'),
    )
    before = {path: (path.stat(), _decoded(path)) for path in (log, junit)}
    config = sweep.SweepConfig(trees=sweep.swept_trees([tmp_path / 'proj'], tmp_path / 'no-claude'), now=NOW)
    poster = Recorder()

    result = sweep.run(config, environ={var: value}, post_escalation=poster)

    assert {hit.path: dict(hit.counts) for hit in result.redacted} == {log: {var: 1}, junit: {var: 1}}
    for path, (st, content) in before.items():
        after = _decoded(path)
        assert len(after) == len(content)
        assert value[:40].encode() not in after
        assert path.stat().st_mtime_ns == st.st_mtime_ns
    assert f'<redacted:{var}>' in _readable_xml(_decoded(junit))
    (arguments,) = poster.calls
    for text in (json.dumps(arguments), json.dumps(result.record())):
        assert value[:16] not in text


def test_one_sweep_redacts_verify_and_offline_lane_logs(logs):
    expected = logs.expected_counts()
    before = {path: (path.stat(), _decoded(path)) for path in expected}
    notes = logs.notes.read_bytes()
    poster = Recorder()

    result = sweep.run(logs.config(), environ=ENVIRON, post_escalation=poster)

    assert result.ok
    for path, counts in expected.items():
        (var,) = counts
        _assert_log_redacted(path, var, *before[path])
    assert {hit.path: dict(hit.counts) for hit in result.redacted} == expected
    assert logs.notes.read_bytes() == notes
    assert result.deferred == (logs.fresh,)
    assert result.unreadable == ()
    (arguments,) = poster.calls
    for path, counts in expected.items():
        ((var, n),) = counts.items()
        assert f'{path}: {var}×{n}' in arguments['detail']
    _no_value_in(json.dumps(arguments))
    _no_value_in(json.dumps(result.record()))


def test_a_corrupt_gzipped_report_is_reported_unreadable_and_its_siblings_are_still_swept(tmp_path):
    project = tmp_path / 'proj'
    attempt = verify_logs_dir(project) / '6830'
    log = _write(attempt / 'attempt-1.test-Z.log', [f'E   {POOL_A}'])
    corrupt = _write_gzip(
        attempt / 'attempt-1.orchestrator.junit-Z.xml.gz',
        _junit_lines(message=f'leaked {POOL_A}', text=f'E   {POOL_A}'),
    )
    whole = corrupt.read_bytes()
    corrupt.write_bytes(whole[: len(whole) // 2])
    os.utime(corrupt, ns=(OLD_MTIME_NS, OLD_MTIME_NS))
    content = corrupt.read_bytes()
    config = sweep.SweepConfig(trees=sweep.swept_trees([project], tmp_path / 'no-claude'), now=NOW)

    result = sweep.run(config, environ=ENVIRON, post_escalation=Recorder())

    assert result.unreadable == (corrupt,)
    assert corrupt.read_bytes() == content
    assert [hit.path for hit in result.redacted] == [log]
    assert POOL_A.encode() not in log.read_bytes()
    assert result.record()['unreadable'] == [str(corrupt)]


def test_a_plain_log_is_matched_for_the_raw_value_and_a_transcript_only_for_its_json_escape(tmp_path):
    value = 'ab"cd\\ef' + 'Zz' * 15
    project = tmp_path / 'proj'
    log = _write(offline_lane_dir(project) / 'p1' / 'detect.log', [f'E   AssertionError: token={value}'])
    transcript = _write(agent_transcripts_dir(project) / '7001' / '-enc' / 'sid.jsonl', [_line_with(value)])
    raw_in_transcript = _write(agent_transcripts_dir(project) / '7002' / '-enc' / 'sid.jsonl', [f'token={value}'])
    untouched = raw_in_transcript.read_bytes()
    config = sweep.SweepConfig(trees=sweep.swept_trees([project], tmp_path / 'no-claude'), now=NOW)

    result = sweep.run(config, environ={'CLAUDE_OAUTH_TOKEN_B': value}, post_escalation=Recorder())

    assert {hit.path: dict(hit.counts) for hit in result.redacted} == {
        log: {'CLAUDE_OAUTH_TOKEN_B': 1},
        transcript: {'CLAUDE_OAUTH_TOKEN_B': 1},
    }
    assert value.encode() not in log.read_bytes()
    assert raw_in_transcript.read_bytes() == untouched


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
    config = sweep.SweepConfig(trees=sweep.swept_trees([project], tmp_path / 'no-claude'), now=NOW)
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
    config = sweep.SweepConfig(trees=sweep.swept_trees([project], tmp_path / 'no-claude'), now=NOW)

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
    config = sweep.SweepConfig(trees=sweep.swept_trees([project], tmp_path / 'no-claude'), now=NOW)

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


def test_a_file_that_vanishes_before_it_is_read_is_passed_over(tree, monkeypatch):
    """pytest prunes its own old base temps while the sweep walks them."""
    real_redact_in_place = sweep.redact_in_place

    def pruned_before_reading(path, needles, **kwargs):
        if path == tree.pool_subagent:
            path.unlink()
            raise FileNotFoundError(2, 'No such file or directory', str(path))
        return real_redact_in_place(path, needles, **kwargs)

    monkeypatch.setattr(sweep, 'redact_in_place', pruned_before_reading)

    result = sweep.run(tree.config(), environ=ENVIRON, post_escalation=Recorder())

    assert result.unreadable == ()
    assert tree.pool_subagent not in result.deferred
    assert {hit.path for hit in result.redacted} == {tree.pool_main, tree.openai}
    assert result.clean == 2


def test_a_directory_that_vanishes_before_it_is_listed_is_passed_over(tree, monkeypatch):
    real_redact_in_place = sweep.redact_in_place
    pruned = tree.pool_subagent.parent.parent

    def pruned_mid_walk(path, needles, **kwargs):
        if path == tree.pool_main:
            shutil.rmtree(pruned)
        return real_redact_in_place(path, needles, **kwargs)

    monkeypatch.setattr(sweep, 'redact_in_place', pruned_mid_walk)

    result = sweep.run(tree.config(), environ=ENVIRON, post_escalation=Recorder())

    assert not pruned.exists()
    assert result.unreadable == ()
    assert {hit.path for hit in result.redacted} == {tree.pool_main, tree.openai}


def test_a_failed_escalation_is_reported_and_the_redaction_still_stands(tree):
    result = sweep.run(tree.config(), environ=ENVIRON, post_escalation=FailingPoster())

    assert result.ok
    assert result.escalation_error == 'ConnectionError: escalation server unreachable'
    assert POOL_A.encode() not in tree.pool_main.read_bytes()
    assert result.record()['escalation_error'] == result.escalation_error


def _served_tool_calls(server: StubMcpServer) -> list[tuple[str, dict]]:
    return [
        (request.body['params']['name'], request.body['params']['arguments'])
        for request in server.requests
        if request.http_method == 'POST' and request.body.get('method') == 'tools/call'
    ]


class TestMain:
    @pytest.fixture
    def secretless_environ(self, monkeypatch):
        for name in list(os.environ):
            if is_secret_env_name(name):
                monkeypatch.delenv(name)

    def _argv(
        self, tree: Tree | LogTree, tmp_path: Path, *, escalation_url: str = 'http://127.0.0.1:1/mcp'
    ) -> list[str]:
        return [
            *(arg for root in tree.project_roots for arg in ('--project-root', str(root))),
            '--units-dir', str(tmp_path / 'no-units'),
            '--claude-projects-dir', str(tree.claude_projects),
            '--escalation-url', escalation_url,
            '--min-age-secs', '0',
            '--pytest-tmp-dir', str(tmp_path / 'no-pytest'),
        ]

    def _record(self, capsys) -> dict:
        (line,) = capsys.readouterr().out.splitlines()
        return json.loads(line)

    def test_a_sweep_with_nothing_to_redact_exits_0(self, secretless_environ, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_A', 'sk-ant-oat01-' + 'Un' * 17 + 'U')
        tree = Tree(tmp_path)

        assert sweep.main(self._argv(tree, tmp_path)) == 0

        record = self._record(capsys)
        assert record['ok'] is True
        assert record['redacted'] == []
        assert record['escalation_error'] is None

    @pytest.mark.parametrize('endpoint_suffix', ['/mcp', '/mcp/'])  # mcp-url-sweep: allow input the code under test must normalise; the stub asserts it reached /mcp
    def test_a_sweep_that_redacts_files_one_escalation_on_the_configured_server(
        self, secretless_environ, monkeypatch, tmp_path, capsys, endpoint_suffix
    ):
        """--escalation-url is the server's MCP URL, with or without a trailing slash."""
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_A', POOL_A)
        tree = Tree(tmp_path)

        with StubMcpServer(stateful=True, payload={'id': 'esc-1', 'status': 'pending'}) as server:
            assert sweep.main(self._argv(tree, tmp_path, escalation_url=f'{server.url}{endpoint_suffix}')) == 0

        assert self._record(capsys)['escalation_error'] is None
        ((name, arguments),) = _served_tool_calls(server)
        assert name == 'escalate_info'
        assert arguments['task_id'] == arguments['agent_role'] == 'transcript-secret-sweep'
        assert arguments['category'] == 'risk_identified'
        assert POOL_A not in json.dumps(arguments)

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
        assert record['escalation_error'].startswith('McpCallError')
        assert 'the escalation was not filed' in err
        assert POOL_A not in out + err

    def test_a_verify_log_under_a_project_root_is_swept(self, secretless_environ, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_A', POOL_A)
        logs = LogTree(tmp_path)

        with StubMcpServer(stateful=True, payload={'id': 'esc-1', 'status': 'pending'}) as server:
            assert sweep.main(self._argv(logs, tmp_path, escalation_url=f'{server.url}/mcp')) == 0

        assert POOL_A.encode() not in logs.verify_log.read_bytes()
        assert str(logs.verify_log) in {hit['path'] for hit in self._record(capsys)['redacted']}

    def test_a_pytest_tmp_dir_is_swept(self, secretless_environ, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_A', POOL_A)
        tree = Tree(tmp_path)
        dumps = PytestTmpTree(tmp_path)

        with StubMcpServer(stateful=True, payload={'id': 'esc-1', 'status': 'pending'}) as server:
            argv = self._argv(tree, tmp_path, escalation_url=f'{server.url}/mcp')
            assert sweep.main([*argv, '--pytest-tmp-dir', str(dumps.root)]) == 0

        out, err = capsys.readouterr()
        (line,) = out.splitlines()
        assert str(dumps.env_dump) in {hit['path'] for hit in json.loads(line)['redacted']}
        assert POOL_A.encode() not in dumps.env_dump.read_bytes()
        assert POOL_A[:16] not in out + err
