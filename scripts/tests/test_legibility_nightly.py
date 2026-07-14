"""Tests for scripts/legibility/nightly.py — the nightly trickle pipeline
assembly (PRD task epsilon: plans/confusion-reduction-prd.md §5.5, decisions
7/8, boundary test §8.8).

Every stage is exercised behind a dependency-injection seam (invoke for the
LLM, status_fetcher for the census, poster for escalation, committer for
git) plus pytest-monkeypatched module functions — no test here ever touches
a real LLM, the machine-operated main git checkout, a live escalation
server, or real systemd. The one true end-to-end test
(test_run_nightly_happy_path_end_to_end) uses a real temp `git init` repo so
the commit path is genuinely exercised without risking main.
"""
from __future__ import annotations

import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from legibility import census_trigger, codebook, nightly
from legibility.config import load_config


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _write_config(
    root: Path, *, project_id: str, escalation_port: int = 8199, cwd_prefixes=None,
) -> Path:
    """Write a minimal valid docs/legibility/legibility.yaml under *root*."""
    cwd_prefixes = cwd_prefixes if cwd_prefixes is not None else [str(root / "work")]
    legibility_dir = root / "docs" / "legibility"
    legibility_dir.mkdir(parents=True, exist_ok=True)
    config_path = legibility_dir / "legibility.yaml"
    lines = [
        f"project_id: {project_id}",
        f"project_root: {root}",
        f"escalation_port: {escalation_port}",
        "cwd_prefixes:",
    ]
    lines += [f"  - {prefix}" for prefix in cwd_prefixes]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _encode_cwd(cwd: str) -> str:
    """Mirror inventory.encode_cwd (kept independent -- this is fixture code,
    not a reuse of the module under test)."""
    return cwd.replace('/', '-').replace('.', '-')


def _write_transcript(
    path: Path, *, cwd: str, timestamp: str, session_id: str = 'session-1',
) -> None:
    """Write a minimal fixture session transcript JSONL with a genuine user
    turn plus a structural tool-error (nonzero confusion signal): a user
    turn, an assistant tool_use, and a user tool_result carrying
    ``is_error: true`` with 'No such file or directory' content (matches
    sampling._NOT_FOUND_PATTERNS too, for a score > 0 on more than one
    signal class)."""
    lines = [
        {
            'type': 'user',
            'cwd': cwd,
            'timestamp': timestamp,
            'sessionId': session_id,
            'message': {'role': 'user', 'content': 'please fix this confusing bug'},
        },
        {
            'type': 'assistant',
            'cwd': cwd,
            'timestamp': timestamp,
            'sessionId': session_id,
            'message': {
                'role': 'assistant',
                'content': [
                    {'type': 'tool_use', 'id': 'tool-1', 'name': 'Bash',
                     'input': {'command': 'cat missing-file'}},
                ],
            },
        },
        {
            'type': 'user',
            'cwd': cwd,
            'timestamp': timestamp,
            'sessionId': session_id,
            'message': {
                'role': 'user',
                'content': [
                    {'type': 'tool_result', 'tool_use_id': 'tool-1', 'is_error': True,
                     'content': 'No such file or directory'},
                ],
            },
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(line) for line in lines) + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# step-3/4: select_scored_records (+ the stratified sampler)
# ---------------------------------------------------------------------------

def test_select_scored_records_includes_matching_session(tmp_path):
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z')

    scored = nightly.select_scored_records(cfg, projects_root, date(2026, 7, 13))

    assert len(scored) == 1
    record = scored[0]
    assert record.session.path == session_path
    assert record.score > 0
    assert record.stratum


def test_select_scored_records_excludes_out_of_scope_sessions(tmp_path):
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'

    # Wrong cwd -- outside cfg.cwd_prefixes.
    other_cwd = str(tmp_path / 'elsewhere')
    _write_transcript(
        projects_root / _encode_cwd(other_cwd) / 'session-2.jsonl',
        cwd=other_cwd, timestamp='2026-07-13T10:00:00Z',
    )

    # Right cwd, wrong date.
    _write_transcript(
        projects_root / _encode_cwd(work_cwd) / 'session-3.jsonl',
        cwd=work_cwd, timestamp='2026-07-01T10:00:00Z',
    )

    scored = nightly.select_scored_records(cfg, projects_root, date(2026, 7, 13))

    assert scored == []


# ---------------------------------------------------------------------------
# step-5/6: build_digests
# ---------------------------------------------------------------------------

def test_build_digests_happy_path(tmp_path):
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z')
    selected = nightly.select_scored_records(cfg, projects_root, date(2026, 7, 13))

    digests, extractor_failures = nightly.build_digests(selected)

    assert extractor_failures == []
    assert len(digests) == 1
    assert digests[0].startswith('---\n')


def test_build_digests_isolates_extractor_crash(tmp_path):
    work_cwd = str(tmp_path / 'work')
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', cwd_prefixes=[work_cwd]))
    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    _write_transcript(session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z')
    selected = nightly.select_scored_records(cfg, projects_root, date(2026, 7, 13))

    def _crashing_build(path, *, agent_class_override=None, max_bytes=15360):
        raise ValueError('boom: corrupt transcript')

    digests, extractor_failures = nightly.build_digests(selected, build=_crashing_build)

    assert digests == []
    assert len(extractor_failures) == 1
    session_name, reason = extractor_failures[0]
    assert session_name == session_path.name
    assert 'boom' in reason


# ---------------------------------------------------------------------------
# step-7/8: git helpers (_git_status_changed, _git_commit_docs_only) --
# against a REAL tmp `git init` repo, plus a fake-runner ref-lock retry test.
# ---------------------------------------------------------------------------

_CODEBOOK_RELPATH = Path('docs') / 'legibility' / 'confusion-codebook.yaml'


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', '-b', 'main'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True)
    docs = repo / 'docs' / 'legibility'
    docs.mkdir(parents=True)
    (docs / 'confusion-codebook.yaml').write_text('version: 2\nentries: []\n', encoding='utf-8')
    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'initial'], cwd=repo, check=True)
    return repo


def test_git_status_changed_detects_dirty_codebook(tmp_path):
    repo = _init_repo(tmp_path)

    assert nightly._git_status_changed(repo, _CODEBOOK_RELPATH) is False

    (repo / _CODEBOOK_RELPATH).write_text(
        'version: 2\nentries: []\ncandidates: []\n', encoding='utf-8',
    )

    assert nightly._git_status_changed(repo, _CODEBOOK_RELPATH) is True


def test_git_commit_docs_only_commits_only_that_path(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / _CODEBOOK_RELPATH).write_text(
        'version: 2\nentries: []\ncandidates: []\n', encoding='utf-8',
    )
    (repo / 'scratch.txt').write_text('unrelated wip\n', encoding='utf-8')

    result = nightly._git_commit_docs_only(repo, [_CODEBOOK_RELPATH], 'legibility: nightly sightings')

    assert result.ok is True
    assert result.sha

    status = subprocess.run(
        ['git', 'status', '--porcelain'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout
    assert 'scratch.txt' in status
    assert 'confusion-codebook.yaml' not in status

    shown = subprocess.run(
        ['git', 'show', '--stat', '--format=', 'HEAD'],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout
    assert 'confusion-codebook.yaml' in shown
    assert 'scratch.txt' not in shown


def test_git_commit_docs_only_noop_returns_not_ok(tmp_path):
    repo = _init_repo(tmp_path)

    result = nightly._git_commit_docs_only(repo, [_CODEBOOK_RELPATH], 'legibility: no-op')

    assert result.ok is False


def test_git_commit_docs_only_retries_on_ref_lock_then_succeeds(tmp_path):
    calls = {'commit_attempts': 0}

    def fake_runner(args, **kwargs):
        if 'commit' in args:
            calls['commit_attempts'] += 1
            if calls['commit_attempts'] < 3:
                return subprocess.CompletedProcess(args, 1, stdout='', stderr='fatal: cannot lock ref ...')
            return subprocess.CompletedProcess(args, 0, stdout='', stderr='')
        if 'rev-parse' in args:
            return subprocess.CompletedProcess(args, 0, stdout='deadbeef\n', stderr='')
        raise AssertionError(f'unexpected git invocation: {args}')

    result = nightly._git_commit_docs_only(
        tmp_path, [_CODEBOOK_RELPATH], 'msg', runner=fake_runner, retries=5, backoff=0,
    )

    assert result.ok is True
    assert result.sha == 'deadbeef'
    assert calls['commit_attempts'] == 3


def test_git_commit_docs_only_gives_up_on_persistent_lock_error(tmp_path):
    def fake_runner(args, **kwargs):
        if 'commit' in args:
            return subprocess.CompletedProcess(args, 1, stdout='', stderr='fatal: cannot lock ref ...')
        raise AssertionError(f'unexpected git invocation: {args}')

    result = nightly._git_commit_docs_only(
        tmp_path, [_CODEBOOK_RELPATH], 'msg', runner=fake_runner, retries=3, backoff=0,
    )

    assert result.ok is False


def test_git_commit_docs_only_never_invokes_stash(tmp_path, monkeypatch):
    calls = []
    real_run = subprocess.run

    def spy_run(args, **kwargs):
        calls.append(args)
        return real_run(args, **kwargs)

    monkeypatch.setattr(nightly.subprocess, 'run', spy_run)

    repo = _init_repo(tmp_path)
    (repo / _CODEBOOK_RELPATH).write_text(
        'version: 2\nentries: []\ncandidates: []\n', encoding='utf-8',
    )

    nightly._git_commit_docs_only(repo, [_CODEBOOK_RELPATH], 'msg')

    assert not any('stash' in call for call in calls)


# ---------------------------------------------------------------------------
# step-9/10: escalation (_build_escalation_arguments, post_escalation)
# ---------------------------------------------------------------------------

def test_build_escalation_arguments_shape(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    arguments = nightly._build_escalation_arguments(cfg, 'summary text', 'detail text')

    assert arguments == {
        'task_id': 'legibility-trickle-proj_a',
        'agent_role': 'legibility-trickle',
        'category': 'infra_issue',
        'severity': 'info',
        'summary': 'summary text',
        'detail': 'detail text',
    }


def test_post_escalation_calls_poster_with_mcp_envelope(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a', escalation_port=8199))
    calls = []

    def fake_poster(url, envelope):
        calls.append((url, envelope))

    ok = nightly.post_escalation(cfg, 'summary text', 'detail text', poster=fake_poster)

    assert ok is True
    assert len(calls) == 1
    url, envelope = calls[0]
    assert url == 'http://localhost:8199/mcp'
    assert envelope['method'] == 'tools/call'
    assert envelope['params']['name'] == 'escalate_info'
    assert envelope['params']['arguments'] == {
        'task_id': 'legibility-trickle-proj_a',
        'agent_role': 'legibility-trickle',
        'category': 'infra_issue',
        'severity': 'info',
        'summary': 'summary text',
        'detail': 'detail text',
    }


def test_post_escalation_is_best_effort_on_poster_failure(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def raising_poster(url, envelope):
        raise RuntimeError('escalation server unreachable')

    ok = nightly.post_escalation(cfg, 'summary text', 'detail text', poster=raising_poster)

    assert ok is False


# ---------------------------------------------------------------------------
# step-11/12: evaluate_census_step
# ---------------------------------------------------------------------------

def test_evaluate_census_step_no_fire_never_launches(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        return census_trigger.Decision(fire=False, reasons=['max-interval: 1.0d (threshold 10d)'])

    launcher_calls = []
    line, fire = nightly.evaluate_census_step(
        cfg, now=None, status_fetcher=None, decide=fake_decide,
        entrypoint_exists=lambda: True, launcher=lambda: launcher_calls.append(1),
    )

    assert fire is False
    assert 'NO-FIRE' in line
    assert launcher_calls == []


def test_evaluate_census_step_fire_without_entrypoint_logs_loud_no_launch(tmp_path, caplog):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        return census_trigger.Decision(fire=True, reasons=['max-interval: 11.0d -> FIRE'])

    launcher_calls = []
    with caplog.at_level('WARNING'):
        line, fire = nightly.evaluate_census_step(
            cfg, now=None, status_fetcher=None, decide=fake_decide,
            entrypoint_exists=lambda: False, launcher=lambda: launcher_calls.append(1),
        )

    assert fire is True
    assert 'FIRE' in line
    assert launcher_calls == []
    assert any('FIRE-WITHOUT-LAUNCH' in record.message for record in caplog.records)


def test_evaluate_census_step_fire_with_entrypoint_launches(tmp_path):
    cfg = load_config(_write_config(tmp_path, project_id='proj_a'))

    def fake_decide(project_root, *, now=None, status_fetcher=None):
        return census_trigger.Decision(fire=True, reasons=['max-interval: 11.0d -> FIRE'])

    launcher_calls = []
    line, fire = nightly.evaluate_census_step(
        cfg, now=None, status_fetcher=None, decide=fake_decide,
        entrypoint_exists=lambda: True, launcher=lambda: launcher_calls.append(1),
    )

    assert fire is True
    assert launcher_calls == [1]


# ---------------------------------------------------------------------------
# step-1/2: resolve_config_path
# ---------------------------------------------------------------------------

def test_resolve_config_path_matches_by_project_id_field(tmp_path):
    config_a = _write_config(tmp_path / "proj-a", project_id="proj_a")
    _write_config(tmp_path / "proj-b", project_id="proj_b")

    resolved = nightly.resolve_config_path("proj_a", search_roots=[tmp_path])

    assert resolved == config_a


def test_resolve_config_path_unknown_id_raises(tmp_path):
    _write_config(tmp_path / "proj-a", project_id="proj_a")

    with pytest.raises(FileNotFoundError):
        nightly.resolve_config_path("does-not-exist", search_roots=[tmp_path])


def test_resolve_config_path_env_override(tmp_path, monkeypatch):
    config_a = _write_config(tmp_path / "proj-a", project_id="proj_a")
    _write_config(tmp_path / "proj-b", project_id="proj_b")
    monkeypatch.setenv("LEGIBILITY_SEARCH_ROOTS", str(tmp_path))

    resolved = nightly.resolve_config_path("proj_a")

    assert resolved == config_a


# ---------------------------------------------------------------------------
# step-13/14: run_nightly -- end-to-end happy path (real tmp git repo)
# ---------------------------------------------------------------------------

def _write_known_cause_codebook(repo: Path) -> Path:
    """Write a v2 confusion-codebook.yaml under *repo* with one entry,
    id='known-cause', sightings: [] -- the pre-existing codebook state for
    the end-to-end fixture (step-13)."""
    codebook_path = repo / 'docs' / 'legibility' / 'confusion-codebook.yaml'
    cb = {
        'version': 2,
        'entries': [
            {
                'id': 'known-cause',
                'title': 'Known Cause',
                'severity': 'medium',
                'status': 'open',
                'origin_phase': 'implement',
                'manifested_phase': 'implement',
                'sightings': [],
            },
        ],
        'candidates': [],
    }
    codebook.dump(cb, codebook_path)
    return codebook_path


def _init_e2e_repo(tmp_path: Path, *, work_cwd: str) -> tuple[Path, Path]:
    """Build a tmp git repo with a committed legibility.yaml (project_id=
    'testproj') + a v2 codebook carrying one entry (id='known-cause',
    sightings: []), tree clean -- the end-to-end fixture (step-13)."""
    repo = tmp_path / 'e2e-repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', '-b', 'main'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=repo, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True)

    config_path = _write_config(
        repo, project_id='testproj', escalation_port=8199, cwd_prefixes=[work_cwd],
    )
    _write_known_cause_codebook(repo)

    subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'initial'], cwd=repo, check=True)
    return repo, config_path


def _fake_invoke_known_cause(prompt: str, model: str) -> str:
    return json.dumps({
        'matches': [{
            'entry_id': 'known-cause',
            'origin_phase': 'implement',
            'manifested_phase': 'implement',
            'note': 'matched the known cause',
        }],
        'candidates': [],
    })


def test_run_nightly_happy_path_end_to_end(tmp_path):
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=timezone.utc)
    escalation_calls = []

    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_known_cause,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result.exit_code == 0
    assert result.commit_made is True
    assert result.applied == 1
    assert result.coder_status == 'ok'
    assert result.census_line is not None and 'NO-FIRE' in result.census_line
    assert escalation_calls == []

    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert len(after_log) == len(before_log) + 1

    shown = subprocess.run(
        ['git', 'show', '--stat', '--format=', 'HEAD'],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout
    assert 'confusion-codebook.yaml' in shown
    touched_files = [line for line in shown.splitlines() if '|' in line]
    assert len(touched_files) == 1

    committed = codebook.load(repo / 'docs' / 'legibility' / 'confusion-codebook.yaml')
    entry = next(e for e in committed['entries'] if e['id'] == 'known-cause')
    assert len(entry['sightings']) == 1
    sighting = entry['sightings'][0]
    assert sighting['date'] == '2026-07-13'
    assert sighting['project'] == 'testproj'
    assert sighting['session'] == 'session-1'

    # Idempotency: a second run makes no new commit and leaves exactly one sighting.
    result_2 = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_known_cause,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result_2.exit_code == 0
    assert result_2.commit_made is False
    assert result_2.applied == 0
    assert escalation_calls == []

    after_log_2 = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert after_log_2 == after_log

    committed_2 = codebook.load(repo / 'docs' / 'legibility' / 'confusion-codebook.yaml')
    entry_2 = next(e for e in committed_2['entries'] if e['id'] == 'known-cause')
    assert len(entry_2['sightings']) == 1


# ---------------------------------------------------------------------------
# step-15/16: run_nightly -- fail-loud on coder storm (decision 8, §8.6)
# ---------------------------------------------------------------------------

def _fake_invoke_unparseable(prompt: str, model: str) -> str:
    return 'not valid json at all'


def test_run_nightly_fail_loud_on_coder_storm(tmp_path):
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    codebook_path = repo / 'docs' / 'legibility' / 'confusion-codebook.yaml'
    before_bytes = codebook_path.read_bytes()
    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=timezone.utc)
    escalation_calls = []

    # A single selected session whose invoke fails to parse -> 1/1 failed ->
    # failed/total (1.0) strictly exceeds 0.5 -> coder.code_digests reports
    # status="failure" (the storm threshold), even with just one digest.
    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_fake_invoke_unparseable,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result.exit_code != 0
    assert result.coder_status == 'failure'
    assert result.commit_made is False
    assert result.escalated is True

    assert len(escalation_calls) == 1
    url, envelope = escalation_calls[0]
    arguments = envelope['params']['arguments']
    assert arguments['category'] == 'infra_issue'
    assert 'storm' in arguments['summary'].lower()
    assert '1/1' in arguments['summary']

    # Merge/dump/commit skipped entirely: codebook untouched, no new commit.
    assert codebook_path.read_bytes() == before_bytes
    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert after_log == before_log


# ---------------------------------------------------------------------------
# step-17/18: run_nightly -- fail-loud on extractor crash (decision 8)
# ---------------------------------------------------------------------------

def _crashing_build_digests(selected, **kwargs):
    """A fake build_digests: one digest builds fine, one session crashes --
    extractor_failures is non-empty even though a usable digest also
    exists, so the RED test can prove the coder is never reached."""
    ok_digest = (
        '---\n'
        'session: "ok-session"\n'
        'date: "2026-07-13"\n'
        'agent_class: "interactive"\n'
        '---\n'
        'body text\n'
    )
    return [ok_digest], [('session-1.jsonl', 'boom: corrupt transcript')]


def test_run_nightly_fail_loud_on_extractor_crash(tmp_path, monkeypatch):
    work_cwd = str(tmp_path / 'work')
    repo, config_path = _init_e2e_repo(tmp_path, work_cwd=work_cwd)

    projects_root = tmp_path / 'projects'
    session_path = projects_root / _encode_cwd(work_cwd) / 'session-1.jsonl'
    target_date = date(2026, 7, 13)
    _write_transcript(
        session_path, cwd=work_cwd, timestamp='2026-07-13T10:00:00Z', session_id='session-1',
    )

    codebook_path = repo / 'docs' / 'legibility' / 'confusion-codebook.yaml'
    before_bytes = codebook_path.read_bytes()
    before_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()

    monkeypatch.setattr(nightly, 'build_digests', _crashing_build_digests)

    coder_calls = []

    def _unexpected_invoke(prompt, model):
        coder_calls.append(prompt)
        return 'not valid json'

    fixed_now = datetime(2026, 7, 14, 3, 0, 0, tzinfo=timezone.utc)
    escalation_calls = []

    result = nightly.run_nightly(
        config_path=config_path,
        projects_root=projects_root,
        target_date=target_date,
        now=fixed_now,
        invoke=_unexpected_invoke,
        status_fetcher=None,
        poster=lambda url, envelope: escalation_calls.append((url, envelope)),
    )

    assert result.exit_code != 0
    assert result.escalated is True
    assert coder_calls == []  # the coder must never be reached after a crash

    assert len(escalation_calls) == 1
    url, envelope = escalation_calls[0]
    arguments = envelope['params']['arguments']
    assert arguments['category'] == 'infra_issue'
    assert 'extractor' in arguments['summary'].lower()
    assert '1' in arguments['summary']

    # Merge/dump/commit skipped entirely: codebook untouched, no new commit.
    assert codebook_path.read_bytes() == before_bytes
    after_log = subprocess.run(
        ['git', 'log', '--oneline'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert after_log == before_log
