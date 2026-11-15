"""Tests for scripts/locate_session_transcript.py — "where is the transcript of
session <id>" answered without a ``find /`` (task 4637).

THE CONTRACT UNDER TEST IS AGREEMENT WITH THE ARCHIVE'S OWN WRITER AND LOCATOR,
never a remembered layout. Every archived session here is produced by
``shared.transcript_archive.archive_task_transcripts`` and every expected path
is computed by ``shared.transcript_archive.durable_archive_path``, so a layout
change in ``shared/`` moves the fixture and the expectation together instead of
leaving a hand-written ``<task_id>/<enc>/<sid>.jsonl`` copy here to rot.

Every test drives the real CLI as a subprocess, with an explicit ``cwd`` (the
tool resolves the main checkout from where it is standing) and ``HOME`` pointed
at a fake home, so the developer's real ``~/.claude`` is never searched.

COVERAGE — the two chains that run this directory (see
``scripts/tests/test_task_db_scan.py``'s docstring for the near-homograph trap
between ``scripts/tests/`` and ``tests/scripts/``):

    scripts/orchestrator.yaml (test_command)
    dark-factory-orchestrator.yaml (test_command, tail segment)

To run just this suite:

    uv run --project shared pytest scripts/tests/test_locate_session_transcript.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from committed_git_repo import committed_repo
from locate_session_transcript import EXIT_FOUND, EXIT_NO_CHECKOUT, EXIT_NOT_FOUND
from shared.project_layout import agent_transcripts_dir
from shared.transcript_archive import archive_task_transcripts, durable_archive_path
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / 'scripts' / 'locate_session_transcript.py'

TASK_ID = '4242'
SID = 'b976febe-1c2d-4e3f-8a9b-0c1d2e3f4a5b'
SID2 = 'c4a7e2d1-5b6c-4d7e-9f80-a1b2c3d4e5f6'
UNKNOWN_SID = '00000000-1111-4222-8333-444444444444'

_RUN_TIMEOUT_SECS = 60


def _run(cwd: Path, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=str(cwd),
        env={**os.environ, 'HOME': str(home)},
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SECS,
        check=False,
    )


def _describe(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f'rc={result.returncode}\n stdout: {result.stdout!r}\n stderr: {result.stderr!r}'
    )


@pytest.fixture
def fake_home(tmp_path: Path) -> Path:
    home = tmp_path / 'home'
    home.mkdir()
    return home


@pytest.fixture
def main(tmp_path: Path) -> Path:
    return Path(committed_repo(tmp_path / 'checkout', {'README': 'x'})).resolve()


def _linked_worktree(main: Path, name: str = TASK_ID) -> Path:
    dest = main / '.worktrees' / name
    subprocess.run(
        ['git', '-C', str(main), 'worktree', 'add', '-q', '-b', f'task/{name}', str(dest)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return dest


def _archive_session(main: Path, tmp_path: Path, session_id: str = SID) -> Path:
    """Archive *session_id* into *main*'s archive with the real writer; return its path."""
    cfg = tmp_path / 'cfg'
    transcript = cfg / 'projects' / '-tmp-enc' / f'{session_id}.jsonl'
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(json.dumps({'sessionId': session_id}) + '\n', encoding='utf-8')
    archive_root = agent_transcripts_dir(main)
    assert archive_task_transcripts(cfg, TASK_ID, session_id, archive_root=archive_root) == 1
    archived = durable_archive_path(archive_root, TASK_ID, session_id)
    assert archived is not None
    return archived


def test_finds_an_archived_session_from_the_main_checkout(main, tmp_path, fake_home):
    expected = _archive_session(main, tmp_path)

    result = _run(main, fake_home, SID)

    assert result.returncode == EXIT_FOUND, _describe(result)
    assert result.stdout.splitlines() == [str(expected)], _describe(result)


def test_finds_the_main_checkouts_archive_from_a_linked_worktree(main, tmp_path, fake_home):
    """The load-bearing case: a dispatched agent stands in a worktree with no data/."""
    worktree = _linked_worktree(main)
    expected = _archive_session(main, tmp_path)
    assert not (worktree / 'data').exists()

    result = _run(worktree, fake_home, SID)

    assert result.returncode == EXIT_FOUND, _describe(result)
    assert result.stdout.splitlines() == [str(expected)], _describe(result)


def test_a_miss_exits_not_found_and_names_the_archive_root(main, tmp_path, fake_home):
    _archive_session(main, tmp_path)

    result = _run(main, fake_home, UNKNOWN_SID)

    assert result.returncode == EXIT_NOT_FOUND, _describe(result)
    assert result.stdout == '', _describe(result)
    assert UNKNOWN_SID in result.stderr, _describe(result)
    assert str(agent_transcripts_dir(main)) in result.stderr, _describe(result)


def test_an_absent_archive_is_reported_as_absent(main, fake_home):
    assert not (main / 'data').exists()

    result = _run(main, fake_home, SID)

    assert result.returncode == EXIT_NOT_FOUND, _describe(result)
    assert str(agent_transcripts_dir(main)) in result.stderr, _describe(result)
    assert 'absent' in result.stderr, _describe(result)


def test_a_partial_session_id_is_refused_rather_than_silently_missed(main, fake_home):
    result = _run(main, fake_home, 'b976febe')

    assert result.returncode == 2, _describe(result)
    assert 'b976febe' in result.stderr, _describe(result)


def test_outside_any_checkout_exits_no_checkout_unless_project_root_is_given(
    main, tmp_path, fake_home
):
    expected = _archive_session(main, tmp_path)
    outside = tmp_path / 'not-a-checkout'
    outside.mkdir()

    refused = _run(outside, fake_home, SID)

    assert refused.returncode == EXIT_NO_CHECKOUT, _describe(refused)
    assert str(outside.resolve()) in refused.stderr, _describe(refused)

    found = _run(outside, fake_home, SID, '--project-root', str(main))

    assert found.returncode == EXIT_FOUND, _describe(found)
    assert found.stdout.splitlines() == [str(expected)], _describe(found)


def _interactive_session(home: Path, session_id: str = SID2) -> Path:
    """Write *session_id* under *home*'s CLI config, beside a decoy subagent dir."""
    enc = home / '.claude' / 'projects' / '-home-x-proj'
    transcript = enc / f'{session_id}.jsonl'
    decoy = enc / session_id / 'subagents' / 'agent-abc.jsonl'
    decoy.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({'sessionId': session_id}) + '\n', encoding='utf-8')
    decoy.write_text(json.dumps({'sessionId': session_id}) + '\n', encoding='utf-8')
    return transcript


def test_finds_an_interactive_session_under_the_home_cli_config(main, fake_home):
    expected = _interactive_session(fake_home)

    result = _run(main, fake_home, SID2)

    assert result.returncode == EXIT_FOUND, _describe(result)
    assert result.stdout.splitlines() == [str(expected)], _describe(result)


def test_a_miss_names_the_home_cli_projects_root_too(main, fake_home):
    _interactive_session(fake_home)

    result = _run(main, fake_home, UNKNOWN_SID)

    assert result.returncode == EXIT_NOT_FOUND, _describe(result)
    assert str(fake_home / '.claude' / 'projects') in result.stderr, _describe(result)
    assert str(agent_transcripts_dir(main)) in result.stderr, _describe(result)


def test_a_session_in_both_stores_prints_both(main, tmp_path, fake_home):
    archived = _archive_session(main, tmp_path, SID2)
    interactive = _interactive_session(fake_home, SID2)

    result = _run(main, fake_home, SID2)

    assert result.returncode == EXIT_FOUND, _describe(result)
    assert result.stdout.splitlines() == [str(archived), str(interactive)], _describe(result)
