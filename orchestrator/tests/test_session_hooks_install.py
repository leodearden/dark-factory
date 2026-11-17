"""Tests for session_hooks' ``install`` verb: the MERGE-not-clobber settings merge, primary-checkout hook-script resolution, the atomic settings write with backup, and its non-fail-soft exit code."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import pytest

from orchestrator import session_hooks as sh

# ---------------------------------------------------------------------------
# Step-11: merge_hook_settings (MERGE-not-clobber)
# ---------------------------------------------------------------------------


def _real_settings_snapshot() -> dict:
    """A settings dict reproducing the real ~/.claude/settings.json's shape."""
    return {
        'env': {'SOME_VAR': '1'},
        'permissions': {'allow': ['Bash(git *)']},
        'hooks': {
            'PreToolUse': [
                {
                    'matcher': 'Bash',
                    'hooks': [
                        {
                            'type': 'command',
                            'command': '/home/leo/.claude/hooks/some-bash-pretooluse.sh',
                            'timeout': 5,
                        }
                    ],
                },
                {
                    'matcher': 'EnterWorktree',
                    'hooks': [
                        {
                            'type': 'command',
                            'command': '/home/leo/.claude/hooks/worktree-hookspath-capture.sh',
                            'timeout': 10,
                        }
                    ],
                },
            ],
            'PostToolUse': [
                {
                    'matcher': 'ExitWorktree',
                    'hooks': [
                        {
                            'type': 'command',
                            'command': '/home/leo/.claude/hooks/worktree-hookspath-restore.sh',
                            'timeout': 10,
                        }
                    ],
                },
            ],
        },
        'statusLine': {'type': 'command', 'command': 'my-status-line.sh'},
        'enabledPlugins': ['hookify@claude-plugins-official'],
    }


def test_merge_hook_settings_adds_only_the_three_event_keys(tmp_path: Path) -> None:
    before = _real_settings_snapshot()
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    after = sh.merge_hook_settings(before, script_dir)

    hooks = after['hooks']
    assert set(hooks) == {'PreToolUse', 'PostToolUse', 'SessionStart', 'Notification', 'Stop'}
    for event, script_name in (
        ('SessionStart', 'session-start.sh'),
        ('Notification', 'notification.sh'),
        ('Stop', 'stop.sh'),
    ):
        entries = hooks[event]
        assert len(entries) == 1
        leaf = entries[0]['hooks'][0]
        assert leaf['type'] == 'command'
        assert leaf['command'] == str(script_dir / script_name)
        assert leaf['timeout'] == 10


def test_merge_hook_settings_leaves_pretooluse_posttooluse_byte_identical(tmp_path: Path) -> None:
    before = _real_settings_snapshot()
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    after = sh.merge_hook_settings(before, script_dir)

    assert after['hooks']['PreToolUse'] == before['hooks']['PreToolUse']
    assert after['hooks']['PostToolUse'] == before['hooks']['PostToolUse']


def test_merge_hook_settings_leaves_every_other_top_level_key_byte_identical(tmp_path: Path) -> None:
    before = _real_settings_snapshot()
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    after = sh.merge_hook_settings(before, script_dir)

    for key in ('env', 'permissions', 'statusLine', 'enabledPlugins'):
        assert after[key] == before[key]


def test_merge_hook_settings_does_not_mutate_input(tmp_path: Path) -> None:
    before = _real_settings_snapshot()
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    sh.merge_hook_settings(before, script_dir)

    assert before == _real_settings_snapshot()


def test_merge_hook_settings_is_idempotent(tmp_path: Path) -> None:
    before = _real_settings_snapshot()
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    once = sh.merge_hook_settings(before, script_dir)
    twice = sh.merge_hook_settings(once, script_dir)

    assert once == twice


def test_merge_hook_settings_creates_hooks_key_when_absent(tmp_path: Path) -> None:
    before = {'env': {'SOME_VAR': '1'}}
    script_dir = tmp_path / 'skills' / 'spawn' / 'hooks'

    after = sh.merge_hook_settings(before, script_dir)

    assert set(after['hooks']) == {'SessionStart', 'Notification', 'Stop'}
    assert after['env'] == before['env']


# ---------------------------------------------------------------------------
# _hooks_script_dir: resolves to the PRIMARY checkout, not a linked worktree
# ---------------------------------------------------------------------------


def _init_primary_checkout_with_hooks(tmp_path: Path) -> Path:
    """Create a temp git checkout under *tmp_path* with a real hooks trio, committed."""
    primary = tmp_path / 'primary'
    primary.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=primary, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=primary, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=primary, check=True)
    hooks_dir = primary / 'skills' / 'spawn' / 'hooks'
    hooks_dir.mkdir(parents=True)
    for script_name in ('session-start.sh', 'notification.sh', 'stop.sh'):
        (hooks_dir / script_name).write_text('#!/bin/sh\n')
    (primary / 'README.md').write_text('primary checkout\n')
    subprocess.run(['git', 'add', '-A'], cwd=primary, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'init'], cwd=primary, check=True)
    return primary


def _init_primary_checkout_with_partial_hooks(tmp_path: Path) -> Path:
    """Like ``_init_primary_checkout_with_hooks`` but deliberately omits one
    script, so the resolved candidate dir exists (git succeeds) yet fails the
    ``all(...is_file())`` completeness guard in ``_hooks_script_dir``."""
    primary = tmp_path / 'primary'
    primary.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=primary, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=primary, check=True)
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=primary, check=True)
    hooks_dir = primary / 'skills' / 'spawn' / 'hooks'
    hooks_dir.mkdir(parents=True)
    # Deliberately omit stop.sh.
    for script_name in ('session-start.sh', 'notification.sh'):
        (hooks_dir / script_name).write_text('#!/bin/sh\n')
    (primary / 'README.md').write_text('primary checkout with partial hooks\n')
    subprocess.run(['git', 'add', '-A'], cwd=primary, check=True)
    subprocess.run(['git', 'commit', '-q', '-m', 'init'], cwd=primary, check=True)
    return primary


def _fake_session_hooks_path(root: Path) -> Path:
    """Mirror the real repo layout (<root>/orchestrator/src/orchestrator/session_hooks.py)."""
    fake_file = root / 'orchestrator' / 'src' / 'orchestrator' / 'session_hooks.py'
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text('# stub for _hooks_script_dir test\n')
    return fake_file


def test_hooks_script_dir_resolves_to_primary_checkout_from_linked_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _init_primary_checkout_with_hooks(tmp_path)
    worktree = tmp_path / 'worktree-2288'
    subprocess.run(
        ['git', 'worktree', 'add', '-q', str(worktree), '-b', 'wt-2288'],
        cwd=primary,
        check=True,
    )
    fake_file = _fake_session_hooks_path(worktree)
    monkeypatch.setattr(sh, '__file__', str(fake_file))

    resolved = sh._hooks_script_dir()

    assert resolved == (primary / 'skills' / 'spawn' / 'hooks').resolve()
    assert resolved != fake_file.resolve().parents[3] / 'skills' / 'spawn' / 'hooks'


def test_hooks_script_dir_falls_back_when_candidate_missing_a_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """git resolves fine and the candidate dir exists, but it is missing one
    of the three hook scripts -- the ``all(...is_file())`` guard must reject
    it and fall back to parents[3] rather than returning the incomplete
    candidate (e.g. an installed/partial checkout where skills/spawn/hooks is
    stale). Uses a linked worktree, like the resolves-to-primary-checkout
    test above, so the (rejected) candidate and the fallback are observably
    different directories -- otherwise a regression that wrongly accepted the
    incomplete candidate would go unnoticed, since both would resolve to the
    same path when the fake file lives directly under the primary checkout."""
    primary = _init_primary_checkout_with_partial_hooks(tmp_path)
    worktree = tmp_path / 'worktree-partial'
    subprocess.run(
        ['git', 'worktree', 'add', '-q', str(worktree), '-b', 'wt-partial'],
        cwd=primary,
        check=True,
    )
    fake_file = _fake_session_hooks_path(worktree)
    monkeypatch.setattr(sh, '__file__', str(fake_file))

    resolved = sh._hooks_script_dir()

    assert resolved == fake_file.resolve().parents[3] / 'skills' / 'spawn' / 'hooks'
    assert resolved != (primary / 'skills' / 'spawn' / 'hooks').resolve()


def test_hooks_script_dir_falls_back_to_parents3_when_not_a_git_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_file = _fake_session_hooks_path(tmp_path)
    monkeypatch.setattr(sh, '__file__', str(fake_file))
    # Hermetic regardless of ambient git topology: cap git's upward .git
    # search at tmp_path so this test's "not a git checkout" premise holds
    # even if /tmp (or some ancestor) happens to sit inside a real repo.
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))

    resolved = sh._hooks_script_dir()

    assert resolved == fake_file.resolve().parents[3] / 'skills' / 'spawn' / 'hooks'


def test_hooks_script_dir_falls_back_when_git_common_dir_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct fallback path from the not-a-git-checkout case above: git
    itself succeeds (returncode 0) but reports an empty/blank
    --git-common-dir, which must also degrade to the parents[3] fallback."""
    fake_file = _fake_session_hooks_path(tmp_path)
    monkeypatch.setattr(sh, '__file__', str(fake_file))

    def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, returncode=0, stdout='   \n', stderr='')

    monkeypatch.setattr(subprocess, 'run', _fake_run)

    resolved = sh._hooks_script_dir()

    assert resolved == fake_file.resolve().parents[3] / 'skills' / 'spawn' / 'hooks'


# ---------------------------------------------------------------------------
# _run_install: missing-file create, backup-on-existing, atomic JSON write
# ---------------------------------------------------------------------------


def test_run_install_creates_settings_file_when_missing(tmp_path: Path) -> None:
    settings_path = tmp_path / 'settings.json'

    sh._run_install(settings_path=settings_path)

    assert settings_path.is_file()
    written = json.loads(settings_path.read_text())
    assert set(written['hooks']) == {'SessionStart', 'Notification', 'Stop'}
    assert list(tmp_path.glob('settings.json.*.bak')) == []


def test_run_install_backs_up_pre_merge_bytes_when_file_exists(tmp_path: Path) -> None:
    settings_path = tmp_path / 'settings.json'
    before = {'env': {'SOME_VAR': '1'}, 'hooks': {'PreToolUse': []}}
    before_raw = json.dumps(before)
    settings_path.write_text(before_raw)

    sh._run_install(settings_path=settings_path)

    backups = list(tmp_path.glob('settings.json.*.bak'))
    assert len(backups) == 1
    assert backups[0].read_text() == before_raw

    merged = json.loads(settings_path.read_text())
    assert merged['env'] == before['env']
    assert set(merged['hooks']) == {'PreToolUse', 'SessionStart', 'Notification', 'Stop'}


def test_run_install_result_is_valid_json(tmp_path: Path) -> None:
    settings_path = tmp_path / 'settings.json'
    settings_path.write_text(json.dumps({'env': {}}))

    sh._run_install(settings_path=settings_path)

    # json.loads raises on malformed content -- reaching the assertion below
    # at all is the real assertion here.
    reloaded = json.loads(settings_path.read_text())
    assert reloaded['env'] == {}


# ---------------------------------------------------------------------------
# main(['install', ...]): exit-code propagation (install is NOT fail-soft)
# ---------------------------------------------------------------------------


def test_main_install_succeeds_and_returns_zero(tmp_path: Path) -> None:
    settings_path = tmp_path / 'settings.json'

    rc = sh.main(['install', '--settings-path', str(settings_path)])

    assert rc == 0
    assert settings_path.is_file()


def test_main_install_returns_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings_path = tmp_path / 'settings.json'

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sh, '_run_install', _boom)

    with caplog.at_level(logging.ERROR):
        rc = sh.main(['install', '--settings-path', str(settings_path)])

    assert rc == 1
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
