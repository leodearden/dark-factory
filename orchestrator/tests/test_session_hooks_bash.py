"""Tests that run the real ``skills/spawn/hooks/*.sh`` entrypoints end-to-end: PYTHONPATH wiring, stdin passthrough, fail-soft exit, and CLAUDE_SPAWN_SESSION_ID convergence through the shell."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from _process_tree_fakes import claude_process_tree
from _session_hook_fixtures import assert_hook_env_isolated, isolate_hook_env, owner_ppid_env

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOKS_DIR = _REPO_ROOT / 'skills' / 'spawn' / 'hooks'


@pytest.fixture(autouse=True)
def _isolate_hook_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_hook_env(monkeypatch, tmp_path)


def test_hook_env_is_isolated_for_this_module(tmp_path: Path) -> None:
    assert_hook_env_isolated(tmp_path)


# ---------------------------------------------------------------------------
# Step-13: bash-level integration test (skills/spawn/hooks/*.sh entrypoints)
# ---------------------------------------------------------------------------


def _run_hook_script(
    script_name: str,
    hook_input: Mapping[str, object],
    root: Path,
    *,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env['CLAUDE_FLEET_ROOT'] = str(root)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(_HOOKS_DIR / script_name)],
        input=json.dumps(hook_input).encode(),
        env=env,
        capture_output=True,
        timeout=30,
    )


def test_session_start_sh_then_notification_sh_flip_status(tmp_path: Path) -> None:
    # Proves the PYTHONPATH=orchestrator/src wiring, stdin passthrough to
    # python, and fail-soft exit for the real bash entrypoints end-to-end.
    hook_input = {'session_id': 'sess-bash-1', 'cwd': '/home/leo/src/dark-factory'}

    start_result = _run_hook_script('session-start.sh', hook_input, tmp_path)
    assert start_result.returncode == 0

    slug = sh.hook_session_slug(hook_input, env={})
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING

    notification_result = _run_hook_script('notification.sh', hook_input, tmp_path)
    assert notification_result.returncode == 0

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.AWAITING_INPUT


# ---------------------------------------------------------------------------
# Task 2511 step-3: bash-level CLAUDE_SPAWN_SESSION_ID convergence
# ---------------------------------------------------------------------------


def test_session_start_sh_converges_onto_pre_written_launching_record(tmp_path: Path) -> None:
    # Proves CLAUDE_SPAWN_SESSION_ID propagates through the real .sh -> python
    # wiring (not just the pure-python run_session_start path exercised
    # above): a pre-written LAUNCHING record at slug S, keyed the way
    # spawn-claude.sh's own `launching` write keys it, must advance to
    # RUNNING via the real session-start.sh entrypoint -- with a DIFFERENT
    # Claude Code session_id -- when CLAUDE_SPAWN_SESSION_ID=S is in env.
    slug = 'session-cockpit-3215099'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=3215099),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-bash-convergence', 'cwd': '/home/leo/src/dark-factory'}

    result = _run_hook_script(
        'session-start.sh',
        hook_input,
        tmp_path,
        extra_env={'CLAUDE_SPAWN_SESSION_ID': slug},
    )

    assert result.returncode == 0
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING
    # No separate uuid-keyed duplicate.
    dirs = list(sr.sessions_dir(root=tmp_path).iterdir())
    assert len(dirs) == 1
    assert dirs[0].name == slug


def test_nested_compact_cannot_invert_ownership(tmp_path: Path) -> None:
    # Automatic compaction fires SessionStart source='compact' with NO user
    # action, so any nested `claude -p` that runs long enough reaches the
    # remint path. Source alone proves the emitter holds *a* session, not
    # *this* one -- so without the pid condition a nested compact REBOUND the
    # spawner's record to itself and forced the true owner to fork, i.e. a
    # full ownership inversion plus the task-2511 split, reached by routine
    # agent behaviour with no hook fault.
    slug = 'session-cockpit-4400010'
    env = owner_ppid_env(slug, 4400500)
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=4400010,
            result_file='/tmp/result-4400010.md',
        ),
        root=tmp_path,
    )
    # The owner's own SessionStart: direct child of the exported payload
    # bash, so it may claim the record.
    owner = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory'}
    sh.run_session_start(owner, env, root=tmp_path, process_tree=claude_process_tree(4400501, 4400500))
    assert sr.read_record(slug, root=tmp_path).claude_owner_pid is not None

    # A DIFFERENT process (the nested claude) presenting an auto-compact.
    nested_tree = claude_process_tree(999_001, 4400500)
    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory', 'source': 'compact'}
    sh.run_session_start(nested, env, root=tmp_path, process_tree=nested_tree)

    spawner = sr.read_record(slug, root=tmp_path)
    assert spawner.claude_session_id == 'uuid-owner'      # NOT rebound
    assert spawner.result_file == '/tmp/result-4400010.md'
    forked_slug = sh.hook_session_slug(nested, env, root=tmp_path, process_tree=nested_tree)
    forked = sr.read_record(forked_slug, root=tmp_path)
    assert forked.claude_session_id == 'uuid-nested'


def test_owner_auto_compact_still_converges_on_its_own_record(
    tmp_path: Path,
) -> None:
    # The other direction, and the reason 'compact' is not simply dropped
    # from the owner-only set: auto-compaction is routine, so forking on it
    # would re-introduce the task-2511 split for the OWNER on a far more
    # common event than the inversion it would prevent. Same process => same
    # pid => the re-mint is forgiven and the record stays converged.
    slug = 'session-cockpit-4400011'
    env = owner_ppid_env(slug, 4400510)
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=4400011),
        root=tmp_path,
    )
    tree = claude_process_tree(4400511, 4400510)
    owner = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory'}
    sh.run_session_start(owner, env, root=tmp_path, process_tree=tree)

    remint = {'session_id': 'uuid-owner-2', 'cwd': '/home/leo/src/dark-factory', 'source': 'compact'}
    sh.run_session_start(remint, env, root=tmp_path, process_tree=tree)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-owner-2'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_legacy_record_without_owner_pid_adopts_rather_than_splitting_the_owner(
    tmp_path: Path,
) -> None:
    # A record bound before claude_owner_pid existed cannot prove ownership
    # EITHER WAY. Task 4193 L2 ruling item 4-ii: unprovable is not disproved,
    # so it must take the fail-soft direction this module documents at
    # _env_slug_is_owned ("every failure mode resolves to adopt, never to
    # fork"). Forking here would split the OWNER's own record on every
    # routine automatic compaction wherever /proc is unavailable (macOS) --
    # a universal regression, strictly worse than the rare inversion it
    # would prevent. The fork is reserved for a pid that RESOLVES and
    # mismatches (see the test below).
    slug = 'session-cockpit-4400012'
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.RUNNING, launcher_pid=4400012,
            claude_session_id='uuid-owner',  # bound, but no owner pid
        ),
        root=tmp_path,
    )
    remint = {'session_id': 'uuid-other', 'cwd': '/home/leo/src/dark-factory', 'source': 'compact'}
    sh.run_session_start(remint, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-other'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_unresolvable_current_owner_pid_adopts_rather_than_splitting_the_owner(
    tmp_path: Path,
) -> None:
    # The mirror of the case above, and the one the pre-existing pin could
    # never observe (both _owning_claude_pid() calls happened inside one
    # pytest process): the record HAS an owner pid, but the CURRENT probe
    # cannot resolve one -- exactly macOS, where _parent_pid_of has no /proc
    # to read. Unprovable again means adopt.
    slug = 'session-cockpit-4400013'
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.RUNNING, launcher_pid=4400013,
            claude_session_id='uuid-owner', claude_owner_pid=4400099,
        ),
        root=tmp_path,
    )
    remint = {'session_id': 'uuid-owner-2', 'cwd': '/home/leo/src/dark-factory', 'source': 'compact'}
    sh.run_session_start(remint, env, root=tmp_path, process_tree=claude_process_tree(None))

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-owner-2'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1
