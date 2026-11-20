"""Tests for the `orchestrator verify-merge` and `cancel-verify` subcommands."""

import asyncio
import contextlib
import fcntl
import os
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT
from _verify_merge_cli_fixtures import (
    _VERIFY_MERGE_PGID,
    _VERIFY_MERGE_RESULT_JSON,
    _VERIFY_MERGE_SHA,
    _patched_verify_merge,
    _setup_verify_repo,
    _wait_for_pgid_file_or_report_crash,
    _wait_pgid_gone,
)
from click.testing import CliRunner

import orchestrator.cli as cli_module
from orchestrator.cli import main
from orchestrator.config import OrchestratorConfig

# ---------------------------------------------------------------------------
# flock probes for the knob-ON lock tests (Task 2306, Task 2830)
# ---------------------------------------------------------------------------


def _flock_is_held(lock_path: Path) -> bool:
    """True iff an exclusive non-blocking flock through an independent fd is denied.

    flock(2) locks belong to the open file description, so this probe conflicts
    with the CLI's own in-process holds.
    """
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


@contextlib.contextmanager
def _holding_flock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive flock on lock_path through an fd independent of the CLI's."""
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------------------------------------------------------------------------
# Step-7: `orchestrator verify-merge` integration tests (parity + cleanup + errors)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('test_command,expect_pass', [('true', True), ('false', False)])
def test_verify_merge_cli_wrapper_transparency(tmp_path, monkeypatch, test_command, expect_pass):
    """CLI verify-merge is a transparent wrapper: JSON round-trip is lossless and exit code is clean.

    Both the local baseline and the CLI call run_merge_verify_on_worktree on the same
    SHA with the same spec, so the test validates that the CLI scaffolding (config load,
    spec parse, worktree create/cleanup, JSON serialise) adds no observable difference.
    It does *not* validate parity against the merge-queue dispatch path directly.

    SYNC test (asyncio_mode=auto): the local result is computed via asyncio.run()
    before invoking CliRunner so we never nest event loops.
    """
    from orchestrator.git_ops import GitOps
    from orchestrator.verify_runner import (
        MergeVerifySpec,
        UnscopedTypecheckSpec,
        VerifyCommand,
        result_from_json,
        run_merge_verify_on_worktree,
        spec_to_json,
    )

    repo, head_sha = _setup_verify_repo(tmp_path)
    config = OrchestratorConfig(project_root=repo)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: config)

    # A dummy config file just to satisfy click.Path(exists=True)
    cfg_file = tmp_path / 'dummy.yaml'
    cfg_file.write_text('')

    spec = MergeVerifySpec(
        verify_commands=(VerifyCommand('mod', test_command=test_command),),
        unscoped_typecheck=UnscopedTypecheckSpec(
            commands=(VerifyCommand('mod', type_check_command='true'),),
            block_on_timeout=True,
        ),
        task_files=('mod/test_x.py',),
        verify_env={},
        cold_timeout_secs=300.0,
    )

    # Compute local result in-process (SYNC: use asyncio.run to avoid nested loop)
    async def _local_run():
        git_ops = GitOps(config.git, repo)
        wt, _ = await git_ops._create_merge_worktree(base_sha=head_sha)
        try:
            return await run_merge_verify_on_worktree(wt, config, spec)
        finally:
            await git_ops.cleanup_merge_worktree(wt)

    local = asyncio.run(_local_run())

    # Invoke CLI
    r = CliRunner().invoke(main, [
        'verify-merge',
        '--sha', head_sha,
        '--spec', spec_to_json(spec),
        '--config', str(cfg_file),
    ])

    assert r.exit_code == 0, (
        f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    )
    cli_result = result_from_json(r.output)
    assert cli_result == local, (
        f'CLI result != local result: cli={cli_result!r}, local={local!r}'
    )
    assert local.passed is expect_pass


def test_verify_merge_cleanup(tmp_path, monkeypatch):
    """verify-merge must not leak _merge-* worktrees after the run completes."""
    from orchestrator.verify_runner import (
        MergeVerifySpec,
        UnscopedTypecheckSpec,
        VerifyCommand,
        spec_to_json,
    )

    repo, head_sha = _setup_verify_repo(tmp_path)
    config = OrchestratorConfig(project_root=repo)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: config)

    cfg_file = tmp_path / 'dummy.yaml'
    cfg_file.write_text('')

    spec = MergeVerifySpec(
        verify_commands=(VerifyCommand('mod', test_command='true'),),
        unscoped_typecheck=UnscopedTypecheckSpec(
            commands=(VerifyCommand('mod', type_check_command='true'),),
            block_on_timeout=True,
        ),
        task_files=('mod/test_x.py',),
        verify_env={},
        cold_timeout_secs=300.0,
    )

    r = CliRunner().invoke(main, [
        'verify-merge',
        '--sha', head_sha,
        '--spec', spec_to_json(spec),
        '--config', str(cfg_file),
    ])
    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'

    # No _merge-* directories should remain under .worktrees
    worktrees_dir = repo / '.worktrees'
    leaked = (
        any(p.name.startswith('_merge-') for p in worktrees_dir.iterdir())
        if worktrees_dir.exists()
        else False
    )
    assert not leaked, f'leaked worktree under {worktrees_dir}'

    # git worktree list must not show any _merge-* entry
    wt_list = subprocess.run(
        ['git', '-C', str(repo), 'worktree', 'list', '--porcelain'],
        check=True, capture_output=True, text=True,
    ).stdout
    assert '_merge-' not in wt_list, f'git worktree list shows leaked entry:\n{wt_list}'


@pytest.mark.parametrize('bad_case', ['malformed_spec', 'absent_sha'])
def test_verify_merge_clean_error_contract(tmp_path, monkeypatch, bad_case):
    """Bad requests never emit a non-verdict to stdout and always exit non-zero.

    Case A (malformed --spec): JSON parse error → graceful non-zero, empty stdout.
    Case B (absent SHA): non-existent commit → graceful non-zero, empty stdout.
    Both cases must have a concise message on stderr and no uncaught traceback.
    """
    from orchestrator.verify_runner import (
        MergeVerifySpec,
        UnscopedTypecheckSpec,
        VerifyCommand,
        result_from_json,
        spec_to_json,
    )

    repo, head_sha = _setup_verify_repo(tmp_path)
    config = OrchestratorConfig(project_root=repo)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: config)

    cfg_file = tmp_path / 'dummy.yaml'
    cfg_file.write_text('')

    good_spec = MergeVerifySpec(
        verify_commands=(VerifyCommand('mod', test_command='true'),),
        unscoped_typecheck=UnscopedTypecheckSpec(
            commands=(VerifyCommand('mod', type_check_command='true'),),
            block_on_timeout=True,
        ),
        task_files=('mod/test_x.py',),
        verify_env={},
        cold_timeout_secs=300.0,
    )

    if bad_case == 'malformed_spec':
        invoke_args = [
            'verify-merge',
            '--sha', head_sha,
            '--spec', 'not valid json',
            '--config', str(cfg_file),
        ]
    else:  # absent_sha
        invoke_args = [
            'verify-merge',
            '--sha', '0' * 40,
            '--spec', spec_to_json(good_spec),
            '--config', str(cfg_file),
        ]

    r = CliRunner().invoke(main, invoke_args)

    # Must exit non-zero
    assert r.exit_code != 0, (
        f'expected non-zero exit_code for {bad_case}, got {r.exit_code}'
    )
    # stdout must not contain a parseable VerifyResult
    # (click 8.3.2: r.output mixes stdout+stderr; use result_from_json to prove
    # no valid verdict was emitted)
    with pytest.raises((ValueError, TypeError)):
        result_from_json(r.output)
    # No uncaught traceback — exception must be SystemExit (or None)
    assert r.exception is None or isinstance(r.exception, SystemExit), (
        f'expected SystemExit or None, got {type(r.exception).__name__}: {r.exception}'
    )
    # A concise error message on stderr
    assert r.stderr.strip() != '', (
        f'expected non-empty stderr for {bad_case!r}'
    )


# ---------------------------------------------------------------------------
# Task 1699 step-7 — verify-merge subcommand wiring: acquire_host_verify_worktree
# ---------------------------------------------------------------------------


def test_verify_merge_uses_acquire_host_verify_worktree(tmp_path, monkeypatch):
    """verify-merge must call acquire_host_verify_worktree, NOT _create_merge_worktree."""
    patched = _patched_verify_merge(tmp_path, monkeypatch, persistent=True)
    git_ops = patched.git_ops
    git_ops._create_merge_worktree = AsyncMock(return_value=(MagicMock(), _VERIFY_MERGE_SHA))

    r = patched.invoke()

    assert r.exit_code == 0, (
        f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    )
    assert _VERIFY_MERGE_RESULT_JSON in r.output, (
        f'stdout must contain the known JSON; got: {r.output!r}'
    )

    # Core assertion: acquire_host_verify_worktree called; _create_merge_worktree NOT
    git_ops.acquire_host_verify_worktree.assert_awaited_once_with(_VERIFY_MERGE_SHA)
    git_ops._create_merge_worktree.assert_not_called()


# ---------------------------------------------------------------------------
# Task 1732 step-9 — verify-merge --request-id pgid lifecycle + back-compat
# ---------------------------------------------------------------------------


def test_verify_merge_request_id_pgid_lifecycle(tmp_path, monkeypatch):
    """verify-merge --request-id writes pgid file during run, removes it on exit.

    Injects start_own_process_group as a spy; mocks all IO so no real git/build
    work happens. Checks file existence mid-run via the mocked
    run_merge_verify_on_worktree coroutine.
    """
    from orchestrator.verify_cancel import pgid_file

    FAKE_REQUEST_ID = 'test-req-1732'
    invoke = _patched_verify_merge(tmp_path, monkeypatch).invoke

    sopg_calls = []

    def fake_sopg():
        sopg_calls.append(True)
        return _VERIFY_MERGE_PGID

    monkeypatch.setattr(cli_module, 'start_own_process_group', fake_sopg)

    pgf = pgid_file(tmp_path / '.worktrees', FAKE_REQUEST_ID)
    file_existed_mid_run = []

    async def fake_run_merge_verify(wt, cfg, spec, merge_sha=None):
        file_existed_mid_run.append(pgf.exists())
        return MagicMock()

    monkeypatch.setattr('orchestrator.verify_runner.run_merge_verify_on_worktree', fake_run_merge_verify)

    r = invoke('--request-id', FAKE_REQUEST_ID)

    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    assert _VERIFY_MERGE_RESULT_JSON in r.output

    # start_own_process_group must have been called once
    assert len(sopg_calls) == 1, 'start_own_process_group must be called once with --request-id'

    # pgid file must have existed during the run
    assert file_existed_mid_run == [True], (
        f'pgid file was not present mid-run; file_existed_mid_run={file_existed_mid_run!r}'
    )

    # pgid file must be removed after normal exit
    assert not pgf.exists(), f'pgid file must be removed on normal exit; still present at {pgf}'


def test_verify_merge_no_request_id_back_compat(tmp_path, monkeypatch):
    """Without --request-id, start_own_process_group is NOT called and no pgid file created.

    Verifies today's exact behavior is unchanged (back-compat).
    """
    invoke = _patched_verify_merge(tmp_path, monkeypatch).invoke

    sopg_calls = []

    def fake_sopg():
        sopg_calls.append(True)
        return _VERIFY_MERGE_PGID

    monkeypatch.setattr(cli_module, 'start_own_process_group', fake_sopg)

    r = invoke()

    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'

    # start_own_process_group must NOT have been called (no --request-id)
    assert sopg_calls == [], (
        'start_own_process_group must NOT be called without --request-id'
    )

    # No .merge_verify_pgids directory should have been created
    from orchestrator.verify_cancel import PGID_DIR_NAME
    pgid_dir_path = tmp_path / '.worktrees' / PGID_DIR_NAME
    assert not pgid_dir_path.exists(), (
        f'.merge_verify_pgids directory created without --request-id: {pgid_dir_path}'
    )


# ---------------------------------------------------------------------------
# Task 2306 step-9 — verify-merge flock guard: knob-on warm path + knob-off
# back-compat regression guard.
# ---------------------------------------------------------------------------


def test_verify_merge_flock_held_during_warm_run_and_released_after(tmp_path, monkeypatch):
    """Knob ON: the merge-verify flock is held for the whole warm verify span.

    Inside the mocked run_merge_verify_on_worktree coroutine, a fresh
    os.open + fcntl.flock(LOCK_EX|LOCK_NB) on merge_verify_lock_path(worktree_base)
    must raise BlockingIOError (the CLI holds the lock), and
    read_lock_holder_pgid(worktree_base) must return a positive int. After the
    CLI returns, the holder-pgid file must be gone and the lock must be
    re-acquirable (released).
    """
    from orchestrator.verify_cancel import (
        acquire_merge_verify_flock,
        merge_verify_lock_path,
        read_lock_holder_pgid,
        release_merge_verify_flock,
    )

    patched = _patched_verify_merge(tmp_path, monkeypatch, persistent=True)
    git_ops = patched.git_ops
    worktree_base = git_ops.worktree_base
    mid_run = {}

    async def fake_run_merge_verify(wt, cfg, spec, merge_sha=None):
        mid_run['lock_held'] = _flock_is_held(merge_verify_lock_path(worktree_base))
        mid_run['holder_pgid'] = read_lock_holder_pgid(worktree_base)
        return MagicMock()

    monkeypatch.setattr('orchestrator.verify_runner.run_merge_verify_on_worktree', fake_run_merge_verify)

    r = patched.invoke()

    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    assert _VERIFY_MERGE_RESULT_JSON in r.output

    assert mid_run.get('lock_held') is True, (
        f'merge-verify flock was not held during the warm run: {mid_run!r}'
    )
    assert isinstance(mid_run.get('holder_pgid'), int), (
        f'holder pgid was not recorded mid-run: {mid_run!r}'
    )
    assert mid_run['holder_pgid'] > 0

    git_ops.acquire_host_verify_worktree.assert_awaited_once_with(_VERIFY_MERGE_SHA)

    # After the CLI returns: holder-pgid file gone, lock re-acquirable.
    assert read_lock_holder_pgid(worktree_base) is None, (
        'holder-pgid file must be removed after the CLI returns'
    )
    fd = acquire_merge_verify_flock(merge_verify_lock_path(worktree_base), timeout_secs=1.0)
    assert fd is not None, 'lock must be re-acquirable after the CLI releases it'
    release_merge_verify_flock(fd)


def test_verify_merge_knob_off_no_flock_back_compat(tmp_path, monkeypatch):
    """Knob OFF: no lock is taken and no holder-pgid file is created (back-compat)."""
    from orchestrator.verify_cancel import (
        lane_lock_path,
        merge_verify_lock_path,
        read_lock_holder_pgid,
    )

    patched = _patched_verify_merge(tmp_path, monkeypatch)
    git_ops = patched.git_ops
    worktree_base = git_ops.worktree_base

    r = patched.invoke()

    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    assert _VERIFY_MERGE_RESULT_JSON in r.output

    git_ops.acquire_host_verify_worktree.assert_awaited_once_with(_VERIFY_MERGE_SHA)
    assert read_lock_holder_pgid(worktree_base) is None, (
        'no holder-pgid file should be created when the knob is off'
    )
    assert not merge_verify_lock_path(worktree_base).exists(), (
        'no .merge_verify.lock file should be created when the knob is off'
    )
    assert not lane_lock_path(git_ops.persistent_merge_worktree_path).exists(), (
        'no shared <lane_dir>.lock (_merge-verify.lock) should be created when '
        'the knob is off (task 2830)'
    )


# ---------------------------------------------------------------------------
# Task 2306 step-11 — verify-merge flock guard: bounded-wait contention
# outcome (distinguished VerifyResult, no ephemeral fallback).
# ---------------------------------------------------------------------------


def test_verify_merge_flock_contention_emits_distinguished_result(tmp_path, monkeypatch):
    """Knob ON + lock already held: bounded wait times out -> distinguished result.

    verify-merge must NOT call acquire_host_verify_worktree / _create_merge_worktree
    (no tree mutation, no ephemeral fallback). It must instead emit a passed=False
    VerifyResult with category == FLOCK_CONTENTION_CATEGORY and a `contention` dict
    naming the pre-recorded holder pgid and this process's own pgid as waiter, and
    exit 0 (a non-zero exit would make RemoteRunner treat this as RunnerUnavailable
    and beta would never see the discriminant).
    """
    from orchestrator.verify_cancel import merge_verify_lock_path, write_lock_holder_pgid
    from orchestrator.verify_runner import (
        FLOCK_CONTENTION_CATEGORY,
        result_from_json,
        result_to_json,
    )

    patched = _patched_verify_merge(tmp_path, monkeypatch, persistent=True)
    git_ops = patched.git_ops
    worktree_base = git_ops.worktree_base
    git_ops._create_merge_worktree = AsyncMock(return_value=(MagicMock(), _VERIFY_MERGE_SHA))
    # The distinguished payload must round-trip through the real codec on stdout.
    monkeypatch.setattr('orchestrator.verify_runner.result_to_json', result_to_json)
    monkeypatch.setattr(cli_module, 'MERGE_VERIFY_FLOCK_WAIT_SECS', 0.3)

    write_lock_holder_pgid(worktree_base, 9191)
    with _holding_flock(merge_verify_lock_path(worktree_base)):
        r = patched.invoke()

    assert r.exit_code == 0, (
        f'contention must exit 0 (the only delivery channel for the discriminant); '
        f'got {r.exit_code}; output={r.output!r}'
    )

    result = result_from_json(r.output)
    assert result.passed is False
    assert result.category == FLOCK_CONTENTION_CATEGORY
    assert result.contention is not None
    assert isinstance(result.contention['host'], str) and result.contention['host'], (
        f'contention.host must be a non-empty str: {result.contention!r}'
    )
    assert result.contention['holder_pgid'] == 9191
    assert result.contention['waiter_pgid'] == os.getpgrp()

    git_ops.acquire_host_verify_worktree.assert_not_awaited()
    git_ops._create_merge_worktree.assert_not_called()


# ---------------------------------------------------------------------------
# Task 2830 — converge the laptop verify-merge CLI span onto the SHARED lane
# lock (lane_lock_path(persistent_merge_worktree_path) = <worktree_base>/
# _merge-verify.lock), finishing DF 2685 for the remote host. These two pins
# fail against current main (the CLI holds only .merge_verify.lock and ignores
# the lane lock); they go green once step-2 dual-locks the CLI span.
# ---------------------------------------------------------------------------


def test_verify_merge_holds_lane_lock_during_warm_run(tmp_path, monkeypatch):
    """Knob ON: the SHARED lane lock is held for the whole warm verify span (task 2830).

    Converges the laptop CLI span onto
    ``lane_lock_path(persistent_merge_worktree_path)`` = ``<worktree_base>/
    _merge-verify.lock`` — the SAME lock ``GitOps.merge_verify_lease`` and reify's
    seed/thin/gc take (task 2685) — so a laptop lane actor is mutually excluded from
    a live laptop verify. Inside the mocked run_merge_verify_on_worktree coroutine, a
    fresh os.open + fcntl.flock(LOCK_EX|LOCK_NB) on that lane lock must raise
    BlockingIOError (the CLI holds it) and read_lock_holder_pgid(worktree_base) must
    return a positive int. After the CLI returns, the holder-pgid file must be gone and
    the lane lock re-acquirable (released).
    """
    from orchestrator.verify_cancel import (
        acquire_merge_verify_flock,
        lane_lock_path,
        read_lock_holder_pgid,
        release_merge_verify_flock,
    )

    patched = _patched_verify_merge(tmp_path, monkeypatch, persistent=True)
    git_ops = patched.git_ops
    worktree_base = git_ops.worktree_base
    mid_run = {}

    async def fake_run_merge_verify(wt, cfg, spec, merge_sha=None):
        mid_run['lock_held'] = _flock_is_held(lane_lock_path(git_ops.persistent_merge_worktree_path))
        mid_run['holder_pgid'] = read_lock_holder_pgid(worktree_base)
        return MagicMock()

    monkeypatch.setattr('orchestrator.verify_runner.run_merge_verify_on_worktree', fake_run_merge_verify)

    r = patched.invoke()

    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    assert _VERIFY_MERGE_RESULT_JSON in r.output

    assert mid_run.get('lock_held') is True, (
        f'shared lane lock was not held during the warm run: {mid_run!r}'
    )
    assert isinstance(mid_run.get('holder_pgid'), int), (
        f'holder pgid was not recorded mid-run: {mid_run!r}'
    )
    assert mid_run['holder_pgid'] > 0

    git_ops.acquire_host_verify_worktree.assert_awaited_once_with(_VERIFY_MERGE_SHA)

    # After the CLI returns: holder-pgid file gone, lane lock re-acquirable.
    assert read_lock_holder_pgid(worktree_base) is None, (
        'holder-pgid file must be removed after the CLI returns'
    )
    fd = acquire_merge_verify_flock(
        lane_lock_path(git_ops.persistent_merge_worktree_path), timeout_secs=1.0
    )
    assert fd is not None, 'lane lock must be re-acquirable after the CLI releases it'
    release_merge_verify_flock(fd)


def test_verify_merge_lane_actor_contention_emits_distinguished_result(tmp_path, monkeypatch):
    """Knob ON + a laptop lane actor holds <lane_dir>.lock -> distinguished result (task 2830).

    Simulates a laptop reseed/thin/gc holding the SHARED lane lock
    (``lane_lock_path(persistent_merge_worktree_path)`` = ``<worktree_base>/
    _merge-verify.lock``) and having recorded its pgid. The CLI's bounded wait on the
    lane lock must time out and emit the distinguished contention result (passed=False,
    category == FLOCK_CONTENTION_CATEGORY, naming holder 9292 and this process's own
    pgid as waiter) at exit 0, WITHOUT ever touching the tree
    (acquire_host_verify_worktree not awaited).
    """
    from orchestrator.verify_cancel import lane_lock_path, write_lock_holder_pgid
    from orchestrator.verify_runner import (
        FLOCK_CONTENTION_CATEGORY,
        result_from_json,
        result_to_json,
    )

    patched = _patched_verify_merge(tmp_path, monkeypatch, persistent=True)
    git_ops = patched.git_ops
    worktree_base = git_ops.worktree_base
    # The distinguished payload must round-trip through the real codec on stdout.
    monkeypatch.setattr('orchestrator.verify_runner.result_to_json', result_to_json)
    monkeypatch.setattr(cli_module, 'MERGE_VERIFY_FLOCK_WAIT_SECS', 0.3)

    write_lock_holder_pgid(worktree_base, 9292)
    with _holding_flock(lane_lock_path(git_ops.persistent_merge_worktree_path)):
        r = patched.invoke()

    assert r.exit_code == 0, (
        f'contention must exit 0 (the only delivery channel for the discriminant); '
        f'got {r.exit_code}; output={r.output!r}'
    )

    result = result_from_json(r.output)
    assert result.passed is False
    assert result.category == FLOCK_CONTENTION_CATEGORY
    assert result.contention is not None
    assert isinstance(result.contention['host'], str) and result.contention['host'], (
        f'contention.host must be a non-empty str: {result.contention!r}'
    )
    assert result.contention['holder_pgid'] == 9292
    assert result.contention['waiter_pgid'] == os.getpgrp()

    git_ops.acquire_host_verify_worktree.assert_not_awaited()


def test_verify_merge_also_holds_old_compat_lock_during_warm_run(tmp_path, monkeypatch):
    """Knob ON: the OLD .merge_verify.lock co-lock is ALSO held for the whole span (task 2830).

    Rollout back-compat guard (green-on-arrival after step-2). The converged CLI span
    RETAINS the divergent ``.merge_verify.lock`` (``merge_verify_lock_path(worktree_base)``)
    as a transitional rollout CO-lock, held across the WHOLE verify span, so an
    in-flight OLD verify-merge — one holding ONLY that lock, before checkout-sync ships
    the lane-lock code — stays mutually excluded during rollout. This pins the HOLD side
    of the compat arm, deliberately distinct from
    ``test_verify_merge_flock_contention_emits_distinguished_result`` (which exercises
    the compat-arm WAITER side). It regresses the moment the compat acquire is dropped
    from the dual-lock.

    Folds in the lane-lock probe too (as the plan permits) to assert the FULL dual-hold
    in a single mid-build run: with the split lane-lock lifetime (esc-2830-1) the lane
    lock is re-acquired for the build's duration inside ``_run()``, so while
    ``run_merge_verify_on_worktree`` runs BOTH the OLD compat lock (held across the
    whole span) AND the SHARED lane lock (held build-scoped) are exclusive.
    """
    from orchestrator.verify_cancel import (
        acquire_merge_verify_flock,
        lane_lock_path,
        merge_verify_lock_path,
        read_lock_holder_pgid,
        release_merge_verify_flock,
    )

    patched = _patched_verify_merge(tmp_path, monkeypatch, persistent=True)
    git_ops = patched.git_ops
    worktree_base = git_ops.worktree_base
    mid_run = {}

    async def fake_run_merge_verify(wt, cfg, spec, merge_sha=None):
        # NEW pin: the OLD .merge_verify.lock compat co-lock is held mid-run.
        mid_run['old_lock_held'] = _flock_is_held(merge_verify_lock_path(worktree_base))
        # Folded-in full dual-hold: the SHARED lane lock is held mid-build too.
        mid_run['lane_lock_held'] = _flock_is_held(
            lane_lock_path(git_ops.persistent_merge_worktree_path)
        )
        mid_run['holder_pgid'] = read_lock_holder_pgid(worktree_base)
        return MagicMock()

    monkeypatch.setattr('orchestrator.verify_runner.run_merge_verify_on_worktree', fake_run_merge_verify)

    r = patched.invoke()

    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    assert _VERIFY_MERGE_RESULT_JSON in r.output

    assert mid_run.get('old_lock_held') is True, (
        f'OLD .merge_verify.lock compat co-lock was not held during the warm run '
        f'(an in-flight OLD verify-merge would NOT be excluded during rollout): {mid_run!r}'
    )
    assert mid_run.get('lane_lock_held') is True, (
        f'shared lane lock was not held during the warm run (full dual-hold): {mid_run!r}'
    )
    assert isinstance(mid_run.get('holder_pgid'), int), (
        f'holder pgid was not recorded mid-run: {mid_run!r}'
    )
    assert mid_run['holder_pgid'] > 0

    git_ops.acquire_host_verify_worktree.assert_awaited_once_with(_VERIFY_MERGE_SHA)

    # After the CLI returns: holder-pgid file gone, OLD compat lock re-acquirable (released).
    assert read_lock_holder_pgid(worktree_base) is None, (
        'holder-pgid file must be removed after the CLI returns'
    )
    fd = acquire_merge_verify_flock(
        merge_verify_lock_path(worktree_base), timeout_secs=1.0
    )
    assert fd is not None, 'OLD compat lock must be re-acquirable after the CLI releases it'
    release_merge_verify_flock(fd)


def test_verify_merge_build_lane_acquire_fails_open(tmp_path, monkeypatch):
    """Knob ON: the in-_run() BUILD lane re-acquire failing OPEN still completes (task 2830).

    Pins the deliberate fail-OPEN arm of the split lane-lock lifetime (esc-2830-1,
    mirroring ``GitOps.merge_verify_lease``): after the pre-flight gate passes and
    releases the lane lock, ``_run()`` re-acquires the lane lock for the build only.
    If THAT acquire times out (``build_lane_fd`` is None), the build proceeds WITHOUT
    the lane lock and WITHOUT writing the shared holder-pgid — it does NOT raise, does
    NOT emit a contention result, and leaves NO holder-pgid file behind. (A live lane
    actor is already excluded by the pre-flight gate + the compat co-lock held across
    the whole span, so proceeding unrecorded is safe.)

    Forces ONLY the in-_run() lane acquire to fail (return None) by failing the SECOND
    lane-lock acquire while leaving the pre-flight gate's FIRST lane acquire and the
    compat acquire (a DIFFERENT path) untouched. Regresses the moment fail-open turns
    into fail-closed (raise / contention result) or the build starts writing a holder
    on the no-lock path.
    """
    from orchestrator.verify_cancel import (
        acquire_merge_verify_flock,
        lane_lock_path,
        merge_verify_lock_path,
        read_lock_holder_pgid,
        release_merge_verify_flock,
    )

    patched = _patched_verify_merge(tmp_path, monkeypatch, persistent=True)
    git_ops = patched.git_ops
    worktree_base = git_ops.worktree_base

    # --- Force ONLY the in-_run() BUILD lane acquire to fail open (return None). ---
    # The knob-on span acquires the LANE lock twice: (1) at the pre-flight gate (must
    # succeed so the gate passes), then (2) inside _run() for the build (the acquire
    # under test). The compat acquire targets a DIFFERENT path and always succeeds.
    # Fail the 2nd lane-lock acquire onward, leaving the gate + compat acquires real.
    real_acquire = cli_module.acquire_merge_verify_flock
    lane_lock = lane_lock_path(git_ops.persistent_merge_worktree_path)
    lane_calls = {'n': 0}

    def fake_acquire(lock_path, timeout_secs):
        if lock_path == lane_lock:
            lane_calls['n'] += 1
            if lane_calls['n'] >= 2:
                return None  # the BUILD re-acquire times out -> fail OPEN
        return real_acquire(lock_path, timeout_secs)

    monkeypatch.setattr(cli_module, 'acquire_merge_verify_flock', fake_acquire)

    # --- Probe holder + lane-lock state mid-build. ---
    mid_run = {}

    async def fake_run_merge_verify(wt, cfg, spec, merge_sha=None):
        mid_run['reached'] = True
        # Fail-open: the build holds NO lane lock and wrote NO holder-pgid.
        mid_run['lane_lock_held'] = _flock_is_held(lane_lock)
        mid_run['holder_pgid'] = read_lock_holder_pgid(worktree_base)
        return MagicMock()

    monkeypatch.setattr('orchestrator.verify_runner.run_merge_verify_on_worktree', fake_run_merge_verify)

    r = patched.invoke()

    # Fail-OPEN contract: the verify still completes normally (exit 0, result echoed).
    assert r.exit_code == 0, f'fail-open must exit 0; got {r.exit_code}; output={r.output!r}'
    assert _VERIFY_MERGE_RESULT_JSON in r.output, (
        f'the build result must still be echoed on the fail-open path: {r.output!r}'
    )
    assert 'Error:' not in r.output, f'fail-open must NOT raise: {r.output!r}'

    # The BUILD lane re-acquire was genuinely exercised and forced to None (>=2 acquires).
    assert lane_calls['n'] >= 2, (
        f'the in-_run build lane re-acquire was not reached: {lane_calls!r}'
    )
    # The build proceeded despite no lane lock, and recorded NO holder mid-run.
    assert mid_run.get('reached') is True, 'run_merge_verify_on_worktree must still run'
    assert mid_run.get('lane_lock_held') is False, (
        f'fail-open: the build must hold NO lane lock: {mid_run!r}'
    )
    assert mid_run.get('holder_pgid') is None, (
        f'fail-open: no shared holder-pgid may be written on the no-lock build path: {mid_run!r}'
    )

    git_ops.acquire_host_verify_worktree.assert_awaited_once_with(_VERIFY_MERGE_SHA)

    # After the CLI returns: no holder-pgid file, both locks free (compat released,
    # lane never held during the build).
    assert read_lock_holder_pgid(worktree_base) is None, (
        'no holder-pgid file may exist after a fail-open run'
    )
    fd_lane = acquire_merge_verify_flock(lane_lock, timeout_secs=1.0)
    assert fd_lane is not None, 'lane lock must be free after a fail-open run'
    release_merge_verify_flock(fd_lane)
    fd_compat = acquire_merge_verify_flock(
        merge_verify_lock_path(worktree_base), timeout_secs=1.0
    )
    assert fd_compat is not None, 'compat lock must be released after the CLI returns'
    release_merge_verify_flock(fd_compat)


# ---------------------------------------------------------------------------
# Task 1732 step-11 — cancel-verify subcommand (CliRunner)
# ---------------------------------------------------------------------------


def test_cancel_verify_unknown_id_exits_0(tmp_path, monkeypatch):
    """(a) Unknown request id (no pgid file present) -> exit 0 (idempotent)."""
    from unittest.mock import MagicMock

    from orchestrator.config import OrchestratorConfig

    # Minimal config: project_root -> tmp_path (worktree_base = tmp_path/.worktrees)
    fake_config = OrchestratorConfig(project_root=tmp_path)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: fake_config)

    # GitOps mock returning a worktree_base that has no pgid file
    fake_worktree_base = tmp_path / '.worktrees'
    mock_git_ops = MagicMock()
    mock_git_ops.worktree_base = fake_worktree_base
    monkeypatch.setattr('orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops))

    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text('')

    r = CliRunner().invoke(main, [
        'cancel-verify',
        '--request-id', 'nonexistent-req',
        '--config', str(cfg_file),
    ])
    assert r.exit_code == 0, (
        f'expected exit_code 0 for unknown id, got {r.exit_code}; output={r.output!r}'
    )


def test_cancel_verify_rc_wiring(tmp_path, monkeypatch):
    """(b) cancel_request spy: CLI passes pgid_file path and exits with exactly that rc."""
    from unittest.mock import MagicMock

    from orchestrator.config import OrchestratorConfig
    from orchestrator.verify_cancel import pgid_file

    FAKE_REQUEST_ID = 'wiring-test-req'
    EXPECTED_RC = 42  # unusual value to prove propagation

    fake_worktree_base = tmp_path / '.worktrees'
    expected_path = pgid_file(fake_worktree_base, FAKE_REQUEST_ID)

    cancel_calls = []

    def fake_cancel_request(path, **kwargs):
        cancel_calls.append(path)
        return EXPECTED_RC

    monkeypatch.setattr(cli_module, 'cancel_request', fake_cancel_request)

    fake_config = OrchestratorConfig(project_root=tmp_path)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: fake_config)

    mock_git_ops = MagicMock()
    mock_git_ops.worktree_base = fake_worktree_base
    monkeypatch.setattr('orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops))

    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text('')

    r = CliRunner().invoke(main, [
        'cancel-verify',
        '--request-id', FAKE_REQUEST_ID,
        '--config', str(cfg_file),
    ])

    # RC propagated correctly
    assert r.exit_code == EXPECTED_RC, (
        f'expected exit_code {EXPECTED_RC}, got {r.exit_code}; output={r.output!r}'
    )
    # Path derivation: cancel_request was called with the right pgid_file path
    assert len(cancel_calls) == 1, f'cancel_request called {len(cancel_calls)} times'
    assert cancel_calls[0] == expected_path, (
        f'cancel_request called with wrong path: {cancel_calls[0]} != {expected_path}'
    )


def test_cancel_verify_real_impl_dead_pgid(tmp_path, monkeypatch):
    """(c) Real pgid file with a dead pgid -> exit 0, file removed."""
    from unittest.mock import MagicMock

    from orchestrator.config import OrchestratorConfig
    from orchestrator.verify_cancel import pgid_file, write_pgid_file

    FAKE_REQUEST_ID = 'dead-pgid-req'
    # Use pid 1 (init) as the "dead" pgid — we can't kill it, but with a dead
    # process we instead use a non-existent high pid that raises ProcessLookupError.
    # We'll write a nonsense high pid that definitely doesn't exist.
    NONEXISTENT_PID = 2 ** 22  # well above /proc/sys/kernel/pid_max on most systems

    fake_worktree_base = tmp_path / '.worktrees'
    pgf = pgid_file(fake_worktree_base, FAKE_REQUEST_ID)
    write_pgid_file(pgf, NONEXISTENT_PID)

    fake_config = OrchestratorConfig(project_root=tmp_path)
    monkeypatch.setattr(cli_module, 'load_config', lambda _: fake_config)

    mock_git_ops = MagicMock()
    mock_git_ops.worktree_base = fake_worktree_base
    monkeypatch.setattr('orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops))

    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text('')

    r = CliRunner().invoke(main, [
        'cancel-verify',
        '--request-id', FAKE_REQUEST_ID,
        '--config', str(cfg_file),
    ])

    assert r.exit_code == 0, (
        f'expected exit_code 0 for dead pgid, got {r.exit_code}; output={r.output!r}'
    )
    assert not pgf.exists(), f'pgid file must be removed after successful cancel; still at {pgf}'


# ---------------------------------------------------------------------------
# Task 1732 step-13 (part 2) — end-to-end: real subprocess cancel capstone
# ---------------------------------------------------------------------------


# Slow: _wait_pgid_gone's poll-for-drain alone can take up to 20s (task 2733).
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
def test_verify_merge_cancel_end_to_end(tmp_path, monkeypatch):
    """End-to-end: real subprocess 'verify-merge --request-id X' is killed by cancel-verify.

    Spawns a real 'orchestrator verify-merge --request-id X' subprocess with a
    long-running sleep spec.  Polls for the pgid file that verify-merge writes
    after os.setsid().  Calls 'cancel-verify --request-id X' via CliRunner
    (in-process, monkeypatched load_config).  Asserts:

    * cancel-verify exits 0
    * pgid file is removed by cancel_request
    * verify-merge subprocess exits within seconds (SIGKILL delivered)
    * the recorded pgid group is gone (os.killpg raises ESRCH) after wait()
    """
    import os
    import subprocess as subprocess_mod
    import sys

    from orchestrator.config import OrchestratorConfig
    from orchestrator.git_ops import GitOps
    from orchestrator.verify_cancel import pgid_file
    from orchestrator.verify_runner import (
        MergeVerifySpec,
        UnscopedTypecheckSpec,
        VerifyCommand,
        spec_to_json,
    )

    # --- Set up a real git repo ---
    repo, head_sha = _setup_verify_repo(tmp_path)

    # --- Write minimal config YAML (project_root -> repo) ---
    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text(f'project_root: {repo}\n')

    # --- Spec with a long-running sleep test command ---
    spec = MergeVerifySpec(
        verify_commands=(VerifyCommand('mod', test_command='sleep 300'),),
        unscoped_typecheck=UnscopedTypecheckSpec(
            commands=(VerifyCommand('mod', type_check_command='true'),),
            block_on_timeout=True,
        ),
        task_files=('mod/test_x.py',),
        verify_env={},
        cold_timeout_secs=300.0,
    )

    REQUEST_ID = 'e2e-cancel-test'

    # --- Derive the expected pgid file path (same derivation as verify-merge uses) ---
    config_obj = OrchestratorConfig(project_root=repo)
    _git_ops = GitOps(config_obj.git, repo)
    pgf = pgid_file(_git_ops.worktree_base, REQUEST_ID)

    # --- Spawn verify-merge as a real subprocess ---
    # Set PYTHONPATH so the subprocess imports from the worktree's src (not the
    # editable-install main-checkout), ensuring it uses the task-branch code.
    worktree_src = str(Path(__file__).parent.parent / 'src')
    env = dict(os.environ)
    existing_pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = f'{worktree_src}:{existing_pp}' if existing_pp else worktree_src
    # Remove ORCH_PROJECT_ROOT so it does not override the YAML's project_root.
    # The _isolate_orch_config autouse fixture sets ORCH_PROJECT_ROOT=<pytest
    # tmp_path>, and pydantic-settings env_settings has higher priority than
    # yaml_settings.  Without this removal the subprocess would use the pytest
    # tmp_path as project_root instead of the test's git repo, causing git
    # worktree operations to fail with "not a git repository".
    env.pop('ORCH_PROJECT_ROOT', None)

    child = subprocess_mod.Popen(
        [sys.executable, '-c', 'from orchestrator.cli import main; main()',
         'verify-merge',
         '--sha', head_sha,
         '--spec', spec_to_json(spec),
         '--config', str(cfg_file),
         '--request-id', REQUEST_ID],
        env=env,
        stdout=subprocess_mod.PIPE,
        stderr=subprocess_mod.PIPE,
    )

    pgid_val = None
    try:
        # --- Poll for the pgid file (written before asyncio.run) ---
        # Distinguishes a crashed verify-merge subprocess (reported
        # immediately, exit code != 0) from a genuine timeout (subprocess
        # still running after the budget) -- see
        # _wait_for_pgid_file_or_report_crash's docstring.
        _wait_for_pgid_file_or_report_crash(pgf, child, timeout=30)

        # Save pgid BEFORE cancel-verify removes the file
        pgid_val = int(pgf.read_text().strip())

        # --- Cancel via CliRunner (in-process, monkeypatched load_config) ---
        monkeypatch.setattr(cli_module, 'load_config', lambda _: config_obj)
        r = CliRunner().invoke(main, [
            'cancel-verify',
            '--request-id', REQUEST_ID,
            '--config', str(cfg_file),
        ])

        assert r.exit_code == 0, (
            f'cancel-verify expected exit 0, got {r.exit_code}; '
            f'output={r.output!r}'
        )
        assert not pgf.exists(), (
            'pgid file must be removed by cancel-verify on success'
        )

        # --- verify-merge subprocess must exit within seconds ---
        try:
            child.wait(timeout=20)
        except subprocess_mod.TimeoutExpired:
            pytest.fail(
                'verify-merge subprocess did not exit within 20s after cancel-verify'
            )

        # --- Process group must eventually be gone after the subprocess is
        # reaped. Bounded poll (not a one-shot check) -- see _wait_pgid_gone
        # docstring: SIGKILL delivery + reaping is asynchronous. ---
        _wait_pgid_gone(pgid_val, timeout=20)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


# ---------------------------------------------------------------------------
# Task 2308 step-11 — verify-merge watchdog wiring: start_stdin_watchdog spawn
# ---------------------------------------------------------------------------


def test_verify_merge_spawns_watchdog_when_request_id_set(tmp_path, monkeypatch):
    """--request-id set: verify-merge spawns the stdin watchdog with the pgid.

    Over the shared _patched_verify_merge arrange (known pgid, no real
    git/build work), monkeypatches orchestrator.cli.start_stdin_watchdog to a
    recorder. Asserts the recorder is called exactly once with the SAME pgid
    written to the pgid file.
    """
    invoke = _patched_verify_merge(tmp_path, monkeypatch).invoke

    watchdog_calls = []

    def fake_start_stdin_watchdog(pgid, *args, **kwargs):
        watchdog_calls.append(pgid)
        return MagicMock()

    monkeypatch.setattr(cli_module, 'start_stdin_watchdog', fake_start_stdin_watchdog)

    r = invoke('--request-id', 'test-req-2308')

    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    assert watchdog_calls == [_VERIFY_MERGE_PGID], (
        f'start_stdin_watchdog must be called exactly once with pgid={_VERIFY_MERGE_PGID}; '
        f'got {watchdog_calls!r}'
    )


def test_verify_merge_no_watchdog_when_request_id_absent(tmp_path, monkeypatch):
    """Without --request-id, start_stdin_watchdog is NOT spawned (back-compat)."""
    invoke = _patched_verify_merge(tmp_path, monkeypatch).invoke

    watchdog_calls = []

    def fake_start_stdin_watchdog(pgid, *args, **kwargs):
        watchdog_calls.append(pgid)
        return MagicMock()

    monkeypatch.setattr(cli_module, 'start_stdin_watchdog', fake_start_stdin_watchdog)

    r = invoke()

    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    assert watchdog_calls == [], (
        f'start_stdin_watchdog must NOT be called without --request-id; got {watchdog_calls!r}'
    )


# ---------------------------------------------------------------------------
# Task 4194 step-1 — the fire callback forwards the WatchdogTrigger
# ---------------------------------------------------------------------------


def test_verify_merge_watchdog_fire_callback_forwards_trigger(tmp_path, monkeypatch):
    """The fire callback forwards the trigger, and still sets watchdog_fired FIRST.

    The callback the CLI hands start_stdin_watchdog is the last hop before
    fire_watchdog_kill reports the branch identity on stderr, so it has to
    carry that identity through. It must also keep setting watchdog_fired
    strictly before anything else: that happens-before is load-bearing (see
    the comment above the callback in cli.py) and threading a new argument
    through must not reorder it. Both are asserted from one ordered log.
    """
    from orchestrator.verify_cancel import WatchdogTrigger

    invoke = _patched_verify_merge(tmp_path, monkeypatch).invoke

    # One ordered log shared by both seams, so the happens-before is read off
    # the log rather than inferred from two independent recorders.
    ordered = []

    class _LoggingEvent(threading.Event):
        def set(self):
            ordered.append('watchdog_fired.set')
            super().set()

    monkeypatch.setattr(threading, 'Event', _LoggingEvent)

    kill_calls = []

    def fake_fire_watchdog_kill(pgid, **kwargs):
        ordered.append('fire_watchdog_kill')
        kill_calls.append((pgid, kwargs))

    monkeypatch.setattr(cli_module, 'fire_watchdog_kill', fake_fire_watchdog_kill)

    captured_fire = []

    def fake_start_stdin_watchdog(pgid, *args, fire=None, **kwargs):
        captured_fire.append(fire)
        return MagicMock()

    monkeypatch.setattr(cli_module, 'start_stdin_watchdog', fake_start_stdin_watchdog)

    r = invoke('--request-id', 'test-req-4194')

    assert r.exit_code == 0, f'expected exit_code 0, got {r.exit_code}; output={r.output!r}'
    assert len(captured_fire) == 1, f'expected one fire= callback; got {captured_fire!r}'
    assert captured_fire[0] is not None

    ordered.clear()  # scope the log to the callback's own actions
    captured_fire[0](WatchdogTrigger.HEARTBEAT_STARVATION)

    assert ordered == ['watchdog_fired.set', 'fire_watchdog_kill']
    assert len(kill_calls) == 1
    pgid, kwargs = kill_calls[0]
    assert pgid == _VERIFY_MERGE_PGID
    assert kwargs['trigger'] is WatchdogTrigger.HEARTBEAT_STARVATION


# ---------------------------------------------------------------------------
# Task 5904 — a fired watchdog owns the exit on every path out of _run()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('setup_raises', [True, False], ids=['setup_raises', 'returns'])
def test_verify_merge_hands_exit_to_a_fired_watchdog_on_every_path(
    tmp_path, monkeypatch, setup_raises,
):
    """A raise out of _run() hands the exit to a fired watchdog, exactly like a return.

    The watchdog's own SIGTERM routinely makes _run() RAISE (it kills
    ``git worktree add``, rc=-15). If that path exits without joining the
    watchdog, interpreter shutdown tears the daemon watchdog thread down
    mid-grace and its SIGKILL escalation never runs. See
    cli.py::verify_merge.
    """
    from orchestrator.verify_cancel import WatchdogTrigger

    acquire = (
        AsyncMock(side_effect=RuntimeError('git worktree add failed rc=-15'))
        if setup_raises else None
    )
    invoke = _patched_verify_merge(tmp_path, monkeypatch, acquire=acquire).invoke

    log = []

    def fake_fire_watchdog_kill(pgid, **kwargs):
        log.append('escalation_started')

    monkeypatch.setattr(cli_module, 'fire_watchdog_kill', fake_fire_watchdog_kill)

    class _FakeWatchdogThread:
        def join(self, timeout=None):
            log.append('watchdog_joined')

    def fake_start_stdin_watchdog(pgid, *, fire, **kwargs):
        fire(WatchdogTrigger.EOF)
        return _FakeWatchdogThread()

    monkeypatch.setattr(cli_module, 'start_stdin_watchdog', fake_start_stdin_watchdog)

    r = invoke('--request-id', 'test-req-5904')

    assert log == ['escalation_started', 'watchdog_joined'], (
        f'a fired watchdog must own the exit: expected the main thread to block on '
        f'the watchdog after the escalation began; got log={log!r}, '
        f'exit_code={r.exit_code}, output={r.output!r}'
    )
    assert r.exit_code == 1, f'expected exit_code 1, got {r.exit_code}; output={r.output!r}'
    assert _VERIFY_MERGE_RESULT_JSON not in r.stdout, (
        f'never print a VerifyResult for a build the watchdog killed; stdout={r.stdout!r}'
    )
