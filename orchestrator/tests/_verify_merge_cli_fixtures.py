"""Shared scaffolding for tests that drive the `orchestrator verify-merge` / `cancel-verify` CLI."""

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner, Result

import orchestrator.cli as cli_module
from orchestrator.cli import main
from orchestrator.config import GitConfig, OrchestratorConfig


def _setup_verify_repo(tmp_path: Path):
    """Init a minimal git repo with a mod/test_x.py file, return (repo, head_sha)."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    p = str(repo)
    subprocess.run(['git', 'init', '-b', 'main', p], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'config', 'user.name', 'Test User'],
                   check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'config', 'user.email', 'test@example.com'],
                   check=True, capture_output=True)
    mod_dir = repo / 'mod'
    mod_dir.mkdir()
    (mod_dir / 'test_x.py').write_text('# placeholder\n')
    subprocess.run(['git', '-C', p, 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'commit', '-m', 'initial commit'],
                   check=True, capture_output=True)
    result = subprocess.run(['git', '-C', p, 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True)
    head_sha = result.stdout.strip()
    return repo, head_sha


# ---------------------------------------------------------------------------
# Shared arrange for a verify-merge run with no real git/build work
# ---------------------------------------------------------------------------

_VERIFY_MERGE_SHA = 'abc1234567890abc1234567890abc1234567890ab'
_VERIFY_MERGE_PGID = 55555
_VERIFY_MERGE_RESULT_JSON = '{"passed": true, "results": []}'


@dataclass(frozen=True)
class _PatchedVerifyMerge:
    """The GitOps double verify-merge constructs, and the invoker that runs the CLI."""

    git_ops: MagicMock
    invoke: Callable[..., Result]


def _patched_verify_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    acquire: AsyncMock | None = None,
    persistent: bool = False,
) -> _PatchedVerifyMerge:
    """Fake every verify-merge collaborator that does real work; return the double and invoker.

    Faked: GitOps (worktree_base is ``tmp_path/.worktrees``), load_config,
    start_own_process_group (returns _VERIFY_MERGE_PGID), start_stdin_watchdog
    (inert, so it never reads this process's real fd 0), and the verify_runner
    trio (stdout carries _VERIFY_MERGE_RESULT_JSON). A test re-patches the seam
    it is about. ``acquire`` stands in for acquire_host_verify_worktree, which
    otherwise returns a real directory. ``persistent`` sets
    git.persistent_merge_worktree. Either way worktree_base exists and
    persistent_merge_worktree_path is a real path under it, so the knob-ON lock
    paths are real files. The returned record carries the GitOps double and the
    invoker, which appends its arguments to
    ``verify-merge --sha _VERIFY_MERGE_SHA --spec {} --config <file>``.
    """
    fake_wt = tmp_path / '_merge-verify'
    fake_wt.mkdir()
    worktree_base = tmp_path / '.worktrees'
    worktree_base.mkdir()
    mock_git_ops = MagicMock()
    mock_git_ops.worktree_base = worktree_base
    mock_git_ops.persistent_merge_worktree_path = worktree_base / '_merge-verify'
    mock_git_ops.acquire_host_verify_worktree = (
        AsyncMock(return_value=fake_wt) if acquire is None else acquire
    )
    mock_git_ops.cleanup_merge_worktree = AsyncMock(return_value=None)
    monkeypatch.setattr('orchestrator.git_ops.GitOps', MagicMock(return_value=mock_git_ops))

    fake_config = OrchestratorConfig(
        project_root=tmp_path, git=GitConfig(persistent_merge_worktree=persistent),
    )
    monkeypatch.setattr(cli_module, 'load_config', lambda _: fake_config)
    monkeypatch.setattr(cli_module, 'start_own_process_group', lambda: _VERIFY_MERGE_PGID)
    monkeypatch.setattr(cli_module, 'start_stdin_watchdog', lambda pgid, *a, **kw: MagicMock())

    monkeypatch.setattr('orchestrator.verify_runner.spec_from_json', lambda s: MagicMock())
    monkeypatch.setattr(
        'orchestrator.verify_runner.run_merge_verify_on_worktree',
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        'orchestrator.verify_runner.result_to_json', lambda r: _VERIFY_MERGE_RESULT_JSON,
    )

    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text('')

    def invoke(*extra_args: str) -> Result:
        return CliRunner().invoke(main, [
            'verify-merge',
            '--sha', _VERIFY_MERGE_SHA,
            '--spec', '{}',
            '--config', str(cfg_file),
            *extra_args,
        ])

    return _PatchedVerifyMerge(git_ops=mock_git_ops, invoke=invoke)


def _communicate_or_empty(
    child: 'subprocess.Popen', *, timeout: float = 10.0
) -> tuple[bytes, bytes]:
    """child.communicate(), tolerating a still-open pipe held by an orphaned
    descendant instead of propagating subprocess.TimeoutExpired.

    verify-merge's spec spawns descendants (a `sleep`-based test command,
    build subprocesses) that inherit the Popen stdout/stderr pipe write
    ends. If verify-merge itself dies nonzero while an orphaned descendant
    still holds those pipes open, a plain communicate() can time out even
    though *child* has already been reaped -- which would replace the
    intended crash/timeout pytest.fail below with an unrelated
    TimeoutExpired, losing the exit-code and output diagnostics.
    """
    try:
        return child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        return (b'', b'')


def _wait_for_pgid_file_or_report_crash(
    pgf: 'Path',
    child: 'subprocess.Popen',
    *,
    timeout: float = 30.0,
    interval: float = 0.1,
) -> None:
    """Poll for *pgf* to appear; fail promptly and distinctly if *child* has
    already exited instead of burning the full *timeout* budget.

    verify-merge writes its pgid file before doing any work and removes it
    in a `finally` (orchestrator/src/orchestrator/cli.py::verify_merge) --
    so a subprocess that crashes fast writes-and-removes the file well
    inside a single poll interval, and a naive poll-then-timeout mislabels
    that crash as a timeout. Checking child.poll() on every iteration lets
    a crash (nonzero exit) fail immediately with an honest headline; a
    subprocess still running after the budget remains a genuine timeout
    (tasks 2350 and 2770 widened this budget repeatedly for load flakiness,
    not for crash attribution -- do not fold that budget into this change).
    A fast *successful* exit could also legitimately race the pgid-file
    window, so the crash/timeout split is on exit code, not merely on
    exit -- and is reported through its own message below rather than the
    deadline headline, since the elapsed time in that case is not the
    timeout.
    """
    start = time.monotonic()
    deadline = start + timeout
    fast_clean_exit = False
    while True:
        if pgf.exists():
            return
        exit_code = child.poll()
        if exit_code is not None:
            # Re-check once more: a fast SUCCESSFUL exit can legitimately
            # race the write-then-remove window.
            if pgf.exists():
                return
            if exit_code != 0:
                stdout, stderr = _communicate_or_empty(child)
                pytest.fail(
                    f'verify-merge exited with code {exit_code} before/without '
                    f'writing a stable pgid file (crash, not a timeout)\n'
                    f'STDOUT: {stdout.decode()[:2000]!r}\n'
                    f'STDERR: {stderr.decode()[:2000]!r}'
                )
            fast_clean_exit = True
            break  # exit_code == 0 -- fast successful race; fall through below
        if time.monotonic() >= deadline:
            break
        time.sleep(interval)

    stdout, stderr = _communicate_or_empty(child) if child.poll() is not None else (b'', b'')
    if fast_clean_exit:
        pytest.fail(
            f'verify-merge exited cleanly (code 0) after '
            f'{time.monotonic() - start:.1f}s without a visible pgid file -- '
            f'fast successful race or a no-op invocation\n'
            f'STDOUT: {stdout.decode()[:2000]!r}\n'
            f'STDERR: {stderr.decode()[:2000]!r}'
        )
    pytest.fail(
        f'verify-merge did not write pgid file within {timeout:.0f}s '
        f'(subprocess poll={child.poll()!r})\n'
        f'STDOUT: {stdout.decode()[:2000]!r}\n'
        f'STDERR: {stderr.decode()[:2000]!r}'
    )


def _wait_pgid_gone(pgid: int, *, timeout: float = 20.0, interval: float = 0.1) -> None:
    """Poll until process group *pgid* has no live members, else raise.

    SIGKILL delivery and reaping are asynchronous: a one-shot os.killpg
    immediately after child.wait() can observe a not-yet-reaped group member
    (a lingering zombie, or a killed descendant that reparented to init)
    even though the group is guaranteed to empty eventually. This asserts
    the real contract -- eventual group death -- via a bounded poll on a
    monotonic deadline instead of a synchronous one-shot check.

    Mirrors the EPERM-vs-ESRCH errno discrimination formerly inlined in
    test_verify_merge_cancel_end_to_end.
    """
    import errno
    import os
    import time

    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, OSError) as exc:
            if hasattr(exc, 'errno') and exc.errno == errno.EPERM:
                raise AssertionError(f'pgid group {pgid}: got EPERM (unexpected)') from exc
            return  # ESRCH → group is gone — expected
        if time.monotonic() >= deadline:
            raise AssertionError(f'verify-merge pgid group {pgid} still alive after cancel-verify')
        time.sleep(interval)
