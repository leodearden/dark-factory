"""Tests for landlock sandbox backend and backend dispatcher."""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from orchestrator.agents import landlock as landlock_mod
from orchestrator.agents.landlock import (
    _reset_probe as _landlock_reset_probe,
)
from orchestrator.agents.landlock import (
    build_landlock_command,
    is_landlock_available,
)
from orchestrator.agents.sandbox_dispatch import Backend


@pytest.fixture(autouse=True)
def _reset_landlock_probe():
    """Reset cached probe before and after each test."""
    _landlock_reset_probe()
    yield
    _landlock_reset_probe()


_VAR_TMP_SKIP_REASON = '/var/tmp not writable in this sandbox'
_LANDLOCK_SKIP_REASON = 'landlock not supported on this kernel'


@functools.cache
def _var_tmp_writable() -> bool:
    """Probe whether /var/tmp is actually usable for scratch dirs in this process.

    Catches ANY OS-level refusal, not just permission denial: agent sandboxes
    deny the write via a syscall filter (EACCES) even though the directory's
    own mode (1777) permits it, minimal containers may not ship /var/tmp at all
    (ENOENT), and a read-only bind mount surfaces as EROFS. All three must
    degrade to a skip — an escaping OSError here would abort collection of the
    whole module, the exact failure mode these guards exist to prevent. Must be
    an actual write attempt rather than an os.access/stat-mode check.

    NOTE: duplicated verbatim in test_sandbox_enforcement_matrix.py and in
    fused-memory/tests/reconciliation/test_recon_sandbox_guard.py — no shared
    test-helper module spans both packages, so keep the three copies in sync.
    """
    try:
        probe = tempfile.mkdtemp(dir='/var/tmp')
    except OSError:
        return False
    shutil.rmtree(probe, ignore_errors=True)
    return True


def _skip_var_tmp() -> bool:
    """Whether /var/tmp-dependent tests should be skipped in this environment.

    Under ``DF_REQUIRE_SANDBOX_TESTS=1`` — armed via ``verify_env`` in the
    top-level ``dark-factory-orchestrator.yaml`` (task 4635) on hosts known to
    have a writable /var/tmp and a landlock-capable kernel — an unwritable
    /var/tmp is an environment regression, not a reason to skip: quietly
    dropping the real-kernel enforcement surface (TestLandlockEnforcement,
    TestLandlockRefer, TestLandlockClaudeHomeNarrowing, all 12 matrix rows)
    would leave the suite green while every denial assertion stopped running.
    Fail loudly there.
    """
    if _var_tmp_writable():
        return False
    if os.environ.get('DF_REQUIRE_SANDBOX_TESTS') == '1':
        pytest.fail(
            f'DF_REQUIRE_SANDBOX_TESTS=1 but {_VAR_TMP_SKIP_REASON}: refusing to '
            'silently skip the real-kernel sandbox enforcement tests.',
            pytrace=False,
        )
    return True


def _skip_landlock() -> bool:
    """Whether landlock-dependent tests should be skipped in this environment.

    The mirror of ``_skip_var_tmp`` above, for the OTHER arm of the two-part
    guard every real-kernel row carries: those classes are gated on BOTH a
    writable /var/tmp AND a landlock-capable kernel, so arming only the
    /var/tmp arm would still let all 30 enforcement rows vanish silently the
    moment the availability probe went False — the same green-suite /
    empty-surface failure ``DF_REQUIRE_SANDBOX_TESTS`` exists to prevent,
    merely relocated to the other guard (task 4635 amendment). And that arm is
    the MORE likely of the two on a stable host: ``is_landlock_available()`` is
    first-party code — a syscall probe exercised by this very module — so a
    regression IN THE PROBE takes the surface down just as effectively as a
    kernel that dropped Landlock.

    Under ``DF_REQUIRE_SANDBOX_TESTS=1`` — armed via ``verify_env`` in the
    top-level ``dark-factory-orchestrator.yaml``, on hosts known to have a
    writable /var/tmp and a landlock-capable kernel (this one measured ABI 8) —
    an unavailable landlock is therefore an environment-or-probe regression,
    not a reason to skip. Fail loudly there.

    Deliberately NOT applied to ``TestLandlockRefer``'s ``ABI < 2`` arm: an
    ABI-1 kernel HAS landlock and the wrapper correctly omits REFER there, so
    that skip is a genuine capability difference rather than a missing
    enforcement surface, and it stays quiet even when armed.
    """
    if is_landlock_available():
        return False
    if os.environ.get('DF_REQUIRE_SANDBOX_TESTS') == '1':
        pytest.fail(
            f'DF_REQUIRE_SANDBOX_TESTS=1 but {_LANDLOCK_SKIP_REASON}: refusing to '
            'silently skip the real-kernel sandbox enforcement tests.',
            pytrace=False,
        )
    return True


# ---------------------------------------------------------------------------
# Test 1: is_landlock_available probe semantics
# ---------------------------------------------------------------------------

class TestIsLandlockAvailable:
    def test_returns_true_when_abi_positive(self):
        with patch.object(landlock_mod, '_syscall_probe_abi', return_value=7):
            assert is_landlock_available() is True

    def test_returns_false_when_abi_negative(self):
        with patch.object(landlock_mod, '_syscall_probe_abi', return_value=-1):
            assert is_landlock_available() is False

    def test_returns_false_when_abi_zero(self):
        with patch.object(landlock_mod, '_syscall_probe_abi', return_value=0):
            assert is_landlock_available() is False

    def test_caches_result(self):
        with patch.object(landlock_mod, '_syscall_probe_abi', return_value=7) as mock_probe:
            assert is_landlock_available() is True
            assert is_landlock_available() is True
            mock_probe.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2: build_landlock_command argv shape
# ---------------------------------------------------------------------------

class TestBuildLandlockCommand:
    def test_wrapper_argv_shape(self, tmp_path: Path):
        worktree = tmp_path
        (worktree / 'mod_a').mkdir()
        (worktree / 'mod_b').mkdir()

        inner = ['claude', '--print', '--model', 'haiku']
        cmd = build_landlock_command(
            inner, worktree, ['mod_a', 'mod_b'], writable_extras=['/var/tmp/extra'],
        )

        assert cmd[0] == sys.executable
        # Wrapper script path
        assert cmd[1].endswith('landlock_exec.py')
        # Contains the per-module writable flags
        assert '--writable' in cmd
        w_idx = [i for i, x in enumerate(cmd) if x == '--writable']
        w_values = {cmd[i + 1] for i in w_idx}
        assert str((worktree / 'mod_a').resolve()) in w_values
        assert str((worktree / 'mod_b').resolve()) in w_values
        assert '/var/tmp/extra' in w_values
        # .task is always writable
        assert str((worktree / '.task').resolve()) in w_values
        # Terminates with `--` then the inner command
        assert '--' in cmd
        dash_idx = cmd.index('--')
        assert cmd[dash_idx + 1:] == inner

    def test_creates_module_and_task_dirs(self, tmp_path: Path):
        worktree = tmp_path
        assert not (worktree / 'mod_a').exists()
        assert not (worktree / '.task').exists()

        build_landlock_command(['claude'], worktree, ['mod_a'])

        assert (worktree / 'mod_a').is_dir()
        assert (worktree / '.task').is_dir()


# ---------------------------------------------------------------------------
# Test 3: integration — wrapper actually enforces the sandbox
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_skip_landlock(), reason=_LANDLOCK_SKIP_REASON)
@pytest.mark.skipif(_skip_var_tmp(), reason=_VAR_TMP_SKIP_REASON)
class TestLandlockEnforcement:
    def test_allowed_and_denied_writes(self):
        # /var/tmp — outside the wrapper's default writable /tmp. pytest's
        # tmp_path lives under /tmp, where landlock_exec grants blanket write
        # access for agent scratch, so we can't use it as a worktree for
        # enforcement testing.
        base = Path(tempfile.mkdtemp(prefix='landlock-test-', dir='/var/tmp'))
        try:
            worktree = base / 'wt'
            worktree.mkdir()
            (worktree / 'mod_a').mkdir()
            (worktree / 'mod_b').mkdir()  # sibling — must be denied
            denied_parent = base / 'outside'
            denied_parent.mkdir()
            self._run_enforcement(worktree, denied_parent)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def _run_enforcement(self, worktree: Path, denied_parent: Path) -> None:

        inner = [
            '/bin/sh', '-c',
            (
                f'touch {worktree}/mod_a/ok_allowed && '
                f'(touch {worktree}/mod_b/bad_sibling 2>/dev/null || echo sibling_denied) && '
                f'(touch {denied_parent}/bad_outside 2>/dev/null || echo outside_denied) && '
                'echo end'
            ),
        ]
        cmd = build_landlock_command(inner, worktree, ['mod_a'])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        assert result.returncode == 0, f'stderr={result.stderr}'
        assert (worktree / 'mod_a' / 'ok_allowed').exists()
        assert not (worktree / 'mod_b' / 'bad_sibling').exists()
        assert not (denied_parent / 'bad_outside').exists()
        assert 'sibling_denied' in result.stdout
        assert 'outside_denied' in result.stdout
        assert 'end' in result.stdout


# ---------------------------------------------------------------------------
# Test 3b: REFER (cross-directory rename) — fleet-outage regression guard
# ---------------------------------------------------------------------------

# Two separate marks, not one combined condition: landlock being ENTIRELY
# absent is an environment regression (loud under DF_REQUIRE_SANDBOX_TESTS=1,
# via _skip_landlock), whereas ABI 1 is not — that kernel has landlock and the
# wrapper correctly omits REFER — so each arm keeps its own reason and its own
# armed/quiet disposition.
@pytest.mark.skipif(_skip_landlock(), reason=_LANDLOCK_SKIP_REASON)
@pytest.mark.skipif(
    landlock_mod._syscall_probe_abi() < 2,
    reason='landlock ABI < 2 has no REFER; the wrapper correctly omits it there',
)
@pytest.mark.skipif(_skip_var_tmp(), reason=_VAR_TMP_SKIP_REASON)
class TestLandlockRefer:
    """Guard against re-omitting LANDLOCK_ACCESS_FS_REFER from the ruleset.

    When ``handled_access_fs`` omits REFER, the kernel denies *every* rename
    that reparents a file across directories — surfaced to userspace as EXDEV,
    even within one granted tree on a single filesystem. rustc's
    ``encode_and_write_metadata`` writes ``.rmeta`` into a temp subdirectory
    and renames it up one level, so that omission broke every ``cargo
    build/check/test`` under the sandbox, on untouched dependency crates as
    much as on edited ones. It took the whole fleet down on 2026-07-29.

    The second test is the other half of the contract: REFER must restore
    rename *within* the writable set without widening that set.
    """

    def test_cross_directory_rename_inside_writable_tree_is_allowed(self):
        base = Path(tempfile.mkdtemp(prefix='landlock-refer-', dir='/var/tmp'))
        try:
            worktree = base / 'wt'
            worktree.mkdir()
            mod = worktree / 'mod_a'
            (mod / 'sub').mkdir(parents=True)
            (mod / 'sub' / 'f').write_text('x')

            inner = [
                '/bin/sh', '-c',
                f'mv {mod}/sub/f {mod}/f_moved && echo refer_ok',
            ]
            cmd = build_landlock_command(inner, worktree, ['mod_a'])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

            assert result.returncode == 0, f'stderr={result.stderr}'
            assert 'refer_ok' in result.stdout
            assert (mod / 'f_moved').exists()
            assert not (mod / 'sub' / 'f').exists()
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_refer_does_not_widen_the_write_boundary(self):
        """Renaming OUT of the writable set stays denied even with REFER."""
        base = Path(tempfile.mkdtemp(prefix='landlock-refer-esc-', dir='/var/tmp'))
        try:
            worktree = base / 'wt'
            worktree.mkdir()
            mod = worktree / 'mod_a'
            mod.mkdir()
            (mod / 'f').write_text('x')
            outside = base / 'outside'
            outside.mkdir()

            inner = [
                '/bin/sh', '-c',
                f'(mv {mod}/f {outside}/escaped 2>/dev/null || echo escape_denied) && echo end',
            ]
            cmd = build_landlock_command(inner, worktree, ['mod_a'])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

            assert result.returncode == 0, f'stderr={result.stderr}'
            assert 'escape_denied' in result.stdout
            assert not (outside / 'escaped').exists()
            assert (mod / 'f').exists()
        finally:
            shutil.rmtree(base, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: invoke.py dispatches to the correct backend
# ---------------------------------------------------------------------------

class TestInvokeBackendDispatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('backend,expect_bwrap,expect_landlock', [
        ('bwrap', True, False),
        ('landlock', False, True),
        ('none', False, False),
    ])
    async def test_dispatches_by_backend(
        self, tmp_path: Path, backend: str, expect_bwrap: bool, expect_landlock: bool,
    ):
        from orchestrator.agents import sandbox_dispatch

        # Root conftest's _restore_sandbox_backend autouse fixture restores
        # the prior backend after this test, so no try/finally needed here.
        sandbox_dispatch.set_backend(cast(Backend, backend))
        with patch(
            'orchestrator.agents.sandbox.is_bwrap_available', return_value=True,
        ), patch(
            'orchestrator.agents.sandbox.build_bwrap_command',
            return_value=['bwrap-wrapped', 'claude'],
        ) as mock_bwrap, patch(
            'orchestrator.agents.landlock.is_landlock_available', return_value=True,
        ), patch(
            'orchestrator.agents.landlock.build_landlock_command',
            return_value=['landlock-wrapped', 'claude'],
        ) as mock_landlock, patch(
            'asyncio.create_subprocess_exec', new_callable=AsyncMock,
        ) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b'{"result": "ok"}', b'')
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            from orchestrator.agents.invoke import invoke_agent
            await invoke_agent(
                prompt='test', system_prompt='sys', cwd=tmp_path,
                sandbox_modules=['mod_a'],
            )

            assert mock_bwrap.called == expect_bwrap
            assert mock_landlock.called == expect_landlock


# ---------------------------------------------------------------------------
# Test 5: SandboxConfig has a backend field
# ---------------------------------------------------------------------------

class TestSandboxConfigBackendField:
    def test_default_backend_is_auto(self):
        from orchestrator.config import SandboxConfig
        cfg = SandboxConfig()
        assert cfg.backend == 'auto'

    def test_accepts_landlock(self):
        from orchestrator.config import SandboxConfig
        cfg = SandboxConfig(backend='landlock')
        assert cfg.backend == 'landlock'

    def test_accepts_bwrap(self):
        from orchestrator.config import SandboxConfig
        cfg = SandboxConfig(backend='bwrap')
        assert cfg.backend == 'bwrap'

    def test_accepts_none(self):
        from orchestrator.config import SandboxConfig
        cfg = SandboxConfig(backend='none')
        assert cfg.backend == 'none'

    def test_rejects_garbage(self):
        from orchestrator.config import SandboxConfig
        with pytest.raises(ValidationError):
            SandboxConfig(backend='seccomp-magic')  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 6: ~/.claude narrowing — deny settings.json, allow fleet/, allow
# redirected in-worktree transcript writes (PRD enforcement matrix rows 9/10)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_skip_landlock(), reason=_LANDLOCK_SKIP_REASON)
@pytest.mark.skipif(_skip_var_tmp(), reason=_VAR_TMP_SKIP_REASON)
class TestLandlockClaudeHomeNarrowing:
    def test_denies_settings_but_allows_fleet_and_transcript(self):
        # /var/tmp — outside the wrapper's blanket /tmp grant, so writes here
        # are governed only by the rules under test. HOME is overridden to a
        # fake home (below) so this never touches the real ~/.claude.
        base = Path(tempfile.mkdtemp(prefix='landlock-claude-home-', dir='/var/tmp'))
        try:
            fake_home = base / 'home'
            fake_home.mkdir()
            fleet_dir = fake_home / '.claude' / 'fleet'
            fleet_dir.mkdir(parents=True)
            settings_path = fake_home / '.claude' / 'settings.json'
            settings_path.write_text(json.dumps({'orig': True}))

            worktree = base / 'wt'
            worktree.mkdir()

            self._run_enforcement(fake_home, fleet_dir, settings_path, worktree)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def _run_enforcement(
        self, fake_home: Path, fleet_dir: Path, settings_path: Path, worktree: Path,
    ) -> None:
        # Mirrors the production compute_write_set() -> sandbox_extras ->
        # build_landlock_command(writable_extras=...) chain: fleet/ is the
        # only ~/.claude subpath granted, passed the same way the real
        # workflow wiring passes it.
        transcript_dir = worktree / '.task' / 'claude-config-x' / 'projects'
        poisoned = json.dumps({'pwned': True})

        inner = [
            '/bin/sh', '-c',
            (
                f'(touch {fleet_dir}/rec 2>/dev/null && echo fleet_ok || echo fleet_denied) && '
                f"(printf '%s' '{poisoned}' > {settings_path} 2>/dev/null && "
                'echo settings_wrote || echo settings_denied) && '
                f'(mkdir -p {transcript_dir} 2>/dev/null && '
                f'printf transcript > {transcript_dir}/transcript.jsonl 2>/dev/null && '
                'echo transcript_ok || echo transcript_denied) && '
                'echo end'
            ),
        ]
        cmd = build_landlock_command(
            inner, worktree, [], writable_extras=[str(fleet_dir)],
        )
        result = subprocess.run(
            cmd,
            env={**os.environ, 'HOME': str(fake_home)},
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert result.returncode == 0, f'stderr={result.stderr}'
        assert 'end' in result.stdout

        # Row 10 (over-narrow guard): the fleet/ extra stays writable.
        assert 'fleet_ok' in result.stdout, result.stdout

        # Row 9 (the property under test): settings.json is NOT writable via
        # the narrowed ~/.claude grant, and its content is left untouched.
        assert 'settings_denied' in result.stdout, result.stdout
        assert 'settings_wrote' not in result.stdout
        assert json.loads(settings_path.read_text()) == {'orig': True}

        # Acceptance: redirected in-worktree transcript writes still work.
        assert 'transcript_ok' in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# Test 7: an agent concludes a conflicted merge inside the sandbox (task 5844)
# ---------------------------------------------------------------------------

_CONFLICT_BASE = '1\n2\n3\n'
_CONFLICT_RESOLVED = '1\nRESOLVED\n3\n'
_AGENT_SUCCESS_JSON = json.dumps({
    'result': 'ok', 'is_error': False, 'subtype': 'success', 'cost_usd': 0.0,
    'duration_ms': 1, 'num_turns': 1, 'session_id': 's',
})


def _scaffold_git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ['git', *args], cwd=str(cwd), capture_output=True, text=True, timeout=30,
        env={**os.environ, 'GIT_CEILING_DIRECTORIES': str(cwd.resolve().parent)},
    )
    assert proc.returncode == 0, f'git {" ".join(args)} failed: {proc.stderr}'
    return proc.stdout


@pytest.mark.skipif(_skip_landlock(), reason=_LANDLOCK_SKIP_REASON)
@pytest.mark.skipif(_skip_var_tmp(), reason=_VAR_TMP_SKIP_REASON)
class TestLandlockSandboxedConflictedMerge:
    """A sandboxed agent resolves a NEW merge conflict and commits it cleanly.

    The common dir's rr-cache is read-only to the sandbox, so under half-on
    rerere the concluding commit died rc 139 and left MERGE_RR.lock plus a
    dangling MERGE_RR record behind. Real production chain: compute_write_set
    -> invoke_agent -> wrap_command -> build_landlock_command.
    """

    @pytest.mark.asyncio
    async def test_agent_commits_conflicted_merge_without_rerere_debris(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        from orchestrator.agents import sandbox_dispatch
        from orchestrator.agents.invoke import invoke_agent
        from orchestrator.agents.write_set import compute_write_set
        from orchestrator.rebase_recovery import scan_merge_rr

        # /var/tmp, never /tmp: landlock_exec blanket-grants /tmp, which would
        # make rr-cache writable and this test vacuous.
        base = Path(tempfile.mkdtemp(prefix='landlock-rerere-', dir='/var/tmp'))
        try:
            name = 'wt'
            main, worktree, fake_home = self._scaffold(base, name)
            monkeypatch.setenv('PATH', f'{base / "bin"}{os.pathsep}{os.environ.get("PATH", "")}')
            monkeypatch.setenv('HOME', str(fake_home))
            for var in ('DF_AGENT_CPU_NICE', 'DF_AGENT_CPU_GOVERN'):
                monkeypatch.delenv(var, raising=False)
            sandbox_dispatch.set_backend('landlock')
            write_set = compute_write_set(worktree, home=fake_home)
            extras = [str(p) for p in write_set.writable_paths()]

            self._assert_rr_cache_denied(main, worktree, fake_home, extras)
            assert _scaffold_git(worktree, 'config', '--get', 'rerere.enabled').strip() == 'true', (
                'premise broken: git already reads rerere off in the env invoke_agent '
                'inherits, so this test cannot observe invoke_agent turning it off'
            )

            result = await invoke_agent(
                prompt='resolve', system_prompt='s', cwd=worktree, model='opus',
                max_turns=1, max_budget_usd=0.01, timeout_seconds=60,
                sandbox_modules=[], sandbox_extras=extras,
            )

            assert result.success, f'fake claude dispatch did not succeed: {result!r}'
            rcs = dict(
                item.split('=') for item in (worktree / '.task' / 'merge-rcs').read_text().split()
            )
            assert rcs['merge'] == '1', (
                f'the merge must stop on a NEW conflict, or this test is vacuous: {rcs}'
            )
            assert rcs['add'] == '0', rcs
            assert rcs['commit'] == '0', (
                f'the concluding commit must succeed inside the sandbox: {rcs}'
            )
            admin = main / '.git' / 'worktrees' / name
            assert not (admin / 'MERGE_RR.lock').exists(), 'stale MERGE_RR.lock left behind'
            # A 0-byte MERGE_RR from the unsandboxed setup commit is legitimate;
            # the defect is a RECORD.
            assert scan_merge_rr(git_dir=admin, common_dir=main / '.git').records == ()
            parents = _scaffold_git(worktree, 'rev-list', '--parents', '-n1', 'HEAD').split()
            assert len(parents) == 3, f'expected a two-parent merge commit, got {parents}'
            assert (worktree / 'f.txt').read_text() == _CONFLICT_RESOLVED
        finally:
            shutil.rmtree(base, ignore_errors=True)

    @staticmethod
    def _scaffold(base: Path, name: str) -> tuple[Path, Path, Path]:
        main = base / 'main'
        main.mkdir()
        _scaffold_git(main, 'init', '-q', '-b', 'main')
        for key, value in (
            ('user.email', 'landlock-rerere@test.invalid'),
            ('user.name', 'Landlock Rerere Test'),
            ('commit.gpgsign', 'false'),
            ('rerere.enabled', 'true'),
            ('rerere.autoupdate', 'true'),
            ('gc.auto', '0'),
            ('maintenance.auto', 'false'),
        ):
            _scaffold_git(main, 'config', key, value)
        (main / 'f.txt').write_text(_CONFLICT_BASE)
        _scaffold_git(main, 'add', 'f.txt')
        _scaffold_git(main, 'commit', '-qm', 'base')

        worktree = base / name
        _scaffold_git(main, 'worktree', 'add', '-q', '-b', f'task/{name}', str(worktree), 'main')
        (main / 'f.txt').write_text('1\nMAIN\n3\n')
        _scaffold_git(main, 'commit', '-qam', 'main edit')
        (worktree / 'f.txt').write_text('1\nTASK\n3\n')
        _scaffold_git(worktree, 'commit', '-qam', 'task edit')

        (base / '.task-meta' / name).mkdir(parents=True)
        fake_home = base / 'home'
        (fake_home / '.cache' / 'uv').mkdir(parents=True)
        (fake_home / '.claude' / 'fleet').mkdir(parents=True)

        bin_dir = base / 'bin'
        bin_dir.mkdir()
        agent = bin_dir / 'claude'
        agent.write_text(
            '#!/bin/sh\n'
            'cat >/dev/null\n'
            'git merge --no-ff main >/dev/null 2>&1; m=$?\n'
            "printf '1\\nRESOLVED\\n3\\n' > f.txt\n"
            'git add f.txt; a=$?\n'
            'git commit --no-edit >/dev/null 2>&1; c=$?\n'
            'mkdir -p .task\n'
            'printf "merge=%s add=%s commit=%s\\n" "$m" "$a" "$c" > .task/merge-rcs\n'
            f"printf '%s\\n' '{_AGENT_SUCCESS_JSON}'\n"
        )
        agent.chmod(0o755)
        return main, worktree, fake_home

    @staticmethod
    def _assert_rr_cache_denied(
        main: Path, worktree: Path, fake_home: Path, extras: list[str],
    ) -> None:
        """Premise guard: rr-cache really is read-only to this sandbox.

        rr-cache is created OUTSIDE the sandbox first, so the denial below is a
        landlock EACCES rather than ENOENT.
        """
        rr_cache = main / '.git' / 'rr-cache'
        rr_cache.mkdir(exist_ok=True)
        probe = rr_cache / 'premise-probe'
        guard = subprocess.run(
            build_landlock_command(['mkdir', str(probe)], worktree, [], writable_extras=extras),
            env={**os.environ, 'HOME': str(fake_home)},
            capture_output=True, text=True, timeout=15,
        )
        assert guard.returncode != 0 and not probe.exists(), (
            'premise broken: the sandbox could write rr-cache, so this test cannot '
            f'observe the half-on rerere defect (rc={guard.returncode}, stderr={guard.stderr})'
        )
