"""T2: verify-admission wiring (task 2390; PRD
plans/verify-oversubscription-control-prd.md).

Covers the test-leg-only admission gate wired into
``orchestrator.verify._run_or_skip_timed``:

- step-5/step-6: acquire wiring (flock semaphore via
  ``shared.verify_admission.acquire_task_slot``)
- step-7/step-8: role nice-prefix wrapping (``shared.verify_admission.nice_prefix``)

Every test here is marked ``@pytest.mark.real_verify_admission`` to opt out of
the autouse ``_neutralize_verify_admission`` conftest fixture, which otherwise
forces the module seam ``orchestrator.verify._verify_admission_active`` to
False for every other test in the suite (task 2390 pre-1).

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_psi_admission_reload.py's stated rationale).
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
from unittest.mock import patch

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_verification

# module_config commands used across these tests are chosen to be uniquely
# identifiable by exact string match, so a spy `_run_cmd` can label which leg
# is running without needing `label` (which is a `_run_or_skip_timed`-local
# closure variable, never passed down to `_run_cmd`).
_TEST_CMD = 'pytest tests/'
_LINT_CMD = 'ruff'
_TYPE_CMD = 'pyright'
_LEG_BY_CMD = {_TEST_CMD: 'test', _LINT_CMD: 'lint', _TYPE_CMD: 'type'}


def _module_config(**overrides) -> ModuleConfig:
    kwargs = dict(
        prefix='pkg',
        test_command=_TEST_CMD,
        lint_command=_LINT_CMD,
        type_check_command=_TYPE_CMD,
        # Sequential so the three legs run strictly test -> lint -> type,
        # making ordering assertions deterministic (no gather interleaving).
        concurrent_verify=False,
    )
    kwargs.update(overrides)
    return ModuleConfig(**kwargs)


class TestVerifyAdmissionAcquireWiring:
    """Acquire wiring: test leg only, gated by config.verify_admission_enabled."""

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_acquire_called_once_before_test_leg_only(self, tmp_path):
        events: list = []

        @contextlib.contextmanager
        def fake_acquire(role, *, slots_dir, n, wait=True):
            events.append(('acquire', role, slots_dir, n, wait))
            yield True

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            events.append(f'run:{_LEG_BY_CMD.get(cmd, cmd)}')
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with patch('orchestrator.verify.acquire_task_slot', side_effect=fake_acquire) as mock_acquire, \
                patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(),
                role='task',
                attempt_id=None,
            )

        mock_acquire.assert_called_once_with('task', slots_dir=slots_dir, n=1, wait=True)
        assert events == [
            ('acquire', 'task', slots_dir, 1, True),
            'run:test',
            'run:lint',
            'run:type',
        ], f'expected acquire before run:test, no acquire before lint/type; got {events!r}'
        assert slots_dir.is_dir(), 'admission wiring must mkdir slots_dir itself (T1 never does)'

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_disabled_no_acquire_no_dir_byte_identical_cmd(self, tmp_path):
        captured_cmds: list[str] = []

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_cmds.append(cmd)
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_enabled=False,
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with patch('orchestrator.verify.acquire_task_slot') as mock_acquire, \
                patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(),
                role='task',
                attempt_id=None,
            )

        mock_acquire.assert_not_called()
        assert not slots_dir.exists(), 'disabled admission must never create slots_dir'
        assert captured_cmds[0] == _TEST_CMD, 'disabled admission must send the byte-identical, unwrapped cmd'

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_merge_role_acquires_uniformly_but_run_cmd_still_invoked(self, tmp_path):
        events: list = []

        @contextlib.contextmanager
        def fake_acquire(role, *, slots_dir, n, wait=True):
            events.append(('acquire', role))
            yield False  # T1 C-merge-priority: merge never actually holds a slot

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            events.append(f'run:{_LEG_BY_CMD.get(cmd, cmd)}')
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with patch('orchestrator.verify.acquire_task_slot', side_effect=fake_acquire) as mock_acquire, \
                patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(),
                role='merge',
                attempt_id=None,
            )

        mock_acquire.assert_called_once_with('merge', slots_dir=slots_dir, n=1, wait=True)
        assert events[0] == ('acquire', 'merge')
        assert 'run:test' in events, 'merge verify must still run the test leg (T1 no-op bypass, not a skip)'

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_fail_open_when_slots_dir_unmkdirable(self, tmp_path):
        run_cmd_calls: list[str] = []

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            run_cmd_calls.append(cmd)
            return 0, '', False

        blocker_file = tmp_path / 'not_a_dir'
        blocker_file.write_text('x')
        # Parent path component is a FILE, not a directory -> mkdir(parents=True) raises OSError.
        slots_dir = blocker_file / 'slots'

        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        # REAL acquire_task_slot (not mocked) — only _run_cmd is patched, so this
        # exercises T1's actual code path in addition to the mkdir fail-open.
        with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(),
                role='task',
                attempt_id=None,
            )

        assert run_cmd_calls == [_TEST_CMD, _LINT_CMD, _TYPE_CMD], (
            'an unmkdir-able slots_dir must fail open (verify still runs), not raise'
        )

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_real_acquire_serializes_test_leg_across_concurrent_verifies(self, tmp_path):
        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )

        current = 0
        max_seen = 0

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            nonlocal current, max_seen
            if cmd != _TEST_CMD:
                return 0, '', False
            current += 1
            max_seen = max(max_seen, current)
            await asyncio.sleep(0.05)
            current -= 1
            return 0, '', False

        worktree_a = tmp_path / 'wt-a'
        worktree_a.mkdir()
        worktree_b = tmp_path / 'wt-b'
        worktree_b.mkdir()

        # REAL acquire_task_slot: proves both flock serialization (max_seen==1)
        # AND that the blocking poll-wait is off the event loop (a loop-blocking
        # acquire would either push max_seen to 2 or deadlock the second waiter,
        # since the first holder's release depends on its own coroutine — sharing
        # the same loop — getting scheduled).
        with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            results = await asyncio.gather(
                run_verification(
                    worktree=worktree_a, config=config, module_config=_module_config(),
                    role='task', attempt_id=None,
                ),
                run_verification(
                    worktree=worktree_b, config=config, module_config=_module_config(),
                    role='task', attempt_id=None,
                ),
            )

        assert max_seen == 1, f'test leg ran concurrently across verifies (max_seen={max_seen})'
        assert len(results) == 2
        assert all(r.passed for r in results)


class TestResolveNicePrefix:
    """Pure-unit tests: canonical T1 tiers by default, per-role config
    override (shlex-split) when the corresponding knob is non-empty.
    """

    @pytest.mark.real_verify_admission
    def test_task_matches_canonical_tier(self):
        from orchestrator.verify import _resolve_nice_prefix

        config = OrchestratorConfig()
        assert _resolve_nice_prefix(config, 'task') == shlex.split('nice -n 15 ionice -c2 -n7')

    @pytest.mark.real_verify_admission
    def test_merge_matches_canonical_tier(self):
        from orchestrator.verify import _resolve_nice_prefix

        config = OrchestratorConfig()
        assert _resolve_nice_prefix(config, 'merge') == ['nice', '-n', '5']

    @pytest.mark.real_verify_admission
    def test_background_matches_canonical_tier(self):
        from orchestrator.verify import _resolve_nice_prefix

        config = OrchestratorConfig()
        assert _resolve_nice_prefix(config, 'background') == ['nice', '-n', '19', 'ionice', '-c3']

    @pytest.mark.real_verify_admission
    @pytest.mark.parametrize('role', ['offline', 'some-unknown-role'])
    def test_offline_and_unknown_role_have_no_prefix(self, role):
        from orchestrator.verify import _resolve_nice_prefix

        config = OrchestratorConfig()
        assert _resolve_nice_prefix(config, role) == []

    @pytest.mark.real_verify_admission
    def test_task_override_takes_precedence_over_canonical_tier(self):
        from orchestrator.verify import _resolve_nice_prefix

        config = OrchestratorConfig(verify_admission_nice_task='nice -n 3')
        assert _resolve_nice_prefix(config, 'task') == ['nice', '-n', '3']


class TestNicePrefixIntegration:
    """Integration via run_verification + spy _run_cmd: the test leg is
    wrapped ``<nice argv> /bin/bash -c <shlex.quote(cmd)>``; lint/type ride
    un-niced; disabled admission leaves the test leg byte-identical.
    """

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_task_role_wraps_test_leg_with_nice_and_bash_c(self, tmp_path):
        captured_cmds: list[str] = []

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_cmds.append(cmd)
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(),
                role='task',
                attempt_id=None,
            )

        assert captured_cmds[0] == 'nice -n 15 ionice -c2 -n7 /bin/bash -c ' + shlex.quote(_TEST_CMD)
        assert captured_cmds[1] == _LINT_CMD, 'lint leg must not be niced'
        assert captured_cmds[2] == _TYPE_CMD, 'type leg must not be niced'

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_merge_role_wraps_test_leg_with_its_own_nice_tier(self, tmp_path):
        captured_cmds: list[str] = []

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_cmds.append(cmd)
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(),
                role='merge',
                attempt_id=None,
            )

        assert captured_cmds[0] == 'nice -n 5 /bin/bash -c ' + shlex.quote(_TEST_CMD)
        assert captured_cmds[1] == _LINT_CMD, 'lint leg must not be niced'
        assert captured_cmds[2] == _TYPE_CMD, 'type leg must not be niced'

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_disabled_admission_leaves_test_leg_byte_identical(self, tmp_path):
        captured_cmds: list[str] = []

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_cmds.append(cmd)
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_enabled=False,
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(),
                role='task',
                attempt_id=None,
            )

        assert captured_cmds[0] == _TEST_CMD, 'disabled admission must never nice/bash-c wrap the test leg'
        assert captured_cmds[1] == _LINT_CMD
        assert captured_cmds[2] == _TYPE_CMD
