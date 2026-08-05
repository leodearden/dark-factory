"""T6: benchmark-tuned default `-n` for verify admission (task 2394; PRD
plans/verify-oversubscription-control-prd.md T6, follow-up to T2/task 2390).

Covers the new ``verify_admission_pytest_n`` knob end to end:

- step-2/3: config surface + green-tier reload registration (mirrors
  test_config_verify_admission_reload.py).
- step-4/5: the ``verify_cmd.apply_pytest_numprocesses`` mutator (mirrors
  test_verify_cmd.py::TestSerialPytest).
- step-6/7: wiring into ``orchestrator.verify._run_or_skip_timed`` — the
  test-leg-only, role-gated (task/background, not merge) ``-n`` rewrite
  (mirrors test_verify_admission_wiring.py).

The default (`'auto'`) is the T6 benchmark report's recommendation —
see plans/verify-oversubscription-benchmark-2026-07-14.md — not an
arbitrary placeholder; it is behavior-preserving (byte-identical to
pre-T6 `-n auto` addopts) because no clean idle-window measurement
supported a specific worker-count cap on this host.
"""

from __future__ import annotations

import shlex
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    ModuleConfig,
    OrchestratorConfig,
    apply_reload,
    diff_config,
)
from orchestrator.verify import run_verification
from orchestrator.verify_cmd import (
    ToolKind,
    apply_pytest_numprocesses,
    parse_config_command,
    render,
    serial_pytest,
)

# Module-local wiring-test fixtures (not conftest.py — a conftest.py edit
# trips verify.py's has_conftest; mirrors test_verify_admission_wiring.py's
# stated rationale). Deliberately duplicated rather than imported from that
# module: each admission test file is self-contained.
_TEST_CMD = 'pytest tests/'
_LINT_CMD = 'ruff'
_TYPE_CMD = 'pyright'


def _leg_for_cmd(cmd: str) -> str:
    """Label which leg *cmd* belongs to. Checks 'pytest'/'tests/' as two
    separate substrings (not the single _TEST_CMD substring test_verify_
    admission_wiring.py's version checks) because a `-n` injection splices
    new flags between them (``pytest tests/`` -> ``pytest -n 16 tests/``),
    breaking containment of the whole ``_TEST_CMD`` string.
    """
    if 'pytest' in cmd and 'tests/' in cmd:
        return 'test'
    if _LINT_CMD in cmd:
        return 'lint'
    if _TYPE_CMD in cmd:
        return 'type'
    return cmd


def _module_config(**overrides: Any) -> ModuleConfig:
    kwargs: dict[str, Any] = dict(
        prefix='pkg',
        test_command=_TEST_CMD,
        lint_command=_LINT_CMD,
        type_check_command=_TYPE_CMD,
        concurrent_verify=False,
    )
    kwargs.update(overrides)
    return ModuleConfig(**kwargs)


class TestVerifyAdmissionPytestNConfigDefault:
    """Behavior-preserving default: 'auto' means no `-n` rewrite is injected
    (see the wiring tests below), so an untuned install keeps today's
    `-n auto` pyproject addopts byte-for-byte.
    """

    def test_default_is_str_auto(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        assert isinstance(cfg.verify_admission_pytest_n, str)
        assert cfg.verify_admission_pytest_n == 'auto'


class TestVerifyAdmissionPytestNValidation:
    """A malformed value is rejected at construction/reload rather than
    reaching pytest-xdist unvalidated and only failing the whole test leg
    the next time a task/background verify runs (mirrors
    TestVerifyAdmissionTaskSlotsValidation in test_config_verify_admission_
    reload.py). Accepted values mirror pytest-xdist's own `-n` grammar plus
    this knob's own '' no-op sentinel.
    """

    @pytest.mark.parametrize('bad_value', ['1six', 'sixteen', '0', '-1', '3.5', ' 16'])
    def test_malformed_value_rejected(self, bad_value):
        with pytest.raises(ValidationError):
            OrchestratorConfig(verify_admission_pytest_n=bad_value)

    @pytest.mark.parametrize('good_value', ['', 'auto', 'logical', '1', '16'])
    def test_valid_values_accepted(self, good_value):
        cfg = OrchestratorConfig(verify_admission_pytest_n=good_value)
        assert cfg.verify_admission_pytest_n == good_value


class TestVerifyAdmissionPytestNReloadDisposition:
    """Green-tier: hot-reloadable without a process restart, same tier as
    the other five verify_admission_* knobs.
    """

    def test_field_is_reloadable(self):
        assert 'verify_admission_pytest_n' in RELOADABLE_FIELDS, (
            "'verify_admission_pytest_n' is expected to be green-tier "
            'reloadable but is missing from RELOADABLE_FIELDS'
        )

    def test_edit_lands_in_applied_candidates_not_restart_required(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(verify_admission_pytest_n='auto')
        fresh = OrchestratorConfig(verify_admission_pytest_n='16')
        diff = diff_config(live, fresh)
        assert 'verify_admission_pytest_n' in diff.applied_candidates
        assert 'verify_admission_pytest_n' not in diff.restart_required

    def test_apply_reload_applies_in_place(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(verify_admission_pytest_n='auto')
        fresh = OrchestratorConfig(verify_admission_pytest_n='16')
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert report['applied']['verify_admission_pytest_n'] == {'old': 'auto', 'new': '16'}
        assert live.verify_admission_pytest_n == '16'


class TestApplyPytestNumprocesses:
    """apply_pytest_numprocesses(cmd, n) appends a `-n <n>` pytest-xdist
    worker-count flag. Mirrors test_verify_cmd.py::TestSerialPytest's shape.
    """

    def test_structured_single_invocation_appends_dash_n(self):
        cmd = parse_config_command('uv run pytest tests/')
        result = render(apply_pytest_numprocesses(cmd, '16'))
        assert result == 'uv run pytest -n 16 tests/'

    @pytest.mark.parametrize('n', ['auto', ''])
    def test_noop_values_leave_cmd_byte_identical(self, n):
        cmd = parse_config_command('uv run pytest tests/')
        mutated = apply_pytest_numprocesses(cmd, n)
        assert mutated == cmd
        assert render(mutated) == render(cmd)

    def test_rewrites_every_pytest_invocation_in_chained_command(self):
        raw = (
            'cd shared && uv run pytest tests/ && '
            'cd ../orchestrator && uv run pytest tests/'
        )
        cmd = parse_config_command(raw)
        assert cmd.tool is ToolKind.PYTEST
        assert cmd.raw == raw

        result = render(apply_pytest_numprocesses(cmd, '16'))
        assert result.count('pytest') == 2
        assert result.count('-n 16') == 2

    def test_non_pytest_command_returned_unchanged(self):
        cmd = parse_config_command('ruff check .')
        assert apply_pytest_numprocesses(cmd, '16') == cmd

    def test_noop_on_opaque(self):
        cmd = parse_config_command('mypy src/')
        assert apply_pytest_numprocesses(cmd, '16') == cmd


class TestWithPytestNumprocessesStr:
    """The string-level wrapper `verify._with_pytest_numprocesses_str` (task 3478).

    Sibling of ``_with_pytest_timeout_str`` / ``_with_junitxml_str``: parse ->
    mutate -> ``is`` identity check -> render. It exists because task 3478
    applies the `-n` cap at a SECOND site (per segment, inside
    ``_run_one_segment``) as well as the original non-segmented one, and two
    copies of the parse/render/identity dance would be free to drift
    (INV-5 no-lockstep-duplication).

    The identity check is the load-bearing part: a from-scratch ``render`` of
    an untouched parse is only argv-EQUIVALENT, not byte-identical (flag
    order and spacing get normalised), so a no-op that round-tripped would
    silently rewrite every operator's configured command. These tests assert
    ``is``, not ``==``, for exactly that reason.
    """

    @staticmethod
    def _wrapper():
        from orchestrator.verify import _with_pytest_numprocesses_str  # noqa: PLC0415

        return _with_pytest_numprocesses_str

    @pytest.mark.parametrize(
        ('cmd', 'n', 'why'),
        [
            (
                'cd shared && uv run pytest tests/ && cd ../orchestrator && uv run pytest tests/',
                'auto',
                "n='auto' is the shipped default and an apply_pytest_numprocesses no-op",
            ),
            ('uv run pytest tests/', '', "n='' is the other no-op value"),
            ('uv run ruff check src/', '4', 'a non-pytest tool is never -n capped'),
            ('mypy src/', '4', 'an OPAQUE command has no structure to mutate'),
            (
                "uv run pytest -p no:xdist -o addopts='' tests/",
                '4',
                'serial-forced: xdist is off, so -n would fail the run outright',
            ),
        ],
    )
    def test_noop_returns_the_very_same_object(self, cmd, n, why):
        assert self._wrapper()(cmd, n) is cmd, why

    def test_structured_pytest_gains_dash_n_and_keeps_its_flags(self):
        result = self._wrapper()('uv run pytest tests/ --timeout=300', '4')
        assert '-n 4' in result
        assert '--timeout=300' in result
        assert 'tests/' in result

    def test_none_in_none_out(self):
        assert self._wrapper()(None, '4') is None


class TestApplyPytestNumprocessesSerialRecoveryCollision:
    """Regression: `-n` must NOT be injected onto a command already forced
    serial (``-p no:xdist``). verify.py's env-transient recovery
    (verify.py:3458) and flaky-scoped isolated re-run (verify.py:5164) both
    build their recovered command via ``serial_pytest`` — which disables the
    xdist plugin — and then pass it back through ``_run_or_skip_timed``, which
    re-hits the ``apply_pytest_numprocesses`` injection site. With xdist
    disabled pytest does not register the ``-n``/``--numprocesses`` option, so
    injecting ``-n 16`` would make pytest exit with
    ``unrecognized arguments: -n``, defeating the shared-venv-transient
    recovery safety net precisely when the operator has set the knob to the
    recommended value. Dormant at the shipped 'auto' default, but the knob is
    green-tier/hot-reloadable — so setting it live must not break recovery.
    """

    def test_structured_serial_forced_command_is_not_dash_n_injected(self):
        # What verify.py:3458 (_serial_pytest_str) produces for a structured
        # command, fed back through the injection site with a numeric knob.
        recovered = serial_pytest(parse_config_command('uv run pytest tests/'))
        result = apply_pytest_numprocesses(recovered, '16')
        rendered = render(result)
        assert result is recovered, 'apply_pytest_numprocesses must no-op on a serial-forced command'
        assert '-p no:xdist' in rendered
        assert '-n 16' not in rendered

    def test_chained_serial_forced_command_is_not_dash_n_injected(self):
        raw = (
            'cd shared && uv run pytest tests/ && '
            'cd ../orchestrator && uv run pytest tests/'
        )
        recovered = serial_pytest(parse_config_command(raw))
        result = apply_pytest_numprocesses(recovered, '16')
        rendered = render(result)
        assert result is recovered, 'apply_pytest_numprocesses must no-op on a serial-forced chain'
        assert rendered.count('no:xdist') == 2
        assert '-n 16' not in rendered


class TestPytestNWiring:
    """Wiring into ``_run_or_skip_timed``: the test-leg-only, role-gated
    (task/background, not merge) `-n` rewrite, injected before the
    nice-prefix wrap. Mirrors test_verify_admission_wiring.py's spy/
    _leg_for_cmd/_module_config pattern.
    """

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    @pytest.mark.parametrize('role', ['task', 'background'])
    async def test_task_and_background_roles_get_dash_n_injected(self, tmp_path, role):
        captured_cmds: list[str] = []

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_cmds.append(cmd)
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
            verify_admission_pytest_n='16',
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(),
                role=role,
                attempt_id=None,
            )

        test_cmd = next(c for c in captured_cmds if _leg_for_cmd(c) == 'test')
        assert '-n 16' in test_cmd, f'expected -n 16 injected for role={role!r}; got {test_cmd!r}'

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_merge_role_excluded_from_dash_n_injection(self, tmp_path):
        captured_cmds: list[str] = []

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_cmds.append(cmd)
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
            verify_admission_pytest_n='16',
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

        test_cmd = next(c for c in captured_cmds if _leg_for_cmd(c) == 'test')
        assert '-n 16' not in test_cmd, (
            f"merge's test leg must never be -n-capped (bypasses admission "
            f'slot-counting, latency-critical); got {test_cmd!r}'
        )

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_auto_value_injects_nothing(self, tmp_path):
        captured_cmds: list[str] = []

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_cmds.append(cmd)
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
            verify_admission_pytest_n='auto',
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

        assert captured_cmds[0] == 'nice -n 15 ionice -c2 -n7 /bin/bash -c ' + shlex.quote(_TEST_CMD), (
            "verify_admission_pytest_n='auto' must inject no -n rewrite — the "
            'test leg must be byte-identical to pre-T6 (nice-wrap only)'
        )

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_admission_disabled_injects_nothing(self, tmp_path):
        captured_cmds: list[str] = []

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_cmds.append(cmd)
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_enabled=False,
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
            verify_admission_pytest_n='16',
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

        assert captured_cmds[0] == _TEST_CMD, (
            'disabled admission must never inject -n (or nice/bash-c wrap) the test leg'
        )

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('test_command', 'expected_inner'),
        [
            (
                'cd orchestrator && uv run pytest tests/ -x',
                'cd orchestrator && uv run pytest -x -n 16 tests/',
            ),
            (
                # The actual shape used in orchestrator/config.yaml's per-subproject
                # test_command legs today: an '='-joined value flag (a single
                # token), not a separate-token value flag. Confirms the
                # parse->render round-trip (verify.py:3326-3328) is
                # argv-equivalent to the input plus '-n 16' for a realistic,
                # currently-deployed module command shape.
                'cd orchestrator && uv run pytest tests/ --timeout=300',
                'cd orchestrator && uv run pytest --timeout=300 -n 16 tests/',
            ),
        ],
    )
    async def test_realistic_module_command_is_argv_equivalent_with_dash_n_appended(
        self, tmp_path, test_command, expected_inner
    ):
        captured_cmds: list[str] = []

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_cmds.append(cmd)
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
            verify_admission_pytest_n='16',
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(test_command=test_command),
                role='task',
                attempt_id=None,
            )

        test_cmd = next(c for c in captured_cmds if _leg_for_cmd(c) == 'test')
        # role='task' also gets the nice-prefix bash-c wrap (T2); the inner
        # payload is what apply_pytest_numprocesses/render produced.
        expected = 'nice -n 15 ionice -c2 -n7 /bin/bash -c ' + shlex.quote(expected_inner)
        assert test_cmd == expected, (
            f'expected the parse->render round-trip to be argv-equivalent to '
            f'{test_command!r} plus -n 16; got {test_cmd!r}'
        )

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_chained_module_command_gets_dash_n_on_each_invocation(self, tmp_path):
        chained_cmd = (
            'cd shared && uv run pytest tests/ && '
            'cd ../orchestrator && uv run pytest tests/'
        )
        captured_cmds: list[str] = []

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_cmds.append(cmd)
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
            verify_admission_pytest_n='16',
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()

        with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
            await run_verification(
                worktree=worktree,
                config=config,
                module_config=_module_config(test_command=chained_cmd),
                role='task',
                attempt_id=None,
            )

        test_cmd = next(c for c in captured_cmds if _leg_for_cmd(c) == 'test')
        assert test_cmd.count('-n 16') == 2, (
            f'expected -n 16 injected into both chained pytest invocations; got {test_cmd!r}'
        )

    @pytest.mark.real_verify_admission
    @pytest.mark.asyncio
    async def test_env_recovery_rerun_is_not_dash_n_injected(self, tmp_path):
        """End-to-end regression: on an ENV_TRANSIENT (vanished-xdist)
        failure, verify.py's forced-serial recovery re-run passes its
        ``-p no:xdist`` command back through the ``-n`` injection site. With
        xdist disabled pytest does not register the ``-n`` option, so a
        re-injected ``-n 16`` would fail the recovery run with
        ``unrecognized arguments: -n`` — defeating the shared-venv-transient
        safety net at exactly the recommended live knob value. Assert the
        recovered command is serial and carries NO ``-n 16``.
        """
        captured_cmds: list[str] = []
        test_leg_calls = {'n': 0}

        async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            captured_cmds.append(cmd)
            if 'pytest' in cmd and 'tests/' in cmd:
                test_leg_calls['n'] += 1
                if test_leg_calls['n'] == 1:
                    # First test leg: a shared-venv-mutation transient
                    # (concurrent `uv sync` vanished xdist) → ENV_TRANSIENT.
                    return 1, "ModuleNotFoundError: No module named 'xdist'", False
                # The forced-serial recovery re-run passes.
                return 0, '', False
            return 0, '', False

        slots_dir = tmp_path / 'slots'
        config = OrchestratorConfig(
            verify_admission_slots_dir=str(slots_dir),
            verify_admission_task_slots=1,
            verify_admission_pytest_n='16',
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

        # The recovery re-run is the (only) test-leg command carrying the
        # serial marker.
        recovered = next(c for c in captured_cmds if 'no:xdist' in c)
        assert '-n 16' not in recovered, (
            f'forced-serial recovery re-run must not carry -n 16 (xdist '
            f'disabled → unregistered option); got {recovered!r}'
        )
