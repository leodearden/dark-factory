"""Tests for the delivered-check dep-gate (capability-delivered-checks PRD,
task delta — plans/capability-delivered-checks-prd.md).

Covers ``orchestrator.delivered_checks`` (the grep/script runner) and the
scheduler-side dep-gate that consumes it: the ``delivered_check_cache`` arm
on ``Scheduler._deps_satisfied`` / ``Scheduler._eligible_for_dispatch``, the
per-tick sweep (``Scheduler._compute_delivered_check_cache`` +
``Scheduler._phase_delivered_check_gate``), and the hold-visibility event
(``Scheduler._note_delivered_hold`` / ``EventType.delivered_check_gate_held``).

Structurally mirrors ``test_scheduler.py``'s ``TestDepsSatisfiedExternalGate``
/ ``TestExternalDepGateHeld_DepsLive`` / ``TestAcquireNextExternalDepGate`` —
the delivered-check gate is a deliberate structural clone of the
cross-project external-dep gate (task 1580); see the plan's design
decisions for the point-by-point mirror rationale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.delivered_checks import DeliveredCheckResult, run_delivered_check

# ---------------------------------------------------------------------------
# TestRunnerGrepKind (task 2580 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


class TestRunnerGrepKind:
    """``run_delivered_check``'s grep-kind branch: argv shape + rc mapping."""

    def _fake_runner(self, rc: int, out: str = '', err: str = ''):
        """Build an injected fake runner recording every argv it's called with."""
        calls: list[list[str]] = []

        async def _runner(argv, **kwargs):
            calls.append(argv)
            return (rc, out, err)

        return _runner, calls

    @pytest.mark.asyncio
    async def test_argv_shape_with_paths(self):
        runner, calls = self._fake_runner(rc=0)
        check = {
            'name': 'cap',
            'kind': 'grep',
            'pattern': 'FooBar',
            'expect': 'present',
            'paths': ['src/a.py', 'src/b.py'],
        }

        await run_delivered_check(check, project_root='/proj', ref='main', runner=runner)

        assert calls == [
            ['git', '-C', '/proj', 'grep', '-E', 'FooBar', 'main', '--', 'src/a.py', 'src/b.py']
        ]

    @pytest.mark.asyncio
    async def test_argv_omits_dashdash_when_paths_empty(self):
        runner, calls = self._fake_runner(rc=0)
        check = {'name': 'cap', 'kind': 'grep', 'pattern': 'FooBar', 'expect': 'present'}

        await run_delivered_check(check, project_root='/proj', ref='main', runner=runner)

        assert calls == [['git', '-C', '/proj', 'grep', '-E', 'FooBar', 'main']]

    @pytest.mark.asyncio
    async def test_default_ref_is_main(self):
        runner, calls = self._fake_runner(rc=0)
        check = {'name': 'cap', 'kind': 'grep', 'pattern': 'FooBar', 'expect': 'present'}

        # ref= omitted entirely — default must be 'main'.
        await run_delivered_check(check, project_root='/proj', runner=runner)

        assert calls == [['git', '-C', '/proj', 'grep', '-E', 'FooBar', 'main']]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('rc', 'expected'),
        [
            (0, DeliveredCheckResult.DELIVERED),
            (1, DeliveredCheckResult.FAILED),
            (2, DeliveredCheckResult.ERRORED),
            (128, DeliveredCheckResult.ERRORED),
        ],
    )
    async def test_expect_present_rc_mapping(self, rc, expected):
        """expect=present: DELIVERED on rc==0, FAILED on rc==1, ERRORED on rc>=2."""
        runner, _calls = self._fake_runner(rc=rc)
        check = {'name': 'cap', 'kind': 'grep', 'pattern': 'x', 'expect': 'present'}

        result = await run_delivered_check(check, project_root='/proj', runner=runner)

        assert result is expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('rc', 'expected'),
        [
            (0, DeliveredCheckResult.FAILED),
            (1, DeliveredCheckResult.DELIVERED),
            (2, DeliveredCheckResult.ERRORED),
            (128, DeliveredCheckResult.ERRORED),
        ],
    )
    async def test_expect_absent_rc_mapping(self, rc, expected):
        """expect=absent: DELIVERED on rc==1, FAILED on rc==0, ERRORED on rc>=2."""
        runner, _calls = self._fake_runner(rc=rc)
        check = {'name': 'cap', 'kind': 'grep', 'pattern': 'x', 'expect': 'absent'}

        result = await run_delivered_check(check, project_root='/proj', runner=runner)

        assert result is expected


# ---------------------------------------------------------------------------
# TestRunnerScriptKind (task 2580 — step-3 RED / step-4 GREEN)
# ---------------------------------------------------------------------------


class TestRunnerScriptKind:
    """``run_delivered_check``'s script-kind branch: argv/cwd shape + exit/timeout mapping."""

    def _fake_runner(self, rc: int = 0, out: str = '', err: str = ''):
        calls: list[tuple] = []

        async def _runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return (rc, out, err)

        return _runner, calls

    @pytest.mark.asyncio
    async def test_argv_and_cwd_shape(self):
        runner, calls = self._fake_runner(rc=0)
        check = {
            'name': 'cap',
            'kind': 'script',
            'script': 'scripts/check.sh',
            'args': ['--flag', 'value'],
            'timeout_secs': 30,
        }

        await run_delivered_check(check, project_root='/proj', runner=runner)

        assert len(calls) == 1
        argv, kwargs = calls[0]
        assert argv == ['/proj/scripts/check.sh', '--flag', 'value']
        assert kwargs.get('cwd') == Path('/proj')

    @pytest.mark.asyncio
    async def test_exit_zero_delivered(self):
        runner, _calls = self._fake_runner(rc=0)
        check = {'name': 'cap', 'kind': 'script', 'script': 'x.sh', 'timeout_secs': 5}

        result = await run_delivered_check(check, project_root='/proj', runner=runner)

        assert result is DeliveredCheckResult.DELIVERED

    @pytest.mark.asyncio
    async def test_nonzero_exit_failed(self):
        runner, _calls = self._fake_runner(rc=1)
        check = {'name': 'cap', 'kind': 'script', 'script': 'x.sh', 'timeout_secs': 5}

        result = await run_delivered_check(check, project_root='/proj', runner=runner)

        assert result is DeliveredCheckResult.FAILED

    @pytest.mark.asyncio
    async def test_timeout_errored(self):
        """A runner that raises TimeoutError (simulating asyncio.wait_for's
        own timeout firing, mirroring test_deterministic_runner.py's
        seam-injection precedent) must map to ERRORED — no real subprocess,
        no real wait, is needed to exercise this path."""
        called = False

        async def _runner(argv, **kwargs):
            nonlocal called
            called = True
            raise TimeoutError('simulated timeout')

        check = {'name': 'cap', 'kind': 'script', 'script': 'x.sh', 'timeout_secs': 5}

        result = await run_delivered_check(check, project_root='/proj', runner=_runner)

        assert called, 'the injected runner must actually be invoked'
        assert result is DeliveredCheckResult.ERRORED

    @pytest.mark.asyncio
    async def test_missing_executable_errored(self):
        """A runner that raises OSError (e.g. FileNotFoundError for a
        missing/non-executable script) must map to ERRORED."""
        called = False

        async def _runner(argv, **kwargs):
            nonlocal called
            called = True
            raise FileNotFoundError('missing executable')

        check = {'name': 'cap', 'kind': 'script', 'script': 'x.sh', 'timeout_secs': 5}

        result = await run_delivered_check(check, project_root='/proj', runner=_runner)

        assert called, 'the injected runner must actually be invoked'
        assert result is DeliveredCheckResult.ERRORED
