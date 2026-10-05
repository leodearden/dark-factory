"""pytest-timeout's ``timeout_method`` under pytest-xdist: the measurement, and the policy.

This file is the ONE home for the signal-vs-thread trade. Every pyproject.toml
that sets ``timeout_method`` points here rather than restating it.

WHAT EACH METHOD DOES (pytest-timeout's own mechanism, ``pytest_timeout.py``):

  * ``signal`` (``timeout_sigalrm``) arms SIGALRM per test. On expiry the
    handler raises ``pytest.fail`` INSIDE the test: that one test fails, and
    the process (an xdist worker included) lives on to run the next test.
  * ``thread`` (``timeout_timer``) starts a timer thread per test. On expiry it
    dumps every thread's stack and calls ``os._exit(1)``: the whole process
    dies. Under xdist that is a dead worker, and with ``--max-worker-restart=0``
    (which every thread config must carry, see
    ``test_xdist_worker_restart_policy.py``) every test that worker had not
    yet run is lost, so the tally is partial.

WHY SIGNAL IS VALID UNDER XDIST. A Python signal handler only ever runs on the
MAIN thread. pytest-xdist 3.6.0 (changelog #1027) starts its workers with
execnet's ``main_thread_only`` execmodel
(``xdist/workermanage.py::NodeManager``), so tests run on each worker's main
thread and SIGALRM reaches them. The member floors do NOT guarantee that: they
read ``pytest-xdist>=3.5.0``, and orchestrator's ``>=3.0``. What guarantees it
is the single workspace ``uv.lock`` pin, at 3.8.0 when this was written. A
resolve below 3.6.0 would put tests back on a non-main thread, and
``test_runs_on_main_thread`` in the probe below is what turns that red.

THE MEASUREMENT (task 5470; pytest 9.0.3, pytest-xdist 3.8.0, execnet 2.1.2,
pytest-timeout 2.4.0). The same synthetic suite at ``-n 2 --dist loadgroup``
and ``timeout = 2``, with one test blocked in ``threading.Event().wait()`` on
the main thread:

  * signal: that test FAILED with "Timeout (>2.0s) from pytest-timeout". The
    worker SURVIVED, because the next test in the same ``xdist_group`` ran on it
    and passed, and the tally was complete: 1 failed, 8 passed.
  * thread, with ``--max-worker-restart=0``: "worker 'gwN' crashed while
    running ..." and "worker restarting disabled"; two tests never ran.

The two probes below re-run that measurement on every collection.

THE DECISION:

  * ``signal`` is this repo's default: every config outside
    ``THREAD_TIMEOUT_METHOD_CONFIGS`` runs it. dashboard and escalation keep it
    ON THIS MEASUREMENT. Their earlier reason, "runs no pytest-xdist", was
    false: verify injects ``-n`` into every gated pytest leg, and since task
    5408 their own addopts carry ``-n auto``.
  * orchestrator and fused-memory run ``thread``. The premise recorded for that
    choice was "signal only works in the main thread / is incompatible with
    pytest-xdist" (fused-memory, commit 0afc3098f4, 2026-04-24, when uv.lock
    already pinned xdist 3.8.0). The signal probe below falsifies it.
  * What thread still offers over signal is pytest-timeout's own account. Its
    README calls thread "the surest and most portable method" and warns that
    signal "may interfere with the code under test": a test that uses SIGALRM
    itself must take thread. Separately, CPython runs a signal handler only
    when the main thread returns to the interpreter, so a C call that blocks
    without returning is not interrupted by signal. Neither case is measured
    here.
  * The two thread settings are RETAINED, not re-justified. Evaluating a flip
    touches ``--max-worker-restart=0``, ``faulthandler_timeout`` and verify's
    worker-death discriminators, so it is a separate follow-up filed from task
    5470. ``THREAD_TIMEOUT_METHOD_CONFIGS`` pins the partition, so that a flip or
    a new thread config has to update this record and its rationale.

Citations are by ``path::symbol``, never file:line, which is the house style
``test_line_pin_policy.py`` enforces in this directory.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import pathlib
import re
import subprocess
import sys

import pytest

# The one config-discovery walk, with its anti-vacuity floors. Resolves because
# tests/scripts/conftest.py puts this directory on sys.path.
import test_pytest_per_test_timeout_policy as per_test_timeout_policy

SIGNAL_METHOD = 'signal'
THREAD_METHOD = 'thread'

# The configs that run timeout_method = "thread". This is the RECORDED
# DIVERGENCE from the signal default, and the module docstring states why it
# is retained.
THREAD_TIMEOUT_METHOD_CONFIGS = frozenset({'orchestrator', 'fused-memory'})

PROBE_WORKERS = 2
PROBE_TIMEOUT_SECS = 2

# The pytest11 entry-point names of the only plugins the probe child may load.
# pytest's `plugins:` header shows the same names for the two distributions.
PROBE_PLUGINS = ('xdist', 'timeout')

# Every probe test except the blocking one overrides the ini cap with this, so
# that only the blocking test races PROBE_TIMEOUT_SECS. On a loaded host a
# trivial test can take longer than 2s, and the tally must not depend on that.
NON_BLOCKING_TIMEOUT_SECS = 60

# Below each probe's @pytest.mark.timeout, so a wedged child surfaces as a
# TimeoutExpired carrying its captured output rather than as pytest's axe.
PROBE_SUBPROCESS_TIMEOUT_SECS = 100

# Keeps the two probes on one worker wherever --dist loadgroup is in force, so
# two nested pytest sessions never run at once on an already-loaded host.
TIMEOUT_METHOD_PROBE_GROUP = 'timeout_method_probe'

PROBE_MODULE_NAME = 'test_timeout_probe.py'
BLOCKING_TEST = 'test_a_blocks_main_thread'
SAME_WORKER_SUCCESSOR = 'test_b_runs_after_on_same_worker'
MAIN_THREAD_TEST = 'test_runs_on_main_thread'

_PROBE_MODULE = f'''\
import threading

import pytest


@pytest.mark.xdist_group('same_worker')
def {BLOCKING_TEST}():
    threading.Event().wait()


@pytest.mark.xdist_group('same_worker')
@pytest.mark.timeout({NON_BLOCKING_TIMEOUT_SECS})
def {SAME_WORKER_SUCCESSOR}():
    assert threading.current_thread() is threading.main_thread()


@pytest.mark.timeout({NON_BLOCKING_TIMEOUT_SECS})
def {MAIN_THREAD_TEST}():
    assert threading.current_thread() is threading.main_thread()


@pytest.mark.timeout({NON_BLOCKING_TIMEOUT_SECS})
@pytest.mark.parametrize('peer', range(6))
def test_peer(peer):
    pass
'''


@dataclasses.dataclass(frozen=True)
class _ProbeSuite:
    root: pathlib.Path
    ini: pathlib.Path


def _write_probe_suite(tmp_path: pathlib.Path, method: str) -> _ProbeSuite:
    """The synthetic suite under *tmp_path*, its cap enforced by *method*."""
    root = tmp_path / 'suite'
    root.mkdir()
    ini = root / 'pytest.ini'
    ini.write_text(
        f'[pytest]\ntimeout = {PROBE_TIMEOUT_SECS}\ntimeout_method = {method}\n',
        encoding='utf-8',
    )
    (root / PROBE_MODULE_NAME).write_text(_PROBE_MODULE, encoding='utf-8')
    return _ProbeSuite(root=root, ini=ini)


def _run_probe_suite(suite: _ProbeSuite, *extra: str) -> subprocess.CompletedProcess[str]:
    """Run *suite* at ``-n PROBE_WORKERS --dist loadgroup -rA`` plus *extra*.

    ``PYTEST_*`` is scrubbed from the child's environment, so an inherited
    ``PYTEST_ADDOPTS``, ``PYTEST_TIMEOUT`` or ``PYTEST_XDIST_WORKER`` cannot
    change what the child runs or suppress the literals asserted on. Only
    ``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` is then set, so the child loads exactly
    ``PROBE_PLUGINS`` and nothing else installed in the venv.
    """
    missing = [
        plugin for plugin in ('xdist', 'pytest_timeout')
        if importlib.util.find_spec(plugin) is None
    ]
    assert not missing, (
        f'{missing!r} not importable in this environment, so the probe cannot '
        'run; this is a missing plugin, not a measurement.'
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith('PYTEST_')}
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    result = subprocess.run(
        [
            sys.executable, '-m', 'pytest',
            '-c', str(suite.ini),
            '-p', 'no:cacheprovider',
            *(arg for name in PROBE_PLUGINS for arg in ('-p', name)),
            '-n', str(PROBE_WORKERS),
            '--dist', 'loadgroup',
            '-rA',
            *extra,
        ],
        cwd=str(suite.root),
        env=env,
        capture_output=True,
        text=True,
        timeout=PROBE_SUBPROCESS_TIMEOUT_SECS,
        check=False,
    )
    loaded = _header_plugins(result.stdout)
    assert loaded == set(PROBE_PLUGINS), (
        f'the probe child loaded the plugins {loaded!r}, not exactly xdist and '
        'pytest-timeout. The probe measures only those two, so any other plugin '
        f'can change its result.\n{_captured(result)}'
    )
    return result


def _header_plugins(output: str) -> set[str] | None:
    """The plugin names on pytest's ``plugins:`` header line, versions stripped."""
    header = re.search(r'^plugins: (.*)$', output, re.MULTILINE)
    if header is None:
        return None
    return {entry.rsplit('-', 1)[0] for entry in header.group(1).split(', ')}


def _captured(result: subprocess.CompletedProcess[str]) -> str:
    return f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'


@pytest.mark.xdist_group(TIMEOUT_METHOD_PROBE_GROUP)
@pytest.mark.timeout(120)
def test_signal_timeout_fails_the_test_and_the_xdist_worker_survives(
    tmp_path: pathlib.Path,
) -> None:
    """THE MEASUREMENT that keeps dashboard and escalation on signal."""
    result = _run_probe_suite(_write_probe_suite(tmp_path, SIGNAL_METHOD))
    output = result.stdout + result.stderr

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, (
        f'the signal-method probe exited {result.returncode}, not '
        f'{int(pytest.ExitCode.TESTS_FAILED)} (TESTS_FAILED).\n{_captured(result)}'
    )
    failed = re.findall(r'^FAILED .*$', output, re.MULTILINE)
    assert len(failed) == 1 and BLOCKING_TEST in failed[0], (
        f'expected exactly one FAILED line, naming {BLOCKING_TEST}; got '
        f'{failed!r}.\n{_captured(result)}'
    )
    for survivor in (SAME_WORKER_SUCCESSOR, MAIN_THREAD_TEST):
        assert re.search(rf'^PASSED \S*::{survivor}\b', output, re.MULTILINE), (
            f'{survivor} did not pass. If it is {SAME_WORKER_SUCCESSOR}, the '
            'worker did not survive the signal timeout; if either fails its '
            'main-thread assertion, xdist is no longer running tests on the '
            "worker's main thread (pytest-xdist < 3.6.0?), and signal is no "
            'longer valid under xdist. Re-read this file\'s docstring before '
            f'changing any timeout_method.\n{_captured(result)}'
        )
    assert 'crashed' not in output, (
        f'a worker crashed under the signal method.\n{_captured(result)}'
    )
    assert re.search(r'^INTERNALERROR>', output, re.MULTILINE) is None, (
        f'the signal-method probe raised an INTERNALERROR.\n{_captured(result)}'
    )
    assert 'Timeout' in output, (
        f'{BLOCKING_TEST} failed, but not on the pytest-timeout cap.\n'
        f'{_captured(result)}'
    )
    assert '1 failed, 8 passed' in output, (
        'the tally is not the complete "1 failed, 8 passed": a signal timeout '
        f'must cost exactly the one test it fires in.\n{_captured(result)}'
    )


@pytest.mark.xdist_group(TIMEOUT_METHOD_PROBE_GROUP)
@pytest.mark.timeout(120)
def test_thread_timeout_kills_the_xdist_worker(tmp_path: pathlib.Path) -> None:
    """THE PAIRED CONTROL: the same suite under thread, as orchestrator and fused-memory run.

    Run on the signal probe's OWN suite, this is that probe's non-vacuity check:
    on this input, a method that kills the worker does produce the crash
    signature the signal probe asserts absent.
    ``test_xdist_worker_restart_policy.py::test_restarts_disabled_attribute_the_worker_death``
    also shows a thread timeout killing a worker, but on a different suite (a
    fixture-level sleep, its crashes ordered by controller-side signals), so it
    cannot vouch for this one.

    It asserts only the crash signature, not which later tests were lost: that
    set depends on xdist's shutdown ordering.
    """
    result = _run_probe_suite(
        _write_probe_suite(tmp_path, THREAD_METHOD), '--max-worker-restart=0'
    )
    output = result.stdout + result.stderr

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, (
        f'the thread-method probe exited {result.returncode}, not '
        f'{int(pytest.ExitCode.TESTS_FAILED)} (TESTS_FAILED).\n{_captured(result)}'
    )
    # `--dist loadgroup` suffixes a grouped nodeid with `@<group>`.
    assert re.search(rf"crashed while running '[^']*::{BLOCKING_TEST}(@[^']*)?'", output), (
        f'the thread-method timeout in {BLOCKING_TEST} did not kill its xdist '
        'worker. Then the signal probe no longer measures a difference between '
        f'the two methods.\n{_captured(result)}'
    )
    assert 'worker restarting disabled' in output, (
        f'xdist did not report that restarting was disabled.\n{_captured(result)}'
    )


def test_a_timed_out_probe_child_reports_its_partial_output() -> None:
    partial = 'bringing up nodes...\n........'
    timed_out = subprocess.TimeoutExpired(
        ['python', '-m', 'pytest'], 240, output=partial.encode(), stderr=None
    )

    report = _captured(timed_out)

    assert partial in report, (
        "a TimeoutExpired carries the child's streams as bytes, even under "
        f'text=True; they must be decoded, not shown as a repr.\n{report}'
    )
    assert 'stdout:' in report and 'stderr:' in report, report
    assert "b'" not in report, report


def test_thread_timeout_method_is_confined_to_the_recorded_configs() -> None:
    thread_configs = {
        name
        for name, ini_options in per_test_timeout_policy.discovered_pytest_configs().items()
        if ini_options.get('timeout_method') == THREAD_METHOD
    }
    assert thread_configs == THREAD_TIMEOUT_METHOD_CONFIGS, (
        f'the configs running timeout_method = {THREAD_METHOD!r} are '
        f'{sorted(thread_configs)}, but the recorded set is '
        f'{sorted(THREAD_TIMEOUT_METHOD_CONFIGS)}. {SIGNAL_METHOD!r} is the '
        'default: under pytest-xdist a signal timeout fails only the test it '
        'fires in, while a thread timeout kills the worker and truncates the '
        'tally. Adding a thread config, or flipping one either way, must update '
        'THREAD_TIMEOUT_METHOD_CONFIGS and the rationale in '
        'tests/scripts/test_timeout_method_policy.py\'s module docstring in the '
        'same change.'
    )
