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
thread and SIGALRM reaches them. The member floors guarantee that: every
workspace member's pytest-xdist declaration reads ``>=3.6.0`` (esc-5470-3,
ruled 2026-10-07; task 6477), which
``test_every_pytest_xdist_floor_admits_only_main_thread_workers`` enforces, and
the single workspace ``uv.lock`` pin (3.8.0 when this was written) sits above
it. A resolve below 3.6.0 would put tests back on a non-main thread, and
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

THE REAL-CONFTEST MEASUREMENT (task 6354, 2026-10-31; the same versions on
CPython 3.13, load average 175 on 32 cores). A throwaway probe ran inside the
REAL ``orchestrator/tests`` and ``fused-memory/tests`` conftests at
``-o timeout_method=signal -n 2 --max-worker-restart=0``: five tests in one
``xdist_group``, each capped at 3s, then successor tests. Signal interrupted all
five main-thread shapes:

  * ``asyncio.sleep(3600)``, raised from the selector's wait;
  * an async test holding a pending task and a coroutine it never started;
  * ``threading.Event().wait()``;
  * ``subprocess.run`` of a ``sleep 3600`` child, which ``subprocess.run`` then
    killed;
  * ``asyncio.to_thread`` of ``Event().wait``.

In both suites no worker crashed and no INTERNALERROR was raised; the successor
sync and async tests ran on the same worker and passed, and the tally was
complete. Two collateral costs appeared, and NEITHER COST THE TALLY:

  * In orchestrator only, the never-started coroutine raised "coroutine ... was
    never awaited". orchestrator's ``filterwarnings`` promotes that to an error,
    and it was blamed on the NEXT test on that worker, the one whose
    ``gc.collect()`` found it: one extra red. Any test that fails partway can
    leave the same collateral, so it is not specific to signal.
  * A thread parked in asyncio's default executor made the timed-out test's
    loop teardown wait up to 300s in ``shutdown_default_executor``.

That probe is not committed, because it would import the heaviest conftest in
the repo into a nested session on every scripts run. The executable probes
below carry the two shapes the flip depends on, a main thread parked in the
asyncio selector and one parked in a subprocess wait, beside the original
``Event().wait()``, and re-run on every collection. ``to_thread`` is left out
for its 300s teardown, and the unstarted coroutine because it measures
orchestrator's ``filterwarnings``, not signal.

THE DECISION:

  * ``signal`` is every config's method. ``THREAD_TIMEOUT_METHOD_CONFIGS`` is
    empty, and the empty set is the recorded state: a future thread config must
    be added to it, with its reason, in this docstring.
  * orchestrator and fused-memory ran ``thread`` until task 6354. The premise
    recorded for that choice, "signal only works in the main thread / is
    incompatible with pytest-xdist" (fused-memory, commit 0afc3098f4,
    2026-04-24, when uv.lock already pinned xdist 3.8.0), is falsified by the
    signal probe below. Under thread with ``--max-worker-restart=0`` one breach
    truncated the whole session, and verify refuses to confirm a truncated run
    as a flake
    (``orchestrator/src/orchestrator/verify.py::_unmeasured_session_reason``).
    Under signal the starved test fails as itself, with its own traceback, and
    an isolated re-run can confirm it as a flake.
  * dashboard and escalation run signal ON TASK 5470'S MEASUREMENT. Their
    earlier reason, "runs no pytest-xdist", was false: verify injects ``-n``
    into every gated pytest leg, and since task 5408 their own addopts carry
    ``-n auto``.
  * What thread still offers over signal. Each case is UNMEASURED here:

      - CPython runs a signal handler only when the main thread returns to the
        interpreter, so a main thread blocked in a C call that never returns is
        never interrupted. The worker hangs until verify's outer kill, and
        orchestrator's ``faulthandler_timeout`` dump is the only witness.
      - A C call that does return (complexipy's Rust analysis, say) only
        defers the failure until it returns.
      - Code under test that catches ``BaseException`` in the frame where the
        signal lands can swallow pytest-timeout's single alarm.

    pytest-timeout's README also calls thread "the surest and most portable
    method" and warns that signal "may interfere with the code under test": a
    test that uses SIGALRM itself must take thread.
  * ``--max-worker-restart=0`` stays in orchestrator and fused-memory. The
    method and the restart cap are separate axes: under signal the cap still
    governs the worker deaths that remain (segfault, OOM, ``os._exit`` in code
    under test). ``test_xdist_worker_restart_policy.py`` records why, and pins
    it.
  * Every pytest-xdist floor is ``>=3.6.0``, by esc-5470-3's ruling (task
    6477). Three things guard the premise above: the floor test
    (``tests/scripts/test_timeout_method_policy.py::test_every_pytest_xdist_floor_admits_only_main_thread_workers``),
    the ``uv.lock`` pin, and the probe's main-thread assertions.

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
import tomllib
import types

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
from process_group_run import ProcessGroupOverrun, run_process_group
from pytest_config_discovery import (
    REPO_ROOT,
    ROOT_CONFIG_NAME,
    discovered_pytest_configs,
    workspace_members,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from df_pytest_isolation import wait_pid_gone

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SIGNAL_METHOD = 'signal'
THREAD_METHOD = 'thread'

# The configs that run timeout_method = "thread": none since task 6354. Any
# future member is a recorded divergence from signal, with its reason in the
# module docstring.
THREAD_TIMEOUT_METHOD_CONFIGS: frozenset[str] = frozenset()

# The first pytest-xdist release whose workers run under execnet's
# main_thread_only execmodel (changelog #1027), which the docstring's signal
# premise depends on.
MAIN_THREAD_WORKERS_XDIST = Version('3.6.0')
XDIST_DISTRIBUTION = 'pytest-xdist'

# A FLOOR, not an equality: 8 at authorship, every workspace member. Same idiom
# as pytest_config_discovery.MIN_EXPECTED_PYTEST_CONFIGS.
MIN_EXPECTED_XDIST_DECLARERS = 8

PROBE_WORKERS = 2
PROBE_TIMEOUT_SECS = 2

# What pytest-timeout's failure reads when the probe's ini cap fires.
TIMEOUT_FAILURE = f'Timeout (>{float(PROBE_TIMEOUT_SECS)}s) from pytest-timeout'

# The pytest11 entry-point names of the only plugins the probe child may load.
# pytest's `plugins:` header shows the same names for the two distributions.
PROBE_PLUGINS = ('xdist', 'timeout')

# Every probe test except the blocking ones overrides the ini cap with this, so
# that only BLOCKING_TESTS race PROBE_TIMEOUT_SECS. On a loaded host a
# trivial test can take longer than 2s, and the tally must not depend on that.
NON_BLOCKING_TIMEOUT_SECS = 60

# Strictly below the per-test axe the probes run under (300 under verify), as
# test_the_probes_take_this_runs_per_test_timeout_and_the_child_budget_sits_below_it enforces.
PROBE_SUBPROCESS_TIMEOUT_SECS = 240

# Keeps the two probes on one worker wherever --dist loadgroup is in force, so
# two nested pytest sessions never run at once on an already-loaded host.
TIMEOUT_METHOD_PROBE_GROUP = 'timeout_method_probe'

PROBE_MODULE_NAME = 'test_timeout_probe.py'
BLOCKING_TEST = 'test_a_blocks_main_thread'
ASYNCIO_BLOCKING_TEST = 'test_a_blocks_in_asyncio_selector'
SUBPROCESS_BLOCKING_TEST = 'test_a_blocks_waiting_on_a_subprocess'
BLOCKING_TESTS = (BLOCKING_TEST, ASYNCIO_BLOCKING_TEST, SUBPROCESS_BLOCKING_TEST)
SAME_WORKER_SUCCESSOR = 'test_b_runs_after_on_same_worker'
MAIN_THREAD_TEST = 'test_runs_on_main_thread'

_PROBE_MODULE = f'''\
import asyncio
import subprocess
import sys
import threading

import pytest


@pytest.mark.xdist_group('same_worker')
def {BLOCKING_TEST}():
    threading.Event().wait()


@pytest.mark.xdist_group('same_worker')
def {ASYNCIO_BLOCKING_TEST}():
    asyncio.run(asyncio.sleep(3600))


@pytest.mark.xdist_group('same_worker')
def {SUBPROCESS_BLOCKING_TEST}():
    subprocess.run([sys.executable, '-c', 'import time; time.sleep(3600)'], check=False)


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


def _run_probe_suite(
    suite: _ProbeSuite,
    *extra: str,
    budget_secs: float = PROBE_SUBPROCESS_TIMEOUT_SECS,
) -> subprocess.CompletedProcess[str]:
    """Run *suite* at ``-n PROBE_WORKERS --dist loadgroup -rA`` plus *extra*.

    A child still running after *budget_secs* fails the calling test with
    whatever it had printed and the live members of its process group. The
    whole group is killed when the run ends, so no descendant can hold the run
    open or outlive it
    (``tests/scripts/process_group_run.py::run_process_group``).

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
    try:
        result = run_process_group(
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
            cwd=suite.root,
            env=env,
            budget_secs=budget_secs,
        )
    except ProcessGroupOverrun as overrun:
        pytest.fail(
            f'the probe child did not finish within its budget of {budget_secs}s. '
            'Its output goes to files, so no descendant can hold it open: the '
            'controller itself was still running. A stdout ending in the '
            'terminal summary means the controller hung in its own exit after '
            'the session finished, the header alone means it was stuck in '
            'startup, and progress characters without an F mean no test in '
            f'{BLOCKING_TESTS!r} ever timed out.\n'
            f'{overrun}',
            pytrace=False,
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


def _failed_test_names(output: str) -> list[str]:
    """The test names on the short summary's FAILED lines, ``@<group>`` suffix stripped."""
    return re.findall(r'^FAILED \S*::([^\s@]+)', output, re.MULTILINE)


def _failure_sections(output: str) -> dict[str, str]:
    """Each test's section of pytest's FAILURES block, keyed by its head line.

    The short summary truncates a FAILED line's message to the terminal width,
    so why a test failed is read here instead.
    """
    block = re.search(r'^=+ FAILURES =+$(.*?)(?=^=+ .* =+$|\Z)', output, re.MULTILINE | re.DOTALL)
    if block is None:
        return {}
    parts = re.split(r'^_{3,} (\S+) _{3,}$', block.group(1), flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _captured(result: subprocess.CompletedProcess[str]) -> str:
    return f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'


@pytest.mark.xdist_group(TIMEOUT_METHOD_PROBE_GROUP)
def test_signal_timeout_fails_the_test_and_the_xdist_worker_survives(
    tmp_path: pathlib.Path,
) -> None:
    """THE MEASUREMENT that keeps every config on signal."""
    result = _run_probe_suite(_write_probe_suite(tmp_path, SIGNAL_METHOD))
    output = result.stdout + result.stderr

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, (
        f'the signal-method probe exited {result.returncode}, not '
        f'{int(pytest.ExitCode.TESTS_FAILED)} (TESTS_FAILED).\n{_captured(result)}'
    )
    failed = _failed_test_names(output)
    assert set(failed) == set(BLOCKING_TESTS), (
        f'expected the FAILED lines to name exactly {sorted(BLOCKING_TESTS)}; '
        f'got {sorted(failed)}.\n{_captured(result)}'
    )
    sections = _failure_sections(output)
    not_on_the_cap = [
        name for name in BLOCKING_TESTS if TIMEOUT_FAILURE not in sections.get(name, '')
    ]
    assert not not_on_the_cap, (
        f'{not_on_the_cap!r} failed, but not on the pytest-timeout cap: signal '
        'did not interrupt the main thread where that test blocks.\n'
        f'{_captured(result)}'
    )
    for survivor in (SAME_WORKER_SUCCESSOR, MAIN_THREAD_TEST):
        assert re.search(rf'^PASSED \S*::{survivor}\b', output, re.MULTILINE), (
            f'{survivor} did not pass. If it is {SAME_WORKER_SUCCESSOR}, the '
            f'worker did not survive {len(BLOCKING_TESTS)} successive signal '
            'timeouts; if either fails its '
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
    assert '3 failed, 8 passed' in output, (
        'the tally is not the complete "3 failed, 8 passed": a signal timeout '
        f'must cost exactly the one test it fires in.\n{_captured(result)}'
    )


@pytest.mark.xdist_group(TIMEOUT_METHOD_PROBE_GROUP)
def test_thread_timeout_kills_the_xdist_worker(tmp_path: pathlib.Path) -> None:
    """THE PAIRED CONTROL: the same suite under thread, as orchestrator and fused-memory ran until task 6354.

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


def test_a_probe_child_that_outlives_its_budget_fails_the_calling_test(
    tmp_path: pathlib.Path,
) -> None:
    budget_secs = 0.5
    # --timeout=0 lifts the cap on BLOCKING_TEST, so the child can never finish;
    # -n 0 keeps the child to one process.
    never_finishes = ('--timeout=0', '-n', '0')

    with pytest.raises(pytest.fail.Exception) as failure:
        _run_probe_suite(
            _write_probe_suite(tmp_path, SIGNAL_METHOD),
            *never_finishes,
            budget_secs=budget_secs,
        )

    report = str(failure.value)
    assert f'within its budget of {budget_secs}s' in report, report
    assert 'stdout:' in report and 'stderr:' in report, report


def test_a_descendant_still_holding_the_probe_childs_output_does_not_hold_up_the_probe(
    tmp_path: pathlib.Path,
) -> None:
    suite = _write_probe_suite(tmp_path, SIGNAL_METHOD)
    pidfile = tmp_path / 'lingerer.pid'
    # Spawned after the session summary, like a descendant that outlives the
    # controller; it inherits the child's stdout and stderr, and sleeps past
    # the budget so that a run waiting for EOF overruns while a leak still
    # ends by itself.
    (suite.root / 'conftest.py').write_text(
        'import pathlib\n'
        'import subprocess\n'
        'import sys\n'
        '\n'
        '\n'
        'def pytest_unconfigure(config):\n'
        '    lingerer = subprocess.Popen(\n'
        f"        [sys.executable, '-c', 'import time; time.sleep({PROBE_SUBPROCESS_TIMEOUT_SECS + 60})']\n"
        '    )\n'
        f'    pathlib.Path({str(pidfile)!r}).write_text(str(lingerer.pid))\n',
        encoding='utf-8',
    )

    result = _run_probe_suite(suite, '-n', '0', '-k', MAIN_THREAD_TEST)

    assert result.returncode == pytest.ExitCode.OK, (
        f'the probe child exited {result.returncode}, not 0 (OK).\n{_captured(result)}'
    )
    lingerer = int(pidfile.read_text(encoding='utf-8'))
    assert wait_pid_gone(lingerer, timeout=30), (
        f'the lingering descendant {lingerer} is still running: the probe\'s '
        'process group must not outlive the run.'
    )


@pytest.mark.parametrize(
    ('output', 'plugins'),
    [
        (
            'rootdir: /suite\nplugins: xdist-3.8.0, timeout-2.4.0\ncollected 9 items\n',
            {'xdist', 'timeout'},
        ),
        ('plugins: rerun-failures-16.1\n', {'rerun-failures'}),
        ('rootdir: /suite\ncollected 9 items\n', None),
    ],
    ids=['measured-pair', 'hyphenated-name', 'no-header'],
)
def test_header_plugins_names_each_plugin_without_its_version(
    output: str, plugins: set[str] | None,
) -> None:
    assert _header_plugins(output) == plugins


def test_each_failure_is_read_from_its_own_section() -> None:
    output = (
        '==================== FAILURES ====================\n'
        '_______________ test_one _______________\n'
        f'E   Failed: {TIMEOUT_FAILURE}.\n'
        '_______________ test_two _______________\n'
        'E   AssertionError\n'
        '============= short test summary info ============\n'
        'FAILED suite.py::test_one@same_worker - Failed:...\n'
        'FAILED suite.py::test_two\n'
    )

    sections = _failure_sections(output)

    assert _failed_test_names(output) == ['test_one', 'test_two']
    assert sorted(sections) == ['test_one', 'test_two']
    assert TIMEOUT_FAILURE in sections['test_one']
    assert TIMEOUT_FAILURE not in sections['test_two']


def _timeout_marks(module: types.ModuleType) -> list[tuple[str, float]]:
    """``(marked name, seconds)`` for every timeout mark in *module* that sets a timeout.

    Read off the module's own objects rather than its collected items, so a test
    that -k deselected still counts and nothing is collected a second time.
    """
    carriers = [(module.__name__, module)] + [
        (name, obj) for name, obj in vars(module).items() if name.startswith(('test_', 'Test'))
    ]
    found: list[tuple[str, float]] = []
    for name, carrier in carriers:
        marks = getattr(carrier, 'pytestmark', [])
        for mark in marks if isinstance(marks, list) else [marks]:
            if mark.name != 'timeout':
                continue
            secs = mark.args[0] if mark.args else mark.kwargs.get('timeout')
            if secs is not None:
                found.append((name, secs))
    return found


def test_timeout_marks_finds_every_mark_that_sets_a_timeout() -> None:
    @pytest.mark.timeout(120)
    def test_positional() -> None: ...

    @pytest.mark.timeout(timeout=90)
    def test_keyword() -> None: ...

    @pytest.mark.timeout(method=THREAD_METHOD)
    def test_method_only() -> None: ...

    @pytest.mark.timeout(5)
    def helper() -> None: ...

    module = types.ModuleType('marked')
    vars(module).update(
        {fn.__name__: fn for fn in (test_positional, test_keyword, test_method_only, helper)},
        pytestmark=pytest.mark.timeout(30),
    )

    assert sorted(_timeout_marks(module)) == [
        ('marked', 30), ('test_keyword', 90), ('test_positional', 120),
    ]


def test_the_probes_take_this_runs_per_test_timeout_and_the_child_budget_sits_below_it(
    request: pytest.FixtureRequest,
) -> None:
    pytest_timeout = pytest.importorskip('pytest_timeout')
    axe = pytest_timeout.get_env_settings(request.config).timeout
    if not axe:
        pytest.skip('no per-test timeout configured, nothing can truncate a probe')

    tightening = [(name, secs) for name, secs in _timeout_marks(request.module) if secs < axe]
    assert not tightening, (
        f'{tightening!r} carry a timeout mark below this run\'s per-test timeout '
        f'of {axe}s. A pytest-timeout mark REPLACES the run\'s budget in both '
        'directions (precedence recorded at '
        'orchestrator/tests/_orch_helpers.py::VERIFY_CLI_PER_TEST_TIMEOUT), so '
        'such a mark tightens the budget verify grants this leg and takes a slow '
        "probe child's time away before it can report."
    )
    assert axe > PROBE_SUBPROCESS_TIMEOUT_SECS, (
        f'PROBE_SUBPROCESS_TIMEOUT_SECS={PROBE_SUBPROCESS_TIMEOUT_SECS} is not '
        f'below this run\'s per-test timeout of {axe}s, so a wedged probe child '
        'would be cut off by the per-test axe instead of reporting its partial '
        'output.'
    )


def test_thread_timeout_method_is_confined_to_the_recorded_configs() -> None:
    thread_configs = {
        name
        for name, ini_options in discovered_pytest_configs().items()
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


def _xdist_requirements(config_name: str) -> list[Requirement]:
    """Every pytest-xdist entry *config_name*'s pyproject.toml declares.

    Reads all three tables that can declare one: ``[project].dependencies``,
    every list in ``[project.optional-dependencies]``, and every group in
    ``[dependency-groups]``. Dict entries such as ``{include-group = ...}`` name
    another group rather than a distribution, so they are skipped; the group
    they include is walked in its own right.
    """
    pyproject = REPO_ROOT / config_name / 'pyproject.toml'
    data = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    project = data.get('project', {})
    declaration_lists = [
        project.get('dependencies', []),
        *project.get('optional-dependencies', {}).values(),
        *data.get('dependency-groups', {}).values(),
    ]
    requirements = [
        Requirement(entry)
        for entries in declaration_lists
        for entry in entries
        if isinstance(entry, str)
    ]
    return [req for req in requirements if canonicalize_name(req.name) == XDIST_DISTRIBUTION]


def _floor_violation(config_name: str, requirement: Requirement) -> str | None:
    """Why *requirement* admits an xdist below ``MAIN_THREAD_WORKERS_XDIST``, or None."""
    floors = [
        Version(clause.version)
        for clause in requirement.specifier
        if clause.operator == '>='
    ]
    if not floors:
        return (
            f'{config_name}/pyproject.toml declares {str(requirement)!r}, which has no '
            f'`>=` floor; spell it {XDIST_DISTRIBUTION}>={MAIN_THREAD_WORKERS_XDIST}'
        )
    if max(floors) < MAIN_THREAD_WORKERS_XDIST:
        return (
            f'{config_name}/pyproject.toml declares {str(requirement)!r}, below the '
            f'required floor {XDIST_DISTRIBUTION}>={MAIN_THREAD_WORKERS_XDIST}'
        )
    return None


def test_every_pytest_xdist_floor_admits_only_main_thread_workers() -> None:
    """The declaration-level guard on the signal premise.

    The probes above check the resolved xdist at run time and ``uv.lock`` pins
    it; this checks what the root pyproject.toml and every workspace member
    admit, in any dependency table, so a lowered floor or a new declaration at
    an old one fails here, by name, before any relock can resolve below it.
    """
    declarers: list[str] = []
    violations: list[str] = []
    config_names = [
        ROOT_CONFIG_NAME,
        *workspace_members(),
    ]
    for config_name in config_names:
        requirements = _xdist_requirements(config_name)
        if requirements:
            declarers.append(config_name)
        violations.extend(
            violation
            for violation in (_floor_violation(config_name, req) for req in requirements)
            if violation is not None
        )

    assert len(declarers) >= MIN_EXPECTED_XDIST_DECLARERS, (
        f'only {len(declarers)} pyproject.toml files declare {XDIST_DISTRIBUTION} '
        f'({sorted(declarers)}), expected at least {MIN_EXPECTED_XDIST_DECLARERS}. '
        'Discovery reads the root pyproject.toml plus one per '
        '[tool.uv.workspace].members entry, and in each the [project] '
        'dependencies, [project.optional-dependencies] and [dependency-groups]; '
        'a declaration it stopped finding is one this guard no longer checks.'
    )
    assert not violations, (
        f'these {XDIST_DISTRIBUTION} declarations admit a release below '
        f'{MAIN_THREAD_WORKERS_XDIST}:\n'
        + '\n'.join(f'  {violation}' for violation in violations)
        + f'\n\nBelow {MAIN_THREAD_WORKERS_XDIST}, xdist does not run tests on the '
        "worker's main thread, so SIGALRM never reaches them and this file's "
        'premise that timeout_method = "signal" is valid under xdist fails. '
        'See tests/scripts/test_timeout_method_policy.py\'s module docstring and '
        'esc-5470-3.'
    )
