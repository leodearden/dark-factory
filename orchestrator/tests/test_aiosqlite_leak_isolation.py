"""Tests for reap_leaked_aiosqlite_connections() and its autouse fixture wiring.

Task 2413 — fix for the scheduler-test xdist flake where a leaked aiosqlite
connection's background worker thread outlives the test that created it,
then later raises ``RuntimeError: Event loop is closed`` and is blamed on
whatever unrelated test happens to be running when the thread fires.

Design decisions:
- Tests are self-contained unit tests of the helper itself (not a
  cross-test polluter/victim pair), so they are deterministic under
  -n auto --dist loadgroup without requiring an xdist_group co-location tag —
  mirrors test_async_mock_coroutine_isolation.py.
- Each test reaps any residual leaked connection from a *prior* test before
  measuring, matching test_async_mock_coroutine_isolation.py's own
  flush-before-measure pattern for drain_async_mock_coroutines().

Task 4075 adds the "PytestUnhandledThreadExceptionWarning promotion" section
at the bottom of this module. The reaper above is only the FIRST line of
defence; the promotion is the backstop that makes a *degraded* reaper loud
instead of silent (see reap_leaked_aiosqlite_connections()'s MAINTENANCE
NOTE). The backstop contract lives here, next to the reaper it backs, so a
future reader auditing the leak defence sees both halves at once.
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import aiosqlite
import pytest
from _orch_helpers import (
    ORCH_PYPROJECT,
    reap_leaked_aiosqlite_connections,
    require_orchestrator_inifile,
    sanitized_probe_env,
)

try:  # pragma: no cover - the ImportError branch is a pytest-rename tripwire
    from _pytest.config import apply_warning_filters
except ImportError:  # pragma: no cover
    apply_warning_filters = None


@pytest.mark.asyncio
async def test_reap_closes_leaked_open_connection():
    """reap_leaked_aiosqlite_connections() closes a connection nothing else closed.

    Open an aiosqlite connection and execute a statement so its worker thread
    is proven live, then leak it (never call ``.close()``). After reap, the
    thread must be dead — checked directly via the connection's own
    ``_running`` / ``_thread.is_alive()`` state, an observable independent of
    the helper's internal selection predicate.
    """
    # Flush any leaked connection a prior test left behind before measuring.
    await reap_leaked_aiosqlite_connections()

    conn = await aiosqlite.connect(':memory:')
    await conn.execute('CREATE TABLE t(x)')
    assert conn._thread.is_alive(), 'expected the worker thread to be alive after connect'

    closed = await reap_leaked_aiosqlite_connections()
    assert closed >= 1, 'reap_leaked_aiosqlite_connections() should have closed >= 1 connection'

    # Independent observable: verify the specific connection's own state, not
    # just the aggregate count returned above.
    assert conn._running is False, 'reap must stop() the leaked connection'
    assert not conn._thread.is_alive(), 'reap must join the worker thread before returning'


@pytest.mark.asyncio
async def test_reap_is_safe_when_no_leaked_connections():
    """reap_leaked_aiosqlite_connections() returns 0 when nothing is leaked."""
    # First reap to flush any residual leak from a prior test.
    await reap_leaked_aiosqlite_connections()

    count = await reap_leaked_aiosqlite_connections()
    assert count == 0, (
        f'Expected 0 leaked aiosqlite connections but got {count}. '
        'A prior test may have leaked a connection.'
    )


@pytest.mark.asyncio
async def test_reap_ignores_already_closed_connection():
    """A connection that was properly closed is not counted or touched by reap.

    Safety-net assertion: reap must not double-close (and must not raise on)
    a connection a test already cleaned up correctly.
    """
    await reap_leaked_aiosqlite_connections()

    conn = await aiosqlite.connect(':memory:')
    await conn.close()
    # aiosqlite's worker thread resolves the close() future via
    # call_soon_threadsafe() *before* its own Python frame returns and the OS
    # thread actually terminates — an inherent TOCTOU race. A bounded join
    # avoids asserting is_alive() before the thread has had a chance to exit
    # (mirrors dashboard/tests/test_db.py's join(timeout=...) convention for
    # the same aiosqlite worker-exit race).
    conn._thread.join(timeout=2.0)
    assert not conn._thread.is_alive()

    count = await reap_leaked_aiosqlite_connections()
    assert count == 0, 'An already-closed connection must not be counted as reaped'


def test_autouse_reap_fixture_is_active(request):
    """_reap_leaked_aiosqlite_connections is wired as an autouse teardown fixture.

    This assertion is deterministic and worker-placement-agnostic: existing
    autouse fixtures (_drain_async_mock_coroutines, _reap_leaked_merge_workers)
    appear in request.fixturenames; absent names do not. This is a behavioral
    assertion that the reaper is applied to arbitrary tests — not a docstring
    lint.
    """
    assert '_reap_leaked_aiosqlite_connections' in request.fixturenames, (
        '_reap_leaked_aiosqlite_connections must be registered as an autouse '
        'fixture in conftest.py so it runs after every test and closes any '
        'leaked aiosqlite connection before the per-test event loop closes.'
    )


# ---------------------------------------------------------------------------
# PytestUnhandledThreadExceptionWarning promotion (task 4075)
#
# The reaper above prevents the leak; this section pins the backstop for when
# the reaper itself stops working. reap_leaked_aiosqlite_connections()'s
# MAINTENANCE NOTE documents that the helper degrades to a silent no-op
# (returns 0) if aiosqlite renames ``_running`` / ``_thread`` /
# ``core._connection_worker_thread``. In that degraded state the leak still
# happens: a leaked worker thread raises ``RuntimeError: Event loop is
# closed`` and pytest's threadexception plugin attributes it to whatever
# innocent test is running at the time. Unless that warning is an ERROR, the
# suite stays green and nobody finds out. Both
# ``_orch_helpers.reap_leaked_aiosqlite_connections`` and
# ``conftest._reap_leaked_aiosqlite_connections`` promise in prose that
# orchestrator/pyproject.toml makes it an error — these three guards are what
# keep that promise checkable instead of aspirational.
# ---------------------------------------------------------------------------

#: ``ORCH_PYPROJECT`` (the inifile these guards pin), ``sanitized_probe_env``
#: (the env the subprocess arms below run under — it strips ``PYTHONWARNINGS``,
#: which would otherwise contaminate BOTH arms of the end-to-end pin) and
#: ``require_orchestrator_inifile`` all come from ``_orch_helpers``. They live
#: there rather than here because each had already been re-derived in three to
#: five test modules apiece; see that module's "Pytest-invocation helpers"
#: section for the full rationale.

#: The exact ``filterwarnings`` entry these guards pin. Anchored on the
#: CATEGORY alone (empty message field) because the thread-exception message
#: is a formatted traceback with no stable substring to anchor on — and
#: because this is already the spelling shipped at
#: shared/tests/test_async_sqlite_base.py:596.
PROMOTION_ENTRY = 'error::pytest.PytestUnhandledThreadExceptionWarning'

#: Named in every skip reason emitted by ``require_orchestrator_inifile`` here,
#: so a skipped run says which contract went unchecked.
_PIN_SUBJECT = f'the {PROMOTION_ENTRY!r} promotion'

#: A test whose only sin is letting a thread die with an exception — exactly
#: the shape a degraded reaper leaves behind, minus the aiosqlite machinery.
_PROBE_TEST_SOURCE = '''\
import threading


def _boom():
    raise RuntimeError('df-4075 probe')


def test_thread_raises():
    t = threading.Thread(target=_boom, name='df-4075-probe-thread')
    t.start()
    t.join()
'''

#: The CONTROL arm's inifile: structurally identical, deliberately declaring
#: no ``filterwarnings``. It is what attributes the treatment arm's failure to
#: the promotion entry rather than to anything ambient in the environment.
_CONTROL_INIFILE_SOURCE = '''\
[tool.pytest.ini_options]
addopts = ""
'''


def _require_apply_warning_filters():
    """Return ``_pytest.config.apply_warning_filters``, or fail LOUDLY.

    Deliberately NOT ``pytest.skip``. This task exists because a documented
    safety net was absent and nothing said so; a guard that skips itself away
    on a private-API rename would reproduce that failure mode one level up —
    the pin would go quietly green-by-absence exactly when pytest internals
    moved, which is precisely when the promotion most needs re-verification.
    Same loud-on-rename idiom as
    shared/tests/test_async_sqlite_base.py::_ensure_real_conn_closed_at_exit.
    """
    if apply_warning_filters is None:
        raise AssertionError(
            '_pytest.config.apply_warning_filters no longer imports — pytest '
            'has renamed or moved its warning-filter parser. This is NOT a '
            'reason to skip: the promotion pinned by this module can only be '
            "checked through pytest's own parser, and a pytest internals "
            'change is exactly when it most needs re-checking. Update the '
            f'guarded import at the top of {Path(__file__).name} to the new '
            'API (task 4075).'
        )
    return apply_warning_filters


def test_pyprojects_filterwarnings_promotes_unhandled_thread_exceptions(pytestconfig):
    """orchestrator/pyproject.toml's ``filterwarnings`` promotes the warning to an error.

    The SOURCE pin: it reads the EFFECTIVE ``getini('filterwarnings')`` list
    and runs it through pytest's own parser, rather than doing a ``tomllib``
    string match on the file. This repo has already ruled twice that
    "configured" means the effective ``getini(...)`` value and not file text —
    test_marker_registration_drift.py::test_conftest_registered_markers_count_as_registered
    and test_lane_lock_leak_guard.py::_effective_per_test_timeout (task 3836
    review amendment) — and using ``apply_warning_filters`` also avoids
    re-implementing the ``action:message:category:module:lineno`` grammar,
    whose ``:`` field separator is already a documented footgun in
    pyproject.toml's own comment block.
    """
    require_orchestrator_inifile(pytestconfig, subject=_PIN_SUBJECT)
    apply = _require_apply_warning_filters()

    ini_filters = list(pytestconfig.getini('filterwarnings'))

    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter('always')
        # Passing [] for cmdline_filters is LOAD-BEARING, not laziness: a
        # stray ``-W error::pytest.PytestUnhandledThreadExceptionWarning`` on
        # the command line must NOT satisfy this pin, because the claim being
        # checked is that *orchestrator/pyproject.toml* does the promoting —
        # that is what _orch_helpers.py and conftest.py tell their readers.
        apply(ini_filters, [])
        # stacklevel is immaterial to the outcome (the pinned entry anchors on
        # the CATEGORY with an empty module field, so it matches regardless of
        # the attributed frame); it is set only to satisfy ruff's B028.
        try:
            warnings.warn(
                pytest.PytestUnhandledThreadExceptionWarning('df-4075 probe'), stacklevel=2
            )
        except pytest.PytestUnhandledThreadExceptionWarning:
            promoted = True
        else:
            promoted = False

    # pytest.fail rather than pytest.raises: DID NOT RAISE carries none of the
    # diagnostic below, and this failure needs to tell its reader exactly what
    # to restore (same register as test_lane_lock_leak_guard.py's drift fail).
    if not promoted:
        pytest.fail(
            f'{ORCH_PYPROJECT} does not promote '
            f'pytest.PytestUnhandledThreadExceptionWarning to an error. '
            f'Effective [tool.pytest.ini_options] filterwarnings for this run: '
            f'{ini_filters!r}. Add the entry {PROMOTION_ENTRY!r} to that list. '
            f'Without it a leaked aiosqlite worker thread raising '
            f'"RuntimeError: Event loop is closed" is a mere warning attributed '
            f'to an innocent test, so a degraded '
            f'reap_leaked_aiosqlite_connections() (see its MAINTENANCE NOTE) '
            f'leaks silently and the suite stays green. Both that helper and '
            f'conftest._reap_leaked_aiosqlite_connections document this '
            f'promotion as fact — restore the entry, or correct both '
            f'docstrings. Task 4075.'
        )


def test_the_promotion_is_in_effect_for_this_run(pytestconfig):
    """The promotion is live in the run actually happening, not just in the file.

    The IN-EFFECT pin. No filter manipulation at all: pytest's own
    ``catch_warnings_for_item`` context is active during a test body and has
    already applied the ini filters, so warning here exercises the real
    configuration. This catches an invocation that silences the promotion for
    this run (``-W ignore::pytest.PytestUnhandledThreadExceptionWarning``,
    ``-p no:warnings``) even though the ini text is still correct — a state
    the source pin above cannot see.
    """
    require_orchestrator_inifile(pytestconfig, subject=_PIN_SUBJECT)

    # stacklevel only satisfies ruff's B028 — see the note in the source pin.
    try:
        warnings.warn(
            pytest.PytestUnhandledThreadExceptionWarning('df-4075 probe'), stacklevel=2
        )
    except pytest.PytestUnhandledThreadExceptionWarning:
        return

    pytest.fail(
        f'pytest.PytestUnhandledThreadExceptionWarning is NOT an error in this '
        f'run, even though {ORCH_PYPROJECT} is the governing inifile. Either '
        f'the {PROMOTION_ENTRY!r} entry is missing from '
        f'[tool.pytest.ini_options] filterwarnings, or this invocation '
        f'disabled it (-W ignore::..., -p no:warnings, an outer PYTEST_ADDOPTS). '
        f'Either way the backstop documented by '
        f'reap_leaked_aiosqlite_connections() is not protecting this run. '
        f'Task 4075.'
    )


@pytest.mark.timeout(120)
def test_a_thread_exception_actually_fails_a_test_under_this_projects_inifile(
    tmp_path, pytestconfig
):
    """A real thread exception really does fail a real test under our inifile.

    The END-TO-END non-vacuity pin. The two pins above prove a filter is
    registered and active; neither can see whether pytest still routes thread
    exceptions through ``warnings.warn`` at all. On pytest 9.0.3 it does —
    ``_pytest/threadexception.py::collect_thread_exception`` calls
    ``warnings.warn(pytest.PytestUnhandledThreadExceptionWarning(msg))`` inside
    a ``try/except PytestUnhandledThreadExceptionWarning`` and re-raises out of
    the ``trylast`` ``pytest_runtest_setup``/``call``/``teardown`` hooks — but a
    future pytest that changed that would leave the ini entry INERT and the two
    docstrings false again, which is the exact failure class task 4075 exists
    to close.

    Two arms over the same probe file: the TREATMENT run under
    orchestrator/pyproject.toml must FAIL, and the CONTROL run under a
    ``filterwarnings``-free inifile must PASS. The control is what attributes
    the treatment failure to the promotion entry rather than to anything
    ambient. Precedent for a live subprocess non-vacuity probe:
    test_warm_lane_bash_bucket_placement.py's ``--collect-only`` probe, cited
    approvingly by test_marker_registration_drift.py::test_the_sweep_is_not_vacuous.
    """
    require_orchestrator_inifile(pytestconfig, subject=_PIN_SUBJECT)

    probe = tmp_path / 'test_df4075_thread_probe.py'
    probe.write_text(_PROBE_TEST_SOURCE)
    control_inifile = tmp_path / 'pyproject.toml'
    control_inifile.write_text(_CONTROL_INIFILE_SOURCE)

    def _run(inifile: Path) -> subprocess.CompletedProcess:
        # ``-o addopts=`` clears the inifile's own addopts so the probe does
        # not drag in ``-n auto --dist loadgroup -m 'not warm_lane_bash'``;
        # ``-p no:cacheprovider`` keeps the child from writing a .pytest_cache.
        # ``sanitized_probe_env`` strips PYTHONWARNINGS along with the git and
        # PYTEST_* pointers: an ambient warning-filter env var is applied by
        # CPython ahead of the inifile and would contaminate BOTH arms — an
        # inherited ``PYTHONWARNINGS=error`` fails the CONTROL arm (reading as
        # a broken harness) and can pass the TREATMENT arm for a reason that
        # has nothing to do with orchestrator/pyproject.toml.
        return subprocess.run(
            [
                sys.executable, '-m', 'pytest', str(probe),
                '-c', str(inifile),
                '-o', 'addopts=',
                '-p', 'no:cacheprovider',
                '-q',
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env=sanitized_probe_env(),
        )

    control = _run(control_inifile)
    assert control.returncode == 0, (
        f'CONTROL ARM BROKEN — the probe was expected to PASS under a '
        f'filterwarnings-free inifile, but exited {control.returncode}. The '
        f'probe or its harness is at fault, not the promotion; this run cannot '
        f'attribute anything. stdout:\n{control.stdout}\nstderr:\n{control.stderr}'
    )

    treatment = _run(ORCH_PYPROJECT)
    combined = treatment.stdout + treatment.stderr
    assert treatment.returncode != 0, (
        f'a test whose thread died with an unhandled RuntimeError PASSED under '
        f'{ORCH_PYPROJECT} (exit 0). The same probe also passes under a '
        f'filterwarnings-free control inifile, so the {PROMOTION_ENTRY!r} entry '
        f'is either missing or has gone INERT (e.g. a pytest release that no '
        f'longer routes thread exceptions through warnings.warn — see '
        f'_pytest/threadexception.py::collect_thread_exception). This is the '
        f'backstop reap_leaked_aiosqlite_connections() documents; without it a '
        f'leaked aiosqlite worker thread is a warning nobody reads. '
        f'Task 4075.\noutput:\n{combined}'
    )
    assert 'PytestUnhandledThreadExceptionWarning' in combined, (
        f'the probe failed under {ORCH_PYPROJECT}, but not for the reason this '
        f'pin claims — the output never mentions '
        f'PytestUnhandledThreadExceptionWarning, so the failure is incidental '
        f'and the pin is not measuring the promotion.\noutput:\n{combined}'
    )
    assert 'df-4075-probe-thread' in combined, (
        f'the probe failed under {ORCH_PYPROJECT} with a '
        f'PytestUnhandledThreadExceptionWarning, but the output does not name '
        f'the probe thread (df-4075-probe-thread), so the failure may come '
        f'from some other thread rather than the one this test '
        f'started.\noutput:\n{combined}'
    )
