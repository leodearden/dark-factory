"""Thread-method timeouts under xdist must run with worker restarts disabled.

THE INVARIANT. Every pytest config that pairs ``timeout_method = "thread"`` with
an xdist ``-n`` in its addopts carries ``--max-worker-restart=0``. Discovered
from ``[tool.uv.workspace].members``, so a future thread-method member is held
to it with no edit here.

WHY. Under thread method pytest-timeout answers a breach by ``os._exit()``ing
the worker, so a worker death is the routine outcome of any over-cap test. With
restarts permitted, xdist can abort the WHOLE session with an INTERNALERROR that
names no test — ``KeyError: <WorkerController gwN>`` raised from
``xdist/scheduler/loadscope.py::LoadScopeScheduling._assign_work_unit`` — when
one replacement worker finishes collecting while another is still collecting.
The negative control below reproduces that on demand, and asserts the
unregistered controller is a REPLACEMENT (an id at or beyond the initial worker
count). The field ids task 5115 measured, gw35 under ``-n auto`` on 32 cores and
gw19 under ``-n 16``, are replacement ids too.

WHAT THE FLAG DOES NOT CLOSE. Two workers dying near-simultaneously can still
abort the session in either regime:
``xdist/scheduler/loadscope.py::LoadScopeScheduling.remove_node`` hands the
first corpse's tests to a peer that is already dead (``OSError: cannot send``).
That happens before the restart cap is consulted. Task 5114 measured it in 2 of
10 unordered restart-disabled runs, so the reproduction orders its initial
deaths to isolate the path the flag does close.

THE COST is only the tally of tests after the death. The crashed test is
reported FAILED in both regimes, because
``xdist/dsession.py::DSession.worker_errordown`` calls ``handle_crashitem``
before it consults the restart cap, so no green run can turn red. A truncated
tally is already labelled partial by
``orchestrator/src/orchestrator/verify.py::_worker_death_truncation_evidence``.

Provenance: task 1907 adopted the flag in orchestrator; task 5114 measured the
abort and extended it to fused-memory.

OUT OF SCOPE: signal-method members. A signal timeout raises inside the test and
leaves the worker alive, so the case the flag exists for does not arise there.
That side is
``test_merge_gate_parallelism_config.py::test_parallel_member_addopts_do_not_copy_max_worker_restart``.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The ONE accepted spelling. The two-token ``--max-worker-restart 0`` form is
# refused for the reason ``test_merge_gate_parallelism_config.py::_flag_value``
# refuses the ``--flag=value`` form: the configs share one spelling.
RESTART_DISABLED_TOKEN = '--max-worker-restart=0'
RESTART_FLAG_PREFIX = '--max-worker-restart'
THREAD_TIMEOUT_METHOD = 'thread'
WORKERS_FLAG = '-n'

ROOT_CONFIG_NAME = '.'

# A FLOOR, not an equality: orchestrator and fused-memory at authorship.
MIN_EXPECTED_THREAD_TIMEOUT_XDIST_CONFIGS = 2

REPRO_INITIAL_WORKERS = 2

# Below the probes' @pytest.mark.timeout, so a wedged child surfaces as a
# TimeoutExpired carrying its captured output rather than as pytest's axe.
PROBE_SUBPROCESS_TIMEOUT_SECS = 150

RESTART_REPRO_GROUP = 'xdist_restart_reproduction'

SIGNAL_DIR_ENV = 'XDIST_RESTART_REPRO_SIGNAL_DIR'
INITIAL_WORKERS_ENV = 'XDIST_RESTART_REPRO_INITIAL_WORKERS'

REPRO_MODULE_COUNT = 20


def _pytest_ini_options(pyproject: pathlib.Path) -> dict:
    if not pyproject.exists():
        return {}
    data = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    return data.get('tool', {}).get('pytest', {}).get('ini_options', {})


def _thread_timeout_xdist_configs() -> dict[str, str]:
    """Config name -> addopts, for every config pairing thread timeouts with ``-n``."""
    root_data = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    members = root_data.get('tool', {}).get('uv', {}).get('workspace', {}).get('members', [])
    assert members, (
        '[tool.uv.workspace].members is empty or missing in the root '
        'pyproject.toml — this guard reads it to discover which configs to '
        'check, so it would otherwise pass vacuously.'
    )

    configs: dict[str, str] = {}
    for name in [ROOT_CONFIG_NAME, *members]:
        ini_options = _pytest_ini_options(REPO_ROOT / name / 'pyproject.toml')
        addopts = ini_options.get('addopts', '')
        if (
            ini_options.get('timeout_method') == THREAD_TIMEOUT_METHOD
            and WORKERS_FLAG in shlex.split(addopts)
        ):
            configs[name] = addopts

    assert len(configs) >= MIN_EXPECTED_THREAD_TIMEOUT_XDIST_CONFIGS, (
        f'only discovered {sorted(configs)} as configs combining '
        f'timeout_method = {THREAD_TIMEOUT_METHOD!r} with {WORKERS_FLAG} in '
        f'addopts, expected at least {MIN_EXPECTED_THREAD_TIMEOUT_XDIST_CONFIGS} '
        '(orchestrator and fused-memory are the known members). Finding fewer '
        'means this discovery walk rotted, not that the repo changed: a config '
        'that stopped being discovered is one nothing here checks any more.'
    )
    return configs


def _restart_tokens(addopts: str) -> list[str]:
    return [token for token in shlex.split(addopts) if token.startswith(RESTART_FLAG_PREFIX)]


def test_thread_timeout_xdist_configs_disable_worker_restarts() -> None:
    declared_by_config = {
        name: _restart_tokens(addopts)
        for name, addopts in _thread_timeout_xdist_configs().items()
    }
    offenders = {
        name: declared
        for name, declared in declared_by_config.items()
        if declared != [RESTART_DISABLED_TOKEN]
    }
    listing = '\n'.join(
        f'  {name}/pyproject.toml addopts declares {declared!r}'
        for name, declared in sorted(offenders.items())
    )
    assert not offenders, (
        f'these configs run timeout_method = {THREAD_TIMEOUT_METHOD!r} under '
        f'xdist {WORKERS_FLAG} but do not declare exactly '
        f'[{RESTART_DISABLED_TOKEN!r}] in addopts (tasks 1907, 5114):\n'
        f'{listing}\n'
        'Without the flag, a worker killed by a thread-method timeout can abort '
        'the whole session with an unattributed INTERNALERROR KeyError at '
        "xdist's _assign_work_unit, which this file's negative control "
        'reproduces. With it, the death becomes the attributed "crashed while '
        'running" + "worker restarting disabled" signature that verify.py\'s '
        'worker-death discriminators key on.'
    )


_SYNTHETIC_INI = '[pytest]\ntimeout = 1\ntimeout_method = thread\n'

# Orders every step with controller-side events, not sleeps:
#   - an initial worker does not start its timeout clock until the controller
#     has handled the previous initial worker's crash, so the controller never
#     reschedules a dead worker's tests onto a peer that is already dead too;
#   - the second replacement cannot finish collecting until the controller has
#     recorded the first one's collection.
# Worker identity comes from config.workerinput because PYTEST_XDIST_WORKER
# leaks from an outer xdist run.
_SYNTHETIC_CONFTEST = f'''\
import os
import pathlib
import time

import pytest

SIGNALS = pathlib.Path(os.environ['{SIGNAL_DIR_ENV}'])
INITIAL_WORKERS = int(os.environ['{INITIAL_WORKERS_ENV}'])
FIRST_REPLACEMENT = f'gw{{INITIAL_WORKERS}}'
WAIT_CAP_SECS = 60


def _is_replacement(worker_id):
    return int(worker_id[2:]) >= INITIAL_WORKERS


def _worker_id(config):
    workerinput = getattr(config, 'workerinput', None)
    return None if workerinput is None else workerinput['workerid']


def pytest_testnodeready(node):
    if _is_replacement(node.gateway.id):
        (SIGNALS / f'ready-{{node.gateway.id}}').touch()


def pytest_xdist_node_collection_finished(node, ids):
    if _is_replacement(node.gateway.id):
        (SIGNALS / f'collected-{{node.gateway.id}}').touch()


def pytest_handlecrashitem(crashitem, report, sched):
    (SIGNALS / f'crash-handled-{{report.node.gateway.id}}').touch()


def _wait_for(predicate):
    deadline = time.monotonic() + WAIT_CAP_SECS
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.05)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    worker_id = _worker_id(item.config)
    if worker_id is not None and not _is_replacement(worker_id) and worker_id != 'gw0':
        previous = f'gw{{int(worker_id[2:]) - 1}}'
        _wait_for(lambda: (SIGNALS / f'crash-handled-{{previous}}').exists())
    return (yield)


def pytest_collection_modifyitems(session, config, items):
    worker_id = _worker_id(config)
    if worker_id is None or not _is_replacement(worker_id):
        return
    _wait_for(lambda: len(list(SIGNALS.glob('ready-*'))) >= INITIAL_WORKERS)
    if worker_id != FIRST_REPLACEMENT:
        _wait_for(lambda: (SIGNALS / f'collected-{{FIRST_REPLACEMENT}}').exists())


@pytest.fixture(autouse=True)
def _initial_workers_overrun_the_timeout(request):
    if not _is_replacement(_worker_id(request.config)):
        time.sleep(30)
'''

_SYNTHETIC_MODULE = '''\
def test_one():
    pass


def test_two():
    pass


def test_three():
    pass
'''


@dataclasses.dataclass(frozen=True)
class _SyntheticSuite:
    root: pathlib.Path
    ini: pathlib.Path
    signals: pathlib.Path


@pytest.fixture
def synthetic_suite(tmp_path: pathlib.Path) -> _SyntheticSuite:
    """The reproduction suite, under *tmp_path* so no other run can collect it."""
    root = tmp_path / 'suite'
    root.mkdir()
    ini = root / 'pytest.ini'
    ini.write_text(_SYNTHETIC_INI, encoding='utf-8')
    (root / 'conftest.py').write_text(_SYNTHETIC_CONFTEST, encoding='utf-8')
    for index in range(1, REPRO_MODULE_COUNT + 1):
        (root / f'test_m{index:02d}.py').write_text(_SYNTHETIC_MODULE, encoding='utf-8')
    signals = tmp_path / 'signals'
    signals.mkdir()
    return _SyntheticSuite(root=root, ini=ini, signals=signals)


def _run_synthetic_suite(
    suite: _SyntheticSuite, *restart_args: str
) -> subprocess.CompletedProcess[str]:
    """Run the suite at ``-n REPRO_INITIAL_WORKERS --dist loadgroup`` plus *restart_args*.

    ``PYTEST_*`` is scrubbed from the child's environment so an inherited
    ``PYTEST_ADDOPTS`` (a ``-q``, say) cannot suppress the literals asserted on.
    """
    missing = [
        plugin for plugin in ('xdist', 'pytest_timeout')
        if importlib.util.find_spec(plugin) is None
    ]
    assert not missing, (
        f'{missing!r} not importable in this environment, so the reproduction '
        'cannot run; this is a missing plugin, not a reproduction failure.'
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith('PYTEST_')}
    env[SIGNAL_DIR_ENV] = str(suite.signals)
    env[INITIAL_WORKERS_ENV] = str(REPRO_INITIAL_WORKERS)
    return subprocess.run(
        [
            sys.executable, '-m', 'pytest',
            '-c', str(suite.ini),
            '-p', 'no:cacheprovider',
            WORKERS_FLAG, str(REPRO_INITIAL_WORKERS),
            '--dist', 'loadgroup',
            *restart_args,
        ],
        cwd=str(suite.root),
        env=env,
        capture_output=True,
        text=True,
        timeout=PROBE_SUBPROCESS_TIMEOUT_SECS,
        check=False,
    )


def _captured(result: subprocess.CompletedProcess[str]) -> str:
    return f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'


@pytest.mark.xdist_group(RESTART_REPRO_GROUP)
@pytest.mark.timeout(180)
def test_restarts_permitted_abort_the_session_unattributed(
    synthetic_suite: _SyntheticSuite,
) -> None:
    """THE NEGATIVE CONTROL: xdist's default restart cap, as a member without the flag runs."""
    result = _run_synthetic_suite(synthetic_suite)
    output = result.stdout + result.stderr

    assert result.returncode == pytest.ExitCode.INTERNAL_ERROR, (
        f'with restarts permitted the reproduction exited {result.returncode}, '
        f'not {int(pytest.ExitCode.INTERNAL_ERROR)} (INTERNAL_ERROR). If this '
        'stopped reproducing (after an xdist upgrade, say), the positive arm no '
        'longer proves the flag is what prevents the abort: re-measure the '
        "invariant's rationale rather than loosening this assertion.\n"
        f'{_captured(result)}'
    )
    key_error = re.search(r'KeyError: <WorkerController gw(\d+)>', output)
    assert key_error is not None, (
        'the session aborted, but not with the unregistered-controller KeyError '
        'this file documents.\n'
        f'{_captured(result)}'
    )
    assert int(key_error.group(1)) >= REPRO_INITIAL_WORKERS, (
        f'the KeyError names gw{key_error.group(1)}, an INITIAL worker; the '
        'documented mechanism is a REPLACEMENT (id >= '
        f'{REPRO_INITIAL_WORKERS}) that joined the scheduler after the initial '
        'distribution.\n'
        f'{_captured(result)}'
    )
    assert '_assign_work_unit' in output, (
        'the KeyError was not raised from _assign_work_unit, so this is not the '
        'abort path this file documents.\n'
        f'{_captured(result)}'
    )
    assert 'crashed while running' not in output, (
        'the abort attributed the worker death to a test, so it is no longer '
        'the unattributed failure the flag exists to prevent.\n'
        f'{_captured(result)}'
    )


@pytest.mark.xdist_group(RESTART_REPRO_GROUP)
@pytest.mark.timeout(180)
def test_restarts_disabled_attribute_the_worker_death(
    synthetic_suite: _SyntheticSuite,
) -> None:
    """THE POSITIVE ARM: the same suite with the flag ends as an attributed failure."""
    result = _run_synthetic_suite(synthetic_suite, RESTART_DISABLED_TOKEN)
    output = result.stdout + result.stderr

    assert result.returncode == pytest.ExitCode.TESTS_FAILED, (
        f'with {RESTART_DISABLED_TOKEN} the reproduction exited '
        f'{result.returncode}, not {int(pytest.ExitCode.TESTS_FAILED)} '
        '(TESTS_FAILED).\n'
        f'{_captured(result)}'
    )
    assert re.search(r'^INTERNALERROR>', output, re.MULTILINE) is None, (
        f'the session still aborted with an INTERNALERROR under '
        f'{RESTART_DISABLED_TOKEN}, so the flag no longer closes the path.\n'
        f'{_captured(result)}'
    )
    assert 'worker restarting disabled' in output, (
        'xdist did not report that restarting was disabled.\n'
        f'{_captured(result)}'
    )
    assert re.search(r"crashed while running '[^']+\.py::[^']+'", output), (
        'the worker death was not attributed to a named test.\n'
        f'{_captured(result)}'
    )
    assert re.search(r'^FAILED ', output, re.MULTILINE), (
        'no FAILED line in the short test summary.\n'
        f'{_captured(result)}'
    )
