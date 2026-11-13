"""xdist worker restarts stay disabled for thread-method configs and for the recorded configs.

THE INVARIANT has two rules. Both read the configs that
``tests/scripts/pytest_config_discovery.py::discovered_pytest_configs`` discovers
from ``[tool.uv.workspace].members``, so a new member is held to them with no
edit here.

  * Every pytest config that sets ``timeout_method = "thread"`` carries
    ``--max-worker-restart=0``, whether or not its own addopts ask for xdist:
    verify injects ``-n`` into every gated pytest leg
    (``orchestrator/src/orchestrator/verify.py::_with_pytest_numprocesses_str``,
    driven by ``verify_admission_pytest_n``), so a thread-method member runs
    under xdist there regardless. No config runs thread since task 6354; the
    rule holds any future one.
  * The configs in ``RESTART_DISABLED_CONFIGS``, orchestrator and fused-memory,
    keep the flag under signal. The timeout method and the restart cap are
    separate axes, and task 6354 moved only the method: a signal timeout no
    longer kills a worker, but a segfault, an OOM kill or an ``os._exit`` in
    code under test still does, and with restarts permitted each replacement
    re-collects the whole suite.

WHY. Under thread method pytest-timeout answers a breach by ``os._exit()``ing
the worker, so a worker death is the routine outcome of any over-cap test. With
restarts permitted, xdist can abort the WHOLE session with an INTERNALERROR that
names no test — ``KeyError: <WorkerController gwN>`` raised from
``xdist/scheduler/loadscope.py::LoadScopeScheduling._assign_work_unit`` — when
one replacement worker finishes collecting while another is still collecting.
The negative control below reproduces that on demand, and asserts the
unregistered controller is a REPLACEMENT (an id at or beyond the initial worker
count).

WHAT THE FLAG DOES NOT CLOSE. Two workers dying near-simultaneously can still
abort the session in either regime:
``xdist/scheduler/loadscope.py::LoadScopeScheduling.remove_node`` hands the
first corpse's tests to a peer that is already dead (``OSError: cannot send``)
before the restart cap is consulted. The reproduction orders its initial deaths
so that it exercises only the path the flag does close.

THE COST is only the tally of tests after the death. The crashed test is
reported FAILED in both regimes, because
``xdist/dsession.py::DSession.worker_errordown`` calls ``handle_crashitem``
before it consults the restart cap, so no green run can turn red. A truncated
tally is already labelled partial by
``orchestrator/src/orchestrator/verify.py::_worker_death_truncation_evidence``.

OUT OF SCOPE: the other signal-method members. orchestrator and fused-memory are
signal members that DO carry the flag, for the reason above. The other parallel
members do not, because no measured need was found (task 5408); that side is
``test_merge_gate_parallelism_config.py::test_parallel_member_addopts_do_not_copy_max_worker_restart``.
"""
from __future__ import annotations

import dataclasses
import getpass
import importlib.util
import os
import pathlib
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping

import pytest
from pytest_config_discovery import discovered_pytest_configs
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# The configs that keep the flag under timeout_method = "signal". The module
# docstring states why.
RESTART_DISABLED_CONFIGS = frozenset({'orchestrator', 'fused-memory'})

# The ONE accepted spelling. The two-token ``--max-worker-restart 0`` form is
# refused for the reason ``test_merge_gate_parallelism_config.py::_flag_value``
# refuses the ``--flag=value`` form: the configs share one spelling.
RESTART_DISABLED_TOKEN = '--max-worker-restart=0'
RESTART_FLAG_PREFIX = '--max-worker-restart'
RESTART_FLAG_SPELLING = re.compile(rf'{re.escape(RESTART_FLAG_PREFIX)}=\d+')
THREAD_TIMEOUT_METHOD = 'thread'

REPRO_INITIAL_WORKERS = 2

# Below the probes' @pytest.mark.timeout, so a wedged child surfaces as a
# TimeoutExpired carrying its captured output rather than as pytest's axe.
PROBE_SUBPROCESS_TIMEOUT_SECS = 150

RESTART_REPRO_GROUP = 'xdist_restart_reproduction'

SIGNAL_DIR_ENV = 'XDIST_RESTART_REPRO_SIGNAL_DIR'
INITIAL_WORKERS_ENV = 'XDIST_RESTART_REPRO_INITIAL_WORKERS'

REPRO_MODULE_COUNT = 20


def _restart_tokens(addopts: str) -> list[str]:
    return [token for token in shlex.split(addopts) if token.startswith(RESTART_FLAG_PREFIX)]


def _restart_offenders(ini_options_by_config: Mapping[str, Mapping]) -> dict[str, list[str]]:
    """Config name -> its restart tokens, for each config not declaring exactly the one token."""
    declared_by_config = {
        name: _restart_tokens(ini_options.get('addopts', ''))
        for name, ini_options in ini_options_by_config.items()
    }
    return {
        name: declared
        for name, declared in declared_by_config.items()
        if declared != [RESTART_DISABLED_TOKEN]
    }


def _offender_listing(offenders: Mapping[str, list[str]]) -> str:
    return '\n'.join(
        f'  {name}/pyproject.toml addopts declares {declared!r}'
        for name, declared in sorted(offenders.items())
    )


def _thread_method_configs(
    ini_options_by_config: Mapping[str, Mapping],
) -> dict[str, Mapping]:
    return {
        name: ini_options
        for name, ini_options in ini_options_by_config.items()
        if ini_options.get('timeout_method') == THREAD_TIMEOUT_METHOD
    }


def test_restart_offenders_name_every_config_without_exactly_the_one_token() -> None:
    offenders = _restart_offenders({
        'no-addopts': {},
        'flag-missing': {'addopts': '-n auto --dist loadgroup'},
        'two-token-spelling': {'addopts': '-n auto --max-worker-restart 0'},
        'restarts-permitted': {'addopts': '-n auto --max-worker-restart=1'},
        'declared-twice': {'addopts': '--max-worker-restart=0 --max-worker-restart=0'},
        'correct': {'addopts': "-n auto --max-worker-restart=0 -m 'not integration'"},
    })

    assert offenders == {
        'no-addopts': [],
        'flag-missing': [],
        'two-token-spelling': [RESTART_FLAG_PREFIX],
        'restarts-permitted': ['--max-worker-restart=1'],
        'declared-twice': [RESTART_DISABLED_TOKEN, RESTART_DISABLED_TOKEN],
    }


def test_the_thread_rule_holds_thread_method_configs_only() -> None:
    offenders = _restart_offenders(_thread_method_configs({
        'thread-without-flag': {'timeout_method': 'thread', 'addopts': '-n auto'},
        'thread-with-flag': {'timeout_method': 'thread', 'addopts': RESTART_DISABLED_TOKEN},
        'signal-without-flag': {'timeout_method': 'signal', 'addopts': '-n auto'},
        'method-unset': {'addopts': '-n auto'},
    }))

    assert offenders == {'thread-without-flag': []}


def test_thread_timeout_configs_disable_worker_restarts() -> None:
    offenders = _restart_offenders(
        _thread_method_configs(discovered_pytest_configs())
    )
    assert not offenders, (
        f'these configs run timeout_method = {THREAD_TIMEOUT_METHOD!r} but do '
        f'not declare exactly [{RESTART_DISABLED_TOKEN!r}] in addopts (tasks '
        '1907, 5114). A missing -n in addopts does not exempt them: verify '
        'injects one into every gated pytest leg.\n'
        f'{_offender_listing(offenders)}\n'
        'Without the flag, a worker killed by a thread-method timeout can abort '
        'the whole session with an unattributed INTERNALERROR KeyError at '
        "xdist's _assign_work_unit, which this file's negative control "
        'reproduces. With it, the death becomes the attributed "crashed while '
        'running" + "worker restarting disabled" signature that verify.py\'s '
        'worker-death discriminators key on.'
    )


def test_recorded_configs_keep_worker_restarts_disabled() -> None:
    configs = discovered_pytest_configs()
    undiscovered = sorted(RESTART_DISABLED_CONFIGS - configs.keys())
    assert not undiscovered, (
        f'RESTART_DISABLED_CONFIGS records {undiscovered!r}, but no pytest config '
        'is discovered under that name. A renamed or removed member must update '
        'the recorded set in the same change.'
    )
    offenders = _restart_offenders({name: configs[name] for name in RESTART_DISABLED_CONFIGS})
    assert not offenders, (
        'these configs are recorded in RESTART_DISABLED_CONFIGS but do not '
        f'declare exactly [{RESTART_DISABLED_TOKEN!r}] in addopts.\n'
        f'{_offender_listing(offenders)}\n'
        'They run timeout_method = "signal" since task 6354, which moved only the '
        'method. Under signal the cap governs the worker deaths that remain '
        '(segfault, OOM kill, os._exit in code under test). With restarts '
        'permitted, a replacement worker re-collects the whole suite, and two '
        'concurrent replacements abort the session with the unattributed '
        "INTERNALERROR KeyError this file's negative control reproduces. Dropping "
        'the flag is a separate decision that needs its own measurement: take the '
        "config out of RESTART_DISABLED_CONFIGS and record why in this file's "
        'module docstring in the same change.'
    )


_SYNTHETIC_INI = '[pytest]\ntimeout = 1\ntimeout_method = thread\n'

# Orders every step with controller-side events, not sleeps:
#   - an initial worker does not start its timeout clock until the controller
#     has handled the previous initial worker's crash, so the controller never
#     reschedules a dead worker's tests onto a peer that is already dead too;
#   - the second replacement cannot finish collecting until the controller has
#     recorded the first one's collection.
_SYNTHETIC_CONFTEST = '''\
import os
import pathlib
import time

import pytest

SIGNALS = pathlib.Path(os.environ['XDIST_RESTART_REPRO_SIGNAL_DIR'])
INITIAL_WORKERS = int(os.environ['XDIST_RESTART_REPRO_INITIAL_WORKERS'])
FIRST_REPLACEMENT = f'gw{INITIAL_WORKERS}'
WAIT_CAP_SECS = 60


def _is_replacement(worker_id):
    return int(worker_id[2:]) >= INITIAL_WORKERS


def _worker_id(config):
    workerinput = getattr(config, 'workerinput', None)
    return None if workerinput is None else workerinput['workerid']


def pytest_testnodeready(node):
    if _is_replacement(node.gateway.id):
        (SIGNALS / f'ready-{node.gateway.id}').touch()


def pytest_xdist_node_collection_finished(node, ids):
    if _is_replacement(node.gateway.id):
        (SIGNALS / f'collected-{node.gateway.id}').touch()


def pytest_handlecrashitem(crashitem, report, sched):
    (SIGNALS / f'crash-handled-{report.node.gateway.id}').touch()


def _wait_for(predicate):
    deadline = time.monotonic() + WAIT_CAP_SECS
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.05)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    worker_id = _worker_id(item.config)
    if worker_id is not None and not _is_replacement(worker_id) and worker_id != 'gw0':
        previous = f'gw{int(worker_id[2:]) - 1}'
        _wait_for(lambda: (SIGNALS / f'crash-handled-{previous}').exists())
    return (yield)


def pytest_collection_modifyitems(session, config, items):
    worker_id = _worker_id(config)
    if worker_id is None or not _is_replacement(worker_id):
        return
    _wait_for(lambda: len(list(SIGNALS.glob('ready-*'))) >= INITIAL_WORKERS)
    if worker_id != FIRST_REPLACEMENT:
        _wait_for(lambda: (SIGNALS / f'collected-{FIRST_REPLACEMENT}').exists())


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

_RECORDER_INI = '[pytest]\n'

_TEMPROOT_RECORDER_MODULE = '''\
import pathlib
import tempfile


def test_records_the_temproot(tmp_path):
    record = pathlib.Path(__file__).resolve().with_name('seen-temproot')
    record.write_text(tempfile.gettempdir())
'''


@dataclasses.dataclass(frozen=True)
class _SyntheticSuite:
    root: pathlib.Path
    ini: pathlib.Path
    signals: pathlib.Path
    basetemp: pathlib.Path


def _write_suite(
    parent: pathlib.Path, *, ini: str, sources: Mapping[str, str]
) -> _SyntheticSuite:
    """Lay the suite out under *parent*; basetemp is a not-yet-created sibling of its root."""
    root = parent / 'suite'
    root.mkdir()
    ini_path = root / 'pytest.ini'
    ini_path.write_text(ini, encoding='utf-8')
    for name, text in sources.items():
        (root / name).write_text(text, encoding='utf-8')
    signals = parent / 'signals'
    signals.mkdir()
    return _SyntheticSuite(root=root, ini=ini_path, signals=signals, basetemp=parent / 'basetemp')


@pytest.fixture
def synthetic_suite(tmp_path: pathlib.Path) -> _SyntheticSuite:
    """The reproduction suite, under *tmp_path* so no other run can collect it."""
    unread = [
        name for name in (SIGNAL_DIR_ENV, INITIAL_WORKERS_ENV)
        if f"os.environ['{name}']" not in _SYNTHETIC_CONFTEST
    ]
    assert not unread, (
        f'_SYNTHETIC_CONFTEST does not read {unread!r}, the environment '
        '_run_synthetic_suite sets for it: the two spellings drifted apart.'
    )
    return _write_suite(
        tmp_path,
        ini=_SYNTHETIC_INI,
        sources={
            'conftest.py': _SYNTHETIC_CONFTEST,
            **{
                f'test_m{index:02d}.py': _SYNTHETIC_MODULE
                for index in range(1, REPRO_MODULE_COUNT + 1)
            },
        },
    )


def _run_synthetic_suite(
    suite: _SyntheticSuite, *restart_args: str
) -> subprocess.CompletedProcess[str]:
    """Run the suite at ``-n REPRO_INITIAL_WORKERS --dist loadgroup`` plus *restart_args*.

    *restart_args* must each be ``--max-worker-restart=<n>``; anything else is refused.

    ``PYTEST_*`` is scrubbed from the child's environment so an inherited
    ``PYTEST_ADDOPTS`` (a ``-q``, say) cannot suppress the literals asserted on.

    The child runs under the suite's private basetemp, never in the host's
    default pytest temp root; why:
    ``test_xdist_worker_restart_policy.py::test_a_synthetic_suite_run_leaves_the_hosts_default_temp_root_untouched``.
    """
    refused = [arg for arg in restart_args if not RESTART_FLAG_SPELLING.fullmatch(arg)]
    if refused:
        raise ValueError(
            f'_run_synthetic_suite takes only {RESTART_FLAG_PREFIX}=<n> flags (its '
            f'basetemp and output options are fixed); refused: {refused!r}'
        )
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
            '-n', str(REPRO_INITIAL_WORKERS),
            '--dist', 'loadgroup',
            *restart_args,
            f'--basetemp={suite.basetemp}',
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


def _listing(root: pathlib.Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob('*'))


@pytest.mark.xdist_group(RESTART_REPRO_GROUP)
@pytest.mark.timeout(300)
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
@pytest.mark.timeout(300)
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


@pytest.mark.xdist_group(RESTART_REPRO_GROUP)
@pytest.mark.timeout(300)
def test_a_synthetic_suite_run_leaves_the_hosts_default_temp_root_untouched(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = tmp_path / 'host-temp'
    (host / f'pytest-of-{getpass.getuser()}' / 'garbage-seeded' / 'leftover').mkdir(parents=True)
    before = _listing(host)
    monkeypatch.setenv('TMPDIR', str(host))
    suite = _write_suite(
        tmp_path,
        ini=_RECORDER_INI,
        sources={'test_records_temproot.py': _TEMPROOT_RECORDER_MODULE},
    )

    result = _run_synthetic_suite(suite)

    assert result.returncode == pytest.ExitCode.OK, (
        f'the recorder suite exited {result.returncode}, not '
        f'{int(pytest.ExitCode.OK)} (OK).\n'
        f'{_captured(result)}'
    )
    seen = pathlib.Path((suite.root / 'seen-temproot').read_text(encoding='utf-8'))
    assert seen.resolve() == host.resolve(), (
        f'the child ran against temp root {seen}, not the stand-in {host}, so '
        'the listing check below would pass vacuously.\n'
        f'{_captured(result)}'
    )
    after = _listing(host)
    assert after == before, (
        'a synthetic suite run must not create, lock or clean dirs in the host '
        "default temp root: its exit-time sweep of other sessions' dead dirs "
        'runs outside pytest-timeout and counts against the child\'s timeout.\n'
        f'before: {before}\nafter:  {after}\n'
        f'{_captured(result)}'
    )
    assert suite.basetemp.is_dir(), (
        f'the session did not create its private basetemp {suite.basetemp}.\n'
        f'{_captured(result)}'
    )


def test_an_argument_other_than_a_restart_flag_is_refused_before_any_session_starts(
    tmp_path: pathlib.Path,
) -> None:
    smuggled = tmp_path / 'elsewhere'
    smuggle = f'--basetemp={smuggled}'
    suite = _write_suite(
        tmp_path,
        ini=_RECORDER_INI,
        sources={'test_records_temproot.py': _TEMPROOT_RECORDER_MODULE},
    )

    with pytest.raises(ValueError, match=re.escape(RESTART_FLAG_PREFIX)) as refused:
        _run_synthetic_suite(suite, smuggle)

    assert smuggle in str(refused.value)
    assert not suite.basetemp.exists(), 'a session started anyway'
    assert not smuggled.exists(), 'a session started anyway'
    assert not (suite.root / 'seen-temproot').exists(), 'a session started anyway'
