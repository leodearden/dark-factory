"""Every pytest config in this repo declares a per-test wall-clock cap, and they agree on it.

Task 5442. One invariant family, one file — the convention
``test_merge_gate_parallelism_config.py``'s docstring establishes. ``tests/scripts/``
is the right home because the invariant is repo-WIDE (it reads every
``pyproject.toml``, the repo root's included) and because this directory carries
its own module config (``tests/scripts/orchestrator.yaml``), so these guards
actually run on merge verify rather than only when someone remembers them.

WHAT WAS BROKEN. pytest reads exactly ONE ``[tool.pytest.ini_options]`` — the
rootdir's inifile — and never merges across ``pyproject.toml`` files. Five
workspace members declared ``timeout``/``timeout_method``; the repo ROOT declared
neither, and neither did ``cockpit`` or ``sampler``. So any invocation whose
rootdir resolved to one of those three ran with NO per-test wall-clock cap at
all: a hung test hung the whole session with no output instead of failing loud.
MEASURED at base ``832d6faf16``, from the repo root: a probe collected there
resolved ``inipath = pyproject.toml``, ``getini('timeout') == ''`` and
``getini('timeout_method') == ''``, and ``pytest escalation/tests`` ran serial and
uncapped for 111.83s.

THE GUARDS ARE VALUE-AGNOSTIC, DELIBERATELY. Not one assertion here names the
number. A guard spelling ``assert timeout == 300`` would be a ninth copy of the
configuration (SPOT) and would go red the next time the value is honestly
re-measured — punishing exactly the behaviour
``plans/pytest-per-test-timeout-measurement-2026-09-17.md`` establishes. The
number lives in the configs; these guards pin the RELATIONSHIPS between them:
that a root-bound run resolves one at all, that every config declares one, that
they all declare the SAME one, and that CONTRIBUTING.md documents that same one.

The canonical rationale for WHY the cap exists and what it catches is
``shared/pyproject.toml``'s ``timeout`` block, and the derivation of its value is
``plans/pytest-per-test-timeout-measurement-2026-09-17.md``. Neither is restated
here.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile
import tomllib

import pytest

REPO_ROOT = pathlib.Path(__file__).parents[2]

# Spelled out rather than imported from a sibling guard. ``tests/scripts/conftest.py``
# records that this idiom is DELIBERATELY not de-duplicated — 29 files spell it,
# collapsing three of them would create an inconsistency, and a wrong `parents`
# index fails loudly at import rather than silently agreeing with a sibling.

# Bounds the probe subprocess itself, INSIDE the probe test's
# ``@pytest.mark.timeout``, so a wedged pytest surfaces as a ``TimeoutExpired``
# carrying its captured output rather than as pytest's axe carrying nothing.
# Same arrangement, and the same reasoning, as
# ``test_merge_gate_parallelism_config.py::PROBE_SUBPROCESS_TIMEOUT_SECS``.
PROBE_SUBPROCESS_TIMEOUT_SECS = 100

# pytest's exit code for "no tests were collected" — the vacuity signal that
# matters here. A probe that is never collected would let the returncode
# assertion below pass for the wrong reason once the config is fixed.
PYTEST_NO_TESTS_COLLECTED_RC = 5

# The probe directory lives under REPO_ROOT and NOWHERE ELSE, because the
# rootdir is what is under test. MEASURED: the same probe written to a
# ``tmp_path`` and run with ``cwd=REPO_ROOT`` resolved ``rootpath=/tmp/...`` and
# ``inipath=None`` — pytest took the args' own ancestor, so the run read no
# inifile at all and the probe would have been asserting about nothing.
#
# Dot-prefixed so no OTHER pytest run can collect it while it exists: pytest's
# default ``norecursedirs`` skips ``.*``, and passing this probe's path
# explicitly is what still lets THIS run reach it.
PROBE_DIR_PREFIX = '.pytest-root-timeout-probe-'

# The probe asserts from INSIDE a real root-bound session, which is the only
# vantage point that can answer "what does this invocation actually resolve?".
# ``getini('timeout')`` is a STRING here, not an int — pytest-timeout declares it
# as one, and an unset value arrives as ``''`` rather than as None (measured on
# the pinned pytest-timeout). So the probe converts before comparing, and its
# messages name the root pyproject.toml, since that is the file a reader has to
# go edit.
_PROBE_SRC = '''\
def test_the_resolved_config_carries_a_per_test_timeout(request):
    inipath = request.config.inipath
    timeout = request.config.getini("timeout")
    method = request.config.getini("timeout_method")

    assert str(timeout).strip(), (
        f"this run resolved {inipath} as its inifile and that config declares no "
        f"`timeout`, so every test in it runs with NO per-test wall-clock cap: a "
        f"hung test hangs the whole session with no output instead of failing "
        f"loud. Add `timeout` to that file's [tool.pytest.ini_options] (task 5442)"
    )
    assert float(timeout) > 0, (
        f"{inipath} declares timeout={timeout!r}, which is not a positive number "
        f"of seconds; 0 disables the cap outright (task 5442)"
    )
    assert str(method).strip(), (
        f"this run resolved {inipath} as its inifile and that config declares no "
        f"`timeout_method`, so the cap's enforcement mechanism is whatever "
        f"pytest-timeout defaults to rather than one this repo chose (task 5442)"
    )
'''


@pytest.fixture
def root_bound_probe():
    """A throwaway probe test file inside REPO_ROOT, removed afterwards.

    Inside the repo, not in ``tmp_path``: see ``PROBE_DIR_PREFIX`` for the
    measurement that rules ``tmp_path`` out. Removed in a ``finally`` so a failing
    probe cannot leave a stray ``test_*.py`` in the working tree.
    """
    directory = pathlib.Path(tempfile.mkdtemp(dir=REPO_ROOT, prefix=PROBE_DIR_PREFIX))
    try:
        probe = directory / 'test_root_bound_timeout_probe.py'
        probe.write_text(_PROBE_SRC, encoding='utf-8')
        yield probe
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _run_probe(probe: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Collect *probe* in a real pytest session rooted at the repo root.

    ``cwd=REPO_ROOT`` and a probe path under it are what make the ROOT
    ``pyproject.toml`` the inifile, which is the resolution under test.

    ``-o addopts=`` neutralises the root ``addopts``, whose ``-m 'not smoke and not
    integration and not warm_lane_bash'`` would otherwise be free to deselect the
    probe and turn this guard into a "no tests ran" that says nothing about
    timeouts. ``-p no:cacheprovider`` keeps the run from writing a
    ``.pytest_cache`` into the tree.

    ``sys.executable -m pytest`` rather than a bare ``pytest``: the interpreter
    running this guard is the one whose resolution is being asked about, so
    naming it explicitly removes any dependence on an ambient ``PATH``.
    """
    return subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            str(probe),
            '-q',
            '-p',
            'no:cacheprovider',
            '-o',
            'addopts=',
        ],
        capture_output=True,
        text=True,
        timeout=PROBE_SUBPROCESS_TIMEOUT_SECS,
        cwd=str(REPO_ROOT),
        check=False,
    )


@pytest.mark.timeout(120)
def test_root_bound_run_resolves_a_per_test_timeout(root_bound_probe: pathlib.Path) -> None:
    """RUN pytest from the repo root and ask the session what cap it resolved.

    The headline defect is not "a key is missing from a TOML table" — it is "a
    bare root-bound pytest run has no per-test wall-clock cap whatsoever", and
    the mechanism is inifile RESOLUTION. A structural presence check (
    ``test_every_pytest_config_declares_a_per_test_timeout`` below) would go
    green on a key that a root-bound run does not actually resolve: a wrong
    table, a stray second config, a rootdir that resolves elsewhere. Only
    running it can tell the difference.

    The explicit ``@pytest.mark.timeout`` is not decoration. This guard is about
    the suite's default per-test budget, so it must never be able to SIT on that
    default — 120s TIGHTENS rather than raises (the scripts verify leg passes
    ``--timeout=300``), and a probe that runs one trivial test has no business
    taking two minutes even behind a pytest startup on a loaded 32-core host.
    Same reasoning as ``test_merge_gate_parallelism_config.py``'s probes.

    MEASURED RED at base ``832d6faf16``: the probe resolved
    ``inipath=<repo>/pyproject.toml`` with ``getini('timeout') == ''`` and
    ``getini('timeout_method') == ''``.
    """
    result = _run_probe(root_bound_probe)

    # (a) NON-VACUITY, first: a probe that was never collected would satisfy
    # nothing, and exit code 5 is pytest saying exactly that.
    assert result.returncode != PYTEST_NO_TESTS_COLLECTED_RC, (
        f'the root-bound probe collected no tests at all (exit '
        f'{PYTEST_NO_TESTS_COLLECTED_RC}), so this guard would be asserting about '
        f'an empty session rather than about the root inifile (task 5442). Check '
        f'that {PROBE_DIR_PREFIX!r} is still reachable from the repo root and that '
        f'`-o addopts=` still neutralises the root marker deselection.\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    )

    # (b) THE INVARIANT. The probe's own assertions name the resolved inifile
    # and the remedy; they are reproduced in this message rather than restated.
    assert result.returncode == 0, (
        f'a pytest run rooted at the repo root does not resolve a usable per-test '
        f'wall-clock cap (exit {result.returncode}, task 5442). pytest reads '
        f'exactly ONE inifile — the rootdir\'s — and never merges across '
        f'pyproject.toml files, so a member declaring `timeout` does nothing for a '
        f'root-bound run. Add `timeout` and `timeout_method` to the ROOT '
        f'pyproject.toml\'s [tool.pytest.ini_options]; the value\'s provenance is '
        f'plans/pytest-per-test-timeout-measurement-2026-09-17.md and the rationale '
        f'is shared/pyproject.toml.\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    )


# ``'.'`` names the ROOT pyproject.toml, the convention
# ``test_merge_gate_parallelism_config.py::_declared_addopts`` already
# established: the root is a genuinely different file from any member's, and
# naming it this way lets it read as one more entry in the sweep instead of a
# special case bolted onto it.
ROOT_CONFIG_NAME = '.'

# A FLOOR, not an equality — 8 at authorship (the repo root plus all seven
# workspace members) — so a member added later is swept with no edit here, while
# a discovery that rots and finds fewer still fails loudly. Same idiom, and the
# same reason, as ``test_marker_registration_drift.py::_MIN_EXPECTED_TEST_FILES``.
MIN_EXPECTED_PYTEST_CONFIGS = 8

# The two keys every config must carry. `timeout` alone is not enough: without
# `timeout_method` the enforcement mechanism is whatever pytest-timeout defaults
# to rather than one this repo chose against a measured failure.
REQUIRED_TIMEOUT_KEYS = ('timeout', 'timeout_method')


def _pytest_ini_options(pyproject: pathlib.Path) -> dict | None:
    """``[tool.pytest.ini_options]`` for *pyproject*, or ``None`` if it declares none.

    Shape copied from ``test_pytest_workspace_collection.py::_pytest_ini_options``,
    with ONE deliberate difference: that helper returns ``{}`` both for an absent
    file and for a config that declares no pytest table, which is right for a
    caller reading a single key out of it. This file DISCOVERS configs by the
    presence of that table, so the two cases have to be distinguishable — an
    empty-but-present table would otherwise be skipped by discovery rather than
    reported as a config missing both keys, which is a vacuity hole exactly where
    this file can least afford one.
    """
    if not pyproject.exists():
        return None
    data = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    pytest_table = data.get('tool', {}).get('pytest', {})
    if 'ini_options' not in pytest_table:
        return None
    return pytest_table['ini_options']


def discovered_pytest_configs() -> dict[str, dict]:
    """Every pyproject.toml in this repo declaring ``[tool.pytest.ini_options]``.

    DISCOVERED from ``[tool.uv.workspace].members`` rather than hardcoded, so a
    new member joins the sweep with no edit here. That is the whole point: the
    defect this file guards against is a config that nobody remembered to give a
    cap, and a hand-written list is the same class of forgetting.

    Both anti-vacuity guards live HERE rather than in each caller, because they
    are properties of DISCOVERY and every sweep in this file depends on them.
    """
    root_pyproject = REPO_ROOT / 'pyproject.toml'
    root_data = tomllib.loads(root_pyproject.read_text(encoding='utf-8'))
    members = root_data.get('tool', {}).get('uv', {}).get('workspace', {}).get('members', [])

    # A typo'd key here would make the loop below iterate nothing and turn every
    # sweep in this file into a silent pass.
    assert members, (
        '[tool.uv.workspace].members is empty or missing in the root '
        'pyproject.toml — this guard reads it to discover which subprojects to '
        'check, so it would otherwise pass vacuously.'
    )

    configs: dict[str, dict] = {}
    for name in [ROOT_CONFIG_NAME, *members]:
        ini_options = _pytest_ini_options(REPO_ROOT / name / 'pyproject.toml')
        if ini_options is not None:
            configs[name] = ini_options

    assert len(configs) >= MIN_EXPECTED_PYTEST_CONFIGS, (
        f'only discovered {len(configs)} pytest configs '
        f'({sorted(configs)}), expected at least '
        f'{MIN_EXPECTED_PYTEST_CONFIGS} (task 5442). The sweep may be looking in '
        f'the wrong place — it reads the root pyproject.toml plus one per '
        f'[tool.uv.workspace].members entry, with {ROOT_CONFIG_NAME!r} naming the '
        f'root. A config that stopped being discovered is a config nothing in '
        f'this file checks any more.'
    )
    return configs


def test_every_pytest_config_declares_a_per_test_timeout() -> None:
    """All eight configs, root included, must carry `timeout` AND `timeout_method`.

    The generalisation of the behavioural probe above. That one proves the
    MECHANISM is fixed for the rootdir a bare repo-root run resolves; this one
    proves the COVERAGE, and catches the next workspace member added without a
    cap — which is how cockpit and sampler came to have none.

    A config missing these keys is not a style problem. pytest reads exactly ONE
    inifile, so every run whose rootdir resolves to that config runs with no
    per-test wall-clock cap at all: a hung test hangs the whole session with no
    output instead of failing loud with a traceback.

    MEASURED RED at base 832d6faf16 on `cockpit/pyproject.toml` and
    `sampler/pyproject.toml`, which declared neither key while both already
    shipped `pytest-timeout>=2.4.0` in their dev group — so at both the plugin was
    installed and inert.
    """
    configs = discovered_pytest_configs()

    uncapped = {
        name: [key for key in REQUIRED_TIMEOUT_KEYS if key not in ini_options]
        for name, ini_options in sorted(configs.items())
    }
    uncapped = {name: missing for name, missing in uncapped.items() if missing}

    assert not uncapped, (
        'these pytest configs declare no per-test wall-clock cap (task 5442):\n'
        + '\n'.join(
            f'  {name}/pyproject.toml — missing {missing}'
            for name, missing in uncapped.items()
        )
        + '\n\npytest reads exactly ONE [tool.pytest.ini_options] — the rootdir\'s '
        'inifile — and never merges across pyproject.toml files, so a sibling '
        'declaring these keys does nothing for a run rooted at one of the files '
        'above: every test in it runs uncapped, and a hang takes the whole '
        'session with no output rather than failing loud.\n'
        'REMEDY: copy both keys from any sibling config — every config in this '
        'repo carries the same value, which '
        'test_every_pytest_config_declares_the_same_timeout enforces. The value\'s '
        'provenance is plans/pytest-per-test-timeout-measurement-2026-09-17.md; '
        'the rationale for the setting is shared/pyproject.toml.'
    )


def _grouped_by_value(timeouts: dict[str, object]) -> str:
    """Config names grouped under each distinct timeout, for a failure message.

    Grouped rather than listed flat so a reader sees WHICH files disagree and in
    WHICH direction — which is the whole diagnosis — instead of having to
    reconstruct it from eight lines of `name: value`.
    """
    by_value: dict[object, list[str]] = {}
    for name, value in sorted(timeouts.items()):
        by_value.setdefault(value, []).append(f'{name}/pyproject.toml')
    return '\n'.join(
        f'  timeout = {value!r}: {", ".join(names)}'
        for value, names in sorted(by_value.items(), key=lambda item: repr(item[0]))
    )


def test_every_pytest_config_declares_the_same_timeout() -> None:
    """One value across all eight configs — asserted WITHOUT naming the number.

    A split value means the cap a test runs under depends on which directory
    pytest happened to resolve its rootdir from: the same test, unchanged, gets
    one budget from `cd fused-memory && pytest tests` and a different one from
    `pytest fused-memory/tests` at the repo root. That is the same class of
    silent asymmetry ``test_pytest_workspace_collection.py::
    test_root_pyproject_mirrors_member_marker_deselections`` exists to prevent
    for marker deselection, and it is not hypothetical here: the measurement
    behind the current value found root-bound runs 1.3x to 1.7x SLOWER than
    per-member ones, so the two rootdirs are not interchangeable even in
    principle.

    NO LITERAL NUMBER IS ASSERTED, deliberately. A guard spelling
    `assert timeout == 540` would be a ninth copy of the configuration, and it
    would go red the next time the value is honestly re-measured — punishing
    exactly the behaviour this task exists to establish. The number belongs in
    the configs; this guard pins the RELATIONSHIP between them, and the only
    property of the value it is entitled to know is that it is a positive number
    of seconds.
    """
    timeouts = {
        name: ini_options['timeout']
        for name, ini_options in discovered_pytest_configs().items()
        if 'timeout' in ini_options
    }

    # Presence is test_every_pytest_config_declares_a_per_test_timeout's job, not
    # this one's — but a sweep that found NO values at all would agree vacuously.
    assert timeouts, (
        'no discovered pytest config declares a `timeout` at all (task 5442), so '
        'this agreement guard would pass by comparing nothing. Fix '
        'test_every_pytest_config_declares_a_per_test_timeout first.'
    )

    distinct = set(timeouts.values())
    assert len(distinct) == 1, (
        'the pytest configs in this repo disagree about the per-test '
        f'wall-clock cap (task 5442):\n{_grouped_by_value(timeouts)}\n\n'
        'A split value means the cap a test runs under depends on which '
        'directory pytest happened to resolve its rootdir from — the same test, '
        'unchanged, gets a different budget from `cd <member> && pytest tests` '
        'than from `pytest <member>/tests` at the repo root. Pick ONE value for '
        'all of them; its provenance belongs in '
        'plans/pytest-per-test-timeout-measurement-2026-09-17.md, not in a '
        'per-config judgement call.'
    )

    (value,) = distinct
    assert isinstance(value, int) and value > 0, (
        f'every pytest config declares timeout = {value!r} (task 5442), which is '
        'not a positive whole number of seconds. pytest-timeout reads this as a '
        'wall-clock budget; 0 disables the cap outright and a non-integer is not '
        'what any of the surrounding comments describe.'
    )
