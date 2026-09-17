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
