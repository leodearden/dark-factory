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
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

import pytest
import verify_command_invariants as vci

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
# It nests one level deeper, inside ``.pytest-tmp/``, because the fixture's
# ``finally`` does NOT run on SIGKILL, on pytest-timeout's ``os._exit()`` worker
# kill, or on an interrupted session — and REPO_ROOT is the machine-operated
# ``project_root`` checkout, where a stray untracked directory can ride a
# ``git add -- .`` into a commit. That is not hypothetical: .gitignore records a
# marker-probe leftover of exactly this shape riding a WIP commit onto a task
# branch and breaking a directory-wide ruff sweep (task 3581).
# ``.gitignore``'s ``.pytest-tmp/`` rule is this repo's existing,
# deliberately-unanchored home for pytest scratch inside a checkout, so a
# leftover here is already unstageable — and ``_ignored_scratch_root`` refuses
# to create anything if that stops being true rather than trusting this comment.
# MEASURED: a probe at ``<repo>/.pytest-tmp/<dir>/test_*.py`` resolves
# ``inipath=<repo>/pyproject.toml`` and ``rootpath=<repo>``, identical to one
# written directly at the repo root — so the extra level costs the invariant
# under test nothing.
PROBE_SCRATCH_ROOT = REPO_ROOT / '.pytest-tmp'

# Dot-prefixed so no OTHER pytest run can collect it while it exists: pytest's
# default ``norecursedirs`` skips ``.*``, and passing this probe's path
# explicitly is what still lets THIS run reach it. Redundant with the dot on
# ``.pytest-tmp`` itself, deliberately: the property has to keep holding wherever
# the scratch root is next pointed.
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


def _ignored_scratch_root() -> pathlib.Path:
    """``PROBE_SCRATCH_ROOT``, created — but only after git confirms it is ignored.

    Asked BEFORE the directory is created, so a path git would offer to stage is
    never materialised at all. ``git check-ignore`` answers for a path that does
    not exist yet, which is what makes that ordering available.
    """
    relative = PROBE_SCRATCH_ROOT.relative_to(REPO_ROOT)
    probed = subprocess.run(
        ['git', 'check-ignore', '-q', f'{relative}/probe'],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert probed.returncode == 0, (
        f'{relative}/ is no longer git-ignored (`git check-ignore` exit '
        f'{probed.returncode}), so this guard would drop scratch into the working '
        f'tree of a machine-operated checkout — where the fixture\'s `finally` '
        f'does NOT run on SIGKILL or on an interrupted session, and a leftover can '
        f'ride a `git add -- .` into a commit (task 5442; the same shape broke a '
        f'directory-wide ruff sweep in task 3581). Restore the `.pytest-tmp/` rule '
        f'in .gitignore, or point PROBE_SCRATCH_ROOT at another ignored path under '
        f'the repo root.\nstderr:\n{probed.stderr}'
    )
    PROBE_SCRATCH_ROOT.mkdir(exist_ok=True)
    return PROBE_SCRATCH_ROOT


@pytest.fixture
def root_bound_probe():
    """A throwaway probe test file under REPO_ROOT, removed afterwards.

    Inside the repo, not in ``tmp_path``: see ``PROBE_SCRATCH_ROOT`` for the
    measurement that rules ``tmp_path`` out, and for why it nests inside an
    already-ignored scratch root. The ``finally`` covers the runs that reach it;
    the ignored scratch root is what covers the runs that never do.
    """
    directory = pathlib.Path(
        tempfile.mkdtemp(dir=_ignored_scratch_root(), prefix=PROBE_DIR_PREFIX)
    )
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


# pytest-timeout's ENTIRE legal domain for this setting, read from the installed
# package rather than assumed: `_validate_method` is `if method not in ["signal",
# "thread"]: raise ValueError`, and the `--timeout-method` option declares the
# same two as its argparse `choices`.
#
# This is why the value-agnostic principle this file defends does NOT reach
# here. That principle protects `timeout` — a MEASURED number that must stay
# free to move without a guard going red. `timeout_method` is not measured and
# cannot move: it is a two-element enumeration, so naming both members costs the
# next re-measurement nothing.
SUPPORTED_TIMEOUT_METHODS = frozenset({'signal', 'thread'})


def test_every_pytest_config_declares_a_supported_timeout_method() -> None:
    """A typo'd method is caught HERE, once, not one wedged suite at a time.

    Presence is not enough. `timeout_method = "sginal"` satisfies
    ``test_every_pytest_config_declares_a_per_test_timeout`` above and the
    behavioural probe's non-blank check, while pytest-timeout raises
    ``ValueError: Invalid method sginal from config file`` at session start — so
    EVERY run whose rootdir resolves to that config dies before a single test
    executes. Loud, but discovered one suite at a time by whoever next ran it,
    and for the ROOT config that is every root-bound invocation in the repo.

    Which method each config declares is deliberately NOT asserted: `signal` and
    `thread` are a live trade with measured evidence on both sides
    (``fused-memory/pyproject.toml`` records signal proving unreliable under
    xdist workers; ``escalation/pyproject.toml`` records signal measured green
    under the identical flags), and pinning a winner here would settle that trade
    by guard rather than by measurement. This asserts only that the declared
    value is one pytest-timeout will accept.
    """
    declared = {
        name: ini_options['timeout_method']
        for name, ini_options in sorted(discovered_pytest_configs().items())
        if 'timeout_method' in ini_options
    }

    # Presence is the sweep above's job, not this one's — but a mapping that came
    # back empty would satisfy the membership check below by comparing nothing.
    assert declared, (
        'no discovered pytest config declares a `timeout_method` at all (task '
        '5442), so this guard would pass by checking nothing. Fix '
        'test_every_pytest_config_declares_a_per_test_timeout first.'
    )

    unsupported = {
        name: method
        for name, method in declared.items()
        if method not in SUPPORTED_TIMEOUT_METHODS
    }

    assert not unsupported, (
        'these pytest configs declare a `timeout_method` pytest-timeout does not '
        f'accept (task 5442); the legal domain is exactly '
        f'{sorted(SUPPORTED_TIMEOUT_METHODS)}:\n'
        + '\n'.join(
            f'  {name}/pyproject.toml — timeout_method = {method!r}'
            for name, method in unsupported.items()
        )
        + '\n\npytest-timeout validates this at session start and raises '
        '`ValueError: Invalid method <value> from config file`, so every run '
        'whose rootdir resolves to one of the files above dies before collecting '
        'a single test. Fix the spelling; which of the two to pick for a given '
        'config follows that config\'s execution shape, and the trade between '
        'them is recorded in fused-memory/pyproject.toml.'
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


CONTRIBUTING_PATH = REPO_ROOT / 'CONTRIBUTING.md'

# A marker pair of its own rather than "the span matching a number", because
# CONTRIBUTING.md carries other, deliberately-GENERIC timeout spans — the
# pre-commit hook's 300000ms budget in §4 and the --no-verify bullet in §8 —
# which must never be pinned to this config. Any first-matching-span extractor
# would silently re-target onto one of them on a doc reorder.
#
# The `*-mirror:begin/end` spelling follows the idiom the two existing
# CONTRIBUTING mirrors established (`lint-command-mirror`,
# `type-check-command-mirror`) rather than inventing a second convention; the
# naming slot was free.
MIRROR_BEGIN = 'pytest-timeout-mirror:begin'
MIRROR_END = 'pytest-timeout-mirror:end'

# An inline-code span of PURE DIGITS. Deliberately not "any number in the span":
# the prose around it names other figures (`--timeout=300`, a section reference),
# and requiring the backticks plus a digits-only body is what keeps exactly one
# match inside the markers — which is the condition `marked_span` asserts.
_MARKED_SECONDS = re.compile(r'`(\d+)`')


def _documented_per_test_timeout(markdown_text: str) -> int:
    """The per-test timeout documented inside the mirror markers, as an int.

    The four marker assertions — exactly one begin, exactly one end, exactly one
    match in the slice between them (which is also what catches INVERTED
    markers), and a non-blank match — are ``verify_command_invariants.marked_span``'s,
    IMPORTED rather than copied beside it: that module's docstring forbids the
    next copy in as many words, and every failure it raises is a loud
    ``AssertionError`` naming the marker literal and CONTRIBUTING.md rather than
    a quiet ``''``. That is the vacuity hazard and the whole point — an extractor
    that silently yields nothing turns the drift assertion green while pinning
    nothing at all.
    """
    return int(
        vci.marked_span(
            markdown_text,
            begin=MIRROR_BEGIN,
            end=MIRROR_END,
            pattern=_MARKED_SECONDS,
            what='inline-code span of digits',
            source='CONTRIBUTING.md',
            label=(
                "the Tests bullet's documented per-test timeout in CONTRIBUTING.md "
                'that mirrors [tool.pytest.ini_options].timeout in every '
                'pyproject.toml'
            ),
            task='5442',
        )
    )


# Extractor fixtures are hand-written markdown, never the real CONTRIBUTING.md,
# so they stay stable under any future edit to that file's content. They spell
# the marker literals out in full rather than interpolating the constants above:
# a rename must not be able to silently keep a broken parser agreeing with its
# own fixtures. Same convention, and the same reason, as
# test_contributing_lint_command_drift.py's.

# (a) Happy path, modelled on the real block: a fenced ```bash example ABOVE the
# marker carrying its own number, and — the hazard that decides the pattern —
# an inline-code span inside the marked prose that is NOT pure digits.
_HAPPY_DOC = """\
- **Tests** run per-package with `pytest`, e.g.:
  ```bash
  cd orchestrator && uv run pytest tests/ --timeout=300
  ```
<!-- pytest-timeout-mirror:begin
     Mirrors [tool.pytest.ini_options].timeout in every pyproject.toml. Pinned by
     tests/scripts/test_pytest_per_test_timeout_policy.py. -->
  Every pytest config caps a single test at `480` seconds of wall clock; opt a
  slow test up with `@pytest.mark.timeout(N)`.
<!-- pytest-timeout-mirror:end -->
- **Lint**: `uv run ruff check alpha beta`
"""

_HAPPY_SECONDS = 480

# (b) No marker at all — deleted, or the section renamed. The doc still CONTAINS
# a plausible-looking number: the extractor must not fall back to "find
# something that looks right".
_NO_MARKER_DOC = """\
- **Tests** run per-package with `pytest --timeout=300`.

Give a slow hook a timeout of at least `300000` ms.
"""

# (c) Two marker blocks — e.g. a section duplicated in a bad merge. Picking the
# first silently pins one mirror and lets the other rot unwatched.
_DUPLICATE_MARKER_DOC = """\
<!-- pytest-timeout-mirror:begin -->
Every pytest config caps a single test at `480` seconds.
<!-- pytest-timeout-mirror:end -->

## Some later section

<!-- pytest-timeout-mirror:begin -->
Every pytest config caps a single test at `300` seconds.
<!-- pytest-timeout-mirror:end -->
"""

# (d) Inverted markers. They yield an EMPTY slice, so the match assertion catches
# them — but only if it is really made, which is what this fixture pins.
_INVERTED_MARKER_DOC = """\
<!-- pytest-timeout-mirror:end -->
Every pytest config caps a single test at `480` seconds.
<!-- pytest-timeout-mirror:begin -->
"""

# (e) Decoy immunity. §4's pre-commit hook budget and §8's --no-verify bullet are
# GENERIC advice about a different gate, and CONTRIBUTING.md really carries them
# — before AND after the marked span, so neither a "first span" nor a "last span"
# heuristic passes by accident. Pinning either to this config would be wrong
# twice over: it would fail immediately, and "fixing" it would destroy correct,
# audience-appropriate advice.
_DECOY_DOC = """\
pyright can exceed two minutes — give those a timeout of at least `300000` ms.

<!-- pytest-timeout-mirror:begin
     Mirrors [tool.pytest.ini_options].timeout in every pyproject.toml. -->
Every pytest config caps a single test at `480` seconds of wall clock.
<!-- pytest-timeout-mirror:end -->

Don't skip the hook with `--no-verify`; if a check is genuinely too slow, raise
the timeout to `300000` instead.
"""


def test_documented_per_test_timeout_extracts_the_marked_span() -> None:
    """(a) Only the marked span's number is returned, backticks stripped.

    Also the specific failure of an extractor keyed on "a number in the span":
    it would return the `--timeout=300` of the fenced example above the marker,
    or a fragment of the begin comment's own prose — each a plausible-looking
    figure, so the mistake would not announce itself.
    """
    assert _documented_per_test_timeout(_HAPPY_DOC) == _HAPPY_SECONDS


@pytest.mark.parametrize(
    ('markdown_text', 'case'),
    [
        (_NO_MARKER_DOC, 'missing'),
        (_DUPLICATE_MARKER_DOC, 'duplicated'),
        (_INVERTED_MARKER_DOC, 'inverted'),
    ],
)
def test_documented_per_test_timeout_fails_loudly_on_a_broken_marker(
    markdown_text: str, case: str
) -> None:
    """(b, c, d) A broken marker RAISES — never '' or None.

    Missing is the vacuity hazard: an extractor that silently returns nothing
    turns every downstream assertion green while pinning nothing at all.
    Duplicated is the same failure one level down — silently taking the first
    leaves the second mirror unpinned and free to drift. Inverted yields an empty
    slice, which would otherwise extract nothing just as quietly. Every message
    must tell a human what to restore and where.
    """
    with pytest.raises(AssertionError) as excinfo:
        _documented_per_test_timeout(markdown_text)

    message = str(excinfo.value)
    assert MIRROR_BEGIN in message, case
    assert 'CONTRIBUTING.md' in message, case


def test_documented_per_test_timeout_is_immune_to_the_generic_hook_decoy() -> None:
    """(e) The pre-commit hook's millisecond budget is never extracted.

    §4's "give those a timeout of at least 300000ms" and §8's --no-verify bullet
    describe `hooks/pre-commit`, a different gate on a different clock. Pinning
    either to the pytest configs would be wrong twice over: it would fail
    immediately, and "fixing" it would destroy correct advice.
    """
    assert _documented_per_test_timeout(_DECOY_DOC) == _HAPPY_SECONDS


def test_contributing_mirrors_the_configured_per_test_timeout() -> None:
    """CONTRIBUTING.md's documented cap must equal the one the configs declare.

    Reads BOTH sides live from the committed artifacts. This is a value MIRROR,
    not a prose test: it asserts that one extracted NUMBER equals a config value,
    and says nothing about docstrings, comment wording, or whether any prose
    mentions a topic. It must stay green if every sentence around the marker is
    reworded.

    The harm it prevents is the one both sibling CONTRIBUTING mirrors were filed
    for, measured rather than predicted on those: prose pointing at prose does
    not hold. A contributor who reads a stale number here plans a test's budget
    against a cap that no longer exists.

    MEASURED RED at base 832d6faf16: CONTRIBUTING.md carried no such marker pair
    and documented no per-test timeout at all, so `marked_span` raised its
    missing-marker AssertionError.
    """
    timeouts = set(
        ini_options['timeout']
        for ini_options in discovered_pytest_configs().values()
        if 'timeout' in ini_options
    )
    documented = _documented_per_test_timeout(
        CONTRIBUTING_PATH.read_text(encoding='utf-8')
    )

    # (a) NON-VACUITY, both sides. Neither an empty config sweep nor a
    # non-positive documented figure may let this invariant pass by comparing
    # nothing. (The documented side cannot be absent — `marked_span` raises
    # rather than returning '' — so what is left to check is that it is a
    # sensible number.)
    assert timeouts, (
        'no discovered pytest config declares a `timeout` at all (task 5442), so '
        'this mirror would pass vacuously. Fix '
        'test_every_pytest_config_declares_a_per_test_timeout first.'
    )
    assert documented > 0, (
        f'CONTRIBUTING.md documents a per-test timeout of {documented} inside the '
        f'{MIRROR_BEGIN!r} marker (task 5442), which is not a positive number of '
        'seconds'
    )

    # (b) SEMANTIC — the configs must agree with each other before the doc can
    # mirror "the" value at all. Its own guard owns the diagnosis; this reports
    # only that the comparison cannot be made.
    assert len(timeouts) == 1, (
        f'the pytest configs declare {len(timeouts)} different timeouts '
        f'({sorted(timeouts)}), so there is no single value for CONTRIBUTING.md '
        'to mirror (task 5442). '
        'test_every_pytest_config_declares_the_same_timeout names which files '
        'disagree.'
    )

    # (c) EXACT.
    (configured,) = timeouts
    assert documented == configured, (
        f"CONTRIBUTING.md's Tests bullet documents a per-test timeout of "
        f'{documented}s, but every pytest config declares {configured}s (task '
        f'5442). A contributor reading the doc sizes a @pytest.mark.timeout(N) '
        f'against a cap that does not exist — and since that marker OVERRIDES the '
        f'default in both directions rather than raising a floor under it, an N '
        f'chosen from a stale number silently TIGHTENS the run that gates their '
        f'merge. Update the number inside the {MIRROR_BEGIN!r} marker in '
        f'CONTRIBUTING.md to match the configs; the value\'s provenance is '
        f'plans/pytest-per-test-timeout-measurement-2026-09-17.md.'
    )
