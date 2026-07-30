"""Pytest driver for dark-factory's ported warm-lane bash tests.

Task 3073 (PRD ``plans/warm-lane-infra-repatriation-prd.md`` leaf α2, Phase 1).

Task 3072 (leaf α) relocated reify's project-agnostic warm-lane *scripts* into
``orchestrator/scripts/warm-lane/``.  This leaf ports the matching *bash tests*
into ``orchestrator/tests/warm-lane/`` so they exercise dark-factory's own
copies.  ``.sh`` files are not collected by pytest, so this module is the
driver: one parametrized item per ported test, so a failure is attributable to
a single file rather than to an opaque aggregate.

Two non-vacuity guards are load-bearing here, because PRD leaf **κ** deletes
reify's originals on the strength of this suite being green:

* ``PORTED_TESTS`` is an explicit manifest asserted set-equal to the on-disk
  glob.  A glob-only driver reports a green suite when it discovers zero files;
  the manifest turns a dropped, renamed or mis-globbed port into a hard failure.
* Required host tools FAIL rather than skip.  reify's originals hard-require
  ``flock``, ``git``, procfs, ``du``, ``dd`` and (for audit) ``python3`` with no
  guards at all, so a skip on a tool-less host would silently convert this
  leaf's whole deliverable into a no-op.

See ``orchestrator/tests/warm-lane/README.md`` for provenance, the enumerated
deltas from the reify sources, and the measured wall-clock per test.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Resolved from THIS FILE, never from the process CWD: the merge-verify harness
# runs pytest from the ``orchestrator/`` cwd while a plain
# ``pytest orchestrator/tests`` runs from the repo root, and this contract must
# hold identically under both.  Same idiom as test_warm_lane_scripts_shipped.py.
BASH_TEST_DIR = Path(__file__).resolve().parents[0] / 'warm-lane'
WARM_LANE_SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'warm-lane'
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Explicit manifest of the ported bash tests — the non-vacuity guard.  Grown
#: one entry per port; asserted set-equal to the on-disk glob below.
PORTED_TESTS: tuple[str, ...] = (
    'test_warm_lane_disk_guard.sh',
    'test_warm_lane_degenerate_ref.sh',
    'test_thin_warm_lane.sh',
    'test_warm_lane_gc.sh',
    'test_warm_lane_gc_sweep.sh',
    'test_warm_lane_audit.sh',
    'test_warm_lane_sizing_lifecycle.sh',
    'test_provision_warm_lane_fs.sh',
)

#: Every invocable script α relocated, mapped to the ported bash test that owns
#: its coverage.  The parenthesised notes record *additional* real exercise, so a
#: future reader can see which tests break if a script changes; the mapping
#: itself is one-test-per-script so the drift guard stays unambiguous.
#: ``lib_live_refs.sh`` / ``lib_portable.sh`` are deliberately absent — see
#: ``test_every_invocable_script_has_ported_coverage``.
SCRIPT_COVERAGE = {
    # also exercised for real by test_warm_lane_gc_sweep.sh blocks G4/G5/U/W
    'warm-lane-gc.sh': 'test_warm_lane_gc.sh',
    'warm-lane-gc-sweep.sh': 'test_warm_lane_gc_sweep.sh',
    # also test_warm_lane_sizing_lifecycle.sh
    'thin-warm-lane.sh': 'test_thin_warm_lane.sh',
    # also test_warm_lane_sizing_lifecycle.sh and test_warm_lane_gc_sweep.sh V1-V3
    'warm-lane-disk-guard.sh': 'test_warm_lane_disk_guard.sh',
    # also test_warm_lane_sizing_lifecycle.sh
    'warm-lane-audit.sh': 'test_warm_lane_audit.sh',
    'warm-lane-degenerate-ref-check.sh': 'test_warm_lane_degenerate_ref.sh',
    'provision-warm-lane-fs.sh': 'test_provision_warm_lane_fs.sh',
}

#: Kept strictly BELOW the ``timeout(360)`` marker so a hung bash test fails as
#: one clean pytest failure with its stdout/stderr captured, instead of
#: pytest-timeout's thread method ``os._exit()``ing the whole xdist worker
#: (``--max-worker-restart=0`` is set, so that death is unrecoverable).
SUBPROC_TIMEOUT = 300

#: Host tools reify's originals hard-require with no skip guards.
REQUIRED_HOST_TOOLS = ('bash', 'git', 'flock', 'du', 'dd', 'python3')

#: See ``test_warm_lane_scripts_shipped.py::_HOSTILE_ENV_KEYS`` for the full
#: rationale.  ``REIFY_WARM_LANE_MOUNT`` would override a lane root outright;
#: the ``GIT_*`` pair is the subtler hazard — pytest's ``--basetemp`` may legally
#: sit INSIDE a checkout, and this suite runs under git hooks and
#: ``git rebase --exec``, which export ``GIT_DIR``, under which a bare
#: ``git init`` in a test fixture fails outright.  Six of the ported tests build
#: synthetic git repos and ``test_warm_lane_gc.sh`` alone does 34
#: ``git worktree add`` calls, so stripping these is non-negotiable.
_HOSTILE_ENV_KEYS = ('REIFY_WARM_LANE_MOUNT', 'GIT_DIR', 'GIT_WORK_TREE')

# ``timeout(360)``: pytest-timeout ranks a marker above both the ini ``timeout =
# 60`` and the ``--timeout=300`` every configured invocation passes on the CLI,
# so a bare local ``pytest tests/warm-lane`` cannot silently get 60s.
# ``xdist_group``: reify classifies all of these as ``pool`` (serialised), and
# ``lib_live_refs.sh`` walks ``/proc`` for live consumers, where a concurrently
# running sibling's cwd could perturb a liveness probe.  ``--dist loadgroup`` is
# already in addopts, so a shared group co-locates them on one worker while the
# rest of the suite still runs in parallel.
pytestmark = [
    pytest.mark.warm_lane_bash,
    pytest.mark.timeout(360),
    pytest.mark.xdist_group('warm_lane_bash'),
]


def _sanitized_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """A copy of ``os.environ`` with ``_HOSTILE_ENV_KEYS`` removed."""
    env = {k: v for k, v in os.environ.items() if k not in _HOSTILE_ENV_KEYS}
    if extra:
        env.update(extra)
    return env


def _tail(text: str, limit: int = 4000) -> str:
    """The last ``limit`` characters, so a failure message stays readable."""
    return text if len(text) <= limit else '...(truncated)...\n' + text[-limit:]


def test_manifest_matches_disk() -> None:
    """``PORTED_TESTS`` must name exactly the ``test_*.sh`` files on disk.

    The non-vacuity guard.  ``test_helpers.sh`` is excluded: it is the sourced
    harness, not a runnable suite.
    """
    on_disk = {
        p.name for p in BASH_TEST_DIR.glob('test_*.sh')
        if p.name != 'test_helpers.sh'
    }
    assert set(PORTED_TESTS) == on_disk, (
        f'PORTED_TESTS and {BASH_TEST_DIR} disagree.\n'
        f'  manifest-only (dropped/renamed port?): {sorted(set(PORTED_TESTS) - on_disk)}\n'
        f'  disk-only (unregistered port?):        {sorted(on_disk - set(PORTED_TESTS))}'
    )


def test_every_invocable_script_has_ported_coverage() -> None:
    """Every invocable relocated script must be exercised by a ported bash test.

    The INV-5 replacement for reify's ``verify-pipeline-infra-tests.txt``
    drift map, which this leaf excised from ``test_warm_lane_gc_sweep.sh``
    (that file maps reify's paths and has no dark-factory counterpart).  This
    assertion guards the same invariant against the directory that actually
    exists here: it fails LOUDLY if a future leaf ships a new invocable script
    into ``orchestrator/scripts/warm-lane/`` without ported bash coverage, or
    removes one whose test is still listed.  PRD leaf κ needs that condition to
    be false before it deletes reify's originals.

    ``lib_live_refs.sh`` and ``lib_portable.sh`` are excluded from the key set by
    decision, not oversight: both are ``source``-only, neither has a ``--help``
    or any invocable entry point, and both are covered *transitively* — the
    gc/gc-sweep ``exit 2``-on-missing-sibling assertions (``test_warm_lane_gc.sh``
    A9) exercise ``lib_live_refs.sh``, and ``test_warm_lane_audit.sh`` exercises
    ``lib_portable.sh`` on every audit invocation.
    """
    invocable = {
        p.name for p in WARM_LANE_SCRIPT_DIR.glob('*.sh')
        if not p.name.startswith('lib_')
    }
    assert set(SCRIPT_COVERAGE) == invocable, (
        f'SCRIPT_COVERAGE and the invocable scripts in {WARM_LANE_SCRIPT_DIR} disagree.\n'
        f'  mapped but absent (removed script?):  {sorted(set(SCRIPT_COVERAGE) - invocable)}\n'
        f'  shipped but unmapped (no coverage?):  {sorted(invocable - set(SCRIPT_COVERAGE))}'
    )
    unported = {
        script: test for script, test in SCRIPT_COVERAGE.items()
        if test not in PORTED_TESTS
    }
    assert not unported, (
        f'SCRIPT_COVERAGE names bash tests that are not in PORTED_TESTS: {unported}'
    )


def test_required_host_tools_are_present() -> None:
    """Fail — never skip — when a tool the ported tests hard-require is absent.

    reify's originals invoke ``flock``, ``git``, ``du``, ``dd`` and ``python3``
    with no skip guards, and several probe procfs.  A skip here would hand PRD
    leaf κ a green suite that ran nothing.
    """
    missing = [tool for tool in REQUIRED_HOST_TOOLS if shutil.which(tool) is None]
    assert not missing, f'required host tools not on PATH: {missing}'
    assert Path('/proc/self/cwd').exists(), (
        'procfs is unavailable; the ported liveness probes in lib_live_refs.sh '
        'read /proc/<pid>/{exe,cwd,fd,maps} and cannot run here'
    )


def test_script_dir_resolution_ignores_the_env_override() -> None:
    """``lib_warm_lane_paths.sh`` must read NO environment — a behavioural pin.

    ``conftest.py``'s autouse ``_isolate_warm_lane_script_dir`` pins
    ``ORCH_WARM_LANE_SCRIPT_DIR`` at a guaranteed-absent sentinel for EVERY test
    in the orchestrator suite, and that value leaks into any subprocess this
    driver spawns.  If the bash harness honoured it, all ported tests would
    silently run against an empty directory — a vacuous green, the exact failure
    mode this leaf exists to prevent for leaf κ.  Setting the sentinel explicitly
    and asserting the resolver still yields the REAL directory makes "harmless by
    design" a fact rather than an assumption.
    """
    lib = BASH_TEST_DIR / 'lib_warm_lane_paths.sh'
    proc = subprocess.run(
        ['bash', '-c', f'source {lib}; printf %s "$WARM_LANE_SCRIPTS_DIR"'],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO_ROOT,
        env=_sanitized_env(
            {'ORCH_WARM_LANE_SCRIPT_DIR': '/nonexistent/df-warm-lane-scripts'},
        ),
    )
    assert proc.returncode == 0, f'resolver failed: {_tail(proc.stderr)}'
    assert proc.stdout == str(WARM_LANE_SCRIPT_DIR), (
        f'resolver honoured the environment: got {proc.stdout!r}, '
        f'want {str(WARM_LANE_SCRIPT_DIR)!r}'
    )


@pytest.mark.parametrize('name', PORTED_TESTS, ids=PORTED_TESTS)
def test_bash_suite_passes(name: str) -> None:
    """Each ported bash suite exits 0 (``test_summary``'s pass/fail protocol)."""
    proc = subprocess.run(
        ['bash', str(BASH_TEST_DIR / name)],
        capture_output=True,
        text=True,
        timeout=SUBPROC_TIMEOUT,
        cwd=REPO_ROOT,
        env=_sanitized_env(),
    )
    assert proc.returncode == 0, (
        f'{name} failed with exit {proc.returncode}\n'
        f'--- stdout ---\n{_tail(proc.stdout)}\n'
        f'--- stderr ---\n{_tail(proc.stderr)}'
    )
