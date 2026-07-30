"""Pytest driver for dark-factory's ported warm-lane bash tests.

Task 3073 (PRD ``plans/warm-lane-infra-repatriation-prd.md`` leaf α2, Phase 1).

Task 3072 (leaf α) relocated reify's project-agnostic warm-lane *scripts* into
``orchestrator/scripts/warm-lane/``.  This leaf ports the matching *bash tests*
into ``orchestrator/tests/warm-lane/`` so they exercise dark-factory's own
copies.  ``.sh`` files are not collected by pytest, so this module is the
driver: one parametrized item per ported test, so a failure is attributable to
a single file rather than to an opaque aggregate.

Three non-vacuity guards are load-bearing here, because PRD leaf **κ** deletes
reify's originals on the strength of this suite being green:

* ``PORTED_TESTS`` is an explicit manifest asserted set-equal to the on-disk
  glob.  A glob-only driver reports a green suite when it discovers zero files;
  the manifest turns a dropped, renamed or mis-globbed port into a hard failure.
* Required host tools FAIL rather than skip.  reify's originals hard-require
  ``flock``, ``git``, procfs, ``du``, ``dd`` and (for audit) ``python3`` with no
  guards at all, so a skip on a tool-less host would silently convert this
  leaf's whole deliverable into a no-op.
* Greenness is ``exit 0`` **and** a ``Results: N passed, 0 failed`` line with
  ``N`` at or above a per-suite floor — the same strictly-stronger predicate
  reify's own runner (``tests/infra/run_all_ambient_isolation_lib.sh``) applies.
  Exit-status-only would be the weak link in κ's chain: a block that skips or
  short-circuits still exits 0 while its share of the 837 asserts never runs.
  See ``ASSERT_FLOORS``.

See ``orchestrator/tests/warm-lane/README.md`` for provenance, the enumerated
deltas from the reify sources, and the measured wall-clock per test.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
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

#: Minimum ``Results: N passed`` count each ported suite must report.  The floor
#: is the count guaranteed on ANY host that satisfies ``REQUIRED_HOST_TOOLS``,
#: i.e. the measured count minus the asserts in blocks whose skip guards are
#: enumerated below.  Exit status alone cannot see a block that skipped: reify's
#: own runner therefore defines GREEN as exit 0 AND the protocol line, and leaf κ
#: deletes reify's originals on the strength of this suite, so the weaker
#: predicate is not good enough here.
#:
#: Only TWO deductions from the measured counts in
#: ``orchestrator/tests/warm-lane/README.md`` are legitimate, both from skip
#: guards this leaf was required to preserve verbatim:
#:
#: * ``test_provision_warm_lane_fs.sh`` Block I (5 asserts) — real-geometry
#:   ``xfs_info``/``xfs_db`` proof, guarded on ``mkfs.xfs``/``xfs_info``/
#:   ``xfs_db``.  Measured: 111 with xfsprogs, 106 without.  xfsprogs is
#:   deliberately NOT added to ``REQUIRED_HOST_TOOLS`` — that guard exists so the
#:   bash file stays runnable by a developer on any host, and promoting it to a
#:   hard requirement would red the whole orchestrator suite on a host that only
#:   lacks an optional package.
#: * ``test_warm_lane_audit.sh`` L9 (3 asserts) — unreadable-record degradation,
#:   guarded on ``id -u != 0`` because mode 000 is not a barrier for root.
#:
#: ``test_thin_warm_lane.sh``'s Block C df-delta assert is not a deduction: it is
#: gated on ``REIFY_WARM_LANE_MOUNT``, which ``_sanitized_env`` always strips, so
#: 45 is both its floor and its measured count under this driver.
#:
#: Any OTHER shortfall — a block short-circuited by a future edit — fails loudly
#: here instead of passing as a green item that ran less than it claims.  Leaf γ
#: (dark-factory task 3075) was the anticipated case and has now landed: it
#: rewrote ``warm-lane-gc.sh``'s Pass-1 reclaimability predicate around the
#: durable lane record and added 28 asserts (Block S + Block A's A10), so the
#: floor moved 170 → 198.  It moved because the count was RE-MEASURED, not to
#: make a red run green.
ASSERT_FLOORS = {
    'test_warm_lane_disk_guard.sh': 62,
    'test_warm_lane_degenerate_ref.sh': 70,
    'test_thin_warm_lane.sh': 45,
    'test_warm_lane_gc.sh': 198,  # 170 + 28 (Block S 23 + A10 3 + S9b/S10b)
    'test_warm_lane_gc_sweep.sh': 86,
    'test_warm_lane_audit.sh': 225,  # 228 measured − 3 (L9, root-guarded)
    'test_warm_lane_sizing_lifecycle.sh': 65,
    'test_provision_warm_lane_fs.sh': 106,  # 111 measured − 5 (Block I, xfsprogs)
}

#: ``test_helpers.sh::test_summary``'s pass/fail protocol line, verbatim.
_RESULTS_RE = re.compile(r'^Results: (\d+) passed, (\d+) failed$', re.MULTILINE)

#: Kept strictly BELOW the ``timeout(360)`` marker so a hung bash test fails as
#: one clean pytest failure with its stdout/stderr captured, instead of
#: pytest-timeout's thread method ``os._exit()``ing the whole xdist worker
#: (``--max-worker-restart=0`` is set, so that death is unrecoverable).
SUBPROC_TIMEOUT = 300

#: Host tools reify's originals hard-require with no skip guards.
REQUIRED_HOST_TOOLS = ('bash', 'git', 'flock', 'du', 'dd', 'python3')

#: See ``test_warm_lane_scripts_shipped.py::_HOSTILE_ENV_KEYS`` for the full
#: rationale.  The ``GIT_*`` pair is the subtler hazard — pytest's ``--basetemp``
#: may legally sit INSIDE a checkout, and this suite runs under git hooks and
#: ``git rebase --exec``, which export ``GIT_DIR``, under which a bare
#: ``git init`` in a test fixture fails outright.  Six of the ported tests build
#: synthetic git repos and ``test_warm_lane_gc.sh`` alone does 34
#: ``git worktree add`` calls, so stripping these is non-negotiable.
_HOSTILE_ENV_KEYS = ('GIT_DIR', 'GIT_WORK_TREE')

#: A PREFIX rule, not a name list.  The relocated scripts read ~30 ``REIFY_*``
#: override/seam variables (``REIFY_WARM_LANE_MOUNT``,
#: ``REIFY_WARM_LANE_AUDIT_STATE_DIR``, ``REIFY_WARM_LANE_AUDIT_SAFETY``,
#: ``REIFY_WARM_LANE_GC_PROTECT_GLOB``, ``REIFY_WARM_LANE_GC_MOUNT``,
#: ``REIFY_WARM_LANE_SUDO``, ``REIFY_WARM_LANE_ALLOW_MKFS``, …), and the reasoning
#: that made ``REIFY_WARM_LANE_MOUNT`` hostile applies unchanged to every sibling:
#: this host also develops reify, so an operator debugging warm lanes there may
#: have any of them exported.  Enumerating names would leave the set to drift as
#: leaves β/γ/δ/ε add seams; the prefix cannot.  The strip is safe because every
#: ported suite exports the ``REIFY_*`` vars it needs per invocation — nothing is
#: read from the ambient environment.
_HOSTILE_ENV_PREFIXES = ('REIFY_',)

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
    """``os.environ`` minus the hostile keys and every ``REIFY_*`` key.

    ``extra`` is applied AFTER the strip, so a caller can still inject a sentinel
    the rule would otherwise remove.
    """
    env = {
        k: v for k, v in os.environ.items()
        if k not in _HOSTILE_ENV_KEYS and not k.startswith(_HOSTILE_ENV_PREFIXES)
    }
    if extra:
        env.update(extra)
    return env


def _tail(text: str, limit: int = 4000) -> str:
    """The last ``limit`` characters, so a failure message stays readable."""
    return text if len(text) <= limit else '...(truncated)...\n' + text[-limit:]


def test_manifest_matches_disk() -> None:
    """``PORTED_TESTS`` must name exactly the ``test_*.sh`` files on disk.

    The non-vacuity guard, plus its companion: ``ASSERT_FLOORS`` must be keyed by
    exactly the same set.  ``test_helpers.sh`` is excluded: it is the sourced
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
    # Every suite must carry an assert floor, and a removed port must not leave a
    # stale one behind: without this, a dropped floor is a KeyError at run time
    # while a stale floor is invisible.
    assert set(ASSERT_FLOORS) == set(PORTED_TESTS), (
        f'ASSERT_FLOORS and PORTED_TESTS disagree.\n'
        f'  floor without a port:  {sorted(set(ASSERT_FLOORS) - set(PORTED_TESTS))}\n'
        f'  port without a floor:  {sorted(set(PORTED_TESTS) - set(ASSERT_FLOORS))}'
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


def _run_bash_suite(name: str) -> tuple[int, str, str]:
    """Run one ported suite, reaping its whole process group on timeout.

    ``start_new_session=True`` plus an explicit ``killpg`` is load-bearing, not
    tidiness.  A plain ``subprocess.run(timeout=...)`` SIGKILLs only the direct
    ``bash``, so its ``trap cleanup EXIT`` never runs — and ``test_warm_lane_gc.sh``
    backgrounds several ``( flock -x 9 && ... sleep 300 ) &`` and
    ``( cd <lane>/target && exec sleep 300 ) &`` liveness helpers.  Those would
    survive the kill, keep holding lane flocks and keep lane directories busy for
    up to five minutes, and leave the suite's lane trees behind on the verify
    host — turning one timeout into a cascade of unrelated failures.

    A timeout is also converted into a normal ``pytest.fail`` with the captured
    output, rather than propagating ``TimeoutExpired`` (whose stdout/stderr hang
    off the exception object), so the hang path produces the same
    failure-with-captured-tail shape the module docstring and README promise.
    """
    with subprocess.Popen(
        ['bash', str(BASH_TEST_DIR / name)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
        env=_sanitized_env(),
        start_new_session=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=SUBPROC_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                proc.kill()
            # ``Popen.communicate``'s ``TimeoutExpired`` carries no output (only
            # ``subprocess.run`` repopulates it), so the resumed call is what
            # recovers the captured tail.  Bounded: a pipe held open by a
            # survivor must not turn this into the >360s pytest-timeout worker
            # kill that SUBPROC_TIMEOUT exists to stay clear of.
            try:
                stdout, stderr = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover
                stdout, stderr = '', '<unavailable: pipes still held open>'
            pytest.fail(
                f'{name} did not finish within {SUBPROC_TIMEOUT}s; its process '
                f'group was killed.\n'
                f'--- stdout ---\n{_tail(stdout)}\n'
                f'--- stderr ---\n{_tail(stderr)}'
            )
        return proc.returncode, stdout, stderr


@pytest.mark.parametrize('name', PORTED_TESTS, ids=PORTED_TESTS)
def test_bash_suite_passes(name: str) -> None:
    """Each ported bash suite must be green by reify's own greenness predicate.

    ``exit 0`` AND a ``Results: N passed, 0 failed`` line AND ``N`` at or above
    ``ASSERT_FLOORS[name]``.  Exit status alone cannot distinguish "all 837
    asserts ran and passed" from "a block skipped and the rest passed", and leaf
    κ deletes reify's originals on the strength of this item being green.
    """
    returncode, stdout, stderr = _run_bash_suite(name)
    context = (
        f'--- stdout ---\n{_tail(stdout)}\n'
        f'--- stderr ---\n{_tail(stderr)}'
    )
    assert returncode == 0, f'{name} failed with exit {returncode}\n{context}'

    # Last match, mirroring the ``| tail -1`` reify's runner applies: each suite
    # calls test_summary exactly once today, but a future sub-harness that
    # printed its own line must not shadow the authoritative total.
    matches = _RESULTS_RE.findall(stdout)
    assert matches, (
        f'{name} exited 0 but printed no "Results: N passed, M failed" line. '
        f'That line is test_summary\'s entire pass/fail protocol, so its absence '
        f'means the suite never reached test_summary — a green item that proved '
        f'nothing.\n{context}'
    )
    passed, failed = int(matches[-1][0]), int(matches[-1][1])
    assert failed == 0, (
        f'{name} reported {failed} failed assert(s) yet exited {returncode}; '
        f'test_summary\'s exit-1-on-failure contract is broken.\n{context}'
    )
    floor = ASSERT_FLOORS[name]
    assert passed >= floor, (
        f'{name} ran only {passed} asserts, below its floor of {floor}. A block '
        f'skipped or short-circuited: an item that is green on exit status while '
        f'part of its coverage never ran. Either restore the coverage, or lower '
        f'the floor in ASSERT_FLOORS with the new skip guard recorded there.\n'
        f'{context}'
    )
