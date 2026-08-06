"""Suite-wide pytest isolation: git can never escape the basetemp into a repo.

Incident esc-3072-3.  Git repository discovery walks UP the directory tree, so
a git command run with ``cwd=<some tmp dir>`` does not operate on the directory
the caller named — it operates on *whatever repo encloses that directory*.  When
pytest's basetemp lives inside a live task worktree
(``.worktrees/<task>/.pytest-tmp/``), that enclosing repo is production state:
three blobs were written into a real task's object store and ``foo.py`` was
staged at stages 1/2/3, leaving ``UU foo.py`` in its index.

THE ONE NON-OBVIOUS FACT — the ceiling must equal the basetemp ITSELF:

    Git stops the upward walk only when the walk would ascend INTO or above a
    ``GIT_CEILING_DIRECTORIES`` entry.  Everything strictly BELOW an entry is
    still inspected.  So a ceiling at an *ancestor* of the basetemp — the
    tempting ``/tmp``, or ``tempfile.gettempdir()`` — is entirely inert against
    this incident: the walk from ``/tmp/…/.pytest-tmp/test_x0/sub`` still finds
    the repo sitting below ``/tmp``.  Verified against real git before this
    module was written.  Anyone "simplifying" this to a value computable in
    ``pytest_configure`` (where basetemp is not yet derivable without the
    private ``config._tmp_path_factory``) silently disarms the whole defence.

That precision is also what makes a suite-wide ceiling SAFE: a repo created
under the basetemp — every legitimate ``tmp_path`` repo and linked worktree —
sits below the ceiling entry and keeps resolving normally.

Complementary per-call layers from task 3182, which remain in force and are
strictly tighter than this one (``cwd.parent`` rather than the basetemp):

* ``_orch_helpers.assert_isolated_git_repo`` — a pure-filesystem pre-flight
  that runs BEFORE any subprocess, so a rejected call writes nothing anywhere.
  A ceiling cannot give that guarantee: it makes git fail mid-sequence, after
  an earlier ``git hash-object -w`` has already written its blobs.
* ``_orch_helpers.git_env_with_ceiling`` — the same ceiling mechanism applied
  per call, on a private env copy.

SECOND DEFENCE — a test run can never falsify a REAL deploy clock.

Task 3797.  ``scripts/restart-all-orchestrators.sh`` resolves its ``CLOCK_FILE``
from ``$ORCH_FLEET_DEPLOY_CLOCK``, defaulting to
``$REPO_DIR/data/orchestrator/last_redeploy_orchestrator.json`` — the live
checkout the script sits in — and stamps it on its verified-fresh exit-0 path.
``scripts/tests/test_restart_all_orchestrators.py`` drove that path against a
fake ``systemctl`` without setting the env var, so an ordinary green test run
wrote a REAL "the fleet just redeployed" stamp.  Nothing distinguishes it from
a genuine one: ``scripts/orchestrator-watchdog.py`` reads the file and SKIPS
its staleness pass for ``ORCH_RESTART_MIN_INTERVAL_SECS`` (8h default), so the
fleet-staleness backstop was silently disarmed for the rest of the day.

Same structure as the ceiling above — two pure helpers
(``deploy_clock_snapshot`` / ``deploy_clock_violation_reason``) plus a thin
session-scoped autouse fixture wiring them — and the same reason for living
here: the defence must be suite-wide and impossible to opt out of, because the
defect class is "a spawner that forgets the env var" and the next one has not
been written yet.

Import constraint: STDLIB + PYTEST ONLY.  Every subproject conftest imports this
module, so it must import cleanly inside every member venv — escalation's lacks
aiosqlite and stubs ``shared`` in ``sys.modules``, so nothing under ``shared/src``
may be depended on here.  That is why the protected clock paths below are
LITERALS rather than an import of
``orchestrator.service_restart.FLEET_DEPLOY_CLOCK_RELPATH``, exactly as the
stdlib watchdog and the bash script duplicate it; all four mirrors are pinned
together by
``tests/scripts/test_orchestrator_watchdog.py::test_fleet_deploy_clock_path_matches_across_tiers``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_CEILING_ENV = 'GIT_CEILING_DIRECTORIES'

# This repo's own definition of "this directory is a live worktree root",
# taken verbatim from .gitignore:15-17 rather than invented in parallel here.
_WORKTREE_ROOT_COMPONENTS = frozenset({
    '.worktrees',
    '.worktrees-orphaned',
    '.eval-worktrees',
})


def git_ceiling_value(basetemp: str | os.PathLike[str], existing: str | None = None) -> str:
    """Build the ``GIT_CEILING_DIRECTORIES`` value that contains *basetemp*.

    *basetemp* is made absolute and symlink-resolved: git IGNORES non-absolute
    entries outright, and compares entries against the RESOLVED cwd, so an
    unresolved or relative entry is not a weaker ceiling but no ceiling at all.

    Any *existing* value is PRESERVED and the basetemp entry appended after it.
    Git treats the variable as a colon-separated list where any single entry can
    stop the walk, so appending is strictly additive containment — overwriting
    would silently discard an operator- or CI-set ceiling.

    The append is idempotent: conftests nest (a run whose rootdir is the repo
    root loads both the root conftest and a subproject conftest), so re-entry
    must not accumulate duplicate entries.
    """
    entry = str(Path(basetemp).resolve())
    if not existing:
        return entry
    if entry in existing.split(':'):
        return existing
    return f'{existing}:{entry}'


def basetemp_rejection_reason(path: str | os.PathLike[str]) -> str | None:
    """Explain why *path* is an unsafe pytest basetemp, or ``None`` if it is fine.

    Deliberately NARROW: it refuses only a basetemp sitting inside one of this
    repo's own gitignored worktree roots.  The tempting general rule — "reject
    any basetemp with an enclosing git repo" — would hard-fail every developer
    and CI setup that legitimately points ``--basetemp`` inside a checkout,
    turning a safety net into an outage.

    Matching is on resolved path COMPONENTS, never substrings: a directory
    merely *containing* the name (``my.worktrees-backup``) is a different,
    perfectly safe directory.  Resolution is what catches the shape that
    actually caused the incident — ``--basetemp=.pytest-tmp`` run from inside
    the worktree, where the flag value never mentions ``.worktrees`` at all.
    """
    resolved = Path(path).resolve()
    offenders = _WORKTREE_ROOT_COMPONENTS.intersection(resolved.parts)
    if not offenders:
        return None
    return (
        f'unsafe --basetemp: {resolved} is inside a live task worktree '
        f'({"/".join(sorted(offenders))}).\n'
        'Git repository discovery walks UP the directory tree, so every '
        'git command a test runs under that basetemp resolves against the '
        "enclosing worktree's repo and mutates production state "
        '(incident esc-3072-3: blobs written and a file staged at stages '
        '1/2/3 in a live task). The untracked tree it creates can also be '
        'swept into a commit by `git add -A` or trip a clean-worktree gate.\n'
        'Fix: pass --basetemp somewhere OUTSIDE the worktree, or omit it '
        'entirely and let pytest default to /tmp/pytest-of-<user>/.'
    )


def reject_unsafe_basetemp(config: object) -> None:
    """Fail collection loudly when ``--basetemp`` points inside a worktree.

    Reads only the PUBLIC ``config.option.basetemp`` CLI option, which is
    populated by the time ``pytest_configure`` runs.  It deliberately does not
    consult ``config._tmp_path_factory``: that attribute is private and its
    availability during a *conftest* ``pytest_configure`` depends on plugin
    hook ordering.

    A no-op when no ``--basetemp`` was passed — the verify lane's case, which
    runs bare ``uv run pytest tests/`` and lands in ``/tmp/pytest-of-<user>/``.

    ``pytest.UsageError`` is the right register here: pytest renders it as a
    clean ``ERROR: ...`` and exits without a traceback, which is what an
    operator/agent misconfiguration deserves.
    """
    basetemp = getattr(getattr(config, 'option', None), 'basetemp', None)
    if not basetemp:
        return
    reason = basetemp_rejection_reason(basetemp)
    if reason is not None:
        raise pytest.UsageError(reason)


@pytest.fixture(scope='session', autouse=True)
def _df_git_ceiling_at_basetemp(tmp_path_factory: pytest.TempPathFactory):
    """Contain git's upward walk inside this run's pytest basetemp.

    SESSION scope is deliberate, on two counts.  Cost: the repo already
    documents this concern for ``_ABSENT_WARM_LANE_SCRIPT_DIR`` in
    ``orchestrator/tests/conftest.py`` — a per-test fixture runs "thousands of
    times, times every xdist worker".  Coverage: a function-scoped autouse
    fixture would miss git run from module- and session-scoped fixtures, which
    is exactly where expensive repo setup tends to live.

    Under xdist each worker process gets its own ``popen-gwN`` basetemp and
    sets its own ceiling in its own environment; no coordination is needed.

    Restores the previous value EXACTLY on teardown, including restoring
    absence by deleting the key rather than setting an empty string — git reads
    an empty entry as the "subsequent entries are not symlinks" marker, not as
    "unset".
    """
    saved = os.environ.get(_CEILING_ENV)
    os.environ[_CEILING_ENV] = git_ceiling_value(
        tmp_path_factory.getbasetemp(), existing=saved,
    )
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop(_CEILING_ENV, None)
        else:
            os.environ[_CEILING_ENV] = saved


# ---------------------------------------------------------------------------
# Deploy-clock isolation (task 3797)
# ---------------------------------------------------------------------------

# Both entries are MIN-INTERVAL deploy clocks: scripts/orchestrator-watchdog.py
# reads each as "this component was redeployed at <ts>" and SKIPS its staleness
# pass while the corresponding min-interval window is open (8h by default —
# ORCH_RESTART_MIN_INTERVAL_SECS / FM_RESTART_MIN_INTERVAL_SECS). Falsifying
# either one therefore SUPPRESSES a staleness backstop for the rest of the day,
# invisibly and with every test still green.
#
# The fused-memory clock is included even though only the fleet clock has
# actually been falsified so far, because it is the identical defect class, not
# an adjacent one: the same watchdog, the same semantics, and it is stamped
# from a chained `--stamp-fm-deploy-clock` SUBPROCESS — the same shape (a
# spawned process resolving its target from the environment) that produced the
# fleet-clock bug.
#
# LITERALS, not imports: see the module docstring's import constraint. All four
# mirrors of these paths are pinned together by
# tests/scripts/test_orchestrator_watchdog.py::test_fleet_deploy_clock_path_matches_across_tiers.
PROTECTED_DEPLOY_CLOCK_RELPATHS: tuple[str, ...] = (
    'data/orchestrator/last_redeploy_orchestrator.json',
    'data/fused-memory/last_redeploy_fused_memory.json',
)


def deploy_clock_snapshot(
    root: str | os.PathLike[str],
) -> dict[str, tuple[bytes, int] | None]:
    """Record every protected deploy clock under *root*.

    One entry per :data:`PROTECTED_DEPLOY_CLOCK_RELPATHS`, mapping the relpath
    to ``(bytes, st_mtime_ns)`` when the file exists and ``None`` when it does
    not.  Absence is a FIRST-CLASS value rather than an omitted key: the common
    case is absence (a fresh worktree has no ``data/`` dir at all), so "the file
    was created during the run" — the exact shape of this bug — is only
    detectable if absence is recorded.

    ``(bytes, st_mtime_ns)`` rather than bytes alone.  ``stamp_fleet_deploy_clock``
    writes ``{"ts": <integer seconds>, "iso": ...}`` at ONE-SECOND resolution, so
    two stamps landing inside the same second are byte-identical; mtime is the
    only remaining signal, and a rapid restamp is precisely what a test suite
    produces.

    Never raises on a missing file or a missing ``data/`` directory — that is
    the ordinary state, not an error.
    """
    base = Path(root)
    snapshot: dict[str, tuple[bytes, int] | None] = {}
    for relpath in PROTECTED_DEPLOY_CLOCK_RELPATHS:
        target = base / relpath
        try:
            snapshot[relpath] = (target.read_bytes(), target.stat().st_mtime_ns)
        except OSError:
            snapshot[relpath] = None
    return snapshot


def _clock_change_kind(
    before: tuple[bytes, int] | None, after: tuple[bytes, int] | None,
) -> str | None:
    """Name how a single clock entry changed, or ``None`` if it did not."""
    if before == after:
        return None
    if before is None:
        return 'CREATED during the test run'
    if after is None:
        return 'DELETED during the test run'
    if before[0] != after[0]:
        return 'REWRITTEN during the test run (contents changed)'
    return (
        'RESTAMPED during the test run (contents identical, mtime moved — the '
        'clock writes whole seconds, so two stamps in one second look the same)'
    )


def deploy_clock_violation_reason(
    before: dict[str, tuple[bytes, int] | None],
    after: dict[str, tuple[bytes, int] | None],
) -> str | None:
    """Explain which protected deploy clock the run falsified, or ``None``.

    Reports the FIRST offending relpath in :data:`PROTECTED_DEPLOY_CLOCK_RELPATHS`
    order, what happened to it, and both readings of that observation — a test
    that forgot to redirect its clock (the common case, with the concrete
    remedy) and a genuine concurrent redeploy in a machine-operated checkout
    (not a bug at all).  Naming only the first would invite the reader to assume
    whichever one they thought of first.
    """
    for relpath in PROTECTED_DEPLOY_CLOCK_RELPATHS:
        kind = _clock_change_kind(before.get(relpath), after.get(relpath))
        if kind is None:
            continue
        env_var = (
            'FM_DEPLOY_CLOCK' if 'fused-memory' in relpath else 'ORCH_FLEET_DEPLOY_CLOCK'
        )
        return (
            f'this test run falsified a REAL deploy clock: {relpath} was {kind}.\n'
            'scripts/orchestrator-watchdog.py reads that file as "this component '
            'was redeployed at <ts>" and SKIPS its staleness pass while the '
            'min-interval window is open (8h by default), so the stamp silently '
            'disarms staleness recovery for the rest of the day.\n'
            f'Fix (the usual cause): a test spawned a process that resolved its '
            f'clock path from the environment and defaulted to the live checkout. '
            f'Point {env_var} at a tmp file for the whole suite, as '
            'scripts/tests/conftest.py::_df_fleet_deploy_clock_redirect does, or '
            'per call, as tests/scripts/test_orchestrator_watchdog.py::'
            '_boundary_run_drain_script does with its REQUIRED clock_file '
            'parameter.\n'
            'Benign alternative, worth ruling out first in a machine-operated '
            'checkout: a REAL fleet redeploy (the deployed watchdog, or an '
            'operator running restart-all-orchestrators.sh --drain) fired while '
            'this suite was running. That is a genuine stamp and not a test bug — '
            'compare the {ts, iso} body against the deploy you expect.'
        )
    return None


@pytest.fixture(scope='session', autouse=True)
def _df_deploy_clocks_unwritten():
    """Fail the run if it falsified a REAL deploy clock in this checkout.

    The repo root is ``Path(__file__).resolve().parent`` — this module SITS at
    the repo root, and in a task worktree that correctly yields the WORKTREE
    root, which is precisely the ``REPO_DIR`` that
    ``scripts/restart-all-orchestrators.sh`` computes for its ``CLOCK_FILE``
    default.  So the guard watches exactly the file a forgetful spawner would
    hit, in whichever checkout the suite is running from.

    SESSION scope for the same two reasons ``_df_git_ceiling_at_basetemp``
    documents — cost (a function-scoped autouse fixture runs once per test,
    times every xdist worker) and coverage (a write from a module- or
    session-scoped fixture must be caught too, and that is where expensive
    subprocess setup tends to live).

    DETECTS ONLY — it never restores or rolls the file back.  The main checkout
    is machine-operated (CLAUDE.md): the deployed watchdog and a real
    ``restart-all-orchestrators.sh --drain`` can legitimately stamp the clock
    while a suite runs there.  A restoring guard would silently roll back a
    GENUINE fleet-deploy stamp, re-opening the very 8h window this defence
    exists to close — and doing it invisibly, which is strictly worse than the
    bug.  Hence the failure message names the benign concurrent-redeploy
    reading alongside the test-bug one.

    A failure raised in teardown surfaces as a run-level ERROR with a non-zero
    exit code even when every test passed.  That is the intended loudness: the
    damage is to production state, not to any one test's result.
    """
    root = Path(__file__).resolve().parent
    before = deploy_clock_snapshot(root)
    try:
        yield
    finally:
        reason = deploy_clock_violation_reason(before, deploy_clock_snapshot(root))
        if reason is not None:
            pytest.fail(reason, pytrace=False)
