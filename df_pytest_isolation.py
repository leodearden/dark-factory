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

Same structure as the ceiling above — pure helpers (``deploy_clock_guard_roots``
/ ``deploy_clock_snapshot`` / ``deploy_clock_violation_reason``) plus a thin
session-scoped autouse fixture wiring them — and the same reason for living
here: the defence must be suite-wide and impossible to opt out of, because the
defect class is "a spawner that forgets the env var" and the next one has not
been written yet.

Unlike the ceiling, this one guards TWO roots when it runs inside a task
worktree: that worktree AND the main checkout enclosing it.  The bash script
derives its clock path from its own location (so a worktree run hits the
worktree), but ``scripts/orchestrator-watchdog.py`` HARDCODES ``REPO_DIR``, so
its fused-memory clock and the fleet script it spawns land in the main checkout
whichever worktree invoked them.  One root would have left the second protected
path watched-but-unwritable-to, i.e. green and useless.

THIRD DEFENCE — a test's spawn timeout can never leak a process group.

Task 3798.  Both pytest harnesses that drive
``scripts/restart-all-orchestrators.sh --drain`` used a plain
``subprocess.run(..., timeout=N)``.  On POSIX that timeout path ``kill()``s the
DIRECT CHILD ONLY, so the poll loops the drain script forks outlive it, get
reparented to ``systemd --user``, and sit spending their grace — 86 concurrent
orphans on 2026-08-06 and 82 more on 2026-08-07, each holding a hand-typed
``99999``-second (27.8 HOUR) grace.  That number is the second half of the
defect: an orphan that lives 27.8h outlives pytest's tmpdir GC, at which point
its ``PATH`` no longer resolves ``systemctl`` to the test's fake and its
expiry restarts REAL units and stamps a REAL fleet-deploy clock.

Same structure and the same reason for living here as the two defences above —
``run_in_new_session`` / ``wait_proof_grace_secs`` / ``leaked_drain_processes``
as pure helpers plus a thin session-scoped autouse fixture — because the defect
class is again "a spawner that forgets" and the next one has not been written
yet.  Centralising the spawn matters twice over here: ``tests/scripts/`` and
``scripts/tests/`` cannot import each other's test modules, so their two
copies of this harness could only ever be kept in step by a cross-checking
test (``test_boundary_fake_systemctl_matches_unit_suite_verbatim`` exists for
exactly that reason).  This module is the one place BOTH already import.

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

import math
import os
import signal
import subprocess
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

# A real clock body is ~70 bytes (`{"ts": <int>, "iso": "<timestamp>"}`), so this
# truncates nothing legitimate; it exists only so a failure message stays
# readable if the guard ever fires on a file that is not a clock at all.
_MAX_CLOCK_BODY_CHARS = 200


def deploy_clock_guard_roots(
    module_root: str | os.PathLike[str],
) -> tuple[Path, ...]:
    """Every checkout whose deploy clocks a run in *module_root* must leave alone.

    Always *module_root* itself — the checkout this module sits in, which is the
    ``REPO_DIR`` ``scripts/restart-all-orchestrators.sh`` derives from its own
    path, hence the file a forgetful spawner hits in an ordinary worktree run.

    PLUS, when that checkout is a task worktree (``…/.worktrees/<id>``), the MAIN
    checkout enclosing it.  That second root is not belt-and-braces: the OTHER
    protected clock is written against a HARDCODED path.
    ``scripts/orchestrator-watchdog.py`` fixes ``REPO_DIR =
    "/home/leo/src/dark-factory"``, so its ``FM_DEPLOY_CLOCK_PATH`` default — and
    the fleet restart script it spawns — resolve into the main checkout no matter
    which worktree the suite runs from.  A guard rooted only at the worktree would
    watch a path nothing writes and report all-clear while the REAL production
    clock was being falsified, which is the same "green and useless" failure the
    relpath drift pin exists to prevent.

    The cost is accepted deliberately: a GENUINE concurrent redeploy in the
    machine-operated main checkout now fails a worktree suite too, not just a
    main-checkout one.  That is the loud-over-silent trade this whole defence is
    built on, and it is why the failure message prints the absolute path plus
    both observed bodies — so "real redeploy" is one glance away from "test bug".

    Own checkout FIRST and deduped, so the message names the run's own checkout
    when both were touched, and so a suite running in the main checkout (where
    ``module_root`` has no ``.worktrees`` parent) snapshots it exactly once.
    """
    root = Path(module_root).resolve()
    roots = [root]
    if root.parent.name in _WORKTREE_ROOT_COMPONENTS and len(root.parents) >= 2:
        enclosing = root.parents[1]
        if enclosing != root:
            roots.append(enclosing)
    return tuple(roots)


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


def _describe_clock_entry(entry: tuple[bytes, int] | None) -> str:
    """Render one snapshot reading for the failure message.

    The body is printed, not just the fact that it changed, because
    distinguishing a test stamp from a GENUINE concurrent redeploy means
    comparing the ``{ts, iso}`` against the deploy you expect — and by the time
    anyone reads the failure the file may have moved again, so re-reading it is
    not a substitute.  Both readings are already in hand in the snapshot tuples;
    withholding them just makes triage a second investigation.
    """
    if entry is None:
        return 'absent'
    body, mtime_ns = entry
    text = body.decode('utf-8', errors='replace').strip()
    if len(text) > _MAX_CLOCK_BODY_CHARS:
        text = f'{text[:_MAX_CLOCK_BODY_CHARS]}…(truncated)'
    return f'{text!r} st_mtime_ns={mtime_ns}'


def deploy_clock_violation_reason(
    before: dict[str, tuple[bytes, int] | None],
    after: dict[str, tuple[bytes, int] | None],
    root: str | os.PathLike[str] | None = None,
) -> str | None:
    """Explain which protected deploy clock the run falsified, or ``None``.

    Reports the FIRST offending relpath in :data:`PROTECTED_DEPLOY_CLOCK_RELPATHS`
    order, what happened to it, the before/after readings actually observed, and
    both readings of that observation — a test that forgot to redirect its clock
    (the common case, with the concrete remedy) and a genuine concurrent redeploy
    in a machine-operated checkout (not a bug at all).  Naming only the first
    would invite the reader to assume whichever one they thought of first.

    *root* is optional and cosmetic-but-load-bearing: pass the checkout the
    snapshots were taken against and the message names the ABSOLUTE file.  A run
    guards more than one checkout (see :func:`deploy_clock_guard_roots`), so a
    bare relpath leaves the reader unable to tell which one was falsified.
    """
    for relpath in PROTECTED_DEPLOY_CLOCK_RELPATHS:
        before_entry, after_entry = before.get(relpath), after.get(relpath)
        kind = _clock_change_kind(before_entry, after_entry)
        if kind is None:
            continue
        env_var = (
            'FM_DEPLOY_CLOCK' if 'fused-memory' in relpath else 'ORCH_FLEET_DEPLOY_CLOCK'
        )
        where = relpath if root is None else str(Path(root) / relpath)
        return (
            f'this test run falsified a REAL deploy clock: {where} was {kind}.\n'
            f'observed before: {_describe_clock_entry(before_entry)}\n'
            f'observed after:  {_describe_clock_entry(after_entry)}\n'
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
            'the two bodies printed above ARE that comparison: check the {ts, iso} '
            'against the deploy you expect.'
        )
    return None


@pytest.fixture(scope='session', autouse=True)
def _df_deploy_clocks_unwritten():
    """Fail the run if it falsified a REAL deploy clock in any guarded checkout.

    The roots come from :func:`deploy_clock_guard_roots`, seeded with
    ``Path(__file__).resolve().parent`` — this module SITS at the repo root, and
    in a task worktree that correctly yields the WORKTREE root, which is
    precisely the ``REPO_DIR`` that ``scripts/restart-all-orchestrators.sh``
    computes for its ``CLOCK_FILE`` default.  Under a worktree the enclosing MAIN
    checkout is guarded too, because ``scripts/orchestrator-watchdog.py``
    hardcodes its ``REPO_DIR`` and so writes there regardless of which worktree
    spawned it.  So the guard watches exactly the files a forgetful spawner would
    hit, in every checkout it could hit them in.

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
    roots = deploy_clock_guard_roots(Path(__file__).resolve().parent)
    before = [(root, deploy_clock_snapshot(root)) for root in roots]
    try:
        yield
    finally:
        for root, snapshot in before:
            reason = deploy_clock_violation_reason(
                snapshot, deploy_clock_snapshot(root), root=root,
            )
            if reason is not None:
                pytest.fail(reason, pytrace=False)


# ---------------------------------------------------------------------------
# Session-isolated spawning (task 3798)
# ---------------------------------------------------------------------------

# How long the post-kill drain may take before we give up on collecting the
# child's partial output. Bounded on purpose — see run_in_new_session. Generous
# because it is a CEILING, not a latency: once the group is SIGKILLed every
# write end of the pipe is closed and the drain returns in milliseconds.
_POST_KILL_DRAIN_SECS = 10


# ---------------------------------------------------------------------------
# The wait-proving grace (task 3798, fix (c))
# ---------------------------------------------------------------------------

# How far the grace must exceed the spawn timeout that will kill it. 4x absorbs
# scheduling jitter under 32-way xdist load with room to spare, which is the
# load the existing 20s timeout was itself chosen against.
WAIT_PROOF_GRACE_MULTIPLIER = 4

# A purely multiplicative grace off a 1-3s timeout lands near the script's own
# startup cost, so the defer line the test is waiting for might never print.
WAIT_PROOF_GRACE_FLOOR_SECS = 30

# The other side of the invariant: how long a poller that ESCAPES its kill may
# live before its own grace expires and it exits.
LEAK_SELF_TERMINATION_CEILING_SECS = 90


def wait_proof_grace_secs(spawn_timeout_secs: float) -> int:
    """The grace a wait-proving test should hand the script it spawns.

    Several tests prove ``restart-all-orchestrators.sh`` is genuinely POLLING
    rather than fail-opening, by setting a grace large enough that the script
    is still waiting when the test's own short ``subprocess`` timeout kills it,
    then asserting no restart was recorded. That shape needs a number, and all
    three sites independently picked ``99999`` — 27.8 HOURS.

    Each choice was locally reasonable: bigger is safer against the ONE
    constraint the code stated. The unstated second constraint is what this
    function exists to make explicit, because the number is two things at once:

    * LARGE enough that the wait is provably still a real wait at the moment
      the caller's own ``subprocess`` timeout kills it — at least
      :data:`WAIT_PROOF_GRACE_MULTIPLIER` times that timeout, never below
      :data:`WAIT_PROOF_GRACE_FLOOR_SECS`. Too small and the script force-fires
      mid-test, records the restart, and the assertion fails.
    * SMALL enough that a poller which nevertheless escapes the kill
      self-terminates within :data:`LEAK_SELF_TERMINATION_CEILING_SECS`. Too
      large and it does not: 86 concurrent orphans on 2026-08-06 and 82 more on
      2026-08-07, each with 27.8h left to run.

    THE COUNTER-INTUITIVE PART, because it is the objection a reviewer will
    raise: shrinking the grace makes an escaped orphan reach expiry and issue a
    REAL ``systemctl --user restart`` SOONER. That is strictly SAFER, not
    riskier. At 30-80s the fake ``systemctl`` in the test's tmpdir under
    ``/tmp/pytest-of-<user>/pytest-NNN/`` still exists, so the orphan's PATH
    still resolves to the FAKE and its restart goes nowhere. It was only the
    27.8h survivor that outlived pytest's tmpdir GC, fell through to
    ``/usr/bin/systemctl``, and restarted real units while stamping a real
    fleet-deploy clock. A short grace is what keeps an escape contained inside
    the lifetime of its own containment.

    Returns an ``int``: the value is stringified into an env var and compared
    by bash's integer operators, which reject ``30.0``.
    """
    derived = math.ceil(spawn_timeout_secs * WAIT_PROOF_GRACE_MULTIPLIER)
    return max(WAIT_PROOF_GRACE_FLOOR_SECS, derived)


def _unsafe_pgid_reason(pgid: int) -> str | None:
    """Return why *pgid* is unsafe to ``killpg``, or ``None`` if it is fine.

    Ported (not imported — see the module docstring's stdlib-only constraint)
    from ``shared.proc_group._unsafe_pgid_reason``, which is the canonical
    async sibling of this function and the place to look for the full history.

    This is not belt-and-braces. That module's docstring records task 845: a
    ``killpg`` aimed via ``os.getpgid(pid)`` at a pid the kernel had already
    RECYCLED resolved to the new owner's group, which in practice was the
    ``systemd --user`` manager's — and killed the user's entire login session.
    The hazard is unusually live in this module's case, because the orphans it
    exists to reap are themselves reparented to ``systemd --user``.

    Takes no ``proc_pid`` companion argument, unlike the ``shared`` version:
    the single caller captures ``pgid = p.pid`` at the instant of spawn and
    passes that same value here, so a ``pgid != proc.pid`` check would compare
    a variable against itself.
    """
    if pgid <= 1:
        return f'pgid <= 1 ({pgid!r})'
    if pgid == os.getpid():
        return f'pgid == os.getpid() ({pgid})'
    try:
        ppid = os.getppid()
    except OSError:
        ppid = None
    if ppid is not None and pgid == ppid:
        return f'pgid == os.getppid() ({pgid})'
    try:
        own_pgrp = os.getpgrp()
    except OSError:
        own_pgrp = None
    if own_pgrp is not None and pgid == own_pgrp:
        return f'pgid == os.getpgrp() ({pgid})'
    return None


def run_in_new_session(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    cwd: str | os.PathLike[str] | None = None,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """``subprocess.run``, except a timeout kills the whole process GROUP.

    A drop-in for ``subprocess.run(cmd, env=..., capture_output=True,
    text=True, timeout=...)``: same ``CompletedProcess`` on the ordinary path,
    same ``subprocess.TimeoutExpired`` with partial output attached on the slow
    one. Two things differ, both of them the point.

    FIRST — the child leads its own session, and the timeout signals the GROUP.
    ``subprocess.run``'s timeout path calls ``Popen.kill()``, which signals the
    direct child and nothing else. A script that backgrounds a poll loop
    therefore survives its own caller's timeout: the loop is reparented to
    ``systemd --user`` and keeps running until its grace expires. That is task
    3798's 86-orphan pileup, and it is only reachable by signalling the group.

    The pgid is CAPTURED as ``p.pid`` immediately after a ``start_new_session=
    True`` spawn, and that frozen int is what gets signalled. Deliberately NOT
    ``os.killpg(os.getpgid(p.pid), ...)``, the obvious spelling: once the child
    is reaped the kernel may reuse its pid, and ``os.getpgid`` on a reused pid
    returns the NEW owner's group — task 845, where that resolved to the user's
    ``systemd --user`` session. POSIX guarantees the child is the leader of its
    own group with ``pgid == pid`` at the instant of a new-session spawn, so
    capturing it there closes the TOCTOU completely.
    :func:`_unsafe_pgid_reason` is the residual defence, and a refusal falls
    back to ``p.kill()`` — the direct child only, i.e. exactly today's
    behaviour — rather than signalling a group we are unsure of.

    SECOND — the post-kill drain is BOUNDED. This is the non-obvious half, and
    it is what keeps the DEGRADED paths above safe. A surviving grandchild
    inherits the stdout/stderr pipe WRITE ENDS, so a drain run against it never
    sees EOF; unbounded, it blocks forever and the caller's ``timeout=`` stops
    being a wall-clock bound at all. Measured against real processes: a direct
    ``communicate()`` after a plain ``kill()`` of such a child does hang.

    Whenever the group kill lands this is moot — the whole group dies, every
    write end closes, and the drain returns in milliseconds. It becomes real on
    exactly the two fallbacks: an ``_unsafe_pgid_reason`` refusal and a failed
    ``killpg``, both of which degrade to ``p.kill()`` and so leave a
    pipe-holding grandchild alive. Capping at ``_POST_KILL_DRAIN_SECS`` makes
    the worst case ``timeout + drain`` on every path.

    (For the record, since it is the obvious next question: stock
    ``subprocess.run`` does NOT hang here. Its POSIX timeout branch calls
    ``process.wait()``, not a second ``communicate()`` — verified by inspection
    of CPython. Its defect is solely the un-reached grandchild, not a hang.)

    The re-raised ``TimeoutExpired`` carries whatever was drained, because the
    existing callers assert on the script's output from the TIMEOUT path (via
    their ``_decode`` / ``_boundary_decode`` helpers, which normalise both
    ``bytes`` and ``str`` and so keep working unchanged).
    """
    p = subprocess.Popen(
        cmd,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        start_new_session=True,
    )
    # IMMEDIATELY, and never re-derived: this is the whole task-845 defence.
    pgid = p.pid
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(p, pgid)
        try:
            out, err = p.communicate(timeout=_POST_KILL_DRAIN_SECS)
        except subprocess.TimeoutExpired:
            # Something still holds the pipe open despite the group kill.
            # Give up on the output rather than on the timeout: the caller
            # asked for a bound and gets one.
            p.kill()
            out, err = ('', '') if text else (b'', b'')
        raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err) from None
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def _kill_process_group(p: subprocess.Popen, pgid: int) -> None:
    """SIGKILL the captured *pgid*, degrading to the direct child if unsure.

    Every failure mode degrades to ``p.kill()`` rather than propagating: the
    caller is already on its way to raising ``TimeoutExpired``, and masking
    that with a signalling error would replace a legible timeout with a
    confusing one. ``ProcessLookupError`` is the ordinary case of a child that
    exited between the timeout and the signal.
    """
    reason = _unsafe_pgid_reason(pgid)
    if reason is not None:
        p.kill()
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        p.kill()


# ---------------------------------------------------------------------------
# The in-suite leak guard (task 3798)
# ---------------------------------------------------------------------------

# Stamped into os.environ for the session, so EVERY spawner that builds its
# child env from `dict(os.environ)` — both of today's, plus any future one —
# tags its descendants with no per-spawner change. Same mechanism and same
# "the defect class is a spawner that forgets" reasoning as 3797's clock
# redirect.
LEAK_TOKEN_ENV = 'DF_PYTEST_LEAK_TOKEN'

# The one script whose survivors are worth failing a run over.
DRAIN_SCRIPT_CMDLINE_MARKER = 'restart-all-orchestrators.sh'


def leaked_drain_processes(
    token: str | None, *, proc_root: str | os.PathLike[str] = Path('/proc'),
) -> list[tuple[int, str]]:
    """Every live drain-script process belonging to THIS pytest session.

    Returns ``(pid, cmdline)`` for each process whose ``/proc/<pid>/cmdline``
    contains :data:`DRAIN_SCRIPT_CMDLINE_MARKER` AND whose
    ``/proc/<pid>/environ`` carries ``LEAK_TOKEN_ENV=<token>``.

    WHY BOTH, RATHER THAN THE OBVIOUS ``pgrep -f restart-all-orchestrators.sh``:

    * The token is what makes the guard ATTRIBUTABLE. ``merge_verify_breadth:
      "full"`` means many worktrees run this same suite concurrently, and the
      main checkout is machine-operated (CLAUDE.md) — the deployed watchdog and
      the merge coordinator both run ``--drain`` for real. Asserted literally,
      the bare pgrep is a false-positive generator that fails runs for other
      people's processes and for genuine production activity. Requiring this
      session's own token means a match provably belongs to US.
    * The marker is what keeps the guard QUIET. These two test directories hold
      ~90 files that spawn subprocesses; scanning for "any surviving descendant"
      would flake constantly on all of them.

    Fails CLOSED on a falsy token — an empty token must never be read as "match
    everything", which would turn the guard into a suite-wide false-positive
    generator the first time the fixture failed to stamp it.

    Every per-pid read is individually guarded: scanning /proc races process
    exit by construction, and a pid that vanishes mid-scan is the ordinary case,
    not an error. Mirrors the same invariant in ``shared.proc_group``, the
    canonical async sibling of this module's process handling.
    """
    if not token:
        return []

    # The exact NUL-delimited assignment, so a token that merely PREFIXES
    # another session's cannot cross-match. /proc separates entries with NUL
    # and the first entry has no leading one, hence the two accepted forms.
    assignment = LEAK_TOKEN_ENV.encode() + b'=' + token.encode() + b'\x00'
    marker = DRAIN_SCRIPT_CMDLINE_MARKER.encode()

    leaks: list[tuple[int, str]] = []
    try:
        entries = sorted(Path(proc_root).iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / 'cmdline').read_bytes()
            if marker not in cmdline:
                continue
            environ = (entry / 'environ').read_bytes()
        except OSError:
            continue
        if not (environ.startswith(assignment) or b'\x00' + assignment in environ):
            continue
        # errors='replace' only for the human-readable rendering: environ and
        # cmdline are raw bytes and need not be valid UTF-8.
        leaks.append((
            int(entry.name),
            cmdline.replace(b'\x00', b' ').decode('utf-8', errors='replace').strip(),
        ))
    return leaks


def leaked_drain_process_reason(leaks: list[tuple[int, str]]) -> str | None:
    """Explain which drain processes this run leaked, or ``None`` if it did not.

    Names every pid and its cmdline, the mechanism, and the remedy SYMBOL —
    mirroring :func:`deploy_clock_violation_reason`'s convention that a guard's
    message has to be actionable without a second investigation.
    """
    if not leaks:
        return None
    listed = '\n'.join(f'  pid {pid}: {cmdline}' for pid, cmdline in leaks)
    return (
        f'this test run LEAKED {len(leaks)} drain-script process(es) (task 3798):\n'
        f'{listed}\n'
        'Each carries this session\'s own DF_PYTEST_LEAK_TOKEN, so they are '
        'provably ours — not a concurrent worktree\'s run and not a genuine '
        'machine-operated --drain.\n'
        'Mechanism: a spawn whose timeout kill did not reach the process group. '
        "subprocess.run's timeout path kill()s the DIRECT CHILD only, so a poll "
        'loop the script forked survives, is reparented to systemd --user, and '
        'spends its grace unattended — then issues a REAL systemctl restart if '
        'it outlives the tmpdir holding the fake one.\n'
        'Fix: spawn via df_pytest_isolation.run_in_new_session, which starts a '
        'new session and SIGKILLs the whole group on timeout.\n'
        'These processes have been reaped, so the box is clean; this failure is '
        'the signal, not the damage.'
    )
