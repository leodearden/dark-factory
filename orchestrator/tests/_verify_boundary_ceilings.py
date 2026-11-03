"""The timing every verify-merge boundary row runs under.

The pinned watchdog window the rows hand the real CLI, the wedge-detector
ceilings their waits spend, the load-scaled discovery ceiling, and the
per-test pytest budgets derived from all of them.  Deriving the budgets here,
from the same symbols the waits spend, is what keeps a mark from silently
truncating a wait it was meant to cover.
"""

from __future__ import annotations

import math
import os
import warnings

from df_pytest_isolation import load_scaled_grace

# ---------------------------------------------------------------------------
# Task 4025 (de-flake beta) -- pinned SS9 Row 1/2/3 watchdog window.
#
# ROW_WATCHDOG_ENV carries PINNED LITERALS, deliberately NOT imported from
# verify_cancel.  They used to equal production; task 4195 has since landed and
# derived WATCHDOG_HEARTBEAT_TIMEOUT_SECS from the ssh transport's own
# dead-peer verdict (WATCHDOG_TRANSPORT_HEADROOM 1.5 x SSH_SERVER_ALIVE_INTERVAL
# 15 x SSH_SERVER_ALIVE_COUNT_MAX 4 = 90.0; the 5.0 kill grace is unchanged),
# so the pin and production now deliberately DIFFER.  Tracking production here
# would silently multiply the rows' runtime -- Row 3 from ~17s to ~95s and
# ROW_PER_TEST_TIMEOUT_SECS from 390 to ~650 -- for no coverage, because the
# rows already drive the real production code path through ROW_WATCHDOG_ENV ->
# cli.py::_env_float.  The
# pin is now STRUCTURAL, not just asserted: the literals live on
# ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS / ROW_WATCHDOG_KILL_GRACE_SECS below,
# and ROW_TREE_KILL_CEILING_SECS / ROW_PER_TEST_TIMEOUT_SECS are computed
# from those two floats, so they can no longer drift out of step with the
# window.  Nothing here imports verify_cancel's watchdog timing constants
# (WATCHDOG_HEARTBEAT_TIMEOUT_SECS / WATCHDOG_KILL_GRACE_SECS) -- the boundary
# harness and rows deliberately still share verify_cancel's /proc walkers
# (collect_descendants, pgid_file, etc.); a dedicated drift test in
# test_verify_boundary_ceilings.py imports the two timing constants LOCALLY,
# on its own, purely to notice if they ever diverge from this pin.
#
# This REPLACES a prior test-only 1.0s/0.5s tightening that bought speed and
# no coverage, and was the measured cause of the "no descendant appeared
# within 20.0s" flake on wait_subtree_live (6 failures / 490 legs on the
# 1.0s arm vs 0 on both the 10s arm and the separate-process producer arm).
#
# HONEST CAVEAT: this MOVES the cliff from 1.0s to 10s, it does not remove
# it.  The watchdog deadline is a wall-clock select() in
# verify_cancel.run_stdin_watchdog -- LOAD-RIGID -- while the producer
# (_verify_boundary_harness.py::HeartbeatWriter._run) is an Event.wait(0.2)
# -- LOAD-ELASTIC.  Measured producer inter-write gap at loadavg 113-178 was
# p999 0.386s / max 0.845s: only ~1.2x real margin against a 1.0s deadline,
# but ~12x against 10.0s.
# Do not try to buy margin by pre-buffering heartbeats -- run_stdin_watchdog
# drains with a single read(4096) per select, so a burst resets exactly one
# window, not N.
ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS: float = 10.0
ROW_WATCHDOG_KILL_GRACE_SECS: float = 5.0

#: What production read when the divergence above was last DECIDED, not a value
#: the rows spend.  Its only consumer is test_verify_boundary_ceilings.py::
#: test_row_watchdog_window_tracks_the_baselined_production_window,
#: which fails the day production moves again so the next author re-decides
#: whether the rows' fast window is still representative.
BASELINED_PRODUCTION_WATCHDOG_WINDOW: tuple[float, float] = (90.0, 5.0)

#: Worst-case time to fire and finish killing the tree.  Row 3 takes
#: run_stdin_watchdog's select-TIMEOUT branch and pays the full sum; Rows
#: 1/2 take the EOF branch and pay only the grace.
ROW_WATCHDOG_WINDOW_SECS: float = (
    ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS + ROW_WATCHDOG_KILL_GRACE_SECS
)  # 15.0

ROW_WATCHDOG_ENV: dict[str, str] = {
    'ORCH_WATCHDOG_HEARTBEAT_TIMEOUT_SECS': str(ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS),
    'ORCH_WATCHDOG_KILL_GRACE_SECS': str(ROW_WATCHDOG_KILL_GRACE_SECS),
}

#: Ceiling for the rows' child.wait()/wait_pids_exited() polls: the full
#: window plus load headroom.  A WEDGE-DETECTOR, not a speed assertion (the
#: rows assert THAT the tree was killed, never how fast), so on the success
#: path a wider ceiling costs zero wall-clock and is paid only when the test
#: is already failing.
ROW_TREE_KILL_CEILING_SECS: float = ROW_WATCHDOG_WINDOW_SECS + 15.0  # 30.0

#: Ceiling for the marker waits (_verify_boundary_harness.py's
#: :func:`wait_for_marker` / :func:`wait_for_marker_stable`, whose bare
#: defaults resolve to it).  Value unchanged from the literal it replaces;
#: the point of naming it is that Row 5's budget below references the SAME
#: symbol those helpers spend, so the two cannot drift.  NOT load-scaled:
#: unlike descendant discovery, this waits on a single already-forked build
#: shell reaching one ``touch``, and no measurement in this repo implicates it
#: in a load flake.
ROW_MARKER_CEILING_SECS: float = 20.0

#: Base ceiling for the two DISCOVERY waits every row runs BEFORE the
#: watchdog is even armed.  Rows 1-4 run wait_for_pgid_file twice -- once on
#: the leader's pgid file, once on the build's own (sleeper_spec
#: build_pgid_file); Row 5 runs wait_for_pgid_file, then wait_subtree_live.
#: Rows 1-4 pass the resolved ceiling explicitly at their call sites in
#: test_laptop_warm_verify_boundary.py (instead of relying on the bare
#: default) so this value and those defaults
#: cannot silently drift apart.
ROW_DISCOVERY_CEILING_BASE_SECS: int = 20

#: Task 4014.  Top of the measured per-core dilation envelope, NOT a guess.
#: Every dilation figure this repo has actually measured is expressed in
#: loadavg-per-core terms and lands in the 3-6x band: tests/warm-lane's
#: README records 3-6x at loadavg 124 on 32 cores plus a 6.2x module figure;
#: the ROW_WATCHDOG banner above records producer gaps at loadavg 113-178; and the
#: 3689 steward measured THIS test pair stretching 10s -> 51-59s (~5-6x).
#: Scaling keys on loadavg-per-core rather than PSI because this suite's
#: autouse _hermetic_psi_reader fixture injects a stub PSI reader suite-wide,
#: so /proc/pressure readings are deliberately untrustworthy here.
_DISCOVERY_CEILING_MAX_SCALE: int = 6

#: The widest load-scaled ceiling, i.e. the clamp before any operator pin.
ROW_DISCOVERY_CEILING_UNPINNED_MAX_SECS: int = (
    ROW_DISCOVERY_CEILING_BASE_SECS * _DISCOVERY_CEILING_MAX_SCALE
)  # 120

#: Operator knob: pin the discovery ceiling outright (e.g. to reproduce a
#: discovery timeout quickly instead of waiting out a load-scaled deadline).
#: Namespaced ORCH_TEST_*, NOT bare ORCH_*:
#: _verify_boundary_harness.py::subprocess_env copies os.environ into every
#: spawned verify-merge, and the ORCH_* namespace there is production knobs
#: the CLI actually reads.
_DISCOVERY_CEILING_OVERRIDE_ENV: str = 'ORCH_TEST_DISCOVERY_CEILING_SECS'


def resolve_ceiling_override(environ) -> float | None:
    """Parse :data:`_DISCOVERY_CEILING_OVERRIDE_ENV` out of *environ*.

    Returns None when unset, and warns-then-returns-None on a value that is
    unparseable, non-finite, or non-positive.  Deliberately does NOT raise:
    this is resolved at import, so raising would fail COLLECTION for every
    module importing this one -- every row and every helper test taken down by one
    mistyped debugging knob, a strictly worse and far more confusing failure
    than the one being guarded against.  Silently ignoring it would be worse
    still, hence the warning.

    NON-FINITE is rejected for exactly that reason, and is not a hypothetical:
    ``inf`` is the natural spelling an operator reaches for to mean "never
    time out", and ``float('inf')`` parses cleanly while satisfying every
    ordering guard (``inf <= 0`` and ``nan <= 0`` are both False).  It would
    therefore flow into ROW_DISCOVERY_CEILING_MAX_SECS and blow up the very
    next module-level statement -- ``int(... + 2 * inf)`` raises OverflowError,
    ``nan`` raises ValueError -- taking down collection of every importer,
    i.e. precisely the failure this warn-instead-of-raise contract exists to
    prevent.

    Takes an environ MAPPING rather than reading ``os.environ`` so it is
    deterministically testable with no monkeypatch-versus-import-order race.
    """
    raw = environ.get(_DISCOVERY_CEILING_OVERRIDE_ENV)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = None
    if value is None or not math.isfinite(value) or value <= 0:
        warnings.warn(
            f'{_DISCOVERY_CEILING_OVERRIDE_ENV}={raw!r} is not a finite positive number; '
            f'ignoring it and using the load-scaled discovery ceiling instead',
            stacklevel=2,
        )
        return None
    return value


#: Resolved ONCE at import -- see the module's discovery-ceiling notes and
#: :func:`row_discovery_ceiling_secs`.  A per-call env read could return a
#: value the rows' already-baked @pytest.mark.timeout marks cannot accommodate.
DISCOVERY_CEILING_OVERRIDE_SECS: float | None = resolve_ceiling_override(os.environ)

#: The bound the rows' import-time @pytest.mark.timeout marks are derived from.
#: No value row_discovery_ceiling_secs() can return may exceed it -- see
#: test_verify_boundary_ceilings.py::
#: test_row_per_test_timeout_still_covers_the_max_discovery_ceiling.  The
#: override is folded in (rather than sitting outside the bound) so that
#: invariant holds for every reachable value, pinned or scaled.
ROW_DISCOVERY_CEILING_MAX_SECS: float = (
    DISCOVERY_CEILING_OVERRIDE_SECS
    if DISCOVERY_CEILING_OVERRIDE_SECS is not None
    else ROW_DISCOVERY_CEILING_UNPINNED_MAX_SECS
)  # 120 unpinned


def row_discovery_ceiling_secs(*, _override=DISCOVERY_CEILING_OVERRIDE_SECS) -> float:
    """Resolve the discovery ceiling for a wait that is starting NOW.

    A WEDGE DETECTOR, not a speed assertion: the rows assert THAT a tree was
    discovered, never how fast, so on the success path a wider ceiling costs
    ZERO wall clock and is paid only when the test is already failing.  That
    asymmetry is what makes scaling all the way to
    :data:`_DISCOVERY_CEILING_MAX_SCALE` safe for suite runtime -- an idle
    box still pays exactly :data:`ROW_DISCOVERY_CEILING_BASE_SECS`.

    Sampled per CALL rather than at import so a row that starts mid-storm
    sees the storm, and clamped at both ends: never tighter than base (a
    quiet box must not get a stricter deadline than it has always had) and
    never wider than :data:`ROW_DISCOVERY_CEILING_MAX_SECS` (the bound the
    import-time pytest timeout is derived from).

    An operator pin via :data:`_DISCOVERY_CEILING_OVERRIDE_ENV` is returned
    VERBATIM, bypassing load scaling in both directions -- otherwise "pin it
    low to reproduce the timeout quickly" would silently do nothing on a busy
    box.  It is safe against the clamp invariant because
    :data:`ROW_DISCOVERY_CEILING_MAX_SECS` folds the same pin in.

    The scaling is :func:`df_pytest_isolation.load_scaled_grace`'s (floor at
    base, ceil to whole seconds, clamp, fail-safe to base without loadavg);
    only the operator pin is decided here.
    """
    if _override is not None:
        return _override
    # Clamp against the UNPINNED bound, not ROW_DISCOVERY_CEILING_MAX_SECS:
    # the latter folds an operator pin in, and a pin is already returned above.
    # Reading it here would make a low pin silently re-clamp the scaled branch
    # too, so the `_override=None` seam would not mean "as if unpinned".
    return load_scaled_grace(
        ROW_DISCOVERY_CEILING_BASE_SECS, cap_secs=ROW_DISCOVERY_CEILING_UNPINNED_MAX_SECS,
    )


#: Worst-case BOUNDED work in a row on the failure path, at the WIDEST
#: discovery ceiling reachable: both discovery waits, then one full watchdog
#: window, then both ceiling-bounded kill-confirmation waits run to their
#: full ceiling.  (Deliberately excludes _setup_verify_repo's real git work
#: and the finally block's ~5s child.kill()/wait() tail -- both small next to
#: the headroom below.)  Keyed on ROW_DISCOVERY_CEILING_MAX_SECS, not on the
#: base: task 4014 made the discovery ceiling load-scaled, and a widened wait
#: the rows' @pytest.mark.timeout (derived below) cannot accommodate would be TRUNCATED by
#: pytest-timeout's thread-mode os._exit() -- reporting as a worker kill
#: instead of the row's self-diagnosing AssertionError.
_ROW_WORST_CASE_FIXED_SECS: float = (
    ROW_WATCHDOG_WINDOW_SECS + 2 * ROW_TREE_KILL_CEILING_SECS
)  # 75.0

#: Per-test opt-out from a boundary test module's inherited `timeout = 60`
#: (orchestrator/pyproject.toml:103, thread-mode -- os._exit()s the xdist
#: worker on expiry).  2x the FIXED terms, because those terms do not track
#: load themselves while the child's `from orchestrator.cli import main` does:
#: the waiter comment in
#: test_laptop_warm_verify_boundary.py::test_flock_contention_full_two_way_seam_blocks_and_escalates
#: measures that import at ~9s under a full-suite storm.
#: The discovery term is counted ONCE and exempted from that 2x: it now
#: scales with loadavg-per-core on its own, so applying the storm factor to
#: it as well would double-count the same elasticity.
ROW_PER_TEST_TIMEOUT_SECS: int = int(
    2 * _ROW_WORST_CASE_FIXED_SECS + 2 * ROW_DISCOVERY_CEILING_MAX_SECS
)  # 390

#: Row 5's own bounded terms.  It does not share Rows 1-4's shape -- no
#: watchdog window, no kill confirmation -- so it cannot share their
#: constant, only their DERIVATION.  Named here (rather than
#: passed as literals at the call sites) so the budget below and the waits it
#: is supposed to cover move together.
ROW5_WAITER_COMPLETION_CEILING_SECS: float = 60.0
ROW5_HOLDER_TEARDOWN_CEILING_SECS: float = 5.0

#: Worst-case FIXED (non-load-scaled) work in Row 5: the marker wait, the
#: waiter subprocess's completion ceiling, and the finally block's TWO
#: holder teardowns (waiter then holder).  The marker term is counted TWICE
#: because :func:`wait_for_marker_stable` spends its timeout on the existence
#: gate and then a FRESH deadline on the settle loop -- worst case 40s, not
#: 20s, a term the literal mark this replaced hid entirely.  (Deliberately
#: excludes _setup_verify_repo's real git work and the in-process consumer
#: half -- both small next to the headroom below, exactly as for Rows 1-4.)
_ROW5_WORST_CASE_FIXED_SECS: float = (
    2 * ROW_MARKER_CEILING_SECS
    + ROW5_WAITER_COMPLETION_CEILING_SECS
    + 2 * ROW5_HOLDER_TEARDOWN_CEILING_SECS
)  # 110.0

#: Row 5's per-test opt-out from the inherited `timeout = 60`, in the same
#: shape as ROW_PER_TEST_TIMEOUT_SECS above: 2x the FIXED terms (they do not
#: track load while the child's CLI import does -- see the measurement cited
#: there), plus the two
#: discovery waits counted ONCE at MAX and exempted from that 2x -- they now
#: scale with loadavg-per-core themselves, so applying the storm factor to
#: them as well would double-count the same elasticity.
ROW5_PER_TEST_TIMEOUT_SECS: int = int(
    2 * _ROW5_WORST_CASE_FIXED_SECS + 2 * ROW_DISCOVERY_CEILING_MAX_SECS
)  # 460
# ---------------------------------------------------------------------------
