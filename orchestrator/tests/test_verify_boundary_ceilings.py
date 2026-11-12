"""Self-tests for the boundary rows' timing ceilings and budgets in _verify_boundary_ceilings."""

from __future__ import annotations

import math
import os

import pytest
from _verify_boundary_ceilings import (
    BASELINED_PRODUCTION_WATCHDOG_WINDOW,
    DISCOVERY_CEILING_OVERRIDE_SECS,
    ROW5_PER_TEST_TIMEOUT_SECS,
    ROW_DISCOVERY_CEILING_BASE_SECS,
    ROW_DISCOVERY_CEILING_MAX_SECS,
    ROW_DISCOVERY_CEILING_UNPINNED_MAX_SECS,
    ROW_PER_TEST_TIMEOUT_SECS,
    ROW_TREE_KILL_CEILING_SECS,
    ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS,
    ROW_WATCHDOG_KILL_GRACE_SECS,
    ROW_WATCHDOG_WINDOW_SECS,
    resolve_ceiling_override,
    row_discovery_ceiling_secs,
)

# ---------------------------------------------------------------------------
# Task 4025 amendment -- drift detector for the ROW_WATCHDOG pin in
# _verify_boundary_ceilings.py.  ROW_WATCHDOG_ENV is pinned to today's
# production numbers but deliberately does not TRACK verify_cancel's constants
# (see _verify_boundary_ceilings.py's ROW_WATCHDOG banner); that decoupling means nothing else
# notices the day the two diverge.  This is the one test whose whole job is to
# notice, so the import it performs (LOCAL to the test body, not module-level)
# is a deliberate, narrow exception to that banner's "nothing here imports
# verify_cancel's watchdog timing constants" -- it exists specifically to
# compare against them, not to consume them.
# ---------------------------------------------------------------------------


def test_row_watchdog_window_tracks_the_baselined_production_window():
    """Fails the day verify_cancel's WATCHDOG_* moves off its baselined value.

    task 4025 pinned ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS /
    ROW_WATCHDOG_KILL_GRACE_SECS as literals specifically so Rows 1/2/3's
    runtime does NOT track verify_cancel.WATCHDOG_HEARTBEAT_TIMEOUT_SECS /
    WATCHDOG_KILL_GRACE_SECS.  Those two once happened to be equal; task 4195
    has since derived the production timeout from the ssh transport (90.0s),
    so the pin and production now deliberately DIFFER and equality is no
    longer the thing to assert.

    What survives is this test's actual job: tell the next author that
    production moved AGAIN, so they re-decide whether the rows' fast window is
    still representative of the behaviour they exercise.

    If this fails: production changed.  Do NOT "fix" it by importing the
    pinned literals from verify_cancel -- that reintroduces the
    silent-runtime-multiplication task 4025 built the pin to prevent, and buys
    no coverage, because the rows already drive the real production code path
    through ROW_WATCHDOG_ENV -> cli.py::_env_float.  Instead update the three
    Row 1/2/3 docstrings' window wording in test_laptop_warm_verify_boundary.py,
    decide afresh whether
    ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS / ROW_WATCHDOG_KILL_GRACE_SECS should
    move, and re-baseline BASELINED_PRODUCTION_WATCHDOG_WINDOW.
    """
    from orchestrator import verify_cancel

    production = (
        verify_cancel.WATCHDOG_HEARTBEAT_TIMEOUT_SECS,
        verify_cancel.WATCHDOG_KILL_GRACE_SECS,
    )
    assert production == BASELINED_PRODUCTION_WATCHDOG_WINDOW, (
        f'production verify_cancel.WATCHDOG_* {production} no longer matches the '
        f'baselined {BASELINED_PRODUCTION_WATCHDOG_WINDOW} the boundary rows were '
        f'last reconciled against. The pinned ROW_WATCHDOG_* '
        f'{(ROW_WATCHDOG_HEARTBEAT_TIMEOUT_SECS, ROW_WATCHDOG_KILL_GRACE_SECS)} '
        f'intentionally do not track production; re-decide whether they should '
        f'move, then re-baseline BASELINED_PRODUCTION_WATCHDOG_WINDOW.'
    )


# ---------------------------------------------------------------------------
# Task 4014 -- deterministic coverage for the load-scaled discovery ceiling.
# Pure function, no clock, no /proc, no subprocess: load is injected by
# patching os.getloadavg/os.cpu_count.
# ---------------------------------------------------------------------------


def _inject_load(monkeypatch, loadavg: float, cpu_count: int = 32) -> None:
    """Patch the load signal on the ``os`` MODULE, where every reader sees it."""
    monkeypatch.setattr(os, 'getloadavg', lambda: (loadavg, loadavg, loadavg))
    monkeypatch.setattr(os, 'cpu_count', lambda: cpu_count)


def _unpinned_ceiling_at(monkeypatch, loadavg: float) -> float:
    _inject_load(monkeypatch, loadavg)
    return row_discovery_ceiling_secs(_override=None)


def test_row_discovery_ceiling_scales_with_load_and_clamps(monkeypatch):
    """The discovery ceiling widens with per-core load and is bounded at both ends.

    This is a WEDGE DETECTOR, not a speed assertion -- the rows assert THAT
    a tree was discovered, never how fast -- so on the success path a wider
    ceiling costs ZERO wall clock and is paid only when the test is already
    failing.  The four properties that matter:

    (a) an IDLE box must get exactly the base ceiling, so a quiet run is
        never slowed down;
    (b) real per-core load must actually widen it (3x per core is the BOTTOM
        of this repo's measured dilation envelope);
    (c) an absurd load must clamp, so a runaway box cannot stretch a row
        past the pytest timeout that is DERIVED from that clamp;
    (d) the mapping is monotone -- more load never yields a tighter deadline.
    """
    idle = _unpinned_ceiling_at(monkeypatch, 0.1)
    assert idle == ROW_DISCOVERY_CEILING_BASE_SECS, (
        f'an idle box must pay exactly the base ceiling '
        f'({ROW_DISCOVERY_CEILING_BASE_SECS}); got {idle}'
    )

    loaded = _unpinned_ceiling_at(monkeypatch, 96.0)
    assert loaded > ROW_DISCOVERY_CEILING_BASE_SECS, (
        f'at 3x per-core load -- the BOTTOM of the measured dilation envelope -- '
        f'the ceiling must widen past base; got {loaded}'
    )

    absurd = _unpinned_ceiling_at(monkeypatch, 6400.0)
    assert absurd == ROW_DISCOVERY_CEILING_UNPINNED_MAX_SECS, (
        f'a runaway load must clamp to {ROW_DISCOVERY_CEILING_UNPINNED_MAX_SECS} -- that clamp is what '
        f'ROW_DISCOVERY_CEILING_MAX_SECS, and in turn the import-time pytest '
        f'timeout, are derived from; got {absurd}'
    )

    ladder = [
        _unpinned_ceiling_at(monkeypatch, load)
        for load in (0.0, 1.0, 32.0, 64.0, 96.0, 200.0, 1000.0, 6400.0)
    ]
    assert ladder == sorted(ladder), (
        f'more load must never yield a TIGHTER deadline; got {ladder}'
    )
    assert all(v >= ROW_DISCOVERY_CEILING_BASE_SECS for v in ladder), (
        f'no reachable value may drop below the base ceiling; got {ladder}'
    )


def test_row_discovery_ceiling_rounds_up_to_whole_seconds(monkeypatch):
    """A fractional per-core load widens the ceiling to the next whole second."""
    ceiling = _unpinned_ceiling_at(monkeypatch, 33.0)
    expected = math.ceil(ROW_DISCOVERY_CEILING_BASE_SECS * 33.0 / 32)
    assert ceiling == expected, (
        f'at loadavg 33.0 on 32 cores the ceiling must be base * 33/32 rounded '
        f'up to whole seconds ({expected}); got {ceiling}'
    )


def test_row_discovery_ceiling_fails_safe_without_loadavg(monkeypatch):
    """A platform with no loadavg gets the base deadline, never a crash at the call site."""
    def _raise_oserror():
        raise OSError('getloadavg not supported on this platform')

    def _raise_attributeerror():
        raise AttributeError('os has no getloadavg on this platform')

    for no_loadavg in (_raise_oserror, _raise_attributeerror):
        monkeypatch.setattr(os, 'getloadavg', no_loadavg)
        assert row_discovery_ceiling_secs(_override=None) == ROW_DISCOVERY_CEILING_BASE_SECS


def test_discovery_ceiling_env_override_parses_and_pins(monkeypatch):
    """ORCH_TEST_DISCOVERY_CEILING_SECS parses safely and beats load scaling.

    Two halves, both deterministic -- the parser takes an environ MAPPING
    rather than reading os.environ, so neither half monkeypatches import-time
    state or races import order.

    (a) Parsing.  A good value parses; an absent one is None; a typo'd one is
        None AND warns.  Warning rather than raising is deliberate: this
        constant is resolved at IMPORT, so raising would fail COLLECTION for
        every module importing the ceilings -- every row and every helper
        test taken down by one mistyped debugging knob, which is a strictly
        worse and far more confusing failure than the one being guarded
        against.  Silently ignoring it would be worse still (the project's
        no-silent-fail-soft norm), hence the warning.

        NON-FINITE inputs are covered alongside the typo because they defeat
        that contract in a way a typo does not: 'banana' fails float(), but
        'inf'/'nan' PARSE, and `inf <= 0` / `nan <= 0` are both False, so
        without an explicit finiteness guard they sail through into
        ROW_DISCOVERY_CEILING_MAX_SECS and detonate the next module-level
        statement (`int(... + 2*inf)` -> OverflowError; nan -> ValueError),
        failing collection for every module importing it.  'inf' is not an exotic
        input either: it is what an operator writes to mean "never time out".

    (b) Pinning.  An explicit operator pin wins in BOTH directions -- it must
        override load scaling upward and downward alike, or "pin it to 7.5 to
        reproduce the timeout quickly" silently does nothing on a busy box.

    The knob is namespaced ORCH_TEST_*, not bare ORCH_*: subprocess_env()
    copies os.environ into every spawned verify-merge, and the ORCH_*
    namespace there is production knobs the CLI actually reads.
    """
    assert resolve_ceiling_override({'ORCH_TEST_DISCOVERY_CEILING_SECS': '7.5'}) == 7.5
    assert resolve_ceiling_override({}) is None

    for bad in ('banana', 'inf', '-inf', 'nan', '0', '-1'):
        with pytest.warns(UserWarning, match='ORCH_TEST_DISCOVERY_CEILING_SECS'):
            assert resolve_ceiling_override(
                {'ORCH_TEST_DISCOVERY_CEILING_SECS': bad}
            ) is None, f'{bad!r} must be rejected, not pinned'

    # The consequence the rejection above exists to prevent, asserted against
    # the ceilings module's OWN import-time constants -- the values THIS process is
    # actually running under, whatever env it was collected with.  A local
    # re-derivation would have been vacuous twice over: every value the
    # resolver accepts is finite and positive by construction, and a copy of
    # the derivation cannot notice the MODULE's derivation being loosened.
    assert math.isfinite(ROW_DISCOVERY_CEILING_MAX_SECS) and ROW_DISCOVERY_CEILING_MAX_SECS > 0, (
        f'ROW_DISCOVERY_CEILING_MAX_SECS resolved to '
        f'{ROW_DISCOVERY_CEILING_MAX_SECS} -- a non-finite or non-positive '
        f'ceiling reached import, so the finiteness guard above is being bypassed'
    )
    for name, derived in (
        ('ROW_PER_TEST_TIMEOUT_SECS', ROW_PER_TEST_TIMEOUT_SECS),
        ('ROW5_PER_TEST_TIMEOUT_SECS', ROW5_PER_TEST_TIMEOUT_SECS),
    ):
        assert derived > 0, (
            f'{name} derived to {derived} -- the per-test timeouts are int() of a '
            f'sum containing the ceiling, so this is what a non-finite ceiling '
            f'looks like on the far side of the derivation'
        )

    _inject_load(monkeypatch, 6400.0)
    assert row_discovery_ceiling_secs(_override=7.5) == 7.5
    _inject_load(monkeypatch, 0.0)
    assert row_discovery_ceiling_secs(_override=7.5) == 7.5


def test_row_per_test_timeout_still_covers_the_max_discovery_ceiling():
    """The per-test pytest timeout must cover the WIDEST discovery ceiling reachable.

    Extends task 4025's "the coupling is STRUCTURAL, not merely asserted"
    property (commit 71a4d37f17) to the now load-scaled discovery ceiling,
    against the AMBIENT module constants.

    Without (b), a load-widened discovery wait would be silently truncated by
    the import-time ``@pytest.mark.timeout(...)`` decorator -- and a widened
    deadline that never gets to run is WORSE than no fix at all, because
    pytest-timeout's thread mode ``os._exit()``s the xdist worker: the row
    then reports as a worker kill rather than as the self-diagnosing
    AssertionError (leader rc taxonomy + stderr tail) task 4025-alpha built
    precisely so this failure would explain itself.
    """
    resolved = row_discovery_ceiling_secs()
    assert resolved <= ROW_DISCOVERY_CEILING_MAX_SECS, (
        f'row_discovery_ceiling_secs() returned {resolved}, above the '
        f'ROW_DISCOVERY_CEILING_MAX_SECS ({ROW_DISCOVERY_CEILING_MAX_SECS}) that '
        f'ROW_PER_TEST_TIMEOUT_SECS is derived from -- the widened wait would be '
        f'truncated by @pytest.mark.timeout instead of running'
    )
    if DISCOVERY_CEILING_OVERRIDE_SECS is None:
        # An operator pin deliberately escapes the base floor (pinning LOW is
        # how you reproduce a discovery timeout quickly), so the floor is
        # asserted only for the unpinned, CI-reachable configuration.
        assert resolved >= ROW_DISCOVERY_CEILING_BASE_SECS, (
            f'row_discovery_ceiling_secs() returned {resolved}, below the base '
            f'ceiling ({ROW_DISCOVERY_CEILING_BASE_SECS}) -- load scaling must '
            f'never hand a quiet box a TIGHTER deadline than it always had'
        )

    required = (
        2 * ROW_DISCOVERY_CEILING_MAX_SECS
        + ROW_WATCHDOG_WINDOW_SECS
        + 2 * ROW_TREE_KILL_CEILING_SECS
    )
    assert required <= ROW_PER_TEST_TIMEOUT_SECS, (
        f'ROW_PER_TEST_TIMEOUT_SECS ({ROW_PER_TEST_TIMEOUT_SECS}) does not cover '
        f'the bounded worst case at the MAX discovery ceiling ({required}) -- '
        f'both discovery waits, one full watchdog window, and both '
        f'kill-confirmation waits run to their ceiling'
    )
