"""Fleet-wide guard: a declared restart-backoff cap must actually engage.

This module owns the RELATIONAL invariant for the whole repo — every file
that declares ``RestartMaxDelaySec=`` must pair it with ``RestartSteps=``,
because systemd parses the cap, logs "Service has RestartMaxDelaySec= but no
RestartSteps= setting. Ignoring." at unit load, and then discards it.  The
backoff the unit's own comment advertises never happens and nothing in the
file's text reveals that.

The assertion itself is NOT defined here.  It was written for task 3333
(dashboard unit + template) as a private helper inside
tests/scripts/test_dashboard_service_template.py, and task 3408 lifted it
into the importable tests/scripts/systemd_unit_invariants.py so that this
fleet-wide sweep and the dashboard suite share ONE implementation rather
than two copies that can drift apart.  test_restart_backoff_guard_rejects_
ineffective_units in the dashboard module remains the helper's negative-case
guard; this module supplies its fleet-wide application.
"""
import importlib

import systemd_unit_invariants


def test_restart_backoff_helper_is_shared_not_duplicated() -> None:
    """The dashboard suite must USE the shared helper, not its own fork of it.

    A "both modules define a function of this name" test would pass happily
    against two independently-drifting copies, which is precisely the failure
    this task exists to repair: per-unit string asserts forked from the real
    invariant and stayed green for months while the directive they guarded was
    inert.  So the check is object IDENTITY — a future author who re-inlines a
    private copy into either module fails here immediately, at the moment of
    forking, rather than years later when one copy has quietly stopped
    catching the defect.
    """
    for name in ("restart_directive", "assert_restart_backoff_effective"):
        assert callable(getattr(systemd_unit_invariants, name, None)), (
            f"systemd_unit_invariants must export a callable {name!r}; it is "
            "the single shared implementation of the restart-backoff invariant."
        )

    dashboard = importlib.import_module("test_dashboard_service_template")

    assert (
        dashboard._assert_restart_backoff_effective
        is systemd_unit_invariants.assert_restart_backoff_effective
    ), (
        "test_dashboard_service_template._assert_restart_backoff_effective is "
        "not the shared systemd_unit_invariants.assert_restart_backoff_effective. "
        "The invariant has been forked into a private copy; import the shared "
        "one instead so both modules cannot drift apart."
    )
    assert (
        dashboard._restart_directive is systemd_unit_invariants.restart_directive
    ), (
        "test_dashboard_service_template._restart_directive is not the shared "
        "systemd_unit_invariants.restart_directive. The directive reader "
        "encodes measured systemd 255.4 last-wins scalar semantics; a private "
        "copy of it will drift out of agreement with the assertion that uses it."
    )
