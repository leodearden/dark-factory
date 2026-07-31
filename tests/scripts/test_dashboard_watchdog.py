"""Unit tests for scripts/dashboard-watchdog.py.

The watchdog module has a hyphenated filename so it cannot be imported via
``import dashboard_watchdog``.  We use importlib to load it by file path —
the same idiom as tests/scripts/test_orchestrator_watchdog.py.

No live systemd runtime and no running dashboard are needed — every
``subprocess.run`` and ``urllib.request.urlopen`` call is monkeypatched.

This file runs in the ``scripts/`` verify lane
(scripts/orchestrator.yaml: ``uv run --project shared pytest tests/scripts/``),
whose environment is stdlib-only: ``shared`` does NOT depend on ``escalation``,
so nothing here may import it.  The storm escape's literal acceptance signal
("exactly ONE born-at-L2 record on disk") is asserted against the REAL
``escalation.submit`` writer in
dashboard/tests/test_dashboard_watchdog_storm_escape.py instead.
"""

import importlib.util
import pathlib
import types

import pytest  # pyright: ignore[reportMissingImports]

REPO_ROOT = pathlib.Path(__file__).parents[2]
WATCHDOG_PATH = REPO_ROOT / "scripts" / "dashboard-watchdog.py"


def _load_watchdog() -> types.ModuleType:
    """Load scripts/dashboard-watchdog.py as a module (hyphenated name).

    Re-invoking this is how the tests model the oneshot contract: the timer
    fires a FRESH process every 30s, so a fresh module load per simulated tick
    is what proves an in-memory counter would be useless and the streak really
    does have to round-trip through the state file.
    """
    spec = importlib.util.spec_from_file_location("dashboard_watchdog", WATCHDOG_PATH)
    assert spec is not None, f"Could not build spec from {WATCHDOG_PATH}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Contract constants (plans/dashboard-availability-prd.md §Contract)
# ---------------------------------------------------------------------------


def test_probe_url_is_the_shallow_endpoint():
    """The probe targets /api/health, the bare {'status': 'ok'} handler.

    dashboard/src/dashboard/app.py defines /api/health with no DB access,
    immediately above /healthz which performs three _DB_PROBE_TIMEOUT=5.0 DB
    probes.  Probing the deep endpoint is what made a merely-slow dashboard
    look dead during the 2026-07-30 restart storm.
    """
    mod = _load_watchdog()
    assert mod.PROBE_URL == "http://127.0.0.1:8080/api/health"


def test_fail_streak_is_three():
    """FAIL_STREAK is the MANDATED constant name (the sidecar delivered_check
    greps ``scripts/`` for it), and 3 consecutive misses × the timer's 30s
    cadence sets the ~90s sustained-outage detection latency."""
    mod = _load_watchdog()
    assert mod.FAIL_STREAK == 3


def test_grace_secs_is_sixty():
    mod = _load_watchdog()
    assert mod.GRACE_SECS == 60


def test_rate_ceiling_constants():
    """At most MAX_RESTARTS restarts inside a rolling RATE_WINDOW_SECS."""
    mod = _load_watchdog()
    assert mod.MAX_RESTARTS == 3
    assert mod.RATE_WINDOW_SECS == 3600


def test_probe_timeout_is_five_seconds():
    mod = _load_watchdog()
    assert mod.PROBE_TIMEOUT == 5


def test_dashboard_unit_name():
    mod = _load_watchdog()
    assert mod.DASHBOARD_UNIT == "dark-factory-dashboard.service"


def test_healthz_appears_nowhere_in_the_source():
    """The 2026-07-30 incident regression, pinned at the source level.

    The retired inline shell probed ``/healthz`` — three 5s DB probes — and
    restarted on a single miss.  The deep endpoint must not reappear anywhere
    in the watchdog: not in a constant, not in a fallback, not in a comment
    that a future reader could copy back into the probe.
    """
    source = WATCHDOG_PATH.read_text(encoding="utf-8")
    assert "/healthz" not in source, (
        "'/healthz' appears in scripts/dashboard-watchdog.py. The deep "
        "DB-probing endpoint is what turned a slow dashboard into 192 "
        "restarts in 3h on 2026-07-30; the watchdog probes /api/health only."
    )


def test_constants_are_not_accidentally_aliased():
    """FAIL_STREAK and MAX_RESTARTS are both 3 today but mean different things.

    Guards against a future 'DRY' edit collapsing them into one name: the
    streak gate counts consecutive failed PROBES, the ceiling counts RESTARTS
    inside a rolling window.  Changing one must not silently change the other.
    """
    mod = _load_watchdog()
    assert "FAIL_STREAK" in vars(mod)
    assert "MAX_RESTARTS" in vars(mod)
    source = WATCHDOG_PATH.read_text(encoding="utf-8")
    assert "MAX_RESTARTS = FAIL_STREAK" not in source
    assert "FAIL_STREAK = MAX_RESTARTS" not in source


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
