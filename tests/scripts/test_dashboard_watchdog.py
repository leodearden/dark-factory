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


# ---------------------------------------------------------------------------
# probe_health()
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal context-manager stand-in for urllib.request.urlopen's return value."""

    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _patch_urlopen(monkeypatch, mod, outcome):
    """Point *mod*'s urlopen at a fake; return the list of (args, kwargs) calls.

    *outcome* is either a response object to return or an exception to raise.
    """
    calls: list[tuple[tuple, dict]] = []

    def fake_urlopen(*args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_probe_health_true_on_200(monkeypatch):
    mod = _load_watchdog()
    _patch_urlopen(monkeypatch, mod, _FakeResponse(200))
    assert mod.probe_health() is True


def test_probe_health_false_on_503(monkeypatch):
    """A 503 is a FAILURE here — the DELIBERATE inversion of
    scripts/orchestrator-watchdog.py's probe_health, which returns True on 503.

    That probe guards fused-memory, a shared server whose /health returns 503
    when a backing STORE is degraded: restarting the process would not fix a
    down store, so any HTTP response there means "the event loop is alive".

    This probe targets a bare ``{'status': 'ok'}`` handler that performs no
    I/O at all. If it answers anything other than 200, the app router itself
    is broken — which a restart CAN fix. The two scripts sit side by side in
    scripts/ and look almost identical, so do NOT "fix" one to match the other.
    """
    mod = _load_watchdog()
    import urllib.error

    err = urllib.error.HTTPError(mod.PROBE_URL, 503, "Service Unavailable", {}, None)  # type: ignore[arg-type]
    _patch_urlopen(monkeypatch, mod, err)
    assert mod.probe_health() is False


def test_probe_health_false_on_404_httperror(monkeypatch):
    """A 404 (the shallow route missing entirely) is a failure."""
    mod = _load_watchdog()
    import urllib.error

    err = urllib.error.HTTPError(mod.PROBE_URL, 404, "Not Found", {}, None)  # type: ignore[arg-type]
    _patch_urlopen(monkeypatch, mod, err)
    assert mod.probe_health() is False


def test_probe_health_false_on_non_200_response_object(monkeypatch):
    """A non-200 delivered as a RESPONSE rather than an exception is also a
    failure — pins the ``status == 200`` check itself, not just the except
    branches. (A custom opener, or a redirect handler that returned a 3xx,
    reaches probe_health this way.)"""
    mod = _load_watchdog()
    _patch_urlopen(monkeypatch, mod, _FakeResponse(204))
    assert mod.probe_health() is False


def test_probe_health_false_on_urlerror(monkeypatch):
    """Connection refused — the dashboard process is not listening at all."""
    mod = _load_watchdog()
    import urllib.error

    _patch_urlopen(monkeypatch, mod, urllib.error.URLError(ConnectionRefusedError(111)))
    assert mod.probe_health() is False


def test_probe_health_false_on_timeout(monkeypatch):
    """A probe that never returns within PROBE_TIMEOUT is a failure.

    ``socket.timeout`` is an alias of the builtin TimeoutError on 3.10+; both
    spellings are exercised so neither regresses.
    """
    import socket

    mod = _load_watchdog()
    _patch_urlopen(monkeypatch, mod, socket.timeout("timed out"))
    assert mod.probe_health() is False

    mod2 = _load_watchdog()
    _patch_urlopen(monkeypatch, mod2, TimeoutError("timed out"))
    assert mod2.probe_health() is False


def test_probe_health_requests_exactly_probe_url_with_probe_timeout(monkeypatch):
    """The probe must hit PROBE_URL with an explicit bounded timeout.

    Without ``timeout=`` urlopen inherits the (unbounded) global default socket
    timeout, so a hung dashboard would stall the oneshot past the next tick.
    """
    mod = _load_watchdog()
    calls = _patch_urlopen(monkeypatch, mod, _FakeResponse(200))
    mod.probe_health()

    assert len(calls) == 1
    args, kwargs = calls[0]
    requested = args[0] if args else kwargs.get("url")
    assert requested == mod.PROBE_URL
    assert kwargs.get("timeout") == mod.PROBE_TIMEOUT


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
