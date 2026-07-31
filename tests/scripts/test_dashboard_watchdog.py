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


# ---------------------------------------------------------------------------
# Persisted state — the "fresh process every tick" contract
# ---------------------------------------------------------------------------

DEFAULT_STATE = {"streak": 0, "restarts": [], "ceiling_open": False}


@pytest.fixture()
def state_env(monkeypatch, tmp_path):
    """Point DASHBOARD_WATCHDOG_STATE at a tmp file and yield its path.

    STATE_PATH is resolved at module-import time, so this must be applied
    BEFORE any _load_watchdog() call in the test body.
    """
    path = tmp_path / "wd" / "state.json"
    monkeypatch.setenv("DASHBOARD_WATCHDOG_STATE", str(path))
    return path


def test_load_state_missing_file_returns_documented_defaults(state_env):
    """A first-ever tick must not crash and must not create anything.

    The oneshot runs before data/dashboard-watchdog/ exists on a fresh
    checkout; reading is a pure query, so nothing is written until an actual
    state change happens.
    """
    mod = _load_watchdog()
    assert mod.load_state() == DEFAULT_STATE
    assert not state_env.exists()
    assert not state_env.parent.exists(), "load_state() must not create the state dir"


def test_state_round_trips_across_a_fresh_module_load(state_env):
    """THE contract this whole design exists for.

    Each timer tick is a fresh ``Type=oneshot`` process, so an in-memory
    counter would reset to 0 every 30 seconds and the streak gate would never
    reach FAIL_STREAK. Writing with one module instance and reading with a
    SECOND, independently-loaded instance is what actually proves that.
    """
    writer = _load_watchdog()
    writer.save_state({"streak": 2, "restarts": [1753900000, 1753900100], "ceiling_open": True})

    reader = _load_watchdog()
    assert reader is not writer
    loaded = reader.load_state()
    assert loaded["streak"] == 2
    assert loaded["restarts"] == [1753900000, 1753900100]
    assert loaded["ceiling_open"] is True


def test_load_state_corrupt_json_returns_defaults(state_env):
    """A truncated write (power cut mid-tick) must not wedge the watchdog."""
    state_env.parent.mkdir(parents=True, exist_ok=True)
    state_env.write_text('{"streak": 2, "restarts": [17539', encoding="utf-8")

    mod = _load_watchdog()
    assert mod.load_state() == DEFAULT_STATE


def test_load_state_valid_json_but_not_a_dict_returns_defaults(state_env):
    """Valid JSON of the wrong SHAPE is a distinct failure from corrupt JSON:
    json.load succeeds, so only an explicit isinstance check catches it."""
    state_env.parent.mkdir(parents=True, exist_ok=True)
    state_env.write_text("[]", encoding="utf-8")

    mod = _load_watchdog()
    assert mod.load_state() == DEFAULT_STATE


@pytest.mark.parametrize(
    "payload",
    [
        '{}',
        '{"streak": "three", "restarts": [], "ceiling_open": false}',
        '{"streak": 1, "restarts": "nope", "ceiling_open": false}',
        '{"streak": 1, "restarts": [1, "x", null], "ceiling_open": false}',
        '{"streak": -5, "restarts": [], "ceiling_open": false}',
    ],
)
def test_load_state_normalises_missing_and_ill_typed_keys(state_env, payload):
    """A hand-edited state file must not be able to crash a tick.

    Every key is normalised to its declared type; a non-numeric entry inside
    ``restarts`` is dropped rather than poisoning the later ``now - epoch``
    arithmetic in the rolling-window prune.
    """
    state_env.parent.mkdir(parents=True, exist_ok=True)
    state_env.write_text(payload, encoding="utf-8")

    mod = _load_watchdog()
    st = mod.load_state()
    assert isinstance(st["streak"], int)
    assert st["streak"] >= 0
    assert isinstance(st["restarts"], list)
    assert all(isinstance(e, int) for e in st["restarts"])
    assert isinstance(st["ceiling_open"], bool)


def test_save_state_leaves_no_temp_file_behind(state_env):
    """The write is atomic (tmp + os.replace), so the state dir holds exactly
    the state file afterwards — a leftover temp would accumulate one file per
    tick, i.e. 2880 files a day."""
    mod = _load_watchdog()
    mod.save_state({"streak": 1, "restarts": [], "ceiling_open": False})

    entries = sorted(p.name for p in state_env.parent.iterdir())
    assert entries == [state_env.name], f"unexpected leftovers: {entries}"


def test_save_state_creates_the_state_directory(state_env):
    """data/dashboard-watchdog/ does not exist on a fresh checkout."""
    assert not state_env.parent.exists()
    mod = _load_watchdog()
    mod.save_state(DEFAULT_STATE)
    assert state_env.exists()


def test_save_state_is_fail_soft_when_the_path_is_unwritable(monkeypatch, tmp_path):
    """A state path that is a DIRECTORY cannot be written — the tick must
    warn and continue, not raise. An unwritable state file degrades the
    watchdog to stateless (it stops restarting), which is the safe direction;
    crashing the oneshot would put the unit into 'failed'."""
    blocked = tmp_path / "state.json"
    blocked.mkdir()
    monkeypatch.setenv("DASHBOARD_WATCHDOG_STATE", str(blocked))

    mod = _load_watchdog()
    mod.save_state({"streak": 1, "restarts": [], "ceiling_open": False})  # must not raise


# ---------------------------------------------------------------------------
# Tick harness — simulates consecutive timer firings
# ---------------------------------------------------------------------------

#: How long ago the dashboard unit activated, for ticks that should be well
#: clear of the startup-grace window. Any value >> GRACE_SECS works.
OUTSIDE_GRACE_SECS = 10_000


class _TickRecorder:
    """Records what a run of simulated ticks actually did.

    ``actuations`` deliberately excludes the read-only ``systemctl show``
    query (kept separately in ``queries``): "zero systemctl invocations" in
    the behavioural boundary cases means zero ACTUATIONS — the grace gate
    legitimately queries the unit's activation timestamp on every tick.
    """

    def __init__(self) -> None:
        self.actuations: list[list[str]] = []
        self.queries: list[list[str]] = []
        self.escalations: list[list[str]] = []
        self.states: list[dict] = []

    @property
    def restarts(self) -> list[list[str]]:
        return [a for a in self.actuations if "restart" in a]

    @property
    def streaks(self) -> list[int]:
        return [s["streak"] for s in self.states]


def _run_ticks(
    monkeypatch,
    probe_results,
    activated_secs_ago: int | None = OUTSIDE_GRACE_SECS,
    seed_state: dict | None = None,
    recorder: _TickRecorder | None = None,
) -> _TickRecorder:
    """Run one simulated timer tick per entry in *probe_results*.

    Each tick gets a FRESH ``_load_watchdog()`` — that is the whole point:
    the timer fires a new ``Type=oneshot`` process every 30s, so anything the
    watchdog needs to remember has to round-trip through the state file.

    *probe_results* is an iterable of bools; True means the probe answers 200.
    *activated_secs_ago* seeds the unit's ActiveEnterTimestamp (None makes the
    ``systemctl show`` query fail, i.e. activation undeterminable).
    Pass *recorder* to accumulate across several _run_ticks calls.
    """
    import subprocess as _sp
    import time as _time

    rec = recorder if recorder is not None else _TickRecorder()

    if seed_state is not None:
        _load_watchdog().save_state(seed_state)

    active_enter = (
        None if activated_secs_ago is None else int(_time.time()) - activated_secs_ago
    )

    def fake_run(argv, *args, **kwargs):
        argv_list = list(argv)
        if argv_list and argv_list[0] == "systemd-cat":
            return _sp.CompletedProcess(argv_list, 0, stdout="", stderr="")
        if argv_list[:2] == ["systemctl", "--user"]:
            if "show" in argv_list:
                rec.queries.append(argv_list)
                if active_enter is None:
                    return _sp.CompletedProcess(argv_list, 1, stdout="", stderr="")
                return _sp.CompletedProcess(
                    argv_list,
                    0,
                    stdout=f"ActiveEnterTimestamp=@{active_enter}\n",
                    stderr="",
                )
            rec.actuations.append(argv_list)
            return _sp.CompletedProcess(argv_list, 0, stdout="", stderr="")
        # Anything else is the `uv run ... escalation submit ...` invocation.
        rec.escalations.append(argv_list)
        return _sp.CompletedProcess(argv_list, 0, stdout="", stderr="")

    monkeypatch.setattr(_sp, "run", fake_run)

    for healthy in probe_results:
        mod = _load_watchdog()
        outcome = _FakeResponse(200) if healthy else urllib_error_503(mod)
        _patch_urlopen(monkeypatch, mod, outcome)
        mod.tick()
        rec.states.append(_load_watchdog().load_state())

    return rec


def urllib_error_503(mod):
    import urllib.error

    return urllib.error.HTTPError(mod.PROBE_URL, 503, "Service Unavailable", {}, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# B1 — healthy steady state
# ---------------------------------------------------------------------------


def test_b1_healthy_steady_state_never_actuates(monkeypatch, state_env):
    """30 consecutive healthy ticks (15 minutes of real time) touch nothing.

    The floor the whole rewrite has to clear: the retired inline shell would
    restart the dashboard the first time the deep endpoint was slow, so a
    watchdog that is quiet while the service is healthy is the primary signal.
    """
    rec = _run_ticks(monkeypatch, [True] * 30)

    assert rec.actuations == [], f"healthy ticks actuated systemctl: {rec.actuations}"
    assert rec.escalations == [], f"healthy ticks filed escalations: {rec.escalations}"
    assert rec.streaks == [0] * 30
    assert all(s["restarts"] == [] for s in rec.states)
    assert all(s["ceiling_open"] is False for s in rec.states)


def test_b1_healthy_tick_clears_a_pre_existing_streak(monkeypatch, state_env):
    """A recovered service resets the counter — hysteresis is CONSECUTIVE
    failures, not cumulative ones. Without this, failures separated by hours
    of health would eventually add up to a restart."""
    rec = _run_ticks(
        monkeypatch,
        [True],
        seed_state={"streak": 2, "restarts": [], "ceiling_open": False},
    )

    assert rec.streaks == [0]
    assert rec.actuations == []


# ---------------------------------------------------------------------------
# B2 — a single transient miss must not restart anything
# ---------------------------------------------------------------------------


def test_b2_single_transient_miss_never_restarts(monkeypatch, state_env):
    """success, success, FAIL, success, success → nothing happens.

    THE regression this task exists to prevent. The retired inline shell
    restarted on exactly this sequence, which on 2026-07-30 turned a
    momentarily-slow dashboard into 192 restarts in 3 hours (~27% downtime).
    """
    rec = _run_ticks(monkeypatch, [True, True, False, True, True])

    assert rec.actuations == [], f"a single miss actuated systemctl: {rec.actuations}"
    assert rec.escalations == []
    # The failing tick is index 2; the streak must be exactly 1 there, and
    # back to 0 once the service answers again.
    assert rec.streaks == [0, 0, 1, 0, 0]
    assert all(s["restarts"] == [] for s in rec.states)


def test_b2_streak_survives_the_process_boundary(monkeypatch, state_env):
    """Two consecutive misses accumulate to 2 — across two separate module
    loads. A streak that reset every tick would look identical to B2 from the
    outside (still no restart), so this asserts the counter really advances."""
    rec = _run_ticks(monkeypatch, [False, False])

    assert rec.streaks == [1, 2]
    assert rec.actuations == [], "acted before FAIL_STREAK consecutive misses"


def test_b2_alternating_failures_never_reach_the_gate(monkeypatch, state_env):
    """FAIL, ok, FAIL, ok, FAIL, ok … forever — a flapping-but-serving
    dashboard. Cumulative failures far exceed FAIL_STREAK, but no CONSECUTIVE
    run ever does, so the watchdog stays quiet."""
    rec = _run_ticks(monkeypatch, [False, True] * 10)

    assert rec.actuations == []
    assert max(rec.streaks) == 1


# ---------------------------------------------------------------------------
# B3 — sustained outage restarts exactly once per completed streak
# ---------------------------------------------------------------------------


def test_b3_sustained_outage_restarts_on_the_streak_th_tick(monkeypatch, state_env):
    """Exactly ONE restart, and it fires on the FAIL_STREAK-th tick — not earlier.

    Detection latency is therefore FAIL_STREAK × the timer's 30s cadence ≈ 90s,
    which is the number the unit-file comment documents.
    """
    mod_consts = _load_watchdog()
    rec = _run_ticks(monkeypatch, [False] * mod_consts.FAIL_STREAK)

    assert len(rec.restarts) == 1, f"expected 1 restart, got {rec.restarts}"
    # It fired on the last tick: the streak reset to 0 only at the end, and no
    # restart epoch was recorded before then.
    assert rec.streaks == [1, 2, 0][: mod_consts.FAIL_STREAK]
    assert [len(s["restarts"]) for s in rec.states] == [0, 0, 1][
        : mod_consts.FAIL_STREAK
    ]

    restart_argv = rec.restarts[0]
    assert restart_argv[-1] == mod_consts.DASHBOARD_UNIT
    assert "--user" in restart_argv, "must act on the USER manager, not the system one"
    assert rec.escalations == [], "a first restart must not file an escalation"


def test_b3_reset_failed_precedes_the_restart(monkeypatch, state_env):
    """``reset-failed`` clears StartLimitBurst first.

    Without it, a unit that has already exhausted its start limit silently
    ignores the restart — the watchdog would log "restart issued", the streak
    would reset, and nothing would actually happen. That is a fail-soft hole
    that makes the gate LOOK like it fired.
    """
    mod = _load_watchdog()
    rec = _run_ticks(monkeypatch, [False] * mod.FAIL_STREAK)

    verbs = [a[2] for a in rec.actuations]
    assert verbs == ["reset-failed", "restart"], f"unexpected actuation order: {verbs}"
    assert all(a[-1] == mod.DASHBOARD_UNIT for a in rec.actuations)


def test_b3_second_streak_restarts_exactly_once_more(monkeypatch, state_env):
    """Invariant I3: the restart is per COMPLETED STREAK, never per tick.

    2 × FAIL_STREAK failing ticks produce exactly 2 restarts — not 6. A
    per-tick loop is exactly what the incident was.
    """
    mod = _load_watchdog()
    rec = _run_ticks(monkeypatch, [False] * (2 * mod.FAIL_STREAK))

    assert len(rec.restarts) == 2, f"expected 2 restarts, got {len(rec.restarts)}"
    assert len(rec.states[-1]["restarts"]) == 2, "both restart epochs must be recorded"


def test_b3_restart_epoch_is_recorded_for_the_rate_window(monkeypatch, state_env):
    """Each restart appends a plausible epoch — the rolling ceiling window is
    measured against these, so an unrecorded restart would make the storm
    escape unreachable."""
    import time

    before = int(time.time())
    mod = _load_watchdog()
    rec = _run_ticks(monkeypatch, [False] * mod.FAIL_STREAK)
    after = int(time.time())

    epochs = rec.states[-1]["restarts"]
    assert len(epochs) == 1
    assert before <= epochs[0] <= after


def test_b3_streak_resets_after_the_restart(monkeypatch, state_env):
    """Post-restart the counter starts over, so the next restart needs another
    FAIL_STREAK consecutive misses — hysteresis applies to repeat actuation
    too, not just the first one."""
    mod = _load_watchdog()
    rec = _run_ticks(monkeypatch, [False] * (mod.FAIL_STREAK + 1))

    assert rec.streaks[mod.FAIL_STREAK - 1] == 0, "streak not reset after restart"
    assert rec.streaks[mod.FAIL_STREAK] == 1, "next miss must start a fresh streak"
    assert len(rec.restarts) == 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
