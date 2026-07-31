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


# ---------------------------------------------------------------------------
# B5 — startup grace
#
# (a) _unit_active_enter_epoch() parsing, against verbatim systemctl output.
# (b) The behavioural gate: a freshly-activated unit is not probed at all.
# ---------------------------------------------------------------------------


def _patch_systemctl_show(monkeypatch, mod, returncode: int, stdout: str):
    """Fake ``subprocess.run`` for the ``systemctl show`` query only.

    Returns the list of recorded argvs so the test can assert the query
    actually asked for the field the task mandates.
    """
    import subprocess as _sp

    calls: list[list[str]] = []

    def fake_run(argv, *args, **kwargs):
        argv_list = list(argv)
        calls.append(argv_list)
        return _sp.CompletedProcess(argv_list, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(_sp, "run", fake_run)
    return calls


def test_active_enter_epoch_parses_the_unix_timestamp(monkeypatch):
    """``--timestamp=unix`` yields ``@<epoch>``; the leading @ is stripped."""
    mod = _load_watchdog()
    _patch_systemctl_show(
        monkeypatch, mod, 0, "ActiveEnterTimestamp=@1753900000\n"
    )
    assert mod._unit_active_enter_epoch(mod.DASHBOARD_UNIT) == 1753900000


def test_active_enter_epoch_queries_the_mandated_field(monkeypatch):
    """The argv must ask for ActiveEnterTimestamp with ``--timestamp=unix``.

    ActiveEnterTimestamp — not ExecMainStartTimestamp — is the field the task
    mandates: it marks when the unit reached 'active', i.e. when the grace
    window for THIS activation began.  ``--timestamp=unix`` is what makes the
    value a timezone-independent integer instead of a locale-formatted string
    that int() would reject (silently disabling the grace gate).
    """
    mod = _load_watchdog()
    calls = _patch_systemctl_show(
        monkeypatch, mod, 0, "ActiveEnterTimestamp=@1753900000\n"
    )

    mod._unit_active_enter_epoch(mod.DASHBOARD_UNIT)

    assert len(calls) == 1
    argv = calls[0]
    assert argv[:3] == ["systemctl", "--user", "show"]
    assert "--timestamp=unix" in argv
    assert "ActiveEnterTimestamp" in argv
    assert mod.DASHBOARD_UNIT in argv


def test_active_enter_epoch_never_activated_is_none(monkeypatch):
    """``@0`` is systemd's 'never activated' sentinel, not the 1970 epoch.

    Read as an epoch it would put activation 56 years in the past — grace
    would never apply, and the very first tick after a boot-time failure to
    start could restart a unit that had not even tried to come up yet.
    """
    mod = _load_watchdog()
    _patch_systemctl_show(monkeypatch, mod, 0, "ActiveEnterTimestamp=@0\n")
    assert mod._unit_active_enter_epoch(mod.DASHBOARD_UNIT) is None


def test_active_enter_epoch_nonzero_returncode_is_none(monkeypatch):
    """systemctl failing (no user manager, unknown unit) → undeterminable."""
    mod = _load_watchdog()
    _patch_systemctl_show(monkeypatch, mod, 1, "")
    assert mod._unit_active_enter_epoch(mod.DASHBOARD_UNIT) is None


@pytest.mark.parametrize(
    "stdout",
    [
        "ActiveEnterTimestamp=Thu 2026-07-30 11:02:03 BST\n",  # --timestamp=unix ignored
        "ActiveEnterTimestamp=\n",  # empty value
        "ActiveEnterTimestamp=@notanumber\n",
        "",  # no output at all
        "no-equals-sign-here\n",
    ],
)
def test_active_enter_epoch_unparseable_is_none(monkeypatch, stdout):
    """Anything that is not a clean integer is None, never a guess."""
    mod = _load_watchdog()
    _patch_systemctl_show(monkeypatch, mod, 0, stdout)
    assert mod._unit_active_enter_epoch(mod.DASHBOARD_UNIT) is None


def test_active_enter_epoch_is_fail_soft_on_oserror(monkeypatch):
    """A missing systemctl binary must not crash the oneshot."""
    import subprocess as _sp

    mod = _load_watchdog()

    def boom(*args, **kwargs):
        raise FileNotFoundError("systemctl")

    monkeypatch.setattr(_sp, "run", boom)
    assert mod._unit_active_enter_epoch(mod.DASHBOARD_UNIT) is None


def test_b5_freshly_activated_unit_is_not_probed_or_restarted(monkeypatch, state_env):
    """A unit that activated 10s ago is inside GRACE_SECS: no probe, no action.

    uvicorn needs a few seconds to bind and mount the SPA, so probing during
    startup would fail for a perfectly healthy service — and, with a streak
    carried over from before the restart, could restart it again immediately.
    That is the flap the grace window exists to break.
    """
    rec = _run_ticks(monkeypatch, [False] * 5, activated_secs_ago=10)

    assert rec.actuations == [], f"restarted inside the grace window: {rec.actuations}"
    assert rec.escalations == []
    assert rec.streaks == [0] * 5, "a fresh activation must invalidate the streak"


def test_b5_grace_resets_a_streak_carried_across_a_restart(monkeypatch, state_env):
    """The pre-restart streak must not survive into the new activation.

    Without the reset, the two ticks left over from the streak that CAUSED the
    restart would combine with the first post-grace miss to restart again after
    a single failed probe — reintroducing the incident's per-tick behaviour.
    """
    rec = _run_ticks(
        monkeypatch,
        [False],
        activated_secs_ago=5,
        seed_state={"streak": 2, "restarts": [], "ceiling_open": False},
    )

    assert rec.streaks == [0]
    assert rec.actuations == []


def test_b5_grace_is_a_window_not_a_permanent_suppression(monkeypatch, state_env):
    """Once the unit is GRACE_SECS + 120s old, the same failing ticks DO act.

    Paired deliberately with the test above: together they prove the gate is
    bounded in time.  A grace check that never expired would be a watchdog
    that never watches.
    """
    mod = _load_watchdog()
    rec = _run_ticks(
        monkeypatch,
        [False] * mod.FAIL_STREAK,
        activated_secs_ago=mod.GRACE_SECS + 120,
    )

    assert len(rec.restarts) == 1, f"grace suppressed a real outage: {rec.actuations}"


def test_b5_grace_skips_the_probe_entirely(monkeypatch, state_env):
    """Invariant I5: never act on a probe that was not run — so don't run one.

    Inside grace the watchdog cannot use the answer either way, so it must not
    ask — otherwise every 30s tick during a slow startup burns a 5s socket
    timeout for an answer that is discarded.

    The fake RECORDS rather than raises: probe_health() catches Exception by
    design (any failure counts as a failed probe), so an exploding urlopen
    would be swallowed and this test would pass with no grace gate at all.
    """
    import subprocess as _sp
    import time as _time

    active_enter = int(_time.time()) - 5

    def fake_run(argv, *args, **kwargs):
        argv_list = list(argv)
        if argv_list and argv_list[0] == "systemd-cat":
            return _sp.CompletedProcess(argv_list, 0, stdout="", stderr="")
        return _sp.CompletedProcess(
            argv_list, 0, stdout=f"ActiveEnterTimestamp=@{active_enter}\n", stderr=""
        )

    monkeypatch.setattr(_sp, "run", fake_run)

    mod = _load_watchdog()
    probes = _patch_urlopen(monkeypatch, mod, _FakeResponse(200))

    mod.tick()

    assert probes == [], "probed the dashboard inside the startup grace window"


def test_b5_undeterminable_activation_does_not_apply_grace(monkeypatch, state_env):
    """``systemctl show`` failing means grace CANNOT be determined, so it does
    not apply — the watchdog proceeds to probe.

    Mirrors orchestrator-watchdog.py's documented convention (None = "cannot
    determine, don't guess").  Failing the other way would let a broken
    systemctl query permanently disarm the watchdog: the probe would never run
    and a genuinely dead dashboard would stay dead forever.
    """
    mod = _load_watchdog()
    rec = _run_ticks(
        monkeypatch, [False] * mod.FAIL_STREAK, activated_secs_ago=None
    )

    assert len(rec.restarts) == 1
    assert rec.queries, "the grace gate must still have asked for the timestamp"


# ---------------------------------------------------------------------------
# B6 — slow-but-alive (the 2026-07-30 incident regression) + unit-file contract
# ---------------------------------------------------------------------------

WATCHDOG_UNIT_PATH = REPO_ROOT / "dashboard" / "dark-factory-dashboard-watchdog.service"
WATCHDOG_TIMER_PATH = REPO_ROOT / "dashboard" / "dark-factory-dashboard-watchdog.timer"

#: The deep, DB-touching endpoint the retired inline shell probed. Assembled
#: rather than written literally so this test file's own text cannot be
#: mistaken for a reintroduction when the source-absence pin is grepped.
DEEP_ENDPOINT = "/health" + "z"


def test_b6_slow_deep_endpoint_never_restarts_a_serving_dashboard(
    monkeypatch, state_env
):
    """THE incident. A dashboard whose DB probes are slow is still SERVING.

    On 2026-07-30 the deep endpoint's three 5s DB probes exceeded the inline
    shell's ``curl --max-time 5`` while the app itself answered fine, so the
    watchdog restarted a healthy service 192 times in 3 hours (~27% downtime),
    and every restart made the next probe more likely to time out.

    The fake dispatches on URL: the shallow endpoint answers 200 immediately,
    the deep one raises a 503 after a simulated delay. Ten ticks must actuate
    nothing — and the deep endpoint must never even be REQUESTED.
    """
    import subprocess as _sp
    import time as _time
    import urllib.error

    active_enter = int(_time.time()) - OUTSIDE_GRACE_SECS
    actuations: list[list[str]] = []
    requested: list[str] = []

    def fake_run(argv, *args, **kwargs):
        argv_list = list(argv)
        if argv_list and argv_list[0] == "systemd-cat":
            return _sp.CompletedProcess(argv_list, 0, stdout="", stderr="")
        if "show" in argv_list:
            return _sp.CompletedProcess(
                argv_list,
                0,
                stdout=f"ActiveEnterTimestamp=@{active_enter}\n",
                stderr="",
            )
        actuations.append(argv_list)
        return _sp.CompletedProcess(argv_list, 0, stdout="", stderr="")

    monkeypatch.setattr(_sp, "run", fake_run)

    for _ in range(10):
        mod = _load_watchdog()

        def dispatching_urlopen(url, *args, **kwargs):
            requested.append(url)
            if DEEP_ENDPOINT in url:
                # The slow DB probes: eventually a 503, well past any sane
                # client timeout. Simulated, not slept — the point is the
                # OUTCOME differs by endpoint, not the wall clock.
                raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)  # type: ignore[arg-type]
            return _FakeResponse(200)

        monkeypatch.setattr(mod.urllib.request, "urlopen", dispatching_urlopen)
        mod.tick()

    assert actuations == [], f"restarted a serving dashboard: {actuations}"
    assert requested, "the watchdog probed nothing at all"
    assert all(DEEP_ENDPOINT not in url for url in requested), (
        f"the watchdog requested the deep DB-probing endpoint: {requested}"
    )
    assert _load_watchdog().load_state()["streak"] == 0


def test_b6_unit_execstart_invokes_the_watchdog_script():
    """The supervision policy must live in a tested script, not in an sh -c.

    The retired ExecStart was a contract-in-prose (INV-1): probe target,
    timeout, and restart policy were all inside a shell string no test could
    reach, which is why a single-sample restarter shipped unnoticed.
    """
    unit = WATCHDOG_UNIT_PATH.read_text(encoding="utf-8")
    exec_lines = [ln for ln in unit.splitlines() if ln.startswith("ExecStart=")]

    assert len(exec_lines) == 1, f"expected exactly one ExecStart: {exec_lines}"
    assert "scripts/dashboard-watchdog.py" in exec_lines[0], exec_lines[0]


def test_b6_unit_has_no_inline_shell_probe():
    """``curl`` and the deep endpoint appear nowhere in the unit file.

    Pins both halves of the incident at the file level: the single-sample
    ``curl ... || systemctl restart`` and the deep endpoint it sampled.
    """
    unit = WATCHDOG_UNIT_PATH.read_text(encoding="utf-8")

    assert "curl" not in unit, "the inline curl probe is back in the unit file"
    assert DEEP_ENDPOINT not in unit, (
        "the deep DB-probing endpoint is back in the watchdog unit; the "
        "watchdog probes the shallow /api/health only"
    )
    assert "/bin/sh" not in unit


def test_b6_unit_is_oneshot_and_journal_logged():
    """Type=oneshot is what makes each tick a fresh process (hence the
    persisted state), and journal routing is the only way an operator can see
    why the watchdog did or did not act."""
    unit = WATCHDOG_UNIT_PATH.read_text(encoding="utf-8")

    assert "Type=oneshot" in unit
    assert "StandardOutput=journal" in unit
    assert "StandardError=journal" in unit


def test_b6_timer_cadence_is_thirty_seconds():
    """FAIL_STREAK × OnUnitActiveSec is the ~90s sustained-outage detection
    latency the unit-file comment documents. Changing the cadence silently
    changes that latency, so it is pinned here alongside FAIL_STREAK."""
    timer = WATCHDOG_TIMER_PATH.read_text(encoding="utf-8")
    mod = _load_watchdog()

    assert "OnUnitActiveSec=30" in timer
    assert mod.FAIL_STREAK * 30 == 90


def test_b6_timer_install_stanza_is_untouched():
    """This task is repo-side only: RE-ARMING the installed timer is task
    3289's job. Disarming it here — by dropping [Install] — would leave the
    dashboard unsupervised while looking like a fix."""
    timer = WATCHDOG_TIMER_PATH.read_text(encoding="utf-8")

    assert "[Install]" in timer
    assert "WantedBy=timers.target" in timer


# ---------------------------------------------------------------------------
# B4 part 1 — the rate ceiling and the storm escape, at argv level
#
# The literal acceptance signal ("exactly ONE born-at-L2 record on disk") is
# asserted against the REAL escalation.submit writer in
# dashboard/tests/test_dashboard_watchdog_storm_escape.py — this lane is
# stdlib-only and cannot import escalation.
# ---------------------------------------------------------------------------


@pytest.fixture()
def queue_env(monkeypatch, tmp_path):
    """Point DASHBOARD_WATCHDOG_QUEUE_DIR at a tmp dir; resolved at import."""
    qdir = tmp_path / "escalations"
    monkeypatch.setenv("DASHBOARD_WATCHDOG_QUEUE_DIR", str(qdir))
    return qdir


def _argv_value(argv: list[str], flag: str) -> str:
    """Return the value following *flag* in *argv* (fails loudly if absent)."""
    assert flag in argv, f"{flag} missing from {argv}"
    idx = argv.index(flag)
    assert idx + 1 < len(argv), f"{flag} has no value in {argv}"
    return argv[idx + 1]


def _at_ceiling_state(mod, age_secs: int = 60) -> dict:
    """State carrying MAX_RESTARTS restart epochs *age_secs* ago."""
    import time as _time

    now = int(_time.time())
    return {
        "streak": 0,
        "restarts": [now - age_secs] * mod.MAX_RESTARTS,
        "ceiling_open": False,
    }


def test_b4_ceiling_stops_restarting(monkeypatch, state_env, queue_env):
    """INV-4, the storm escape. At the ceiling the watchdog STOPS restarting.

    This is the whole point of the rewrite. MAX_RESTARTS restarts inside one
    RATE_WINDOW_SECS is proof that restarting is not fixing the problem, so
    continuing to restart only adds downtime — on 2026-07-30 it added ~27%.
    The watchdog escalates to a human instead and goes quiet.
    """
    mod = _load_watchdog()
    rec = _run_ticks(
        monkeypatch,
        [False] * mod.FAIL_STREAK,
        seed_state=_at_ceiling_state(mod),
    )

    assert rec.restarts == [], f"restarted at the ceiling: {rec.restarts}"
    assert len(rec.escalations) == 1, f"expected 1 escalation, got {rec.escalations}"
    assert rec.states[-1]["ceiling_open"] is True


def test_b4_ceiling_escalation_argv_is_the_born_at_l2_contract(
    monkeypatch, state_env, queue_env
):
    """The argv must satisfy escalation submit's ENFORCED argparse boundary.

    submit.py stamps level=2 directly, bypassing the server's chokepoint, so
    it restricts --severity to BORN_AT_L2_SEVERITIES and REJECTS an
    --agent-role without a harness-/orchestrator- sentinel prefix. Getting
    either wrong means the escalation is never filed — the watchdog would go
    quiet at the ceiling with nobody told, the worst of both behaviours.
    """
    mod = _load_watchdog()
    rec = _run_ticks(
        monkeypatch,
        [False] * mod.FAIL_STREAK,
        seed_state=_at_ceiling_state(mod),
    )

    argv = rec.escalations[0]
    assert "submit" in argv, f"the submit subcommand must lead the CLI args: {argv}"
    assert _argv_value(argv, "--queue-dir") == mod.ESCALATION_QUEUE_DIR
    assert _argv_value(argv, "--task") == "dashboard-watchdog-restart-ceiling"
    assert _argv_value(argv, "--severity") == "critical"
    assert _argv_value(argv, "--category") == "infra_issue"
    assert _argv_value(argv, "--agent-role") == "harness-dashboard-watchdog"
    assert _argv_value(argv, "--agent-role").startswith("harness-")


def test_b4_escalation_summary_carries_the_measured_evidence(
    monkeypatch, state_env, queue_env
):
    """A human reading only the summary must be able to act on it.

    Naming the unit, the restart COUNT and the WINDOW is what distinguishes
    "the dashboard is flapping and restarting does not fix it" from a generic
    'infra_issue' the steward has to go digging to understand.
    """
    mod = _load_watchdog()
    rec = _run_ticks(
        monkeypatch,
        [False] * mod.FAIL_STREAK,
        seed_state=_at_ceiling_state(mod),
    )

    argv = rec.escalations[0]
    summary = _argv_value(argv, "--summary")
    detail = _argv_value(argv, "--detail")

    assert mod.DASHBOARD_UNIT in summary
    assert str(mod.MAX_RESTARTS) in summary
    assert summary.strip()
    assert detail.strip(), "an empty --detail wastes the one L2 this episode gets"
    assert mod.PROBE_URL in detail, "the detail must name what was actually probed"


def test_b4_summary_is_one_line_within_the_record_contract(
    monkeypatch, state_env, queue_env
):
    """escalation.models.Escalation declares ``summary: str  # one-line``.

    Nothing ENFORCES that — there is no max_length on the field — so an
    embedded newline or a 2KB paragraph would be written to the queue happily
    and then wrap or truncate wherever a steward reads it (dashboard row,
    notification, digest). The contract is documentation-only, which is
    exactly the kind that rots, so it is pinned here instead.
    """
    mod = _load_watchdog()
    rec = _run_ticks(
        monkeypatch,
        [False] * mod.FAIL_STREAK,
        seed_state=_at_ceiling_state(mod),
    )

    summary = _argv_value(rec.escalations[0], "--summary")

    assert "\n" not in summary, f"summary is not one line: {summary!r}"
    assert len(summary) <= 400, f"summary is {len(summary)} chars: {summary!r}"


def test_b4_dedup_at_most_one_l2_per_ceiling_episode(
    monkeypatch, state_env, queue_env
):
    """Ten more failing ticks file NOTHING further.

    Without the dedup the storm escape would replace a restart storm with an
    escalation storm — 2 L2s a minute, forever, each one paging a human.
    """
    mod = _load_watchdog()
    rec = _run_ticks(
        monkeypatch,
        [False] * mod.FAIL_STREAK,
        seed_state=_at_ceiling_state(mod),
    )
    assert len(rec.escalations) == 1

    _run_ticks(monkeypatch, [False] * 10, recorder=rec)

    assert rec.restarts == [], f"restarted after the ceiling tripped: {rec.restarts}"
    assert len(rec.escalations) == 1, (
        f"re-filed while the ceiling was open: {len(rec.escalations)} escalations"
    )


def test_b4_window_is_rolling_not_a_lifetime_counter(monkeypatch, state_env, queue_env):
    """MAX_RESTARTS restarts LAST YEAR must not trip today's ceiling.

    A lifetime counter would eventually wedge the watchdog permanently on a
    service that misbehaves once a month — it would stop supervising and file
    an L2 about a problem that no longer exists.
    """
    mod = _load_watchdog()
    rec = _run_ticks(
        monkeypatch,
        [False] * mod.FAIL_STREAK,
        seed_state=_at_ceiling_state(mod, age_secs=mod.RATE_WINDOW_SECS + 600),
    )

    assert len(rec.restarts) == 1, "an aged-out window must restart normally"
    assert rec.escalations == [], "an aged-out window must not file anything"
    assert rec.states[-1]["ceiling_open"] is False
    # The stale epochs are pruned, so only the fresh restart remains.
    assert len(rec.states[-1]["restarts"]) == 1


def test_b4_ceiling_needs_max_restarts_not_one_fewer(monkeypatch, state_env, queue_env):
    """MAX_RESTARTS - 1 prior restarts still permits one more.

    Pins the boundary: an off-by-one here either escalates a service that had
    two ordinary blips (crying wolf) or permits a fourth restart the ceiling
    was written to prevent.
    """
    mod = _load_watchdog()
    below = _at_ceiling_state(mod)
    below["restarts"] = below["restarts"][:-1]

    rec = _run_ticks(monkeypatch, [False] * mod.FAIL_STREAK, seed_state=below)

    assert len(rec.restarts) == 1
    assert rec.escalations == []
    assert len(rec.states[-1]["restarts"]) == mod.MAX_RESTARTS


def test_b4_the_restart_that_reaches_the_ceiling_still_escalates_next_time(
    monkeypatch, state_env, queue_env
):
    """The Nth restart happens; the (N+1)th is what escalates instead.

    Drives two full streaks from MAX_RESTARTS - 1 prior restarts: the first
    completes the allowance, the second finds the window full and escalates.
    """
    mod = _load_watchdog()
    below = _at_ceiling_state(mod)
    below["restarts"] = below["restarts"][:-1]

    rec = _run_ticks(
        monkeypatch, [False] * (2 * mod.FAIL_STREAK), seed_state=below
    )

    assert len(rec.restarts) == 1, "the allowed restart did not fire"
    assert len(rec.escalations) == 1, "the over-ceiling streak did not escalate"
    assert rec.states[-1]["ceiling_open"] is True


# ---------------------------------------------------------------------------
# Ceiling-episode boundary — the trip must END, not wedge
#
# Invariant I4 says "no further restarts until an operator intervenes". Read
# literally that means a state file an operator has to delete BY HAND: a
# dashboard that fixes itself at 3am would stay unsupervised until someone
# noticed the stale flag. The episode boundary is a self-recovered dashboard.
# ---------------------------------------------------------------------------


def _tripped_state(mod, restart_age: int = 60) -> dict:
    """State with the ceiling already open and a full restart window."""
    import time as _time

    now = int(_time.time())
    return {
        "streak": 0,
        "restarts": [now - restart_age] * mod.MAX_RESTARTS,
        "ceiling_open": True,
    }


def test_episode_ends_when_the_dashboard_recovers(monkeypatch, state_env, queue_env):
    """A successful probe while tripped CLEARS the trip.

    There is nothing left to restart — the service is answering — so ending
    the episode costs nothing and restores normal supervision. Leaving it open
    would mean a dashboard that recovered on its own stays unwatched until a
    human deletes a state file they do not know exists.
    """
    mod = _load_watchdog()
    rec = _run_ticks(monkeypatch, [True], seed_state=_tripped_state(mod))

    assert rec.states[-1]["ceiling_open"] is False, "the trip never clears"
    assert rec.states[-1]["streak"] == 0
    assert rec.actuations == [], "nothing to restart — the probe just succeeded"
    assert rec.escalations == []


def test_tripped_and_still_failing_stays_quiet(monkeypatch, state_env, queue_env):
    """A dashboard that stays down stays cleanly down, with its L2 on file.

    No restarts, no re-filing, and the trip stays open — the escape holds for
    as long as the fault does.
    """
    mod = _load_watchdog()
    rec = _run_ticks(monkeypatch, [False] * 20, seed_state=_tripped_state(mod))

    assert rec.actuations == []
    assert rec.escalations == []
    assert all(s["ceiling_open"] is True for s in rec.states)


def test_tripped_failures_do_not_bank_a_streak(monkeypatch, state_env, queue_env):
    """The streak must NOT advance while the ceiling is open.

    If 20 failing ticks banked a streak of 20, the instant a single successful
    probe cleared the trip the very next miss would satisfy FAIL_STREAK and
    restart immediately — the hysteresis gate bypassed by a counter that ran
    while nobody was acting on it.
    """
    mod = _load_watchdog()
    rec = _run_ticks(monkeypatch, [False] * 20, seed_state=_tripped_state(mod))

    assert max(rec.streaks) == 0, f"banked a streak while tripped: {rec.streaks}"

    # Recover, then fail ONCE. A banked streak would restart here.
    rec2 = _run_ticks(monkeypatch, [True, False], recorder=rec)
    assert rec2.actuations == [], "restarted on the first miss after recovery"
    assert rec2.streaks[-1] == 1


def test_dedup_is_per_episode_not_forever(monkeypatch, state_env, queue_env):
    """A NEW storm after a recovery files a NEW L2.

    The dedup exists to stop an escalation storm inside one episode, not to
    silence the watchdog for the lifetime of the state file. A second,
    genuinely separate outage is new information and a steward needs it.
    """
    import time as _time

    mod = _load_watchdog()

    # Episode 1: tripped, then the dashboard recovers.
    rec = _run_ticks(monkeypatch, [True], seed_state=_tripped_state(mod))
    assert rec.states[-1]["ceiling_open"] is False

    # Time passes: the old restart epochs age out of the rolling window, so
    # the next storm is measured on its own merits.
    now = int(_time.time())
    aged = _load_watchdog().load_state()
    aged["restarts"] = [now - (mod.RATE_WINDOW_SECS + 600)] * mod.MAX_RESTARTS
    _load_watchdog().save_state(aged)

    # Episode 2: a fresh storm — MAX_RESTARTS streaks exhaust the allowance,
    # the next one trips again.
    rec2 = _run_ticks(
        monkeypatch, [False] * (mod.FAIL_STREAK * (mod.MAX_RESTARTS + 1))
    )

    assert len(rec2.escalations) == 1, (
        "a separate outage after a recovery filed no new L2 — the dedup "
        "silenced the watchdog forever rather than for one episode"
    )
    assert rec2.states[-1]["ceiling_open"] is True


def test_recovery_does_not_wipe_the_rolling_restart_window(
    monkeypatch, state_env, queue_env
):
    """Clearing the trip must not also clear the restart history.

    Those epochs are the evidence the ceiling is measured against. Wiping them
    on recovery would let a dashboard alternate crash → recover → crash and
    earn a fresh MAX_RESTARTS allowance every cycle, which is a restart storm
    with extra steps.
    """
    mod = _load_watchdog()
    tripped = _tripped_state(mod)
    rec = _run_ticks(monkeypatch, [True], seed_state=tripped)

    assert len(rec.states[-1]["restarts"]) == mod.MAX_RESTARTS, (
        "recovery wiped the rolling window that bounds the next storm"
    )


# ---------------------------------------------------------------------------
# Fail-soft / invariant I5
#
# The watchdog must never convert its own tooling failure into a restart, and
# must never crash the oneshot unit. A watchdog in `failed` state supervises
# nothing, and one that restarts on a systemctl hiccup is the incident again.
# ---------------------------------------------------------------------------


def test_i5_undeterminable_activation_alone_never_restarts(
    monkeypatch, state_env, queue_env
):
    """`systemctl show` failing is not evidence the dashboard is unhealthy.

    Paired with test_b5_undeterminable_activation_does_not_apply_grace: there,
    a genuinely failing PROBE still restarts. Here the probe SUCCEEDS, so the
    broken query alone must actuate nothing — otherwise a systemd hiccup, not
    the dashboard, decides when to restart.
    """
    rec = _run_ticks(monkeypatch, [True] * 5, activated_secs_ago=None)

    assert rec.actuations == [], f"a broken systemctl query restarted: {rec.actuations}"
    assert rec.escalations == []
    assert rec.streaks == [0] * 5


def test_i5_restart_timeout_is_survived_and_still_recorded(
    monkeypatch, state_env, queue_env
):
    """A TimeoutExpired from systemctl must not raise — and must still count.

    A timed-out restart very often took effect anyway, so recording it is the
    fail-direction that leads to the storm escape. Dropping it would make the
    ceiling unreachable: the watchdog would retry forever on a unit whose
    systemctl always hangs, which is the flap in a different costume.
    """
    import subprocess as _sp
    import time as _time

    active_enter = int(_time.time()) - OUTSIDE_GRACE_SECS
    attempted: list[list[str]] = []

    def fake_run(argv, *args, **kwargs):
        argv_list = list(argv)
        if argv_list and argv_list[0] == "systemd-cat":
            return _sp.CompletedProcess(argv_list, 0, stdout="", stderr="")
        if "show" in argv_list:
            return _sp.CompletedProcess(
                argv_list, 0, stdout=f"ActiveEnterTimestamp=@{active_enter}\n", stderr=""
            )
        attempted.append(argv_list)
        raise _sp.TimeoutExpired(argv_list, timeout=45)

    monkeypatch.setattr(_sp, "run", fake_run)

    mod = _load_watchdog()
    for _ in range(mod.FAIL_STREAK):
        m = _load_watchdog()
        _patch_urlopen(monkeypatch, m, urllib_error_503(m))
        m.tick()  # must not raise

    assert any("restart" in a for a in attempted), "no restart was attempted"
    assert len(_load_watchdog().load_state()["restarts"]) == 1, (
        "a timed-out restart was not recorded — the ceiling becomes unreachable"
    )


def test_i5_unwritable_state_never_raises_from_tick(monkeypatch, tmp_path, queue_env):
    """A state path that is a DIRECTORY degrades the watchdog to stateless.

    No streak can accumulate, so it stops restarting — the safe direction.
    Crashing here would instead put the oneshot into `failed`, where it
    supervises nothing at all and no longer even logs.
    """
    blocked = tmp_path / "state.json"
    blocked.mkdir()
    monkeypatch.setenv("DASHBOARD_WATCHDOG_STATE", str(blocked))

    rec = _run_ticks(monkeypatch, [False] * 5)  # must not raise

    assert rec.actuations == [], "restarted with no working state to gate on"


@pytest.mark.parametrize("failure", ["nonzero", "missing-uv", "timeout"])
def test_i5_escalation_failure_does_not_resume_the_flap(
    monkeypatch, state_env, queue_env, failure
):
    """A failed submit must leave the ceiling TRIPPED, not restarting again.

    The tempting fail-direction — "the escalation didn't go through, so keep
    trying to fix it ourselves" — is exactly the storm. Failing toward "quiet,
    with a journal line" loses the L2 but not the availability guarantee.
    """
    import subprocess as _sp
    import time as _time

    mod = _load_watchdog()
    active_enter = int(_time.time()) - OUTSIDE_GRACE_SECS
    actuations: list[list[str]] = []

    def fake_run(argv, *args, **kwargs):
        argv_list = list(argv)
        if argv_list and argv_list[0] == "systemd-cat":
            return _sp.CompletedProcess(argv_list, 0, stdout="", stderr="")
        if argv_list[:2] == ["systemctl", "--user"]:
            if "show" in argv_list:
                return _sp.CompletedProcess(
                    argv_list,
                    0,
                    stdout=f"ActiveEnterTimestamp=@{active_enter}\n",
                    stderr="",
                )
            actuations.append(argv_list)
            return _sp.CompletedProcess(argv_list, 0, stdout="", stderr="")
        if failure == "nonzero":
            return _sp.CompletedProcess(argv_list, 2, stdout="", stderr="argparse error")
        if failure == "missing-uv":
            raise FileNotFoundError(mod.UV_BIN)
        raise _sp.TimeoutExpired(argv_list, timeout=120)

    monkeypatch.setattr(_sp, "run", fake_run)

    _load_watchdog().save_state(_at_ceiling_state(mod))
    for _ in range(mod.FAIL_STREAK + 10):
        m = _load_watchdog()
        _patch_urlopen(monkeypatch, m, urllib_error_503(m))
        m.tick()  # must not raise

    assert actuations == [], f"resumed restarting after a failed submit: {actuations}"
    assert _load_watchdog().load_state()["ceiling_open"] is True


@pytest.mark.parametrize(
    "boom",
    [OSError("systemctl exploded"), RuntimeError("unexpected"), ValueError("bad")],
)
def test_i5_main_exits_zero_even_when_tick_raises(monkeypatch, state_env, boom):
    """A wedged tick must not put the oneshot unit into `failed`.

    Mirrors orchestrator-watchdog.main()'s isolation. A `failed` oneshot is
    silently skipped by some restart tooling and shows up as a red unit an
    operator has to clear by hand before supervision resumes — a much worse
    outcome than one skipped tick 30 seconds before the next.
    """
    mod = _load_watchdog()

    def exploding_tick():
        raise boom

    monkeypatch.setattr(mod, "tick", exploding_tick)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: None)

    assert mod.main() == 0


def test_i5_main_exits_zero_on_a_normal_tick(monkeypatch, state_env, queue_env):
    """The ordinary path returns 0 too — so a non-zero exit means something."""
    import subprocess as _sp

    monkeypatch.setattr(
        _sp,
        "run",
        lambda argv, *a, **k: _sp.CompletedProcess(list(argv), 0, stdout="", stderr=""),
    )

    mod = _load_watchdog()
    _patch_urlopen(monkeypatch, mod, _FakeResponse(200))

    assert mod.main() == 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
