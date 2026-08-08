"""Host-coupled parity guard: the INSTALLED orchestrator-know-live.service unit.

Deliberately HOST-COUPLED — the opposite contract of every sibling parity
suite in this directory. test_check_dashboard_unit_parity.py states its rule
plainly: "All drift-logic tests run against inline fixture strings and
tmp_path directories — NEVER the host's real ~/.config/systemd/user/". This
module exists BECAUSE that rule is correct for a general parity checker but
cannot answer the one question task 3642 needs answered: has THIS host's
installed unit, and its systemd --user manager, actually been reconciled
with the committed template. scripts/check_orchestrator_unit_parity.py (a
registry-driven installed-vs-committed gate) is confirmed absent from main —
its commits live only on the unmerged task/3424 branch — so there is no
portable checker this module could instead exercise against a fixture. A
future run on a host with the unit installed and reconciled gets a live
green answer about ITS install; a fresh checkout or CI runner with no
installed unit, no user D-Bus session, or a pre-254 systemd that has never
heard of RestartSteps= degrades to a skip (see the guards below), never a
false failure.

Scope is deliberately narrow: two PROPERTIES of exactly one unit
(orchestrator-know-live.service), at two layers — not a byte-parity sweep
across the fleet:

  1. FILE layer — the installed unit FILE on disk.
  2. MANAGER layer — systemd --user's LOADED view of the unit, via
     `systemctl --user show`. Not redundant with the file layer: `cp`-ing a
     corrected unit into place without `daemon-reload` leaves the manager
     holding the stale unit, so only the manager layer proves a reload
     actually happened.

Each layer checks two properties: the RestartMaxDelaySec=/RestartSteps=
pairing (systemd silently drops an unpaired cap — see
tests/scripts/systemd_unit_invariants.py) and the `--config` argument
naming the canonical dark-factory-orchestrator.yaml basename (task 3641;
CANONICAL_CONFIG_BASENAME mirrors tests/scripts/test_orchestrator_service_
files.py:543). Deliberately NOT asserted: ActiveState — liveness is the
watchdog's job (scripts/orchestrator-watchdog.py) and pinning it here would
make this suite fail during any legitimate restart window.

Marking: the four host-touching tests below (two FILE-layer, two
MANAGER-layer) each carry `@pytest.mark.integration`, which is already
registered and deselected by the ROOT pyproject.toml's default addopts
(`-m 'not smoke and not integration and not warm_lane_bash'`) — the same
mechanism fused-memory/tests/test_falkor_fulltext_integration.py and
shared/tests/test_cli_invoke_integration.py use to keep host/service-coupled
assertions out of the default, unmarked suite, so a legitimate operator
action against ~/.config/systemd/user (a manual `systemctl --user stop`, a
hand drop-in, a fleet redeploy in progress) can no longer turn every
concurrent task worktree's default test run red. The marker is applied
per-FUNCTION rather than as a single module-level `pytestmark`, because the
small fixture-string parser tests in the PARSER layer section at the bottom
of this module read no host state at all — marking them too would silently
reintroduce the "zero coverage off-host" gap this module is otherwise
written to avoid.

Importable `from systemd_unit_invariants import ...` because
tests/scripts/conftest.py puts this directory on sys.path — pytest's
--import-mode=importlib (pyproject.toml addopts) deliberately does not do
that on its own.
"""

import pathlib
import shutil
import subprocess

import pytest

from systemd_unit_invariants import assert_restart_backoff_effective, restart_directive

# Mirrors scripts/setup-host.sh:114 (`UNIT_DIR="$HOME/.config/systemd/user"`)
# exactly. Deliberately NOT XDG_CONFIG_HOME-aware: the installer itself does
# not honour that variable, so making this guard honour it would only ever
# point the guard at a directory setup-host.sh never writes to — the unit
# would not be found there, _require_installed_unit would skip, and the
# guard would silently check nothing while the real installed unit (at
# $HOME/.config/systemd/user, unconditionally) drifted. Mirroring the
# installer's actual, non-configurable path is the whole point of a parity
# guard.
UNIT_DIR = pathlib.Path.home() / ".config" / "systemd" / "user"

UNIT_BASENAME = "orchestrator-know-live.service"
INSTALLED_UNIT_PATH = UNIT_DIR / UNIT_BASENAME

# Mirrors tests/scripts/test_orchestrator_service_files.py:543 — copied
# inline (not imported across test modules) per this module's reuse note.
CANONICAL_CONFIG_BASENAME = "dark-factory-orchestrator.yaml"

_SYSTEMCTL_SKIP_REASON = (
    "systemctl is not installed; this test requires a live systemd --user "
    "manager and has no fixture-based fallback (see module docstring)"
)


def _require_installed_unit() -> pathlib.Path:
    """Return INSTALLED_UNIT_PATH, or skip if it does not exist on this host.

    A fresh checkout or a CI runner has no ~/.config/systemd/user at all —
    that is an environment fact, not a defect in the unit, so this skips
    rather than fails.
    """
    if not INSTALLED_UNIT_PATH.exists():
        pytest.skip(
            f"{INSTALLED_UNIT_PATH} does not exist on this host (fresh "
            "checkout or CI runner with no installed orchestrator units)"
        )
    return INSTALLED_UNIT_PATH


def _config_arg_from_exec_start(exec_start_value: str) -> str | None:
    """Token immediately after `--config` in an ExecStart= value, or None.

    Mirrors test_orchestrator_service_files.py's _exec_start_config_arg
    token parse (also accepts `--config=VALUE`), copied inline rather than
    imported across test modules.

    Known, intentional divergence from that canonical parser: the canonical
    helper raises a MalformedExecStart-style exception for a dangling
    trailing `--config` (the flag with no value), keeping that case
    distinct from "no --config at all". This copy collapses both to None.
    Every caller in this module already asserts `config_arg is not None`,
    so a dangling `--config` still fails the test here too — just with this
    module's generic "carries no --config argument" message rather than a
    MalformedExecStart-specific one. Lifting both parsers into a shared
    helper (e.g. tests/scripts/systemd_unit_invariants.py) would close this
    gap for real and is the better long-term fix, but that module and
    test_orchestrator_service_files.py both sit outside this task's locked
    scope (scripts/setup-host.sh and this module only) — tracked as
    follow-up rather than reached for here.
    """
    tokens = exec_start_value.split()
    for i, token in enumerate(tokens):
        if token == "--config":
            return tokens[i + 1] if i + 1 < len(tokens) else None
        if token.startswith("--config="):
            return token.split("=", 1)[1]
    return None


def _argv_from_exec_start_show(exec_start_value: str) -> str | None:
    """Extract the `argv[]=...` segment from a `systemctl show -p ExecStart` struct.

    Measured shape (systemd 255.4, this host, 2026-08-06): `{ path=... ;
    argv[]=<full argv> ; ignore_errors=no ; start_time=... ; stop_time=... ;
    pid=... ; code=... ; status=... }` — a `" ; "`-delimited sequence of
    `key=value` segments. Isolating the `argv[]=` segment before handing the
    remainder to _config_arg_from_exec_start (rather than scanning the whole
    raw struct blob for a `--config` token) keeps both parser layers asking
    the identical question — "what is argv[i + 1] after --config" — of the
    actual argument vector, not of struct metadata that happens not to
    collide with it today but is not guaranteed to stay that way. Returns
    None, never raises, if no `argv[]=` segment is present, so callers can
    skip/fail explicitly on an unexpected shape rather than mis-parse one.
    """
    for segment in exec_start_value.split(" ; "):
        segment = segment.strip()
        if segment.startswith("argv[]="):
            return segment[len("argv[]=") :]
    return None


def _systemctl_user_show(unit: str, *properties: str) -> dict[str, str] | None:
    """Run `systemctl --user show <unit> -p <prop> ...`, parsed to a dict.

    Returns None — never raises — when the query cannot be answered at all:
    a non-zero exit, output naming a bus connection failure ("Failed to
    connect to bus" — the shape seen from a container/CI sandbox with no
    user D-Bus session), or the query timing out — a wedged systemd --user
    manager or a stuck D-Bus leaving this call hung past its 30s timeout,
    which surfaces as subprocess.TimeoutExpired. That last case is caught
    deliberately and degrades to the same skip: subprocess.TimeoutExpired
    is a subprocess.SubprocessError, NOT an OSError (verified MRO:
    TimeoutExpired -> SubprocessError -> Exception), so the handler below
    must name both classes — narrowing it back to `except OSError` alone
    is the exact regression a prior review caught here. Callers treat None
    as "skip", not "fail": this invariant requires a live systemd --user
    manager to answer, and its absence is an environment fact rather than
    a defect in the unit.

    A property systemd does not implement at all (verified: `-p
    SomeUnknownProperty` against a live unit exits 0 with EMPTY stdout, on
    this host's systemd 255.4) is NOT surfaced as None here — it is simply
    absent from the returned dict, distinct from a recognised-but-blank
    value. Callers that care about that distinction (e.g. RestartSteps=,
    unsupported before systemd 254) must check `"Prop" not in shown`
    themselves; collapsing "unsupported" into the same falsy shape as
    "supported and empty" is how a guard meant to skip cleanly on an old
    systemd instead fails loudly and misleadingly on one.
    """
    argv = ["systemctl", "--user", "show", unit]
    for prop in properties:
        argv += ["-p", prop]
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    combined = f"{result.stdout}{result.stderr}"
    if result.returncode != 0 or "failed to connect to bus" in combined.lower():
        return None
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            parsed[key] = value
    return parsed


# ---------------------------------------------------------------------------
# FILE layer — the installed unit on disk
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_installed_unit_file_restart_backoff_effective() -> None:
    """The INSTALLED unit file's RestartMaxDelaySec= must be paired with RestartSteps=.

    Expected GREEN on arrival — a regression pin, not the RED signal this
    module was originally written to add. Measured 2026-08-06, before task
    3642's GREEN step: the installed copy at ~/.config/systemd/user/
    orchestrator-know-live.service declared RestartMaxDelaySec=60 with no
    RestartSteps= line — the committed scripts/orchestrator-know-live.service
    was ahead by exactly that directive (plus its two-line explanatory
    comment), landed for the whole fleet by commit f7459a1c49 but not yet
    propagated to this host for know-live. Task 3642's step-2 reinstalled
    the committed template and reloaded the manager, closing that gap; this
    test now pins the reconciled state so a future stale re-install
    regresses loudly instead of silently. Reuses
    systemd_unit_invariants.assert_restart_backoff_effective, the same
    RELATIONAL invariant test_systemd_restart_backoff.py already applies to
    the COMMITTED template — this is that same check applied to the
    INSTALLED file instead.
    """
    path = _require_installed_unit()
    assert_restart_backoff_effective(path)


@pytest.mark.integration
def test_installed_unit_file_execstart_config_is_canonical() -> None:
    """The INSTALLED unit file's --config must name the canonical basename.

    Expected GREEN on arrival — a regression pin, not the RED signal this
    module exists to add. Task 3641 already canonicalized the installed
    file's ExecStart= (measured: it agrees byte-for-byte with the committed
    template on this line); this guards against a FUTURE setup-host.sh run
    re-installing a stale template that regresses it.
    """
    path = _require_installed_unit()
    exec_start = restart_directive(path, "ExecStart")
    assert exec_start is not None, f"{path} declares no ExecStart= line"
    config_arg = _config_arg_from_exec_start(exec_start)
    assert config_arg is not None, (
        f"{path}'s ExecStart= carries no --config argument: {exec_start!r}"
    )
    actual = pathlib.PurePosixPath(config_arg).name
    assert actual == CANONICAL_CONFIG_BASENAME, (
        f"{path}'s ExecStart= points --config at {config_arg!r}, whose "
        f"basename is {actual!r}. It must be the canonical "
        f"{CANONICAL_CONFIG_BASENAME!r} (CLAUDE.md; task 3641) — the "
        "dashboard's _discover_escalation_urls keys on that exact filename."
    )


# ---------------------------------------------------------------------------
# MANAGER layer — systemd --user's LOADED view of the unit
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("systemctl") is None, reason=_SYSTEMCTL_SKIP_REASON)
def test_installed_unit_manager_restart_steps_effective() -> None:
    """systemd --user's LOADED view must report RestartSteps=4, not just the file.

    Expected GREEN on arrival — a regression pin, mirroring the file-layer
    check above. Not redundant with it: `cp`-ing a corrected unit into place
    without `systemctl --user daemon-reload` leaves the MANAGER holding the
    old unit, so a file-only check would bless a host whose backoff is still
    inert. Measured 2026-08-06, before task 3642's GREEN step: `systemctl
    --user show orchestrator-know-live.service -p RestartSteps` reported
    RestartSteps=0 — systemd's zero-value default, confirming the manager
    had not reloaded the RestartSteps=4 line at all. Task 3642's step-2 ran
    `daemon-reload` and restarted the unit, so this now pins RestartSteps=4
    as the reconciled state rather than asserting the pre-fix RED.
    """
    _require_installed_unit()
    shown = _systemctl_user_show(UNIT_BASENAME, "RestartSteps")
    if shown is None:
        pytest.skip(
            "systemctl --user show could not be queried (no user D-Bus "
            "session in this runner)"
        )
    if "RestartSteps" not in shown:
        pytest.skip(
            f"systemctl --user show {UNIT_BASENAME} -p RestartSteps returned "
            "no RestartSteps= property at all (empty stdout, not merely an "
            "empty value) — most likely this host's systemd predates 254, "
            "which introduced RestartSteps=/RestartMaxDelaySec=. Verified: "
            "`systemctl --user show <unit> -p <unsupported-property>` exits "
            "0 with empty stdout, so there is nothing for this guard to "
            "assert against an unsupported property."
        )
    steps = shown.get("RestartSteps")
    assert steps == "4", (
        f"systemctl --user show {UNIT_BASENAME} -p RestartSteps reports "
        f"RestartSteps={steps!r}, not '4'. The committed "
        "scripts/orchestrator-know-live.service declares RestartSteps=4; "
        "propagate it with `install -m 0644 scripts/orchestrator-know-live"
        f".service {INSTALLED_UNIT_PATH}` followed by `systemctl --user "
        f"daemon-reload` and `systemctl --user restart {UNIT_BASENAME}`."
    )


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("systemctl") is None, reason=_SYSTEMCTL_SKIP_REASON)
def test_installed_unit_manager_execstart_config_is_canonical() -> None:
    """systemd --user's LOADED ExecStart= argv must carry the canonical --config basename.

    Expected GREEN on arrival, mirroring the file-layer config check above —
    the manager already agrees with the (already-canonical) installed file.
    Kept as its own assertion, at its own layer, so a future regression that
    only reaches the file (an install without a daemon-reload) or only
    reaches the manager (a hand-edited drop-in) is still caught by whichever
    layer it actually lands in.

    Asserts EXACT basename equality via the same _config_arg_from_exec_start
    token parser the file layer uses, applied to the argv[] field of the
    manager's struct view (_argv_from_exec_start_show) — not a substring
    check on the raw ExecStart= blob. A substring check would also accept a
    drop-in that repointed --config at, say,
    backup-dark-factory-orchestrator.yaml, since that value still CONTAINS
    the canonical basename as a substring; exact-basename equality is what
    the drop-in-only regression this layer exists to catch actually
    requires.
    """
    _require_installed_unit()
    shown = _systemctl_user_show(UNIT_BASENAME, "ExecStart")
    if shown is None:
        pytest.skip(
            "systemctl --user show could not be queried (no user D-Bus "
            "session in this runner)"
        )
    exec_start = shown.get("ExecStart", "")
    argv = _argv_from_exec_start_show(exec_start)
    assert argv is not None, (
        f"systemctl --user show {UNIT_BASENAME} -p ExecStart returned a "
        f"struct with no argv[]= segment: {exec_start!r}"
    )
    config_arg = _config_arg_from_exec_start(argv)
    assert config_arg is not None, (
        f"systemctl --user show {UNIT_BASENAME} -p ExecStart's argv[] "
        f"carries no --config argument: {argv!r}"
    )
    actual = pathlib.PurePosixPath(config_arg).name
    assert actual == CANONICAL_CONFIG_BASENAME, (
        f"systemctl --user show {UNIT_BASENAME} -p ExecStart's argv[] "
        f"points --config at {config_arg!r}, whose basename is {actual!r}. "
        f"It must be the canonical {CANONICAL_CONFIG_BASENAME!r} (CLAUDE.md; "
        "task 3641) — the dashboard's _discover_escalation_urls keys on "
        "that exact filename."
    )


# ---------------------------------------------------------------------------
# PARSER layer — portable, fixture-string coverage for the token parsers
#
# Deliberately NOT @pytest.mark.integration and NOT gated on
# _require_installed_unit()/systemctl: these read no host state, only
# inline fixture strings, so they run in the default suite on every
# machine — including CI and every other dev box, where the four tests
# above skip. Without a negative-case owner of its own, a regression in
# either parser (an off-by-one on tokens[i + 1], a broken --config= or
# argv[]= prefix branch) would be invisible everywhere except the one host
# that actually exercises it, and would surface there as a confusing
# failure about the unit rather than about the parser — mirroring why
# test_systemd_restart_backoff.py's docstring records that this directory's
# convention is for a shared assertion to have its own negative-case owner.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exec_start_value", "expected"),
    [
        pytest.param(
            "/usr/bin/orchestrator run --config /etc/dark-factory-orchestrator.yaml",
            "/etc/dark-factory-orchestrator.yaml",
            id="space-separated",
        ),
        pytest.param(
            "/usr/bin/orchestrator run --config=/etc/dark-factory-orchestrator.yaml",
            "/etc/dark-factory-orchestrator.yaml",
            id="equals-form",
        ),
        pytest.param(
            "/usr/bin/orchestrator run --config",
            None,
            id="dangling-flag-no-value",
        ),
        pytest.param(
            "/usr/bin/orchestrator run --project foo",
            None,
            id="no-config-flag-at-all",
        ),
    ],
)
def test_config_arg_from_exec_start_parses_all_forms(exec_start_value, expected) -> None:
    """_config_arg_from_exec_start's token scan, pinned against inline fixtures."""
    assert _config_arg_from_exec_start(exec_start_value) == expected


def test_argv_from_exec_start_show_extracts_argv_segment() -> None:
    """_argv_from_exec_start_show isolates argv[] from a full struct blob.

    Fixture is the measured `systemctl --user show ... -p ExecStart` shape
    (see the function's own docstring), with paths shortened for
    readability.
    """
    blob = (
        "{ path=/usr/bin/uv ; argv[]=/usr/bin/uv run --config /etc/x.yaml ; "
        "ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; "
        "code=(null) ; status=0/0 }"
    )
    assert _argv_from_exec_start_show(blob) == "/usr/bin/uv run --config /etc/x.yaml"


def test_argv_from_exec_start_show_returns_none_without_argv_segment() -> None:
    """An unexpected struct shape with no argv[]= segment yields None, not a crash."""
    assert _argv_from_exec_start_show("{ path=/usr/bin/uv ; ignore_errors=no }") is None


@pytest.mark.parametrize(
    "raised",
    [
        pytest.param(
            subprocess.TimeoutExpired(cmd=["systemctl"], timeout=30),
            id="timeout-expired-wedged-manager",
        ),
        pytest.param(
            FileNotFoundError("systemctl missing"),
            id="file-not-found-oserror-branch",
        ),
        pytest.param(
            PermissionError("systemctl not executable"),
            id="permission-error-oserror-branch",
        ),
        pytest.param(
            subprocess.CalledProcessError(1, "systemctl"),
            id="called-process-error-future-check-true",
        ),
    ],
)
def test_systemctl_user_show_returns_none_when_subprocess_raises(
    monkeypatch: pytest.MonkeyPatch, raised: BaseException
) -> None:
    """_systemctl_user_show's docstring promises "Returns None — never raises".

    Regression pin for a swallowed-exception gap (reviewer_comprehensive,
    robustness): the only handler was `except OSError`, but
    subprocess.TimeoutExpired — raised when a wedged systemd --user manager
    or a stuck D-Bus leaves `systemctl show` hung past its 30s timeout — is
    NOT an OSError subclass (verified MRO: TimeoutExpired -> SubprocessError
    -> Exception), so it propagated uncaught instead of degrading to the
    documented skip, i.e. the exact "false failure on a degraded
    environment" the module docstring promises never to produce.
    subprocess.CalledProcessError is pinned for the same reason, as a
    forward-guard: it cannot arise today (the call uses check=False) but
    would hit the identical gap if a future edit flips that flag.
    FileNotFoundError and PermissionError are already-green regression pins
    for the existing, correctly-handled OSError branch.

    Portable by construction — subprocess.run is fully monkeypatched, so no
    host or live systemd is needed — which is why this lives in the PARSER
    layer section (see its header comment) rather than only being covered
    by the host-coupled MANAGER-layer tests above, which are deselected by
    default and would give this helper zero protection on CI or any other
    dev box.
    """

    def fake_run(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise raised

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _systemctl_user_show(UNIT_BASENAME, "RestartSteps") is None, (
        "_systemctl_user_show's docstring promises None, never raises — a "
        f"{type(raised).__name__} from subprocess.run must degrade to a "
        "skip via None, not propagate uncaught"
    )
