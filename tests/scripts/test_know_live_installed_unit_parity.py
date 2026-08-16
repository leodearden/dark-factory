"""Host-coupled parity guard: the INSTALLED orchestrator-know-live.service unit.

Deliberately HOST-COUPLED — the opposite contract of every sibling parity
suite in this directory. test_check_dashboard_unit_parity.py states its rule
plainly: "All drift-logic tests run against inline fixture strings and
tmp_path directories — NEVER the host's real ~/.config/systemd/user/". This
module exists BECAUSE that rule is correct for a general parity checker but
cannot answer the one question task 3642 needs answered: has THIS host's
installed unit, and its systemd --user manager, actually been reconciled
with the committed template. scripts/check_orchestrator_unit_parity.py (a
registry-driven installed-vs-committed gate) has since LANDED on main — task
3424, picked up here when this branch was rebased onto it — and setup-host.sh
now runs it as a pre-install gate. That does not make this module redundant,
and the two answer different questions: the checker compares two FILES for
symmetric equality, so it is silent about the MANAGER layer below (an install
without a daemon-reload passes it), and it is a script an operator must run
rather than an assertion the suite enforces. What it does retire is the older
claim here that no portable checker existed to exercise against a fixture —
one now does, and drift-logic coverage of it belongs in its own fixture-based
suite (tests/scripts/test_check_orchestrator_unit_parity.py), not here. A
run on a host with the unit installed and reconciled gets a live
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
CANONICAL_CONFIG_BASENAME is now IMPORTED from
tests/scripts/systemd_unit_invariants.py rather than mirrored from
tests/scripts/test_orchestrator_service_files.py — task 3773 lifted it
there together with the parser that consumes it, so there is one
definition instead of two copies to keep in step). Deliberately NOT
asserted: ActiveState — liveness is the
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

Reuse note, updated by task 3763 (this module previously recorded lifting
its helpers into tests/scripts/systemd_unit_invariants.py as "the better
long-term fix", untaken because that module sat outside task 3642's lock).
Half of that is now done, and the split is deliberate:

  - MOVED — `systemctl_user_show`, formerly the module-local
    `_systemctl_user_show`, now lives in systemd_unit_invariants.py and is
    imported above. It gained a second consumer
    (test_pump_web_ui_installed_unit_parity.py, task 3763), which is the
    trigger condition that module's own docstring names: "duplicating it
    into both is how the two copies drift until one silently stops
    catching the defect". Its correctness rests on four non-obvious
    MEASURED behaviours, so a divergent copy is the expensive kind of
    drift. Its negative-case owner stays HERE (the four-case parametrized
    test at the bottom of this file), mirroring how
    test_dashboard_service_template.py owns assert_restart_backoff_
    effective's guard.
  - ALSO MOVED — `UNIT_DIR` (now `INSTALLED_UNIT_DIR`),
    `_require_installed_unit` (now `require_installed_unit`, parameterized
    by basename) and `_SYSTEMCTL_SKIP_REASON` (now
    `SYSTEMCTL_SKIP_REASON`), on the same trigger and in the same amendment
    pass. These carry no parsing logic, so no negative-case owner moved
    with them, but INSTALLED_UNIT_DIR is the sharpest single-sourcing case
    in the set: it mirrors a path defined in another language's file
    (scripts/setup-host.sh:114), so each copy is another chance to
    mis-mirror, and a mis-mirror does not fail — it degrades to
    require_installed_unit() skipping, i.e. a guard that silently checks
    nothing, the exact failure mode the mirroring exists to prevent. Two
    consumers today, one more per future per-unit parity module.
  - ALSO MOVED, by task 3773 — `_config_arg_from_exec_start` (now
    `config_arg_from_exec_start`, taking a unit_name so its failure
    messages name the offender), `CANONICAL_CONFIG_BASENAME`, and the
    `MalformedExecStart` class the canonical parser raises. Same trigger,
    found by reviewer_comprehensive on task 3642 and filed as its
    follow-up: this module hand-copied a parser whose canonical copy
    already lived in tests/scripts/test_orchestrator_service_files.py.
    The lift ALSO closed the divergence the bullet this one replaces said
    a lift alone could NOT close — the copy here collapsed a dangling
    `--config` (the flag with no value after it) into the same None as
    "no --config at all", where the canonical parser raises. The shared
    parser adopts the RAISE contract, because None is load-bearing in the
    sibling suite (orchestrator-watchdog.service legitimately takes no
    --config and must SKIP), so overloading None with "malformed" is
    exactly how a guard waves through the drift it was written to catch.
    Reconciling cost this module nothing: every call site here already
    asserted `config_arg is not None`, so a dangling `--config` already
    failed — just with a generic "carries no --config argument" message
    instead of one naming the defect. The empty `--config=` spelling was
    reconciled in the same direction, being the same defect (the
    orchestrator would start with no config path at all) in the other
    spelling. Its negative-case owner stays HERE, in the PARSER layer
    section at the bottom of this file, exactly as systemctl_user_show's
    did through task 3763's lift.
  - STAYED LOCAL — `_argv_from_exec_start_show`, which still has exactly
    one consumer (this module), so lifting it would buy no de-duplication
    today. This directory's lift trigger is a SECOND consumer, not
    proximity: test_orchestrator_service_files.py's `_exec_start_line`
    (file content -> the ExecStart= line) stayed with ITS single consumer
    for the same reason, in the same pass.
"""

import pathlib
import shutil
import subprocess

import pytest
from systemd_unit_invariants import (
    CANONICAL_CONFIG_BASENAME,
    INSTALLED_UNIT_DIR,
    SYSTEMCTL_SKIP_REASON,
    MalformedExecStart,
    assert_restart_backoff_effective,
    config_arg_from_exec_start,
    require_installed_unit,
    restart_directive,
    systemctl_user_show,
)

UNIT_BASENAME = "orchestrator-know-live.service"
INSTALLED_UNIT_PATH = INSTALLED_UNIT_DIR / UNIT_BASENAME

# The committed template the installed copy is propagated FROM; parents[2]
# because this file is <repo>/tests/scripts/<name>.py. Read at assertion
# time rather than hard-coding the expected RestartSteps value — see the
# derivation note in the MANAGER-layer test below.
COMMITTED_UNIT_PATH = pathlib.Path(__file__).parents[2] / "scripts" / UNIT_BASENAME

def _argv_from_exec_start_show(exec_start_value: str) -> str | None:
    """Extract the `argv[]=...` segment from a `systemctl show -p ExecStart` struct.

    Measured shape (systemd 255.4, this host, 2026-08-06): `{ path=... ;
    argv[]=<full argv> ; ignore_errors=no ; start_time=... ; stop_time=... ;
    pid=... ; code=... ; status=... }` — a `" ; "`-delimited sequence of
    `key=value` segments. Isolating the `argv[]=` segment before handing the
    remainder to config_arg_from_exec_start (rather than scanning the whole
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
    path = require_installed_unit(UNIT_BASENAME)
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
    path = require_installed_unit(UNIT_BASENAME)
    exec_start = restart_directive(path, "ExecStart")
    assert exec_start is not None, f"{path} declares no ExecStart= line"
    config_arg = config_arg_from_exec_start(exec_start, str(path))
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
@pytest.mark.skipif(shutil.which("systemctl") is None, reason=SYSTEMCTL_SKIP_REASON)
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
    `daemon-reload` and restarted the unit, so this now pins the reconciled
    state rather than asserting the pre-fix RED.

    The expected value is DERIVED from the committed template rather than
    hard-coded (task 3763 amendment, applied here because this module was
    already being edited for the shared-helper lift). A literal '4' would
    turn any future legitimate template edit into a RED test on a fully
    reconciled host, with a failure message that misdirects by asserting the
    template says 4 when it no longer does — and a prescribed remediation
    that had already been carried out.
    """
    require_installed_unit(UNIT_BASENAME)
    shown = systemctl_user_show(UNIT_BASENAME, "RestartSteps")
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
    assert COMMITTED_UNIT_PATH.is_file(), (
        f"{COMMITTED_UNIT_PATH} is missing, so this guard cannot derive the "
        "RestartSteps= value the host is supposed to have. That path is "
        "in-repo, not host state — a missing committed template is a repo "
        "defect, not an environment fact, so this fails rather than skips."
    )
    expected_steps = restart_directive(COMMITTED_UNIT_PATH, "RestartSteps")
    assert expected_steps is not None, (
        f"the committed {COMMITTED_UNIT_PATH} declares no RestartSteps= line, "
        "so its own RestartMaxDelaySec= is inert at the source and there is "
        "nothing coherent for this host guard to require. Fix the template "
        "first — tests/scripts/test_systemd_restart_backoff.py is the gate "
        "that owns that invariant for committed units."
    )
    steps = shown.get("RestartSteps")
    assert steps == expected_steps, (
        f"systemctl --user show {UNIT_BASENAME} -p RestartSteps reports "
        f"RestartSteps={steps!r}, but the committed {COMMITTED_UNIT_PATH} "
        f"declares RestartSteps={expected_steps}; propagate it with "
        f"`install -m 0644 scripts/{UNIT_BASENAME} {INSTALLED_UNIT_PATH}` "
        f"followed by `systemctl --user daemon-reload` and `systemctl --user "
        f"restart {UNIT_BASENAME}`."
    )


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("systemctl") is None, reason=SYSTEMCTL_SKIP_REASON)
def test_installed_unit_manager_execstart_config_is_canonical() -> None:
    """systemd --user's LOADED ExecStart= argv must carry the canonical --config basename.

    Expected GREEN on arrival, mirroring the file-layer config check above —
    the manager already agrees with the (already-canonical) installed file.
    Kept as its own assertion, at its own layer, so a future regression that
    only reaches the file (an install without a daemon-reload) or only
    reaches the manager (a hand-edited drop-in) is still caught by whichever
    layer it actually lands in.

    Asserts EXACT basename equality via the same config_arg_from_exec_start
    token parser the file layer uses, applied to the argv[] field of the
    manager's struct view (_argv_from_exec_start_show) — not a substring
    check on the raw ExecStart= blob. A substring check would also accept a
    drop-in that repointed --config at, say,
    backup-dark-factory-orchestrator.yaml, since that value still CONTAINS
    the canonical basename as a substring; exact-basename equality is what
    the drop-in-only regression this layer exists to catch actually
    requires.
    """
    require_installed_unit(UNIT_BASENAME)
    shown = systemctl_user_show(UNIT_BASENAME, "ExecStart")
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
    config_arg = config_arg_from_exec_start(
        argv, f"systemctl --user show {UNIT_BASENAME} -p ExecStart argv[]"
    )
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
# require_installed_unit(UNIT_BASENAME)/systemctl: these read no host state, only
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
            "/usr/bin/orchestrator run --project foo",
            None,
            id="no-config-flag-at-all",
        ),
    ],
)
def test_config_arg_from_exec_start_returns_value_or_none(exec_start_value, expected) -> None:
    """config_arg_from_exec_start's token scan, pinned against inline fixtures.

    Half of the RECONCILED contract task 3773 settled when this module's
    hand-copy of the parser was retired in favour of the shared one: None
    is returned for EXACTLY one input shape — an ExecStart= carrying no
    `--config` flag at all. That answer is load-bearing rather than
    incidental, which is why it is asserted as a RETURN and not folded in
    with the raising cases below: the sibling suite
    (test_orchestrator_service_files.py) SKIPs its canonical-filename
    guard on None for orchestrator-watchdog.service, which runs a probe
    script that legitimately takes no --config, so a shared parser that
    started raising here would turn that unit's correct shape into a hard
    failure.

    The `--config` and `--config=` spellings are both accepted because
    both call sites in this module hand the scan a real-world string it
    does not control — an installed unit file's ExecStart= value, and the
    argv[] segment of a `systemctl show` struct.
    """
    assert config_arg_from_exec_start(exec_start_value, "<fixture>") == expected


@pytest.mark.parametrize(
    "exec_start_value",
    [
        pytest.param(
            "/usr/bin/orchestrator run --config",
            id="dangling-flag-no-value",
        ),
        pytest.param(
            "/usr/bin/orchestrator run --config=",
            id="equals-form-empty-value",
        ),
    ],
)
def test_config_arg_from_exec_start_raises_on_malformed(exec_start_value) -> None:
    """The other half of the reconciled contract: no value means RAISE, not None.

    This is the divergence task 3773 closed. The copy that used to live in
    this module collapsed a dangling trailing `--config` into the same
    None as "no --config at all", while test_orchestrator_service_files.py's
    canonical parser raised — two parsers answering the same question two
    different ways, which is precisely the silent drift
    systemd_unit_invariants.py exists to prevent. The shared parser adopts
    the RAISE contract, so None now means one thing only (see the test
    above) and a unit that would start the orchestrator with NO config
    path at all fails loudly.

    Adopting it cost this module nothing: both call sites already asserted
    `config_arg is not None`, so a dangling `--config` already failed here
    — with a generic "carries no --config argument" message that named
    the symptom rather than the defect. The unit_name assertion below is
    what that reconciliation actually bought, and is why the call sites
    now pass one: the installed unit's PATH, or the `systemctl --user show
    ... argv[]` provenance, survives into the failure text instead of the
    reader being left to guess which of the two layers produced it.

    The empty `--config=` spelling is pinned alongside the dangling flag
    because it is the SAME defect wearing the other spelling — both copies
    used to return "" for it, which each call site then failed on with a
    confusing message about a basename. Verified safe to tighten: every
    committed unit uses the space-separated form with a real path, so no
    live verdict moves.
    """
    with pytest.raises(MalformedExecStart) as excinfo:
        config_arg_from_exec_start(exec_start_value, UNIT_BASENAME)
    assert UNIT_BASENAME in str(excinfo.value), (
        "the raise must name the unit it was asked about — the caller-supplied "
        "context (a unit path, or a `systemctl show` provenance string) is the "
        "whole diagnostic improvement this reconciliation bought over the "
        f"generic None both call sites already failed on. Got: {excinfo.value}"
    )


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
    """systemctl_user_show's docstring promises "Returns None — never raises".

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
    assert systemctl_user_show(UNIT_BASENAME, "RestartSteps") is None, (
        "systemctl_user_show's docstring promises None, never raises — a "
        f"{type(raised).__name__} from subprocess.run must degrade to a "
        "skip via None, not propagate uncaught"
    )
