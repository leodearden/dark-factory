"""The dashboard unit's restart stanza is effective in both committed copies.

Covers the Restart=/RestartSec=/RestartSteps=/RestartMaxDelaySec= backoff and
the SuccessExitStatus= clean-stop classification, in
scripts/dashboard.service.template and dashboard/dark-factory-dashboard.service.
These tests read the files directly, so no systemd runtime is required.
test_systemd_analyze_verify_reports_no_ignored_directives is the one skippable
corroboration that asks systemd itself.

Convention: each assertion helper earns its own negative tests, so it cannot
silently no-op, and every bad case pins its own failure mode with ``match=``.
"""

import os
import pathlib
import re
import shutil
import subprocess

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from systemd_unit_invariants import assert_restart_backoff_effective, restart_directive

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]
TEMPLATE = REPO_ROOT / "scripts" / "dashboard.service.template"
HARDCODED = REPO_ROOT / "dashboard" / "dark-factory-dashboard.service"


# ---------------------------------------------------------------------------
# Restart backoff
#
# The invariant and its directive reader live in
# tests/scripts/systemd_unit_invariants.py (assert_restart_backoff_effective,
# restart_directive), shared with the fleet-wide sweep in
# tests/scripts/test_systemd_restart_backoff.py so that both apply the SAME
# assertion rather than two copies of it.
#
# Nothing asserts that the sharing holds, deliberately: the `from
# systemd_unit_invariants import ...` at the top of this module IS the
# sharing, structurally, and a drifted private copy would be caught by that
# fleet sweep failing on whichever unit the copy stopped checking.
#
# This module owns the helper's negative-case guard,
# test_restart_backoff_guard_rejects_ineffective_units below.
# ---------------------------------------------------------------------------


def _write_restart_unit(path: pathlib.Path, body: str) -> pathlib.Path:
    """Write a minimal synthetic unit carrying only a [Service] restart stanza."""
    path.write_text(f"[Service]\n{body}\n", encoding="utf-8")
    return path


# Shared by test_restart_backoff_guard_rejects_ineffective_units' `no_steps`
# case and _write_backoff_canary below: both need the exact pre-fix "a cap
# declared with nothing to interpolate over" shape, and hoisting it to one
# constant makes their "same shape" claim hold structurally instead of by
# comment — editing one no longer leaves the other's claim quietly false.
_INERT_CAP_BODY = "Restart=on-failure\nRestartSec=5\nRestartMaxDelaySec=60"


def test_restart_backoff_guard_rejects_ineffective_units(
    tmp_path: pathlib.Path,
) -> None:
    """assert_restart_backoff_effective must fire on every inert-cap spelling.

    Follows this module's convention: an assertion helper earns its own
    negatives so it cannot silently no-op, and every case pins its own failure
    mode with ``match=``.  A bare
    ``pytest.raises(AssertionError)`` is satisfied by ANY assertion firing
    anywhere in the chain, so the distinct bad cases below — absent steps / zero
    steps / absent floor — would all still "pass" while tripping one shared
    branch, which is exactly the confusion they exist to rule out.
    """
    # Bad: the pre-fix state — a cap declared with nothing to interpolate over.
    no_steps = _write_restart_unit(tmp_path / "no_steps.service", _INERT_CAP_BODY)
    with pytest.raises(AssertionError, match="but no RestartSteps="):
        assert_restart_backoff_effective(no_steps)

    # Bad: present but zero — a zero-step curve is no backoff at all, so mere
    # presence of the directive is not the property being asserted.
    zero_steps = _write_restart_unit(
        tmp_path / "zero_steps.service",
        "Restart=on-failure\nRestartSec=5\nRestartSteps=0\nRestartMaxDelaySec=60",
    )
    with pytest.raises(AssertionError, match="produces no backoff curve"):
        assert_restart_backoff_effective(zero_steps)

    # Bad: overridden to zero by a later repeat.  systemd is LAST-WINS for these
    # scalars, so this unit's effective RestartSteps is 0 and its cap is
    # discarded — measured on systemd 255.4, which emits the pairing warning for
    # exactly this file while staying silent on the reverse order (0 then 4).
    # A first-match read of the directive would report 4 and pass, so this case
    # pins restart_directive's last-wins semantics through the guard that
    # depends on them.
    overridden_steps = _write_restart_unit(
        tmp_path / "overridden_steps.service",
        "Restart=on-failure\nRestartSec=5\nRestartSteps=4\n"
        "RestartSteps=0\nRestartMaxDelaySec=60",
    )
    with pytest.raises(AssertionError, match="produces no backoff curve"):
        assert_restart_backoff_effective(overridden_steps)
    assert restart_directive(overridden_steps, "RestartSteps") == "0"

    # Bad: cap and steps, but no floor to interpolate up FROM.
    no_floor = _write_restart_unit(
        tmp_path / "no_floor.service",
        "Restart=on-failure\nRestartSteps=4\nRestartMaxDelaySec=60",
    )
    with pytest.raises(AssertionError, match="no RestartSec="):
        assert_restart_backoff_effective(no_floor)

    # Good: the full curve — must not raise (guards against over-tightening).
    good = _write_restart_unit(
        tmp_path / "good.service",
        "Restart=on-failure\nRestartSec=5\nRestartSteps=4\nRestartMaxDelaySec=60",
    )
    assert_restart_backoff_effective(good)

    # Good: no cap declared at all — the invariant is conditional on the cap
    # being declared, not a blanket requirement that every unit implement
    # backoff.  Pins that a later author cannot tighten it into an
    # unconditional "RestartSteps= must be present".
    no_cap = _write_restart_unit(
        tmp_path / "no_cap.service", "Restart=on-failure\nRestartSec=5"
    )
    assert_restart_backoff_effective(no_cap)

    # Good: a suffixed spec is still a declared cap.  This is why
    # restart_directive captures the value opaquely — a (\d+) parse would read
    # `60s` as an absent directive and skip the invariant without a word.
    suffixed = _write_restart_unit(
        tmp_path / "suffixed.service",
        "Restart=on-failure\nRestartSec=5s\nRestartSteps=4\nRestartMaxDelaySec=60s",
    )
    assert_restart_backoff_effective(suffixed)
    assert restart_directive(suffixed, "RestartMaxDelaySec") == "60s"

    # ...and the inert form of that same spelling must STILL be caught, so the
    # opaque capture above bought precision, not laxity.
    suffixed_no_steps = _write_restart_unit(
        tmp_path / "suffixed_no_steps.service",
        "Restart=on-failure\nRestartSec=5s\nRestartMaxDelaySec=60s",
    )
    with pytest.raises(AssertionError, match="but no RestartSteps="):
        assert_restart_backoff_effective(suffixed_no_steps)

    # Bad: the same inert cap, spelled with the leading and around-separator
    # whitespace systemd.syntax permits.  This is a VALID unit whose cap systemd
    # parses and discards, so it must be caught — a column-0-only anchor reads
    # it as declaring no cap at all and returns early, skipping the invariant
    # without a word.  That is the one failure direction this guard cannot
    # afford: it masks the very defect the helper exists to surface, unlike the
    # opaque-value capture above, which errs toward checking MORE.
    spaced_no_steps = _write_restart_unit(
        tmp_path / "spaced_no_steps.service",
        "Restart=on-failure\n  RestartSec = 5\n\tRestartMaxDelaySec\t=\t60",
    )
    with pytest.raises(AssertionError, match="but no RestartSteps="):
        assert_restart_backoff_effective(spaced_no_steps)
    assert restart_directive(spaced_no_steps, "RestartMaxDelaySec") == "60"

    # Good: the whitespace-tolerant read is complete, not just detection-only —
    # a properly paired unit in that same spelling must not raise.
    spaced_good = _write_restart_unit(
        tmp_path / "spaced_good.service",
        "Restart=on-failure\n  RestartSec = 5\n  RestartSteps = 4\n"
        "  RestartMaxDelaySec = 60",
    )
    assert_restart_backoff_effective(spaced_good)


def test_restart_backoff_is_effective_in_both_unit_files() -> None:
    """Both unit files must pair RestartMaxDelaySec= with a usable RestartSteps=.

    Without the pairing systemd discards the cap at load time, so a dashboard
    stuck in a crash loop retries every 5s indefinitely instead of backing off
    toward 60s — the unit's comment claims a backoff that the unit does not do.
    """
    for path in (TEMPLATE, HARDCODED):
        assert_restart_backoff_effective(path)


def _ignored_directive_lines(output: str, unit: pathlib.Path) -> list[str]:
    """Return systemd-analyze lines reporting a discarded directive in *unit*.

    Split out of the test below so it can be exercised against captured
    systemd-analyze output WITHOUT a systemd runtime — the filter is the part
    that can silently no-op, and a guard that only runs where systemd is
    installed is no guard at all.  See
    test_ignored_directive_filter_matches_both_systemd_spellings for the
    negatives; the measured samples it pins are quoted there verbatim.
    """
    return [
        line
        for line in output.splitlines()
        # Case-insensitive: systemd writes "Ignoring." for a semantic discard
        # and "ignoring." for an unknown key, and on systemd < 254 the latter is
        # the ONLY form this unit's backoff directives can produce.
        if "ignoring." in line.lower()
        # Scoped to the unit under test: verify also loads dependencies, and
        # each warning names the unit it belongs to.  A semantic warning is
        # prefixed with the bare unit name, a parse warning with the full path
        # and line number.
        and (line.startswith(f"{unit.name}:") or str(unit) in line)
    ]


def _write_backoff_canary(directory: pathlib.Path) -> pathlib.Path:
    """Write a unit systemd-analyze will actually warn about, as a positive control.

    Thin wrapper over ``_write_restart_unit`` rather than a second
    hand-rolled unit writer.  The body is ``_INERT_CAP_BODY`` — the same
    pre-fix "cap declared with nothing to interpolate over" shape
    ``test_restart_backoff_guard_rejects_ineffective_units``'s ``no_steps``
    case writes — with ``ExecStart=/bin/true`` prepended.

    The ExecStart is not decoration: measured 2026-08-16 in this worktree
    (systemd 255.4-1ubuntu8.17), the bare restart stanza alone is REFUSED by
    systemd-analyze before the pairing check ever runs
    (``canary-noexec.service: Service has no ExecStart=, ExecStop=, or
    SuccessAction=. Refusing.``, exit 1, zero "Ignoring." line) — a canary in
    that shape would fail on every healthy host instead of drawing the
    warning it exists to draw.  With the ExecStart present it instead draws
    ``canary-exec.service: Service has RestartMaxDelaySec= but no
    RestartSteps= setting. Ignoring.`` at exit 0.  See
    test_backoff_canary_is_the_shape_systemd_must_warn_about for the pinned,
    host-independent contract this shape must satisfy.

    ``df-restart-backoff-canary.service`` is a name chosen not to collide
    with any real unit on any host; measured working.
    """
    return _write_restart_unit(
        directory / "df-restart-backoff-canary.service",
        f"ExecStart=/bin/true\n{_INERT_CAP_BODY}",
    )


def test_ignored_directive_filter_matches_both_systemd_spellings() -> None:
    """_ignored_directive_lines must catch both casings and only this unit.

    The samples below are real ``systemd-analyze verify`` output captured on
    2026-08-01 (systemd 255.4), not invented strings: the capital-I form from a
    unit with RestartMaxDelaySec= and no RestartSteps=, the lowercase form from
    a deliberately misspelled ``RestartStepz=``, and the dependency noise from
    ``systemd-analyze verify --user`` on the real dashboard unit.

    This is the guard that keeps the systemd test below honest.  Both of its
    ways of degrading into a silent pass — wrong casing, or another unit's
    warning counted as this one's — are pinned here, and they are pinned
    without needing systemd, so they hold on every host.
    """
    semantic = (
        f"{HARDCODED.name}: Service has RestartMaxDelaySec= but no "
        "RestartSteps= setting. Ignoring."
    )
    parse = (
        f"{HARDCODED}:42: Unknown key name 'RestartStepz' in section "
        "'Service', ignoring."
    )
    # Another unit's identical defect, plus genuinely unrelated noise.
    foreign = (
        "orchestrator-dark-factory.service: Service has RestartMaxDelaySec= "
        "but no RestartSteps= setting. Ignoring."
    )
    noise = "Failed to bind private socket: Address already in use"

    # Caught: the capital-I semantic warning...
    assert _ignored_directive_lines(semantic, HARDCODED) == [semantic]
    # ...and the lowercase parse warning, which is the ONLY signal available on
    # systemd < 254, where both backoff directives degrade to unknown keys.
    assert _ignored_directive_lines(parse, HARDCODED) == [parse]

    # Ignored: a warning that belongs to a different unit must not be blamed on
    # this file, and non-warning chatter must not trip the assertion.
    assert _ignored_directive_lines(f"{foreign}\n{noise}", HARDCODED) == []

    # Mixed stream: exactly the two lines that name this unit, in order.
    combined = "\n".join([noise, foreign, semantic, parse])
    assert _ignored_directive_lines(combined, HARDCODED) == [semantic, parse]

    # A clean run reports nothing.
    assert _ignored_directive_lines("", HARDCODED) == []


def _systemd_analyze_verify_skip_reason() -> str | None:
    """Return why the systemd-analyze verify test must skip, or None to run it.

    Two independent preconditions, checked in order:

    1. The ``systemd-analyze`` binary must be on PATH.
    2. ``XDG_RUNTIME_DIR`` must be set to a non-empty value.  Measured
       2026-08-12 in this worktree (systemd 255.4): with the variable unset,
       ``systemd-analyze verify --user`` cannot initialise a --user manager
       at all and its combined stdout+stderr is ONLY "Failed to lookup
       RuntimeDirectory path: No such device or address" / "Failed to
       initialize manager: No such device or address" (exit 1) -- it never
       gets far enough to emit the per-unit diagnostic lines the guarded
       test greps for, so ``assert not ignored`` there would pass having
       checked nothing.  An empty value reproduces that identical no-op
       byte-for-byte, which is why this checks truthiness
       (``os.environ.get(...)``) rather than key membership -- and a
       nonexistent-but-declared path (e.g. /tmp/does-not-exist) still yields
       real per-unit diagnostics at exit 0, which is why this does not
       "harden" into a directory-existence check either.  Contexts that hit
       this in practice: cron, ssh host-cmd, any headless invocation with no
       pam_systemd session -- ``shutil.which()`` succeeds there and the old
       inline decorator condition never fired.  DBUS_SESSION_BUS_ADDRESS was
       also measured (unsetting it alone still yields real diagnostics at
       exit 0) and is deliberately NOT checked here: requiring it too would
       over-skip on hosts where the guarded test works correctly.
    """
    if shutil.which("systemd-analyze") is None:
        return (
            "no systemd-analyze on this PATH; the file-content invariants "
            "above are the primary guard and this module requires no "
            "systemd runtime"
        )
    if not os.environ.get("XDG_RUNTIME_DIR"):
        return (
            "XDG_RUNTIME_DIR is not set (or is empty): systemd-analyze "
            "verify --user cannot initialise a --user manager without it "
            'and prints only "Failed to lookup RuntimeDirectory path" / '
            '"Failed to initialize manager", emitting zero per-unit '
            "diagnostic lines -- the guarded assertion would pass having "
            "checked nothing"
        )
    return None


_SYSTEMD_ANALYZE_SKIP_REASON = _systemd_analyze_verify_skip_reason()


def test_systemd_analyze_skip_reason_requires_both_binary_and_user_runtime_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_systemd_analyze_verify_skip_reason must gate on BOTH preconditions.

    Four cases below, each host-independent: both the binary lookup and
    XDG_RUNTIME_DIR are monkeypatched rather than read from this host's real
    environment, so this test itself needs no systemd runtime -- honouring
    the module docstring's "no systemd runtime is required" contract.  See
    _systemd_analyze_verify_skip_reason's docstring for the measured basis
    (the observed output strings, exit codes, and why DBUS_SESSION_BUS_ADDRESS
    is deliberately not part of the discriminator) that each case below
    exercises:

      (a) systemd-analyze binary absent.
      (b) binary present, XDG_RUNTIME_DIR absent -- the headless case a
          binary lookup alone cannot detect.
      (c) binary present, XDG_RUNTIME_DIR="" -- empty must count as absent.
      (d) binary present, XDG_RUNTIME_DIR set to a nonexistent path -- must
          still run.
    """
    # (a) Binary absent: the predicate must still gate the test, even with a
    # perfectly valid XDG_RUNTIME_DIR alongside it.
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    reason = _systemd_analyze_verify_skip_reason()
    assert reason is not None

    # (b) Binary present, XDG_RUNTIME_DIR absent: the predicate must still
    # gate the test. This is the headless case: shutil.which() finds the
    # binary, yet systemd-analyze verify --user checks nothing here.
    monkeypatch.setattr(
        shutil, "which", lambda *_a, **_k: "/usr/bin/systemd-analyze"
    )
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    reason = _systemd_analyze_verify_skip_reason()
    assert reason is not None

    # (c) Binary present, XDG_RUNTIME_DIR="": empty must count as absent, or
    # a membership test ("XDG_RUNTIME_DIR" in os.environ) would treat this as
    # present and let the vacuous pass through.
    monkeypatch.setenv("XDG_RUNTIME_DIR", "")
    reason = _systemd_analyze_verify_skip_reason()
    assert reason is not None

    # (d) Binary present, XDG_RUNTIME_DIR set to a nonexistent path: the
    # guarded test must be able to RUN. A directory-existence check would
    # wrongly skip here, on a host where systemd-analyze verify still
    # succeeds against a declared-but-missing runtime dir.
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/tmp/does-not-exist")
    reason = _systemd_analyze_verify_skip_reason()
    assert reason is None


def test_backoff_canary_is_the_shape_systemd_must_warn_about(
    tmp_path: pathlib.Path,
) -> None:
    """_write_backoff_canary must produce a unit systemd will actually warn about.

    This pins the shape of the positive control that
    test_systemd_analyze_verify_reports_no_ignored_directives uses below (its
    docstring point 7): a canary deliberately defective in exactly the way
    systemd's own pairing warning reports, so a verify run that draws no
    warning for it means the invocation checked nothing, not that everything
    is fine.  Pinned here host-independently -- (a) and (c) need no systemd
    runtime -- so it holds even on a host where the canary can never actually
    be FIRED through systemd-analyze.

    (a) The trap is armed and not defused, checked through the repo's own
        backoff invariant rather than by re-deriving the pairing rule by
        hand: the canary must trip assert_restart_backoff_effective's "but
        no RestartSteps=" branch, backed by two direct reads of the
        directives that branch depends on.

    (b) The canary declares an ExecStart=.  LOAD-BEARING, not decoration.
        Measured 2026-08-16 in this worktree (systemd 255.4-1ubuntu8.17): a
        [Service]-only unit carrying just the restart stanza -- ``_INERT_CAP_BODY``,
        the same body test_restart_backoff_guard_rejects_ineffective_units'
        ``no_steps`` case writes -- is REFUSED by systemd-analyze before it
        ever reaches the pairing check::

          canary-noexec.service: Service has no ExecStart=, ExecStop=, or
          SuccessAction=. Refusing.
          Unit canary-noexec.service has a bad unit file setting.

        (exit 1, and ZERO "Ignoring." line for the canary).  Adding
        ``ExecStart=/bin/true`` to the same stanza restores the warning::

          canary-exec.service: Service has RestartMaxDelaySec= but no
          RestartSteps= setting. Ignoring.

        (exit 0).  A canary left in the first shape would make the positive
        control below FAIL on every healthy host: a guard reporting the
        opposite of the truth.

    (c) The written trap is detectable by the same filter the positive
        control uses below (_ignored_directive_lines), read under the
        canary's own path -- connecting the two halves of the mechanism
        rather than pinning them in isolation.  The line asserted here is
        real output measured today, not invented.
    """
    canary = _write_backoff_canary(tmp_path)

    # (a) Armed, not defused.
    with pytest.raises(AssertionError, match="but no RestartSteps="):
        assert_restart_backoff_effective(canary)
    assert restart_directive(canary, "RestartMaxDelaySec") is not None
    assert restart_directive(canary, "RestartSteps") is None

    # (b) ExecStart present -- see docstring point (b) for why this is
    # load-bearing rather than incidental.
    assert restart_directive(canary, "ExecStart") is not None

    # (c) Detectable by the same filter the positive control below uses.
    line = (
        f"{canary.name}: Service has RestartMaxDelaySec= but no "
        "RestartSteps= setting. Ignoring."
    )
    assert _ignored_directive_lines(line, canary) == [line]


@pytest.mark.skipif(
    _SYSTEMD_ANALYZE_SKIP_REASON is not None,
    reason=_SYSTEMD_ANALYZE_SKIP_REASON or "",
)
def test_systemd_analyze_verify_reports_no_ignored_directives(
    tmp_path: pathlib.Path,
) -> None:
    """systemd itself must not report discarding any directive in the unit.

    Confirmatory corroboration of the file-content invariant above, deliberately
    skippable so the module keeps the "no systemd runtime is required" contract
    stated in its docstring.

    Seven details here are load-bearing, each verified by measurement rather
    than assumed (2026-08-01, systemd 255.4 unless noted):

    1. stderr is captured, not just stdout.  The warning goes to stderr while
       stdout is empty, so a stdout-only assertion would be a silent no-op that
       passes forever regardless of the unit's contents.
    2. The assertion is on the WARNING TEXT, never on ``returncode``.
       ``systemd-analyze verify`` exits 1 when the ExecStart binary does not
       exist ("Command /... is not executable"), and this unit bakes in
       /home/leo/.local/bin/uv — so a returncode check would fail on any other
       host for a reason entirely unrelated to this invariant.  (It exits 0 even
       while emitting the warnings this test hunts, so the code carries no
       signal in either direction.)
    3. It runs against HARDCODED only.  The raw template cannot be verified at
       all: its ``__REPO_ROOT__`` sentinel trips a fatal "WorkingDirectory= path
       is not absolute" error.  tests/scripts/test_dashboard_service_template.py::
       test_template_renders_to_hardcoded_file already asserts the two files are
       byte-identical modulo the sentinels, so the property carries across
       without verifying the same bytes twice.
    4. The match is CASE-INSENSITIVE, because systemd spells its two discard
       paths differently and only one of them is the pairing warning:
         - ``t1.service: Service has RestartMaxDelaySec= but no RestartSteps=
           setting. Ignoring.``           (capital I)
         - ``/path/t2.service:8: Unknown key name 'RestartStepz' in section
           'Service', ignoring.``          (lowercase)
       This matters precisely where the guard is needed most.  RestartSteps= AND
       RestartMaxDelaySec= both landed in systemd v254, so on any host below
       that — and scripts/setup-host.sh renders this template onto arbitrary
       machines — the whole backoff fix degrades to two unknown keys, systemd
       emits only the lowercase form, and a capital-I filter would report green
       on a unit doing no backoff at all.  Matching one casing would blind this
       test to its single most important failure mode.
    5. Warnings are SCOPED to the unit under test.  ``systemd-analyze verify``
       loads the unit's dependencies too and prefixes each warning with the
       offending unit's own name, so an unscoped filter blames this file for
       another unit's defect.  Measured: ``--user`` verification of this unit
       emits eight such lines for orchestrator-*.service and fused-memory.service
       (which genuinely lack RestartSteps=) and zero for the dashboard unit.
       Both prefix forms above are covered — bare unit name for a semantic
       warning, full path plus line number for a parse warning.
    6. The test needs a user RUNTIME DIR, not just the binary on PATH, or it
       passes having checked nothing — the same silent-no-op class point 1
       guards against for stdout-only capture, reached here through a
       different missing precondition.  See ``_systemd_analyze_verify_skip_reason``
       above for the measured basis (the observed output strings, exit
       codes, the contexts this bites in, and why DBUS_SESSION_BUS_ADDRESS is
       deliberately not part of the gate) and for the gate itself.
    7. The assertion below is corroborated by a POSITIVE CONTROL, not just a
       skip gate, because a wired gate is not the only way this test can pass
       vacuously — see test_backoff_canary_is_the_shape_systemd_must_warn_about
       for why the control's own shape needed pinning first.  A
       ``df-restart-backoff-canary.service`` built by ``_write_backoff_canary``
       is verified in the SAME invocation as HARDCODED, and the run must draw
       an "Ignoring." line for the canary before the real assertion is trusted
       at all.  Measured 2026-08-16 in this worktree (systemd
       255.4-1ubuntu8.17):
         - A healthy run emits, verbatim: ``df-restart-backoff-canary.service:
           Service has RestartMaxDelaySec= but no RestartSteps= setting.
           Ignoring.``
         - The degenerate run emits it NOT AT ALL: with XDG_RUNTIME_DIR both
           unset and set to ``""``, the combined output for this same
           two-unit invocation collapses to only "Failed to lookup
           RuntimeDirectory path: No such device or address" / "Failed to
           initialize manager: No such device or address" (exit 1) — the
           exact state point 6 describes, and this control fires on any other
           degeneration too, whatever its cause, because it asserts the
           property that actually matters (this invocation can detect a
           defective unit) rather than a specific failure signature.
         - WHY both units are verified in ONE invocation rather than two: a
           control run in a separate subprocess would attest to a different
           run than the one that produced this assertion's evidence.  The two
           are also demonstrably not interchangeable — verifying the canary
           ALONE loads the INSTALLED ``~/.config/systemd/user/dark-factory-dashboard.service``,
           which on this host is stale (``RestartMaxDelaySec=60``, no
           ``RestartSteps=``) and DOES emit the pairing warning under the
           same unit name the filter scopes on; passing HARDCODED's repo path
           explicitly shadows it, measured as zero
           ``dark-factory-dashboard.service:`` lines in the combined run.
         - WHY the canary is passed LAST, after HARDCODED: systemd-analyze
           verify processes positional units in argument order, so an
           "Ignoring." line for the LAST argument implies every earlier
           argument was already reached — making the control transitively
           attest to HARDCODED having actually been verified, not merely to
           the invocation being alive in general.  This is a forward-looking
           hardening rather than a fix for an observed failure: measured
           2026-08-16 in this worktree (systemd 255.4-1ubuntu8.17), passing a
           unit systemd REFUSES outright first (``bad-first.service: Service
           has no ExecStart=, ExecStop=, or SuccessAction=. Refusing.``)
           still lets a well-formed SECOND unit draw its own "Ignoring." line
           in the same invocation, so today's systemd does not stop at the
           first argument either way.  Ordering the canary last removes the
           dependency on that continuing to hold.
         - On a systemd < 254 host the canary would degrade to the lowercase
           full-path "Unknown key name 'RestartMaxDelaySec' ... ignoring."
           form instead, which ``_ignored_directive_lines`` still matches via
           its ``str(unit) in line`` arm, so the control survives the version
           split point 4 exists for.  This is an INFERENCE from point 4's
           measured ``RestartStepz`` behaviour, not a fresh measurement on
           such a host — this host is 255.

    Scoping is what makes ``--user`` safe to pass, and passing it is the point:
    this is a --user unit (WantedBy=default.target, installed under
    ~/.config/systemd/user), so system-scope verification checks it in a manager
    scope it never runs in and silently fails to resolve its user-scope
    dependencies.
    """
    canary = _write_backoff_canary(tmp_path)
    result = subprocess.run(
        ["systemd-analyze", "verify", "--user", str(HARDCODED), str(canary)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    combined = f"{result.stdout}{result.stderr}"
    assert _ignored_directive_lines(combined, canary), (
        f"systemd-analyze verify produced no 'Ignoring.' line for {canary.name}, a unit "
        "built specifically to draw one — so this invocation did not actually verify "
        "anything and the assertion below would pass having checked nothing. Causes, in "
        "likelihood order: no usable --user manager in this environment (the skip gate "
        "above should have caught that); this systemd no longer warns for a "
        "RestartMaxDelaySec= without RestartSteps=; or _ignored_directive_lines has "
        f"regressed.\nFull output:\n{combined}"
    )
    ignored = _ignored_directive_lines(combined, HARDCODED)
    assert not ignored, (
        f"systemd-analyze verify reports directives it is IGNORING in {HARDCODED}:\n"
        + "\n".join(f"  {line}" for line in ignored)
        + "\nEach such line is a directive systemd parsed, warned about, and then "
        "discarded — the unit does not do what its text says it does. An "
        "'Unknown key name' line means this host's systemd predates the "
        "directive (RestartSteps= and RestartMaxDelaySec= both require v254), "
        "so the backoff is inert here even though the file reads correctly."
    )


# ---------------------------------------------------------------------------
# Clean-stop exit-code classification
#
# `uv run` propagates its child's signal death as a numeric exit code, so a
# perfectly graceful SIGTERM stop surfaces to systemd as exit 143 (128+15).
# Under systemd's default classification that is a FAILURE: the journal records
# "Failed with result exit-code" (status=143) and the unit is left in a failed
# state for a stop that was clean, expected and successful — observed
# 2026-07-30 22:23:53 BST on the bounded-drain unit from task 3306, with six
# keep-alive clients attached.  The cost is diagnostic, and compounding: every
# routine restart plants a false failure, so the failed state stops meaning
# anything and a real crash reads exactly like a deploy.
#
# Note there is no systemd-analyze corroboration in this section, unlike the
# restart-backoff one above.  A missing SuccessExitStatus= is a perfectly VALID
# configuration, not a misconfiguration, so systemd emits nothing about it — the
# file-content invariant is the only signal available.
# ---------------------------------------------------------------------------


def _success_exit_statuses(path: pathlib.Path) -> set[str]:
    """Return every exit status *path* declares as success, as a flat token set.

    Two systemd behaviours are load-bearing here and neither is optional:

    1. A single directive accepts a SPACE-SEPARATED LIST
       (``SuccessExitStatus=143 SIGPIPE``), so each value is split.
    2. Repeated directives ACCUMULATE rather than override, so every match is
       collected — ``re.findall``, not ``re.search``.

    A ``re.search(...).group(1)`` parser would satisfy neither: it would read a
    unit that correctly declares ``SuccessExitStatus=SIGPIPE`` on one line and
    ``SuccessExitStatus=143`` on another as missing 143, sending a future author
    to "fix" a unit that is already right.  Same failure mode
    tests/scripts/test_dashboard_unit_uvicorn_timeouts.py::_timeout_stop_sec
    guards against, where a valid ``15s`` must not be misdiagnosed as absent.

    The ``^`` anchor also keeps this reading DIRECTIVES, not prose: both real
    unit files discuss this directive by name in the comment above it, and a
    commented-out mention must never be counted as a live declaration.
    """
    values = re.findall(
        r"^SuccessExitStatus=(.*)$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return {token for value in values for token in value.split()}


def _assert_clean_sigterm_is_success(path: pathlib.Path) -> None:
    """Assert *path* classifies a clean SIGTERM stop (exit 143) as success."""
    statuses = _success_exit_statuses(path)
    assert statuses, (
        f"{path} declares no SuccessExitStatus= at all, so systemd applies its "
        "default classification and records every clean stop as a failure. "
        "`uv run` propagates its child's SIGTERM death as exit code 143 "
        "(128+15), which the journal reports as 'Failed with result exit-code' "
        "(status=143) — observed 2026-07-30 22:23:53 BST on the bounded-drain "
        "unit from task 3306. Declare SuccessExitStatus=143."
    )
    assert "143" in statuses, (
        f"{path} declares SuccessExitStatus={sorted(statuses)}, which does not "
        "include 143 — the status this unit actually exits with on a graceful "
        "stop. `uv run` translates its child's SIGTERM death into exit 143 "
        "(128+15), so a stop that was clean and expected is still recorded as "
        "'Failed with result exit-code'. Declaring some OTHER status successful "
        "does not fix the exit this unit really produces."
    )

    # Context for the trade being made, not an independent property: 143 being
    # "success" is precisely what stops Restart=on-failure re-firing on it, so a
    # reader of this assertion should be able to see the restart policy it
    # interacts with rather than having to go find it.
    restart = restart_directive(path, "Restart")
    assert restart is not None, (
        f"{path} declares SuccessExitStatus but no Restart= policy. "
        "SuccessExitStatus= reclassifies exactly the exits Restart=on-failure "
        "reacts to, so the two belong together; without a restart policy the "
        "directive only relabels the recorded result and this assertion is "
        "pinning an interaction that is not there."
    )


def test_success_exit_status_parser_handles_systemd_spellings(
    tmp_path: pathlib.Path,
) -> None:
    """_success_exit_statuses must handle every spelling systemd itself accepts.

    Per the module convention, the helper earns its own negatives so it cannot
    silently no-op, and each case pins its own failure mode with ``match=``
    rather than a bare ``pytest.raises(AssertionError)`` that any assertion in
    the chain would satisfy.

    The good cases matter as much as the bad ones here: a parser too strict to
    read a valid unit would send a future author to "fix" something already
    correct, which is the more expensive error of the two.
    """
    # Bad: no directive at all — the pre-fix state.
    absent = _write_restart_unit(
        tmp_path / "absent.service", "Restart=on-failure\nRestartSec=5"
    )
    assert _success_exit_statuses(absent) == set()
    with pytest.raises(AssertionError, match="declares no SuccessExitStatus"):
        _assert_clean_sigterm_is_success(absent)

    # Bad: named only in a comment.  Not hypothetical — the real unit files
    # carry an explanatory comment naming this directive directly above it, so a
    # parser that dropped the ^ anchor would report those files as compliant
    # while the live directive had been deleted.  Same trap
    # tests/scripts/test_dashboard_unit_uvicorn_timeouts.py::
    # test_uvicorn_flag_lookup_is_scoped_to_exec_start guards for the flags.
    commented = _write_restart_unit(
        tmp_path / "commented.service",
        "# SuccessExitStatus=143 because `uv run` exits 143 on SIGTERM\n"
        "Restart=on-failure\nRestartSec=5",
    )
    assert _success_exit_statuses(commented) == set(), (
        "a commented-out mention was counted as a live declaration"
    )
    with pytest.raises(AssertionError, match="declares no SuccessExitStatus"):
        _assert_clean_sigterm_is_success(commented)

    # Bad: present, but not the status this unit actually exits with.  Guards
    # against a helper that merely checks the directive exists.
    wrong = _write_restart_unit(
        tmp_path / "wrong.service", "Restart=on-failure\nSuccessExitStatus=75"
    )
    with pytest.raises(AssertionError, match="does not include 143"):
        _assert_clean_sigterm_is_success(wrong)

    # Bad: the right status but no restart policy to interact with.
    no_restart = _write_restart_unit(
        tmp_path / "no_restart.service", "SuccessExitStatus=143"
    )
    with pytest.raises(AssertionError, match="no Restart= policy"):
        _assert_clean_sigterm_is_success(no_restart)

    # Good: the plain form — must not raise.
    plain = _write_restart_unit(
        tmp_path / "plain.service", "Restart=on-failure\nSuccessExitStatus=143"
    )
    _assert_clean_sigterm_is_success(plain)

    # Good: space-separated list on ONE line.  Pins the whitespace split — a
    # whole-value parser would compare the string "143 SIGPIPE" against "143".
    listed = _write_restart_unit(
        tmp_path / "listed.service", "Restart=on-failure\nSuccessExitStatus=143 SIGPIPE"
    )
    assert _success_exit_statuses(listed) == {"143", "SIGPIPE"}
    _assert_clean_sigterm_is_success(listed)

    # Good: split across REPEATED directives, which systemd accumulates.  Pins
    # the findall — a single re.search would read only the SIGPIPE line and
    # declare this correctly-configured unit broken.
    repeated = _write_restart_unit(
        tmp_path / "repeated.service",
        "Restart=on-failure\nSuccessExitStatus=SIGPIPE\nSuccessExitStatus=143",
    )
    assert _success_exit_statuses(repeated) == {"143", "SIGPIPE"}
    _assert_clean_sigterm_is_success(repeated)


def test_clean_sigterm_stop_is_not_recorded_as_a_unit_failure() -> None:
    """Both unit files must classify exit 143 as a successful stop.

    Without it every ordinary `systemctl restart` leaves a "Failed with result
    exit-code" entry in the journal for a shutdown that worked exactly as
    designed, so a genuinely failed unit is indistinguishable from a routine one.
    """
    for path in (TEMPLATE, HARDCODED):
        _assert_clean_sigterm_is_success(path)
