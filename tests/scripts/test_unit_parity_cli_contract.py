"""The frozen CLI contract of the four unit-parity checkers.

Characterization goldens: every scenario runs one checker's ``main`` in-process
over a SYNTHETIC tmp_path tree (never ~/.config/systemd/user, never a committed
unit text, so editing a shipped unit cannot move a golden) and compares the
whole ``(rc, stdout, stderr)`` triple — with ``str(tmp_path)`` spelled
``<TMP>`` — against a literal captured from the checkers BEFORE the parity
driver was lifted into scripts/systemd_unit_parity.py.

A golden holds two kinds of text, and each changes under its own rule:

- What consumers parse. setup-host.sh, orchestrator-watchdog.py and
  install-lms-units.sh read only the exit status, the checker's tag on every
  line, and the orchestrator's ``verdict`` lines. Each of these is also pinned
  apart from the prose: exit status by scenario class in ``LADDER``, the tag
  by ``test_every_output_line_carries_the_tag_consumers_grep_for``, and the
  verdict lines in test_check_orchestrator_unit_parity.py. A red in any of
  them breaks a consumer. Fix the code, or escalate the change as a design
  concern.
- Operator prose: block headers, entries and remedy lines. No consumer parses
  these. Rewording one is a legitimate change: edit the goldens it moves in
  the same commit, and say in the commit message that the wording changed on
  purpose.

Never regenerate goldens wholesale to turn an unexplained red green. Read the
diff first; that is how an unintended output change gets caught.
"""

import dataclasses
import functools
import importlib.util
import os
import pathlib
import types
from collections.abc import Callable, Mapping

import pytest

SCRIPTS_DIR = pathlib.Path(__file__).parents[2] / "scripts"


@functools.cache
def _load(module_name: str) -> types.ModuleType:
    """Load scripts/<module_name>.py by file path (scripts/ is not a package)."""
    path = SCRIPTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ProbeRunner = Callable[[list[str]], tuple[int, str]]


@dataclasses.dataclass(frozen=True)
class Invocation:
    """One in-process CLI call: which checker, its argv, and (lms) its probe."""

    checker: str
    argv: tuple[str, ...]
    probe_runner: ProbeRunner | None = None


def _run(invocation: Invocation, tmp_path: pathlib.Path, capsys) -> tuple[int, str, str]:
    main = _load(invocation.checker).main
    if invocation.probe_runner is None:
        rc = main(list(invocation.argv))
    else:
        rc = main(list(invocation.argv), probe_runner=invocation.probe_runner)
    captured = capsys.readouterr()
    tmp = str(tmp_path)
    return rc, captured.out.replace(tmp, "<TMP>"), captured.err.replace(tmp, "<TMP>")


def _write(path: pathlib.Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _trees(
    tmp_path: pathlib.Path,
    repo_files: Mapping[str, str | bytes],
    installed_files: Mapping[str, str | bytes],
) -> tuple[pathlib.Path, pathlib.Path]:
    """Build ``<tmp>/repo`` and ``<tmp>/installed`` holding exactly the given files."""
    repo_root = tmp_path / "repo"
    installed_dir = tmp_path / "installed"
    repo_root.mkdir(parents=True)
    installed_dir.mkdir(parents=True)
    for relpath, content in repo_files.items():
        _write(repo_root / relpath, content)
    for relpath, content in installed_files.items():
        _write(installed_dir / relpath, content)
    return repo_root, installed_dir


UNDECODABLE = b"[Timer]\nOnBootSec=\xff\xfe\n"
DROPIN_TEXT = "[Service]\nRestartSec=1\n"

# ---------------------------------------------------------------------------
# dashboard  (units committed under <repo>/dashboard/)
# ---------------------------------------------------------------------------

DASH_SERVICE = "dark-factory-dashboard.service"
DASH_TIMER = "dark-factory-dashboard-watchdog.timer"
DASH_SERVICE_TEXT = "[Service]\nType=simple\nRestart=always\n"
DASH_SERVICE_DRIFTED = "[Service]\nType=exec\nRestart=always\n"
DASH_TIMER_TEXT = (
    "[Timer]\nOnBootSec=1min\nOnUnitActiveSec=30s\n\n[Install]\nWantedBy=timers.target\n"
)


def _dashboard(
    tmp_path: pathlib.Path,
    repo: Mapping[str, str | bytes],
    installed: Mapping[str, str | bytes],
    units: tuple[str, ...],
) -> Invocation:
    repo_root, installed_dir = _trees(
        tmp_path, {f"dashboard/{name}": text for name, text in repo.items()}, installed
    )
    argv = ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)]
    for unit in units:
        argv += ["--unit", unit]
    return Invocation("check_dashboard_unit_parity", tuple(argv))


DASHBOARD_SCENARIOS: dict[str, Callable[[pathlib.Path], Invocation]] = {
    "parity": lambda tmp: _dashboard(
        tmp,
        {DASH_SERVICE: DASH_SERVICE_TEXT, DASH_TIMER: DASH_TIMER_TEXT},
        {DASH_SERVICE: DASH_SERVICE_TEXT, DASH_TIMER: DASH_TIMER_TEXT},
        (DASH_SERVICE, DASH_TIMER),
    ),
    "vanished": lambda tmp: _dashboard(
        tmp, {}, {DASH_TIMER: DASH_TIMER_TEXT}, (DASH_TIMER,)
    ),
    "missing": lambda tmp: _dashboard(
        tmp, {DASH_TIMER: DASH_TIMER_TEXT}, {}, (DASH_TIMER,)
    ),
    "override": lambda tmp: _dashboard(
        tmp,
        {DASH_TIMER: DASH_TIMER_TEXT},
        {DASH_TIMER: DASH_TIMER_TEXT, f"{DASH_TIMER}.d/override.conf": DROPIN_TEXT},
        (DASH_TIMER,),
    ),
    "drift": lambda tmp: _dashboard(
        tmp, {DASH_SERVICE: DASH_SERVICE_TEXT}, {DASH_SERVICE: DASH_SERVICE_DRIFTED}, (DASH_SERVICE,)
    ),
    "drift+absent": lambda tmp: _dashboard(
        tmp,
        {DASH_SERVICE: DASH_SERVICE_TEXT, DASH_TIMER: DASH_TIMER_TEXT},
        {DASH_SERVICE: DASH_SERVICE_DRIFTED},
        (DASH_SERVICE, DASH_TIMER),
    ),
    "zero-compared": lambda tmp: _dashboard(tmp, {}, {}, ()),
}

# ---------------------------------------------------------------------------
# orchestrator  (units committed under <repo>/scripts/)
# ---------------------------------------------------------------------------

ORCH_TIMER = "orchestrator-watchdog.timer"
ORCH_SERVICE = "orchestrator-watchdog.service"
ORCH_DF = "orchestrator-dark-factory.service"
ORCH_REIFY = "orchestrator-reify.service"
ORCH_VIDEO = "orchestrator-autopilot-video.service"
ORCH_TIMER_TEXT = "[Timer]\nOnBootSec=2min\nOnUnitActiveSec=5min\n\n[Install]\nWantedBy=timers.target\n"
ORCH_SERVICE_TEXT = "[Service]\nType=oneshot\nTimeoutStartSec=60\n"
ORCH_SERVICE_DRIFTED = "[Service]\nType=simple\nTimeoutStartSec=60\nAccuracySec=5s\n"


def _orchestrator(
    tmp_path: pathlib.Path,
    repo: Mapping[str, str | bytes],
    installed: Mapping[str, str | bytes],
    units: tuple[str, ...],
    *,
    print_verdicts: bool = False,
) -> Invocation:
    repo_root, installed_dir = _trees(
        tmp_path, {f"scripts/{name}": text for name, text in repo.items()}, installed
    )
    argv = ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)]
    for unit in units:
        argv += ["--unit", unit]
    if print_verdicts:
        argv.append("--print-verdicts")
    return Invocation("check_orchestrator_unit_parity", tuple(argv))


def _orchestrator_mixed_verdicts(tmp_path: pathlib.Path) -> Invocation:
    """One run whose five units are clean, absent, drift+override, vanished, unreadable."""
    return _orchestrator(
        tmp_path,
        {
            ORCH_TIMER: ORCH_TIMER_TEXT,
            ORCH_SERVICE: ORCH_SERVICE_TEXT,
            ORCH_DF: ORCH_SERVICE_TEXT,
            ORCH_VIDEO: ORCH_TIMER_TEXT,
        },
        {
            ORCH_TIMER: ORCH_TIMER_TEXT,
            ORCH_DF: ORCH_SERVICE_DRIFTED,
            f"{ORCH_DF}.d/override.conf": DROPIN_TEXT,
            ORCH_REIFY: ORCH_SERVICE_TEXT,
            ORCH_VIDEO: UNDECODABLE,
        },
        (ORCH_VIDEO, ORCH_DF, ORCH_REIFY, ORCH_SERVICE, ORCH_TIMER),
        print_verdicts=True,
    )


ORCHESTRATOR_SCENARIOS: dict[str, Callable[[pathlib.Path], Invocation]] = {
    "parity": lambda tmp: _orchestrator(
        tmp,
        {ORCH_SERVICE: ORCH_SERVICE_TEXT, ORCH_TIMER: ORCH_TIMER_TEXT},
        {ORCH_SERVICE: ORCH_SERVICE_TEXT, ORCH_TIMER: ORCH_TIMER_TEXT},
        (ORCH_SERVICE, ORCH_TIMER),
    ),
    "vanished": lambda tmp: _orchestrator(
        tmp, {}, {ORCH_TIMER: ORCH_TIMER_TEXT}, (ORCH_TIMER,)
    ),
    "missing": lambda tmp: _orchestrator(
        tmp, {ORCH_TIMER: ORCH_TIMER_TEXT}, {}, (ORCH_TIMER,)
    ),
    "override": lambda tmp: _orchestrator(
        tmp,
        {ORCH_TIMER: ORCH_TIMER_TEXT},
        {ORCH_TIMER: ORCH_TIMER_TEXT, f"{ORCH_TIMER}.d/override.conf": DROPIN_TEXT},
        (ORCH_TIMER,),
    ),
    "drift": lambda tmp: _orchestrator(
        tmp, {ORCH_SERVICE: ORCH_SERVICE_TEXT}, {ORCH_SERVICE: ORCH_SERVICE_DRIFTED}, (ORCH_SERVICE,)
    ),
    "drift+absent": lambda tmp: _orchestrator(
        tmp,
        {ORCH_SERVICE: ORCH_SERVICE_TEXT, ORCH_TIMER: ORCH_TIMER_TEXT},
        {ORCH_SERVICE: ORCH_SERVICE_DRIFTED},
        (ORCH_SERVICE, ORCH_TIMER),
    ),
    "zero-compared": lambda tmp: _orchestrator(tmp, {}, {}, ()),
    "unreadable": lambda tmp: _orchestrator(
        tmp, {ORCH_TIMER: ORCH_TIMER_TEXT}, {ORCH_TIMER: UNDECODABLE}, (ORCH_TIMER,)
    ),
    "verdicts-clean": lambda tmp: _orchestrator(
        tmp, {ORCH_TIMER: ORCH_TIMER_TEXT}, {ORCH_TIMER: ORCH_TIMER_TEXT}, (ORCH_TIMER,),
        print_verdicts=True,
    ),
    "verdicts-mixed": _orchestrator_mixed_verdicts,
}

# ---------------------------------------------------------------------------
# lms  (one template committed under <repo>/scripts/local-model-serving/)
# ---------------------------------------------------------------------------

LMS_UNIT = "lms-arm@.service"
LMS_RELPATH = f"scripts/local-model-serving/{LMS_UNIT}"
LMS_TEMPLATE = (
    "[Unit]\nDescription=golden arm\n\n"
    "[Service]\nType=exec\nWorkingDirectory=/srv/lms-golden\n\n"
    "[Install]\nWantedBy=default.target\n"
)
LMS_DRIFTED = LMS_TEMPLATE.replace("Type=exec", "Type=simple")
CLEAN_PROBE_STDOUT = "WorkingDirectory=/srv/lms-golden\nDropInPaths=\n"


def _probe(stdout: str, returncode: int = 0) -> ProbeRunner:
    def runner(argv: list[str]) -> tuple[int, str]:
        return returncode, stdout

    return runner


def _lms(
    tmp_path: pathlib.Path,
    *,
    repo: str | bytes | None = LMS_TEMPLATE,
    installed: Mapping[str, str | bytes] | None = None,
    probe_stdout: str = CLEAN_PROBE_STDOUT,
    probe_rc: int = 0,
    installed_dir_arg: pathlib.Path | None = None,
) -> Invocation:
    repo_root, installed_dir = _trees(
        tmp_path,
        {} if repo is None else {LMS_RELPATH: repo},
        {LMS_UNIT: LMS_TEMPLATE} if installed is None else installed,
    )
    argv = (
        "--repo-root", str(repo_root),
        "--installed-dir", str(installed_dir_arg or installed_dir),
    )
    return Invocation("check_lms_unit_parity", argv, _probe(probe_stdout, probe_rc))


def _lms_symlinked_dropin(tmp_path: pathlib.Path) -> Invocation:
    """The glob sees the drop-in through a symlinked dir; systemd names its real path."""
    real_dropin = tmp_path / "installed" / f"{LMS_UNIT}.d" / "10-x.conf"
    link = tmp_path / "link"
    invocation = _lms(
        tmp_path,
        installed={LMS_UNIT: LMS_TEMPLATE, f"{LMS_UNIT}.d/10-x.conf": DROPIN_TEXT},
        probe_stdout=f"WorkingDirectory=/srv/lms-golden\nDropInPaths={real_dropin}\n",
        installed_dir_arg=link,
    )
    os.symlink(tmp_path / "installed", link)
    return invocation


LMS_SCENARIOS: dict[str, Callable[[pathlib.Path], Invocation]] = {
    "parity": lambda tmp: _lms(tmp),
    "vanished": lambda tmp: _lms(tmp, repo=None),
    "missing": lambda tmp: _lms(tmp, installed={}),
    "override": lambda tmp: _lms(
        tmp, installed={LMS_UNIT: LMS_TEMPLATE, f"{LMS_UNIT}.d/10-x.conf": DROPIN_TEXT}
    ),
    "drift": lambda tmp: _lms(tmp, installed={LMS_UNIT: LMS_DRIFTED}),
    "systemd-only-dropin": lambda tmp: _lms(
        tmp,
        probe_stdout=(
            "WorkingDirectory=/srv/lms-golden\n"
            "DropInPaths=/etc/systemd/user/lms-arm@.service.d/50-site.conf\n"
        ),
    ),
    "symlinked-dropin": _lms_symlinked_dropin,
    "effective": lambda tmp: _lms(
        tmp, probe_stdout="WorkingDirectory=/srv/elsewhere\nDropInPaths=\n"
    ),
    "unverifiable": lambda tmp: _lms(tmp, probe_stdout="", probe_rc=1),
}

# ---------------------------------------------------------------------------
# fused-memory  (single unit: --installed FILE --template FILE)
# ---------------------------------------------------------------------------


def _fm_unit(service_lines: tuple[str, ...]) -> str:
    return "[Unit]\nDescription=golden\n\n[Service]\n" + "".join(f"{line}\n" for line in service_lines)


FM_CLEAN = _fm_unit(
    (
        *_load("check_fused_memory_unit_parity").REQUIRED_SERVICE_DIRECTIVES,
        "ExecStartPre=/bin/true",
        "ExecStopPost=/bin/true",
    )
)
FM_DRIFTED = _fm_unit(("Restart=on-failure", "RestartSec=5", "ExecStartPre=/bin/true"))
FM_TEMPLATE_DRIFTED = FM_CLEAN.replace("WatchdogSec=120\n", "")


def _fused_memory(
    tmp_path: pathlib.Path,
    *,
    installed: str | None = FM_CLEAN,
    template: str | None = FM_CLEAN,
    dropin: bool = False,
) -> Invocation:
    installed_path = tmp_path / "installed" / "fused-memory.service"
    template_path = tmp_path / "repo" / "fused-memory.service.template"
    if installed is not None:
        _write(installed_path, installed)
    if template is not None:
        _write(template_path, template)
    if dropin:
        _write(installed_path.parent / "fused-memory.service.d" / "override.conf", DROPIN_TEXT)
    argv = ("--installed", str(installed_path), "--template", str(template_path))
    return Invocation("check_fused_memory_unit_parity", argv)


FUSED_MEMORY_SCENARIOS: dict[str, Callable[[pathlib.Path], Invocation]] = {
    "parity": lambda tmp: _fused_memory(tmp),
    "missing": lambda tmp: _fused_memory(tmp, installed=None),
    "drift": lambda tmp: _fused_memory(tmp, installed=FM_DRIFTED),
    "override": lambda tmp: _fused_memory(tmp, dropin=True),
    "override+drift": lambda tmp: _fused_memory(tmp, installed=FM_DRIFTED, dropin=True),
    "template-missing": lambda tmp: _fused_memory(tmp, template=None),
    "template-drift": lambda tmp: _fused_memory(tmp, template=FM_TEMPLATE_DRIFTED),
}

SCENARIOS: dict[str, Callable[[pathlib.Path], Invocation]] = {
    f"{checker}/{name}": build
    for checker, table in (
        ("dashboard", DASHBOARD_SCENARIOS),
        ("orchestrator", ORCHESTRATOR_SCENARIOS),
        ("lms", LMS_SCENARIOS),
        ("fused-memory", FUSED_MEMORY_SCENARIOS),
    )
    for name, build in table.items()
}

# ---------------------------------------------------------------------------
# The goldens
# ---------------------------------------------------------------------------

GOLDENS: dict[str, tuple[int, str, str]] = {
    'dashboard/drift': (
        1,
        (
            '[dashboard_unit_parity] [drift] 1 directive(s) differ between repo and installed units:\n'
            '[dashboard_unit_parity]   dark-factory-dashboard.service [Service] Type\n'
            '[dashboard_unit_parity]       value differs between the repo copy and the installed copy\n'
            '[dashboard_unit_parity]       repo      <TMP>/repo/dashboard/dark-factory-dashboard.service: simple\n'
            '[dashboard_unit_parity]       installed <TMP>/installed/dark-factory-dashboard.service: exec\n'
            '[dashboard_unit_parity] [drift] To propagate the committed units to this host, run: scripts/setup-host.sh  (this checker is read-only by design — see the module docstring for why there is no --fix)\n'
        ),
        "",
    ),
    'dashboard/drift+absent': (
        1,
        (
            '[dashboard_unit_parity] [skip] installed unit not found: <TMP>/installed/dark-factory-dashboard-watchdog.timer (not installed on this host)\n'
            '[dashboard_unit_parity] [drift] 1 directive(s) differ between repo and installed units:\n'
            '[dashboard_unit_parity]   dark-factory-dashboard.service [Service] Type\n'
            '[dashboard_unit_parity]       value differs between the repo copy and the installed copy\n'
            '[dashboard_unit_parity]       repo      <TMP>/repo/dashboard/dark-factory-dashboard.service: simple\n'
            '[dashboard_unit_parity]       installed <TMP>/installed/dark-factory-dashboard.service: exec\n'
            '[dashboard_unit_parity] [drift] To propagate the committed units to this host, run: scripts/setup-host.sh  (this checker is read-only by design — see the module docstring for why there is no --fix)\n'
        ),
        "",
    ),
    'dashboard/missing': (
        2,
        (
            '[dashboard_unit_parity] [skip] installed unit not found: <TMP>/installed/dark-factory-dashboard-watchdog.timer (not installed on this host)\n'
        ),
        "",
    ),
    'dashboard/override': (
        1,
        (
            '[dashboard_unit_parity] [override] 1 unit(s) have drop-in overrides — the unit files were compared, but the EFFECTIVE configuration was NOT verified:\n'
            '[dashboard_unit_parity]   dark-factory-dashboard-watchdog.timer: <TMP>/installed/dark-factory-dashboard-watchdog.timer.d/override.conf\n'
            '[dashboard_unit_parity] [override] systemd merges these over the unit at load time, so a directive set here silently wins over the committed value. Inspect with: systemctl --user cat <unit>  (and remove the drop-in, or move the setting into the committed unit).\n'
        ),
        "",
    ),
    'dashboard/parity': (
        0,
        (
            '[dashboard_unit_parity] [ok] parity — 2 unit(s) match their committed copies.\n'
        ),
        "",
    ),
    'dashboard/vanished': (
        1,
        (
            '[dashboard_unit_parity] [vanished] 1 committed unit(s) not found — nothing was verified for them:\n'
            '[dashboard_unit_parity]   dark-factory-dashboard-watchdog.timer: expected committed copy at <TMP>/repo/dashboard/dark-factory-dashboard-watchdog.timer\n'
            "[dashboard_unit_parity] [vanished] The committed unit is this checker's source of truth. Check --repo-root, and whether the unit was renamed or moved (the paths live in UNITS in this script).\n"
        ),
        "",
    ),
    'dashboard/zero-compared': (
        1,
        (
            '[dashboard_unit_parity] [vanished] 3 committed unit(s) not found — nothing was verified for them:\n'
            '[dashboard_unit_parity]   dark-factory-dashboard-watchdog.service: expected committed copy at <TMP>/repo/dashboard/dark-factory-dashboard-watchdog.service\n'
            '[dashboard_unit_parity]   dark-factory-dashboard-watchdog.timer: expected committed copy at <TMP>/repo/dashboard/dark-factory-dashboard-watchdog.timer\n'
            '[dashboard_unit_parity]   dark-factory-dashboard.service: expected committed copy at <TMP>/repo/dashboard/dark-factory-dashboard.service\n'
            "[dashboard_unit_parity] [vanished] The committed unit is this checker's source of truth. Check --repo-root, and whether the unit was renamed or moved (the paths live in UNITS in this script).\n"
        ),
        "",
    ),
    'fused-memory/drift': (
        1,
        (
            '[fused_memory_unit_parity] [drift] <TMP>/installed/fused-memory.service: missing required directives:\n'
            '[fused_memory_unit_parity]   - Environment=MEM0_TELEMETRY=false\n'
            '[fused_memory_unit_parity]   - Environment=PATH=%h/.local/bin:%h/bin:%h/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n'
            '[fused_memory_unit_parity]   - WatchdogSec=120\n'
            '[fused_memory_unit_parity]   - RestartSteps=4\n'
            '[fused_memory_unit_parity]   - TimeoutStartSec=300\n'
            '[fused_memory_unit_parity]   - TimeoutStopSec=90\n'
            '[fused_memory_unit_parity]   - ExecStopPost=\n'
        ),
        (
            '[fused_memory_unit_parity] Run with --fix to append missing directives without clobbering host-specific lines.\n'
        ),
    ),
    'fused-memory/missing': (
        2,
        "",
        (
            '[fused_memory_unit_parity] [skip] Installed unit not found at <TMP>/installed/fused-memory.service (unit may not be installed on this host)\n'
        ),
    ),
    'fused-memory/override': (
        1,
        (
            '[fused_memory_unit_parity] [override] 1 drop-in(s) apply to <TMP>/installed/fused-memory.service — the unit file was compared, but the EFFECTIVE configuration was NOT verified by that comparison:\n'
            '[fused_memory_unit_parity]   <TMP>/installed/fused-memory.service.d/override.conf\n'
            '[fused_memory_unit_parity] [override] systemd merges these over the unit at load time, so a directive set here silently wins over the committed value. Inspect the merged result with: systemctl --user cat fused-memory.service\n'
            '[fused_memory_unit_parity] [override] Nothing was removed: this checker is read-only about drop-ins BY DESIGN, because a drop-in can be load-bearing (task 3750). --fix appends to the unit FILE and can neither synthesize nor resolve an override living in a different one — remove it by hand, or move the setting into the committed template.\n'
        ),
        "",
    ),
    'fused-memory/override+drift': (
        1,
        (
            '[fused_memory_unit_parity] [override] 1 drop-in(s) apply to <TMP>/installed/fused-memory.service — the unit file was compared, but the EFFECTIVE configuration was NOT verified by that comparison:\n'
            '[fused_memory_unit_parity]   <TMP>/installed/fused-memory.service.d/override.conf\n'
            '[fused_memory_unit_parity] [override] systemd merges these over the unit at load time, so a directive set here silently wins over the committed value. Inspect the merged result with: systemctl --user cat fused-memory.service\n'
            '[fused_memory_unit_parity] [override] Nothing was removed: this checker is read-only about drop-ins BY DESIGN, because a drop-in can be load-bearing (task 3750). --fix appends to the unit FILE and can neither synthesize nor resolve an override living in a different one — remove it by hand, or move the setting into the committed template.\n'
            '[fused_memory_unit_parity] [drift] <TMP>/installed/fused-memory.service: missing required directives:\n'
            '[fused_memory_unit_parity]   - Environment=MEM0_TELEMETRY=false\n'
            '[fused_memory_unit_parity]   - Environment=PATH=%h/.local/bin:%h/bin:%h/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n'
            '[fused_memory_unit_parity]   - WatchdogSec=120\n'
            '[fused_memory_unit_parity]   - RestartSteps=4\n'
            '[fused_memory_unit_parity]   - TimeoutStartSec=300\n'
            '[fused_memory_unit_parity]   - TimeoutStopSec=90\n'
            '[fused_memory_unit_parity]   - ExecStopPost=\n'
        ),
        (
            '[fused_memory_unit_parity] Run with --fix to append missing directives without clobbering host-specific lines.\n'
        ),
    ),
    'fused-memory/parity': (
        0,
        (
            '[fused_memory_unit_parity] [ok] <TMP>/installed/fused-memory.service: parity — all required directives present.\n'
        ),
        "",
    ),
    'fused-memory/template-drift': (
        0,
        (
            '[fused_memory_unit_parity] [ok] <TMP>/installed/fused-memory.service: parity — all required directives present.\n'
        ),
        (
            "[fused_memory_unit_parity] [warn] Template <TMP>/repo/fused-memory.service.template is itself missing: ['WatchdogSec=120']\n"
        ),
    ),
    'fused-memory/template-missing': (
        0,
        (
            '[fused_memory_unit_parity] [ok] <TMP>/installed/fused-memory.service: parity — all required directives present.\n'
        ),
        "",
    ),
    'lms/drift': (
        1,
        (
            '[lms_unit_parity] [drift] 1 directive(s) differ between the committed and installed units:\n'
            '[lms_unit_parity]   lms-arm@.service [Service] Type\n'
            '[lms_unit_parity]       value differs between the committed and installed copies\n'
            '[lms_unit_parity]       committed <TMP>/repo/scripts/local-model-serving/lms-arm@.service: exec\n'
            '[lms_unit_parity]       installed <TMP>/installed/lms-arm@.service: simple\n'
            '[lms_unit_parity] [drift] To propagate the committed template to this host, run: scripts/local-model-serving/install-lms-units.sh  (this checker is read-only by design — see the module docstring for why there is no --fix)\n'
        ),
        "",
    ),
    'lms/effective': (
        1,
        (
            '[lms_unit_parity] [effective] 1 unit(s) resolve to a WorkingDirectory that is NOT the one the committed template declares:\n'
            '[lms_unit_parity]   lms-arm@.service\n'
            '[lms_unit_parity]       committed template: /srv/lms-golden\n'
            '[lms_unit_parity]       systemd resolved:   /srv/elsewhere\n'
            '[lms_unit_parity] [effective] An arm started now would run from the resolved path, not the committed one. Inspect with: systemctl --user cat lms-arm@<arm>.service\n'
        ),
        "",
    ),
    'lms/missing': (
        2,
        (
            '[lms_unit_parity] [skip] installed unit not found: <TMP>/installed/lms-arm@.service (not installed on this host)\n'
        ),
        "",
    ),
    'lms/override': (
        1,
        (
            '[lms_unit_parity] [override] 1 unit(s) have drop-in overrides — the unit files were compared, but the EFFECTIVE configuration was NOT verified by that comparison:\n'
            '[lms_unit_parity]   lms-arm@.service: <TMP>/installed/lms-arm@.service.d/10-x.conf\n'
            '[lms_unit_parity] [override] systemd merges these over the unit at load time, so a directive set here silently wins over the committed value. Inspect the merged result with: systemctl --user cat lms-arm@<arm>.service\n'
            '[lms_unit_parity] [override] Nothing was removed: this checker is read-only BY DESIGN, because a drop-in can be load-bearing (task 3750). To remove the known worktree drop-in, run scripts/remove-lms-arm-worktree-dropin.sh, which gates removal behind the preconditions that make it safe.\n'
        ),
        "",
    ),
    'lms/parity': (
        0,
        (
            '[lms_unit_parity] [effective] lms-arm@.service: the committed template IS the effective configuration — systemd resolves WorkingDirectory=/srv/lms-golden and merges no drop-in.\n'
            '[lms_unit_parity] [ok] parity — 1 unit(s) match their committed copies.\n'
        ),
        "",
    ),
    'lms/symlinked-dropin': (
        1,
        (
            '[lms_unit_parity] [override] 1 unit(s) have drop-in overrides — the unit files were compared, but the EFFECTIVE configuration was NOT verified by that comparison:\n'
            '[lms_unit_parity]   lms-arm@.service: <TMP>/link/lms-arm@.service.d/10-x.conf\n'
            '[lms_unit_parity] [override] systemd merges these over the unit at load time, so a directive set here silently wins over the committed value. Inspect the merged result with: systemctl --user cat lms-arm@<arm>.service\n'
            '[lms_unit_parity] [override] Nothing was removed: this checker is read-only BY DESIGN, because a drop-in can be load-bearing (task 3750). To remove the known worktree drop-in, run scripts/remove-lms-arm-worktree-dropin.sh, which gates removal behind the preconditions that make it safe.\n'
        ),
        "",
    ),
    'lms/systemd-only-dropin': (
        1,
        (
            '[lms_unit_parity] [override] 1 unit(s) have drop-in overrides — the unit files were compared, but the EFFECTIVE configuration was NOT verified by that comparison:\n'
            '[lms_unit_parity]   lms-arm@.service: /etc/systemd/user/lms-arm@.service.d/50-site.conf\n'
            '[lms_unit_parity] [override] systemd merges these over the unit at load time, so a directive set here silently wins over the committed value. Inspect the merged result with: systemctl --user cat lms-arm@<arm>.service\n'
            '[lms_unit_parity] [override] Nothing was removed: this checker is read-only BY DESIGN, because a drop-in can be load-bearing (task 3750). To remove the known worktree drop-in, run scripts/remove-lms-arm-worktree-dropin.sh, which gates removal behind the preconditions that make it safe.\n'
        ),
        "",
    ),
    'lms/unverifiable': (
        1,
        (
            '[lms_unit_parity] [unverifiable] could not verify the effective configuration for 1 unit(s):\n'
            '[lms_unit_parity]   lms-arm@.service: `systemctl --user show -p WorkingDirectory -p DropInPaths lms-arm@probe.service` exited 1\n'
            '[lms_unit_parity] [unverifiable] The unit files may well match; what this run could NOT establish is what systemd would actually apply. Check that a systemd user manager is running for this user (systemctl --user is-system-running).\n'
        ),
        "",
    ),
    'lms/vanished': (
        1,
        (
            '[lms_unit_parity] [vanished] 1 committed unit(s) not found — nothing was verified for them:\n'
            '[lms_unit_parity]   lms-arm@.service: expected committed copy at <TMP>/repo/scripts/local-model-serving/lms-arm@.service\n'
            "[lms_unit_parity] [vanished] The committed template is this checker's source of truth. Check --repo-root, and whether the unit was renamed or moved (the paths live in UNITS in this script).\n"
        ),
        "",
    ),
    'orchestrator/drift': (
        1,
        (
            '[orchestrator_unit_parity] [drift] 2 directive(s) differ between repo and installed units:\n'
            '[orchestrator_unit_parity]   orchestrator-watchdog.service [Service] AccuracySec\n'
            '[orchestrator_unit_parity]       AccuracySec declared in the installed copy, absent from the repo copy\n'
            '[orchestrator_unit_parity]       repo      <TMP>/repo/scripts/orchestrator-watchdog.service: <absent>\n'
            '[orchestrator_unit_parity]       installed <TMP>/installed/orchestrator-watchdog.service: 5s\n'
            '[orchestrator_unit_parity]   orchestrator-watchdog.service [Service] Type\n'
            '[orchestrator_unit_parity]       Type values differ\n'
            '[orchestrator_unit_parity]       repo      <TMP>/repo/scripts/orchestrator-watchdog.service: oneshot\n'
            '[orchestrator_unit_parity]       installed <TMP>/installed/orchestrator-watchdog.service: simple\n'
            '[orchestrator_unit_parity] [drift] To propagate the committed units to this host, run: DF_INSTALL_ORCH_UNITS=1 bash scripts/setup-host.sh  (a plain run runs this same check and DECLINES the unit install on drift, because the committed side is not always the correct one — see below. This checker is read-only by design; the module docstring says why there is no --fix.)\n'
            "[orchestrator_unit_parity] [drift] BUT CHECK DIRECTION FIRST: drift does not mean the installed copy is the stale one. Confirm the committed value is the one you want on this host before installing — see the KNOWN RED section of this script's module docstring.\n"
        ),
        "",
    ),
    'orchestrator/drift+absent': (
        1,
        (
            '[orchestrator_unit_parity] [skip] orchestrator-watchdog.timer: installed unit not found: <TMP>/installed/orchestrator-watchdog.timer (not installed on this host)\n'
            '[orchestrator_unit_parity] [drift] 2 directive(s) differ between repo and installed units:\n'
            '[orchestrator_unit_parity]   orchestrator-watchdog.service [Service] AccuracySec\n'
            '[orchestrator_unit_parity]       AccuracySec declared in the installed copy, absent from the repo copy\n'
            '[orchestrator_unit_parity]       repo      <TMP>/repo/scripts/orchestrator-watchdog.service: <absent>\n'
            '[orchestrator_unit_parity]       installed <TMP>/installed/orchestrator-watchdog.service: 5s\n'
            '[orchestrator_unit_parity]   orchestrator-watchdog.service [Service] Type\n'
            '[orchestrator_unit_parity]       Type values differ\n'
            '[orchestrator_unit_parity]       repo      <TMP>/repo/scripts/orchestrator-watchdog.service: oneshot\n'
            '[orchestrator_unit_parity]       installed <TMP>/installed/orchestrator-watchdog.service: simple\n'
            '[orchestrator_unit_parity] [drift] To propagate the committed units to this host, run: DF_INSTALL_ORCH_UNITS=1 bash scripts/setup-host.sh  (a plain run runs this same check and DECLINES the unit install on drift, because the committed side is not always the correct one — see below. This checker is read-only by design; the module docstring says why there is no --fix.)\n'
            "[orchestrator_unit_parity] [drift] BUT CHECK DIRECTION FIRST: drift does not mean the installed copy is the stale one. Confirm the committed value is the one you want on this host before installing — see the KNOWN RED section of this script's module docstring.\n"
        ),
        "",
    ),
    'orchestrator/missing': (
        2,
        (
            '[orchestrator_unit_parity] [skip] orchestrator-watchdog.timer: installed unit not found: <TMP>/installed/orchestrator-watchdog.timer (not installed on this host)\n'
        ),
        "",
    ),
    'orchestrator/override': (
        1,
        (
            '[orchestrator_unit_parity] [override] 1 unit(s) have drop-in overrides — the unit files were compared, but the EFFECTIVE configuration was NOT verified:\n'
            '[orchestrator_unit_parity]   orchestrator-watchdog.timer: <TMP>/installed/orchestrator-watchdog.timer.d/override.conf\n'
            '[orchestrator_unit_parity] [override] systemd merges these over the unit at load time, so a directive set here silently wins over the committed value. Inspect with: systemctl --user cat <unit>  (and remove the drop-in, or move the setting into the committed unit).\n'
        ),
        "",
    ),
    'orchestrator/parity': (
        0,
        (
            '[orchestrator_unit_parity] [ok] parity — 2 unit(s) match their committed copies.\n'
        ),
        "",
    ),
    'orchestrator/unreadable': (
        1,
        (
            '[orchestrator_unit_parity] [unreadable] 1 unit file(s) could not be read — nothing was verified for them:\n'
            "[orchestrator_unit_parity]   orchestrator-watchdog.timer: <TMP>/installed/orchestrator-watchdog.timer: 'utf-8' codec can't decode byte 0xff in position 18: invalid start byte\n"
            '[orchestrator_unit_parity] [unreadable] The file exists but could not be decoded as UTF-8 or opened. Check its permissions and encoding — this is not a directive diff, so there is nothing to propagate.\n'
        ),
        "",
    ),
    'orchestrator/vanished': (
        1,
        (
            '[orchestrator_unit_parity] [vanished] 1 committed unit(s) not found — nothing was verified for them:\n'
            '[orchestrator_unit_parity]   orchestrator-watchdog.timer: expected committed copy at <TMP>/repo/scripts/orchestrator-watchdog.timer\n'
            "[orchestrator_unit_parity] [vanished] The committed unit is this checker's source of truth. Check --repo-root, and whether the unit was renamed or moved (the paths live in UNITS in this script).\n"
        ),
        "",
    ),
    'orchestrator/verdicts-clean': (
        0,
        (
            '[orchestrator_unit_parity] verdict orchestrator-watchdog.timer clean\n'
            '[orchestrator_unit_parity] [ok] parity — 1 unit(s) match their committed copies.\n'
        ),
        "",
    ),
    'orchestrator/verdicts-mixed': (
        1,
        (
            '[orchestrator_unit_parity] verdict orchestrator-autopilot-video.service unreadable\n'
            '[orchestrator_unit_parity] verdict orchestrator-dark-factory.service drift,override\n'
            '[orchestrator_unit_parity] verdict orchestrator-reify.service vanished\n'
            '[orchestrator_unit_parity] verdict orchestrator-watchdog.service absent\n'
            '[orchestrator_unit_parity] verdict orchestrator-watchdog.timer clean\n'
            '[orchestrator_unit_parity] [skip] orchestrator-watchdog.service: installed unit not found: <TMP>/installed/orchestrator-watchdog.service (not installed on this host)\n'
            '[orchestrator_unit_parity] [vanished] 1 committed unit(s) not found — nothing was verified for them:\n'
            '[orchestrator_unit_parity]   orchestrator-reify.service: expected committed copy at <TMP>/repo/scripts/orchestrator-reify.service\n'
            "[orchestrator_unit_parity] [vanished] The committed unit is this checker's source of truth. Check --repo-root, and whether the unit was renamed or moved (the paths live in UNITS in this script).\n"
            '[orchestrator_unit_parity] [unreadable] 1 unit file(s) could not be read — nothing was verified for them:\n'
            "[orchestrator_unit_parity]   orchestrator-autopilot-video.service: <TMP>/installed/orchestrator-autopilot-video.service: 'utf-8' codec can't decode byte 0xff in position 18: invalid start byte\n"
            '[orchestrator_unit_parity] [unreadable] The file exists but could not be decoded as UTF-8 or opened. Check its permissions and encoding — this is not a directive diff, so there is nothing to propagate.\n'
            '[orchestrator_unit_parity] [override] 1 unit(s) have drop-in overrides — the unit files were compared, but the EFFECTIVE configuration was NOT verified:\n'
            '[orchestrator_unit_parity]   orchestrator-dark-factory.service: <TMP>/installed/orchestrator-dark-factory.service.d/override.conf\n'
            '[orchestrator_unit_parity] [override] systemd merges these over the unit at load time, so a directive set here silently wins over the committed value. Inspect with: systemctl --user cat <unit>  (and remove the drop-in, or move the setting into the committed unit).\n'
            '[orchestrator_unit_parity] [drift] 2 directive(s) differ between repo and installed units:\n'
            '[orchestrator_unit_parity]   orchestrator-dark-factory.service [Service] AccuracySec\n'
            '[orchestrator_unit_parity]       AccuracySec declared in the installed copy, absent from the repo copy\n'
            '[orchestrator_unit_parity]       repo      <TMP>/repo/scripts/orchestrator-dark-factory.service: <absent>\n'
            '[orchestrator_unit_parity]       installed <TMP>/installed/orchestrator-dark-factory.service: 5s\n'
            '[orchestrator_unit_parity]   orchestrator-dark-factory.service [Service] Type\n'
            '[orchestrator_unit_parity]       Type values differ\n'
            '[orchestrator_unit_parity]       repo      <TMP>/repo/scripts/orchestrator-dark-factory.service: oneshot\n'
            '[orchestrator_unit_parity]       installed <TMP>/installed/orchestrator-dark-factory.service: simple\n'
            '[orchestrator_unit_parity] [drift] To propagate the committed units to this host, run: DF_INSTALL_ORCH_UNITS=1 bash scripts/setup-host.sh  (a plain run runs this same check and DECLINES the unit install on drift, because the committed side is not always the correct one — see below. This checker is read-only by design; the module docstring says why there is no --fix.)\n'
            "[orchestrator_unit_parity] [drift] BUT CHECK DIRECTION FIRST: drift does not mean the installed copy is the stale one. Confirm the committed value is the one you want on this host before installing — see the KNOWN RED section of this script's module docstring.\n"
        ),
        "",
    ),
    'orchestrator/zero-compared': (
        1,
        (
            '[orchestrator_unit_parity] [vanished] 18 committed unit(s) not found — nothing was verified for them:\n'
            '[orchestrator_unit_parity]   orchestrator-autopilot-video.service: expected committed copy at <TMP>/repo/scripts/orchestrator-autopilot-video.service\n'
            '[orchestrator_unit_parity]   orchestrator-autopilot-video.socket: expected committed copy at <TMP>/repo/scripts/orchestrator-autopilot-video.socket.template\n'
            '[orchestrator_unit_parity]   orchestrator-dark-factory.service: expected committed copy at <TMP>/repo/scripts/orchestrator-dark-factory.service\n'
            '[orchestrator_unit_parity]   orchestrator-dark-factory.socket: expected committed copy at <TMP>/repo/scripts/orchestrator-dark-factory.socket.template\n'
            '[orchestrator_unit_parity]   orchestrator-know-live.service: expected committed copy at <TMP>/repo/scripts/orchestrator-know-live.service\n'
            '[orchestrator_unit_parity]   orchestrator-know-live.socket: expected committed copy at <TMP>/repo/scripts/orchestrator-know-live.socket.template\n'
            '[orchestrator_unit_parity]   orchestrator-my-solar-challenge.service: expected committed copy at <TMP>/repo/scripts/orchestrator-my-solar-challenge.service\n'
            '[orchestrator_unit_parity]   orchestrator-my-solar-challenge.socket: expected committed copy at <TMP>/repo/scripts/orchestrator-my-solar-challenge.socket.template\n'
            '[orchestrator_unit_parity]   orchestrator-pump-web-ui.service: expected committed copy at <TMP>/repo/scripts/orchestrator-pump-web-ui.service\n'
            '[orchestrator_unit_parity]   orchestrator-pump-web-ui.socket: expected committed copy at <TMP>/repo/scripts/orchestrator-pump-web-ui.socket.template\n'
            '[orchestrator_unit_parity]   orchestrator-reify.service: expected committed copy at <TMP>/repo/scripts/orchestrator-reify.service\n'
            '[orchestrator_unit_parity]   orchestrator-reify.socket: expected committed copy at <TMP>/repo/scripts/orchestrator-reify.socket.template\n'
            '[orchestrator_unit_parity]   orchestrator-solar-challenge-platform.service: expected committed copy at <TMP>/repo/scripts/orchestrator-solar-challenge-platform.service\n'
            '[orchestrator_unit_parity]   orchestrator-solar-challenge-platform.socket: expected committed copy at <TMP>/repo/scripts/orchestrator-solar-challenge-platform.socket.template\n'
            '[orchestrator_unit_parity]   orchestrator-stream-touch.service: expected committed copy at <TMP>/repo/scripts/orchestrator-stream-touch.service\n'
            '[orchestrator_unit_parity]   orchestrator-stream-touch.socket: expected committed copy at <TMP>/repo/scripts/orchestrator-stream-touch.socket.template\n'
            '[orchestrator_unit_parity]   orchestrator-watchdog.service: expected committed copy at <TMP>/repo/scripts/orchestrator-watchdog.service\n'
            '[orchestrator_unit_parity]   orchestrator-watchdog.timer: expected committed copy at <TMP>/repo/scripts/orchestrator-watchdog.timer\n'
            "[orchestrator_unit_parity] [vanished] The committed unit is this checker's source of truth. Check --repo-root, and whether the unit was renamed or moved (the paths live in UNITS in this script).\n"
        ),
        "",
    ),
}


def test_every_scenario_has_exactly_one_golden():
    assert sorted(SCENARIOS) == sorted(GOLDENS)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_cli_output_is_the_frozen_golden(scenario: str, tmp_path: pathlib.Path, capsys):
    assert _run(SCENARIOS[scenario](tmp_path), tmp_path, capsys) == GOLDENS[scenario]


# Spelled out rather than read from each checker's LOG_TAG, because the
# consumers grep for these literals: renaming a LOG_TAG must turn this red.
CONSUMER_TAGS: dict[str, str] = {
    "check_dashboard_unit_parity": "dashboard_unit_parity",
    "check_orchestrator_unit_parity": "orchestrator_unit_parity",
    "check_lms_unit_parity": "lms_unit_parity",
    "check_fused_memory_unit_parity": "fused_memory_unit_parity",
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_every_output_line_carries_the_tag_consumers_grep_for(
    scenario: str, tmp_path: pathlib.Path, capsys
):
    invocation = SCENARIOS[scenario](tmp_path)
    _, out, err = _run(invocation, tmp_path, capsys)
    tag = f"[{CONSUMER_TAGS[invocation.checker]}] "
    lines = (out + err).splitlines()

    assert lines, "a run that prints nothing gives a consumer no tag to find"
    assert [line for line in lines if not line.startswith(tag)] == []


# ---------------------------------------------------------------------------
# One ladder for the family
# ---------------------------------------------------------------------------

LADDER: dict[str, tuple[int, tuple[str, ...]]] = {
    "parity": (
        0,
        ("dashboard/parity", "orchestrator/parity", "lms/parity", "fused-memory/parity"),
    ),
    "absent-only": (
        2,
        ("dashboard/missing", "orchestrator/missing", "lms/missing", "fused-memory/missing"),
    ),
    "drift": (
        1,
        ("dashboard/drift", "orchestrator/drift", "lms/drift", "fused-memory/drift"),
    ),
    "override": (
        1,
        ("dashboard/override", "orchestrator/override", "lms/override", "fused-memory/override"),
    ),
    "vanished-or-zero-compared": (
        1,
        (
            "dashboard/vanished",
            "dashboard/zero-compared",
            "orchestrator/vanished",
            "orchestrator/zero-compared",
            "lms/vanished",
        ),
    ),
    "drift+absent": (1, ("dashboard/drift+absent", "orchestrator/drift+absent")),
}


@pytest.mark.parametrize("scenario_class", sorted(LADDER))
def test_every_checker_maps_a_scenario_class_to_one_exit_code(
    scenario_class: str, tmp_path: pathlib.Path, capsys
):
    expected, scenarios = LADDER[scenario_class]
    observed = {}
    for index, scenario in enumerate(scenarios):
        root = tmp_path / str(index)
        observed[scenario] = _run(SCENARIOS[scenario](root), root, capsys)[0]
    assert observed == dict.fromkeys(scenarios, expected)


# ---------------------------------------------------------------------------
# An unreadable unit is a report, never a traceback
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UnreadableCase:
    """How one driver-run checker meets a unit file it cannot decode."""

    build: Callable[[pathlib.Path, str | bytes, str | bytes], Invocation]
    tag: str
    unit: str
    committed_relpath: str
    text: str


UNREADABLE_CASES: dict[str, UnreadableCase] = {
    "orchestrator": UnreadableCase(
        lambda tmp, repo, installed: _orchestrator(
            tmp, {ORCH_TIMER: repo}, {ORCH_TIMER: installed}, (ORCH_TIMER,)
        ),
        "orchestrator_unit_parity",
        ORCH_TIMER,
        f"scripts/{ORCH_TIMER}",
        ORCH_TIMER_TEXT,
    ),
    "dashboard": UnreadableCase(
        lambda tmp, repo, installed: _dashboard(
            tmp, {DASH_TIMER: repo}, {DASH_TIMER: installed}, (DASH_TIMER,)
        ),
        "dashboard_unit_parity",
        DASH_TIMER,
        f"dashboard/{DASH_TIMER}",
        DASH_TIMER_TEXT,
    ),
    "lms": UnreadableCase(
        lambda tmp, repo, installed: _lms(tmp, repo=repo, installed={LMS_UNIT: installed}),
        "lms_unit_parity",
        LMS_UNIT,
        LMS_RELPATH,
        LMS_TEMPLATE,
    ),
}


def _recording_clean_probe(calls: list[list[str]]) -> ProbeRunner:
    def runner(argv: list[str]) -> tuple[int, str]:
        calls.append(argv)
        return 0, CLEAN_PROBE_STDOUT

    return runner


@pytest.mark.parametrize("side", ["installed", "committed"])
@pytest.mark.parametrize("checker", sorted(UNREADABLE_CASES))
def test_an_unreadable_unit_is_reported_naming_the_file_that_failed(
    checker: str, side: str, tmp_path: pathlib.Path, capsys
):
    case = UNREADABLE_CASES[checker]
    if side == "installed":
        invocation = case.build(tmp_path, case.text, UNDECODABLE)
        failed = f"<TMP>/installed/{case.unit}"
    else:
        invocation = case.build(tmp_path, UNDECODABLE, case.text)
        failed = f"<TMP>/repo/{case.committed_relpath}"
    probe_calls: list[list[str]] = []
    if invocation.probe_runner is not None:
        invocation = dataclasses.replace(
            invocation, probe_runner=_recording_clean_probe(probe_calls)
        )

    rc, out, _ = _run(invocation, tmp_path, capsys)

    assert rc == 1
    assert (
        f"[{case.tag}] [unreadable] 1 unit file(s) could not be read — "
        "nothing was verified for them:\n"
    ) in out
    assert f"[{case.tag}]   {case.unit}: {failed}: " in out
    assert "[ok]" not in out
    assert probe_calls == [], "a run that could not read the unit must not probe it"
