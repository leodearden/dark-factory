"""Starting a unit must not mutate the shared root .venv.

SCOPE, stated precisely because the honest boundary is narrower than the
invariant one would want.  This module sweeps every committed unit whose
ExecStart is DIRECTLY a ``uv run`` command.  It does NOT follow a unit whose
ExecStart is a shell wrapper running ``uv run`` internally, and two committed
timer-driven units are exactly that shape — scripts/fused-memory-flag-marker-
sweep.service and scripts/memory-metadata-coverage-census.service, whose
wrappers still carry the old ``--frozen`` and so still sync into the shared venv
on every timer elapse.  Those wrappers lie outside task 5553's locks; extending
discovery through them is filed as a follow-up.  Recording the hole here is the
point: a guard reporting green while a known instance of its own defect runs
nightly is worse than one that says what it does not cover.

WHY THE INVARIANT.  Every member of the root pyproject.toml's
``[tool.uv.workspace]`` resolves to ONE ``.venv`` at the repo root, so a unit's
ExecStart acts not on an environment of its own but on the one the seven
orchestrators, the dashboard, the load sampler, fused-memory and every verify
subprocess are already running out of.  ``uv run`` is an INEXACT sync: at
process start it INSTALLS the named member's missing closure into that shared
venv.  ``--frozen`` does not stop it (a LOCKFILE option); ``--no-sync`` does.
The measurement behind those two sentences — transcript, exit codes, lockfile
digest — is written out ONCE, in scripts/orchestrator-autopilot-video.service
above its ExecStart, and cited from here rather than restated.

From 2026-05-29 until task 5553 the fleet carried ``--frozen`` believing it was
the venv guard, and nothing mechanical contradicted the belief, because the
per-unit assertions of the day pinned the flag's TEXT rather than its effect.
This sweep asserts the effect's precondition instead, by content discovery, so a
unit added next month is covered the day it lands.  Structurally modelled on
tests/scripts/test_systemd_restart_backoff.py.
"""
import pathlib
import re
import subprocess

import pytest
from systemd_unit_invariants import MalformedExecStart

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The ExecStart= anchor, shared by the grep and by logical_exec_start so the
# discoverer and the parser answer the SAME question.  They did not at first:
# the grep tolerated systemd's legal `  ExecStart = /usr/bin/uv` while the
# parser matched a bare `ExecStart=` prefix, so such a unit was discovered and
# then reported as declaring no ExecStart at all.  The trailing `=` is what
# excludes `ExecStartPre=`, which offers `P` where the pattern needs `=`.
_EXEC_START_PREFIX = r"ExecStart[ \t]*="
_EXEC_START_RE = re.compile(rf"^{_EXEC_START_PREFIX}")

# Copied from tests/scripts/test_systemd_restart_backoff.py::
# _NON_UNIT_PATHSPECS, which holds the reasoning and the measurements: test
# files embed whole units as fixtures, and docs may legitimately quote the
# DEFECT.  The glob forms are load-bearing — a plain `:!tests/` excludes only
# the top-level directory.
_NON_UNIT_PATHSPECS = (
    ":(exclude,glob)**/tests/**",
    ":(exclude,glob)**/*.md",
)

# Every committed unit or template whose ExecStart is a `uv run` against a
# workspace member.  Asserted by EQUALITY below, unlike the one-sided coverage
# guard in the restart-backoff sweep, because discovery here ends in a PARSER
# and a parser can silently answer None for a unit it stopped understanding.
_EXPECTED_UV_RUN_UNITS = frozenset(
    {
        "dashboard/dark-factory-dashboard.service",
        "dashboard/dark-factory-load-sampler.service",
        "fused-memory/fused-memory.service.example-systemd-config",
        "scripts/dashboard.service.template",
        "scripts/fused-memory.service.template",
        "scripts/legibility-transcript-check@.service",
        "scripts/legibility-trickle-health@.service",
        "scripts/legibility-trickle@.service",
        "scripts/local-model-serving/lms-arm@.service",
        "scripts/orchestrator-autopilot-video.service",
        "scripts/orchestrator-dark-factory.service",
        "scripts/orchestrator-know-live.service",
        "scripts/orchestrator-my-solar-challenge.service",
        "scripts/orchestrator-pump-web-ui.service",
        "scripts/orchestrator-reify.service",
        "scripts/orchestrator-solar-challenge-platform.service",
    }
)

# uv's render sentinel in the two `.service.template` files, substituted for an
# absolute uv path by scripts/setup-host.sh.  Recognised so a template is swept
# in its COMMITTED form rather than only after rendering.
_UV_PATH_SENTINEL = "__UV_PATH__"

# The one markdown file opted back in, mirroring
# tests/scripts/test_systemd_restart_backoff.py::_FACTORY_INIT_REFERENCE.  It is
# not prose ABOUT a unit but the unit new projects are minted from, declaring in
# its Layer 1 section that the scripts/orchestrator-*.service files are "cp'd
# verbatim by setup-host.sh".  A copy-source still showing the old flags mints
# this defect into every new project, with nothing in the sweep able to see it.
_FACTORY_INIT_REFERENCE = "skills/factory-init/references/supervised-unit.md"

# How the flag walk classifies each run-level token.  PARTIAL BY DESIGN and
# backed by a raise, not by a guess: `uv run --help` lists ~80 options, and
# embedding all of them here would be a copy of uv's interface that drifts
# silently at the next upgrade.  These cover what a systemd unit plausibly
# carries; anything else raises, so an unclassified option is a one-line
# decision rather than the mis-walk the raise message describes.
_VALUE_TAKING_RUN_FLAGS = frozenset(
    {
        "--project",
        "--package",
        "--python",
        "--directory",
        "--with",
        "--env-file",
        "--extra",
        "--group",
        "--color",
    }
)
_BOOLEAN_RUN_FLAGS = frozenset(
    {
        "--no-sync",
        "--frozen",
        "--locked",
        "--active",
        "--isolated",
        "--offline",
        "--exact",
        "--all-packages",
        "--all-extras",
        "--no-dev",
        "--no-project",
        "--no-config",
        "--no-env-file",
        "--no-editable",
        "--quiet",
        "--verbose",
    }
)


def discover_exec_start_files() -> list[str]:
    """Return every git-tracked file declaring ``ExecStart=``, sans exclusions.

    By CONTENT rather than by filename glob or a hand-maintained list, for the
    reason test_systemd_restart_backoff.py::
    discover_units_declaring_a_restart_cap states in full: the affected files
    span four naming conventions across five directories, so any glob broad
    enough to catch them all is broader than the invariant, and a hand list
    fails the stated goal outright.

    The anchor is ``ExecStart=`` rather than ``uv run``, i.e. deliberately WIDER
    than the obligation.  Narrowing it to `uv` would let a unit whose ExecStart
    stopped being recognisable as a uv invocation vanish from discovery
    silently; anchoring on the directive every unit must have keeps it in the
    discovered set, where the coverage guard can notice the answer changed.
    Filtering to actual `uv run` commands is the SECOND stage, below.

    The returncode assertion is load-bearing: `git grep` exits >1 on a real
    error, and a helper that swallowed that would return [], collect ZERO
    parametrized cases, and report the sweep green while checking nothing.
    """
    proc = subprocess.run(
        [
            "git",
            "grep",
            "-lE",
            rf"^[ \t]*{_EXEC_START_PREFIX}",
            "--",
            ".",
            *_NON_UNIT_PATHSPECS,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), (
        f"`git grep` for ExecStart= exited {proc.returncode} in {REPO_ROOT} "
        "(0=matches, 1=no matches, >1=error), so unit discovery produced "
        "nothing to check. Failing loudly here is deliberate: returning no "
        "paths would collect zero parametrized cases and report this whole "
        f"sweep green while checking nothing. stderr: {proc.stderr.strip()!r}"
    )
    return sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())


def logical_exec_start(text: str, unit_name: str = "<unit>") -> str:
    """Return the effective ExecStart COMMAND in *text* as one logical line.

    Two normalisations, each of which a naive read gets wrong on a file
    committed in this repo today.

    LAST OCCURRENCE WINS, mirroring systemd and
    test_orchestrator_service_files.py::_exec_start_line, whose docstring holds
    the reasoning: a drop-in override lands as an empty ``ExecStart=`` RESET
    followed by the real command, so a first-match read finds the reset, sees no
    flags, and passes a unit whose real command may well be wrong.

    CONTINUATIONS ARE JOINED, following
    test_dashboard_service_template.py::_logical_exec_start.  The ExecStart= of
    scripts/dashboard.service.template and scripts/fused-memory.service.template
    (with its committed mirror) spans several physical lines.  For those three
    the first fragment happens to hold the run-level flags today, which is worse
    than useless: an unjoined read would pass them VACUOUSLY and stop noticing
    the day a flag moved to a continuation line.

    Returns the command only, with the directive prefix removed.  Raises
    MalformedExecStart — the shared class, so a broken unit surfaces as ONE
    class whichever layer notices it first — when there is no ExecStart= at all,
    or when the effective one carries no command.  Neither is a legitimate "this
    unit has no uv flags" answer; that is the None return below.
    """
    lines = text.splitlines()
    start_indices = [
        i for i, ln in enumerate(lines) if _EXEC_START_RE.match(ln.strip())
    ]
    if not start_indices:
        raise MalformedExecStart(
            f"{unit_name} declares no ExecStart= line, so there is no command "
            "to check for run-level uv flags. Treating this as 'not a uv run "
            "command' would silently drop the unit out of the sweep — the exact "
            "direction a guard against a silently-mutated venv must refuse."
        )

    parts: list[str] = []
    idx = start_indices[-1]
    while True:
        line = lines[idx].strip()
        continued = line.endswith("\\")
        if continued:
            line = line[:-1]
        parts.append(line.strip())
        if not continued or idx + 1 >= len(lines):
            break
        idx += 1

    command = _EXEC_START_RE.sub("", " ".join(p for p in parts if p), count=1).strip()
    if not command:
        raise MalformedExecStart(
            f"{unit_name}'s effective ExecStart= carries no command: the last "
            "assignment is a list RESET with nothing appended after it, so "
            "systemd has no command to run at all. Treating this as 'not a uv "
            "run command' would silently drop a unit that cannot start."
        )
    return command


def uv_run_level_flags(exec_start: str) -> list[str] | None:
    """Return *exec_start*'s RUN-LEVEL uv flag names, or None if it is not `uv run`.

    None means exactly one thing: this command is not a ``uv run`` invocation,
    so the invariant does not apply and callers SKIP.  That is a real answer —
    scripts/orchestrator-watchdog.service runs a bare Python probe and
    scripts/jcodemunch-watcher.service.template runs ``uvx``, an EPHEMERAL
    environment that never touches the shared venv.  The None-vs-raise split
    follows the contract stated once at
    systemd_unit_invariants.py::config_arg_from_exec_start.  Note what None does
    NOT cover: a unit whose ExecStart is a shell wrapper running ``uv run``
    inside it also answers None, which is the scope hole this module's docstring
    records rather than hides.

    RUN-LEVEL is the load-bearing word, and the reason this walks a prefix
    instead of substring-checking the line.  In

        uv run --project orchestrator orchestrator run --config <path>

    ``orchestrator`` is the COMMAND token and everything after it belongs to the
    orchestrator CLI, not to uv.  A flag appended there would leave the unit
    textually satisfying a naive presence check while uv never saw it and the
    venv was still mutated at start — a guard blessing the exact defect it was
    written for.  So the walk starts after ``run`` and STOPS at the first token
    that is neither a ``--flag`` nor the value of one.
    orchestrator/tests/test_mcp_lifecycle.py::TestMcpServerArgs::
    test_run_level_flags_before_python guards the same positional property for
    the plan-tools fast-start argv, where the repo already spelled the flag
    ``--no-sync`` — so the hot path had it right all along while the units did
    not.

    Flag NAMES are returned, i.e. ``--project=orchestrator`` reports
    ``--project``, so the two spellings of a valued flag cannot give two
    different answers about whether a flag is present.
    """
    tokens = exec_start.split()
    if len(tokens) < 2:
        return None
    executable, second = tokens[0], tokens[1]
    is_uv = executable.rsplit("/", 1)[-1] == "uv" or executable == _UV_PATH_SENTINEL
    if not is_uv or second != "run":
        return None

    flags: list[str] = []
    i = 2
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            break
        name = token.partition("=")[0]
        if name not in _VALUE_TAKING_RUN_FLAGS and name not in _BOOLEAN_RUN_FLAGS:
            raise MalformedExecStart(
                f"this walker does not know the run-level flag {name!r} in "
                f"{exec_start!r}, so it cannot tell where uv's flags end and "
                "the command begins. Guessing is what makes this raise rather "
                "than continue. Assumed boolean, the flag's VALUE is mistaken "
                "for the command token and the walk stops early, reporting a "
                "unit as missing `--no-sync` when it carries it. Assumed "
                "valued, a real command token is swallowed and the walk runs "
                "on into the command's own arguments, where a `--no-sync` "
                "belonging to something else would satisfy the guard. Classify "
                "it against `uv run --help` into _VALUE_TAKING_RUN_FLAGS or "
                "_BOOLEAN_RUN_FLAGS in this module."
            )
        flags.append(name)
        # A `--flag=value` token carries its own value; only the space-separated
        # spelling consumes the token after it.
        if name in _VALUE_TAKING_RUN_FLAGS and "=" not in token:
            i += 1
        i += 1
    return flags


def discover_uv_run_units() -> list[str]:
    """Repo-relative paths of every discovered file whose ExecStart is a `uv run`.

    A MalformedExecStart from either parser INCLUDES the path rather than
    propagating, for two reasons, the second sharper.  A raise here happens
    inside the ``@pytest.mark.parametrize`` argument, i.e. at COLLECTION time,
    so it takes down the whole module — including the coverage guard that exists
    to report which unit stopped being understood.  And "cannot answer" must
    never resolve to "silently skipped": the path stays in the sweep, where the
    per-case body re-parses it and fails with the parser's own message, naming
    the one unit at fault.
    """
    units = []
    for rel_path in discover_exec_start_files():
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        try:
            if uv_run_level_flags(logical_exec_start(text, rel_path)) is None:
                continue
        except MalformedExecStart:
            pass
        units.append(rel_path)
    return units


def swept_paths() -> list[str]:
    """The discovered units plus the markdown opt-in, which is never a skip."""
    return sorted([*discover_uv_run_units(), _FACTORY_INIT_REFERENCE])


def _run_level_flags_of(rel_path: str) -> list[str]:
    """The run-level uv flags of *rel_path*, failing rather than returning None.

    swept_paths only yields paths that parsed as a `uv run` command, so a None
    here means the file changed shape between collection and the test body.
    """
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    flags = uv_run_level_flags(logical_exec_start(text, rel_path))
    assert flags is not None, (
        f"{rel_path} was swept as a `uv run` unit but no longer parses as one. "
        "It was discovered by its ExecStart= and must not leave the sweep "
        "silently — check whether the command shape changed."
    )
    return flags


def test_factory_init_reference_is_swept() -> None:
    """The opt-in copy-source must exist and still parse as a `uv run` command.

    Needs its own statement because this path does not come from discovery: a
    rename or a reformat of its fenced ini block would otherwise drop the ONLY
    copy-source out of both arms silently, turning sixteen cases into fifteen
    with nothing red.
    """
    path = REPO_ROOT / _FACTORY_INIT_REFERENCE
    assert path.exists(), (
        f"{_FACTORY_INIT_REFERENCE} does not exist. New projects' supervised "
        "units are copied from it, so if it moved this guard must follow it "
        "rather than silently stop checking anything."
    )
    flags = uv_run_level_flags(
        logical_exec_start(path.read_text(encoding="utf-8"), _FACTORY_INIT_REFERENCE)
    )
    assert flags is not None, (
        f"{_FACTORY_INIT_REFERENCE}'s ExecStart no longer parses as a `uv run` "
        "command, so both arms below would pass it vacuously. Either the fenced "
        "ini block was reformatted past logical_exec_start, or the command "
        "changed shape — check it deliberately rather than letting the repo's "
        "only unit copy-source leave the sweep."
    )


def test_discovery_covers_every_known_uv_run_unit() -> None:
    """Coverage guard: the swept set must be non-empty and exactly the known units.

    The counterpart to test_orchestrator_service_files.py::
    test_orchestrator_service_glob_covers_all_known_units.  Without it a broken
    discovery shrinks the sweep to zero cases, and a zero-case parametrize
    collects no tests and reports no failure — masking the very defect the sweep
    exists to catch.

    EQUALITY here, where test_systemd_restart_backoff.py::
    test_discovery_covers_every_known_unit checks missing-only, and the
    difference is deliberate.  There, discovery is a single git grep whose whole
    point is that a NEW unit declaring a cap is swept automatically, so an
    equality assertion would turn that success into a failure.  Here the second
    stage is a parser that answers None for anything it no longer recognises, so
    a unit reformatted past the walker leaves the sweep quietly rather than
    failing in it.  A new unit is expected to add its path, and that one-line
    diff is the price of its coverage being a decision rather than an accident.
    """
    discovered = set(discover_uv_run_units())
    assert discovered, (
        "discovery found no `uv run` units at all. Most likely causes: the git "
        f"grep ran outside a checkout (REPO_ROOT={REPO_ROOT}), or "
        "uv_run_level_flags stopped recognising every ExecStart. Either way the "
        "parametrized sweeps below would collect zero cases and report green "
        "while checking nothing."
    )
    missing = _EXPECTED_UV_RUN_UNITS - discovered
    extra = discovered - _EXPECTED_UV_RUN_UNITS
    assert not missing, (
        f"these known `uv run` units dropped out of the sweep: {sorted(missing)}. "
        "They were not deleted (discovery reads git-tracked files), so either "
        "the ExecStart= moved/was reformatted past logical_exec_start, or "
        "uv_run_level_flags stopped recognising it as a uv invocation. A unit "
        "that leaves this sweep silently is the failure mode the sweep exists "
        "to prevent."
    )
    assert not extra, (
        f"the sweep picked up files not in _EXPECTED_UV_RUN_UNITS: {sorted(extra)}. "
        "If these are genuinely new units running `uv run` against a workspace "
        "member, add them to the constant — they share the one root .venv and "
        "so carry the same obligation. A file whose ExecStart= is MALFORMED also "
        "lands here, deliberately (see discover_uv_run_units); its own case "
        "below will name the defect. If they are not units at all, exclude them "
        "via _NON_UNIT_PATHSPECS rather than checking them."
    )


@pytest.mark.parametrize("rel_path", swept_paths(), ids=lambda p: p)
def test_uv_run_unit_passes_no_sync(rel_path: str) -> None:
    """A unit's `uv run` must carry `--no-sync` among its RUN-LEVEL flags."""
    flags = _run_level_flags_of(rel_path)
    assert "--no-sync" in flags, (
        f"{rel_path} runs `uv run` without a run-level `--no-sync` (run-level "
        f"flags found: {flags}). Without it, process start INSTALLS the named "
        "member's missing dependency closure into the ONE shared root .venv "
        "every workspace member, every running orchestrator and every verify "
        "subprocess resolves against. `--frozen` does NOT prevent this — it is "
        "a LOCKFILE option, measured reinstalling into the venv (the full "
        "measurement is in scripts/orchestrator-autopilot-video.service, above "
        "its ExecStart). The flag must sit BEFORE the command token, or uv "
        "never sees it. The accepted cost is that an unsynced venv now fails "
        "the unit with ModuleNotFoundError instead of repairing itself; the "
        "repair path is scripts/sync-orchestrator-env.sh (`uv sync "
        "--all-packages`)."
    )


@pytest.mark.parametrize("rel_path", swept_paths(), ids=lambda p: p)
def test_uv_run_unit_carries_no_lockfile_flag(rel_path: str) -> None:
    """Beside `--no-sync`, a run-level `--frozen`/`--locked` is a no-op — so forbid it.

    Its own test rather than a second assert in the arm above: the two fail for
    different reasons, a reader of a failure should see which one fired, and the
    presence arm must stay green independently of this one.
    """
    flags = _run_level_flags_of(rel_path)
    lockfile_flags = [f for f in flags if f in ("--frozen", "--locked")]
    assert not lockfile_flags, (
        f"{rel_path} carries run-level {lockfile_flags} beside `--no-sync` "
        f"(run-level flags found: {flags}). `--no-sync` skips the LOCK step "
        "along with the sync, so both lockfile flags buy literally nothing next "
        "to it — measured, with the exit codes and the unchanged lockfile "
        "digest, in scripts/orchestrator-autopilot-video.service above its "
        "ExecStart. A flag that survives review by looking like it strengthens "
        "the line while doing nothing is worse than no flag: `--frozen` is "
        "exactly how this defect was introduced, because it READS as a venv "
        "guarantee and is in fact a lockfile option. If loud failure on lockfile "
        "drift is ever genuinely wanted for a unit, it cannot be bought here — "
        "it would mean dropping `--no-sync`, which reinstates the hazard. Check "
        "the lockfile somewhere that is not a unit start."
    )
