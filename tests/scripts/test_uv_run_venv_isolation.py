"""Neither starting a unit nor running a checked-in shell script may mutate the shared root .venv.

SCOPE, stated precisely because the honest boundary is narrower than the
invariant one would want.  This module sweeps every committed unit whose start
runs ``uv run``: in its ExecStart, or one level down, in the checkout's shell
wrapper its ExecStart executes (read as text by shell_uv_run_sites, never run).
A second arm sweeps every non-test ``*.sh`` the checkout lists, by content,
whether or not a unit runs it.
It does NOT follow a wrapper's own callees, a wrapper not named ``*.sh``, a uv
the wrapper reaches through a variable whose name never mentions uv, or a
``*_CMD`` seam a unit overrides through ``Environment=`` (no committed unit
does today).  Inside that boundary "cannot answer" raises rather than skips: a
``.sh`` the checkout does not hold, and a uv-then-``run`` the scan cannot read.
Python code that BUILDS a ``uv run`` argv or command string is invisible to a
text scan of shell and is NOT swept.  Three such sites are pinned at their own
seams:
tests/scripts/test_dashboard_watchdog.py::test_b4_ceiling_escalation_runs_uv_without_syncing_the_shared_venv,
tests/scripts/test_watchdog_crash_loop.py::test_submit_argv_is_the_born_at_l2_shape
and scripts/tests/test_sitting_nightly_prepare.py::test_the_prepare_command_never_syncs_the_shared_root_venv.
Any other Python-built ``uv run`` is checked by nothing, whatever it passes.
Command examples in markdown are not swept, apart from the one unit copy-source
FACTORY_INIT_REFERENCE names.
Recording the boundary here is the point: a guard reporting green over what it
never read is worse than one that says what it does not cover.

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
import dataclasses
import pathlib
import re
import subprocess
from collections.abc import Callable

import pytest
from git_listing import git, listed_files
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from shell_logical_lines import logical_lines
from systemd_unit_invariants import (
    EXEC_START_PREFIX,
    FACTORY_INIT_REFERENCE,
    NON_UNIT_PATHSPECS,
    MalformedExecStart,
    logical_exec_start,
)

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]

# Every committed unit or template whose start runs `uv run` against a workspace
# member, in its ExecStart or in the checkout's shell wrapper its ExecStart names.
# Asserted by EQUALITY below, unlike the one-sided coverage guard in the
# restart-backoff sweep, because discovery here ends in a PARSER and a parser
# can silently answer None for a unit it stopped understanding.
_EXPECTED_UV_RUN_UNITS = frozenset(
    {
        "dashboard/dark-factory-dashboard.service",
        "dashboard/dark-factory-load-sampler.service",
        "fused-memory/fused-memory.service.example-systemd-config",
        "scripts/client-feed.service.template",
        "scripts/dashboard.service.template",
        "scripts/exception-owner-sweep.service",
        "scripts/fused-memory-duplicate-audit.service",
        "scripts/fused-memory-flag-marker-sweep.service",
        "scripts/fused-memory-link-heal.service",
        "scripts/fused-memory-memory-eval.service",
        "scripts/fused-memory.service.template",
        "scripts/legibility-transcript-check@.service",
        "scripts/legibility-trickle-health@.service",
        "scripts/legibility-trickle@.service",
        "scripts/local-model-serving/lms-arm@.service",
        "scripts/lock-charter-drift-check.service",
        "scripts/memory-metadata-coverage-census.service",
        "scripts/orchestrator-autopilot-video.service",
        "scripts/orchestrator-dark-factory.service",
        "scripts/orchestrator-know-live.service",
        "scripts/orchestrator-my-solar-challenge.service",
        "scripts/orchestrator-pump-web-ui.service",
        "scripts/orchestrator-reify.service",
        "scripts/orchestrator-solar-challenge-platform.service",
        "scripts/orchestrator-stream-touch.service",
        "scripts/return-brief.service",
        "scripts/timeout-headroom-selfcheck.service",
        "scripts/transcript-secret-sweep.laptop.conf",
        "scripts/transcript-secret-sweep.service",
    }
)

# Every non-test shell script in the checkout that runs `uv run`, whether or not
# a unit runs it; asserted by EQUALITY for the reason _EXPECTED_UV_RUN_UNITS states.
_EXPECTED_UV_RUN_SCRIPTS = frozenset(
    {
        "fused-memory/scripts/cgl_eta_auto_apply.sh",
        "scripts/check_dashboard_payload_budgets.sh",
        "scripts/check_sandbox_soak.sh",
        "scripts/check_write_triage_flip_preconditions.sh",
        "scripts/deploy/install_laptop_transcript_secret_sweep.sh",
        "scripts/eval_bootstrap_smoke.sh",
        "scripts/exception-owner-sweep.sh",
        "scripts/fused-memory-duplicate-audit.sh",
        "scripts/fused-memory-flag-marker-check.sh",
        "scripts/fused-memory-flag-marker-sweep.sh",
        "scripts/fused-memory-link-heal.sh",
        "scripts/fused-memory-memory-eval.sh",
        "scripts/install-exception-owner-sweep-timer.sh",
        "scripts/install-lock-charter-drift-check-timer.sh",
        "scripts/install-timeout-headroom-selfcheck-timer.sh",
        "scripts/legibility/install-transcript-check-timer.sh",
        "scripts/legibility/install-trickle-health-timer.sh",
        "scripts/legibility/install-trickle-timer.sh",
        "scripts/local-model-serving/install-lms-units.sh",
        "scripts/lock-charter-drift-check.sh",
        "scripts/memory-metadata-coverage-census.sh",
        "scripts/rerun_cap_tainted.sh",
        "scripts/return-brief.sh",
        "scripts/timeout-headroom-selfcheck.sh",
        "scripts/transcript-secret-sweep.sh",
    }
)

# uv's render sentinel in the `.service.template` files, substituted for an
# absolute uv path by scripts/setup-host.sh.  Recognised so a template is swept
# in its COMMITTED form rather than only after rendering.
_UV_PATH_SENTINEL = "__UV_PATH__"

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

# The uv executable as a shell script spells it, followed by `run`: a bare
# `uv` (also a `${X:-uv run ...}` seam default, or a path ending /uv), or a
# parameter expansion naming a variable containing UV, e.g. "$UV_RESOLVED".
_SHELL_UV_RUN_RE = re.compile(r'(?:\buv|\$\{?\w*UV\w*\}?)["\']?\s+run(?=\s|$)')

# WIDER than _SHELL_UV_RUN_RE on purpose: uv mentioned in any case and inside
# any longer word, then `run` with only `-`-led tokens between.  A `run` this
# reaches that _SHELL_UV_RUN_RE did not end at is a uv run the scan cannot read,
# e.g. "${UV:-uv}" run, "$(command -v uv)" run, "$uv_bin" run, uv --quiet run.
_SHELL_UV_MENTION_RUN_RE = re.compile(r'(?i:\buv|\$\{?\w*uv)\S*(?:\s+-\S+)*\s+run\b')

# The prefixes systemd allows before an ExecStart executable (systemd.service(5)).
_SYSTEMD_EXEC_PREFIXES = "-@:+!"

# Git pathspecs naming every shell script outside a test tree; a plain `*.sh`
# matches at every depth.
_NON_TEST_SHELL_SCRIPTS = ("*.sh", ":(exclude,glob)**/tests/**")


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
            # The parser's own anchor, so discovery and parsing cannot disagree.
            rf"^[ \t]*{EXEC_START_PREFIX}",
            "--",
            ".",
            *NON_UNIT_PATHSPECS,
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


def uv_run_level_flags(exec_start: str) -> list[str] | None:
    """Return *exec_start*'s RUN-LEVEL uv flag names, or None if it is not `uv run`.

    None means exactly one thing: this command is not a ``uv run`` invocation,
    so the invariant does not apply and callers SKIP.  That is a real answer —
    scripts/orchestrator-watchdog.service runs a bare Python probe that never
    touches the shared venv.  The None-vs-raise split
    follows the contract stated once at
    systemd_unit_invariants.py::config_arg_from_exec_start.  A unit whose
    ExecStart is a shell wrapper running ``uv run`` inside it also answers None
    here; uv_run_sites is what follows such a wrapper.

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
    return run_level_flags(tokens[2:], exec_start)


def run_level_flags(args: list[str], command: str) -> list[str]:
    """The run-level flag names leading *args*, the tokens after `run` (see uv_run_level_flags)."""
    flags: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            break
        name = token.partition("=")[0]
        if name not in _VALUE_TAKING_RUN_FLAGS and name not in _BOOLEAN_RUN_FLAGS:
            raise MalformedExecStart(
                f"this walker does not know the run-level flag {name!r} in "
                f"{command!r}, so it cannot tell where uv's flags end and "
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


class UnreadableShellUvRun(ValueError):
    """A shell script holds a `uv run` this scan cannot read; its case FAILS, never skips."""


@dataclasses.dataclass(frozen=True)
class UvRunSite:
    """One `uv run` a unit's start or a shell script executes: where it is written, its text, its run-level flags."""

    source: str
    command: str
    flags: tuple[str, ...]


def shell_uv_run_sites(script: str, source: str) -> list[UvRunSite]:
    """Every `uv run` in shell *script*, read from the TEXT and never by running it.

    Running a script is not safe from a test: the census wrapper creates git
    worktrees and lands through the merge queue.  Every non-comment occurrence
    counts, one inside a string included, since the flag walk stops at the
    first token that is not a flag anyway.

    Raises UnreadableShellUvRun, naming *source*, for a uv-then-`run` that
    _SHELL_UV_RUN_RE does not recognise or whose run-level flags the walk does
    not know.  Answering [] there would class the unit or script "runs no uv
    run" and skip it, or drop one of a two-site script's sites while the other
    kept it green.
    """
    sites: list[UvRunSite] = []
    for line in logical_lines(script):
        if line.startswith("#"):
            continue
        read_ends = set()
        for match in _SHELL_UV_RUN_RE.finditer(line):
            read_ends.add(match.end())
            args = [t.rstrip(")}\"';") for t in line[match.end() :].split()]
            sites.append(UvRunSite(source, line, _shell_run_level_flags(args, line, source)))
        if any(m.end() not in read_ends for m in _SHELL_UV_MENTION_RUN_RE.finditer(line)):
            raise UnreadableShellUvRun(
                f"{source}: {line!r} mentions uv and then `run`, in a spelling "
                "this scan does not read as a `uv run`, so it cannot check that "
                "command's run-level flags. Spell it as the committed scripts "
                'do (`uv run ...`, `"$UV_RESOLVED" run ...`, or a '
                "`${X_CMD:-uv run ...}` seam default), or teach "
                "_SHELL_UV_RUN_RE the new spelling and pin it in "
                "test_shell_uv_run_sites_reads_every_committed_spelling."
            )
    return sites


def _shell_run_level_flags(args: list[str], line: str, source: str) -> tuple[str, ...]:
    """run_level_flags for a shell line, its raise naming the script file."""
    try:
        return tuple(run_level_flags(args, line))
    except MalformedExecStart as exc:
        raise UnreadableShellUvRun(f"{source}: {exc}") from exc


def wrapper_script_of(exec_start: str, root: pathlib.Path = REPO_ROOT) -> str | None:
    """The *root*-relative path of the shell script *exec_start* executes; None if it runs none.

    Matched by the LONGEST suffix of the absolute path that is a file under
    *root*, so a same-named file nearer the root never shadows it, and the
    deployed path answers the same in every checkout, worktrees included.
    """
    tokens = exec_start.split()
    if not tokens:
        return None
    executable = tokens[0].lstrip(_SYSTEMD_EXEC_PREFIXES)
    if not executable.endswith(".sh"):
        return None
    parts = pathlib.PurePosixPath(executable).parts
    for i in range(1, len(parts)):
        candidate = "/".join(parts[i:])
        if (root / candidate).is_file():
            return candidate
    raise MalformedExecStart(
        f"ExecStart= runs the shell script {executable!r}, and no suffix of that "
        f"path is a file in this checkout ({root}). Answering 'runs no shell "
        "script' would class a renamed or mistyped wrapper's unit as running no "
        "`uv run` and skip it. Fix the path; a unit that genuinely runs a script "
        "from outside this repo cannot be vouched for by this sweep, so exempt "
        "it here as a deliberate decision rather than letting it pass unread."
    )


def uv_run_sites(rel_path: str, root: pathlib.Path = REPO_ROOT) -> list[UvRunSite]:
    """Every `uv run` starting the unit at *rel_path* under *root* runs, directly or via its wrapper.

    An empty list means the unit's start runs no `uv run` and callers SKIP it.
    MalformedExecStart (the unit) and UnreadableShellUvRun (its wrapper) propagate.
    """
    text = (root / rel_path).read_text(encoding="utf-8")
    exec_start = logical_exec_start(text, rel_path)
    flags = uv_run_level_flags(exec_start)
    if flags is not None:
        return [UvRunSite(rel_path, exec_start, tuple(flags))]
    wrapper = wrapper_script_of(exec_start, root)
    if wrapper is None:
        return []
    return script_uv_run_sites(wrapper, root)


def script_uv_run_sites(rel_path: str, root: pathlib.Path = REPO_ROOT) -> list[UvRunSite]:
    """Every `uv run` in the shell script at *rel_path* under *root*."""
    return shell_uv_run_sites((root / rel_path).read_text(encoding="utf-8"), rel_path)


def discover_uv_run_units() -> list[str]:
    """Repo-relative paths of every discovered file whose start runs `uv run`.

    A MalformedExecStart or UnreadableShellUvRun from any parser INCLUDES the
    path rather than propagating, for two reasons, the second sharper.  A raise here happens
    inside the ``@pytest.mark.parametrize`` argument, i.e. at COLLECTION time,
    so it takes down the whole module — including the coverage guard that exists
    to report which unit stopped being understood.  And "cannot answer" must
    never resolve to "silently skipped": the path stays in the sweep, where the
    per-case body re-parses it and fails with the parser's own message, naming
    the one unit at fault.
    """
    units = []
    for rel_path in discover_exec_start_files():
        try:
            if not uv_run_sites(rel_path):
                continue
        except (MalformedExecStart, UnreadableShellUvRun):
            pass
        units.append(rel_path)
    return units


def discover_uv_run_scripts(root: pathlib.Path = REPO_ROOT) -> list[str]:
    """*root*-relative paths of every non-test shell script git lists that runs `uv run`.

    An UnreadableShellUvRun INCLUDES the path rather than skipping it, for the
    reasons discover_uv_run_units gives.  listed_files raises on a git failure,
    so discovery never silently answers [].
    """
    scripts = []
    for path in listed_files(root, *_NON_TEST_SHELL_SCRIPTS):
        rel_path = path.relative_to(root).as_posix()
        try:
            if not script_uv_run_sites(rel_path, root):
                continue
        except UnreadableShellUvRun:
            pass
        scripts.append(rel_path)
    return sorted(scripts)


def swept_paths() -> list[str]:
    """The discovered units plus the markdown opt-in, which is never a skip."""
    return sorted([*discover_uv_run_units(), FACTORY_INIT_REFERENCE])


def swept_cases() -> list:
    """Every swept source paired with the reader that yields its `uv run` sites."""
    return [pytest.param(p, uv_run_sites, id=p) for p in swept_paths()] + [
        pytest.param(p, script_uv_run_sites, id=p) for p in discover_uv_run_scripts()
    ]


def _uv_run_sites_of(
    rel_path: str, read_sites: Callable[[str], list[UvRunSite]]
) -> list[UvRunSite]:
    """The `uv run` sites *read_sites* finds in *rel_path*, failing rather than returning none.

    swept_cases only yields paths that ran a `uv run`, so an empty list here
    means a file changed shape between collection and the test body.
    """
    sites = read_sites(rel_path)
    assert sites, (
        f"{rel_path} was swept as running `uv run` but no longer runs `uv run`. "
        "It was discovered by its ExecStart= or as a shell script and must not "
        "leave the sweep silently — check whether the command shape changed."
    )
    return sites


def _describe(site: UvRunSite) -> str:
    """One offending site, as a failure message names it."""
    return f"{site.source}: {site.command!r} (run-level flags: {list(site.flags)})"


def test_factory_init_reference_is_swept() -> None:
    """The opt-in copy-source must exist and still parse as a `uv run` command.

    Needs its own statement because this path does not come from discovery: a
    rename or a reformat of its fenced ini block would otherwise drop the ONLY
    copy-source out of both arms silently, one case fewer with nothing red.
    """
    path = REPO_ROOT / FACTORY_INIT_REFERENCE
    assert path.exists(), (
        f"{FACTORY_INIT_REFERENCE} does not exist. New projects' supervised "
        "units are copied from it, so if it moved this guard must follow it "
        "rather than silently stop checking anything."
    )
    flags = uv_run_level_flags(
        logical_exec_start(path.read_text(encoding="utf-8"), FACTORY_INIT_REFERENCE)
    )
    assert flags is not None, (
        f"{FACTORY_INIT_REFERENCE}'s ExecStart no longer parses as a `uv run` "
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
        "uv_run_sites stopped recognising every unit's start. Either way the "
        "parametrized sweeps below would collect zero cases and report green "
        "while checking nothing."
    )
    missing = _EXPECTED_UV_RUN_UNITS - discovered
    extra = discovered - _EXPECTED_UV_RUN_UNITS
    assert not missing, (
        f"these known `uv run` units dropped out of the sweep: {sorted(missing)}. "
        "They were not deleted (discovery reads git-tracked files), so either "
        "the ExecStart= moved/was reformatted past logical_exec_start, or "
        "uv_run_sites stopped recognising its `uv run`, in the ExecStart or "
        "in the wrapper it names. A unit "
        "that leaves this sweep silently is the failure mode the sweep exists "
        "to prevent."
    )
    assert not extra, (
        f"the sweep picked up files not in _EXPECTED_UV_RUN_UNITS: {sorted(extra)}. "
        "If these are genuinely new units running `uv run` against a workspace "
        "member, add them to the constant — they share the one root .venv and "
        "so carry the same obligation. A file whose ExecStart= is MALFORMED, or "
        "whose wrapper this scan cannot read, also lands here, deliberately "
        "(see discover_uv_run_units); its own case "
        "below will name the defect. If they are not units at all, exclude them "
        "via NON_UNIT_PATHSPECS rather than checking them."
    )


def test_script_discovery_covers_every_known_uv_run_script() -> None:
    """Coverage guard for the script arm: non-empty and exactly the known scripts.

    Equality for the reason test_discovery_covers_every_known_uv_run_unit
    gives: discovery ends in a parser, so a script re-spelled past it would
    otherwise leave the sweep quietly.
    """
    discovered = set(discover_uv_run_scripts())
    assert discovered, (
        "discovery found no shell script running `uv run` at all, so the "
        "parametrized sweeps below would check no script and report green. "
        f"Check that git lists the checkout's *.sh files (REPO_ROOT={REPO_ROOT})."
    )
    missing = _EXPECTED_UV_RUN_SCRIPTS - discovered
    extra = discovered - _EXPECTED_UV_RUN_SCRIPTS
    assert not missing, (
        f"these known `uv run` shell scripts dropped out of the sweep: {sorted(missing)}. "
        "Either the script stopped running `uv run` (then remove it from "
        "_EXPECTED_UV_RUN_SCRIPTS deliberately), or it moved, or shell_uv_run_sites "
        "stopped recognising its `uv run`. A script that leaves this sweep "
        "silently is the failure mode the sweep exists to prevent."
    )
    assert not extra, (
        f"the sweep picked up shell scripts not in _EXPECTED_UV_RUN_SCRIPTS: {sorted(extra)}. "
        "A shell script that newly runs `uv run` shares the one root .venv and "
        "carries the same obligation, so add it to the constant. A script whose "
        "`uv run` this scan cannot read also lands here, deliberately (see "
        "discover_uv_run_scripts); its own case below names the defect. A test "
        "fixture belongs under a tests/ directory, which discovery excludes."
    )


@pytest.mark.parametrize(("rel_path", "read_sites"), swept_cases())
def test_uv_run_site_passes_no_sync(
    rel_path: str, read_sites: Callable[[str], list[UvRunSite]]
) -> None:
    """Every `uv run` a unit's start or a shell script runs must carry `--no-sync` among its RUN-LEVEL flags."""
    sites = _uv_run_sites_of(rel_path, read_sites)
    missing = [s for s in sites if "--no-sync" not in s.flags]
    assert not missing, (
        f"{rel_path} runs `uv run` without a run-level `--no-sync` at: "
        f"{'; '.join(_describe(s) for s in missing)}. Without it, process "
        "start INSTALLS the named member's missing dependency closure into the "
        "ONE shared root .venv "
        "every workspace member, every running orchestrator and every verify "
        "subprocess resolves against. `--frozen` does NOT prevent this — it is "
        "a LOCKFILE option, measured reinstalling into the venv (the full "
        "measurement is in scripts/orchestrator-autopilot-video.service, above "
        "its ExecStart). The flag must sit BEFORE the command token, or uv "
        "never sees it. The accepted cost is that an unsynced venv now fails "
        "the command with ModuleNotFoundError instead of repairing itself; the "
        "repair path is scripts/sync-orchestrator-env.sh (`uv sync "
        "--all-packages`)."
    )


@pytest.mark.parametrize(("rel_path", "read_sites"), swept_cases())
def test_uv_run_site_carries_no_lockfile_flag(
    rel_path: str, read_sites: Callable[[str], list[UvRunSite]]
) -> None:
    """Beside `--no-sync`, a run-level `--frozen`/`--locked` is a no-op — so forbid it.

    Its own test rather than a second assert in the arm above: the two fail for
    different reasons, a reader of a failure should see which one fired, and the
    presence arm must stay green independently of this one.
    """
    sites = _uv_run_sites_of(rel_path, read_sites)
    offending = [s for s in sites if {"--frozen", "--locked"} & set(s.flags)]
    assert not offending, (
        f"{rel_path} runs `uv run` with a run-level `--frozen`/`--locked` at: "
        f"{'; '.join(_describe(s) for s in offending)}. `--no-sync` skips the LOCK step "
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


@pytest.mark.parametrize(
    ("script", "expected_flags"),
    [
        pytest.param(
            'SWEEP_CMD=("$UV_RESOLVED" run --no-sync --project "$FM" python)\n',
            [("--no-sync", "--project")],
            id="resolved-uv-variable",
        ),
        pytest.param(
            'PREPARE_CMD=(${PREPARE_CMD:-uv run --frozen --project "$REPO/shared" python})\n',
            [("--frozen", "--project")],
            id="seam-default-array",
        ),
        pytest.param(
            'read -r -a SWEEP_CMD <<<"${X_CMD:-uv run --frozen --project $REPO/shared python}"\n',
            [("--frozen", "--project")],
            id="seam-default-read",
        ),
        pytest.param(
            "/home/leo/.local/bin/uv run --no-sync --project=shared python x.py\n",
            [("--no-sync", "--project")],
            id="absolute-uv-path",
        ),
        pytest.param(
            '"$UV" run \\\n    --no-sync --project "$FM" python x.py\n',
            [("--no-sync", "--project")],
            id="continuation",
        ),
        pytest.param(
            'PREPARE_CMD=(${PREPARE_CMD:-uv run --frozen --project "$REPO/shared" python})\n'
            'RETURN_BRIEF_CMD=(${RETURN_BRIEF_CMD:-uv run --no-sync --project "$REPO/shared" python})\n',
            [("--frozen", "--project"), ("--no-sync", "--project")],
            id="two-commands-in-file-order",
        ),
    ],
)
def test_shell_uv_run_sites_reads_every_committed_spelling(
    script: str, expected_flags: list[tuple[str, ...]]
) -> None:
    """Guard the guard: every way a committed script spells `uv run` yields its site."""
    sites = shell_uv_run_sites(script, "planted.sh")
    assert [s.flags for s in sites] == expected_flags


@pytest.mark.parametrize(
    ("script", "expected_flags"),
    [
        pytest.param(
            "# uv run --frozen --project fused-memory python \\\n"
            "X=(uv run --no-sync python)\n",
            [("--no-sync",)],
            id="comment-skipped-and-its-backslash-joins-nothing",
        ),
        pytest.param('"$UV" sync --all-packages\n', [], id="uv-sync"),
        pytest.param("uvx --from foo bar\n", [], id="uvx"),
        pytest.param('python3 "$REPO/x.py"\n', [], id="python3"),
        pytest.param(
            'source "$_uv_lib" || { echo "cannot load the shared uv resolver $_uv_lib '
            '-- refusing to guess which uv to run." >&2; exit 127; }\n',
            [],
            id="uv-and-run-in-prose",
        ),
        pytest.param('CENSUS_CMD=("${UV_PYTHON[@]}")\n', [], id="uv-array-expansion"),
    ],
)
def test_shell_uv_run_sites_ignores_what_is_not_a_uv_run(
    script: str, expected_flags: list[tuple[str, ...]]
) -> None:
    """Guard the guard: comments, `uv sync`, `uvx`, bare python and prose are not sites."""
    sites = shell_uv_run_sites(script, "planted.sh")
    assert [s.flags for s in sites] == expected_flags


@pytest.mark.parametrize(
    "script",
    [
        pytest.param('"${UV:-uv}" run --frozen --project shared python\n', id="defaulted-expansion"),
        pytest.param(
            '"$(command -v uv)" run --frozen --project shared python\n',
            id="command-substitution",
        ),
        pytest.param('"$uv_bin" run --frozen --project shared python\n', id="lowercase-variable"),
        pytest.param(
            "uv --quiet run --frozen --project shared python\n", id="global-flag-before-run"
        ),
        pytest.param(
            "A=(uv run --no-sync --project shared python)\n"
            'B=("${UV:-uv}" run --frozen --project shared python)\n',
            id="one-of-two-sites-respelled",
        ),
    ],
)
def test_shell_uv_run_sites_raises_on_a_uv_run_it_cannot_read(script: str) -> None:
    """Guard the guard: an unread uv run fails naming its script, never answers "none"."""
    with pytest.raises(UnreadableShellUvRun, match=r"^scripts/planted\.sh: "):
        shell_uv_run_sites(script, "scripts/planted.sh")


def test_shell_uv_run_sites_names_the_script_holding_an_unknown_run_level_flag() -> None:
    """The flag walk's raise names the script file, not only the command text."""
    with pytest.raises(UnreadableShellUvRun, match=r"^scripts/planted\.sh: .*'--reinstall'"):
        shell_uv_run_sites('echo "try uv run --reinstall"\n', "scripts/planted.sh")


def test_shell_uv_run_site_names_its_source_and_logical_line() -> None:
    """A site carries the file it was read from and the JOINED command, for diagnostics."""
    [site] = shell_uv_run_sites(
        '"$UV" run \\\n    --no-sync --project "$FM" python x.py\n', "scripts/planted.sh"
    )
    assert site.source == "scripts/planted.sh"
    assert site.command == '"$UV" run --no-sync --project "$FM" python x.py'


def _plant(root: pathlib.Path, files: dict[str, str]) -> None:
    """Write each *root*-relative path in *files* with its text."""
    for rel_path, text in files.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    ("exec_start", "expected"),
    [
        pytest.param("/srv/deployed/scripts/w.sh --flag", "scripts/w.sh", id="longest-suffix"),
        pytest.param("-/srv/deployed/scripts/w.sh", "scripts/w.sh", id="systemd-exec-prefix"),
        pytest.param("/srv/deployed/scripts/w.py", None, id="not-a-shell-script"),
    ],
)
def test_wrapper_script_of_finds_a_shell_script_in_this_checkout(
    tmp_path: pathlib.Path, exec_start: str, expected: str | None
) -> None:
    """A unit's wrapper is found by path SUFFIX, so the deployed path answers in any checkout."""
    _plant(tmp_path, {"scripts/w.sh": "", "w.sh": "", "scripts/w.py": ""})
    assert wrapper_script_of(exec_start, tmp_path) == expected


def test_wrapper_script_of_raises_for_a_shell_script_the_checkout_lacks(
    tmp_path: pathlib.Path,
) -> None:
    """A renamed or mistyped wrapper fails loudly instead of reading as "runs no shell script"."""
    with pytest.raises(MalformedExecStart, match=r"no-such-wrapper\.sh"):
        wrapper_script_of("/srv/deployed/scripts/no-such-wrapper.sh", tmp_path)


_WRAPPER_UNIT = "[Service]\nExecStart=/srv/deployed/scripts/u.sh\n"


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            {"u.service": "[Service]\nExecStart=/usr/bin/uv run --no-sync --project shared x\n"},
            [("u.service", ("--no-sync", "--project"))],
            id="direct-uv-run",
        ),
        pytest.param(
            {
                "u.service": _WRAPPER_UNIT,
                "scripts/u.sh": "#!/bin/bash\nexec uv run --frozen --project shared python x.py\n",
            },
            [("scripts/u.sh", ("--frozen", "--project"))],
            id="followed-wrapper",
        ),
        pytest.param(
            {"u.service": "[Service]\nExecStart=/usr/bin/python3 -m x\n"},
            [],
            id="no-wrapper",
        ),
        pytest.param(
            {"u.service": _WRAPPER_UNIT, "scripts/u.sh": '#!/bin/bash\nexec python3 "$R/x.py"\n'},
            [],
            id="wrapper-running-no-uv-run",
        ),
    ],
)
def test_uv_run_sites_reads_the_start_directly_or_through_its_wrapper(
    tmp_path: pathlib.Path,
    files: dict[str, str],
    expected: list[tuple[str, tuple[str, ...]]],
) -> None:
    """Each branch of a unit's start: its own `uv run`, its wrapper's, or none at all."""
    _plant(tmp_path, files)
    assert [(s.source, s.flags) for s in uv_run_sites("u.service", tmp_path)] == expected


def test_uv_run_sites_raises_for_a_wrapper_the_checkout_lacks(tmp_path: pathlib.Path) -> None:
    """A unit naming a missing wrapper fails rather than leaving the sweep as "no uv run"."""
    _plant(tmp_path, {"u.service": _WRAPPER_UNIT})
    with pytest.raises(MalformedExecStart, match=r"scripts/u\.sh"):
        uv_run_sites("u.service", tmp_path)


def test_discover_uv_run_scripts_reads_every_shell_script_outside_test_trees(
    tmp_path: pathlib.Path,
) -> None:
    """Guard the guard: every *.sh with a site, an unreadable one included; test trees and non-.sh excluded."""
    git(tmp_path, "init", "-q")
    _plant(
        tmp_path,
        {
            "top.sh": "uv run --no-sync --project shared python x.py\n",
            "scripts/a.sh": "uv run --no-sync --project shared python x.py\n",
            "scripts/odd.sh": '"${UV:-uv}" run --frozen --project shared python\n',
            "scripts/plain.sh": "python3 x.py\n",
            "scripts/tests/t.sh": "uv run --frozen --project shared python x.py\n",
            "pkg/tests/u.sh": "uv run --frozen --project shared python x.py\n",
            "tests/root.sh": "uv run --frozen --project shared python x.py\n",
            "scripts/notes.txt": "uv run --frozen --project shared python x.py\n",
        },
    )
    assert discover_uv_run_scripts(tmp_path) == ["scripts/a.sh", "scripts/odd.sh", "top.sh"]


def test_script_uv_run_sites_names_the_script_it_read(tmp_path: pathlib.Path) -> None:
    """A script's sites carry its *root*-relative path as their source."""
    _plant(tmp_path, {"scripts/w.sh": "exec uv run --frozen --project shared python x.py\n"})
    assert [(s.source, s.flags) for s in script_uv_run_sites("scripts/w.sh", tmp_path)] == [
        ("scripts/w.sh", ("--frozen", "--project"))
    ]
