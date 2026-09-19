"""Fleet-wide guard: starting a unit must not mutate the shared root .venv.

Every workspace member named in the root pyproject.toml's
``[tool.uv.workspace]`` resolves to ONE ``.venv`` at the repo root — there is
no member-local environment.  Measured on uv 0.11.6 against a throwaway
two-member workspace: ``uv run --project a`` reports ``sys.prefix`` as the
WORKSPACE ROOT ``.venv`` and never creates ``a/.venv``.  So a unit's ExecStart
does not act on an environment of its own; it acts on the environment the
seven orchestrators, the dashboard, the load sampler, fused-memory and every
verify subprocess are already running out of.

``uv run`` is an INEXACT sync: at process start it INSTALLS the named member's
missing dependency closure into that shared venv (measured: "Installed 1
package in 42ms"), and never prunes.  ``--frozen`` does NOT stop it — that flag
is "run without updating the uv.lock file", a LOCKFILE option, and a
``uv run --frozen --project a`` start installed the deleted package right back
("Installed 1 package in 50ms").  ``--no-sync`` is the flag that stops it, and
under it the venv was untouched.

That distinction is the whole reason this module exists.  From 2026-05-29 until
task 5553 the fleet carried ``--frozen`` believing it was the venv guard, and
nothing mechanical contradicted the belief, because the per-unit assertions of
the day pinned the flag's TEXT rather than its effect.  This sweep asserts the
effect's precondition instead, fleet-wide and by content discovery, so a unit
added next month is covered the day it lands.

Structurally modelled on tests/scripts/test_systemd_restart_backoff.py — the
same exclusions for the same measured reasons, the same non-empty discovery
guard, and the same single markdown opt-in.  Its ``_NON_UNIT_PATHSPECS``
comment states that reasoning at length and is CITED rather than restated here,
so the two sweeps cannot drift into two different answers about what counts as
a unit.
"""
import pathlib
import subprocess

import pytest
from systemd_unit_invariants import MalformedExecStart

REPO_ROOT = pathlib.Path(__file__).parents[2]

# What discovery refuses to treat as a unit, excluded by CATEGORY rather than
# by naming individual paths.  Verbatim from
# tests/scripts/test_systemd_restart_backoff.py:31-65, whose comment holds the
# full reasoning and the measurements behind it; the summary is that
# ``**/tests/**`` files embed whole units as column-0 triple-quoted fixtures
# (several per file), and ``**/*.md`` files are prose that may legitimately
# quote the DEFECT — plans/afk-C1-systemd.md is the live instance, an as-built
# record whose ExecStart carries no lockfile flag at all and which must not be
# edited.  Both categories break a file-wide last-occurrence-wins read the same
# way.  The glob forms are load-bearing: a plain ``:!tests/`` excludes only the
# top-level directory, leaving fused-memory/tests/, orchestrator/tests/,
# scripts/tests/ and dashboard/tests/ swept.
_NON_UNIT_PATHSPECS = (
    ":(exclude,glob)**/tests/**",
    ":(exclude,glob)**/*.md",
)

# Every committed unit or template whose ExecStart is a `uv run` against a
# workspace member, i.e. every file that carries this obligation today.  Fifteen
# of them, spanning four naming conventions and five directories.
#
# Unlike test_systemd_restart_backoff.py's one-sided coverage guard, this set is
# asserted by EQUALITY (see test_discovery_covers_every_known_uv_run_unit) —
# discovery here is two-stage (a git grep, then a `uv run` filter), and the
# second stage is a parser that can silently answer None for a unit it stopped
# understanding.  A missing-only check cannot see that, and a unit that drops
# out of the sweep is exactly the failure this module exists to prevent.
_EXPECTED_UV_RUN_UNITS = frozenset(
    {
        "dashboard/dark-factory-dashboard.service",
        "dashboard/dark-factory-load-sampler.service",
        "fused-memory/fused-memory.service.example-systemd-config",
        "scripts/dashboard.service.template",
        "scripts/fused-memory.service.template",
        "scripts/legibility-transcript-check@.service",
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

# uv's own render sentinel in the two `.service.template` files, substituted for
# an absolute uv path at install time by scripts/setup-host.sh.  Recognised so a
# template is swept in its COMMITTED form rather than only after rendering.
_UV_PATH_SENTINEL = "__UV_PATH__"

# The one markdown file that OPTS BACK IN to both sweeps, mirroring
# tests/scripts/test_systemd_restart_backoff.py:61-64.  The `**/*.md` exclusion
# above is kept for the reason that module states — a doc may legitimately quote
# the DEFECT, and plans/afk-C1-systemd.md is the live as-built record that must
# not be edited — but this file is not prose ABOUT a unit, it is the unit new
# projects are minted from: its own line 16 says the scripts/orchestrator-*
# .service files are "cp'd verbatim by setup-host.sh" and points at
# orchestrator-reify.service as the model.  A copy-source still showing the old
# flags mints this defect into every new project, one unit at a time, with
# nothing in the sweep able to see it happen.
_FACTORY_INIT_REFERENCE = "skills/factory-init/references/supervised-unit.md"

# Run-level flags that take a separate value token, so the walk below consumes
# the value instead of mistaking it for the command token and stopping early.
# `--project` is the one every unit here uses; the rest are uv's other
# environment-selecting options, listed so a unit that grows one does not
# silently truncate its own flag list.
_VALUE_TAKING_RUN_FLAGS = frozenset(
    {"--project", "--package", "--python", "--with", "--directory"}
)


def discover_exec_start_files() -> list[str]:
    """Return every git-tracked file declaring ``ExecStart=``, sans exclusions.

    Discovery is by CONTENT rather than by filename glob or a hand-maintained
    list, for the reason
    test_systemd_restart_backoff.discover_units_declaring_a_restart_cap states
    in full: the affected files span four naming conventions (``*.service``,
    ``*.service.template``, ``*.example-systemd-config``, and a fenced ini block
    inside a ``.md``) across five directories, so any glob broad enough to catch
    them all is broader than the invariant, and a hand-maintained list fails the
    stated goal outright — a unit added next month is simply not in it.

    The anchor is ``ExecStart=`` rather than ``uv run``, i.e. deliberately
    WIDER than the obligation.  Narrowing the grep to `uv` would make a unit
    whose ExecStart stopped being recognisable as a uv invocation vanish from
    discovery silently; anchoring on the directive every unit must have keeps it
    in the discovered set, where the parser can answer for it and the coverage
    guard can notice if the answer changed.  Filtering to actual `uv run`
    commands is the SECOND stage, in uv_run_level_flags.

    The returncode assertion is load-bearing.  ``git grep`` exits 0 on matches,
    1 on no matches and >1 on a real error; a helper that swallows non-zero
    returns [], the parametrize below collects ZERO cases, and the sweep reports
    green while checking nothing.

    The pattern tolerates leading whitespace and whitespace around the
    separator, matching systemd.syntax — a valid ``  ExecStart = /usr/bin/uv``
    must not go undiscovered.  The trailing ``=`` in the pattern is what keeps
    ``ExecStartPre=`` from matching on its own; a unit whose only match is a
    pre-hook has no command to check.
    """
    proc = subprocess.run(
        [
            "git",
            "grep",
            "-lE",
            r"^[ \t]*ExecStart[ \t]*=",
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
        "nothing to check. A non-zero exit here must fail loudly: silently "
        "returning no paths would collect zero parametrized cases and report "
        f"this whole sweep green while checking nothing. stderr: "
        f"{proc.stderr.strip()!r}"
    )
    return sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())


def logical_exec_start(text: str, unit_name: str = "<unit>") -> str:
    """Return the effective ExecStart COMMAND in *text* as one logical line.

    Two independent normalisations, each of which a naive read gets wrong on a
    file committed in this repo today.

    LAST OCCURRENCE WINS, mirroring systemd itself and
    test_orchestrator_service_files._exec_start_line, whose docstring holds the
    reasoning: a drop-in override under ``<unit>.d/`` lands as an empty
    ``ExecStart=`` list RESET followed by the real command, so a first-match
    read finds the reset, sees no flags at all, and passes a unit whose real
    command may well be wrong.  Lines are stripped before matching because
    systemd permits leading whitespace on a directive, and the trailing ``=``
    in the prefix is what keeps ``ExecStartPre=`` out of the match.

    CONTINUATIONS ARE JOINED.  scripts/dashboard.service.template:34 and
    scripts/fused-memory.service.template:39 (with its committed mirror) write
    ExecStart as a systemd backslash continuation spanning several physical
    lines, so a single-physical-line read sees only the first fragment.  For
    those three that fragment happens to hold the run-level flags today, which
    is worse than useless: the read would pass them VACUOUSLY and stop noticing
    the day a flag moved to a continuation line.  The join follows
    test_dashboard_service_template._logical_exec_start — drop the trailing
    ``\\``, strip continuation indentation, separate with a single space.

    Returns the command only, with the ``ExecStart=`` prefix removed, since
    every caller wants to tokenise a command line and none wants the directive
    name.  Raises MalformedExecStart — the shared class from
    systemd_unit_invariants, so a broken unit surfaces as ONE class whichever
    layer notices it first — when there is no ExecStart= at all, or when the
    effective one carries no command.  Neither is a legitimate "this unit has
    no uv flags" answer: the None return belongs to uv_run_level_flags and
    means something quite different.
    """
    lines = text.splitlines()
    start_indices = [
        i for i, ln in enumerate(lines) if ln.strip().startswith("ExecStart=")
    ]
    if not start_indices:
        raise MalformedExecStart(
            f"{unit_name} declares no ExecStart= line, so there is no command "
            "to check for run-level uv flags. Treating this as 'not a uv run "
            "command' would silently drop the unit out of the fleet sweep — "
            "the exact direction a guard against a silently-mutated venv must "
            "refuse."
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

    command = " ".join(p for p in parts if p).partition("=")[2].strip()
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
    so the invariant does not apply to it and callers SKIP.  That is a real
    answer — scripts/orchestrator-watchdog.service runs a bare Python probe and
    scripts/jcodemunch-watcher.service.template runs ``uvx``, which builds an
    EPHEMERAL environment and never touches the shared root venv.  The
    None-vs-raise split follows the contract stated once at
    systemd_unit_invariants.config_arg_from_exec_start: a real answer returns,
    a defect raises.

    RUN-LEVEL is the load-bearing word, and the reason this returns a walked
    prefix rather than doing a substring check on the whole line.  In

        uv run --project orchestrator orchestrator run --config <path>

    ``orchestrator`` is the COMMAND token, and every argument after it belongs
    to the orchestrator CLI, not to uv.  A flag appended there would leave the
    unit textually satisfying a naive presence check while uv never saw it and
    the venv was still mutated at start — a guard blessing the exact defect it
    was written for.  So the walk starts after ``run`` and STOPS at the first
    token that is neither a ``--flag`` nor the value of one.
    orchestrator/tests/test_mcp_lifecycle.py:335-363 already guards this same
    positional property for the plan-tools fast-start argv, where the repo
    spells the flag ``--no-sync`` — so the hot path had the right flag all
    along while the units did not.

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
        flags.append(name)
        # A `--flag=value` token carries its own value; only the space-separated
        # spelling consumes the token after it.
        if name in _VALUE_TAKING_RUN_FLAGS and "=" not in token:
            i += 1
        i += 1
    return flags


def discover_uv_run_units() -> list[str]:
    """Repo-relative paths of every discovered file whose ExecStart is a `uv run`."""
    units = []
    for rel_path in discover_exec_start_files():
        text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
        if uv_run_level_flags(logical_exec_start(text, rel_path)) is not None:
            units.append(rel_path)
    return units


def swept_paths() -> list[str]:
    """Every path both arms below run against: the discovered units plus the opt-in.

    The markdown opt-in is added UNCONDITIONALLY — never as a skip, mirroring
    test_systemd_restart_backoff.py:61-64 — so the exclusion of the `**/*.md`
    category costs this file no coverage.  Its guard is therefore strictly
    stronger than the sweep, not a weaker substitute for it.
    """
    return sorted([*discover_uv_run_units(), _FACTORY_INIT_REFERENCE])


def test_factory_init_reference_is_swept() -> None:
    """The opt-in copy-source must exist and still parse as a `uv run` command.

    Without this, a rename of the file or a reformat of its fenced ini block
    would drop the ONLY copy-source out of both arms silently — the file would
    simply stop being a path the parametrize produced, and two tests would
    quietly become fourteen cases instead of fifteen with nothing red.  That is
    the same silent-shrink hazard the coverage guard above exists for, and it
    needs its own statement here because this path does not come from discovery.
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

    The counterpart to test_orchestrator_service_glob_covers_all_known_units and
    to test_exec_start_config_parser_answers_for_every_orchestrator_run_unit.
    Without it a broken discovery shrinks the sweep to zero cases, and a
    zero-case parametrize collects no tests and reports no failure — masking the
    very defect the sweep exists to catch.

    EQUALITY, not the one-sided missing-only check its structural model
    test_systemd_restart_backoff.test_discovery_covers_every_known_unit uses,
    and the difference is deliberate.  There, discovery is a single git grep
    whose whole point is that a NEW unit declaring a cap is swept automatically,
    so an equality assertion would turn that success into a failure.  Here the
    second stage is a PARSER, and a parser has a silent failure mode a grep does
    not: uv_run_level_flags answers None for anything it no longer recognises as
    ``uv run``, so a unit reformatted in a way the walker mishandles leaves the
    sweep quietly rather than failing in it.  A new unit landing here is
    expected to add its path, and the one-line diff is the price of that being
    a decision rather than an accident.
    """
    discovered = set(discover_uv_run_units())
    assert discovered, (
        "discovery found no `uv run` units at all. Most likely causes: the "
        f"git grep ran outside a checkout (REPO_ROOT={REPO_ROOT}), or "
        "uv_run_level_flags stopped recognising every ExecStart. Either way "
        "the parametrized sweeps below would collect zero cases and report "
        "green while checking nothing."
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
        "so carry the same obligation. If they are not units at all, they must "
        "be excluded via _NON_UNIT_PATHSPECS rather than checked."
    )


@pytest.mark.parametrize("rel_path", swept_paths(), ids=lambda p: p)
def test_uv_run_unit_passes_no_sync(rel_path: str) -> None:
    """A unit's `uv run` must carry `--no-sync` among its RUN-LEVEL flags.

    The fleet-wide arm of the invariant this module's docstring states.
    """
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    flags = uv_run_level_flags(logical_exec_start(text, rel_path))
    assert flags is not None and "--no-sync" in flags, (
        f"{rel_path} runs `uv run` without a run-level `--no-sync` "
        f"(run-level flags found: {flags}). Measured on uv 0.11.6: a start "
        "without it INSTALLS the named member's missing dependency closure "
        "into the ONE shared root .venv — the same environment every other "
        "workspace member, every running orchestrator and every verify "
        "subprocess resolves against — because `uv run --project <member>` "
        "reports sys.prefix as the workspace root and no member-local venv "
        "exists. `--frozen` does NOT prevent this: it is a LOCKFILE option "
        "('run without updating the uv.lock file'), and a `uv run --frozen` "
        "start was measured reinstalling a package deleted from the venv. "
        "The flag must sit BEFORE the command token, or uv never sees it. "
        "The accepted cost is that an unsynced venv now fails the unit with "
        "ModuleNotFoundError instead of repairing itself; the repair path is "
        "scripts/sync-orchestrator-env.sh (`uv sync --all-packages`)."
    )


@pytest.mark.parametrize("rel_path", swept_paths(), ids=lambda p: p)
def test_uv_run_unit_carries_no_lockfile_flag(rel_path: str) -> None:
    """Beside `--no-sync`, a run-level `--frozen`/`--locked` is a no-op — so forbid it.

    Its own test rather than a second assert inside
    test_uv_run_unit_passes_no_sync: the two arms fail for different reasons and
    a reader of a failure should see which one fired, and the presence arm must
    stay green independently of this one.
    """
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    flags = uv_run_level_flags(logical_exec_start(text, rel_path))
    assert flags is not None
    lockfile_flags = [f for f in flags if f in ("--frozen", "--locked")]
    assert not lockfile_flags, (
        f"{rel_path} carries run-level {lockfile_flags} beside `--no-sync` "
        f"(run-level flags found: {flags}). Measured on uv 0.11.6: `--no-sync` "
        "skips the LOCK step along with the sync, so both lockfile flags buy "
        "literally nothing next to it. `uv run --locked` exits 2 on lockfile "
        "drift, but `uv run --locked --no-sync` exits 0 SILENTLY; and "
        "`uv run --frozen --no-sync` leaves uv.lock's sha256 byte-identical to "
        "what `--no-sync` alone leaves it — the two invocations are "
        "indistinguishable. A flag that survives review by looking like it "
        "strengthens the line while doing nothing is worse than no flag: "
        "`--frozen` is exactly how this defect was introduced, because it "
        "READS as a venv guarantee and is in fact a lockfile option ('run "
        "without updating the uv.lock file'). If loud failure on lockfile "
        "drift is ever genuinely wanted for a unit, it cannot be bought here — "
        "it would mean dropping `--no-sync`, which reinstates the hazard. "
        "Check the lockfile somewhere that is not a unit start."
    )
