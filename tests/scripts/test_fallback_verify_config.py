"""Config-integrity test for the FALLBACK full-suite verify command.

Per task 2361: the monorepo FALLBACK full-suite verify path is driven by the
fleet-chain ``test_command`` string in ``dark-factory-orchestrator.yaml``, returned
verbatim by ``_build_fallback_config``'s ``__fallback__`` branch whenever
touched files map to no single subproject (e.g. a change confined to the
repo-root ``tests/scripts/`` suite, which is not a workspace member).

Two defects this guards against:

  (1) Coverage gap — the fleet chain ran each of the 5 subprojects' own
      ``tests/`` dirs but never ``tests/scripts/`` itself, so a task that only
      touches ``tests/scripts/`` was gated on unrelated subprojects while its
      own tests never ran in the gating path.

  (2) Host-oversubscription fragility — ``orchestrator`` and ``fused-memory``
      run pytest-xdist with a 60s per-test ``pyproject.toml`` timeout. Under
      host CPU oversubscription an xdist worker can starve past that 60s
      wall-clock ceiling; pytest-timeout's thread method then ``os._exit()``s
      the worker, and ``--max-worker-restart=0`` degrades that into a clean
      per-test "node down: Not properly terminated" failure — tripping on
      whatever test happens to be running at the time, not a real per-test
      defect. Raising the per-test timeout well above 60s (CLI ``--timeout``
      overrides the pyproject default) removes the trigger.

This test loads the *committed* ``dark-factory-orchestrator.yaml`` directly
(not through the verify.py fallback-builder code path) — mirrors the pattern
in ``test_orchestrator_restart_config_drift.py``.
"""

import os
import pathlib
import re
import shlex
import sys
import tomllib
from typing import NamedTuple

import yaml
from orchestrator.config import ModuleConfig, _discover_module_configs
from orchestrator.verify import _AND_CLAUSE_SPLIT_RE, _cd_clause_target

REPO_ROOT = pathlib.Path(__file__).parents[2]
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"

# Task 2769: the per-module merge-verify orchestrator.yaml files each carry
# their own subproject-scoped test_command (distinct from the repo-root
# FALLBACK fleet chain in dark-factory-orchestrator.yaml above). Rather than
# hardcode the list — which silently fails to cover a NEW subproject that
# later adds its own orchestrator.yaml + test_command (reviewer drift
# concern, task 2769 amendment) — the guard below DISCOVERS them at runtime
# via ``_discover_per_module_configs``: every module config the orchestrator
# itself registers, at ANY depth, that defines a ``test_command``. A
# newly-added subproject is therefore auto-covered and cannot regress to the
# flaky 60s pyproject default without failing this test.
#
# The known names below are retained only as a *floor* (proof discovery still
# resolves them), NOT as the authoritative list. Entries are repo-relative
# module PREFIXES, not bare directory names — task 3350, so the depth-2
# ``tests/scripts`` config cannot collide with the depth-1 ``scripts`` one.
# Depth-1 prefixes are identical to their bare names, so the pre-existing
# entries are unchanged.
KNOWN_PER_MODULE_CONFIG_NAMES = frozenset(
    {
        "shared",
        "escalation",
        "orchestrator",
        "fused-memory",
        "dashboard",
        "sampler",
        "scripts",
        "tests/scripts",
    }
)

# Documented, temporary carve-out: a discovered config known to lack the
# --timeout override AND lying outside task 2769's module locks, so it can't
# be fixed in this change. ``cockpit/orchestrator.yaml`` landed on main after
# task 2769 was scoped and still inherits the flaky 60s default; it is
# surfaced for a follow-up fix via an escalate_info observation. DELETE the
# entry here once cockpit's test_command carries ``--timeout>=300`` — at
# which point dynamic discovery covers it automatically with no further edit.
TIMEOUT_GUARD_EXCLUSIONS = frozenset({"cockpit"})


def _fleet_test_command() -> str:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))["test_command"]


def _fleet_type_check_command() -> str:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))["type_check_command"]


def _fleet_lint_command() -> str:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))["lint_command"]


def _pyproject_at(rel_dir: str) -> dict:
    """Parse the ``pyproject.toml`` of the repo-relative directory *rel_dir*."""
    path = REPO_ROOT / rel_dir / "pyproject.toml"
    assert path.is_file(), (
        f"no pyproject.toml at {rel_dir}/ (task 3367) — the interpreter-pin "
        "invariant cannot be evaluated for a directory with no pyright config"
    )
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _workspace_member_dirs() -> list[str]:
    """Every ``[tool.uv.workspace].members`` entry, discovered at runtime.

    Lifted (task 3397) from what was originally
    ``TestWorkspacePyrightInterpreterPinned._workspace_member_dirs`` (task
    3367) to module level, so the fleet TYPE/LINT coverage invariants below
    discover workspace members from the exact same runtime source as the
    interpreter-pin invariant rather than a second, independently-maintained
    list that could silently drift from it. The class method now delegates
    here.
    """
    root = _pyproject_at(".")
    members = root.get("tool", {}).get("uv", {}).get("workspace", {}).get("members")
    assert members, (
        "root pyproject.toml declares no [tool.uv.workspace].members (task "
        "3367) — the workspace-wide interpreter-pin invariant cannot discover "
        "its subjects and would pass vacuously"
    )
    return list(members)


def _assert_pyright_pins_worktree_venv(rel_dir: str, pyright: dict, why: str) -> None:
    """Assert *rel_dir*'s ``[tool.pyright]`` table pins the worktree-root ``.venv``.

    Shared by both interpreter-pin invariants below (task 3367): the narrow
    fleet-chain guard and the general every-workspace-member guard assert the
    SAME property, and must not drift apart.
    """
    for key in ("venvPath", "venv"):
        assert key in pyright, (
            f"{rel_dir}/pyproject.toml [tool.pyright] does not declare {key!r} "
            f"(task 3367, esc-3359-1). {why} Without an explicit venvPath/venv "
            "pin, pyright resolves its Python interpreter from the ambient "
            "VIRTUAL_ENV/PATH — which verify._target_subprocess_env deliberately "
            "strips — so in a cold merge worktree it type-checks against an "
            "environment holding none of the workspace's third-party packages "
            "and emits hundreds of phantom 'could not be resolved' "
            "(reportMissingImports) errors, false-reddening TYPE on a branch "
            "with no defect (509+ errors on a DOCS-ONLY diff in esc-3359-1)"
        )
    resolved = (REPO_ROOT / rel_dir / pyright["venvPath"] / pyright["venv"]).resolve()
    expected = (REPO_ROOT / ".venv").resolve()
    assert resolved == expected, (
        f"{rel_dir}/pyproject.toml [tool.pyright] pins venvPath="
        f"{pyright['venvPath']!r} venv={pyright['venv']!r}, which resolves to "
        f"{resolved} — not this worktree's own root .venv at {expected} (task "
        "3367, esc-3359-1). The pin is resolved by pyright RELATIVE TO THE "
        "CONFIG FILE'S OWN DIRECTORY, which is what makes it correct per-worktree; "
        "a pin that resolves anywhere else reintroduces cross-worktree "
        "interpreter leakage"
    )


def _discover_per_module_configs() -> dict[str, ModuleConfig]:
    """Every discovered module config (ANY depth) that defines a ``test_command``.

    Keyed by repo-relative module prefix (``shared``, ``tests/scripts``), which
    is what the orchestrator itself registers configs under.

    Delegates to the PRODUCTION walk ``config._discover_module_configs`` rather
    than globbing (task 3350). The previous ``REPO_ROOT/*/orchestrator.yaml``
    glob looked ONE level deep, so a config like ``tests/scripts/
    orchestrator.yaml`` silently escaped the per-test timeout guard below —
    breaking this helper's own promise that "a newly-added subproject is
    auto-covered ... and cannot silently regress to the 60s default". It also
    keyed on ``path.parent.name``, colliding ``scripts`` with ``tests/scripts``.

    Delegating closes both by construction: the guard now checks exactly the set
    the orchestrator registers, at any depth, and inherits the production
    pruning of ``.worktrees``, ``.venv``, ``node_modules``, ``build``, ``target``,
    ``.claude`` and any nested ``.git`` checkout. A hand-rolled
    ``**/orchestrator.yaml`` was rejected: run from the main checkout it would
    descend ``.worktrees/`` and ``.venv/``, and it would remain free to drift
    from what discovery actually does.

    Still naturally excludes the repo-root ``dark-factory-orchestrator.yaml``
    (different filename; the production walk also skips a root-level
    ``orchestrator.yaml`` at prefix ``"."``) — that file is checked by the
    FALLBACK tests above.

    Importing ``orchestrator.config`` from this suite follows the precedent set
    by ``test_orchestrator_restart_config_drift.py`` and
    ``test_offline_lane_qdrant_config.py`` in this same directory.
    """
    return {
        prefix: mc
        for prefix, mc in _discover_module_configs(REPO_ROOT).items()
        if mc.test_command
    }


def _pytest_segments(cmd: str) -> list[str]:
    return [seg for seg in cmd.split("&&") if "pytest" in seg]


def test_fallback_verify_runs_tests_scripts() -> None:
    """The fallback fleet chain must also run the repo-root tests/scripts/ suite.

    Task 2361: tests/scripts/ (e.g. test_spawn_claude.py) is not a workspace
    member and is therefore never covered by any of the 5 subprojects' own
    ``tests/`` dirs. Without this, a task scoped only to tests/scripts/ is
    gated on unrelated subprojects' flakes while its own tests never run in
    the FALLBACK gating path.

    Checked on the pytest-bearing segments (not the raw string) so the
    assertion proves tests/scripts/ is actually invoked by a pytest
    command — not merely mentioned somewhere else in the chain (e.g. inside
    an --ignore flag or an unrelated path fragment).
    """
    cmd = _fleet_test_command()
    segments = _pytest_segments(cmd)
    assert any("tests/scripts" in seg for seg in segments), (
        "dark-factory-orchestrator.yaml test_command (FALLBACK full-suite verify) "
        "must have a pytest segment that runs the repo-root tests/scripts/ "
        f"suite (task 2361), not merely mention it elsewhere; got: {cmd!r}"
    )


def test_fanout_includes_sampler_member() -> None:
    """The fallback fleet chain must also run sampler's own tests/ suite.

    Task 2368: ``sampler`` is a workspace member (root pyproject.toml
    ``[tool.uv.workspace].members``) with its own ``pyproject.toml`` and
    ``tests/``, but was never added to the fleet fanout — so a task touching
    sampler alongside a root-level file (which disqualifies the
    subproject-scoped fallback and falls through to this fleet chain) never
    exercised sampler's own tests in the gating path. The segment must be a
    HARD ``cd ../sampler`` (sampler is present on main, unlike cockpit —
    see :func:`test_fanout_includes_cockpit_presence_guarded`) and carry a
    pytest segment with ``--timeout>60`` like every other segment in this
    chain.
    """
    cmd = _fleet_test_command()
    assert re.search(r"cd \.\./sampler\b", cmd), (
        "dark-factory-orchestrator.yaml test_command (FALLBACK full-suite verify) "
        "must fan out to sampler/ via 'cd ../sampler' (task 2368) so "
        f"sampler's own tests run; got: {cmd!r}"
    )
    # Matched directly against the raw command (not _pytest_segments' crude
    # &&-split) because the identifying token 'sampler' lives in the 'cd'
    # clause, not the pytest clause that follows it (mirrors the existing
    # per-subproject segments, e.g. 'cd shared && uv run pytest tests/ …').
    match = re.search(
        r"cd \.\./sampler\s*&&\s*uv run pytest tests/\s*--timeout[=\s](\d+)",
        cmd,
    )
    assert match, (
        "dark-factory-orchestrator.yaml test_command (FALLBACK full-suite verify) "
        "must run a pytest segment for sampler's own tests/ suite immediately "
        f"after 'cd ../sampler' (task 2368); got: {cmd!r}"
    )
    timeout_value = int(match.group(1))
    assert timeout_value > 60, (
        f"sampler's pytest segment sets --timeout={timeout_value}, which does "
        "not raise the ceiling above the flaky 60s pyproject default (task 2368)"
    )


def test_fanout_includes_cockpit_presence_guarded() -> None:
    """The fallback fleet chain must run cockpit's tests, guarded by a presence check.

    Task 2368: ``cockpit`` is introduced by the un-merged Fleet Cockpit
    batch (2291-2303) and is absent from main. A HARD ``cd ../cockpit``
    segment would return non-zero and abort this entire ``&&`` chain —
    breaking BOTH the fleet fallback AND merge-queue verify for every task
    until cockpit lands. The segment must instead be guarded by a
    ``[ -d cockpit ]`` presence check that skips cleanly when the directory
    is absent, while still running (and propagating failures from) cockpit's
    own tests/ suite once it exists — carrying a pytest segment with
    ``--timeout>60`` like every other segment in this chain.
    """
    cmd = _fleet_test_command()
    assert "cockpit" in cmd, (
        "dark-factory-orchestrator.yaml test_command (FALLBACK full-suite verify) "
        f"must reference cockpit (task 2368); got: {cmd!r}"
    )
    assert re.search(r"\[\s*-d\s+cockpit\s*\]", cmd), (
        "dark-factory-orchestrator.yaml test_command (FALLBACK full-suite verify) "
        "references cockpit but is not presence-guarded with a '[ -d cockpit ]' "
        "test (task 2368) — a hard 'cd ../cockpit' would abort the whole "
        f"&&-chain while cockpit is absent from main; got: {cmd!r}"
    )
    # Matched directly against the raw command (not _pytest_segments' crude
    # &&-split, which would split this guarded subshell's own internal &&
    # into a non-pytest guard piece and a pytest piece that no longer
    # contains the word 'cockpit') so the presence guard and its pytest
    # invocation are verified as one associated unit.
    match = re.search(
        r"\[\s*-d\s+cockpit\s*\].*?cd\s+cockpit\s*&&\s*uv run pytest tests/\s*"
        r"--timeout[=\s](\d+)",
        cmd,
    )
    assert match, (
        "dark-factory-orchestrator.yaml test_command (FALLBACK full-suite verify) "
        "must run a pytest segment for cockpit's own tests/ suite guarded by "
        f"'[ -d cockpit ]' (task 2368); got: {cmd!r}"
    )
    timeout_value = int(match.group(1))
    assert timeout_value > 60, (
        f"cockpit's pytest segment sets --timeout={timeout_value}, which does "
        "not raise the ceiling above the flaky 60s pyproject default (task 2368)"
    )


def test_fallback_verify_raises_per_test_timeout() -> None:
    """Every pytest segment in the fallback chain must raise --timeout above 60s.

    Task 2361: orchestrator/fused-memory run pytest-xdist with a 60s
    pyproject.toml per-test timeout; under host-oversubscription an xdist
    worker starving past that ceiling gets os._exit()'d by pytest-timeout's
    thread method, and --max-worker-restart=0 turns that into a false-failing
    "node down" for whatever test happens to be running. A CLI --timeout
    (which overrides the pyproject default) above 60s removes the trigger.
    """
    cmd = _fleet_test_command()
    segments = _pytest_segments(cmd)
    assert segments, (
        "dark-factory-orchestrator.yaml test_command (FALLBACK full-suite verify) "
        f"has no pytest segments to check (task 2361); got: {cmd!r}"
    )
    for seg in segments:
        match = re.search(r"--timeout[=\s](\d+)", seg)
        assert match, (
            f"pytest segment {seg!r} in dark-factory-orchestrator.yaml test_command "
            "(FALLBACK full-suite verify) has no --timeout override (task "
            "2361) — the flaky 60s pyproject default is left in place, so "
            "xdist worker starvation under host oversubscription can still "
            "manufacture a false 'node down' failure"
        )
        timeout_value = int(match.group(1))
        assert timeout_value > 60, (
            f"pytest segment {seg!r} in dark-factory-orchestrator.yaml test_command "
            f"sets --timeout={timeout_value}, which does not raise the ceiling "
            "above the flaky 60s pyproject default (task 2361)"
        )


def test_per_module_merge_verify_raises_per_test_timeout() -> None:
    """Every per-module merge test_command's pytest segment must carry --timeout>=300.

    Task 2769 (PRD plans/cpu-load-robust-verify-prd.md, task beta): unlike
    the repo-root FALLBACK fleet chain checked above (which already sets
    --timeout=300), the per-module merge-verify orchestrator.yaml files
    (shared, escalation, orchestrator, fused-memory, dashboard, sampler,
    scripts) carried no --timeout override at all, so they silently
    inherited the flaky 60s pyproject.toml default. Under host CPU
    oversubscription a starved-but-correct xdist worker can cross that 60s
    wall-clock ceiling; pytest-timeout's thread method then os._exit()s the
    worker, and --max-worker-restart=0 degrades that into a false "node
    down" failure on whatever test happens to be running. Raising the
    per-test timeout to 300s (mirroring the fallback chain's own
    convention) removes the trigger for all but genuinely-hung tests. This
    guard fails CI if any per-module test_command's pytest segment omits
    --timeout, or sets it below 300, so a future edit can't silently
    regress back to the 60s default.

    The set of per-module configs is DISCOVERED dynamically (task 2769
    amendment) rather than hardcoded, so a future subproject that adds its
    own orchestrator.yaml + pytest test_command is auto-covered and cannot
    silently regress to the 60s default — the exact class of drift this
    guard exists to prevent. Task 3350 widened that discovery from an
    immediate-subdir glob to the production walk, so configs at ANY depth
    (e.g. ``tests/scripts``) are covered too. See
    ``TIMEOUT_GUARD_EXCLUSIONS`` for the single documented, out-of-scope
    carve-out.
    """
    discovered = _discover_per_module_configs()

    # Floor: the known configs must still be resolved by discovery. If any is
    # missing, discovery itself has silently broken and the loop below would
    # vacuously pass on a shrunken set — fail loudly instead.
    missing = KNOWN_PER_MODULE_CONFIG_NAMES - set(discovered)
    assert not missing, (
        "dynamic discovery (config._discover_module_configs, filtered to "
        f"configs defining a test_command) failed to resolve known per-module "
        f"config(s) {sorted(missing)} (task 2769) — discovery has regressed; "
        f"discovered: {sorted(discovered)}"
    )

    for prefix, module_config in sorted(discovered.items()):
        if prefix in TIMEOUT_GUARD_EXCLUSIONS:
            # Documented out-of-scope carve-out (see TIMEOUT_GUARD_EXCLUSIONS).
            continue
        cmd = module_config.test_command
        # Non-None by construction: _discover_per_module_configs filters on
        # ``mc.test_command``. Asserted so the narrowing is checkable rather
        # than implicit — ModuleConfig.test_command is typed ``str | None``.
        assert cmd is not None
        segments = _pytest_segments(cmd)
        if not segments:
            # A non-pytest subproject has no xdist worker to starve, so the
            # --timeout concern doesn't apply — skip it. The known configs are
            # contractually pytest-based, so an empty segment set there is a
            # real regression, not a legitimate non-pytest config.
            assert prefix not in KNOWN_PER_MODULE_CONFIG_NAMES, (
                f"{prefix}/orchestrator.yaml test_command (per-module merge "
                f"verify) has no pytest segments to check (task 2769); got: "
                f"{cmd!r}"
            )
            continue
        for seg in segments:
            match = re.search(r"--timeout[=\s](\d+)", seg)
            assert match, (
                f"pytest segment {seg!r} in {prefix}/orchestrator.yaml "
                "test_command (per-module merge verify) has no --timeout "
                "override (task 2769) — the flaky 60s pyproject default is "
                "left in place, so xdist worker starvation under host "
                "oversubscription can still manufacture a false 'node down' "
                "failure"
            )
            timeout_value = int(match.group(1))
            assert timeout_value >= 300, (
                f"pytest segment {seg!r} in {prefix}/orchestrator.yaml "
                f"test_command sets --timeout={timeout_value}, which is below "
                "the 300s floor (task 2769) mirroring the FALLBACK chain's own "
                "convention"
            )


#: Shell operators that end one `uv run` invocation and begin another. A
#: pairing is only self-defeating WITHIN a single clause, so the scan resets
#: at each of these rather than pooling flags across an `&&`-chain (every
#: subproject's lint_command chains a `python3 .../check_*.py` sibling gate).
_CHAIN_OPERATOR_TOKENS = frozenset({"&&", "||", ";", "|"})

#: The two spellings uv accepts for a value-taking long flag. The sibling
#: parser in ``test_skills_module_config_decision.py`` (``_pytest_collected_dirs``)
#: already handles ``--directory`` this way; mirrored here for BOTH flags so a
#: future ``--project=X --directory=X`` cannot slip past the guard.
_UV_VALUE_FLAGS = ("--project", "--directory")


def _uv_same_value_flag_pairings(cmd: str, *, label: str) -> list[str]:
    """Every value ``V`` for which one clause of *cmd* passes BOTH flags as ``V``.

    Returns the offending values (usually at most one per clause). A clause
    that sets only one of the two, or sets them to DIFFERENT values, is not
    reported: ``--project shared --directory scripts`` means "run from
    ``scripts/`` against ``shared``'s project", which is a genuinely different
    and legitimate instruction, not the self-defeating same-value pairing.
    """
    try:
        tokens = shlex.split(cmd)
    except ValueError as exc:
        raise AssertionError(
            f"{label} is not shell-parseable ({exc.__class__.__name__}: {exc}); "
            f"command: {cmd!r}.\nFIX: repair that config. Skipping an unparseable "
            f"command would silently defeat this guard — an unquoted config would "
            f"be indistinguishable from a clean one."
        ) from exc

    pairings: list[str] = []
    seen: dict[str, str] = {}

    def _end_clause() -> None:
        project, directory = seen.get("--project"), seen.get("--directory")
        if project is not None and project == directory:
            pairings.append(project)
        seen.clear()

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _CHAIN_OPERATOR_TOKENS:
            _end_clause()
        else:
            for flag in _UV_VALUE_FLAGS:
                if token == flag and index + 1 < len(tokens):
                    seen[flag] = tokens[index + 1]
                    index += 1
                    break
                if token.startswith(f"{flag}="):
                    seen[flag] = token[len(flag) + 1 :]
                    break
        index += 1
    _end_clause()
    return pairings


def test_per_module_verify_commands_never_pair_project_with_directory() -> None:
    """No module verify command may pass ``--project V`` and ``--directory V`` together.

    Task 3830. The pairing is SELF-DEFEATING, not merely redundant: uv applies
    ``--directory V`` FIRST (the process cwd becomes ``V/``) and only then
    resolves ``--project V`` — against that NEW cwd — so it looks for ``V/V``,
    which does not exist. uv 0.11.6 emits::

        warning: Project directory `shared` does not exist. This will become an
        error in a future release.

    It is a warning today and the command still runs, which is exactly what
    makes it worth guarding: nothing goes red until the uv release that
    promotes it to an error, at which point EVERY module verify command breaks
    simultaneously — and the repo root sets ``merge_verify_breadth: "full"``,
    so that is a repo-wide merge outage rather than one subproject's problem.

    The set of configs is DISCOVERED (``_discover_per_module_configs`` ->
    the production walk), so a newly-added subproject is auto-covered at any
    depth with no edit here. All three command slots are checked, because the
    defect is per-command, not per-module. No exclusions frozenset: task 3830
    fixed all 21 sites at once, leaving nothing to carve out.
    """
    discovered = _discover_per_module_configs()

    # Same discovery floor as the timeout guard above: if a known config stops
    # resolving, the loop below would vacuously pass on a shrunken set.
    missing = KNOWN_PER_MODULE_CONFIG_NAMES - set(discovered)
    assert not missing, (
        "dynamic discovery (config._discover_module_configs, filtered to "
        f"configs defining a test_command) failed to resolve known per-module "
        f"config(s) {sorted(missing)} (task 3830) — discovery has regressed; "
        f"discovered: {sorted(discovered)}"
    )

    offenders: list[str] = []
    for prefix, module_config in sorted(discovered.items()):
        for slot in ("test_command", "lint_command", "type_check_command"):
            cmd = getattr(module_config, slot)
            if cmd is None:
                continue
            label = f"{prefix}/orchestrator.yaml::{slot}"
            for value in _uv_same_value_flag_pairings(cmd, label=label):
                offenders.append(f"  {label} pairs --project {value} with --directory {value}")

    assert not offenders, (
        f"{len(offenders)} module verify command(s) pair `--project V` with "
        "`--directory V` for the same V (task 3830):\n"
        + "\n".join(offenders)
        + "\n\nWHY THIS IS WRONG: uv applies `--directory V` first (cwd becomes "
        "V/), then resolves `--project V` against that NEW cwd — looking for "
        "V/V, which does not exist. uv 0.11.6 warns 'This will become an error "
        "in a future release'; when it does, every one of these breaks at once, "
        "and the repo root's merge_verify_breadth: \"full\" makes that a "
        "repo-wide merge outage.\n\n"
        "FIX: delete the `--project V ` token and KEEP `--directory V`. The "
        "in-repo worked examples of an unpaired command are "
        "scripts/orchestrator.yaml and tests/scripts/orchestrator.yaml.\n\n"
        "Do NOT 'fix' this the other way by deleting `--directory` instead: cwd "
        "is load-bearing. pyright resolves `[tool.pyright]` from its cwd, and "
        "each module's own table carries a `venvPath`/`venv` interpreter pin "
        "that is resolved RELATIVE TO THAT FILE'S DIRECTORY (plus, for "
        "orchestrator, module-specific extraPaths). Moving cwd to the repo root "
        "silently substitutes the ROOT pyright config for the module's own — "
        "which type-checks green against the wrong settings, the false-GREEN "
        "direction (its own comment records 'measured 496 phantom "
        "reportMissingImports here, 0 with this pin')."
    )


# Task 4902 RE-MEASUREMENT of the `orchestrator` fleet segment, in seconds.
#
# WHY THIS EXISTS: the table below was frozen at a task-3062 single run from
# 2026-07-31. On 2026-08-20 commit 685f558728 landed
# `verify_admission_pytest_n: "8"`, capping orchestrator's xdist fanout, and
# the segment's cost stepped up ~2.6x. Nothing re-measured it, because nothing
# in this file CAN — see the SCOPE paragraph on
# ``test_fallback_verify_budget_clears_the_measured_fleet_chain_floor`` below.
# So it was re-measured by hand, once, and the result recorded here.
#
# CORPUS: `.worktrees/*/.task/verify/*.orchestrator.summary.json` — 246 files
# across 1354 worktrees at the time of the mine (2026-08-28, main 07eebf1c26),
# yielding 244 records carrying a `test`-label command with both a numeric
# `duration_secs` and a `started_at`.
#
# SELECTION (both filters are load-bearing):
#   FULL-SUITE only. Of the 244 test-leg records, 110 target the bare `tests/`
#     directory and 134 are file-scoped rescopes listing explicit
#     orchestrator/tests/*.py paths. A rescope's duration measures one diff's
#     blast radius, not a fleet segment; folding them in drags the median down
#     by an order of magnitude. The discriminator is the positional pytest
#     target: tokens after the last `pytest` that are non-flag, non-`k=v` and
#     non-numeric, kept only when that list is exactly ['tests/'].
#   GREEN only (rc == 0 and not timed_out). The table records honest green wall
#     clock: a timed-out run contributes its 3600s ceiling instead of its true
#     cost, and a red run may abort early — biases in opposite directions.
#
# REGIME SPLIT at started_at >= 2026-08-21T00:00:00Z (the day after 685f558728):
#   pre-cap  green: n=70, 2026-07-07..2026-08-20, min 218.62  p50  691.40  max 1783.83, 0 of 70 over 1800s
#   post-cap green: n=28, 2026-08-22..2026-08-28, min 864.83  p50 1765.95  max 3310.50, 14 of 28 over 1800s
# The 2.6x step in the median straddles the cap and is confirmed on the
# unfiltered arm too (all full-suite: before n=79 p50 698.04, after n=31
# p50 1803.40 with one 3600.65s timeout). This task does NOT reopen its
# attribution — tasks 3589 and 4456 own the -n cap and its memory ground.
#
# PERCENTILE CONVENTIONS, stated so the next re-measurement is a repeat rather
# than a re-derivation: p50 is ``statistics.median`` (n is even, so it is the
# mean of the two central observations and is NOT itself an observed run);
# p90 is the lower order statistic ``sorted[floor(0.90 * n) - 1]`` =
# ``sorted[24]`` of 28, which IS an observed run; max is the observed maximum.
POST_CAP_ORCHESTRATOR_GREEN_SECS = {
    'p50': 1765.95,
    'p90': 2552.09,
    'max': 3310.50,
}

# Sample size behind POST_CAP_ORCHESTRATOR_GREEN_SECS, kept next to it so the
# percentiles and the n that produced them cannot drift apart.
POST_CAP_ORCHESTRATOR_GREEN_N = 28

# Measured per-segment wall-clock of the FALLBACK fleet chain, in seconds.
#
# PROVENANCE — this table now spans TWO measurement epochs. Do not read it as
# one run.
#
#   shared / escalation / fused-memory / tests/scripts — task 3062,
#     .task/verify/attempt-2.__fallback__.{summary.json,test.log}; run started
#     2026-07-31T02:00:48Z under `nice -n 15 ionice -c2 -n7`; surfaced as
#     escalation esc-3062-3. One logged run each, except `tests/scripts`, which
#     uses the LOWEST of four independent measurements (105-127s).
#
#   orchestrator — RE-MEASURED by task 4902 on 2026-08-28 from the per-module
#     verify corpus (`.worktrees/*/.task/verify/*.orchestrator.summary.json`),
#     selecting FULL-SUITE GREEN runs only: n=28, window 2026-08-22..2026-08-28,
#     p50 1765.95 / p90 2552.09 / max 3310.50. The old figure (1366.23, one run)
#     predates commit 685f558728's `verify_admission_pytest_n: "8"` fanout cap
#     and understated this segment by ~400s. Full selection rules, both regime
#     arms and the percentile conventions are on
#     POST_CAP_ORCHESTRATOR_GREEN_SECS above.
#
# WHAT THE SUM IS, PRECISELY. `dashboard`, `sampler` and `cockpit` are OMITTED
# ENTIRELY — task 3062's run timed out at 1800.66s before dashboard even
# started, so no figure exists for them. So sum() is a lower bound on the MEDIAN
# green chain cost: five of eight segments, each a representative green run.
#
# It is NOT a bound on an individual run, and the earlier wording here ("the
# real green-path chain is strictly more expensive than this, never less") was
# wrong to imply otherwise — task 3062 itself logged 1366.23s and 1157.62s for
# the same segment on the same day, and the 4902 corpus shows green full-suite
# orchestrator runs spanning 864.83s to 3310.50s. Individual runs land on both
# sides of this sum; the median chain does not.
#
# FINDING (data for the pinned operator decision, task 3353's L1 — deliberately
# NOT acted on here). At the observed green MAXIMUM the five-segment floor is
# 472.37 + 3310.50 = 3782.87s, already above the 3600s
# `verify_command_timeout_secs`; adding the yaml's own ~407s of estimates for
# the three unmeasured segments puts that path at ~4190s. At the time of the
# mine one run had already consumed the full ceiling and been recorded as a
# false infra_timeout: 3600.649s, started 2026-08-28T17:25:05Z, observed at
# .worktrees/4023/.task/verify/attempt-1.orchestrator.summary.json.
#
# THE INLINED FIGURES ARE THE EVIDENCE; THE PATH IS NOT — the same caveat the
# repo-root yaml carries beside this finding. A `.task/verify/*.summary.json`
# is a TRANSIENT, per-attempt artifact, overwritten by the next attempt in its
# worktree and pruned with that worktree; the numbers are inlined because that
# is the only durable form the observation has, and the path says where it was
# read, not where it can be re-read. Re-checked 2026-08-30: that path had
# already been rewritten by a later, unrelated attempt (rc 1, timed_out FALSE,
# 2884.15s) while a different worktree showed the same ceiling hit (3605.06s)
# that day. Re-mine the corpus glob to re-establish the phenomenon; never
# re-read one path. This is also the reason the guards in this file compare
# recorded constants and read no corpus at test time.
#
# This task changes NO budget, NOT the -n cap, and NOT
# orchestrator/orchestrator.yaml — it only records the measurement those
# decisions need.
MEASURED_FLEET_SEGMENT_SECS = {
    'shared': 120.21,
    'escalation': 123.29,
    # Task 4902: 1366.23 -> 1765.95, the post-cap median. See PROVENANCE above.
    'orchestrator': 1765.95,
    'fused-memory': 123.87,
    'tests/scripts': 105.0,
}


class _SegmentProvenance(NamedTuple):
    """Where one MEASURED_FLEET_SEGMENT_SECS figure came from."""

    measured_at: str      # ISO YYYY-MM-DD
    sample_size: int      # number of runs actually observed
    task_id: str
    corpus: str


# Task 4902. The measurement table above now spans two epochs, and a bare
# {name: float} mapping cannot say which entry belongs to which. This records
# each figure's age and sample so the next reader — and the next re-measurement
# — can see at a glance what is being replaced, without archaeology.
#
# Read the sample sizes literally. Three of these are n=1: a single logged run
# each, which is exactly why a single later regime change (commit 685f558728's
# `pytest -n 8` cap) invalidated the table wholesale and nothing noticed for
# eight days. `tests/scripts` is n=4 because the comment above records it as
# the lowest of four independent measurements. Only `orchestrator` rests on a
# real sample, and only because task 4902 mined one.
#
# This makes the table DATED, not CURRENT. Nothing here re-measures anything —
# see the SCOPE paragraph on the floor guard below.
MEASURED_FLEET_SEGMENT_PROVENANCE: dict[str, _SegmentProvenance] = {
    'shared': _SegmentProvenance(
        '2026-07-31', 1, '3062', '.task/verify/attempt-2.__fallback__.summary.json'),
    'escalation': _SegmentProvenance(
        '2026-07-31', 1, '3062', '.task/verify/attempt-2.__fallback__.summary.json'),
    'orchestrator': _SegmentProvenance(
        # Spelled literally rather than as a reference to
        # POST_CAP_ORCHESTRATOR_GREEN_N so this record reads standalone. The
        # two are the same n by construction — a re-mine must move both.
        '2026-08-28', 28, '4902',
        '.worktrees/*/.task/verify/*.orchestrator.summary.json'),
    'fused-memory': _SegmentProvenance(
        '2026-07-31', 1, '3062', '.task/verify/attempt-2.__fallback__.summary.json'),
    'tests/scripts': _SegmentProvenance(
        '2026-07-31', 4, '3062', '.task/verify/attempt-2.__fallback__.summary.json'),
}


def _verify_budgets() -> dict:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding='utf-8'))


def test_every_measured_fleet_segment_carries_dated_provenance() -> None:
    """Every segment figure must have a provenance record, and vice versa.

    Task 4902. ``MEASURED_FLEET_SEGMENT_SECS`` and
    ``MEASURED_FLEET_SEGMENT_PROVENANCE`` are two structures describing one
    table, so they can drift: a segment added or removed in one and not the
    other leaves either an un-ageable figure or a provenance record for a
    segment that no longer exists. This guard is that referential integrity
    check and nothing more.

    SCOPE, stated because an earlier version of this file overreached. The
    dates, sample sizes and percentiles recorded beside the table are durable
    by virtue of being committed constants with comments; asserting a literal
    against another literal in the same file adds no regression detection over
    version control, so this file no longer does it. In particular there is no
    guard that the table is CURRENT — that would have to read the
    ``.task/verify/*.summary.json`` corpus at test time, which is pruned with
    its worktrees and would therefore be non-deterministic and eventually
    vacuous. Re-measurement is a human act (see the PROVENANCE comment above);
    this test proves only that the table's bookkeeping is self-consistent.
    """
    assert set(MEASURED_FLEET_SEGMENT_PROVENANCE) == set(MEASURED_FLEET_SEGMENT_SECS), (
        'MEASURED_FLEET_SEGMENT_PROVENANCE keys '
        f'{sorted(MEASURED_FLEET_SEGMENT_PROVENANCE)} do not match '
        f'MEASURED_FLEET_SEGMENT_SECS keys {sorted(MEASURED_FLEET_SEGMENT_SECS)} '
        f'(missing provenance: {sorted(set(MEASURED_FLEET_SEGMENT_SECS) - set(MEASURED_FLEET_SEGMENT_PROVENANCE))}; '
        f'orphaned provenance: {sorted(set(MEASURED_FLEET_SEGMENT_PROVENANCE) - set(MEASURED_FLEET_SEGMENT_SECS))}). '
        'Every segment figure must carry its date and sample, or the table goes '
        'back to being un-ageable — the defect task 4902 was filed to fix.'
    )


def test_fallback_verify_budget_clears_the_measured_fleet_chain_floor() -> None:
    """The warm per-command budget must exceed the MEASURED fleet-chain floor.

    Task 3350. ``verify_command_timeout_secs`` is a PER-COMMAND budget and the
    fleet chain is ONE shell command, so this single ceiling bounds all seven
    suites together. It was set to 1800s under the comment "Full warm verify
    here is ~2 min" — false by roughly an order of magnitude.

    A ceiling below a five-of-seven-segment measured floor cannot be cleared by
    a healthy run, so it does not surface hangs; it manufactures ``infra_timeout``
    on the honest green path. That is what task 3062 attempt-2 hit at 1800.66s.

    This asserts against the measured floor rather than pinning the chosen
    value, deliberately. Pinning a number would re-encode a constant with no
    stated basis — the exact failure mode of the "~2 min" comment this test
    exists to replace. A floor derived from logged per-suite durations cannot be
    wrong in the direction that matters: three segments are excluded, so it is
    provably a lower bound on the chain's real cost.

    SCOPE — what this guard does NOT do. It is a floor-REGRESSION guard: it
    fails if someone lowers ``verify_command_timeout_secs`` back below the
    measured floor (now 2238.32s). It is NOT a suite-growth detector, and
    nothing here re-measures anything. ``MEASURED_FLEET_SEGMENT_SECS`` is a
    frozen literal asserted against a config value.

    That limitation is no longer hypothetical. This paragraph used to warn: "if
    the orchestrator segment doubles to 2700s tomorrow, the table still reads
    1366.23, the floor still reads 1838.60, and this test passes green while the
    budget is once again provably below the honest green path." That is what
    happened. Commit 685f558728 (2026-08-20) capped the orchestrator fanout at
    ``pytest -n 8``; its median green full-suite cost went from 691.40s to
    1765.95s; the table stayed at 1366.23 and this test stayed green throughout,
    for eight days, until task 4902 re-measured the segment BY HAND on
    2026-08-28. The prediction was correct in kind and roughly correct in
    magnitude, and the guard did not fire.

    So the disclaimer stands, with one thing added and nothing softened.
    Genuine growth detection still has to come from RE-MEASUREMENT — an operator
    runbook step, or a check against durations recorded by a recent verify run —
    not from a hardcoded table asserting against itself, and 4902 built no such
    detector. What 4902 did add is age: every entry now carries dated provenance
    (``MEASURED_FLEET_SEGMENT_PROVENANCE``), so the next reader can see which
    epoch a figure came from without archaeology; and it removed the second
    copy of these figures rather than policing it, so there is no republished
    copy left to drift out of step — they live once, here, and
    dark-factory-orchestrator.yaml's budget comment now points at them instead
    of restating them. Neither makes the table measure itself. Stating that
    plainly is the point: task 3350 exists because a justification nobody
    re-checked was left standing until it was off by an order of magnitude,
    and a guard that overstates its own reach is the same defect wearing a
    test's clothes.
    """
    budgets = _verify_budgets()
    warm = budgets['verify_command_timeout_secs']
    floor = sum(MEASURED_FLEET_SEGMENT_SECS.values())

    assert warm > floor, (
        f'dark-factory-orchestrator.yaml verify_command_timeout_secs={warm} is '
        f'below the measured fleet-chain floor of {floor:.2f}s — short by '
        f'{floor - warm:.2f}s (task 3350). That floor sums only FIVE of seven '
        f'logged segments ({", ".join(sorted(MEASURED_FLEET_SEGMENT_SECS))}); '
        'dashboard, sampler and cockpit are excluded entirely because task 3062 '
        'attempt-2 timed out at 1800.66s before dashboard even started. A '
        'per-command ceiling below a five-of-seven floor surfaces no hangs — it '
        'manufactures infra_timeout on the honest green path. Raise the budget, '
        'or split the chain and re-measure this table.'
    )

    # Internal coherence: a cold run does strictly MORE work than a warm one —
    # the same chain PLUS verify_cold_preprovision_command (`uv sync
    # --all-packages && npm ci …`; task 4538 added the npm clause that installs
    # the pinned pyright the TYPE chain resolves) — so a warm ceiling above the
    # cold one is incoherent by construction, regardless of either value.
    cold = budgets['verify_cold_command_timeout_secs']
    assert warm <= cold, (
        f'verify_command_timeout_secs={warm} exceeds '
        f'verify_cold_command_timeout_secs={cold} (task 3350). A cold verify runs '
        'the same command chain plus the verify_cold_preprovision_command '
        'preprovision (uv sync + npm ci), so it is strictly more expensive; a '
        'warm budget above the cold one is incoherent by construction'
    )


def test_nested_module_configs_are_covered_by_the_per_test_timeout_guard() -> None:
    """Discovery must reach module configs at ANY depth, keyed by module prefix.

    Task 3350. ``_discover_per_module_configs`` globbed
    ``REPO_ROOT/*/orchestrator.yaml`` — ONE level only — while its own docstring
    promised "a newly-added subproject is auto-covered by the per-test timeout
    guard ... and cannot silently regress to the 60s default". A depth-2 config
    such as ``tests/scripts/orchestrator.yaml`` silently escaped it, so the
    guard's coverage claim was false exactly when a new config appeared
    somewhere the glob did not look — the drift class the dynamic discovery was
    introduced to prevent.

    It also keyed the discovered set on ``path.parent.name``, which collides
    ``scripts`` with ``tests/scripts``: two distinct module configs, one key.
    Assertion (b) pins that they stay DISTINCT, which is only possible if the
    set is keyed on the repo-relative module PREFIX rather than the bare
    directory name.
    """
    discovered = _discover_per_module_configs()

    # (a) The depth-2 config is reached at all.
    assert 'tests/scripts' in discovered, (
        "per-module config discovery did not find 'tests/scripts' (task 3350) — "
        'a config nested deeper than one level escapes the per-test timeout '
        'guard, so it can silently inherit the flaky 60s pyproject default while '
        'this test vacuously passes on a shrunken set. Discovered: '
        f'{sorted(discovered)}'
    )

    # (b) ... without colliding with the depth-1 'scripts' config. Both exist,
    # they are different files with different test_commands, and a name-keyed
    # set would silently drop one of them.
    assert 'scripts' in discovered, (
        "per-module config discovery lost the depth-1 'scripts' config while "
        f'gaining nested ones (task 3350). Discovered: {sorted(discovered)}'
    )
    assert discovered['scripts'] is not discovered['tests/scripts'], (
        "'scripts' and 'tests/scripts' resolved to the SAME module config (task "
        '3350) — the discovered set is keyed on the bare directory name, so the '
        'two collide and one is silently dropped from the timeout guard'
    )

    # (c) The known-7 floor still holds against the re-keyed set. Depth-1
    # prefixes are identical to their bare names, so KNOWN_PER_MODULE_CONFIG_NAMES
    # needed only the new 'tests/scripts' entry.
    missing = KNOWN_PER_MODULE_CONFIG_NAMES - set(discovered)
    assert not missing, (
        f'discovery failed to resolve known per-module config(s) {sorted(missing)} '
        f'(task 3350); discovered: {sorted(discovered)}'
    )


def _pyright_clause_cwds(cmd: str) -> list[str]:
    """Return, in order, the normalised cwd of each bare-pyright clause in *cmd*.

    Walks the ``&&``-chain tracking cwd through ``cd <dir>`` clauses, using the
    same PRODUCTION helpers ``verify._AND_CLAUSE_SPLIT_RE`` /
    ``verify._cd_clause_target`` that ``verify._scope_fallback_tool_to_subproject``
    (task 3022) itself uses to read this exact command — so this helper cannot
    drift from how the scoper interprets the chain.

    A "bare" pyright clause mentions ``pyright`` and is not already wrapped in
    ``uv run --project`` (interpreter-pinned by uv itself, not by
    ``[tool.pyright]``, so it is excluded from the result).

    Extracted (task 3397) from what was originally inlined in
    ``TestRootTypeCheckCommandPyrightInterpreterPinned``'s own test method
    (task 3367), so that test and the fleet TYPE-chain coverage invariant
    below walk the chain identically and cannot drift apart — the same "must
    not drift apart" convention ``_assert_pyright_pins_worktree_venv`` already
    states.
    """
    parts = _AND_CLAUSE_SPLIT_RE.split(cmd)
    cwd = "."
    cwds: list[str] = []
    for i in range(0, len(parts), 2):
        clause = parts[i]
        cd_target = _cd_clause_target(clause)
        if cd_target is not None:
            cwd = os.path.normpath(os.path.join(cwd, cd_target))
            continue
        if "pyright" not in clause:
            continue
        if "uv run --project" in clause:
            # Already interpreter-pinned, by uv rather than by [tool.pyright]:
            # `uv run --project <sub>` selects the workspace venv itself.
            continue
        cwds.append(cwd)
    return cwds


class TestRootTypeCheckCommandPyrightInterpreterPinned:
    """Every bare-``pyright`` clause of the fleet chain must be interpreter-pinned.

    Task 3367 / esc-3359-1 — the DOCS-ONLY-DIFF guard.

    A docs-only diff contains zero ``.py`` files, so ``verify._build_fallback_config``
    returns ``None`` and the RAW ``type_check_command`` from
    ``dark-factory-orchestrator.yaml`` runs verbatim — none of the
    ``_scope_fallback_tool_to_subproject`` rescoping (tasks 2355/3022) applies.
    Its outcome then depends on exactly one property: whether every directory the
    chain ``cd``s into pins pyright's interpreter at that worktree's own ``.venv``.

    If a clause does not, pyright falls back to ambient VIRTUAL_ENV/PATH resolution
    — and ``verify._target_subprocess_env`` deliberately strips both — so a cold
    merge worktree type-checks against an interpreter holding none of the
    workspace's third-party packages. Measured under exactly that scrubbed env:
    fused-memory 514 errors, orchestrator 496, all phantom reportMissingImports;
    0 in both once the pin is present.
    """

    def test_every_bare_pyright_clause_runs_in_an_interpreter_pinned_dir(self) -> None:
        cmd = _fleet_type_check_command()
        checked = _pyright_clause_cwds(cmd)

        for cwd in checked:
            pyproject = _pyproject_at(cwd)
            pyright = pyproject.get("tool", {}).get("pyright")
            assert pyright is not None, (
                f"fleet type_check_command runs a bare pyright clause in {cwd!r}, "
                "whose pyproject.toml has no [tool.pyright] table at all (task "
                "3367, esc-3359-1) — so pyright resolves its interpreter from "
                "ambient VIRTUAL_ENV/PATH, which verify._target_subprocess_env "
                "strips"
            )
            _assert_pyright_pins_worktree_venv(
                cwd,
                pyright,
                why=(
                    "It is the cwd of a bare-pyright clause in fleet "
                    "type_check_command, which a DOCS-ONLY diff runs verbatim "
                    "(no .py files -> _build_fallback_config returns None -> no "
                    "rescoping)."
                ),
            )

        # Non-vacuity: if the chain's shape changes such that no bare-pyright
        # clause is found, this guard must fail loudly rather than pass on an
        # empty set.
        assert checked, (
            "no bare (non-`uv run --project`) pyright clause was found in "
            f"dark-factory-orchestrator.yaml type_check_command (task 3367, "
            f"esc-3359-1) — this interpreter-pin guard would pass vacuously. "
            f"Either the chain no longer runs pyright, or its shape changed such "
            f"that the &&-clause walk no longer resolves its cwds; got: {cmd!r}"
        )


# Floor for the workspace-wide interpreter-pin invariant below: proof that
# runtime discovery from the root pyproject's ``[tool.uv.workspace].members``
# still resolves the members that actually carry pyright configs. NOT the
# authoritative list — same convention as KNOWN_PER_MODULE_CONFIG_NAMES above,
# so a newly-added workspace member is auto-covered with no edit here.
#
# Task 3397: extended from 4 to all 7 members — shared, escalation,
# orchestrator, fused-memory (pre-existing) plus dashboard, sampler, cockpit,
# which the task-3397 fan-out confirmed all declare a
# ``[tool.pyright] venvPath=".." venv=".venv"`` table too.
KNOWN_PYRIGHT_PINNED_MEMBERS = frozenset(
    {
        "shared",
        "escalation",
        "orchestrator",
        "fused-memory",
        "dashboard",
        "sampler",
        "cockpit",
    }
)


class TestWorkspacePyrightInterpreterPinned:
    """EVERY uv-workspace member declaring ``[tool.pyright]`` must pin the worktree venv.

    Task 3367 / esc-3359-1 — the generalisation of the fleet-chain guard above.

    Before task 3397, ``TestRootTypeCheckCommandPyrightInterpreterPinned``
    covered only the 3 directories ``type_check_command`` happened to ``cd``
    into, leaving the hole one config edit away from reopening: ``shared`` and
    ``escalation`` were type-checked only through their own
    ``<sub>/orchestrator.yaml`` ``uv run --directory X pyright``
    commands, where uv (not ``[tool.pyright]``) supplies the interpreter, and
    ``sampler``/``cockpit``/``dashboard`` weren't in the fleet chain at all.
    Task 3397 extended the fleet chain to all 7 workspace members, so today
    both guards cover the same set in practice.

    This class survives that as more than a near-duplicate: it is
    CHAIN-INDEPENDENT. It checks every workspace member that declares a
    ``[tool.pyright]`` table AT ALL — via the fleet chain, a per-module
    ``orchestrator.yaml``, or a developer running ``npx pyright`` directly in
    that directory — rather than deriving coverage from walking one
    particular command string. A future rewrite of ``type_check_command``
    (e.g. back down to a subset, or into a different shape entirely) would
    silently narrow ``TestRootTypeCheckCommandPyrightInterpreterPinned``'s
    coverage without this class noticing, because this class does not read
    ``type_check_command`` at all.

    Members are DISCOVERED at runtime from the root ``pyproject.toml``'s
    ``[tool.uv.workspace].members``, so a newly-added subproject is covered on
    day one rather than escaping a hardcoded list.
    """

    def _workspace_member_dirs(self) -> list[str]:
        return _workspace_member_dirs()

    def test_every_workspace_member_pyright_config_pins_the_worktree_venv(self) -> None:
        # The root pyproject is checked too: it is the mirror the members' ".."
        # pins resolve to, spelled ``venvPath = "."`` from its own directory.
        checked: set[str] = set()
        for rel_dir in [".", *self._workspace_member_dirs()]:
            # cockpit is presence-guarded elsewhere in this file (it landed on
            # main after some guards were scoped); skip any member directory that
            # is genuinely absent rather than failing on a stale members list.
            if not (REPO_ROOT / rel_dir / "pyproject.toml").is_file():
                continue
            pyright = _pyproject_at(rel_dir).get("tool", {}).get("pyright")
            if pyright is None:
                # A member that never runs pyright has no interpreter to pin.
                continue
            _assert_pyright_pins_worktree_venv(
                rel_dir,
                pyright,
                why=(
                    f"{rel_dir!r} is a uv-workspace member that declares a "
                    "[tool.pyright] table, so pyright can be invoked there — by "
                    "the fleet chain, by its own orchestrator.yaml, or by a "
                    "developer — and every such invocation must resolve THIS "
                    "worktree's own .venv."
                ),
            )
            checked.add(rel_dir)

        assert checked, (
            "no workspace member with a [tool.pyright] table was discovered "
            "(task 3367, esc-3359-1) — this invariant would pass vacuously"
        )
        missing = KNOWN_PYRIGHT_PINNED_MEMBERS - checked
        assert not missing, (
            f"runtime discovery from [tool.uv.workspace].members failed to resolve "
            f"known pyright-configured member(s) {sorted(missing)} (task 3367); "
            f"checked: {sorted(checked)}. Either a member was dropped from the "
            f"workspace, or its [tool.pyright] table was removed — both need an "
            f"explicit decision, not a silently shrinking guard"
        )


# Floor for the fleet-chain coverage invariants below (TYPE here; the LINT
# coverage invariant reuses the same frozenset — task 3397, named for what it
# is: the workspace members every fleet chain must cover, not just TYPE's).
# NOT the authoritative list.
#
# UNLIKE KNOWN_PER_MODULE_CONFIG_NAMES / KNOWN_PYRIGHT_PINNED_MEMBERS above,
# this floor is NOT proof that runtime discovery from
# ``[tool.uv.workspace].members`` still resolves these members: the sets it
# is subtracted from below (``walked`` / ``targets``) are parsed from the
# CONFIG COMMAND strings themselves (``type_check_command`` /
# ``lint_command``), not from ``_workspace_member_dirs()``. Removing an entry
# from the root pyproject's members list would therefore NOT fail either
# guard below on its own. What this floor DOES catch: the fleet TYPE/LINT
# chain STRINGS silently shrinking — a ``cd``/``npx pyright`` pair, or a
# ruff/magicmock target, dropped from the yaml. A newly-added workspace
# member is still auto-covered with no edit here, because the per-member
# loops just above each of these assertions discover members from
# ``_workspace_member_dirs()`` at runtime.
KNOWN_FLEET_MEMBERS = frozenset(
    {"cockpit", "dashboard", "escalation", "fused-memory", "orchestrator", "sampler", "shared"}
)


# Measured per-member wall-clock of the fleet TYPE chain
# (dark-factory-orchestrator.yaml type_check_command), in seconds.
#
# PROVENANCE: task 3397, measured 2026-08-02 in a synced worktree (`env -u
# VIRTUAL_ENV uv sync --all-packages`), each member run standalone as
# `env -u VIRTUAL_ENV bash -c "cd <member> && npx pyright"` — mirrors what
# verify._target_subprocess_env strips. All seven: exit 0, "0 errors". The
# full 7-clause chain also ran end-to-end as one command: exit 0, 576s
# wall-clock — a dated snapshot, not a self-maintaining invariant (same
# honesty as MEASURED_FLEET_SEGMENT_SECS above).
MEASURED_FLEET_TYPE_SEGMENT_SECS = {
    "fused-memory": 113,
    "orchestrator": 220,
    "dashboard": 51,
    "shared": 64,
    "escalation": 42,
    "sampler": 11,
    "cockpit": 17,
}


class TestFleetTypeCheckCoversEveryWorkspaceMember:
    """The fleet TYPE chain must ``cd`` into and pyright-check every workspace member.

    Task 3397. ``type_check_command`` is, like ``test_command`` above, the
    FALLBACK chain: ``verify._build_fallback_config`` returns ``None`` for a
    zero-``.py``-file diff, so a docs-only or cross-cutting diff runs this
    chain verbatim with no ``_scope_fallback_tool_to_subproject`` rescoping.
    Before task 3397 the chain covered only 3 of 7
    ``[tool.uv.workspace].members`` (fused-memory, orchestrator, dashboard),
    so shared, escalation, sampler and cockpit were never type-checked at
    this gating layer — the same defect class task 2361/2368 closed for
    ``test_command``.
    """

    def test_every_present_workspace_member_is_type_checked(self) -> None:
        walked = set(_pyright_clause_cwds(_fleet_type_check_command()))

        for member in _workspace_member_dirs():
            if not (REPO_ROOT / member / "pyproject.toml").is_file():
                # Mirrors the presence tolerance in
                # TestWorkspacePyrightInterpreterPinned: a member genuinely
                # absent from this checkout is skipped rather than failed.
                continue
            assert member in walked, (
                f"fleet type_check_command does not cd into and pyright-check "
                f"workspace member {member!r} (task 3397) — a docs-only or "
                "cross-cutting diff (zero .py files touched) runs this chain "
                f"verbatim with no rescoping, so {member!r} would never be "
                f"type-checked at the gating layer; walked: {sorted(walked)}"
            )

        assert walked, (
            "the fleet type_check_command &&-walk resolved no cwds at all "
            "(task 3397) — this coverage invariant would pass vacuously"
        )
        missing = KNOWN_FLEET_MEMBERS - walked
        assert not missing, (
            f"fleet type_check_command is missing known workspace member(s) "
            f"{sorted(missing)} (task 3397) — either a member was dropped from "
            "the chain, or it was rewritten into a form the &&-walk cannot "
            "follow (e.g. a subshell-guarded clause, which "
            "verify._scope_fallback_tool_to_subproject's own cwd tracker also "
            f"cannot follow); walked: {sorted(walked)}"
        )

    def test_measured_type_chain_floor_clears_the_verify_budget(self) -> None:
        """The warm per-command budget must exceed the MEASURED TYPE-chain floor.

        Task 3397. ``verify_command_timeout_secs`` is a PER-COMMAND budget, and
        TEST/LINT/TYPE are three SEPARATE commands dispatched concurrently in
        one ``asyncio.gather`` (verify.py:3745-3768, 4201-4207) — each bounded
        by its own copy of the same ceiling. Extending the TYPE chain from 3 to
        7 members therefore adds nothing to the TEST floor asserted by
        ``test_fallback_verify_budget_clears_the_measured_fleet_chain_floor``
        above; this is TYPE's own equivalent guard, over its own table.

        SCOPE — what this guard does NOT do, mirroring that same test's own
        scope note. It is a floor-REGRESSION guard: it fails if someone lowers
        ``verify_command_timeout_secs`` back below the measured 518s lower
        bound. It is NOT a suite-growth detector, and nothing here re-measures
        anything. ``MEASURED_FLEET_TYPE_SEGMENT_SECS`` is a frozen literal
        asserted against a config value; if a member's pyright run doubles
        tomorrow, the table still reads its 2026-08-02 figure, the floor still
        reads 518, and this test stays green while the budget is once again
        unmeasured against the honest green path. Genuine growth detection
        would have to come from RE-MEASUREMENT, not from a hardcoded table
        asserting against itself.
        """
        budgets = _verify_budgets()
        warm = budgets["verify_command_timeout_secs"]
        cold = budgets["verify_cold_command_timeout_secs"]
        floor = sum(MEASURED_FLEET_TYPE_SEGMENT_SECS.values())

        assert warm > floor, (
            f"dark-factory-orchestrator.yaml verify_command_timeout_secs={warm} "
            f"is below the measured fleet TYPE-chain floor of {floor}s (task "
            "3397) — a per-command ceiling below the honest green path "
            "manufactures infra_timeout rather than surfacing hangs. Raise the "
            "budget, or re-measure this table."
        )
        # Internal coherence, mirrors test_fallback_verify_budget_clears_the_
        # measured_fleet_chain_floor's own cold/warm assertion: a cold verify
        # does strictly MORE work than a warm one, so a warm ceiling above the
        # cold one is incoherent regardless of what either value is.
        assert warm <= cold, (
            f"verify_command_timeout_secs={warm} exceeds "
            f"verify_cold_command_timeout_secs={cold} (task 3397). A cold "
            "verify runs the same chains plus the "
            "verify_cold_preprovision_command preprovision (uv sync + npm ci), "
            "so it is strictly more expensive; a warm budget above the cold one "
            "is incoherent by construction"
        )

    def test_type_chain_table_matches_the_chain_it_measures(self) -> None:
        """MEASURED_FLEET_TYPE_SEGMENT_SECS must describe exactly the shipped chain.

        Task 3397. Guards against the table silently drifting from the actual
        ``type_check_command`` — a member added to the chain without adding its
        measured seconds would silently UNDER-count the floor asserted by
        ``test_measured_type_chain_floor_clears_the_verify_budget`` above; a
        member removed from the chain while its stale figure lingers would
        silently OVER-count it.
        """
        walked = set(_pyright_clause_cwds(_fleet_type_check_command()))
        assert set(MEASURED_FLEET_TYPE_SEGMENT_SECS) == walked, (
            f"MEASURED_FLEET_TYPE_SEGMENT_SECS keys "
            f"{sorted(MEASURED_FLEET_TYPE_SEGMENT_SECS)} do not match the fleet "
            f"type_check_command chain it is meant to cost (task 3397); walked: "
            f"{sorted(walked)}. Update the table to match whenever the chain "
            "changes."
        )


def _lint_leg_targets(cmd: str, marker: str) -> list[str]:
    """Return the positional targets of *cmd*'s ``&&``-leg identified by *marker*.

    *cmd* is the fleet ``lint_command``, an ``&&``-chain of two legs: a ``ruff
    check <targets...>`` leg and a ``check_bare_magicmock_config.py
    <targets...>`` sibling-checker leg. *marker* selects which leg — pass
    ``"ruff check"`` or ``"check_bare_magicmock_config.py"`` — by substring
    after a plain ``&&`` split (unlike the TYPE chain, this command has no
    ``cd`` clauses to walk: every target is an explicit repo-root-relative
    path, so the production ``_AND_CLAUSE_SPLIT_RE``/``_cd_clause_target``
    cwd-tracking walk does not apply here).

    Returns only the tokens AFTER *marker* itself (``shlex.split(marker)``
    located as a contiguous window in the leg's own ``shlex.split`` tokens,
    matching the last window token by suffix so a marker like
    ``"check_bare_magicmock_config.py"`` still matches the full invoked path
    ``fused-memory/scripts/check_bare_magicmock_config.py``) — NOT
    ``shlex.split(leg)`` over the whole leg. The whole-leg split always
    contains the command's own tokens (``uv``, ``run``, ``ruff``, ``check`` /
    ``python3``, ``<script>.py``), so an ``assert targets`` non-vacuity guard
    over it can never fire empty even if every positional target were
    deleted; trimming to the tail after *marker* keeps that guard live.

    Callers must compare whole path TOKENS (as returned here) against member
    names, never substring-match the raw command — ``"shared" in cmd`` is
    already true via the OTHER leg's ``shared/tests`` argument, so it would
    pass vacuously for a member a given leg never actually checks.
    """
    marker_tokens = shlex.split(marker)
    n = len(marker_tokens)
    for leg in cmd.split("&&"):
        if marker not in leg:
            continue
        tokens = shlex.split(leg)
        for i in range(len(tokens) - n + 1):
            window = tokens[i : i + n]
            if window == marker_tokens or (
                n == 1 and window[0].endswith(marker_tokens[0])
            ):
                return tokens[i + n :]
        return []
    return []


class TestFleetLintCoversEveryWorkspaceMember:
    """The fleet LINT chain must ruff-check every workspace member.

    Task 3397. ``lint_command`` is, like ``test_command``/``type_check_command``
    above, the FALLBACK chain: ``verify._build_fallback_config`` returns
    ``None`` for a zero-``.py``-file diff, so a docs-only or cross-cutting
    diff runs this chain verbatim with no
    ``_scope_fallback_tool_to_subproject`` rescoping. Before task 3397 the
    ``ruff check`` leg covered only 5 of 7 ``[tool.uv.workspace].members``
    (sampler and cockpit were absent), so a docs-only or cross-cutting diff
    never ruff-checked either at the gating layer.
    """

    def test_every_present_workspace_member_is_ruff_checked(self) -> None:
        cmd = _fleet_lint_command()
        targets = _lint_leg_targets(cmd, "ruff check")

        for member in _workspace_member_dirs():
            if not (REPO_ROOT / member / "pyproject.toml").is_file():
                # Mirrors the presence tolerance used throughout this file: a
                # member genuinely absent from this checkout is skipped
                # rather than failed.
                continue
            assert member in targets, (
                f"fleet lint_command's ruff-check leg does not target "
                f"workspace member {member!r} (task 3397) — a docs-only or "
                "cross-cutting diff (zero .py files touched) runs this chain "
                f"verbatim with no rescoping, so {member!r} would never be "
                f"ruff-checked at the gating layer; targets: {targets}"
            )

        assert targets, (
            "the fleet lint_command's ruff-check leg had no positional "
            "targets at all (task 3397) — this coverage invariant would pass "
            "vacuously"
        )
        missing = KNOWN_FLEET_MEMBERS - set(targets)
        assert not missing, (
            f"fleet lint_command's ruff-check leg is missing known workspace "
            f"member(s) {sorted(missing)} (task 3397); targets: {targets}"
        )

        # A typo'd or stale target (e.g. "sampler/test", "cocpit") is invisible
        # to the two assertions above — they only catch a KNOWN member being
        # MISSING, not a bogus EXTRA one — yet it would make `ruff check` exit
        # non-zero on every fallback/merge-queue verify. Catch it here instead
        # of at the gating layer.
        for target in targets:
            assert (REPO_ROOT / target).exists(), (
                f"fleet lint_command's ruff-check leg names {target!r}, which "
                f"does not exist under {REPO_ROOT} (task 3397) — this would "
                "make `ruff check` exit non-zero on every fallback/merge-queue "
                f"verify; targets: {targets}"
            )

    def test_every_present_workspace_member_tests_dir_is_magicmock_checked(self) -> None:
        cmd = _fleet_lint_command()
        targets = _lint_leg_targets(cmd, "check_bare_magicmock_config.py")

        for member in _workspace_member_dirs():
            if not (REPO_ROOT / member / "tests").is_dir():
                # check_bare_magicmock_config.py takes directories; naming one
                # that doesn't exist would make this leg exit non-zero, so a
                # member with no tests/ dir is skipped rather than failed.
                continue
            tests_dir = f"{member}/tests"
            assert tests_dir in targets, (
                f"fleet lint_command's check_bare_magicmock_config.py leg "
                f"does not target {tests_dir!r} (task 3397) — a docs-only or "
                "cross-cutting diff (zero .py files touched) runs this chain "
                f"verbatim with no rescoping, so {tests_dir!r} would never be "
                f"checked for bare MagicMock config at the gating layer; "
                f"targets: {targets}"
            )

        assert targets, (
            "the fleet lint_command's check_bare_magicmock_config.py leg had "
            "no positional targets at all (task 3397) — this coverage "
            "invariant would pass vacuously"
        )

        # Same typo-blind-spot rationale as the ruff-leg assertion above: a
        # bogus extra target (not merely a missing known one) would make this
        # leg exit non-zero on every fallback/merge-queue verify.
        for target in targets:
            assert (REPO_ROOT / target).exists(), (
                f"fleet lint_command's check_bare_magicmock_config.py leg "
                f"names {target!r}, which does not exist under {REPO_ROOT} "
                "(task 3397) — this would make the script exit non-zero on "
                f"every fallback/merge-queue verify; targets: {targets}"
            )


# ---------------------------------------------------------------------------
# Pyright scope parity: members importing a scripts/-only module (task 3931)
# ---------------------------------------------------------------------------

# Floor for the discovery below. NOT the authoritative list — the importing
# members are DISCOVERED at runtime by scanning member sources, so a NEW
# member that starts importing a ``scripts/``-only module is covered on day
# one. This floor only proves the scan still resolves the member we know
# imports one (``orchestrator/tests/test_run_vllm_eval.py`` imports
# ``run_vllm_eval``), so a scan that silently stops matching anything fails
# loudly instead of passing vacuously.
KNOWN_SCRIPTS_IMPORTING_MEMBERS = frozenset({"orchestrator"})


def _scripts_only_module_stems() -> set[str]:
    """Top-level module names importable ONLY from repo-root ``scripts/``.

    ``scripts/*.py`` stems, minus (a) non-identifier stems (``wait-for-port``
    et al. are runnable files, never importable modules), (b) stdlib names,
    and (c) any stem that is ALSO a top-level module/package under a workspace
    member's ``src/`` or at the repo root — those resolve for pyright through
    an existing ``extraPaths`` entry and say nothing about ``scripts/``.
    """
    stems = {p.stem for p in (REPO_ROOT / "scripts").glob("*.py") if p.stem.isidentifier()}
    resolvable_elsewhere: set[str] = {p.stem for p in REPO_ROOT.glob("*.py")}
    for member in _workspace_member_dirs():
        src = REPO_ROOT / member / "src"
        if not src.is_dir():
            continue
        resolvable_elsewhere |= {
            child.stem if child.suffix == ".py" else child.name for child in src.iterdir()
        }
    return stems - set(sys.stdlib_module_names) - resolvable_elsewhere


def _members_importing_scripts_only_modules() -> dict[str, set[str]]:
    """Map each workspace member to the ``scripts/``-only modules its own files import."""
    stems = _scripts_only_module_stems()
    assert stems, (
        "no repo-root scripts/ module is importable-only-from-scripts (task "
        "3931) — this scope-parity invariant would pass vacuously"
    )
    pattern = re.compile(
        r"^[ \t]*(?:import|from)[ \t]+(" + "|".join(sorted(map(re.escape, stems))) + r")\b",
        re.MULTILINE,
    )
    found: dict[str, set[str]] = {}
    for member in _workspace_member_dirs():
        member_dir = REPO_ROOT / member
        if not member_dir.is_dir():
            # Presence tolerance, per TestWorkspacePyrightInterpreterPinned: a
            # member genuinely absent from this checkout is skipped, not failed.
            continue
        for path in member_dir.rglob("*.py"):
            if ".venv" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for match in pattern.finditer(text):
                found.setdefault(member, set()).add(match.group(1))
    return found


def _assert_pyright_resolves_scripts(rel_dir: str, pyright: dict, why: str) -> None:
    """Assert *rel_dir*'s ``[tool.pyright] extraPaths`` resolves repo-root ``scripts/``.

    Modelled on ``_assert_pyright_pins_worktree_venv`` above (task 3367): one
    shared assertion carrying the caller's context through ``why=``, so the
    per-member invariant here and the module-local guard in
    ``orchestrator/tests/test_run_vllm_eval.py`` state the same property and
    cannot drift apart.

    Asserted by RESOLUTION, never by string equality: ``../scripts`` and any
    other spelling landing on the same directory both pass.
    """
    extra_paths = pyright.get("extraPaths")
    assert extra_paths, (
        f"{rel_dir}/pyproject.toml [tool.pyright] declares no extraPaths (task "
        f"3931). {why}"
    )
    scripts_dir = (REPO_ROOT / "scripts").resolve()
    resolved = [(REPO_ROOT / rel_dir / entry).resolve() for entry in extra_paths]
    assert scripts_dir in resolved, (
        f"{rel_dir}/pyproject.toml [tool.pyright] extraPaths {list(extra_paths)!r} "
        f"contains no entry resolving to {scripts_dir} (task 3931, esc-3805-1 "
        f"2026-08-09 / esc-3805-6 2026-08-12). {why} The root pyproject.toml's "
        "extraPaths DOES list 'scripts', so ROOT-scoped pyright resolves the "
        "import and reports real errors that PACKAGE-scoped pyright cannot see "
        "— MEASURED at 1.1.408 with hotfix 27ac22a6a6 reverse-applied: 14 "
        "reportArgumentType errors root-scoped, 0 package-scoped. verify's "
        "FILE_SCOPED fallback runs pyright from the worktree ROOT while "
        "pre-commit (hooks/project-checks) and the fleet chain run it "
        "PACKAGE-scoped, so without this entry the two gates disagree about "
        f"whether the same file type-checks; resolved: {[str(p) for p in resolved]}"
    )


class TestMembersImportingScriptsResolveScriptsOnTheirPyrightPath:
    """A member importing a ``scripts/``-only module must resolve ``scripts/`` itself.

    Task 3931 — the CHAIN-INDEPENDENT generalisation of the module-local guard
    in ``orchestrator/tests/test_run_vllm_eval.py``, in the same spirit as
    ``TestWorkspacePyrightInterpreterPinned`` generalising the fleet-chain
    interpreter pin: that guard names one module in one package, this one holds
    for every workspace member, discovered at runtime.

    The property: if a member's OWN sources import a top-level module that
    exists only under repo-root ``scripts/``, then package-scoped pyright must
    be able to resolve it — otherwise the imported names degrade to ``Unknown``
    and every defect involving them goes unreported in that scope, while the
    root-scoped verify gate (whose config lists ``scripts``) still reports
    them. That asymmetry IS esc-3805-1/esc-3805-6.

    Presence-tolerant and MEMBERSHIP-only (never list equality — the rule stated at
    tests/scripts/test_scripts_module_config.py::test_root_pyright_extrapaths_resolves_scripts_imports),
    so an unrelated extraPaths addition does not false-red this guard.
    """

    def test_members_importing_scripts_modules_resolve_scripts(self) -> None:
        importing = _members_importing_scripts_only_modules()
        assert importing, (
            "no workspace member was found importing a repo-root scripts/-only "
            "module (task 3931) — either the scan stopped matching (an import "
            "spelling it cannot see) or the import genuinely went away; both "
            "need an explicit decision, not a silently vacuous guard"
        )
        missing = KNOWN_SCRIPTS_IMPORTING_MEMBERS - set(importing)
        assert not missing, (
            f"runtime discovery failed to resolve known scripts/-importing "
            f"member(s) {sorted(missing)} (task 3931); discovered: "
            f"{ {k: sorted(v) for k, v in importing.items()} }"
        )

        for member, modules in sorted(importing.items()):
            if not (REPO_ROOT / member / "pyproject.toml").is_file():
                continue
            pyright = _pyproject_at(member).get("tool", {}).get("pyright")
            if pyright is None:
                # A member that never runs pyright has no search path to widen.
                continue
            _assert_pyright_resolves_scripts(
                member,
                pyright,
                why=(
                    f"{member!r} imports {sorted(modules)!r}, which exist(s) "
                    "ONLY under repo-root scripts/, so package-scoped pyright "
                    "can resolve the import only via an extraPaths entry "
                    "pointing there."
                ),
            )
