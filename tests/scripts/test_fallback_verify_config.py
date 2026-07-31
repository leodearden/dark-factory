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

import pathlib
import re

import yaml

from orchestrator.config import ModuleConfig, _discover_module_configs

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


# Measured per-segment wall-clock of the FALLBACK fleet chain, in seconds.
#
# PROVENANCE: task 3062, .task/verify/attempt-2.__fallback__.{summary.json,
# test.log}; run started 2026-07-31T02:00:48Z under `nice -n 15 ionice -c2 -n7`;
# surfaced as escalation esc-3062-3. These are LOGGED durations, not estimates.
#
# `tests/scripts` uses the LOWEST of four independent measurements (105-127s),
# and `dashboard`, `sampler` and `cockpit` are OMITTED ENTIRELY — the run timed
# out at 1800.66s before dashboard even started, so no figure exists for them.
# The sum is therefore a hard measured LOWER BOUND on the chain's cost: the real
# green-path chain is strictly more expensive than this, never less.
MEASURED_FLEET_SEGMENT_SECS = {
    'shared': 120.21,
    'escalation': 123.29,
    'orchestrator': 1366.23,
    'fused-memory': 123.87,
    'tests/scripts': 105.0,
}


def _verify_budgets() -> dict:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding='utf-8'))


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
    measured 1838.60s lower bound. It is NOT a suite-growth detector, and
    nothing here re-measures anything. ``MEASURED_FLEET_SEGMENT_SECS`` is a
    frozen literal asserted against a config value; if the orchestrator segment
    doubles to 2700s tomorrow, the table still reads 1366.23, the floor still
    reads 1838.60, and this test passes green while the budget is once again
    provably below the honest green path. Genuine growth detection would have to
    come from RE-MEASUREMENT — an operator runbook step, or a check against
    durations recorded by a recent verify run — not from a hardcoded table
    asserting against itself. Stating that plainly is the point: task 3350
    exists because a justification nobody re-checked was left standing until it
    was off by an order of magnitude, and a guard that overstates its own reach
    is the same defect wearing a test's clothes.
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
    # the same chain PLUS verify_cold_preprovision_command (uv sync
    # --all-packages) — so a warm ceiling above the cold one is incoherent by
    # construction, regardless of what either value is.
    cold = budgets['verify_cold_command_timeout_secs']
    assert warm <= cold, (
        f'verify_command_timeout_secs={warm} exceeds '
        f'verify_cold_command_timeout_secs={cold} (task 3350). A cold verify runs '
        'the same command chain plus the uv sync --all-packages preprovision, so '
        'it is strictly more expensive; a warm budget above the cold one is '
        'incoherent by construction'
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
