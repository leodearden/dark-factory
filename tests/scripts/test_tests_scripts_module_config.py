"""Routing contract: the repo-root ``tests/scripts/`` suite owns a module config.

Task 3350 (escalation esc-3062-3). A diff confined to this suite must resolve
to its OWN module config so it never reaches ``_build_fallback_config``'s
``__fallback__`` branch.

Why that branch is fatal here: ``_build_fallback_config`` is consulted only
when the discovered ``module_configs`` list is EMPTY. For a diff touching only
``tests/scripts/``, ``_single_subproject_prefix`` and
``_root_plus_single_subproject_prefix`` both return None (``tests`` has no
pyproject.toml and no real subproject is touched), so it falls through to the
``config.test_command != 'pytest'`` branch: prefix ``__fallback__``,
``test_command`` = the whole seven-segment fleet chain, and
``verify_command_timeout_secs=None`` so the global ceiling applies. Task 3062
attempt-2 measured that path consuming 1733.60s across four of seven segments
before dashboard even started, then timing out at 1800.66s — category
infra_timeout, on a diff whose own tests take ~105s ON THAT SAME PATH. That
last figure describes the 3062 incident and nothing else: it is not a run of
the command this module config now declares, and it must never be used to size
or describe this budget — see ``MEASURED_SUITE_WORST_SECS`` below, which
carries the runs that may be.

Registering a real module config makes the list non-empty, which is by itself
sufficient to bypass the fallback builder entirely. These tests assert that
routing against the PRODUCTION functions (``_discover_module_configs``,
``derive_modules``, ``OrchestratorConfig.for_module``,
``derive_verify_plan``, ``_executed_module_configs_from_plan``) rather than
reimplementations, so the guard cannot drift from the routing it protects.

Importing ``orchestrator.config`` from this suite is established precedent —
see ``test_orchestrator_restart_config_drift.py`` and
``test_offline_lane_qdrant_config.py`` in this same directory; the root
conftest.py puts every subproject's ``src/`` on sys.path.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml
from orchestrator.config import OrchestratorConfig, _discover_module_configs
from orchestrator.module_charter import derive_modules

from orchestrator import verify, verify_plan

REPO_ROOT = pathlib.Path(__file__).parents[2]

# This worktree's own top-level orchestrator config — the one `_root_config`
# reads. `dark-factory-orchestrator.yaml` is the canonical, REQUIRED filename
# for a project's top-level config (it is what the dashboard's escalation-URL
# discovery keys on); the legacy spellings are a discovery fallback for
# unmigrated projects, not a choice this repo has. Anchored to REPO_ROOT rather
# than taken from the ambient ORCH_CONFIG_PATH for the reason that helper's
# docstring records at length.
ROOT_CONFIG_PATH = REPO_ROOT / 'dark-factory-orchestrator.yaml'

MODULE_PREFIX = 'tests/scripts'

# A file in this suite, used as the representative touched-file for the
# derive_modules -> for_module routing assertions below.
SAMPLE_TOUCHED_FILE = 'tests/scripts/test_spawn_claude.py'


def _discovered() -> dict:
    return _discover_module_configs(REPO_ROOT)


def _root_config(monkeypatch: pytest.MonkeyPatch) -> OrchestratorConfig:
    """Load the repo-root config through the PRODUCTION loader, anchored at ROOT_CONFIG_PATH.

    COPIED (task 3703) from the two sibling guards
    (``test_scripts_module_config.py`` and ``test_module_verify_budgets.py``),
    not imported: a test file importing a sibling test file couples two guards
    that must be able to fail independently, and this anchor is load-bearing
    enough that it must be visibly present in the file that depends on it.

    THE COST OF THAT, RECORDED RATHER THAN LEFT IMPLICIT (task 3703 amendment
    pass, reviewer-flagged): this makes THREE verbatim copies of a helper that
    is pure setup, and the no-cross-import argument — sound for ASSERTIONS —
    does not reach ``tests/scripts/conftest.py``, which already exists and is
    pytest's idiomatic home for exactly this. One anchoring implementation as a
    conftest fixture would leave each guard's assertions just as independent.
    NOT done here: conftest.py is outside this task's locked module list, so
    de-triplicating it is filed as a follow-up rather than reached for.

    ANCHORING ``ORCH_CONFIG_PATH`` IS LOAD-BEARING, not hygiene.
    ``project_root`` is only a model FIELD and selects nothing:
    ``OrchestratorConfig.settings_customise_sources`` builds its
    ``YamlSettingsSource`` from ``os.environ['ORCH_CONFIG_PATH']`` alone,
    falling back to a CWD-relative ``config.yaml``. Both ambient states are
    wrong here, in OPPOSITE directions:

      * UNSET — the state INSIDE VERIFY, because
        ``verify._target_subprocess_env`` deliberately scrubs the whole
        ``ORCH_`` prefix (task 2957) — finds no file, so every value collapses
        to the pydantic DEFAULTS, a config this repo does not declare.
      * SET, as an operator's shell has it, points at whichever checkout that
        orchestrator serves — typically the MAIN one, not this worktree. Every
        assertion would then be about a different checkout's yaml and report
        GREEN on a worktree that had actually regressed: the exact
        reports-green-while-checking-something-else failure this file exists
        to prevent, one env var over.

    Setting the env var IS the production load path (``config.load_config``
    stamps ``os.environ['ORCH_CONFIG_PATH']`` before constructing), so this
    stays a read through the real loader, pinned to THIS worktree's committed
    yaml rather than left to the ambient environment.

    Fails LOUDLY on a missing file rather than silently: ``YamlSettingsSource``
    SKIPS a non-existent ``config_path`` instead of raising, so a bad path
    would yield the pydantic DEFAULTS with no error at all.
    """
    assert ROOT_CONFIG_PATH.is_file(), (
        f'{ROOT_CONFIG_PATH} does not exist, so anchoring ORCH_CONFIG_PATH at '
        'it would silently load the pydantic DEFAULTS instead (YamlSettingsSource '
        'skips a non-existent path rather than raising), and every value read '
        'from the returned config would be about a config this repo does not '
        'declare. dark-factory-orchestrator.yaml is the canonical, required '
        "filename for a project's top-level orchestrator config"
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(ROOT_CONFIG_PATH))
    return OrchestratorConfig(project_root=REPO_ROOT)


def test_tests_scripts_is_a_registered_module_config() -> None:
    """tests/scripts/ must be discovered, and must be REACHABLE by the routing chain.

    The five assertions below are one cohesive contract — discovery alone is
    not enough, because a config the scheduler/workflow path cannot reach is
    silently half-applied (``load_config`` warns about exactly this at
    config.py:4712-4725 but does not fail).

    NOTE on assertion (3)/(4) — a falsified premise corrected against measured
    reality (the same class of defect this task exists to fix). Task 3350's
    plan asserted ``lock_depth == 2``, reading the pydantic Field default at
    config.py:2593. That is not the effective value, and neither is the
    package-bundled ``orchestrator/src/orchestrator/defaults.yaml:7`` layered
    over it on every load: THIS project's ``dark-factory-orchestrator.yaml``
    overrides both, and that override is what a real load resolves here.

        CORRECTED IN PLACE (task 3866): this note used to say "The EFFECTIVE
        value is 4" and reason "at depth 4", which broke when lock_depth moved
        4 -> 12. Noted rather than silently rewritten — this docstring exists
        BECAUSE task 3350 encoded a falsified constant, and a second silent
        substitution would defeat its purpose. Which is also why the repair
        names no new constant: assertion (3) reads ``cfg.lock_depth`` live, so
        the next retune of that knob falsifies nothing written here.

        CORRECTED AGAIN (task 3866, review repair): that first repair broke
        the rule it had just stated. In place of the stale constant it
        asserted a resolved path — ``derive_modules([SAMPLE_TOUCHED_FILE],
        cfg.lock_depth)`` "returns the full path ..., NOT ['tests/scripts'] —
        3 path components is below the truncation threshold at every layer" —
        and that quantifier is false at a layer the paragraph above
        enumerates by hand. MEASURED:
        ``normalize_lock('tests/scripts/test_spawn_claude.py', 2)`` ->
        ``'tests/scripts'``, which is exactly the value the sentence claimed
        was excluded; depths 3 / 4 / 12 do leave it whole. A 3-component path
        survives only at ``lock_depth >= 3``, and the pydantic Field default
        of 2 sits below that threshold. The sentence is DELETED rather than
        re-scoped, because a resolved lock key IS a depth assertion in prose
        and the note directly above already forbids those; re-scoping would
        merely re-stake the claim on ``SAMPLE_TOUCHED_FILE`` remaining 3
        components. What survives is the depth-invariant claim the assertions
        actually check — every derived key RESOLVES back to this config.

    That does not weaken the fix, and the plan's conclusion still holds: what
    actually matters is that the derived lock key RESOLVES to this module
    config, and ``for_module`` walks candidate prefixes from the full path
    inward (``tests/scripts/test_spawn_claude.py`` -> ``tests/scripts`` ->
    ``tests``), returning the deepest registered match. So assertion (4) is
    written in the lock_depth-agnostic form — every derived key must resolve
    to this config — instead of pinning a literal that is only true at depth
    2. Pinning ``== ['tests/scripts']`` would re-encode a falsified constant,
    which is precisely the failure mode task 3350 is repairing elsewhere.
    """
    discovered = _discovered()

    # (1) Discovery registers it at all.
    assert MODULE_PREFIX in discovered, (
        f'{MODULE_PREFIX}/orchestrator.yaml is not discovered by the production '
        f'config._discover_module_configs walk (task 3350). Without it, a diff '
        f'confined to {MODULE_PREFIX}/ leaves module_configs EMPTY, which is the '
        'sole trigger for _build_fallback_config — whose __fallback__ branch '
        'hands back the whole-fleet test_command that measured 1800.66s / '
        f'infra_timeout in task 3062 attempt-2. Discovered: {sorted(discovered)}'
    )

    mc = discovered[MODULE_PREFIX]

    # (2) Registered under the repo-relative POSIX prefix, as for_module expects.
    assert mc.prefix == MODULE_PREFIX, (
        f'module config discovered for {MODULE_PREFIX} carries prefix '
        f'{mc.prefix!r}; for_module resolves by repo-relative POSIX prefix, so a '
        'mismatch makes it unroutable'
    )

    cfg = OrchestratorConfig(project_root=REPO_ROOT)
    prefix_depth = len(MODULE_PREFIX.split('/'))

    # (3) Reachability precondition: a prefix DEEPER than lock_depth is honoured
    # by run_full_verification (which iterates module_configs.values() directly)
    # but is unreachable via scheduler/workflow, which pass normalize_lock-
    # truncated keys. config.py:4719 warns; it does not fail. Asserted against
    # the live cfg.lock_depth rather than any named constant, so it holds at
    # whichever layer is in force — Field default, shipped defaults.yaml, or
    # this project's override.
    assert prefix_depth <= cfg.lock_depth, (
        f'module config prefix {MODULE_PREFIX!r} has depth {prefix_depth} but '
        f'lock_depth={cfg.lock_depth}; the scheduler (_limit_for) and workflow '
        '(_resolve_module_configs) truncate module paths to lock_depth '
        'components via normalize_lock, so this config would be unreachable '
        'through the path that actually dispatches verify (config.py:4712-4725)'
    )

    # (4) The production derivation for a touched file in this suite yields
    # lock key(s) that all resolve BACK to this module config.
    cfg._module_configs = discovered
    derived = derive_modules([SAMPLE_TOUCHED_FILE], cfg.lock_depth)
    assert derived, (
        f'derive_modules([{SAMPLE_TOUCHED_FILE!r}], {cfg.lock_depth}) derived no '
        'module lock keys at all, so the workflow would fall through to its '
        'task-<id> synthetic lock and never resolve a module config'
    )
    for key in derived:
        resolved = cfg.for_module(key)
        assert resolved is not None and resolved.prefix == MODULE_PREFIX, (
            f'derived module lock key {key!r} resolves to '
            f'{resolved.prefix if resolved else None!r}, not {MODULE_PREFIX!r} '
            '(task 3350) — workflow._resolve_module_configs would then produce '
            'an EMPTY module list, which is the exact condition that sends '
            'run_scoped_verification into the _build_fallback_config fleet chain'
        )

    # (5) The prefix itself resolves — the form scheduler/workflow pass when the
    # touched path IS truncated to lock_depth (i.e. whenever lock_depth is below
    # the sample path's own depth; above it the path is passed whole, which
    # assertion (4) covers). Both forms are pinned so neither side of that
    # threshold depends on where the knob currently sits.
    resolved_prefix = cfg.for_module(MODULE_PREFIX)
    assert resolved_prefix is not None and resolved_prefix.prefix == MODULE_PREFIX, (
        f'cfg.for_module({MODULE_PREFIX!r}) did not resolve to the discovered '
        f'module config (got {resolved_prefix!r}); module_configs would be empty '
        'on the workflow path and the fleet-chain fallback would be reached'
    )


# Command fragments that would betray the fleet chain leaking back in: each is a
# segment of dark-factory-orchestrator.yaml's seven-suite test_command. If any
# appears in the executed test_command, the module config is not doing its job.
_FLEET_CHAIN_MARKERS = (
    'cd shared',
    'cd ../escalation',
    'cd ../orchestrator',
    'cd ../fused-memory',
    'cd ../dashboard',
    'cd ../sampler',
    'cockpit',
)


def _executed_for_touched(files: list[str], cfg: OrchestratorConfig):
    """Run the PRODUCTION plan->execution bridge and return the single executed config.

    ``derive_verify_plan`` decides scope; ``_executed_module_configs_from_plan``
    renders those PlannedRuns into the exact ModuleConfig ``run_verification``
    executes. Asserting on THAT is what makes "the tests/scripts segment ran" a
    structural claim rather than an exit-code claim.

    *cfg* IS A REQUIRED PARAMETER, not a convenience (task 3703, applying the
    shape commit 6c72a7da5a landed next door in
    ``test_module_verify_budgets.py``). It must be a config built by
    ``_root_config``, whose docstring spells out why the ``ORCH_CONFIG_PATH``
    anchor is load-bearing: an unset anchor collapses every value to the
    pydantic defaults, SILENTLY. This helper used to construct its own
    ``OrchestratorConfig(project_root=REPO_ROOT)``, and — unlike the sibling
    guard, whose single anchoring caller at least left the env var set as a
    SIDE EFFECT of an earlier assertion — NO caller in this file anchored it at
    all. So it read whatever ambient ``ORCH_CONFIG_PATH`` the process happened
    to carry, with no failure signal either way. Taking the config as an
    argument makes the dependency structural instead of ambient.

    The ``lambda _f: None`` worktree_reader keeps this hermetic: no file reads,
    and nothing classifies STRUCTURAL, so pyright stays FILE_SCOPED.
    """
    mc = _discovered()[MODULE_PREFIX]
    plan = verify_plan.derive_verify_plan(files, [mc], cfg, lambda _f: None)
    executed = verify._executed_module_configs_from_plan([mc], plan)
    assert len(executed) == 1, (
        f'expected exactly one executed module config for {files!r}, got '
        f'{[e.prefix for e in executed]!r}'
    )
    return executed[0]


def test_tests_scripts_diff_executes_its_own_suite_and_keeps_lint_and_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The task's acceptance criterion, asserted STRUCTURALLY rather than via rc.

    "A diff touching only tests/scripts/ should complete verify well inside
    budget, and its own tests/scripts/ tests must actually execute (assert the
    segment ran, not merely that the command exited 0)." An exit code cannot
    carry that claim — task 3338's failure mode is precisely an &&-chain that
    exits 0 without the trailing segment ever running. So this asserts on the
    command strings the production bridge hands to ``run_verification``.

    (c) and (d) are the anti-regression half, and they are not hypothetical:
    ``verify_plan.py:334-363`` emits an explicit SKIPPED PlannedRun for a falsy
    lint_command/type_check_command, ``_executed_module_configs_from_plan``
    leaves the field None (verify.py:4892), and ``_run_or_skip_timed`` turns a
    None command into a ``CheckRun.skipped`` that is VACUOUSLY PASSING at rc=0.
    A test-only module config would therefore silently DELETE lint and type
    gating for every tests/scripts diff — strictly WORSE than the fallback path
    it replaces, which esc-3062-3 logs narrowing both legs correctly
    (``ruff check tests/scripts/test_spawn_claude.py`` rc=0,
    ``pyright tests/scripts/test_spawn_claude.py`` rc=0). Closing the TEST gap
    must not open a LINT/TYPE one.
    """
    executed = _executed_for_touched([SAMPLE_TOUCHED_FILE], _root_config(monkeypatch))

    # (a) The tests/scripts segment ACTUALLY RUNS.
    assert executed.test_command is not None, (
        'executed test_command is None for a tests/scripts-only diff (task '
        '3350) — derive_verify_plan emitted a SKIPPED run, so _run_or_skip_timed '
        'records a vacuously-passing CheckRun.skipped and this suite is not '
        'gated at all'
    )
    assert 'pytest' in executed.test_command, (
        f'executed test_command {executed.test_command!r} invokes no pytest, so '
        'the tests/scripts suite does not run'
    )
    assert MODULE_PREFIX in executed.test_command, (
        f'executed test_command {executed.test_command!r} does not target '
        f'{MODULE_PREFIX}/ — the suite that was actually touched is not the '
        'suite being run'
    )

    # (b) The fleet chain is gone. This is the whole point of the module config:
    # no other subproject's suite may appear in the command for this diff.
    for marker in _FLEET_CHAIN_MARKERS:
        assert marker not in executed.test_command, (
            f'executed test_command contains fleet-chain segment {marker!r} '
            f'(task 3350): {executed.test_command!r}. A tests/scripts-only diff '
            'has fallen back to dark-factory-orchestrator.yaml\'s whole-fleet '
            'test_command, which measured 1800.66s / infra_timeout in task 3062 '
            'attempt-2 — the exact defect this module config exists to remove'
        )

    # (c) LINT still gates. See the docstring: a None command is a vacuous pass.
    assert executed.lint_command is not None and 'ruff' in executed.lint_command, (
        f'executed lint_command is {executed.lint_command!r} (task 3350) — '
        'declaring only test_command on the module config downgrades LINT to a '
        'vacuously-passing CheckRun.skipped at rc=0, silently DELETING gating '
        'that the fallback path performs correctly today (esc-3062-3 logs '
        'ruff check tests/scripts/test_spawn_claude.py rc=0)'
    )

    # (d) TYPE still gates, same reasoning.
    assert (
        executed.type_check_command is not None
        and 'pyright' in executed.type_check_command
    ), (
        f'executed type_check_command is {executed.type_check_command!r} (task '
        '3350) — declaring only test_command on the module config downgrades '
        'TYPE to a vacuously-passing CheckRun.skipped at rc=0, silently '
        'DELETING gating that the fallback path performs correctly today '
        '(esc-3062-3 logs pyright tests/scripts/test_spawn_claude.py rc=0)'
    )


def test_executed_for_touched_is_hermetic_against_the_ambient_orch_config_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_executed_for_touched`` must not read the ambient ``ORCH_CONFIG_PATH``.

    Task 3703, reviewer-flagged — the mirror of the repair commit 6c72a7da5a
    landed next door in ``test_module_verify_budgets.py``. This file's helper
    used to construct its OWN ``OrchestratorConfig(project_root=REPO_ROOT)``,
    and ``project_root`` selects NOTHING: ``settings_customise_sources`` builds
    the ``YamlSettingsSource`` from ``os.environ['ORCH_CONFIG_PATH']`` alone,
    falling back to a CWD-relative ``config.yaml``. NO caller in this file
    anchored that variable at all — worse than the sibling's ordering
    dependency, which at least anchors it somewhere — so the helper read
    whatever yaml the ambient process happened to point at: the pydantic
    DEFAULTS inside verify (``verify._target_subprocess_env`` scrubs the whole
    ``ORCH_`` prefix), or the MAIN checkout's config in an operator shell.
    Taking the config as an argument makes the dependency structural instead of
    ambient.

    WHY THE HOSTILE YAML IS ONE THE LOADER REJECTS, rather than one that merely
    holds different values. This is not a stylistic choice; it is the only
    formulation that can fail BEFORE the fix. MEASURED at base d6a5e32535:
    ``derive_verify_plan`` consults its ``config`` argument in exactly two
    places — ``_merge_breadth_is_full(config)`` on the role='merge' branch, and
    ``_derive_fallback_runs`` when ``module_configs`` is EMPTY. This call is
    role='task' with one non-empty ModuleConfig, so it reaches NEITHER: a
    differently-VALUED config yields a byte-identical plan, and a test built on
    one would pass before the fix as readily as after — a guard that reads as
    enforcement while enforcing nothing. A type-invalid value instead makes the
    PRE-fix helper's own ``OrchestratorConfig`` construction raise pydantic
    ``ValidationError``, which is the single observable signal that the ambient
    dependency exists at all.

    The post-fix GREEN is reachable rather than assumed, measured at that same
    base: ``_discover_module_configs``, ``derive_verify_plan`` and
    ``verify._executed_module_configs_from_plan`` are all unaffected by the
    poisoned env — they read this module's own yaml and the config they are
    handed, never ``ORCH_CONFIG_PATH``.

    ORDER IS LOAD-BEARING: the anchored config is built FIRST, while the
    environment is still sane, and the poison applied SECOND. Anchoring
    afterwards would overwrite the poison and leave this test vacuous.
    """
    # (1) The anchored config, built while the environment is still sane.
    cfg = _root_config(monkeypatch)

    # (2) NOW poison the ambient environment, with a config the PRODUCTION
    # loader REJECTS — see the docstring for why rejected and not merely
    # different.
    hostile = tmp_path / 'hostile.yaml'
    hostile.write_text(
        'verify_command_timeout_secs: "not-a-number"\n', encoding='utf-8'
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(hostile))

    # (3) The helper must not consult that variable. A helper that builds its
    # own config raises pydantic ValidationError here instead of returning.
    executed = _executed_for_touched([SAMPLE_TOUCHED_FILE], cfg)

    assert executed.prefix == MODULE_PREFIX, (
        f'under a poisoned ORCH_CONFIG_PATH the production bridge executed '
        f'{executed.prefix!r}, not {MODULE_PREFIX!r} (task 3703) — module '
        'routing is reading the ambient environment'
    )
    assert executed.test_command is not None and 'pytest' in executed.test_command, (
        f'executed test_command is {executed.test_command!r} under a poisoned '
        f'ORCH_CONFIG_PATH (task 3703); {MODULE_PREFIX} must still run its own '
        'suite regardless of what the ambient environment points at'
    )
    assert MODULE_PREFIX in executed.test_command, (
        f'executed test_command {executed.test_command!r} does not target '
        f'{MODULE_PREFIX}/ under a poisoned ORCH_CONFIG_PATH (task 3703)'
    )

    # The lint/type legs, for the same reason assertions (c)/(d) of
    # test_tests_scripts_diff_executes_its_own_suite_and_keeps_lint_and_type
    # exist: a None command is a VACUOUSLY PASSING CheckRun.skipped at rc=0, so
    # an ambient-config leak that silently dropped them would report green.
    assert executed.lint_command is not None, (
        'executed lint_command is None under a poisoned ORCH_CONFIG_PATH (task '
        '3703) — a None command is a vacuously-passing CheckRun.skipped at '
        'rc=0, so this leak would delete lint gating without reporting anything'
    )
    assert executed.type_check_command is not None, (
        'executed type_check_command is None under a poisoned ORCH_CONFIG_PATH '
        '(task 3703) — same vacuous-pass mechanism as the lint leg above'
    )

    # The module budget survives too: the figure must come from THIS module's
    # yaml, which the poisoned env cannot reach, not from whatever the ambient
    # config declares.
    declared = _discovered()[MODULE_PREFIX].verify_command_timeout_secs
    assert executed.verify_command_timeout_secs == declared, (
        f'executed verify_command_timeout_secs='
        f'{executed.verify_command_timeout_secs} under a poisoned '
        f'ORCH_CONFIG_PATH, not the {declared} this module declares (task '
        '3703) — the budget is being resolved from the ambient environment'
    )


def _min_budget(worst: float) -> int:
    """~2x the worst measured run, rounded DOWN to the nearest 100s.

    The EXACT expression both sibling guards use for their own floors
    (``test_scripts_module_config.MIN_MODULE_BUDGET_SECS``, task 3458, and
    ``test_module_verify_budgets._min_budget``, task 3473), COPIED rather than
    imported so the three cannot drift in SHAPE while staying free to fail
    INDEPENDENTLY — a test file importing a sibling test file couples two
    guards that must be able to fail apart, the convention
    ``test_module_verify_budgets.py``'s header states. The price of that
    convention, stated rather than glossed: nothing in THIS file can observe a
    sibling's copy, so
    ``test_min_module_budget_is_derived_from_the_measured_worst_run`` pins the
    SHAPE of the expression below against synthetic cases and claims no more
    than that.

    DERIVED from the measurement rather than hand-set beside it, because that
    exact pair has already rotted once undetected — in THIS file: a hand-set
    300s floor left standing against a 127.0s figure while
    ``tests/scripts/orchestrator.yaml`` had since recorded a 233.50s worst run
    of the command this module actually declares (task 3703,
    reviewer-flagged; both sibling guards had NAMED the staleness in prose,
    which is precisely what a comment can do and an assertion could not).

    DEGENERATES TO ZERO for cheap suites — ``_min_budget(22.49) == 0``, the
    sampler case ``test_module_verify_budgets.py`` records — which would make
    the floor assertion below VACUOUS. Assertion (iii) of the derivation test
    pins that this suite is not in that regime.
    """
    return (int(2 * worst) // 100) * 100


# The tests/scripts suite's own measured wall-clock. EVERY recorded run, the
# inconvenient ones included: task 3350 exists because a single unrecorded
# estimate ("Full warm verify here is ~2 min") was left standing in the root
# yaml until it was off by an order of magnitude.
#
#   VERBATIM runs of the test_command this module config declares
#   (`uv run --project shared pytest tests/scripts/ --tb=short -q --timeout=300`):
#     233.50s, 167.73s               task 3350's worktree at HEAD 9d6af5289f;
#                                    369 passed / 1 skipped each
#     146.93s wall / 142.11s pytest  task 3703 pre-1 run A, base d6a5e32535;
#                                    rc=0, 1072 passed / 2 skipped / 6
#                                    deselected; /proc/loadavg 51.66 at start
#                                    on a 32-core host
#     185.29s wall / 180.47s pytest  task 3703 pre-1 run B, same base, same
#                                    counts, rc=0; loadavg 61.43 at start
#
#   FALLBACK-PATH runs — A DIFFERENT COMMAND, kept for history and LABELLED so
#   nobody sizes this budget against them again:
#     105-127s                       four runs on the __fallback__ path (task
#                                    3062, esc-3062-3) — the fleet chain's
#                                    legs narrowed to this diff, not the
#                                    command declared here
#
# SIZING RULE: the WORST RUN, never the mean and never fresh-only. Both of
# pre-1's fresh runs came in BELOW 233.50s, so sizing on them would LOWER the
# floor — the unsafe direction. The spread is this oversubscribed host's LOAD at
# measurement time, not suite variance: 167.73s and 233.50s are the same command
# on the same tree.
#
# HONEST CAVEAT, recorded rather than left implicit: 233.50s was measured when
# this suite collected 369 tests, and it now collects 1072 (measured in pre-1).
# It is therefore a LOWER BOUND on today's contended worst case, not a current
# measurement. Do NOT scale it by the test-count ratio — RE-MEASURE under load
# before ever tightening the budget against it.
MEASURED_SUITE_WORST_SECS = 233.50
# DERIVED from the measurement above, never hand-set beside it, so the two
# cannot silently diverge again — see _min_budget's docstring for the rot that
# motivated this and test_min_module_budget_is_derived_from_the_measured_worst_run
# for the guard that now makes it a property. 2 * 233.50 -> 467.0 -> 400.
MIN_MODULE_BUDGET_SECS = _min_budget(MEASURED_SUITE_WORST_SECS)


def test_min_module_budget_is_derived_from_the_measured_worst_run() -> None:
    """The floor must be DERIVED from the measurement, in the family's canonical shape.

    Task 3703, reviewer-flagged. This exact pair had already rotted once,
    undetected: ``MEASURED_SUITE_WORST_SECS`` sat at 127.0 with a HAND-SET
    ``MIN_MODULE_BUDGET_SECS = 300`` while ``tests/scripts/orchestrator.yaml``
    had since recorded a 233.50s worst run of the VERBATIM command this module
    declares — so the yaml's own worst figure was gated by NOTHING, and the two
    sibling guards could only NAME the staleness in their comment blocks
    (``test_scripts_module_config.py``'s ``MIN_MODULE_BUDGET_SECS`` comment and
    ``test_module_verify_budgets.py``'s ``_min_budget`` docstring both call it
    out verbatim) rather than fail on it. Deriving the floor is what turns that
    from a documented observation into a property.

    SCOPE, NARROWED TO WHAT IS ACTUALLY VERIFIED (task 3703 amendment pass,
    reviewer-flagged: this docstring previously described (i) as cross-file
    enforcement, which it is not and cannot be). ``_min_budget`` is a LOCAL
    copy, per the family's no-cross-import convention, so nothing here can
    observe a sibling's expression: if ``test_scripts_module_config.py``
    changed its own inline derivation, every assertion below would still pass.
    (i) is a purity check on THIS file's copy, (ii) is an edit tripwire on THIS
    file's pair. Real cross-file enforcement — one guard reading every family
    member's published pair and checking the derivation holds for all of them —
    would be a new mechanism rather than an amendment, and is filed as a
    follow-up instead of being claimed here.

    Three claims, each falsifiable on its own:

    (i) The SHAPE of the derivation: 2x the worst run, floored to the nearest
        100s. Pinned against SYNTHETIC cases chosen so a different multiple, or
        a round UP, fails. Synthetic rather than a sibling's published pair on
        purpose — an earlier draft quoted ``(930.59, 1800)`` from
        ``test_scripts_module_config.py``, which made this file a SECOND COPY of
        that module's measurement: exactly the duplication
        ``test_module_verify_budgets.py``'s ``MEASURED_BY_SIBLING_GUARD``
        declines to create ("a copy that would have to be raised in lockstep"),
        and one that would start asserting a figure the sibling no longer
        publishes the moment it re-measures. That is the drift class this whole
        task targets, so it is not reintroduced in the test against it.

    (ii) An EDIT TRIPWIRE, not a property: ``MIN_MODULE_BUDGET_SECS`` is
        DEFINED as ``_min_budget(MEASURED_SUITE_WORST_SECS)`` a few lines above,
        so this can only fail if a future edit puts a literal back in its place
        — which is precisely the rot described above, and precisely what the
        two sibling guards' comments could name but not catch.

    (iii) The derivation is NON-DEGENERATE for this suite. ``_min_budget``
        floors to the nearest 100s, so a cheap suite derives a floor of ZERO:
        ``test_module_verify_budgets.py`` documents exactly that case for
        sampler (``_min_budget(22.49) == 0``), where the corresponding budget
        assertion becomes VACUOUSLY true. At this suite's measured cost it does
        not degenerate, and this pins that — so assertion (b) of
        ``test_tests_scripts_module_carries_its_own_tight_verify_budget`` below
        is checking something real.
    """
    # (i) The family's SHAPE, pinned against synthetic cases — see the docstring
    # for why synthetic and not a sibling's published pair.
    #   100.0 -> 200  the 2x multiple, exact at a round boundary
    #   299.9 -> 500  rounds DOWN (599.8); rounding UP would give 600
    #    50.0 -> 100  does not degenerate at the boundary
    #    49.9 ->   0  degenerates just below it (the regime (iii) rules out)
    for worst, expected in ((100.0, 200), (299.9, 500), (50.0, 100), (49.9, 0)):
        assert _min_budget(worst) == expected, (
            f'_min_budget({worst}) == {_min_budget(worst)}, expected {expected} '
            '(task 3703). The family derivation is ~2x the worst measured run '
            'floored to the nearest 100s; a different multiple, or rounding UP '
            "instead of DOWN, changes the SHAPE of this file's floor and must "
            'fail here rather than quietly produce a differently-sized budget'
        )

    # (ii) DERIVED, not hand-set — an edit tripwire on the pair that already
    # rotted once here (see the docstring). Written derivation-first rather than
    # constant-first only because ruff's SIM300 reads the other order as a Yoda
    # condition; equality is symmetric and the claim is unchanged.
    assert _min_budget(MEASURED_SUITE_WORST_SECS) == MIN_MODULE_BUDGET_SECS, (
        f'MIN_MODULE_BUDGET_SECS = {MIN_MODULE_BUDGET_SECS} but '
        f'_min_budget(MEASURED_SUITE_WORST_SECS={MEASURED_SUITE_WORST_SECS}) = '
        f'{_min_budget(MEASURED_SUITE_WORST_SECS)} (task 3703) — a literal has '
        'been put back where the derivation was. Raise '
        'MEASURED_SUITE_WORST_SECS from a fresh measurement and let the floor '
        'follow; do not hand-set the floor'
    )

    # (iii) Non-degenerate for THIS suite, so the budget floor is not vacuous.
    assert MIN_MODULE_BUDGET_SECS > 0, (
        f'MIN_MODULE_BUDGET_SECS derives to {MIN_MODULE_BUDGET_SECS} from '
        f'MEASURED_SUITE_WORST_SECS={MEASURED_SUITE_WORST_SECS} (task 3703). '
        '_min_budget floors to the nearest 100s, so a worst run under 50s '
        'derives a ZERO floor — the degenerate case test_module_verify_budgets.py '
        'records for sampler (_min_budget(22.49) == 0), where the >= floor '
        'assertion passes for ANY declared budget. This suite measures far above '
        'that, so a zero here means the measurement was lost, not that the suite '
        'got cheap'
    )


def test_tests_scripts_module_carries_its_own_tight_verify_budget() -> None:
    """The module must carry its own budget, tighter than the repo-root ceiling.

    This is what makes the fix a real NARROWING rather than a relabelling. Two
    sides are bounded:

    - Below (b): at least ``MIN_MODULE_BUDGET_SECS``, which is not a literal
      but ``_min_budget(MEASURED_SUITE_WORST_SECS)`` — ~2x the worst RECORDED
      run of this module's VERBATIM test_command, floored to the nearest 100s.
      An achievable floor derived from measurement, not a guess, and derived
      rather than transcribed so it cannot fall out of step with the figure it
      is derived from.

        CORRECTED IN PLACE (task 3703). This bullet used to read "at least
        300s, which is >= 2.36x the worst of the four independent measurements
        task 3062 logged for this suite (105-127s)", and it was stale in BOTH
        numbers and in WHICH COMMAND it credited: those four runs are
        esc-3062-3's __fallback__-path runs, not runs of the command this
        module config declares, and tests/scripts/orchestrator.yaml had since
        recorded a 233.50s VERBATIM run that the hand-set 300s floor did not
        reflect at all. Named rather than silently substituted, for the reason
        the note in ``test_tests_scripts_is_a_registered_module_config`` gives
        at length: this file exists BECAUSE a falsified constant was left
        standing, so a quiet swap would defeat its purpose.
    - Above (c): strictly below the repo-root ceiling. The "tight timeouts
      surface hangs" intent that the root yaml states but — at a ceiling the
      honest green path provably cannot clear — could not deliver, now lives
      here, at the level where it can actually hold: a hang in this suite
      surfaces in minutes instead of consuming the whole-fleet budget.

    (d) exercises the REAL precedence mechanism (``verify._resolve_verify_timeout``)
    rather than restating it, so the assertion cannot drift from the code that
    implements it.
    """
    mc = _discovered()[MODULE_PREFIX]

    # (a) Declared at all — otherwise the global ceiling silently applies. The
    # message cites MEASURED_SUITE_WORST_SECS rather than a literal (task 3703
    # amendment pass, reviewer-flagged): it used to read "a suite that measures
    # ~105-127s", which are the __fallback__-path figures the provenance block
    # above LABELS as a different command and forbids sizing or describing this
    # budget with — so an operator read the discredited number in the one place
    # they see on failure. Naming the constant instead means the message cannot
    # fall out of step with the figure the file derives everything from.
    assert mc.verify_command_timeout_secs is not None, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares no '
        'verify_command_timeout_secs (task 3350), so this suite silently '
        'inherits the repo-root whole-fleet ceiling — the budget sized for '
        'seven subprojects, applied to a suite whose worst RECORDED run of its '
        f'own declared test_command is {MEASURED_SUITE_WORST_SECS}s'
    )

    # (b) Measurement-derived floor — DERIVED by _min_budget from
    # MEASURED_SUITE_WORST_SECS, not hand-set beside it (task 3703).
    assert mc.verify_command_timeout_secs >= MIN_MODULE_BUDGET_SECS, (
        f'{MODULE_PREFIX} verify_command_timeout_secs='
        f'{mc.verify_command_timeout_secs} is below the '
        f'{MIN_MODULE_BUDGET_SECS}s floor (task 3350). That floor is not a '
        f'literal: it is _min_budget(MEASURED_SUITE_WORST_SECS='
        f'{MEASURED_SUITE_WORST_SECS}), ~2x the worst RECORDED run of this '
        "module's VERBATIM test_command floored to the nearest 100s — see the "
        'provenance block above that constant for every run behind it. A budget '
        'under the floor would manufacture infra_timeout on the honest green '
        'path — the exact defect this task exists to remove, reintroduced one '
        'level down. Raise MEASURED_SUITE_WORST_SECS from a fresh measurement '
        'and the floor follows; do not hand-set the floor back'
    )

    # (c) Strictly tighter than the repo-root ceiling: a real narrowing.
    root = yaml.safe_load(
        (REPO_ROOT / 'dark-factory-orchestrator.yaml').read_text(encoding='utf-8')
    )
    root_budget = root['verify_command_timeout_secs']
    assert mc.verify_command_timeout_secs < root_budget, (
        f'{MODULE_PREFIX} verify_command_timeout_secs='
        f'{mc.verify_command_timeout_secs} is not strictly below the repo-root '
        f'verify_command_timeout_secs={root_budget} (task 3350). A per-module '
        'budget at or above the global one is a relabelling, not a narrowing: '
        'it surfaces a hang no sooner than the whole-fleet ceiling would'
    )

    # (d) verify.py's documented precedence actually honours it. Called against
    # the real resolver, whose docstring reads "module_config.
    # verify_command_timeout_secs takes precedence over config.
    # verify_command_timeout_secs".
    cfg = OrchestratorConfig(
        project_root=REPO_ROOT, verify_command_timeout_secs=root_budget
    )
    resolved = verify._resolve_verify_timeout(cfg, mc, is_cold=False)
    assert resolved == mc.verify_command_timeout_secs, (
        f'_resolve_verify_timeout returned {resolved} for a warm verify, not the '
        f'module budget {mc.verify_command_timeout_secs} (task 3350) — the '
        'per-module override is not reaching the code path that enforces it'
    )
    assert resolved != root_budget, (
        f'_resolve_verify_timeout returned the repo-root global {root_budget} '
        f'rather than the module budget for {MODULE_PREFIX}'
    )
