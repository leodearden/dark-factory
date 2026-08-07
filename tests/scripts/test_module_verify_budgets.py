"""Every module config must declare its OWN warm verify budget, sized from measurement.

Task 3473. ``verify_command_timeout_secs`` is a PER-COMMAND budget, and
``verify.run_full_verification`` ``asyncio.gather``s over
``module_configs.values()`` — so each module's budget is genuinely independent
of every other module's and of the repo root's. A module that declares none
does NOT get something "inherit-ish": per ``verify._resolve_verify_timeout``'s
cascade it is a hard fall-through to ``config.verify_command_timeout_secs``,
the whole-fleet ceiling (3600s warm). Declaring is the only way to narrow.

Before this task, seven of nine discovered module configs declared nothing, so
a hang in any of them surfaced only after the whole-fleet ceiling — about an
hour rather than the minutes the root yaml's own "tight timeouts surface hangs"
intent claims. This guard closes six of those seven and REPORTS the remaining
one (``orchestrator``, task 3353) through ``MODULE_BUDGET_EXCLUSIONS`` rather
than hiding it.

GENERALISED FROM, not imported from, two sibling guards:
``tests/scripts/test_scripts_module_config.py::
test_scripts_module_carries_its_own_measured_verify_budget`` (task 3458) and
its ancestor ``tests/scripts/test_tests_scripts_module_config.py::
test_tests_scripts_module_carries_its_own_tight_verify_budget`` (task 3350).
Their scalar ``MEASURED_SUITE_WORST_SECS`` / ``MIN_MODULE_BUDGET_SECS`` pair is
promoted here to per-module tables and the single test to a parametrization.

HELPERS ARE COPIED, NOT IMPORTED, deliberately. A test file importing a sibling
test file couples two guards that must be able to fail independently, and
``_root_config`` in particular carries a load-bearing ``ORCH_CONFIG_PATH``
anchor whose absence silently collapses every config value to pydantic
defaults — it must be visibly present in the file that depends on it rather
than inherited from a neighbour.

PLACEMENT. ``tests/scripts/`` rather than a per-module ``tests/`` dir, and NOT
appended to ``test_fallback_verify_config.py``. That file is listed in task
3353's ``files``, so editing it manufactures lock contention with a pending
sibling for no benefit. ``tests/scripts/`` is nonetheless the right directory
for the reason both sibling guards already document: it is ADDITIONALLY covered
by its own registered module config (``tests/scripts/orchestrator.yaml``), so a
guard living here cannot be silenced by an edit to the very command it asserts
about.

Cited by SYMBOL, never by file:line. Both sibling guards record that their own
line pins had all rotted at HEAD and sent operators to unrelated code; a stale
pin is worse than no pin because it reads as authoritative.
"""
from __future__ import annotations

import pathlib

import pytest

from orchestrator import verify, verify_plan
from orchestrator.config import OrchestratorConfig, _discover_module_configs

REPO_ROOT = pathlib.Path(__file__).parents[2]

# This worktree's own top-level orchestrator config. ``dark-factory-orchestrator
# .yaml`` is the canonical, REQUIRED filename for a project's top-level config;
# the legacy spellings are a discovery fallback for unmigrated projects, not a
# choice this repo has. Anchored to REPO_ROOT rather than taken from the ambient
# ORCH_CONFIG_PATH, for the reason ``_root_config``'s docstring records at length.
DF_CONFIG_PATH = REPO_ROOT / 'dark-factory-orchestrator.yaml'


# MEASURED wall-clock, in seconds: the WORST run of each module's own VERBATIM
# test_command, copied byte-for-byte out of that module's orchestrator.yaml.
#
# WHERE THE PROVENANCE LIVES — deliberately NOT here. Each module's own
# orchestrator.yaml carries the full per-run table (every run's wall-clock,
# loadavg, rc and pytest counts, convenient and inconvenient alike) in its
# PER-MODULE VERIFY BUDGET block, and is the SINGLE durable home for those
# numbers; .task/measurements.md holds the raw sweep log but is worktree-local
# scratch and does not survive. Only the scalar W is duplicated here, and only
# because the guard computes ``_min_budget(W)`` from it — which is exactly why
# every one of those yaml blocks carries a RAISE-TOGETHER rule naming its entry
# in this table. Reproducing the per-run tables here as well would create a
# THIRD copy to raise in lockstep, which is the drift this file already declines
# to create for the sibling-owned modules (see MEASURED_BY_SIBLING_GUARD) and
# the drift the repo-root yaml's own paragraph refuses on the same grounds.
#
# EVERY FIGURE IS CONTENDED and none is presented as an unloaded one. The sweep
# ran on a 32-core host at /proc/loadavg 72-252 throughout — roughly 2x to 8x
# oversubscribed — and the norm at orchestrator/tests/warm-lane/README.md (whose
# own argument had to abandon wall-clock for load-independent counters, calling
# it "nearly unusable" on this box) forbids quoting such a figure as unloaded.
# Contended figures are ACCEPTABLE here specifically because the error runs in
# the SAFE direction: an inflated worst-case yields a LARGER budget, and the
# failure this task exists to prevent is an UNDER-sized ceiling.
#
# SIZED AGAINST THE MAX, NEVER THE MEAN. scripts/orchestrator.yaml records a ~3x
# spread (310.33s -> 914.61s) for a BYTE-IDENTICAL command on a BYTE-IDENTICAL
# tree; that is host load at measurement time, not suite variance, and a mean
# would have under-sized every budget here.
#
# CAVEATS THAT CHANGE HOW A NUMBER MUST BE READ are kept here rather than left
# only in the yamls, because they bear on whether this table may be reused at
# all. Everything else — the run-by-run figures each is drawn from — is in the
# owning yaml.
#
#   fused-memory  (i) NO GREEN RUN EXISTS. All six runs are rc=1 on the
#                 PRE-EXISTING failure escalation esc-3473-1 records, red at
#                 base main 5a7770d239 before this task touched anything. The
#                 sizing rule says "worst across recorded GREEN runs"; here it
#                 is applied to RED runs, on the grounds that the failure is one
#                 fast assertion in a suite that ran to COMPLETION (no -x) and
#                 adds no measurable wall-clock. An assumption with an error bar,
#                 not a clean measurement; the 2.6x multiple absorbs it.
#                 (ii) The ONLY in-scope module under pytest-xdist (addopts =
#                 "-n auto --dist loadgroup -m 'not integration'"). W is an XDIST
#                 figure, so a future change to -n or to the dist mode
#                 INVALIDATES it rather than merely shifting it — RE-MEASURE, do
#                 not scale.
#
#   dashboard     NO PRIOR FIGURE OF ANY KIND existed for this suite (task 3062
#                 attempt-2 timed out before dashboard's segment started), so
#                 unlike shared/escalation/fused-memory there is no independent
#                 earlier measurement to reconcile W against. Widest spread in
#                 the sweep, 2.3x. The error bar is wider here; stated, not
#                 implied.
#
#   escalation    W is a SERIAL run, LARGER than either concurrent-wave run.
#                 Sizing against the wave alone — on the reasoning that
#                 concurrency must be the worst case — would have UNDER-sized
#                 this module. The rule is the worst RUN, not the harshest
#                 SWEEP.
#
#   cockpit       W is taken from a GREEN run. One run in the sweep was rc=1
#                 with an unidentified failure (its log was lost to a concurrent
#                 driver collision); nothing here is sized from it.
#
#   sampler       MAKES ASSERTION (b)'s DEGENERACY CONCRETE, so it is stated
#                 here as well as in the test docstring: `_min_budget(22.49)` ==
#                 `(int(44.98) // 100) * 100` == 0, so (b) asserts `budget >= 0`
#                 for sampler — literally unfalsifiable. What carries this module
#                 is (a) declared-at-all, (c) strictly-below-root, (d) honoured
#                 by the real resolver, (e) survives the plan->execution bridge
#                 and (f) the cold fall-through. Do not infer a measurement floor
#                 here that does not exist.
#
# ONE METHODOLOGICAL LIMIT ON THE WHOLE TABLE. The concurrent arm of the sweep
# was a 6-WAY wave (the six modules this task closed). Production's
# ``verify.run_full_verification`` gathers over EVERY registered module config —
# nine today, including the dominant `orchestrator` segment — so the measured
# wave is a LOWER BOUND on the fully-concurrent cost, not a reproduction of it.
# See the same note in each yaml's provenance block for what bounds the gap.
MEASURED_MODULE_SUITE_WORST_SECS: dict[str, float] = {
    'shared': 219.08,
    'escalation': 354.56,
    'fused-memory': 439.80,
    'dashboard': 653.54,
    'sampler': 22.49,
    'cockpit': 130.50,
}

# One real tracked file under each module prefix, used to drive the production
# plan->execution bridge in assertion (e). Deliberately a PRODUCTION file rather
# than a test file: verify_plan._derive_module_runs routes a source-only diff
# through the task-3294 owning-module floor, which is the arm that runs the
# declared command verbatim.
SAMPLE_TOUCHED_FILE: dict[str, str] = {
    'shared': 'shared/src/shared/__init__.py',
    'escalation': 'escalation/src/escalation/__init__.py',
    'fused-memory': 'fused-memory/src/fused_memory/__init__.py',
    'dashboard': 'dashboard/src/dashboard/__init__.py',
    'sampler': 'sampler/src/sampler/__init__.py',
    'cockpit': 'cockpit/src/cockpit/__init__.py',
}


def _min_budget(worst: float) -> int:
    """~2x the worst measured run, rounded DOWN to the nearest 100s.

    The EXACT expression task 3458 uses for
    ``test_scripts_module_config.MIN_MODULE_BUDGET_SECS``, reused verbatim so
    the two guards cannot silently drift in SHAPE. DERIVED from the
    measurement rather than hand-set, because that exact pair has already
    rotted once undetected: the tests/scripts sibling still hard-codes its
    floor against a 127.0s worst run while its own yaml has since recorded
    233.50s, and nothing caught it.

    DEGENERATES TO ZERO for cheap suites — see the parametrized test's
    docstring, where the consequence is spelled out rather than left for a
    reader to discover.
    """
    return (int(2 * worst) // 100) * 100


def _discovered() -> dict:
    """Every module config the PRODUCTION walk registers, keyed by repo-relative prefix.

    Delegates to ``config._discover_module_configs`` rather than globbing
    ``**/orchestrator.yaml``. A hand-rolled glob run from the main checkout
    would descend ``.worktrees/`` and ``.venv/``, and would remain free to
    drift from what the orchestrator actually registers; delegating inherits
    the production pruning of ``.worktrees``, ``.venv``, ``node_modules``,
    ``build``, ``target``, ``.claude`` and nested ``.git`` checkouts, and
    covers configs at ANY depth (``tests/scripts``, not just depth-1 names).
    """
    return _discover_module_configs(REPO_ROOT)


def _root_config(monkeypatch: pytest.MonkeyPatch) -> OrchestratorConfig:
    """Load the repo-root config through the PRODUCTION loader, anchored at DF_CONFIG_PATH.

    ANCHORING ``ORCH_CONFIG_PATH`` IS LOAD-BEARING, not hygiene. ``project_root``
    is only a model FIELD and selects nothing:
    ``OrchestratorConfig.settings_customise_sources`` builds its
    ``YamlSettingsSource`` from ``os.environ['ORCH_CONFIG_PATH']`` alone,
    falling back to a CWD-relative ``config.yaml``. Both ambient states are
    wrong here, in OPPOSITE directions:

      * UNSET — the state INSIDE VERIFY, because
        ``verify._target_subprocess_env`` deliberately scrubs the whole
        ``ORCH_`` prefix — finds no file, so every value collapses to the
        pydantic DEFAULTS. Assertion (c) below would then compare each module
        budget against a default this repo does not declare.
      * SET, as an operator's shell has it, points at whichever checkout that
        orchestrator serves — typically the MAIN one, not this worktree. Every
        assertion would then be about a different checkout's yaml and report
        GREEN on a worktree that had actually regressed.

    Setting the env var IS the production load path (``config.load_config``
    stamps it before constructing), so this stays a read through the real
    loader, pinned to THIS worktree's committed yaml.

    Fails LOUDLY on a missing file: ``YamlSettingsSource`` SKIPS a non-existent
    ``config_path`` rather than raising, so a bad path would silently yield the
    pydantic defaults instead of an error.
    """
    assert DF_CONFIG_PATH.is_file(), (
        f'{DF_CONFIG_PATH} does not exist, so anchoring ORCH_CONFIG_PATH at it '
        'would silently load the pydantic DEFAULTS instead (YamlSettingsSource '
        'skips a non-existent path rather than raising), and every value read '
        'from the returned config would be about a config this repo does not '
        'declare. dark-factory-orchestrator.yaml is the canonical, required '
        "filename for a project's top-level orchestrator config"
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(DF_CONFIG_PATH))
    return OrchestratorConfig(project_root=REPO_ROOT)


def _executed_for_touched(prefix: str, files: list[str], cfg: OrchestratorConfig):
    """Run the PRODUCTION plan->execution bridge and return the single executed config.

    ``derive_verify_plan`` decides scope; ``_executed_module_configs_from_plan``
    renders those PlannedRuns into the exact ModuleConfig ``run_verification``
    executes. Asserting on THAT is what makes "the budget survives to
    execution" a structural claim rather than a claim about the yaml.

    *cfg* IS A REQUIRED PARAMETER, not a convenience. It must be a config built
    by ``_root_config``, whose docstring spells out why the ``ORCH_CONFIG_PATH``
    anchor is load-bearing: an unset anchor collapses every value to the
    pydantic defaults, SILENTLY. This helper used to construct its own
    ``OrchestratorConfig(project_root=REPO_ROOT)`` and read the right yaml only
    because its caller happened to have called ``_root_config`` earlier in the
    same test body, leaving the env var set as a SIDE EFFECT. Reordering the
    assertions, or calling this helper from a new test, would have handed it a
    defaults-collapsed config with no failure signal. Taking the config as an
    argument makes the dependency structural instead of ordering-dependent.

    The ``lambda _f: None`` worktree_reader keeps this hermetic: no file reads,
    and nothing classifies STRUCTURAL.
    """
    mc = _discovered()[prefix]
    plan = verify_plan.derive_verify_plan(files, [mc], cfg, lambda _f: None)
    executed = verify._executed_module_configs_from_plan([mc], plan)
    assert len(executed) == 1, (
        f'expected exactly one executed module config for {files!r}, got '
        f'{[e.prefix for e in executed]!r}'
    )
    return executed[0]


@pytest.mark.parametrize('prefix', sorted(MEASURED_MODULE_SUITE_WORST_SECS))
def test_module_carries_its_own_measured_verify_budget(
    prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each measured module declares a warm budget, narrower than the repo-root ceiling.

    Six assertions, one contract per module:

    (a) DECLARED AT ALL. Otherwise ``verify._resolve_verify_timeout`` falls
        hard through to the repo-root whole-fleet ceiling — a budget sized for
        nine suites chained into one shell command, applied to one suite.
    (b) At least ``_min_budget(W)``, ~2x the worst measured run.
    (c) STRICTLY below the repo-root warm ceiling: a real narrowing rather
        than a relabelling. A per-module budget at or above the global one
        surfaces a hang no sooner than the fleet ceiling already would.
    (d) ``verify._resolve_verify_timeout`` actually honours it warm —
        exercising the REAL precedence mechanism rather than restating it, so
        the assertion cannot drift from the code that implements it.
    (e) It survives the PRODUCTION plan->execution bridge. That bridge uses
        ``dataclasses.replace`` today, but ``verify._apply_cargo_scope`` on
        the same bridge reconstructs ModuleConfig by HAND-LISTING every field
        — precisely the shape where a per-module knob gets silently dropped by
        a future field addition.
    (f) A COLD verify does NOT get this warm value. Per
        ``_resolve_verify_timeout``'s cascade an unset module cold knob falls
        through to ``config.verify_cold_command_timeout_secs``, NOT to the
        module's warm budget. This is exactly the misreading the sibling guard
        was written to make un-silent, and it is folded in here so six modules
        get it for the cost of one assertion.

    SCOPE, STATED HONESTLY — this guard's advertised reach must equal its real
    reach, because a guard that overstates itself is the same defect wearing a
    test's clothes (the standard
    ``test_fallback_verify_budget_clears_the_measured_fleet_chain_floor``
    already sets in this directory).

    * This is a floor-REGRESSION guard, NOT a suite-growth detector. Nothing
      here re-measures anything: ``MEASURED_MODULE_SUITE_WORST_SECS`` is a
      frozen table, so if a suite doubles tomorrow the constant still passes.
      Detecting that requires RE-MEASUREMENT, which a table cannot do to
      itself.
    * ASSERTION (b) DEGENERATES FOR CHEAP SUITES AND IS VACUOUS FOR SOME
      MODULES HERE. ``_min_budget`` floors to the nearest 100s, so any suite
      whose worst run is under 50s yields a floor of 0 and (b) asserts
      ``budget >= 0`` — literally unfalsifiable. sampler's worst run is
      22.49s, giving exactly that. For such modules the load-bearing
      assertions are (a), (c), (d), (e) and (f); (b) carries nothing. Two
      dishonest repairs were rejected: clamping the floor to a hand-set
      minimum would destroy the DERIVED-not-hand-set property task 3458
      introduced precisely because a hand-coded pair had already rotted, and
      dropping (b) for cheap modules would make the parametrization
      non-uniform.
    """
    worst = MEASURED_MODULE_SUITE_WORST_SECS[prefix]
    discovered = _discovered()

    assert prefix in discovered, (
        f'{prefix}/orchestrator.yaml is not discovered by the production '
        f'config._discover_module_configs walk (task 3473), so nothing below '
        f'can bound its verify. Discovered: {sorted(discovered)}'
    )
    mc = discovered[prefix]

    # (a) Declared at all — otherwise the whole-fleet ceiling silently applies.
    assert mc.verify_command_timeout_secs is not None, (
        f'{prefix}/orchestrator.yaml declares no verify_command_timeout_secs '
        f'(task 3473), so this module silently inherits the repo-root '
        f'whole-fleet ceiling. That is not an "inherit-ish" default: '
        f'verify._resolve_verify_timeout falls HARD through to '
        f'config.verify_command_timeout_secs, the budget sized for nine '
        f'subprojects chained into ONE shell command, applied to a suite whose '
        f'own worst measured run is {worst}s. A hang here would surface after '
        f'the fleet ceiling (~an hour), not in minutes'
    )

    # (b) Measurement-derived floor. VACUOUS where _min_budget degenerates to
    # 0 (any suite under ~50s) — see the docstring; not silently so.
    floor = _min_budget(worst)
    assert mc.verify_command_timeout_secs >= floor, (
        f'{prefix} verify_command_timeout_secs='
        f'{mc.verify_command_timeout_secs} is below the {floor}s floor derived '
        f'from its worst measured run ({worst}s, CONTENDED — see '
        f'.task/measurements.md and the module yaml\'s provenance block). A '
        f'budget under the floor would manufacture infra_timeout on the honest '
        f'green path — the exact defect task 3350 exists to remove, '
        f'reintroduced one level down'
    )

    # (c) Strictly tighter than the repo-root ceiling: a real narrowing. Read
    # through the PRODUCTION loader (see _root_config's docstring for why the
    # ORCH_CONFIG_PATH anchoring is load-bearing rather than hygiene).
    cfg = _root_config(monkeypatch)
    root_warm = cfg.verify_command_timeout_secs
    assert mc.verify_command_timeout_secs < root_warm, (
        f'{prefix} verify_command_timeout_secs='
        f'{mc.verify_command_timeout_secs} is not strictly below the repo-root '
        f'verify_command_timeout_secs={root_warm} (task 3473). A per-module '
        f'budget at or above the global one is a relabelling, not a narrowing: '
        f'it surfaces a hang no sooner than the whole-fleet ceiling would'
    )

    # (d) The REAL precedence mechanism honours it, warm.
    resolved = verify._resolve_verify_timeout(cfg, mc, is_cold=False)
    assert resolved == mc.verify_command_timeout_secs, (
        f'_resolve_verify_timeout returned {resolved} for a warm verify of '
        f'{prefix}, not the declared module budget '
        f'{mc.verify_command_timeout_secs} (task 3473) — the per-module '
        f'override is not reaching the code path that enforces it'
    )

    # (e) Survives the PRODUCTION plan -> execution bridge, not just the
    # resolver in isolation. See the docstring for why _apply_cargo_scope makes
    # this a live risk rather than a formality.
    executed = _executed_for_touched(prefix, [SAMPLE_TOUCHED_FILE[prefix]], cfg)
    assert executed.verify_command_timeout_secs == mc.verify_command_timeout_secs, (
        f'the production plan->execution bridge '
        f'(verify._executed_module_configs_from_plan) rendered '
        f'verify_command_timeout_secs={executed.verify_command_timeout_secs} '
        f'for a {prefix}-scoped run, not the declared module budget '
        f'{mc.verify_command_timeout_secs} (task 3473) — the per-module '
        f'override is dropped somewhere between discovery and execution'
    )

    # (f) The warm knob deliberately does NOT cover cold. An unset module cold
    # knob falls through to the ROOT cold ceiling, not to this warm value.
    assert mc.verify_cold_command_timeout_secs is None, (
        f'{prefix}/orchestrator.yaml now declares '
        f'verify_cold_command_timeout_secs='
        f'{mc.verify_cold_command_timeout_secs} (task 3473 deliberately left '
        f'this UNSET: none of these six modules has a build step whose cold '
        f'cost would justify a separate cold budget). If that was set '
        f'intentionally, this assertion and the module yaml\'s cold-cascade '
        f'note must be updated together, not just this line'
    )
    root_cold = cfg.verify_cold_command_timeout_secs
    resolved_cold = verify._resolve_verify_timeout(
        cfg, mc, is_cold=True, is_merge_verify=False
    )
    assert resolved_cold == root_cold, (
        f'_resolve_verify_timeout(is_cold=True) returned {resolved_cold} for '
        f'{prefix}, not the repo-root verify_cold_command_timeout_secs='
        f'{root_cold} (task 3473). The module leaves its own cold knob UNSET '
        f'deliberately, so a cold verify must fall through to the root ceiling'
    )
    assert resolved_cold != mc.verify_command_timeout_secs, (
        f'_resolve_verify_timeout(is_cold=True) returned {prefix}\'s WARM '
        f'budget ({mc.verify_command_timeout_secs}) rather than falling '
        f'through to the root cold ceiling (task 3473) — this is precisely the '
        f'misreading assertion (f) exists to make un-silent: an unset cold '
        f'knob does NOT "inherit" the warm value'
    )


# The floor set: module configs known to exist at base main 5a7770d239 that the
# production walk must still resolve. Same purpose as
# KNOWN_PER_MODULE_CONFIG_NAMES in test_fallback_verify_config.py — if discovery
# silently regresses, the coverage loop below would pass VACUOUSLY on a shrunken
# set, so it is asserted rather than trusted. Includes the six this task closed
# plus the two sibling tasks 3350/3458 closed and the one excluded below.
KNOWN_MODULE_CONFIG_PREFIXES = frozenset(
    {
        'shared',
        'escalation',
        'fused-memory',
        'dashboard',
        'sampler',
        'cockpit',
        'orchestrator',
        'scripts',
        'tests/scripts',
    }
)


# Modules whose budget IS measurement-backed, but whose figures are recorded in
# the sibling guard or yaml that owns them rather than in this file's table.
# Copying those numbers here would create a second copy of a measurement that
# must be raised in lockstep — exactly the drift that has already occurred once
# between these two configs and was caught only by a reviewer.
#
# These entries are NOT a statement that the owning guard is CURRENT. They
# record only WHERE a module's provenance lives, so this file does not become a
# second home for it; see the tests/scripts entry, which names a known staleness
# in its own owner rather than papering over it.
MEASURED_BY_SIBLING_GUARD: dict[str, str] = {
    'scripts': (
        'MEASURED_SUITE_WORST_SECS = 930.59 in '
        'tests/scripts/test_scripts_module_config.py (task 3458), sourced from '
        "scripts/orchestrator.yaml's own PER-MODULE VERIFY BUDGET block, whose "
        'floor is DERIVED from that constant by the same expression _min_budget '
        'uses.'
    ),
    'tests/scripts': (
        'MEASURED worst run 233.50s recorded in tests/scripts/orchestrator.yaml. '
        'NOTE THE OWNING GUARD IS STALE: '
        'tests/scripts/test_tests_scripts_module_config.py (task 3350) still '
        'pins MEASURED_SUITE_WORST_SECS = 127.0 with a HAND-SET '
        'MIN_MODULE_BUDGET_SECS = 300, so the 233.50s figure in the yaml is '
        'gated by nothing — this is the exact rot _min_budget\'s docstring '
        'names, and it is why task 3458 made the floor derived rather than '
        'hand-set. Excluded here to avoid creating a SECOND copy of the '
        'measurement, NOT because the sibling floor is current. Fixing that '
        'guard is not task 3473\'s scope (it does not hold the lock).'
    ),
}


# Documented carve-outs from the coverage guard below: prefix -> justification.
# Modelled on TIMEOUT_GUARD_EXCLUSIONS in test_fallback_verify_config.py, but a
# dict rather than a frozenset so each justification lives AT THE DEFINITION
# SITE rather than in a detached comment that drifts from the entry it excuses.
# That is documentation, not an enforced property — see the coverage guard's
# docstring for why no in-file assertion could enforce it.
#
# THIS IS THE ONE REMAINING GAP in the repo-root yaml's claim that "tight
# timeouts surface hangs ... on the PER-MODULE budgets". Nine of ten configs
# now declare their own; this is the tenth. Recorded here so the guard REPORTS
# that gap on every run rather than hiding it.
MODULE_BUDGET_EXCLUSIONS: dict[str, str] = {
    'orchestrator': (
        'Owned by task 3353, which is still pending. Deliberately not fixed by '
        'task 3473 to avoid scope creep onto files 3473 does not lock. The '
        'exclusion is TECHNICALLY correct, not merely deferential: orchestrator '
        'is the dominant fleet segment (measured 1366.23s and 1157.62s on one '
        'day, ~18% apart, in task 3062 attempt-2 / esc-3062-3), and task 3353 '
        "has since been rewritten to \"split or shard the orchestrator test "
        'suite\" — so a budget declared now against an unsharded 1366s figure '
        "would be invalidated by 3353's own change rather than merely refined "
        'by it. DELETION CONDITION: once orchestrator/orchestrator.yaml declares '
        'its own verify_command_timeout_secs, delete this entry — the coverage '
        'guard then covers it automatically with no further edit.'
    ),
}


def test_every_discovered_module_config_declares_its_own_verify_budget() -> None:
    """Every module config defining a test_command must declare a warm budget.

    THE ANTI-DRIFT HALF of this file, and the generalisation task 3473 exists
    to make. The parametrized guard above covers exactly the six modules whose
    figures are in ``MEASURED_MODULE_SUITE_WORST_SECS``; nothing there notices
    a NEWLY-REGISTERED module config landing with no budget at all. This
    walks what discovery actually registers, so a new subproject is covered
    automatically with no edit here.

    Written FRESH rather than generalised from an existing guard. Task 3473's
    description points at ``test_the_dominant_fleet_segment_declares_its_own_
    verify_budget`` as the thing to widen; that symbol DOES NOT EXIST anywhere
    in this repo (task 3353, which would have written it, is still pending and
    its scope has since been rewritten to sharding the orchestrator suite), so
    there was nothing to generalise FROM.

    Modelled instead on ``TIMEOUT_GUARD_EXCLUSIONS`` in
    ``test_fallback_verify_config.py`` — a documented, single-entry carve-out
    carrying its own stated deletion condition, consulted inside a
    dynamic-discovery loop. WIDENED from that ``frozenset`` to a
    ``dict[str, str]`` so each justification lives AT THE DEFINITION SITE,
    next to the prefix it excuses, rather than in a detached comment that
    drifts from the entry it describes. That is DOCUMENTATION, not an enforced
    property: nothing here checks the reason, and deliberately so. Whether an
    exclusion is legitimately owned is settled in REVIEW, by a human who can
    tell whether the named task is real, open, and actually owns the module.
    No in-file assertion could do that — a regex over this file's own string
    literals cannot verify a task exists (``task 0`` would satisfy it), which
    would make the check read as enforcement while enforcing nothing.

    THE GUARD REPORTS THE REMAINING GAP RATHER THAN HIDING IT. Exactly one
    entry is excluded today (``orchestrator``, task 3353), and that is the one
    place where the repo-root yaml's "tight timeouts live on the per-module
    budgets" claim is still false.

    AND THE CARVE-OUT EXPIRES ON ITS OWN TERMS. Each entry states a deletion
    condition; this test enforces it, by asserting an excluded module still
    declares NO budget. Without that, the day the owning task lands, the
    ``continue`` would skip the prefix permanently — past the declared-at-all
    check AND past the ``MEASURED_MODULE_SUITE_WORST_SECS`` provenance check —
    so a budget could land for the dominant fleet segment with nothing recorded
    behind it while this file stayed green and this docstring stayed wrong. A
    dead-key check over all three prefix-keyed tables closes the mirror-image
    hole: a renamed or removed module leaving an entry that suppresses coverage
    for a prefix that no longer exists.

    Both are assertions about CONFIG STATE, not about the prose of a string
    literal in this file — the distinction that governs what belongs in a guard
    here at all.
    """
    discovered = {
        prefix: mc
        for prefix, mc in _discovered().items()
        if mc.test_command
    }

    # Floor: discovery must still resolve the known configs. Without this, a
    # regression in the production walk would shrink the set and every
    # assertion below would pass vacuously on what remained.
    missing = KNOWN_MODULE_CONFIG_PREFIXES - set(discovered)
    assert not missing, (
        f'the production walk (config._discover_module_configs, filtered to '
        f'configs defining a test_command) failed to resolve known module '
        f'config(s) {sorted(missing)} (task 3473) — discovery has regressed, '
        f'and the coverage loop below would pass vacuously on the shrunken '
        f'set. Discovered: {sorted(discovered)}'
    )

    # DEAD KEYS FAIL LOUDLY. Every table in this file is keyed by a module
    # prefix, and a module that is RENAMED or REMOVED leaves an entry behind
    # that excuses nothing, records nothing, and breaks nothing — so nothing
    # ever prompts its deletion. A stale MODULE_BUDGET_EXCLUSIONS key is the
    # worst of the three: it would go on suppressing coverage for a prefix that
    # no longer exists while reading, to anyone auditing the file, as a live and
    # justified carve-out.
    tabled = (
        set(MODULE_BUDGET_EXCLUSIONS)
        | set(MEASURED_BY_SIBLING_GUARD)
        | set(MEASURED_MODULE_SUITE_WORST_SECS)
    )
    dead = tabled - set(discovered)
    assert not dead, (
        f'module prefix(es) {sorted(dead)} appear in this file\'s tables '
        f'(MODULE_BUDGET_EXCLUSIONS / MEASURED_BY_SIBLING_GUARD / '
        f'MEASURED_MODULE_SUITE_WORST_SECS) but are no longer registered by '
        f'config._discover_module_configs with a test_command (task 3473). '
        f'Either the module was renamed or removed — in which case delete the '
        f'entries — or discovery has regressed. Discovered: {sorted(discovered)}'
    )

    for prefix, mc in sorted(discovered.items()):
        reason = MODULE_BUDGET_EXCLUSIONS.get(prefix)
        if reason is not None:
            # THE CARVE-OUT MUST STILL BE NEEDED. Every entry states its own
            # deletion condition — "once <prefix>/orchestrator.yaml declares its
            # own verify_command_timeout_secs, delete this entry" — and until
            # this assertion existed, nothing enforced it. When the owning task
            # landed, `continue` would have kept skipping the prefix forever, so
            # it would have escaped BOTH the declared-at-all assertion and the
            # measurement-provenance assertion below: a budget could land for
            # the dominant fleet segment with no recorded measurement behind it
            # and no guard, while this file still reported green and its
            # docstring still claimed to REPORT the remaining gap.
            #
            # This asserts on real config state, not on the justification's
            # prose — the distinction the amendment removing the `task \d+`
            # regex turned on.
            assert mc.verify_command_timeout_secs is None, (
                f'{prefix} now declares '
                f'verify_command_timeout_secs={mc.verify_command_timeout_secs}, '
                f'so its MODULE_BUDGET_EXCLUSIONS entry is STALE and has met '
                f'its own stated deletion condition (task 3473). Delete the '
                f'entry: the coverage guard then covers this module '
                f'automatically, which also makes the '
                f'MEASURED_MODULE_SUITE_WORST_SECS provenance requirement below '
                f'apply to it — the point of deleting it rather than leaving a '
                f'harmless-looking key in place. Excluded for: {reason}'
            )
            continue

        assert mc.verify_command_timeout_secs is not None, (
            f'{prefix}/orchestrator.yaml defines a test_command but declares no '
            f'verify_command_timeout_secs (task 3473), so it silently inherits '
            f'the repo-root whole-fleet ceiling — verify._resolve_verify_timeout '
            f'falls HARD through to config.verify_command_timeout_secs. Either '
            f'MEASURE this module\'s verbatim test_command and declare a budget '
            f'(see any of the six closed by task 3473 for the provenance shape), '
            f'or add a justified entry to MODULE_BUDGET_EXCLUSIONS naming the '
            f'task that owns it'
        )

        # A budget with no recorded measurement behind it is a guess wearing a
        # number's clothes. This is what stops a newly-registered config from
        # landing with a plausible-looking value and no provenance.
        #
        # MEASURED_BY_SIBLING_GUARD is not a loophole: those modules' figures
        # ARE recorded, just in the guard that owns them rather than in this
        # file's table. Requiring them here would demand a DUPLICATE copy of a
        # measurement, which is precisely the drift the RAISE-TOGETHER rule in
        # every provenance block exists to prevent — and the drift that has
        # already happened once between the two sibling configs.
        if prefix in MEASURED_BY_SIBLING_GUARD:
            continue
        assert prefix in MEASURED_MODULE_SUITE_WORST_SECS, (
            f'{prefix}/orchestrator.yaml declares '
            f'verify_command_timeout_secs={mc.verify_command_timeout_secs} but '
            f'has no entry in MEASURED_MODULE_SUITE_WORST_SECS (task 3473), so '
            f'nothing records what it was sized from. Measure the module\'s own '
            f'VERBATIM test_command, record the runs with their loadavg in the '
            f'table above and in the yaml\'s provenance block, then add the '
            f'worst wall-clock here'
        )
