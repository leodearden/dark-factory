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

CORRECTED IN PLACE (task 4320), never silently rewritten. This header used to
read: "HELPERS ARE COPIED, NOT IMPORTED, deliberately. A test file importing a
sibling test file couples two guards that must be able to fail independently,
and ``_root_config`` in particular carries a load-bearing ``ORCH_CONFIG_PATH``
anchor whose absence silently collapses every config value to pydantic
defaults — it must be visibly present in the file that depends on it rather
than inherited from a neighbour."

BOTH HALVES OF THAT ARE NOW FALSE, and they were false in different ways.

The FIRST half over-generalised a sound argument. What the argument actually
supports is narrower: a test file must not import a SIBLING TEST FILE, because
that couples two guards which have to be able to fail independently and lets a
regression in one silence the other. It does not reach a shared NON-TEST
module, and it never reached ``conftest.py`` — pytest's idiomatic home for
exactly this. Generalised to "helpers are copied", it licensed the triplication
task 4320 removed: three verbatim ``_root_config`` helpers and three spellings
of one budget derivation, none of which any mechanism could see drift between.
This file now imports ``min_budget``, ``FAMILY_PUBLISHER_PATHS`` and
``published_pairs`` from ``module_budget_family`` — a non-test sibling, on the
``setup_host_sections`` / ``systemd_unit_invariants`` precedent already
established in this directory — and every guard in the family still fails
entirely on its own.

The SECOND half named the wrong remedy for a real hazard. The
``ORCH_CONFIG_PATH`` anchor IS load-bearing exactly as described, and its
absence does silently collapse every config value to the pydantic defaults. But
VISIBILITY was never what enforced it — three copies is three places for the
anchor to be dropped from, and one of them had already been drafted without it.
It now lives once, as the ``root_config`` fixture in ``tests/scripts/
conftest.py``, where it is stronger than a copied helper in two ways: fixture
setup runs BEFORE the test body, so the anchor-then-poison ordering two guards
in this family depend on is structural rather than a comment; and
``test_root_config_fixture_anchors_this_worktrees_own_yaml`` below pins the
behaviour the move had to preserve, which no copy ever did.

READING A SIBLING IS NOT IMPORTING IT, and this file now does the former on
purpose. ``published_pairs`` ``ast``-parses each publisher's SOURCE rather than
importing it, because importing would hand back only the already-evaluated
``MIN_MODULE_BUDGET_SECS`` int — which is precisely what cannot tell a
derivation from a literal that happens to agree with it today. So the
"GENERALISED FROM, not imported from" framing above still holds literally, and
is now also load-bearing in a second sense: no publisher is imported, no
publisher imports this file, and the coupling runs only through source text
this guard reads and can report on by name.

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

import ast
import os
import pathlib

import pytest
import yaml
from module_budget_family import (
    FAMILY_PUBLISHER_PATHS,
    FAMILY_READER_PATH,
    min_budget,
    published_pairs,
)
from orchestrator.config import OrchestratorConfig, _discover_module_configs

from orchestrator import verify, verify_plan

REPO_ROOT = pathlib.Path(__file__).parents[2]

# MEASURED wall-clock, in seconds: the WORST run of each module's own VERBATIM
# test_command, copied byte-for-byte out of that module's orchestrator.yaml.
#
# WHERE THE PROVENANCE LIVES — deliberately NOT here. Each module's own
# orchestrator.yaml carries the full per-run table (every run's wall-clock,
# loadavg, rc and pytest counts, convenient and inconvenient alike) in its
# PER-MODULE VERIFY BUDGET block, and is the SINGLE durable home for those
# numbers; .task/measurements.md holds the raw sweep log but is worktree-local
# scratch and does not survive. Only the scalar W is duplicated here, and only
# because the guard computes ``min_budget(W)`` from it — which is exactly why
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
#                 here as well as in the test docstring: `min_budget(22.49)` ==
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


def _executed_for_touched(prefix: str, files: list[str], cfg: OrchestratorConfig):
    """Run the PRODUCTION plan->execution bridge and return the single executed config.

    ``derive_verify_plan`` decides scope; ``_executed_module_configs_from_plan``
    renders those PlannedRuns into the exact ModuleConfig ``run_verification``
    executes. Asserting on THAT is what makes "the budget survives to
    execution" a structural claim rather than a claim about the yaml.

    *cfg* IS A REQUIRED PARAMETER, not a convenience. It must be the config
    built by the directory-wide ``root_config`` fixture in
    ``tests/scripts/conftest.py``, whose docstring spells out why the
    ``ORCH_CONFIG_PATH`` anchor is load-bearing: an unset anchor collapses every
    value to the pydantic defaults, SILENTLY. This helper used to construct its
    own ``OrchestratorConfig(project_root=REPO_ROOT)`` and read the right yaml
    only because its caller happened to have called the (then file-local)
    anchoring helper earlier in the same test body, leaving the env var set as a
    SIDE EFFECT. Reordering the
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
    prefix: str, root_config: OrchestratorConfig
) -> None:
    """Each measured module declares a warm budget, narrower than the repo-root ceiling.

    Six assertions, one contract per module:

    (a) DECLARED AT ALL. Otherwise ``verify._resolve_verify_timeout`` falls
        hard through to the repo-root whole-fleet ceiling — a budget sized for
        nine suites chained into one shell command, applied to one suite.
    (b) At least ``min_budget(W)``, ~2x the worst measured run.
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
      MODULES HERE. ``min_budget`` floors to the nearest 100s, so any suite
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

    # (b) Measurement-derived floor. VACUOUS where min_budget degenerates to
    # 0 (any suite under ~50s) — see the docstring; not silently so.
    floor = min_budget(worst)
    assert mc.verify_command_timeout_secs >= floor, (
        f'{prefix} verify_command_timeout_secs='
        f'{mc.verify_command_timeout_secs} is below the {floor}s floor derived '
        f'from its worst measured run ({worst}s, CONTENDED — see '
        f'.task/measurements.md and the module yaml\'s provenance block). A '
        f'budget under the floor would manufacture infra_timeout on the honest '
        f'green path — the exact defect task 3350 exists to remove, '
        f'reintroduced one level down'
    )

    # (c) Strictly tighter than the repo-root ceiling: a real narrowing. The
    # repo-root config arrives from the directory-wide `root_config` fixture,
    # which reads it through the PRODUCTION loader — see its docstring in
    # tests/scripts/conftest.py for why the ORCH_CONFIG_PATH anchoring is
    # load-bearing rather than hygiene.
    root_warm = root_config.verify_command_timeout_secs
    assert mc.verify_command_timeout_secs < root_warm, (
        f'{prefix} verify_command_timeout_secs='
        f'{mc.verify_command_timeout_secs} is not strictly below the repo-root '
        f'verify_command_timeout_secs={root_warm} (task 3473). A per-module '
        f'budget at or above the global one is a relabelling, not a narrowing: '
        f'it surfaces a hang no sooner than the whole-fleet ceiling would'
    )

    # (d) The REAL precedence mechanism honours it, warm.
    resolved = verify._resolve_verify_timeout(root_config, mc, is_cold=False)
    assert resolved == mc.verify_command_timeout_secs, (
        f'_resolve_verify_timeout returned {resolved} for a warm verify of '
        f'{prefix}, not the declared module budget '
        f'{mc.verify_command_timeout_secs} (task 3473) — the per-module '
        f'override is not reaching the code path that enforces it'
    )

    # (e) Survives the PRODUCTION plan -> execution bridge, not just the
    # resolver in isolation. See the docstring for why _apply_cargo_scope makes
    # this a live risk rather than a formality.
    executed = _executed_for_touched(
        prefix, [SAMPLE_TOUCHED_FILE[prefix]], root_config
    )
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
    root_cold = root_config.verify_cold_command_timeout_secs
    resolved_cold = verify._resolve_verify_timeout(
        root_config, mc, is_cold=True, is_merge_verify=False
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
# second home for it. When an owner IS known to be stale, say so in its entry
# rather than papering over it — the tests/scripts entry did exactly that until
# task 3703 fixed the owner, and now records the repair as history instead.
MEASURED_BY_SIBLING_GUARD: dict[str, str] = {
    'scripts': (
        'MEASURED_SUITE_WORST_SECS = 930.59 in '
        'tests/scripts/test_scripts_module_config.py (task 3458), sourced from '
        "scripts/orchestrator.yaml's own PER-MODULE VERIFY BUDGET block, whose "
        'floor is DERIVED from that constant by module_budget_family.min_budget, '
        'the family\'s one implementation of that expression.'
    ),
    'tests/scripts': (
        'The measured worst run is recorded in tests/scripts/orchestrator.yaml, '
        "whose PER-MODULE VERIFY BUDGET block carries every run behind it; the "
        'owning guard is tests/scripts/test_tests_scripts_module_config.py, '
        'which pins MEASURED_SUITE_WORST_SECS with MIN_MODULE_BUDGET_SECS '
        'DERIVED from it by module_budget_family.min_budget, the same one '
        'implementation. NO FIGURE IS '
        'QUOTED HERE, corrected in place by task 4320: this entry used to read '
        '"MEASURED worst run 233.50s ... which pins MEASURED_SUITE_WORST_SECS = '
        '233.50", and both numbers went stale the moment task 4320 re-measured '
        'the suite and moved that constant to a larger figure. A quoted number '
        'is a THIRD copy that has to be raised in lockstep with the yaml and '
        'the owning guard — the exact drift this dict exists to avoid, '
        'reintroduced inside the dict itself. What survives is the POINTER, '
        'which cannot go stale that way and which '
        'test_the_budget_family_derives_every_floor_from_one_canonical_expression '
        'now checks actually resolves to a published pair. HISTORY, kept '
        'because it is why this family derives rather than hand-sets: that '
        'guard USED to pin 127.0 with a hand-set 300s floor (task 3350), so '
        "the yaml's own worst-run figure was gated by nothing and only a "
        'reviewer caught it; task 3703 refreshed the constant and made the '
        'floor derived, so it is now gated.'
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


# ---------------------------------------------------------------------------
# CROSS-FILE DERIVATION (task 4320)
# ---------------------------------------------------------------------------

# The family members that PUBLISH a (measurement, derived floor) pair, expressed
# as prefixes rather than as file paths so the floor below cannot be satisfied
# by a reader that found the right NUMBER of files but the wrong ones.
# ``FAMILY_PUBLISHER_PATHS`` names the files; this names what they must turn out
# to be about. Modelled on ``KNOWN_MODULE_CONFIG_PREFIXES`` above, for the same
# reason: without it a shrunken discovery set makes every per-pair assertion
# below pass VACUOUSLY.
FAMILY_PUBLISHED_PREFIXES = frozenset({'scripts', 'tests/scripts'})


def test_the_budget_family_derives_every_floor_from_one_canonical_expression() -> None:
    """No family member may spell the floor derivation for itself (task 4320).

    THE HOLE THIS CLOSES, in the family's own words. Until this guard existed,
    ``test_tests_scripts_module_config.py``'s derivation test carried a scope
    paragraph conceding that ``_min_budget`` there "is a LOCAL copy ... nothing
    here can observe a sibling's expression: if ``test_scripts_module_config.py``
    changed its own inline derivation, every assertion below would still pass",
    and that "real cross-file enforcement — one guard reading every family
    member's published pair and checking the derivation holds for all of them —
    would be a new mechanism rather than an amendment, and is filed as a
    follow-up instead of being claimed here". This IS that follow-up. Before it,
    the derivation existed in THREE spellings — two ``def _min_budget`` copies
    and one inlined ``(int(2 * W) // 100) * 100`` — and nothing in the repo could
    observe drift between them.

    WHY IT LIVES HERE. This file is the family's designated anti-drift member:
    it already owns ``MEASURED_BY_SIBLING_GUARD`` (the table pointing at the
    owning guards) and ``KNOWN_MODULE_CONFIG_PREFIXES`` (the floor that stops a
    shrunken discovery set passing vacuously). A cross-file check belongs with
    the tables it makes checkable, not in either publisher — a publisher checking
    its sibling is exactly the guard-couples-guard shape this directory refuses.

    SEVEN ASSERTIONS, each closing a distinct way the derivation can rot:

      (1) FLOOR SET. Every path in ``FAMILY_PUBLISHER_PATHS`` exists and yields
          a pair, and the discovered prefixes are EXACTLY
          ``FAMILY_PUBLISHED_PREFIXES``. Without this the loop below passes on
          an empty or shrunken set — the vacuous-pass hole
          ``KNOWN_MODULE_CONFIG_PREFIXES`` closes for the sibling guard. It
          also pins ``FAMILY_READER_PATH`` to THIS file, so the shared module's
          record of where the family's only cross-file enforcement lives cannot
          rot into a pointer at nothing — the same stale-pointer failure (7)
          closes for ``MEASURED_BY_SIBLING_GUARD``, applied to the constant
          that names this guard.

      (2) THE MEASUREMENT IS A MEASUREMENT. ``MEASURED_SUITE_WORST_SECS`` must
          be a plain numeric literal, not a computed expression. A measurement
          that is itself derived from something has no provenance a yaml block
          can record, and would let the pair be satisfied by two expressions
          that agree with each other and with no observed run.

      (3) THE FLOOR IS DERIVED, NOT TRANSCRIBED. The ``MIN_MODULE_BUDGET_SECS``
          expression must REFERENCE ``MEASURED_SUITE_WORST_SECS`` by name. This
          is the rot task 3703 fixed for ONE file and nothing enforced for the
          others: a hand-set ``MIN_MODULE_BUDGET_SECS = 1800`` evaluates fine
          and agrees with today's measurement, then silently stops agreeing the
          next time the measurement moves. It now fails HERE, cross-file.

      (4) ONE CANONICAL DERIVATION. The expression is evaluated in a namespace
          whose ONLY bindings are the published worst and
          ``module_budget_family.min_budget`` — deliberately without
          ``__builtins__``. So ``min_budget(MEASURED_SUITE_WORST_SECS)``
          succeeds, while a local ``_min_budget`` copy or an inlined
          ``(int(2 * W) // 100) * 100`` raises ``NameError`` and fails here.
          That is what makes "one canonical expression" a mechanism rather than
          a convention: re-spelling the derivation locally is now impossible to
          land green, whatever value it happens to produce today.

      (5) SAME SHAPE. The evaluated value equals ``min_budget(published
          worst)``. (4) rejects a different SPELLING; (5) rejects a different
          MEANING — a 3x multiple, or a round-UP, reached through the canonical
          helper.

      (6) RAISE-TOGETHER MADE STRUCTURAL. For each family prefix, the
          PRODUCTION-discovered ``verify_command_timeout_secs`` (via
          ``config._discover_module_configs``, not a yaml re-read) is declared
          and is >= ``min_budget(published worst)``. Every yaml in the family
          carries a RAISE-TOGETHER rule in prose; this makes refreshing a
          measurement without raising its budget fail in ONE place for the WHOLE
          family, and keeps it failing even if an owning guard is later deleted
          or weakened. Task 4320's own step 1 is the worked example: the trigger
          fired, and this assertion is where it would have fired even had the
          owning guard not existed.

      (7) ``MEASURED_BY_SIBLING_GUARD`` BECOMES CHECKED, NOT DOCUMENTED. Every
          key in that table resolves to a pair the reader actually FOUND. That
          table's whole purpose is to say "this module's provenance lives over
          there" instead of copying a number here; until now nothing verified
          the pointer still landed on a guard that publishes one. A pointer that
          silently stops resolving is worse than a copied number, because it
          reads as delegation rather than as a gap.

    ASSERTS ON STRUCTURE AND VALUES, NEVER ON PROSE. Every check above is
    against a parsed AST node, an evaluated expression, or a ModuleConfig field.
    Nothing here reads a docstring, a comment or a string literal — the same
    line ``test_every_discovered_module_config_declares_its_own_verify_budget``
    draws when it declines to regex ``MODULE_BUDGET_EXCLUSIONS`` justifications.
    """
    # ---- (1) FLOOR SET -----------------------------------------------------
    absent = [str(p) for p in FAMILY_PUBLISHER_PATHS if not p.is_file()]
    assert not absent, (
        f'module_budget_family.FAMILY_PUBLISHER_PATHS names {absent} which do '
        f'not exist (task 4320). Either a publisher was renamed — in which case '
        f'update FAMILY_PUBLISHER_PATHS — or it was deleted, in which case the '
        f'family has lost a member and every assertion below would go quiet '
        f'about it rather than failing'
    )

    assert FAMILY_READER_PATH.is_file(), (
        f'module_budget_family.FAMILY_READER_PATH names {FAMILY_READER_PATH}, '
        f'which does not exist (task 4320). That constant records WHERE the '
        f'family\'s sole cross-file enforcement lives, so that a reader which '
        f'is renamed or deleted is visible from the shared module every '
        f'publisher already depends on rather than only from the file that '
        f'disappeared'
    )
    assert FAMILY_READER_PATH.resolve() == pathlib.Path(__file__).resolve(), (
        f'module_budget_family.FAMILY_READER_PATH names {FAMILY_READER_PATH}, '
        f'but the guard that actually reads the published pairs is '
        f'{pathlib.Path(__file__)} (task 4320). The constant exists to be true, '
        f'not to be decorative: a pointer nothing checks reads as authoritative '
        f'while naming the wrong file, which is the same stale-pointer failure '
        f'assertion (7) below closes for MEASURED_BY_SIBLING_GUARD'
    )

    pairs = published_pairs()

    assert len(pairs) == len(FAMILY_PUBLISHER_PATHS), (
        f'module_budget_family.published_pairs() returned {len(pairs)} pair(s) '
        f'{sorted(pairs)} for {len(FAMILY_PUBLISHER_PATHS)} publisher file(s) '
        f'(task 4320). A publisher that no longer defines a module-level '
        f'MODULE_PREFIX yields no pair at all, so it would drop out of every '
        f'check below silently'
    )
    assert set(pairs) == FAMILY_PUBLISHED_PREFIXES, (
        f'the budget family publishes pairs for {sorted(pairs)}, expected '
        f'exactly {sorted(FAMILY_PUBLISHED_PREFIXES)} (task 4320). This floor '
        f'exists so the per-pair assertions below cannot pass VACUOUSLY on a '
        f'shrunken set — the same hole KNOWN_MODULE_CONFIG_PREFIXES closes for '
        f'the coverage guard. If a module genuinely joined or left the family, '
        f'update FAMILY_PUBLISHED_PREFIXES and FAMILY_PUBLISHER_PATHS together'
    )

    discovered = _discovered()

    for prefix, pair in sorted(pairs.items()):
        # ---- (2) THE MEASUREMENT IS A MEASUREMENT --------------------------
        assert pair.worst_source is not None, (
            f'{prefix}: {pair.path.name} defines no module-level '
            f'MEASURED_SUITE_WORST_SECS (task 4320), so this family member '
            f'publishes a floor with no measurement behind it. The pair is the '
            f'unit — a floor without its worst run is a number with no '
            f'provenance, which is what the whole family exists to prevent'
        )
        worst_node = ast.parse(pair.worst_source, mode='eval').body
        assert (
            isinstance(worst_node, ast.Constant)
            and isinstance(worst_node.value, (int, float))
            and not isinstance(worst_node.value, bool)
        ), (
            f'{prefix}: MEASURED_SUITE_WORST_SECS in {pair.path.name} is '
            f'`{pair.worst_source}`, which is not a plain numeric literal '
            f'(task 4320). A MEASUREMENT is transcribed from an observed run '
            f'and recorded run-by-run in the owning yaml\'s PER-MODULE VERIFY '
            f'BUDGET block; a computed one has no run behind it, and would let '
            f'this pair be satisfied by two expressions that agree with each '
            f'other and with nothing that was ever measured'
        )
        assert pair.worst is not None, (
            f'{prefix}: MEASURED_SUITE_WORST_SECS in {pair.path.name} could not '
            f'be read as a literal value (task 4320): {pair.worst_source!r}'
        )

        # ---- (3) THE FLOOR IS DERIVED, NOT TRANSCRIBED ---------------------
        assert pair.floor_source is not None, (
            f'{prefix}: {pair.path.name} defines no module-level '
            f'MIN_MODULE_BUDGET_SECS (task 4320), so it records a measurement '
            f'that gates nothing'
        )
        floor_names = {
            node.id
            for node in ast.walk(ast.parse(pair.floor_source, mode='eval'))
            if isinstance(node, ast.Name)
        }
        assert 'MEASURED_SUITE_WORST_SECS' in floor_names, (
            f'{prefix}: MIN_MODULE_BUDGET_SECS in {pair.path.name} is '
            f'`{pair.floor_source}`, which does not REFERENCE '
            f'MEASURED_SUITE_WORST_SECS (task 4320) — it is transcribed rather '
            f'than derived. That is the exact rot task 3703 repaired for '
            f'tests/scripts and which nothing enforced for the rest of the '
            f'family: a hand-set floor agrees with the measurement on the day '
            f'it is written and then silently stops agreeing, and only a '
            f'reviewer ever caught it. Names referenced: {sorted(floor_names)}'
        )

        # ---- (4) ONE CANONICAL DERIVATION ----------------------------------
        assert pair.error is None and pair.floor is not None, (
            f'{prefix}: MIN_MODULE_BUDGET_SECS in {pair.path.name} is '
            f'`{pair.floor_source}`, which does not evaluate against the ONE '
            f'canonical derivation (task 4320): {pair.error}. The expression is '
            f'evaluated with only `min_budget` and `MEASURED_SUITE_WORST_SECS` '
            f'bound and no __builtins__, so a locally re-spelled derivation — a '
            f'`def _min_budget` copy, or an inlined `(int(2 * W) // 100) * 100` '
            f'— raises NameError HERE even when it computes the right number '
            f'today. Import module_budget_family.min_budget and call it: the '
            f'family may hold exactly one implementation of this expression'
        )

        # ---- (5) SAME SHAPE ------------------------------------------------
        expected_floor = min_budget(pair.worst)
        assert pair.floor == expected_floor, (
            f'{prefix}: MIN_MODULE_BUDGET_SECS in {pair.path.name} evaluates to '
            f'{pair.floor}, but min_budget(MEASURED_SUITE_WORST_SECS='
            f'{pair.worst}) is {expected_floor} (task 4320). Assertion (4) '
            f'rejects a different SPELLING of the derivation; this one rejects a '
            f'different MEANING reached through the canonical helper — a '
            f'different multiple, or a round-UP where the family rounds DOWN'
        )

        # ---- (6) RAISE-TOGETHER MADE STRUCTURAL ----------------------------
        mc = discovered.get(prefix)
        assert mc is not None, (
            f'{prefix} publishes a measurement/floor pair in {pair.path.name} '
            f'but config._discover_module_configs no longer registers a module '
            f'config at that prefix (task 4320), so the floor gates nothing in '
            f'production'
        )
        assert mc.verify_command_timeout_secs is not None, (
            f'{prefix}/orchestrator.yaml declares no verify_command_timeout_secs '
            f'(task 4320) while {pair.path.name} publishes a derived floor of '
            f'{pair.floor}s for it, so the module silently inherits the '
            f'repo-root whole-fleet ceiling and the floor is decorative'
        )
        assert mc.verify_command_timeout_secs >= expected_floor, (
            f'{prefix}/orchestrator.yaml declares '
            f'verify_command_timeout_secs={mc.verify_command_timeout_secs}, '
            f'below the {expected_floor}s floor min_budget derives from '
            f'MEASURED_SUITE_WORST_SECS={pair.worst} in {pair.path.name} '
            f'(task 4320). THIS IS THE RAISE-TOGETHER RULE, enforced rather '
            f'than merely written in that yaml\'s provenance block: a refreshed '
            f'measurement must be accompanied by a raised budget IN THE SAME '
            f'CHANGE. Raise the yaml — never lower the measurement, and never '
            f'hand-set the floor back'
        )

    # ---- (7) MEASURED_BY_SIBLING_GUARD BECOMES CHECKED ---------------------
    unresolved = sorted(set(MEASURED_BY_SIBLING_GUARD) - set(pairs))
    assert not unresolved, (
        f'MEASURED_BY_SIBLING_GUARD points at an owning guard for '
        f'{unresolved}, but module_budget_family.published_pairs() found no '
        f'published (measurement, floor) pair for those prefixes (task 4320). '
        f'That table exists so this file does not become a SECOND home for a '
        f'measurement; the price is that its entries are pointers, and a '
        f'pointer that has stopped resolving reads as delegation while '
        f'delegating to nothing. Either restore the pair in the owning guard, '
        f'or move the prefix into MEASURED_MODULE_SUITE_WORST_SECS and record '
        f'the figure here. Pairs found: {sorted(pairs)}'
    )


def test_root_config_fixture_anchors_this_worktrees_own_yaml(
    root_config: OrchestratorConfig,
) -> None:
    """The shared ``root_config`` fixture must pin the config to THIS worktree's yaml.

    Task 4320. Three near-identical ``_root_config`` helpers used to live in this
    directory — one in each member of the budget-guard family — and the anchoring
    they performed was load-bearing rather than hygiene. De-triplicating them into
    a single ``tests/scripts/conftest.py`` fixture is only safe if the BEHAVIOUR
    survives the move, so this pins the behaviour rather than the arrangement.

    Nothing here asserts about naming, placement or prose. Three properties, each
    the negation of a way the anchor can silently stop working:

      (a) THE ANCHOR IS APPLIED. ``ORCH_CONFIG_PATH`` names THIS worktree's
          ``dark-factory-orchestrator.yaml`` while the test runs.
          ``project_root`` is only a model FIELD and selects nothing:
          ``OrchestratorConfig.settings_customise_sources`` builds its
          ``YamlSettingsSource`` from ``os.environ['ORCH_CONFIG_PATH']`` alone,
          falling back to a CWD-relative ``config.yaml``. Both ambient states are
          wrong in OPPOSITE directions — UNSET (the state inside verify, which
          scrubs the whole ``ORCH_`` prefix) collapses every value to the
          pydantic defaults, while SET as an operator's shell has it points at
          whichever checkout that orchestrator serves, so every assertion would
          be about a DIFFERENT checkout and would report green on a worktree that
          had regressed.

      (b) THE YAML WAS ACTUALLY READ. The returned config's ``test_command`` and
          ``verify_command_timeout_secs`` equal what ``yaml.safe_load`` reads
          from that exact file. Asserted AGAINST THE FILE rather than against
          pinned literals, deliberately: pinning the warm ceiling as a number
          here would create a second copy of a value the root yaml owns, and it
          would start failing the day that yaml is legitimately retuned. The
          parse being a mapping, and both keys being PRESENT in it, are
          asserted before either is indexed — otherwise a root config that
          stopped declaring one died with a bare ``KeyError`` and said nothing
          about which file or which key.

      (c) IT IS NOT A DEFAULTS COLLAPSE. ``test_command`` is not the pydantic
          default ``'pytest'``. This is the SILENT failure mode: a missing or
          unset ``config_path`` makes ``YamlSettingsSource`` SKIP the file rather
          than raise, so a broken anchor yields a fully-populated config of
          DEFAULTS with no error anywhere. (a) and (b) would both still hold if
          the repo happened to declare exactly the defaults; (c) is what makes
          the distinction observable at all.
    """
    expected_path = REPO_ROOT / 'dark-factory-orchestrator.yaml'

    # (a) THE ANCHOR IS APPLIED.
    assert os.environ.get('ORCH_CONFIG_PATH') == str(expected_path), (
        f'ORCH_CONFIG_PATH is {os.environ.get("ORCH_CONFIG_PATH")!r} during this '
        f'test, not {str(expected_path)!r} (task 4320). The root_config fixture '
        f'exists to pin config loading to THIS worktree; project_root is only a '
        f'model field and selects nothing, so without the env var the config is '
        f'read from whatever checkout the ambient environment points at — or, '
        f'when unset, from nowhere at all'
    )

    # (b) THE YAML WAS ACTUALLY READ — compared against the file, not a literal.
    declared = yaml.safe_load(expected_path.read_text(encoding='utf-8'))
    # THE KEYS ARE ASSERTED BEFORE THEY ARE INDEXED (task 4320 amendment).
    # Indexing straight into the parsed yaml made a root config that stopped
    # declaring either key die with a bare KeyError on the comparison line —
    # none of the messages below, and no hint that the failure is about the
    # root yaml declaring nothing. That is the inverse of the loud-structured-
    # failure norm the rest of this file follows, and of the `is_file()`
    # precheck the fixture itself carries for the same class of silent failure.
    assert isinstance(declared, dict), (
        f'{expected_path} parsed to {type(declared).__name__}, not a mapping '
        f'(task 4320), so nothing below can be compared against it. An empty '
        f'or malformed root config yields None here while the fixture\'s '
        f'config silently collapses to the pydantic defaults'
    )
    missing = [
        key
        for key in ('test_command', 'verify_command_timeout_secs')
        if key not in declared
    ]
    assert not missing, (
        f'{expected_path.name} declares no {missing!r} (task 4320), so the '
        f'comparisons below have nothing to compare the fixture\'s config '
        f'against. Both keys are read through this fixture by every "strictly '
        f'below the root warm ceiling" assertion in this directory: a root '
        f'config that stopped declaring them would leave those assertions '
        f'reading the pydantic defaults, which is the exact silent collapse (c) '
        f'exists to make observable'
    )
    assert root_config.test_command == declared['test_command'], (
        f'root_config.test_command is {root_config.test_command!r} but '
        f'{expected_path.name} declares {declared["test_command"]!r} '
        f'(task 4320) — the fixture did not load this worktree\'s yaml'
    )
    assert (
        root_config.verify_command_timeout_secs
        == declared['verify_command_timeout_secs']
    ), (
        f'root_config.verify_command_timeout_secs is '
        f'{root_config.verify_command_timeout_secs} but {expected_path.name} '
        f'declares {declared["verify_command_timeout_secs"]} (task 4320). Every '
        f'"strictly below the root warm ceiling" assertion in this directory is '
        f'read through this fixture, so a wrong value here silently changes what '
        f'those assertions mean'
    )

    # (c) IT IS NOT A DEFAULTS COLLAPSE.
    assert root_config.test_command != 'pytest', (
        "root_config.test_command is the pydantic default 'pytest' (task 4320), "
        'which is what OrchestratorConfig yields when YamlSettingsSource finds '
        'no file — it SKIPS a non-existent config_path rather than raising, so '
        'a broken anchor produces a fully-populated config of DEFAULTS and no '
        'error anywhere. That silence is the whole reason the fixture asserts '
        'the path is a real file before setting the env var'
    )
