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
infra_timeout, on a diff whose own tests take ~105s.

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

from orchestrator import verify, verify_plan
from orchestrator.config import OrchestratorConfig, _discover_module_configs
from orchestrator.module_charter import derive_modules

REPO_ROOT = pathlib.Path(__file__).parents[2]

MODULE_PREFIX = 'tests/scripts'

# A file in this suite, used as the representative touched-file for the
# derive_modules -> for_module routing assertions below.
SAMPLE_TOUCHED_FILE = 'tests/scripts/test_spawn_claude.py'


def _discovered() -> dict:
    return _discover_module_configs(REPO_ROOT)


def test_tests_scripts_is_a_registered_module_config() -> None:
    """tests/scripts/ must be discovered, and must be REACHABLE by the routing chain.

    The five assertions below are one cohesive contract — discovery alone is
    not enough, because a config the scheduler/workflow path cannot reach is
    silently half-applied (``load_config`` warns about exactly this at
    config.py:4712-4725 but does not fail).

    NOTE on assertion (3)/(4) — a falsified premise corrected against measured
    reality (the same class of defect this task exists to fix). Task 3350's
    plan asserted ``lock_depth == 2``, reading the pydantic Field default at
    config.py:2593. The EFFECTIVE value is 4: the package-bundled
    ``orchestrator/src/orchestrator/defaults.yaml:7`` ships ``lock_depth: 4``
    and is layered over the Field default on every load. At depth 4,
    ``derive_modules([SAMPLE_TOUCHED_FILE], 4)`` returns the full path
    ``['tests/scripts/test_spawn_claude.py']``, NOT ``['tests/scripts']`` —
    3 path components is below the depth-4 truncation threshold, so
    ``normalize_lock`` leaves it whole.

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
    # truncated keys. config.py:4719 warns; it does not fail. Holds at both the
    # Field default (2) and the shipped defaults.yaml value (4).
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
    # touched path IS truncated to lock_depth (i.e. at any lock_depth <= 2).
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


def _executed_for_touched(files: list[str]):
    """Run the PRODUCTION plan->execution bridge and return the single executed config.

    ``derive_verify_plan`` decides scope; ``_executed_module_configs_from_plan``
    renders those PlannedRuns into the exact ModuleConfig ``run_verification``
    executes. Asserting on THAT is what makes "the tests/scripts segment ran" a
    structural claim rather than an exit-code claim.

    The ``lambda _f: None`` worktree_reader keeps this hermetic: no file reads,
    and nothing classifies STRUCTURAL, so pyright stays FILE_SCOPED.
    """
    mc = _discovered()[MODULE_PREFIX]
    cfg = OrchestratorConfig(project_root=REPO_ROOT)
    plan = verify_plan.derive_verify_plan(files, [mc], cfg, lambda _f: None)
    executed = verify._executed_module_configs_from_plan([mc], plan)
    assert len(executed) == 1, (
        f'expected exactly one executed module config for {files!r}, got '
        f'{[e.prefix for e in executed]!r}'
    )
    return executed[0]


def test_tests_scripts_diff_executes_its_own_suite_and_keeps_lint_and_type() -> None:
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
    executed = _executed_for_touched([SAMPLE_TOUCHED_FILE])

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
