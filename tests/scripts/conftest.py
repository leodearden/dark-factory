"""conftest.py for tests/scripts/ — sys.path insertions, and the shared module-config anchors.

Two sys.path insertions, both load-bearing for the same reason and neither
redundant:

1. ``scripts/`` — mirrors scripts/tests/conftest.py so that
   `import reviewer_redundancy_diagnostic` (and the other module-under-test
   imports in this directory) resolves when pytest collects here.
2. ``this directory`` — so that shared, non-test helper modules living
   alongside the tests (e.g. `import systemd_unit_invariants`,
   `import module_budget_family`) resolve.

The second is not something pytest does for us: pyproject.toml sets
``addopts = "--import-mode=importlib ..."``, and under importlib mode pytest
deliberately does NOT insert a test file's own directory onto sys.path (that
sys.path mutation is a `prepend`/`append`-mode behaviour it exists to avoid).
So a sibling helper module is unimportable from these tests without this line,
and the failure surfaces at COLLECTION as a bare ModuleNotFoundError rather
than as anything resembling the invariant under test.

AND THREE FIXTURES, each the DIRECTORY-WIDE single home for something all three
members of the module-config guard family had a verbatim copy of before task
4320: ``root_config`` (anchoring ``ORCH_CONFIG_PATH`` at this worktree's own
``dark-factory-orchestrator.yaml`` before reading it through the production
loader), ``discover_module_configs`` and ``executed_for_touched``.
``test_module_verify_budgets.py::
test_root_config_fixture_anchors_this_worktrees_own_yaml`` pins the behaviour
the first move had to preserve.

WHY conftest.py AND NOT A SIBLING HELPER MODULE, given that task 4320 also
added ``module_budget_family.py`` next door. That module is named for, and
scoped to, the budget derivation: a plan->execution bridge is not part of it,
and widening it to "shared things the guards use" would make it the junk drawer
this directory has so far avoided. pytest already provides the directory-wide
home, and these are setup rather than assertion — the only thing the family's
no-cross-import convention actually forbids is one GUARD importing another.

``REPO_ROOT`` IS DELIBERATELY NOT DE-DUPLICATED, and the omission is reasoned
rather than overlooked. ``pathlib.Path(__file__).parents[2]`` is spelled at
module level in TWENTY-NINE files in this directory; collapsing it in the three
that happen to be family members would leave twenty-six copies and turn a
uniform idiom into an inconsistency, for no drift the change would remove. It
is also not the drift class the helpers above are: a wrong ``parents[]`` index
does not silently agree with the right one for a while, it fails every test in
the file immediately and loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from orchestrator.config import ModuleConfig, OrchestratorConfig

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / 'scripts'
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

REPO_ROOT = Path(__file__).parents[2]

# This worktree's own top-level orchestrator config — the one ``root_config``
# reads. ``dark-factory-orchestrator.yaml`` is the canonical, REQUIRED filename
# for a project's top-level config (it is what the dashboard's escalation-URL
# discovery keys on); the legacy spellings are a discovery fallback for
# unmigrated projects, not a choice this repo has. Anchored to REPO_ROOT rather
# than taken from the ambient ORCH_CONFIG_PATH, for the reason the fixture's
# docstring records at length.
ROOT_CONFIG_PATH = REPO_ROOT / 'dark-factory-orchestrator.yaml'


@pytest.fixture
def root_config(monkeypatch: pytest.MonkeyPatch) -> OrchestratorConfig:
    """The repo-root config read through the PRODUCTION loader, anchored at this worktree.

    ONE IMPLEMENTATION, DIRECTORY-WIDE (task 4320). This was three verbatim
    ``_root_config`` helpers — in ``test_tests_scripts_module_config.py``,
    ``test_scripts_module_config.py`` and ``test_module_verify_budgets.py`` —
    each anchoring the same env var at the same path and returning the same
    object. The no-cross-import convention those files state is about a test
    file importing a SIBLING TEST FILE, which couples guards that must be able
    to fail independently; it does not reach conftest.py, which is pytest's
    idiomatic home for exactly this, and every guard still fails entirely on its
    own. ``test_tests_scripts_module_config.py``'s own copy said so, and filed
    the de-triplication as a follow-up because conftest.py was outside task
    3703's locked scope. That follow-up is task 4320 and it landed here.

    THE SAME DEFECT HAD ALREADY BEEN FIXED ONCE AT A NARROWER SCOPE, and that
    lineage is kept because it is the argument for fixing it at THIS scope. The
    deleted ``test_scripts_module_config.py`` copy carried the note that "task
    3458's amendment pass extracted this from three near-identical copies —
    see git history for the pre-extraction shape": three copies WITHIN that one
    file. Collapsing them there was right and did not generalise, so the
    triplication simply reappeared one level up, as one copy per file, and
    survived two further tasks that each touched the family. A convention that
    has to be re-applied by hand at every scope is not a mechanism; a
    directory-wide fixture is where this one stops recurring.

    A FIXTURE IS STRUCTURALLY STRONGER THAN A CALLED HELPER, not merely tidier.
    Two guards in this directory turn on ORDER — they build the anchored config
    FIRST and poison ``ORCH_CONFIG_PATH`` SECOND, to show the config was read
    at construction rather than resolved lazily from the ambient environment.
    As a called helper that ordering was a COMMENT, enforceable only by reading.
    Fixture setup runs before the test body, so it is now structural: a test
    that takes this fixture cannot accidentally construct its config after
    poisoning the environment.

    ANCHORING ``ORCH_CONFIG_PATH`` IS LOAD-BEARING, not hygiene, and an early
    draft of one of the deleted copies omitted it on the false premise that
    ``project_root=REPO_ROOT`` selects which yaml is read. It does not:
    ``project_root`` is only a model FIELD, and
    ``OrchestratorConfig.settings_customise_sources`` builds its
    ``YamlSettingsSource`` from ``os.environ['ORCH_CONFIG_PATH']`` alone,
    falling back to a CWD-relative ``config.yaml``. Both ambient states are
    wrong, in OPPOSITE directions:

      * UNSET — which is the state INSIDE VERIFY, because
        ``verify._target_subprocess_env`` deliberately scrubs the whole
        ``ORCH_`` prefix (task 2957) — finds no file, so every value collapses
        to the pydantic DEFAULTS, where e.g. ``test_command`` is the bare
        literal ``'pytest'``. A caller would then fail with a message about the
        fleet chain having dropped a suite, when the chain is in fact correct
        and was simply never read.
      * SET, as an operator's shell has it, points at whichever checkout that
        orchestrator serves — typically the MAIN one, not this worktree. A
        caller would then assert about a different checkout's yaml and report
        GREEN on a worktree that had actually regressed: the exact
        reports-green-while-checking-something-else failure these guards exist
        to prevent, one env var over.

    Setting the env var IS the production load path (``config.load_config``
    stamps ``os.environ['ORCH_CONFIG_PATH']`` before constructing), so this
    stays a read through the real loader — pinned to THIS worktree's committed
    yaml rather than left to the ambient environment. Same remedy, same reason,
    as ``tests/scripts/test_orchestrator_watchdog.py``'s
    ``test_orch_restart_min_interval_secs_matches_config_default``.

    FAILS LOUDLY ON A MISSING FILE rather than silently: ``YamlSettingsSource``
    SKIPS a non-existent ``config_path`` instead of raising, so a bad path would
    yield the pydantic DEFAULTS — a config this repo does not declare — with no
    error anywhere.

    ``orchestrator.config`` is imported INSIDE the fixture, not at module level.
    A conftest is imported for every test in the directory, so a module-level
    import would turn an unimportable ``orchestrator`` into a collection failure
    for the whole of tests/scripts — including the many files here that never
    touch it — instead of a failure in the tests that actually depend on it.
    """
    from orchestrator.config import OrchestratorConfig

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


@pytest.fixture
def discover_module_configs() -> Callable[[], dict[str, ModuleConfig]]:
    """Run the PRODUCTION module-config walk, keyed by repo-relative prefix.

    ONE IMPLEMENTATION, DIRECTORY-WIDE (task 4320 amendment). This was a third
    verbatim helper — ``_discovered`` in each of ``test_module_verify_
    budgets.py``, ``test_scripts_module_config.py`` and
    ``test_tests_scripts_module_config.py`` — of exactly the shape ``_root_
    config`` had: pure, non-asserting setup, coupling no guard to any other. The
    argument recorded at length in ``root_config``'s docstring for fixing that
    one here applies to this one unchanged, and leaving it behind would have
    left the same drift class alive in the neighbouring helper.

    DELEGATES to ``config._discover_module_configs`` rather than globbing
    ``**/orchestrator.yaml``. A hand-rolled glob run from the main checkout
    would descend ``.worktrees/`` and ``.venv/``, and would remain free to drift
    from what the orchestrator actually registers; delegating inherits the
    production pruning of ``.worktrees``, ``.venv``, ``node_modules``,
    ``build``, ``target``, ``.claude`` and nested ``.git`` checkouts, and covers
    configs at ANY depth (``tests/scripts``, not just depth-1 names).

    A CALLABLE, NOT THE DICT ITSELF, and that is load-bearing rather than
    stylistic. A fixture yielding the dict would perform the walk during SETUP,
    which is exactly what two guards in this directory must NOT have: the
    ``test_executed_for_touched_is_hermetic_against_the_ambient_orch_config_
    path`` pair poison ``ORCH_CONFIG_PATH`` and then read a module's declared
    budget through this walk, to show that the figure comes from that module's
    own yaml and not from whatever the ambient environment points at. Hoisting
    the walk to setup time would move it BEFORE the poison and make that claim
    vacuous. Note the contrast with ``root_config``, where setup-time
    construction is precisely the point — the two fixtures want opposite
    timings, for opposite reasons, and both reasons are about the same env var.

    ``orchestrator.config`` is imported INSIDE the fixture for the reason
    ``root_config``'s docstring gives: a conftest is imported for every test in
    the directory, so a module-level import would turn an unimportable
    ``orchestrator`` into a collection failure for the whole of tests/scripts.
    """
    from orchestrator.config import _discover_module_configs

    def _discover() -> dict[str, ModuleConfig]:
        return _discover_module_configs(REPO_ROOT)

    return _discover


@pytest.fixture
def executed_for_touched(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> Callable[[str, list[str], OrchestratorConfig], ModuleConfig]:
    """Run the PRODUCTION plan->execution bridge and return the single executed config.

    ``derive_verify_plan`` decides scope; ``_executed_module_configs_from_plan``
    renders those PlannedRuns into the exact ModuleConfig ``run_verification``
    executes. Asserting on THAT is what makes "ruff ran over scripts/", "the
    tests/scripts segment ran" and "the budget survives to execution"
    STRUCTURAL claims rather than exit-code claims.

    ONE IMPLEMENTATION, DIRECTORY-WIDE (task 4320 amendment). The body was
    byte-identical in all three family members; only the prefix differed, and
    only in how it was obtained — ``test_module_verify_budgets.py`` took it as
    a parameter while the two publishers closed over their own
    ``MODULE_PREFIX``. The parameterised form is the one kept, since a closure
    over a module constant is what made the other two copies look
    file-specific when they were not.

    *cfg* IS A REQUIRED PARAMETER, not a convenience (task 3703, applying the
    shape commit 6c72a7da5a landed in ``test_module_verify_budgets.py``). It
    must be the config built by the ``root_config`` fixture above, whose
    docstring spells out why the ``ORCH_CONFIG_PATH`` anchor is load-bearing:
    an unset anchor collapses every value to the pydantic defaults, SILENTLY.
    Each of the three copies used to construct its own
    ``OrchestratorConfig(project_root=REPO_ROOT)``, and each was broken in a
    slightly different way by it:

      * ``test_scripts_module_config.py`` — ORDERING-DEPENDENT.
        ``test_scripts_module_carries_its_own_measured_verify_budget`` built an
        anchored config for assertion (c) and only then reached assertion (e)'s
        call here, so the helper read the right yaml purely as a SIDE EFFECT of
        that earlier line, while three other callers anchored nothing at all.
      * ``test_tests_scripts_module_config.py`` — worse: NO caller anchored it,
        so it read whatever ambient ``ORCH_CONFIG_PATH`` the process carried.
      * ``test_module_verify_budgets.py`` — repaired first, and the source of
        the shape the other two were then given.

    Taking the config as an argument makes the dependency structural instead of
    ordering-dependent or ambient. The pair of
    ``test_executed_for_touched_is_hermetic_against_the_ambient_orch_config_
    path`` guards is what holds that property down.

    The ``lambda _f: None`` worktree_reader keeps this hermetic: no file reads,
    and nothing classifies STRUCTURAL, so the lint/type legs stay FILE_SCOPED.
    """
    from orchestrator import verify, verify_plan

    def _executed(
        prefix: str, files: list[str], cfg: OrchestratorConfig
    ) -> ModuleConfig:
        mc = discover_module_configs()[prefix]
        plan = verify_plan.derive_verify_plan(files, [mc], cfg, lambda _f: None)
        executed = verify._executed_module_configs_from_plan([mc], plan)
        assert len(executed) == 1, (
            f'expected exactly one executed module config for {files!r}, got '
            f'{[e.prefix for e in executed]!r}'
        )
        return executed[0]

    return _executed
