"""conftest.py for tests/scripts/ — sys.path insertions, and the shared root_config anchor.

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

AND ONE FIXTURE. ``root_config`` is the DIRECTORY-WIDE single home for anchoring
``ORCH_CONFIG_PATH`` at this worktree's own ``dark-factory-orchestrator.yaml``
before reading it through the production loader. Task 4320 de-triplicated it
from three verbatim ``_root_config`` helpers, one in each member of the
module-config guard family; ``test_module_verify_budgets.py::
test_root_config_fixture_anchors_this_worktrees_own_yaml`` pins the behaviour
the move had to preserve.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig

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
